"""HAN classification on original and reduced DBLP main structures."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import hhin_core as core
import gnn_data
import run_hgb_rgcn_adapter as hgb
from runtime_helpers import fresh_run_dir, parse_int_list


HGB_HAN_SOURCE = "https://github.com/THUDM/HGB/tree/master/NC/benchmark/methods/HAN"
VIEW_NAMES = ("main_original", "main_reduced")
DBLP_METAPATHS = {
    "APA": (0, 3),
    "APVPA": (0, 2, 5, 3),
}


@dataclass
class HANView:
    name: str
    graphs: list[object]
    features: torch.Tensor
    train_records: list[tuple[int, int, int]]
    test_records: list[tuple[int, int, int]]
    author_count: int
    main_node_count: int
    main_edge_count: int
    metapath_edge_counts: dict[str, int]


class SemanticAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = self.project(embeddings).mean(0)
        weights = torch.softmax(weights, dim=0)
        return (weights.expand_as(embeddings) * embeddings).sum(1)


class HANLayer(nn.Module):
    def __init__(
        self,
        gat_conv,
        metapath_count: int,
        input_size: int,
        hidden_size: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.gat_layers = nn.ModuleList(
            gat_conv(
                input_size,
                hidden_size,
                heads,
                dropout,
                dropout,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
            for _ in range(metapath_count)
        )
        self.semantic_attention = SemanticAttention(hidden_size * heads)

    def forward(self, graphs, features):
        path_embeddings = [
            layer(graph, features).flatten(1)
            for graph, layer in zip(graphs, self.gat_layers)
        ]
        return self.semantic_attention(torch.stack(path_embeddings, dim=1))


class HAN(nn.Module):
    def __init__(
        self,
        gat_conv,
        metapath_count: int,
        input_size: int,
        hidden_size: int,
        heads: int,
        class_count: int,
        dropout: float,
    ):
        super().__init__()
        self.layer = HANLayer(
            gat_conv,
            metapath_count,
            input_size,
            hidden_size,
            heads,
            dropout,
        )
        self.predict = nn.Linear(hidden_size * heads, class_count)

    def forward(self, graphs, features):
        return self.predict(self.layer(graphs, features))


def make_heterograph(dgl, data, relation_mats, counts):
    graph_data = {}
    for relation_id in sorted(relation_mats):
        src_type, dst_type, _ = data["link_defs"][relation_id]
        matrix = relation_mats[relation_id].tocoo()
        graph_data[(str(src_type), f"r{relation_id}", str(dst_type))] = (
            torch.from_numpy(matrix.row.astype(np.int64)),
            torch.from_numpy(matrix.col.astype(np.int64)),
        )
    return dgl.heterograph(
        graph_data,
        num_nodes_dict={str(node_type): count for node_type, count in counts.items()},
    )


def mean_cluster_features(features: sp.csr_matrix, mapping: np.ndarray) -> np.ndarray:
    groups = core.build_group_matrix(mapping).astype(np.float32)
    counts = np.asarray(groups.sum(axis=1)).ravel().astype(np.float32)
    totals = groups @ features.astype(np.float32)
    means = totals.multiply((1.0 / np.maximum(counts, 1.0))[:, None])
    return means.toarray().astype(np.float32)


def prepare_views(dgl, data, mats, cfg, cluster_of):
    rgcn_original, rgcn_reduced, audit = hgb.build_main_views(
        dgl, data, mats, cfg, cluster_of
    )
    reduced_mats = hgb.reduce_relation_matrices(data, mats, cfg, cluster_of)
    relation_sets = {
        "main_original": {relation_id: mats[relation_id] for relation_id in cfg["core_link_types"]},
        "main_reduced": reduced_mats,
    }
    counts = {
        "main_original": rgcn_original.counts,
        "main_reduced": rgcn_reduced.counts,
    }
    author_features = data["feats_by_type"][cfg["target_type"]].tocsr()
    features = {
        "main_original": author_features.toarray().astype(np.float32),
        "main_reduced": mean_cluster_features(
            author_features, cluster_of[cfg["target_type"]]
        ),
    }
    label_views = {
        "main_original": rgcn_original,
        "main_reduced": rgcn_reduced,
    }

    views = {}
    preparation = {}
    for view_name in VIEW_NAMES:
        started = time.perf_counter()
        heterograph = make_heterograph(
            dgl, data, relation_sets[view_name], counts[view_name]
        )
        graphs = []
        edge_counts = {}
        for path_name, relation_ids in DBLP_METAPATHS.items():
            metapath = [f"r{relation_id}" for relation_id in relation_ids]
            graph = dgl.metapath_reachable_graph(heterograph, metapath)
            graphs.append(graph)
            edge_counts[path_name] = int(graph.num_edges())
        elapsed = time.perf_counter() - started
        labels = label_views[view_name]
        main_edges = sum(matrix.nnz for matrix in relation_sets[view_name].values())
        views[view_name] = HANView(
            view_name,
            graphs,
            torch.from_numpy(features[view_name]),
            labels.train_records,
            labels.test_records,
            counts[view_name][cfg["target_type"]],
            sum(counts[view_name].values()),
            int(main_edges),
            edge_counts,
        )
        preparation[view_name] = {
            "seconds": elapsed,
            "metapath_edges": edge_counts,
        }
    return views, audit, preparation


def local_split(split, target_offset: int):
    train_idx, train_y, val_idx, val_y = split
    return (
        train_idx - target_offset,
        train_y,
        val_idx - target_offset,
        val_y,
    )


def local_records(records, target_offset: int):
    indices = np.asarray(
        [record[0] - target_offset for record in records], dtype=np.int64
    )
    labels = np.asarray([record[1] for record in records], dtype=np.int64)
    return indices, labels


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_once(dgl, gat_conv, view: HANView, split, seed: int, args):
    hgb.set_seed(seed, dgl)
    device = torch.device(args.device)
    train_idx, train_y, val_idx, val_y = split
    test_idx, test_y = local_records(view.test_records, 0)
    class_count = int(max(train_y.max(), val_y.max(), test_y.max())) + 1

    graphs = [graph.to(device) for graph in view.graphs]
    features = view.features.to(device)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)
    test_idx_t = torch.from_numpy(test_idx).to(device)
    model = HAN(
        gat_conv,
        len(graphs),
        features.shape[1],
        args.hidden_dim,
        args.heads,
        class_count,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_micro = -1.0
    best_macro = -1.0
    best_micro_epoch = 0
    best_macro_epoch = 0
    best_micro_state = None
    best_macro_state = None
    sync(device)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graphs, features)
        loss = F.cross_entropy(logits[train_idx_t], train_y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(graphs, features)[val_idx_t].argmax(1).cpu().numpy()
        metrics = hgb.evaluate(val_y, predictions)
        if metrics["micro_f1"] > best_micro + 1e-8:
            best_micro = metrics["micro_f1"]
            best_micro_epoch = epoch + 1
            best_micro_state = copy.deepcopy(model.state_dict())
        if metrics["macro_f1"] > best_macro + 1e-8:
            best_macro = metrics["macro_f1"]
            best_macro_epoch = epoch + 1
            best_macro_state = copy.deepcopy(model.state_dict())
    sync(device)
    elapsed = time.perf_counter() - started

    def test(state):
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predictions = model(graphs, features)[test_idx_t].argmax(1).cpu().numpy()
        return hgb.evaluate(test_y, predictions)

    micro_test = test(best_micro_state)
    macro_test = test(best_macro_state)
    return {
        "graph_view": view.name,
        "seed": seed,
        "validation_micro_f1": best_micro,
        "validation_macro_f1": best_macro,
        "best_micro_epoch": best_micro_epoch,
        "best_macro_epoch": best_macro_epoch,
        "test_accuracy": micro_test["accuracy"],
        "test_micro_f1": micro_test["micro_f1"],
        "test_macro_f1": macro_test["macro_f1"],
        "macro_at_micro_checkpoint": micro_test["macro_f1"],
        "micro_at_macro_checkpoint": macro_test["micro_f1"],
        "training_seconds": elapsed,
        "seconds_per_epoch": elapsed / args.epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "author_nodes": view.author_count,
        "main_nodes": view.main_node_count,
        "main_edges": view.main_edge_count,
        "apa_edges": view.metapath_edge_counts["APA"],
        "apvpa_edges": view.metapath_edge_counts["APVPA"],
        "train_examples": len(train_idx),
        "validation_examples": len(val_idx),
        "test_examples": len(test_idx),
    }


def summarize(raw: pd.DataFrame):
    return raw.groupby("graph_view", as_index=False).agg(
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        test_micro_f1_mean=("test_micro_f1", "mean"),
        test_micro_f1_std=("test_micro_f1", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", "std"),
        validation_micro_f1_mean=("validation_micro_f1", "mean"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        training_seconds_mean=("training_seconds", "mean"),
        training_seconds_std=("training_seconds", "std"),
        seconds_per_epoch_mean=("seconds_per_epoch", "mean"),
        seconds_per_epoch_std=("seconds_per_epoch", "std"),
        parameters=("parameters", "first"),
        author_nodes=("author_nodes", "first"),
        main_nodes=("main_nodes", "first"),
        main_edges=("main_edges", "first"),
        apa_edges=("apa_edges", "first"),
        apvpa_edges=("apvpa_edges", "first"),
        train_examples=("train_examples", "first"),
        validation_examples=("validation_examples", "first"),
        test_examples=("test_examples", "first"),
    )


def paired_comparison(raw: pd.DataFrame):
    comparison = {}
    for metric in ("test_accuracy", "test_micro_f1", "test_macro_f1"):
        pivot = raw.pivot(index="seed", columns="graph_view", values=metric)
        original = pivot["main_original"].to_numpy(dtype=np.float64)
        reduced = pivot["main_reduced"].to_numpy(dtype=np.float64)
        comparison[metric] = {
            "reduced_minus_original_mean": float(np.mean(reduced - original)),
            "paired_ttest_pvalue": core.safe_paired_ttest_pvalue(
                original, reduced
            ),
        }
    times = raw.pivot(
        index="seed", columns="graph_view", values="seconds_per_epoch"
    )
    speedups = (
        times["main_original"].to_numpy(dtype=np.float64)
        / times["main_reduced"].to_numpy(dtype=np.float64)
    )
    comparison["training"] = {
        "mean_of_paired_speedups": float(np.mean(speedups)),
        "speedup_from_mean_times": float(
            times["main_original"].mean() / times["main_reduced"].mean()
        ),
    }
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-thresholds",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "selected_thresholds.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--run-label", default="han_validation")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    output_dir = fresh_run_dir(args.output_root, args.run_label, args.allow_overwrite)
    torch.set_num_threads(args.threads)
    dgl, _, graphbolt_bypassed = hgb.require_dgl()
    from dgl.nn.pytorch import GATConv

    loaded = core.load_hgb_zip(str(args.data_dir / "DBLP.zip"))
    mats = core.build_link_matrices(loaded)
    data = core._strip_unpickleable_data(loaded)
    cfg = core.CONFIGS["DBLP"]
    tau_h, tau_x = gnn_data.selected_dblp_thresholds(args.selected_thresholds)
    cluster_of, fixed_iterations = gnn_data.build_reduction(
        data, mats, cfg, tau_h, tau_x
    )
    views, data_audit, preparation = prepare_views(
        dgl, data, mats, cfg, cluster_of
    )

    units = gnn_data.cluster_split_units(
        views["main_original"].train_records,
        views["main_reduced"].train_records,
        cluster_of[cfg["target_type"]],
        0,
    )
    rows = []
    split_audit = {}
    for seed in seeds:
        paired = gnn_data.paired_split(units, seed, 0)
        split_audit[str(seed)] = {
            "train_clusters": paired["train_clusters"],
            "validation_clusters": paired["validation_clusters"],
        }
        for view_name in VIEW_NAMES:
            split_name = "Original" if view_name == "main_original" else "Reduced"
            print(f"[HAN] view={view_name} seed={seed}", flush=True)
            rows.append(
                train_once(
                    dgl,
                    GATConv,
                    views[view_name],
                    local_split(paired[split_name], 0),
                    seed,
                    args,
                )
            )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(output_dir / "han_raw.csv", index=False)
    summary.to_csv(output_dir / "han_summary.csv", index=False)
    metadata = {
        "dataset": "DBLP",
        "official_source": HGB_HAN_SOURCE,
        "implementation_scope": (
            "HGB HAN architecture and DBLP model/optimizer hyperparameters, "
            "adapted to the original and reduced HHIN main structures with "
            "shared splits and a fixed training budget."
        ),
        "metapaths": {name: list(path) for name, path in DBLP_METAPATHS.items()},
        "metapath_scope": (
            "APA and APVPA use only main-structure relations. APTPA is omitted "
            "because Term belongs to the strong attribute layer and is excluded "
            "from both compared carriers."
        ),
        "feature_rule": (
            "Original public HGB author features; arithmetic mean of member "
            "features for each reduced author cluster."
        ),
        "edge_rule": (
            "Metapath reachable graphs use binary reachability, as in HAN; "
            "reduced relation multiplicities do not become attention weights."
        ),
        "timing_boundary": (
            "Reduction, heterograph construction, and metapath graph preparation "
            "are excluded. Each timed epoch includes one optimization update and "
            "one validation inference pass."
        ),
        "checkpoint_rule": (
            "Validation Micro-F1 selects Accuracy/Micro-F1; validation Macro-F1 "
            "selects Macro-F1."
        ),
        "tau_h": tau_h,
        "tau_x": tau_x,
        "fixed_point_iterations": fixed_iterations,
        "seeds": seeds,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "threads": args.threads,
        "dgl_version": dgl.__version__,
        "torch_version": torch.__version__,
        "graphbolt_import_bypassed": graphbolt_bypassed,
        "main_data_audit": data_audit,
        "split_audit": split_audit,
        "preparation": preparation,
        "paired_comparison": paired_comparison(raw),
    }
    with (output_dir / "han_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

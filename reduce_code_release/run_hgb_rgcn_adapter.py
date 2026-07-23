"""Run the HGB R-GCN architecture on complete and main-structure DBLP views."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import hhin_core as base
import gnn_data


OFFICIAL_HGB_SOURCE = (
    "https://github.com/THUDM/HGB/tree/master/NC/benchmark/methods/RGCN"
)


@dataclass
class GraphView:
    name: str
    graph: object
    edge_types: torch.Tensor
    edge_norm: torch.Tensor
    node_types: list[int]
    counts: dict[int, int]
    offsets: dict[int, int]
    relation_ids: list[int]
    relation_names: list[str]
    train_records: list[tuple[int, int, int]]
    test_records: list[tuple[int, int, int]]


def require_dgl():
    """Import DGL while bypassing an unused GraphBolt binary when necessary."""
    graphbolt_bypassed = False
    dgl_spec = importlib.util.find_spec("dgl")
    if dgl_spec is None or not dgl_spec.submodule_search_locations:
        raise RuntimeError(
            "DGL is not installed. Create the separate environment described in "
            "requirements-hgb-rgcn.txt."
        )
    dgl_root = Path(next(iter(dgl_spec.submodule_search_locations)))
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    expected_graphbolt = (
        dgl_root / "graphbolt" / f"graphbolt_pytorch_{torch_version}.dll"
    )
    if sys.platform.startswith("win") and not expected_graphbolt.exists():
        # DGL imports GraphBolt through its distributed package even for full-batch
        # RelGraphConv.  This experiment never calls GraphBolt or distributed APIs.
        sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))
        graphbolt_bypassed = True
    try:
        import dgl
        from dgl.nn.pytorch import RelGraphConv
    except (ImportError, FileNotFoundError) as exc:
        raise RuntimeError(
            "A compatible DGL environment is required. Use the separate "
            "requirements-hgb-rgcn.txt environment; DGL 2.2.1 is incompatible "
            "with PyTorch 2.8 because no matching GraphBolt binary is shipped."
        ) from exc
    return dgl, RelGraphConv, graphbolt_bypassed


def set_seed(seed: int, dgl_module):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl_module.seed(seed)


def offsets_for_types(counts: dict[int, int], node_types: list[int]):
    offsets = {}
    cursor = 0
    for node_type in node_types:
        offsets[node_type] = cursor
        cursor += counts[node_type]
    return offsets, cursor


def reduce_relation_matrices(data, mats, cfg, cluster_of):
    groups = {
        node_type: base.build_group_matrix(cluster_of[node_type])
        for node_type in cfg["main_types"]
    }
    reduced = {}
    for relation_id in cfg["core_link_types"]:
        src_type, dst_type, _ = data["link_defs"][relation_id]
        reduced[relation_id] = base.reduce_relation_matrix(
            mats[relation_id], src_type, dst_type, groups
        )
    return reduced


def make_dgl_graph(
    dgl_module,
    data,
    relation_mats: dict[int, sp.csr_matrix],
    node_types: list[int],
    counts: dict[int, int],
    relation_ids: list[int],
):
    offsets, total_nodes = offsets_for_types(counts, node_types)
    src_blocks = []
    dst_blocks = []
    type_blocks = []
    norm_blocks = []
    relation_names = []

    for compact_type, relation_id in enumerate(relation_ids):
        src_type, dst_type, relation_name = data["link_defs"][relation_id]
        matrix = relation_mats[relation_id].tocoo().astype(np.float32)
        local_src = matrix.row.astype(np.int64)
        local_dst = matrix.col.astype(np.int64)
        weights = matrix.data.astype(np.float32)
        incoming = np.bincount(
            local_dst, weights=weights, minlength=counts[dst_type]
        ).astype(np.float32)
        norm = weights / np.maximum(incoming[local_dst], 1e-12)
        src_blocks.append(local_src + offsets[src_type])
        dst_blocks.append(local_dst + offsets[dst_type])
        type_blocks.append(np.full(len(local_src), compact_type, dtype=np.int64))
        norm_blocks.append(norm)
        relation_names.append(relation_name)

    source = torch.from_numpy(np.concatenate(src_blocks))
    destination = torch.from_numpy(np.concatenate(dst_blocks))
    graph = dgl_module.graph((source, destination), num_nodes=total_nodes)
    edge_types = torch.from_numpy(np.concatenate(type_blocks)).long()
    edge_norm = torch.from_numpy(np.concatenate(norm_blocks)).float().unsqueeze(1)
    return graph, edge_types, edge_norm, offsets, relation_names


def convert_labels(data, target_type: int, target_offset: int, cluster_mapping=None):
    global_to_local = {
        int(global_id): local
        for local, global_id in enumerate(data["ids_by_type"][target_type])
    }

    def convert(source):
        records = []
        for global_id, node_type, label_text in source:
            if int(node_type) != target_type:
                continue
            local = global_to_local[int(global_id)]
            if cluster_mapping is None:
                model_index = target_offset + local
            else:
                model_index = target_offset + int(cluster_mapping[local])
            records.append((model_index, int(label_text), local))
        return records

    return convert(data["labels"]["label.dat"]), convert(
        data["labels"]["label.dat.test"]
    )


def build_main_views(dgl_module, data, mats, cfg, cluster_of):
    node_types = list(cfg["main_types"])
    relation_ids = list(cfg["core_link_types"])
    original_counts = {
        node_type: len(data["ids_by_type"][node_type]) for node_type in node_types
    }
    reduced_counts = {
        node_type: int(cluster_of[node_type].max()) + 1 for node_type in node_types
    }
    reduced_mats = reduce_relation_matrices(data, mats, cfg, cluster_of)

    original_graph_data = make_dgl_graph(
        dgl_module, data, mats, node_types, original_counts, relation_ids
    )
    reduced_graph_data = make_dgl_graph(
        dgl_module, data, reduced_mats, node_types, reduced_counts, relation_ids
    )
    original_graph, original_etypes, original_norm, original_offsets, names = (
        original_graph_data
    )
    reduced_graph, reduced_etypes, reduced_norm, reduced_offsets, _ = (
        reduced_graph_data
    )
    original_train, original_test = convert_labels(
        data, cfg["target_type"], original_offsets[cfg["target_type"]]
    )
    reduced_train, reduced_test, excluded_overlap, excluded_conflict = (
        gnn_data.label_records(
            data, cfg, reduced_offsets, cluster_of=cluster_of
        )
    )
    eligible_clusters = {
        record[0] - reduced_offsets[cfg["target_type"]]
        for record in reduced_train
    }
    original_train = [
        record
        for record in original_train
        if int(cluster_of[cfg["target_type"]][record[2]]) in eligible_clusters
    ]
    original_view = GraphView(
        "main_original",
        original_graph,
        original_etypes,
        original_norm,
        node_types,
        original_counts,
        original_offsets,
        relation_ids,
        names,
        original_train,
        original_test,
    )
    reduced_view = GraphView(
        "main_reduced",
        reduced_graph,
        reduced_etypes,
        reduced_norm,
        node_types,
        reduced_counts,
        reduced_offsets,
        relation_ids,
        names,
        reduced_train,
        reduced_test,
    )
    audit = {
        "excluded_due_to_test_cluster_overlap": excluded_overlap,
        "excluded_due_to_label_conflict": excluded_conflict,
    }
    return original_view, reduced_view, audit


class TypeIdentityInput(nn.Module):
    """Memory-efficient equivalent of HGB's Linear(I_type) input layers."""

    def __init__(self, counts: list[int], hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(node_count, hidden_dim) for node_count in counts
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.empty(hidden_dim)) for _ in counts
        )
        for node_count, embedding, bias in zip(counts, self.embeddings, self.biases):
            bound = 1.0 / math.sqrt(node_count)
            nn.init.uniform_(embedding.weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self):
        return torch.cat(
            [embedding.weight + bias for embedding, bias in zip(self.embeddings, self.biases)],
            dim=0,
        )


class HGBEntityClassify(nn.Module):
    """HGB EntityClassify architecture using DGL's official RelGraphConv."""

    def __init__(
        self,
        rel_graph_conv,
        counts: list[int],
        hidden_dim: int,
        class_count: int,
        relation_count: int,
        layer_count: int,
        basis_count: int,
        dropout: float,
        self_loop: bool,
    ):
        super().__init__()
        if layer_count < 2:
            raise ValueError("HGB R-GCN requires at least two layers")
        if basis_count <= 0:
            basis_count = relation_count
        self.input_layer = TypeIdentityInput(counts, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            rel_graph_conv(
                hidden_dim,
                hidden_dim,
                relation_count,
                regularizer="basis",
                num_bases=basis_count,
                activation=F.relu,
                self_loop=self_loop,
                dropout=dropout,
            )
            for _ in range(layer_count - 2)
        )
        self.output_layer = rel_graph_conv(
            hidden_dim,
            class_count,
            relation_count,
            regularizer="basis",
            num_bases=basis_count,
            activation=None,
            self_loop=self_loop,
        )

    def forward(self, graph, edge_types, edge_norm):
        hidden = self.input_layer()
        for layer in self.hidden_layers:
            hidden = layer(graph, hidden, edge_types, norm=edge_norm)
        return self.output_layer(graph, hidden, edge_types, norm=edge_norm)


def records_to_arrays(records):
    indices = np.array([record[0] for record in records], dtype=np.int64)
    labels = np.array([record[1] for record in records], dtype=np.int64)
    return indices, labels


def evaluate(labels: np.ndarray, predictions: np.ndarray):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "micro_f1": f1_score(labels, predictions, average="micro"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def train_once(
    dgl_module,
    rel_graph_conv,
    view: GraphView,
    split,
    seed: int,
    args,
):
    set_seed(seed, dgl_module)
    device = torch.device(args.device)
    train_idx, train_y, val_idx, val_y = split
    test_idx, test_y = records_to_arrays(view.test_records)
    class_count = int(max(train_y.max(), val_y.max(), test_y.max())) + 1
    model = HGBEntityClassify(
        rel_graph_conv,
        [view.counts[node_type] for node_type in view.node_types],
        args.hidden_dim,
        class_count,
        len(view.relation_ids),
        args.layers,
        args.bases,
        args.dropout,
        args.self_loop,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    graph = view.graph.to(device)
    edge_types = view.edge_types.to(device)
    edge_norm = view.edge_norm.to(device)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)
    test_idx_t = torch.from_numpy(test_idx).to(device)

    best_micro = -1.0
    best_macro = -1.0
    best_micro_epoch = 0
    best_macro_epoch = 0
    best_micro_state = None
    best_macro_state = None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph, edge_types, edge_norm)
        loss = F.cross_entropy(logits[train_idx_t], train_y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(graph, edge_types, edge_norm)[val_idx_t]
            validation_predictions = validation_logits.argmax(dim=1).cpu().numpy()
        validation_metrics = evaluate(val_y, validation_predictions)
        if validation_metrics["micro_f1"] > best_micro + 1e-8:
            best_micro = validation_metrics["micro_f1"]
            best_micro_epoch = epoch + 1
            best_micro_state = copy.deepcopy(model.state_dict())
        if validation_metrics["macro_f1"] > best_macro + 1e-8:
            best_macro = validation_metrics["macro_f1"]
            best_macro_epoch = epoch + 1
            best_macro_state = copy.deepcopy(model.state_dict())
    training_seconds = time.perf_counter() - start

    def test_checkpoint(state):
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predictions = (
                model(graph, edge_types, edge_norm)[test_idx_t]
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
        return evaluate(test_y, predictions)

    micro_checkpoint = test_checkpoint(best_micro_state)
    macro_checkpoint = test_checkpoint(best_macro_state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "graph_view": view.name,
        "seed": seed,
        "validation_micro_f1": best_micro,
        "validation_macro_f1": best_macro,
        "best_micro_epoch": best_micro_epoch,
        "best_macro_epoch": best_macro_epoch,
        "test_accuracy": micro_checkpoint["accuracy"],
        "test_micro_f1": micro_checkpoint["micro_f1"],
        "test_macro_f1": macro_checkpoint["macro_f1"],
        "macro_at_micro_checkpoint": micro_checkpoint["macro_f1"],
        "micro_at_macro_checkpoint": macro_checkpoint["micro_f1"],
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / args.epochs,
        "parameters": parameter_count,
        "nodes": view.graph.num_nodes(),
        "edges": view.graph.num_edges(),
        "relations": len(view.relation_ids),
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
        seconds_per_epoch_mean=("seconds_per_epoch", "mean"),
        parameters=("parameters", "first"),
        nodes=("nodes", "first"),
        edges=("edges", "first"),
        relations=("relations", "first"),
        train_examples=("train_examples", "first"),
        validation_examples=("validation_examples", "first"),
        test_examples=("test_examples", "first"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-thresholds",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "selected_thresholds.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "hgb_rgcn",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--bases", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--self-loop", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    dgl_module, rel_graph_conv, graphbolt_bypassed = require_dgl()

    loaded = base.load_hgb_zip(str(args.data_dir / "DBLP.zip"))
    mats = base.build_link_matrices(loaded)
    data = base._strip_unpickleable_data(loaded)
    cfg = base.CONFIGS["DBLP"]
    tau_h, tau_x = gnn_data.selected_dblp_thresholds(args.selected_thresholds)
    cluster_of, fixed_iterations = gnn_data.build_reduction(
        data, mats, cfg, tau_h, tau_x
    )

    original, reduced, audit = build_main_views(
        dgl_module, data, mats, cfg, cluster_of
    )
    views = {"main_original": original, "main_reduced": reduced}
    split_units = gnn_data.cluster_split_units(
        original.train_records,
        reduced.train_records,
        cluster_of[cfg["target_type"]],
        reduced.offsets[cfg["target_type"]],
    )

    records = []
    split_audit = {}
    for seed in seeds:
        paired = gnn_data.paired_split(
            split_units,
            seed,
            reduced.offsets[cfg["target_type"]],
        )
        split_audit[str(seed)] = {
            "train_clusters": paired["train_clusters"],
            "validation_clusters": paired["validation_clusters"],
        }
        for view_name in ("main_original", "main_reduced"):
            view = views[view_name]
            if view_name == "main_original":
                split = paired["Original"]
            else:
                split = paired["Reduced"]
            print(f"[HGB R-GCN] view={view_name} seed={seed}", flush=True)
            records.append(
                train_once(
                    dgl_module,
                    rel_graph_conv,
                    view,
                    split,
                    seed,
                    args,
                )
            )

    raw = pd.DataFrame(records)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / "hgb_rgcn_raw.csv", index=False)
    summary.to_csv(args.output_dir / "hgb_rgcn_summary.csv", index=False)
    metadata = {
        "dataset": "DBLP",
        "official_source": OFFICIAL_HGB_SOURCE,
        "implementation_scope": (
            "HGB R-GCN architecture and feats-type=3 identity protocol using "
            "DGL RelGraphConv; local code adapts graph construction, weighted "
            "reduced edges, paired splits, checkpoints, and result export."
        ),
        "views": ["main_original", "main_reduced"],
        "comparison_boundary": (
            "Both views contain the same main-structure node and relation types."
        ),
        "main_split_rule": (
            "Shared stratified 80/20 split of label-consistent reduced clusters."
        ),
        "checkpoint_rule": (
            "Validation Micro-F1 selects the accuracy/Micro-F1 checkpoint; "
            "validation Macro-F1 selects the Macro-F1 checkpoint."
        ),
        "tau_h": tau_h,
        "tau_x": tau_x,
        "fixed_point_iterations": fixed_iterations,
        "seeds": seeds,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "bases": args.bases,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "self_loop": args.self_loop,
        "device": args.device,
        "threads": args.threads,
        "dgl_version": dgl_module.__version__,
        "torch_version": torch.__version__,
        "graphbolt_import_bypassed": graphbolt_bypassed,
        "graphbolt_boundary": (
            "GraphBolt is not used by this full-batch RelGraphConv experiment."
        ),
        "main_data_audit": audit,
        "split_audit": split_audit,
    }
    with (args.output_dir / "hgb_rgcn_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run HAN on the DBLP 2x2 Guard-component ablation."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import gnn_data
import hhin_core as core
import run_guard_component_ablation as guard
import run_han_validation as han
import run_hgb_rgcn_adapter as hgb


def build_han_view(dgl, data, matrices, config, mapping, original: bool):
    original_labels, reduced_labels, audit = hgb.build_main_views(
        dgl, data, matrices, config, mapping
    )
    if original:
        label_view = original_labels
        relation_matrices = {
            relation_id: matrices[relation_id]
            for relation_id in config["core_link_types"]
        }
        features = data["feats_by_type"][config["target_type"]].toarray().astype(
            np.float32
        )
        view_name = "main_original"
    else:
        label_view = reduced_labels
        relation_matrices = hgb.reduce_relation_matrices(
            data, matrices, config, mapping
        )
        features = han.mean_cluster_features(
            data["feats_by_type"][config["target_type"]].tocsr(),
            mapping[config["target_type"]],
        )
        view_name = "main_reduced"

    started = time.perf_counter()
    heterograph = han.make_heterograph(
        dgl, data, relation_matrices, label_view.counts
    )
    graphs = []
    edge_counts = {}
    for path_name, relation_ids in han.DBLP_METAPATHS.items():
        graph = dgl.metapath_reachable_graph(
            heterograph, [f"r{relation_id}" for relation_id in relation_ids]
        )
        graphs.append(graph)
        edge_counts[path_name] = int(graph.num_edges())
    preparation_seconds = time.perf_counter() - started
    view = han.HANView(
        view_name,
        graphs,
        torch.from_numpy(features),
        label_view.train_records,
        label_view.test_records,
        label_view.counts[config["target_type"]],
        sum(label_view.counts.values()),
        int(sum(matrix.nnz for matrix in relation_matrices.values())),
        edge_counts,
    )
    return view, audit, {
        "seconds": preparation_seconds,
        "metapath_edges": edge_counts,
    }


def paired_pvalue(raw: pd.DataFrame, metric: str, configuration: str) -> float:
    original = raw[raw["configuration"] == "original"].set_index("seed")[metric]
    reduced = raw[raw["configuration"] == configuration].set_index("seed")[metric]
    common = original.index.intersection(reduced.index)
    return core.safe_paired_ttest_pvalue(
        original.loc[common].to_numpy(dtype=np.float64),
        reduced.loc[common].to_numpy(dtype=np.float64),
    )


def summarize(raw: pd.DataFrame):
    summary = raw.groupby(["configuration", "label"], as_index=False).agg(
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
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
    summary["accuracy_paired_p"] = np.nan
    summary["macro_f1_paired_p"] = np.nan
    for setting in guard.SETTINGS:
        mask = summary["configuration"] == setting.key
        summary.loc[mask, "accuracy_paired_p"] = paired_pvalue(
            raw, "test_accuracy", setting.key
        )
        summary.loc[mask, "macro_f1_paired_p"] = paired_pvalue(
            raw, "test_macro_f1", setting.key
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau-h", type=float, default=0.4)
    parser.add_argument("--tau-x", type=float, default=1.0)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    torch.set_num_threads(args.threads)
    dgl, _, graphbolt_bypassed = hgb.require_dgl()
    from dgl.nn.pytorch import GATConv

    print("[setup] Loading DBLP and computing Guard mappings", flush=True)
    data, matrices, config, partitions, attributes, terms, history = guard.load_dblp(
        args.data_dir
    )
    mappings = guard.build_mappings(
        config, partitions, attributes, terms, args.tau_h, args.tau_x
    )
    target_type = config["target_type"]

    original_labels, coarse_labels, _ = hgb.build_main_views(
        dgl, data, matrices, config, mappings["structure_only"]
    )
    units = gnn_data.cluster_split_units(
        original_labels.train_records,
        coarse_labels.train_records,
        mappings["structure_only"][target_type],
        coarse_labels.offsets[target_type],
    )
    split_units = {}
    for seed in seeds:
        train_units, validation_units = guard.split_unit_ids(units, seed)
        split_units[seed] = (train_units, validation_units)
    del original_labels, coarse_labels
    gc.collect()

    train_args = SimpleNamespace(
        device=args.device,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    records = []
    preparation = {}
    audits = {}

    print("[HAN] Building original view", flush=True)
    view, audit, prep = build_han_view(
        dgl, data, matrices, config, mappings["structure_only"], original=True
    )
    preparation["original"] = prep
    audits["original"] = audit
    for seed in seeds:
        train_units, validation_units = split_units[seed]
        train_idx, train_y = guard.original_split(units, train_units)
        val_idx, val_y = guard.original_split(units, validation_units)
        print(f"[HAN] Original seed={seed}", flush=True)
        record = han.train_once(
            dgl,
            GATConv,
            view,
            (train_idx, train_y, val_idx, val_y),
            seed,
            train_args,
        )
        record.update({"configuration": "original", "label": "Original"})
        records.append(record)
    del view
    gc.collect()

    for setting in guard.SETTINGS:
        print(f"[HAN] Building {setting.label} view", flush=True)
        view, audit, prep = build_han_view(
            dgl, data, matrices, config, mappings[setting.key], original=False
        )
        preparation[setting.key] = prep
        audits[setting.key] = audit
        mapping = mappings[setting.key][target_type]
        for seed in seeds:
            train_units, validation_units = split_units[seed]
            train_idx, train_y = guard.reduced_split(
                units, train_units, mapping, 0
            )
            val_idx, val_y = guard.reduced_split(
                units, validation_units, mapping, 0
            )
            print(f"[HAN] {setting.label} seed={seed}", flush=True)
            record = han.train_once(
                dgl,
                GATConv,
                view,
                (train_idx, train_y, val_idx, val_y),
                seed,
                train_args,
            )
            record.update(
                {"configuration": setting.key, "label": setting.label}
            )
            records.append(record)
        del view
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.DataFrame(records)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / "guard_han_raw.csv", index=False)
    summary.to_csv(args.output_dir / "guard_han_summary.csv", index=False)
    metadata = {
        "dataset": "DBLP",
        "design": "2x2 Guard-component HAN ablation",
        "tau_h": args.tau_h,
        "tau_x": args.tau_x,
        "fixed_point_iterations": len(history),
        "shared_split_basis": "Structure-only candidate classes",
        "eligible_coarse_training_clusters": len(units),
        "seeds": seeds,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "threads": args.threads,
        "feature_rule": (
            "Original 334-dimensional HGB author features; arithmetic mean "
            "within each reduced author supernode."
        ),
        "metapaths": {name: list(path) for name, path in han.DBLP_METAPATHS.items()},
        "dgl_version": dgl.__version__,
        "torch_version": torch.__version__,
        "graphbolt_import_bypassed": graphbolt_bypassed,
        "preparation": preparation,
        "data_audits": audits,
    }
    with (args.output_dir / "guard_han_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

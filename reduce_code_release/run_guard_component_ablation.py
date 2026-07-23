#!/usr/bin/env python3
"""Run the DBLP 2x2 ablation of term and attached-feature Guards."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import gnn_data
import hhin_core as core
import hhin_experiments as exp
import run_hgb_rgcn_adapter as rgcn


@dataclass(frozen=True)
class GuardSetting:
    key: str
    label: str
    use_term: bool
    use_attribute: bool


SETTINGS = (
    GuardSetting("structure_only", "Structure only", False, False),
    GuardSetting("term_guard", "Term Guard", True, False),
    GuardSetting("attribute_guard", "Attribute Guard", False, True),
    GuardSetting("both_guards", "Both Guards", True, True),
)


def load_dblp(data_dir: Path):
    loaded = core.load_hgb_zip(str(data_dir / "DBLP.zip"))
    matrices = core.build_link_matrices(loaded)
    data = core._strip_unpickleable_data(loaded)
    config = core.CONFIGS["DBLP"]
    partitions, history = core.fixed_point_partition(
        data, matrices, config["core_link_types"]
    )
    attributes = {}
    terms = {}
    for node_type in config["main_types"]:
        attributes[node_type] = core.get_raw_attr_for_type(
            data, node_type, config["strong_attr_types"]
        )
        terms[node_type] = core.derive_term_matrix(
            matrices, config["term_paths_by_type"][node_type]
        ).tocsr().astype(np.float32)
    return data, matrices, config, partitions, attributes, terms, history


def build_mappings(config, partitions, attributes, terms, tau_h: float, tau_x: float):
    mappings = {}
    for setting in SETTINGS:
        by_type = {}
        for node_type in config["main_types"]:
            node_count = len(partitions[node_type])
            raw_attribute = (
                attributes[node_type]
                if setting.use_attribute
                else sp.csr_matrix((node_count, 0), dtype=np.float32)
            )
            term_matrix = (
                terms[node_type]
                if setting.use_term
                else sp.csr_matrix((node_count, 0), dtype=np.float32)
            )
            mapping, _ = core.make_layered_threshold_mapping(
                partitions[node_type],
                raw_attribute,
                term_matrix,
                tau_term=tau_h,
                tau_attr=tau_x,
                term_sim="cosine",
            )
            by_type[node_type] = mapping
        mappings[setting.key] = by_type
    return mappings


def semantic_risk(metrics: dict) -> float:
    risks = []
    for prefix in ("pathsim", "hetesim"):
        risks.extend(
            [
                metrics[f"{prefix}_mae"] / 0.10,
                metrics[f"{prefix}_p95e"] / 0.30,
                metrics[f"{prefix}_p99e"] / 0.50,
                (1.0 - metrics[f"{prefix}_ndcg10"]) / 0.05,
            ]
        )
    return float(max(risks))


def reduction_and_semantics(data, matrices, config, mappings, tau_h, tau_x):
    original_nodes = sum(len(nodes) for nodes in data["ids_by_type"].values())
    original_main_nodes = sum(
        len(data["ids_by_type"][node_type]) for node_type in config["main_types"]
    )
    original_edges = core.count_original_edges(data)
    target_type = config["target_type"]
    target_nodes = len(data["ids_by_type"][target_type])
    non_main_nodes = original_nodes - original_main_nodes
    original_cache = core.build_original_semantic_cache(data, matrices, config)
    rows = []
    path_rows = []

    for setting in SETTINGS:
        mapping = mappings[setting.key]
        reduced_main = sum(int(values.max()) + 1 for values in mapping.values())
        reduced_target = int(mapping[target_type].max()) + 1
        reduced_nodes = reduced_main + non_main_nodes
        reduced_edges = core.count_reduced_edges(data, matrices, mapping)
        per_path, metrics = core.evaluate_semantics(
            data, matrices, config, mapping, orig_cache=original_cache
        )
        row = {
            "configuration": setting.key,
            "label": setting.label,
            "term_guard": int(setting.use_term),
            "attribute_guard": int(setting.use_attribute),
            "tau_h": tau_h if setting.use_term else np.nan,
            "tau_x": tau_x if setting.use_attribute else np.nan,
            "full_node_rr": 1.0 - reduced_nodes / original_nodes,
            "main_node_rr": 1.0 - reduced_main / original_main_nodes,
            "author_rr": 1.0 - reduced_target / target_nodes,
            "edge_rr": 1.0 - reduced_edges / original_edges,
            "reduced_full_nodes": reduced_nodes,
            "reduced_main_nodes": reduced_main,
            "reduced_authors": reduced_target,
            "reduced_edges": reduced_edges,
            "r_sem": semantic_risk(metrics),
            "feasible": int(semantic_risk(metrics) <= 1.0 + 1e-12),
        }
        row.update(metrics)
        rows.append(row)
        for path_row in per_path:
            path_rows.append(
                {
                    "configuration": setting.key,
                    "label": setting.label,
                    **path_row,
                }
            )
        print(
            f"[semantics] {setting.label}: full RR={row['full_node_rr']:.4f}, "
            f"author RR={row['author_rr']:.4f}, R_sem={row['r_sem']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(path_rows)


def clustering_ablation(data, matrices, config, mappings, seeds):
    target_type = config["target_type"]
    label_idx, labels, _ = exp.extract_target_labels(
        data, target_type, policy="first", use_test_labels=True
    )
    class_count = len(np.unique(labels))
    rows = []

    for method in ("pathsim", "hetesim"):
        print(f"[clustering] original {method}", flush=True)
        original = exp.build_similarity_feature(
            data, matrices, config, None, method=method, combine_paths="concat"
        )
        scaler = StandardScaler(with_mean=True, with_std=True)
        original = scaler.fit_transform(np.asarray(original[label_idx], dtype=np.float32))
        original_scores = {}
        for seed in seeds:
            prediction = exp.run_kmeans(original, class_count, seed, n_init=10)
            score = exp.clustering_metrics(
                original, labels, prediction, include_internal=False
            )["nmi"]
            original_scores[seed] = score
            rows.append(
                {
                    "configuration": "original",
                    "label": "Original",
                    "similarity": method,
                    "seed": seed,
                    "nmi": score,
                }
            )

        for setting in SETTINGS:
            print(f"[clustering] {setting.label} {method}", flush=True)
            reduced = exp.build_similarity_feature(
                data,
                matrices,
                config,
                mappings[setting.key],
                method=method,
                combine_paths="concat",
            )
            reduced = scaler.transform(
                np.asarray(reduced[label_idx], dtype=np.float32)
            )
            for seed in seeds:
                prediction = exp.run_kmeans(reduced, class_count, seed, n_init=10)
                score = exp.clustering_metrics(
                    reduced, labels, prediction, include_internal=False
                )["nmi"]
                rows.append(
                    {
                        "configuration": setting.key,
                        "label": setting.label,
                        "similarity": method,
                        "seed": seed,
                        "nmi": score,
                        "delta_nmi": score - original_scores[seed],
                    }
                )
            del reduced
            gc.collect()
        del original
        gc.collect()

    raw = pd.DataFrame(rows)
    summary = raw.groupby(
        ["configuration", "label", "similarity"], as_index=False, dropna=False
    ).agg(nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"))
    original = summary[summary["configuration"] == "original"][
        ["similarity", "nmi_mean"]
    ].rename(columns={"nmi_mean": "original_nmi_mean"})
    summary = summary.merge(original, on="similarity", how="left")
    summary["delta_nmi"] = summary["nmi_mean"] - summary["original_nmi_mean"]
    return raw, summary


def prediction_ablation(data, matrices, config, mappings):
    target_type = config["target_type"]
    label_idx, labels, _ = exp.extract_target_labels(
        data, target_type, policy="first", use_test_labels=True
    )
    rows = []
    for method in ("pathsim", "hetesim"):
        print(f"[label consistency] original {method}", flush=True)
        original = exp._combined_similarity_matrix(
            data, matrices, config, method, None, combine_mode="mean"
        )
        original_metric = exp._label_prediction_metrics(
            original, label_idx, labels, [10]
        )[0]
        for setting in SETTINGS:
            print(f"[label consistency] {setting.label} {method}", flush=True)
            reduced = exp._combined_similarity_matrix(
                data,
                matrices,
                config,
                method,
                mappings[setting.key],
                combine_mode="mean",
            )
            metric = exp._label_prediction_metrics(reduced, label_idx, labels, [10])[0]
            rows.append(
                {
                    "configuration": setting.key,
                    "label": setting.label,
                    "similarity": method,
                    "accuracy_original": original_metric["accuracy"],
                    "accuracy_reduced": metric["accuracy"],
                    "delta_accuracy": metric["accuracy"] - original_metric["accuracy"],
                    "macro_f1_original": original_metric["f1_macro"],
                    "macro_f1_reduced": metric["f1_macro"],
                    "delta_macro_f1": metric["f1_macro"] - original_metric["f1_macro"],
                }
            )
            del reduced
            gc.collect()
        del original
        gc.collect()
    return pd.DataFrame(rows)


def split_unit_ids(units, seed: int):
    indices = np.arange(len(units), dtype=np.int64)
    labels = np.array([unit[1] for unit in units], dtype=np.int64)
    return train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )


def original_split(units, selected):
    records = [record for unit_id in selected for record in units[unit_id][2]]
    return (
        np.array([record[0] for record in records], dtype=np.int64),
        np.array([record[1] for record in records], dtype=np.int64),
    )


def reduced_split(units, selected, mapping, offset: int):
    labels_by_cluster = {}
    for unit_id in selected:
        for _, label, local_id in units[unit_id][2]:
            cluster = int(mapping[local_id])
            previous = labels_by_cluster.setdefault(cluster, int(label))
            if previous != int(label):
                raise ValueError("A reduced training cluster contains conflicting labels")
    clusters = sorted(labels_by_cluster)
    return (
        np.array([offset + cluster for cluster in clusters], dtype=np.int64),
        np.array([labels_by_cluster[cluster] for cluster in clusters], dtype=np.int64),
    )


def paired_ttest(raw: pd.DataFrame, metric: str, configuration: str):
    original = raw[raw["configuration"] == "original"].set_index("seed")[metric]
    reduced = raw[raw["configuration"] == configuration].set_index("seed")[metric]
    common = original.index.intersection(reduced.index)
    if len(common) < 2:
        return np.nan
    return float(stats.ttest_rel(original.loc[common], reduced.loc[common]).pvalue)


def rgcn_ablation(data, matrices, config, mappings, seeds, args):
    dgl_module, rel_graph_conv, graphbolt_bypassed = rgcn.require_dgl()
    torch.set_num_threads(args.threads)
    views = {}
    audits = {}

    original_view = None
    coarse_view = None
    for setting in SETTINGS:
        original, reduced, audit = rgcn.build_main_views(
            dgl_module, data, matrices, config, mappings[setting.key]
        )
        if setting.key == "structure_only":
            original_view = original
            coarse_view = reduced
        views[setting.key] = reduced
        audits[setting.key] = audit

    units = gnn_data.cluster_split_units(
        original_view.train_records,
        coarse_view.train_records,
        mappings["structure_only"][config["target_type"]],
        coarse_view.offsets[config["target_type"]],
    )
    train_args = SimpleNamespace(
        device=args.device,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        bases=args.bases,
        dropout=args.dropout,
        self_loop=args.self_loop,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )
    records = []
    split_audit = {}

    for seed in seeds:
        train_units, validation_units = split_unit_ids(units, seed)
        split_audit[str(seed)] = {
            "train_coarse_clusters": int(len(train_units)),
            "validation_coarse_clusters": int(len(validation_units)),
        }
        train_idx, train_y = original_split(units, train_units)
        val_idx, val_y = original_split(units, validation_units)
        print(f"[R-GCN] Original seed={seed}", flush=True)
        record = rgcn.train_once(
            dgl_module,
            rel_graph_conv,
            original_view,
            (train_idx, train_y, val_idx, val_y),
            seed,
            train_args,
        )
        record.update({"configuration": "original", "label": "Original"})
        records.append(record)

        for setting in SETTINGS:
            view = views[setting.key]
            mapping = mappings[setting.key][config["target_type"]]
            offset = view.offsets[config["target_type"]]
            train_idx, train_y = reduced_split(units, train_units, mapping, offset)
            val_idx, val_y = reduced_split(units, validation_units, mapping, offset)
            print(f"[R-GCN] {setting.label} seed={seed}", flush=True)
            record = rgcn.train_once(
                dgl_module,
                rel_graph_conv,
                view,
                (train_idx, train_y, val_idx, val_y),
                seed,
                train_args,
            )
            record.update({"configuration": setting.key, "label": setting.label})
            records.append(record)

    raw = pd.DataFrame(records)
    summary = raw.groupby(
        ["configuration", "label"], as_index=False, dropna=False
    ).agg(
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
        seconds_per_epoch_mean=("seconds_per_epoch", "mean"),
        nodes=("nodes", "first"),
        edges=("edges", "first"),
        parameters=("parameters", "first"),
        train_examples=("train_examples", "first"),
        validation_examples=("validation_examples", "first"),
        test_examples=("test_examples", "first"),
    )
    summary["accuracy_paired_p"] = np.nan
    summary["macro_f1_paired_p"] = np.nan
    for setting in SETTINGS:
        mask = summary["configuration"] == setting.key
        summary.loc[mask, "accuracy_paired_p"] = paired_ttest(
            raw, "test_accuracy", setting.key
        )
        summary.loc[mask, "macro_f1_paired_p"] = paired_ttest(
            raw, "test_macro_f1", setting.key
        )
    completed_stages = (
        ["non_gnn", "rgcn"] if args.stage == "all" else [args.stage]
    )
    metadata = {
        "shared_split_basis": "Structure-only candidate classes",
        "eligible_coarse_training_clusters": len(units),
        "split_audit": split_audit,
        "view_audits": audits,
        "dgl_version": dgl_module.__version__,
        "graphbolt_import_bypassed": graphbolt_bypassed,
    }
    return raw, summary, metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("non_gnn", "rgcn", "all"), default="all")
    parser.add_argument("--tau-h", type=float, default=0.4)
    parser.add_argument("--tau-x", type=float, default=1.0)
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
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    print("[setup] Loading DBLP and computing the shared fixed-point partition", flush=True)
    data, matrices, config, partitions, attributes, terms, history = load_dblp(
        args.data_dir
    )
    print("[setup] Building the four Guard configurations", flush=True)
    mappings = build_mappings(
        config, partitions, attributes, terms, args.tau_h, args.tau_x
    )
    metadata = {
        "dataset": "DBLP",
        "design": "2x2 component ablation of term and attached-feature Guards",
        "completed_stages": completed_stages,
        "tau_h": args.tau_h,
        "tau_x": args.tau_x,
        "seeds": seeds,
        "fixed_point_iterations": len(history),
        "configurations": [setting.__dict__ for setting in SETTINGS],
    }

    if args.stage in ("non_gnn", "all"):
        reduction, paths = reduction_and_semantics(
            data, matrices, config, mappings, args.tau_h, args.tau_x
        )
        clustering_raw, clustering_summary = clustering_ablation(
            data, matrices, config, mappings, seeds
        )
        prediction = prediction_ablation(data, matrices, config, mappings)
        reduction.to_csv(
            args.output_dir / "guard_ablation_reduction_semantics.csv", index=False
        )
        paths.to_csv(
            args.output_dir / "guard_ablation_semantics_by_path.csv", index=False
        )
        clustering_raw.to_csv(
            args.output_dir / "guard_ablation_clustering_raw.csv", index=False
        )
        clustering_summary.to_csv(
            args.output_dir / "guard_ablation_clustering_summary.csv", index=False
        )
        prediction.to_csv(
            args.output_dir / "guard_ablation_label_consistency.csv", index=False
        )

    if args.stage in ("rgcn", "all"):
        rgcn_raw, rgcn_summary, rgcn_metadata = rgcn_ablation(
            data, matrices, config, mappings, seeds, args
        )
        rgcn_raw.to_csv(args.output_dir / "guard_ablation_rgcn_raw.csv", index=False)
        rgcn_summary.to_csv(
            args.output_dir / "guard_ablation_rgcn_summary.csv", index=False
        )
        metadata["rgcn"] = rgcn_metadata

    metadata_path = args.output_dir / "guard_ablation_metadata.json"
    if metadata_path.exists() and args.stage != "all":
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        completed = previous.get("completed_stages", []) + metadata["completed_stages"]
        previous.update(metadata)
        metadata = previous
        metadata["completed_stages"] = list(dict.fromkeys(completed))
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

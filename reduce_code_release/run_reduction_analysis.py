"""Measure HHIN construction cost, architecture controls, and DBLP cases."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import threading
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import hhin_core as base


def monitor_call(func):
    """Return result, elapsed seconds, incremental RSS, and absolute peak RSS."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample_memory():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(0.005)

    thread = threading.Thread(target=sample_memory, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
        peak[0] = max(peak[0], process.memory_info().rss)
    return (
        result,
        elapsed,
        max(0, peak[0] - baseline) / (1024.0**2),
        peak[0] / (1024.0**2),
    )


def load_dataset(data_dir: Path, dataset: str):
    cfg = base.CONFIGS[dataset]
    loaded = base.load_hgb_zip(str(data_dir / f"{dataset}.zip"))
    mats = base.build_link_matrices(loaded)
    data = base._strip_unpickleable_data(loaded)
    return cfg, data, mats


def selected_thresholds(selected: pd.DataFrame, dataset: str):
    row = selected.loc[selected["dataset"].str.upper() == dataset.upper()].iloc[0]
    return float(row["semantic_threshold"]), float(row["attribute_threshold"]), row


def build_hhin_once(data, mats, cfg, tau_h: float, tau_x: float):
    fixed_start = time.perf_counter()
    parts, history = base.fixed_point_partition(data, mats, cfg["core_link_types"])
    fixed_seconds = time.perf_counter() - fixed_start

    guard_start = time.perf_counter()
    cluster_of = {}
    for node_type in cfg["main_types"]:
        raw_attr = base.get_raw_attr_for_type(data, node_type, cfg["strong_attr_types"])
        high_value = base.derive_term_matrix(
            mats, cfg["term_paths_by_type"][node_type]
        ).tocsr().astype(np.float32)
        mapping, _ = base.make_layered_threshold_mapping(
            parts[node_type],
            raw_attr,
            high_value,
            tau_term=tau_h,
            tau_attr=tau_x,
            term_sim="cosine",
        )
        cluster_of[node_type] = mapping
    guard_seconds = time.perf_counter() - guard_start

    for node_type in cfg["semantic_types"]:
        cluster_of[node_type] = np.arange(
            len(data["ids_by_type"][node_type]), dtype=np.int32
        )
    return parts, history, cluster_of, fixed_seconds, guard_seconds


def reduction_counts(data, mats, cfg, cluster_of):
    original_main = sum(len(data["ids_by_type"][t]) for t in cfg["main_types"])
    reduced_main = sum(int(cluster_of[t].max()) + 1 for t in cfg["main_types"])
    semantic_nodes = sum(len(data["ids_by_type"][t]) for t in cfg["semantic_types"])
    reduced_full = reduced_main + semantic_nodes
    original_edges = base.count_original_edges(data)
    reduced_edges = base.count_reduced_edges(data, mats, cluster_of)
    return {
        "original_main_nodes": int(original_main),
        "reduced_main_nodes": int(reduced_main),
        "main_node_reduction_ratio": 1.0 - reduced_main / original_main,
        "original_full_nodes": int(data["N"]),
        "reduced_full_nodes": int(reduced_full),
        "full_node_reduction_ratio": 1.0 - reduced_full / int(data["N"]),
        "original_edges": int(original_edges),
        "reduced_edges": int(reduced_edges),
        "edge_reduction_ratio": 1.0 - reduced_edges / original_edges,
    }


def measure_reduction_cost(data, mats, cfg, tau_h, tau_x, repeats: int):
    records = []
    retained = None
    for repeat in range(repeats):
        gc.collect()
        result, total_seconds, peak_mb, absolute_peak_mb = monitor_call(
            lambda: build_hhin_once(data, mats, cfg, tau_h, tau_x)
        )
        parts, history, cluster_of, fixed_seconds, guard_seconds = result
        records.append(
            {
                "repeat": repeat + 1,
                "fixed_point_seconds": fixed_seconds,
                "guard_seconds": guard_seconds,
                "total_reduction_seconds": total_seconds,
                "incremental_peak_rss_mb": peak_mb,
                "absolute_peak_rss_mb": absolute_peak_mb,
                "fixed_point_iterations": len(history),
            }
        )
        if retained is None:
            retained = (parts, history, cluster_of)
        else:
            del parts, history, cluster_of
    return pd.DataFrame(records), retained


def build_mixed_hin_control(data, mats, cfg, tau_x: float):
    """Single-layer control: semantic-node edges enter structural refinement."""
    semantic_relation_types = {
        relation_type
        for path in cfg["term_paths_by_type"].values()
        for relation_type in path
    }
    mixed_relation_types = sorted(
        set(cfg["core_link_types"]) | semantic_relation_types
    )
    parts, history = base.fixed_point_partition(data, mats, mixed_relation_types)
    cluster_of = {}
    for node_type in cfg["main_types"]:
        raw_attr = base.get_raw_attr_for_type(data, node_type, cfg["strong_attr_types"])
        empty_semantic_channel = sp.csr_matrix(
            (len(data["ids_by_type"][node_type]), 0), dtype=np.float32
        )
        mapping, _ = base.make_layered_threshold_mapping(
            parts[node_type],
            raw_attr,
            empty_semantic_channel,
            tau_term=0.0,
            tau_attr=tau_x,
            term_sim="cosine",
        )
        cluster_of[node_type] = mapping
    for node_type in cfg["semantic_types"]:
        cluster_of[node_type] = np.arange(
            len(data["ids_by_type"][node_type]), dtype=np.int32
        )
    return parts, history, cluster_of


def evaluate_control(data, mats, cfg, cluster_of):
    cache = base.build_original_semantic_cache(data, mats, cfg)
    sample_idx = np.arange(len(data["ids_by_type"][cfg["target_type"]]))
    _, aggregate = base.evaluate_semantics(
        data, mats, cfg, cluster_of, sample_idx=sample_idx, orig_cache=cache
    )
    return aggregate


def sparse_cosine_pair(matrix, i: int, j: int):
    if matrix.shape[1] == 0:
        return np.nan
    rows = normalize(matrix[[i, j]].tocsr(), norm="l2", axis=1, copy=True)
    value = float(rows.getrow(0).multiply(rows.getrow(1)).sum())
    return float(np.clip(value, -1.0, 1.0))


def dblp_case_candidates(data, mats, cfg, parts, tau_h: float, tau_x: float):
    rows = []
    for node_type in cfg["main_types"]:
        high_value = base.derive_term_matrix(
            mats, cfg["term_paths_by_type"][node_type]
        ).tocsr().astype(np.float32)
        raw_attr = base.get_raw_attr_for_type(data, node_type, cfg["strong_attr_types"])
        classes = {}
        for local_idx, class_id in enumerate(parts[node_type]):
            classes.setdefault(int(class_id), []).append(local_idx)
        for class_id, members in classes.items():
            if len(members) < 2:
                continue
            for left, right in combinations(members, 2):
                sim_h = sparse_cosine_pair(high_value, left, right)
                sim_x = sparse_cosine_pair(raw_attr, left, right)
                pass_h = np.isnan(sim_h) or sim_h >= tau_h - 1e-8
                pass_x = np.isnan(sim_x) or sim_x >= tau_x - 1e-8
                if pass_h and pass_x:
                    decision = "accepted"
                elif not pass_h and pass_x:
                    decision = "rejected_by_high_value_guard"
                elif pass_h and not pass_x:
                    decision = "rejected_by_attached_attribute_guard"
                else:
                    decision = "rejected_by_both_guards"
                rows.append(
                    {
                        "node_type": int(node_type),
                        "candidate_class": class_id,
                        "left_global_id": int(data["ids_by_type"][node_type][left]),
                        "right_global_id": int(data["ids_by_type"][node_type][right]),
                        "high_value_cosine": sim_h,
                        "attached_attribute_cosine": sim_x,
                        "tau_h": tau_h,
                        "tau_x": tau_x,
                        "decision": decision,
                    }
                )
    all_pairs = pd.DataFrame(rows)
    representatives = []
    preferred = [
        "accepted",
        "rejected_by_high_value_guard",
        "rejected_by_attached_attribute_guard",
        "rejected_by_both_guards",
    ]
    for decision in preferred:
        subset = all_pairs.loc[all_pairs["decision"] == decision]
        if subset.empty:
            continue
        if decision == "accepted":
            chosen = subset.sort_values(
                ["high_value_cosine", "attached_attribute_cosine"],
                ascending=False,
                na_position="last",
            ).iloc[0]
        elif decision == "rejected_by_high_value_guard":
            chosen = subset.sort_values("high_value_cosine", ascending=True).iloc[0]
        elif decision == "rejected_by_attached_attribute_guard":
            chosen = subset.sort_values("attached_attribute_cosine", ascending=True).iloc[0]
        else:
            chosen = subset.sort_values(
                ["high_value_cosine", "attached_attribute_cosine"], ascending=True
            ).iloc[0]
        representatives.append(chosen)
    return all_pairs, pd.DataFrame(representatives)


def metadata(repeats: int, data_dir: Path):
    process = psutil.Process()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "psutil": psutil.__version__,
        "repeats": repeats,
        "data_dir": str(data_dir.resolve()),
        "process_id": process.pid,
        "memory_definition": "Absolute and incremental peak resident set size, sampled every 5 ms after dataset loading.",
    }


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
        default=Path(__file__).with_name("revision_additional_results"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="Measure construction cost without recomputing the mixed-HIN control.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_thresholds)

    raw_cost_rows = []
    cost_summary_rows = []
    control_rows = []
    case_pairs = None
    case_representatives = None

    for dataset in args.datasets:
        dataset = dataset.upper()
        print(f"[{dataset}] loading data", flush=True)
        cfg, data, mats = load_dataset(args.data_dir, dataset)
        tau_h, tau_x, selected_row = selected_thresholds(selected, dataset)

        print(f"[{dataset}] measuring reduction construction", flush=True)
        raw_cost, retained = measure_reduction_cost(
            data, mats, cfg, tau_h, tau_x, args.repeats
        )
        parts, history, cluster_of = retained
        raw_cost.insert(0, "dataset", dataset)
        raw_cost.insert(1, "tau_h", tau_h)
        raw_cost.insert(2, "tau_x", tau_x)
        raw_cost_rows.append(raw_cost)
        counts = reduction_counts(data, mats, cfg, cluster_of)
        cost_summary_rows.append(
            {
                "dataset": dataset,
                "tau_h": tau_h,
                "tau_x": tau_x,
                "fixed_point_seconds_mean": raw_cost["fixed_point_seconds"].mean(),
                "fixed_point_seconds_std": raw_cost["fixed_point_seconds"].std(ddof=1),
                "guard_seconds_mean": raw_cost["guard_seconds"].mean(),
                "guard_seconds_std": raw_cost["guard_seconds"].std(ddof=1),
                "total_reduction_seconds_mean": raw_cost[
                    "total_reduction_seconds"
                ].mean(),
                "total_reduction_seconds_std": raw_cost[
                    "total_reduction_seconds"
                ].std(ddof=1),
                "incremental_peak_rss_mb_max": raw_cost[
                    "incremental_peak_rss_mb"
                ].max(),
                "absolute_peak_rss_mb_max": raw_cost[
                    "absolute_peak_rss_mb"
                ].max(),
                "fixed_point_iterations": len(history),
                **counts,
            }
        )

        if not args.skip_control:
            print(f"[{dataset}] evaluating single-layer control", flush=True)
            mixed_parts, mixed_history, mixed_clusters = build_mixed_hin_control(
                data, mats, cfg, tau_x
            )
            mixed_counts = reduction_counts(data, mats, cfg, mixed_clusters)
            mixed_semantics = evaluate_control(data, mats, cfg, mixed_clusters)
            control_rows.append(
                {
                    "dataset": dataset,
                    "method": "Single-layer mixed HIN + attached-attribute constraint",
                    "tau_h": tau_h,
                    "tau_x": tau_x,
                    "fixed_point_iterations": len(mixed_history),
                    **mixed_counts,
                    **mixed_semantics,
                }
            )
            control_rows.append(
                {
                    "dataset": dataset,
                    "method": "HHIN layered reduction",
                    "tau_h": tau_h,
                    "tau_x": tau_x,
                    "fixed_point_iterations": len(history),
                    **counts,
                    "pathsim_mae": float(selected_row["avg_pathsim_mae"]),
                    "pathsim_top10": float(selected_row["avg_pathsim_top10"]),
                    "pathsim_ndcg10": float(selected_row["avg_pathsim_ndcg10"]),
                    "hetesim_mae": float(selected_row["avg_hetesim_mae"]),
                    "hetesim_top10": float(selected_row["avg_hetesim_top10"]),
                    "hetesim_ndcg10": float(selected_row["avg_hetesim_ndcg10"]),
                }
            )

        if dataset == "DBLP":
            case_pairs, case_representatives = dblp_case_candidates(
                data, mats, cfg, parts, tau_h, tau_x
            )

        del data, mats, parts, history, cluster_of
        if not args.skip_control:
            del mixed_parts, mixed_history, mixed_clusters
        gc.collect()

    raw_cost_df = pd.concat(raw_cost_rows, ignore_index=True)
    cost_summary_df = pd.DataFrame(cost_summary_rows)
    control_df = pd.DataFrame(control_rows)
    raw_cost_df.to_csv(args.output_dir / "reduction_cost_raw.csv", index=False)
    cost_summary_df.to_csv(
        args.output_dir / "reduction_cost_summary.csv", index=False
    )
    if not control_df.empty:
        control_df.to_csv(args.output_dir / "single_layer_control.csv", index=False)
    if case_pairs is not None:
        case_pairs.to_csv(args.output_dir / "dblp_case_all_candidate_pairs.csv", index=False)
    if case_representatives is not None:
        case_representatives.to_csv(
            args.output_dir / "dblp_case_representative_pairs.csv", index=False
        )
    with (args.output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata(args.repeats, args.data_dir), handle, indent=2)
    print(cost_summary_df.to_string(index=False), flush=True)
    print(control_df.to_string(index=False), flush=True)
    if case_representatives is not None:
        print(case_representatives.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

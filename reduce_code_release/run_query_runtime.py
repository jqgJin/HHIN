"""Benchmark retrieval and ranking over HHIN main-structure node types."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import hhin_experiments as exp
import runtime_helpers as rt


DEFAULT_SELECTED = Path(__file__).resolve().parent / "config" / "selected_thresholds.csv"


def short_name(name: str) -> str:
    return str(name).strip()[:1].upper()


def compact_type_label(dataset: object, target_name: object) -> str:
    """Compact label used in figures, e.g. ACM-A for ACM author nodes."""
    return f"{str(dataset).upper()}-{short_name(str(target_name))}"


def path_label(type_seq: list[int], type_names: dict[int, str]) -> str:
    half = [short_name(type_names[t]) for t in type_seq]
    full = half + half[-2::-1]
    return "".join(full)


def enumerate_main_structure_half_paths(
    data: dict,
    cfg: dict,
    type_names: dict[int, str],
    max_len: int,
) -> list[dict[str, object]]:
    main_types = set(cfg["main_types"])
    out_by_type: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for lt in cfg["core_link_types"]:
        st, et, _meaning = data["link_defs"][lt]
        if st in main_types and et in main_types:
            out_by_type[st].append((lt, et))

    rows: list[dict[str, object]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()

    def dfs(start_t: int, cur_t: int, link_seq: list[int], type_seq: list[int]) -> None:
        if len(link_seq) >= max_len:
            return
        for lt, next_t in out_by_type.get(cur_t, []):
            new_links = link_seq + [lt]
            new_types = type_seq + [next_t]
            if next_t != start_t:
                key = (start_t, tuple(new_links))
                if key not in seen:
                    seen.add(key)
                    rows.append(
                        {
                            "target_type": start_t,
                            "target_name": type_names[start_t],
                            "path": path_label(new_types, type_names),
                            "link_seq": new_links,
                            "end_type": next_t,
                            "half_path_len": len(new_links),
                        }
                    )
            dfs(start_t, next_t, new_links, new_types)

    for t in cfg["main_types"]:
        dfs(t, t, [], [t])
    return rows


def pathsim_row(B, diag: np.ndarray, idx: int) -> np.ndarray:
    return exp.base.pathsim_row_from_B(B, diag, int(idx))


def hetesim_row(BL_norm, BR_norm, idx: int) -> np.ndarray:
    return (BL_norm.getrow(int(idx)) @ BR_norm.T).toarray().ravel().astype(np.float32)


def prepare_path_factors(data: dict, mats: dict, link_seq: list[int], method: str, g_by_type: dict | None):
    method = method.lower()
    if g_by_type is None:
        if method == "pathsim":
            B = exp.base.compose_path(mats, link_seq).tocsr().astype(np.float32)
            diag = np.asarray(B.multiply(B).sum(axis=1)).ravel().astype(np.float32)
            return {"B": B, "diag": diag}
        if method == "hetesim":
            _BL, _BR, BLn, BRn = exp.base.compute_hetesim_path_factors(
                mats, data["link_defs"], link_seq, {}
            )
            return {"BLn": BLn, "BRn": BRn}
    else:
        if method == "pathsim":
            B, _end_t = exp.base.reduced_path_matrix(mats, data["link_defs"], link_seq, g_by_type)
            B = B.tocsr().astype(np.float32)
            diag = np.asarray(B.multiply(B).sum(axis=1)).ravel().astype(np.float32)
            return {"B": B, "diag": diag}
        if method == "hetesim":
            _BL, _BR, BLn, BRn = exp.base.compute_hetesim_path_factors(
                mats, data["link_defs"], link_seq, g_by_type
            )
            return {"BLn": BLn, "BRn": BRn}
    raise ValueError(f"Unknown similarity method: {method}")


def one_row_similarity(factors: dict, method: str, idx: int) -> np.ndarray:
    if method == "pathsim":
        return pathsim_row(factors["B"], factors["diag"], idx)
    if method == "hetesim":
        return hetesim_row(factors["BLn"], factors["BRn"], idx)
    raise ValueError(f"Unknown similarity method: {method}")


def query_loop(
    original_factors: dict,
    reduced_factors: dict,
    method: str,
    query_idx: np.ndarray,
    target_cluster_of: np.ndarray,
    k_values: list[int],
    topk: int,
    collect_metrics: bool,
    original_first: bool = True,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    def measure(factors, compact: bool):
        rows: list[np.ndarray] = []
        t0 = time.perf_counter()
        for qi in query_idx:
            idx = int(target_cluster_of[int(qi)]) if compact else int(qi)
            rows.append(one_row_similarity(factors, method, idx))
        similarity_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for qi, row in zip(query_idx, rows):
            self_idx = int(target_cluster_of[int(qi)]) if compact else int(qi)
            rt.topk_indices(row, topk, self_idx)
        ranking_time = time.perf_counter() - t0
        return rows, similarity_time, ranking_time

    if original_first:
        original_rows, original_similarity_time, original_ranking_time = measure(original_factors, False)
        reduced_rows, reduced_similarity_time, reduced_ranking_time = measure(reduced_factors, True)
    else:
        reduced_rows, reduced_similarity_time, reduced_ranking_time = measure(reduced_factors, True)
        original_rows, original_similarity_time, original_ranking_time = measure(original_factors, False)

    metric_rows: list[dict[str, float]] = []
    if collect_metrics:
        for qi, row_o, row_r_compact in zip(query_idx, original_rows, reduced_rows):
            expanded_r = row_r_compact[target_cluster_of]
            metric_rows.append(exp._retrieval_row_metrics(row_o, expanded_r, int(qi), k_values))

    original_time = original_similarity_time + original_ranking_time
    reduced_time = reduced_similarity_time + reduced_ranking_time
    timing = {
        "original_similarity_query_time_sec": float(original_similarity_time),
        "reduced_similarity_query_time_sec": float(reduced_similarity_time),
        "speedup_similarity_query": float(original_similarity_time / max(reduced_similarity_time, 1e-12)),
        "original_ranking_time_sec": float(original_ranking_time),
        "reduced_ranking_time_sec": float(reduced_ranking_time),
        "speedup_ranking": float(original_ranking_time / max(reduced_ranking_time, 1e-12)),
        "original_query_time_sec": float(original_time),
        "reduced_query_time_sec": float(reduced_time),
        "speedup_query": float(original_time / max(reduced_time, 1e-12)),
    }
    return timing, metric_rows


def experiment_args(datasets: list[str], selected_thresholds_csv: Path, base_dir: Path, semantic_sim: str):
    return SimpleNamespace(
        datasets=datasets,
        selected_thresholds_csv=str(selected_thresholds_csv),
        base_dir=str(base_dir),
        semantic_sim=semantic_sim,
    )


def run_experiment(args: argparse.Namespace, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = rt.parse_csv_list(args.datasets)
    methods = rt.parse_csv_list(args.methods)
    k_values = rt.parse_int_list(args.k_values)
    topk = max(k_values) if k_values else 10

    base_args = experiment_args(datasets, args.selected_thresholds_csv, args.base_dir, args.semantic_sim)
    selected_rows = exp._selected_threshold_rows(base_args)

    raw_rows: list[dict[str, object]] = []
    metric_rows_out: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []

    for dataset in datasets:
        print(f"[main-query] Dataset={dataset}: loading graph and selected reduction...")
        data, mats, cfg, parts, type_names, raw_x_by_type, term_b_by_type = exp.prepare_dataset(
            dataset, args.base_dir
        )
        selected_ds = selected_rows[selected_rows["dataset"].astype(str) == dataset]
        if selected_ds.empty:
            raise ValueError(f"No selected threshold row for dataset {dataset}")
        sel = selected_ds.iloc[0]
        cluster_of_by_type, mode, tau_sem, tau_attr = exp._build_selected_cluster_mapping(
            data, mats, cfg, parts, raw_x_by_type, term_b_by_type, sel, base_args
        )
        g_by_type = {t: exp.base.build_group_matrix(cluster_of_by_type[t]) for t in cluster_of_by_type}
        paths = enumerate_main_structure_half_paths(data, cfg, type_names, args.max_half_path_len)
        pd.DataFrame(paths).drop(columns=["link_seq"]).to_csv(
            run_dir / f"{dataset.lower()}_main_structure_paths.csv", index=False
        )

        for t in cfg["main_types"]:
            n = len(parts[t])
            c = int(np.max(cluster_of_by_type[t])) + 1
            type_rows.append(
                {
                    "dataset": dataset,
                    "type_id": t,
                    "type_name": type_names[t],
                    "original_nodes": n,
                    "reduced_clusters": c,
                    "target_type_reduction_ratio": 1.0 - c / max(n, 1),
                    "reduction_setting": mode,
                    "semantic_threshold": tau_sem,
                    "attribute_threshold": tau_attr,
                }
            )

        for pinfo in paths:
            target_t = int(pinfo["target_type"])
            target_name = str(pinfo["target_name"])
            path_name = str(pinfo["path"])
            link_seq = list(pinfo["link_seq"])
            n_target = len(parts[target_t])
            n_clusters = int(np.max(cluster_of_by_type[target_t])) + 1
            query_idx = rt.choose_query_indices(n_target, args.query_sample_size, args.query_seed)
            target_cluster_of = np.asarray(cluster_of_by_type[target_t], dtype=np.int32)

            path_rows.append(
                {
                    "dataset": dataset,
                    "target_type": target_t,
                    "target_name": target_name,
                    "path": path_name,
                    "link_seq": "-".join(str(x) for x in link_seq),
                    "half_path_len": int(pinfo["half_path_len"]),
                    "original_nodes": n_target,
                    "reduced_clusters": n_clusters,
                    "query_count": int(len(query_idx)),
                }
            )

            for method in methods:
                print(f"[main-query] {dataset} {target_name} {method} {path_name}")
                gc.collect()
                t0 = time.perf_counter()
                original_factors = prepare_path_factors(data, mats, link_seq, method, None)
                original_prep_time = time.perf_counter() - t0

                gc.collect()
                t0 = time.perf_counter()
                reduced_factors = prepare_path_factors(data, mats, link_seq, method, g_by_type)
                reduced_prep_time = time.perf_counter() - t0

                warmup_count = min(int(args.warmup_queries), len(query_idx))
                if warmup_count > 0:
                    query_loop(
                        original_factors,
                        reduced_factors,
                        method,
                        query_idx[:warmup_count],
                        target_cluster_of,
                        k_values,
                        topk,
                        collect_metrics=False,
                        original_first=False,
                    )

                for repeat in range(args.repeats):
                    collect_metrics = repeat == 0
                    original_first = repeat % 2 == 0
                    timing, metric_rows = query_loop(
                        original_factors,
                        reduced_factors,
                        method,
                        query_idx,
                        target_cluster_of,
                        k_values,
                        topk,
                        collect_metrics=collect_metrics,
                        original_first=original_first,
                    )
                    raw_rows.append(
                        {
                            "dataset": dataset,
                            "target_type": target_t,
                            "target_name": target_name,
                            "path": path_name,
                            "method": method,
                            "repeat": repeat + 1,
                            "execution_order": "original-first" if original_first else "reduced-first",
                            "original_nodes": n_target,
                            "reduced_clusters": n_clusters,
                            "query_count": int(len(query_idx)),
                            "topk": topk,
                            "original_prep_time_sec": original_prep_time,
                            "reduced_prep_time_sec": reduced_prep_time,
                            "speedup_prep": original_prep_time / max(reduced_prep_time, 1e-12),
                            **timing,
                        }
                    )
                    if collect_metrics:
                        for row in metric_rows:
                            metric_rows_out.append(
                                {
                                    "dataset": dataset,
                                    "target_type": target_t,
                                    "target_name": target_name,
                                    "path": path_name,
                                    "method": method,
                                    **row,
                                }
                            )

                del original_factors, reduced_factors

    raw = pd.DataFrame(raw_rows)
    metrics = pd.DataFrame(metric_rows_out)
    types = pd.DataFrame(type_rows)
    paths_df = pd.DataFrame(path_rows)

    raw.to_csv(run_dir / "main_structure_query_runtime_raw.csv", index=False)
    metrics.to_csv(run_dir / "main_structure_query_retrieval_metrics.csv", index=False)
    types.to_csv(run_dir / "main_structure_type_reduction.csv", index=False)
    paths_df.to_csv(run_dir / "main_structure_query_paths.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "target_type", "target_name", "method"]
    for keys, frame in raw.groupby(group_cols, sort=False):
        dataset, target_t, target_name, method = keys
        metric_frame = metrics[
            (metrics["dataset"] == dataset)
            & (metrics["target_type"] == target_t)
            & (metrics["method"] == method)
        ]
        type_frame = types[(types["dataset"] == dataset) & (types["type_id"] == target_t)]
        by_repeat = frame.groupby("repeat", sort=False)[
            [
                "original_query_time_sec",
                "reduced_query_time_sec",
                "original_similarity_query_time_sec",
                "reduced_similarity_query_time_sec",
                "original_ranking_time_sec",
                "reduced_ranking_time_sec",
                "original_prep_time_sec",
                "reduced_prep_time_sec",
            ]
        ].sum()
        summary = {
            "dataset": dataset,
            "target_type": int(target_t),
            "target_name": target_name,
            "method": method,
            "path_count": int(frame["path"].nunique()),
            "query_count": int(frame["query_count"].iloc[0]),
            "original_nodes": int(frame["original_nodes"].iloc[0]),
            "reduced_clusters": int(frame["reduced_clusters"].iloc[0]),
            "target_type_reduction_ratio": float(type_frame["target_type_reduction_ratio"].iloc[0]),
            "original_query_time_sec": float(by_repeat["original_query_time_sec"].mean()),
            "original_query_time_std_sec": float(by_repeat["original_query_time_sec"].std(ddof=1)),
            "reduced_query_time_sec": float(by_repeat["reduced_query_time_sec"].mean()),
            "reduced_query_time_std_sec": float(by_repeat["reduced_query_time_sec"].std(ddof=1)),
            "speedup_query": float(
                by_repeat["original_query_time_sec"].mean()
                / max(by_repeat["reduced_query_time_sec"].mean(), 1e-12)
            ),
            "original_similarity_query_time_sec": float(by_repeat["original_similarity_query_time_sec"].mean()),
            "reduced_similarity_query_time_sec": float(by_repeat["reduced_similarity_query_time_sec"].mean()),
            "speedup_similarity_query": float(
                by_repeat["original_similarity_query_time_sec"].mean()
                / max(by_repeat["reduced_similarity_query_time_sec"].mean(), 1e-12)
            ),
            "original_ranking_time_sec": float(by_repeat["original_ranking_time_sec"].mean()),
            "original_ranking_time_std_sec": float(by_repeat["original_ranking_time_sec"].std(ddof=1)),
            "reduced_ranking_time_sec": float(by_repeat["reduced_ranking_time_sec"].mean()),
            "reduced_ranking_time_std_sec": float(by_repeat["reduced_ranking_time_sec"].std(ddof=1)),
            "speedup_ranking": float(
                by_repeat["original_ranking_time_sec"].mean()
                / max(by_repeat["reduced_ranking_time_sec"].mean(), 1e-12)
            ),
            "original_prep_time_sec": float(by_repeat["original_prep_time_sec"].mean()),
            "reduced_prep_time_sec": float(by_repeat["reduced_prep_time_sec"].mean()),
            "speedup_prep": float(
                by_repeat["original_prep_time_sec"].mean()
                / max(by_repeat["reduced_prep_time_sec"].mean(), 1e-12)
            ),
        }
        for col in metric_frame.columns:
            if (
                col in {"mae", "p95e", "p99e", "maxe"}
                or col.startswith("overlap@")
                or col.startswith("tie_overlap@")
                or col.startswith("ndcg@")
            ):
                summary[col] = float(metric_frame[col].mean())
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "main_structure_query_runtime_summary.csv", index=False)

    dataset_summary = summary.groupby(["dataset", "method"], as_index=False).agg(
        target_types=("target_name", "nunique"),
        original_nodes_total=("original_nodes", "sum"),
        reduced_clusters_total=("reduced_clusters", "sum"),
        original_query_time_sec=("original_query_time_sec", "sum"),
        reduced_query_time_sec=("reduced_query_time_sec", "sum"),
        original_similarity_query_time_sec=("original_similarity_query_time_sec", "sum"),
        reduced_similarity_query_time_sec=("reduced_similarity_query_time_sec", "sum"),
        original_ranking_time_sec=("original_ranking_time_sec", "sum"),
        reduced_ranking_time_sec=("reduced_ranking_time_sec", "sum"),
        original_prep_time_sec=("original_prep_time_sec", "sum"),
        reduced_prep_time_sec=("reduced_prep_time_sec", "sum"),
        **{f"ndcg@{topk}": (f"ndcg@{topk}", "mean"), f"overlap@{topk}": (f"overlap@{topk}", "mean")},
    )
    dataset_repeat = raw.groupby(["dataset", "method", "repeat"], as_index=False).agg(
        original_query_time_sec=("original_query_time_sec", "sum"),
        reduced_query_time_sec=("reduced_query_time_sec", "sum"),
        original_ranking_time_sec=("original_ranking_time_sec", "sum"),
        reduced_ranking_time_sec=("reduced_ranking_time_sec", "sum"),
    )
    dataset_std = dataset_repeat.groupby(["dataset", "method"], as_index=False).agg(
        original_query_time_std_sec=("original_query_time_sec", "std"),
        reduced_query_time_std_sec=("reduced_query_time_sec", "std"),
        original_ranking_time_std_sec=("original_ranking_time_sec", "std"),
        reduced_ranking_time_std_sec=("reduced_ranking_time_sec", "std"),
    )
    dataset_summary = dataset_summary.merge(dataset_std, on=["dataset", "method"], how="left")
    dataset_summary["main_type_reduction_ratio"] = 1.0 - (
        dataset_summary["reduced_clusters_total"] / dataset_summary["original_nodes_total"].clip(lower=1)
    )
    dataset_summary["speedup_query"] = (
        dataset_summary["original_query_time_sec"] / dataset_summary["reduced_query_time_sec"].clip(lower=1e-12)
    )
    dataset_summary["speedup_similarity_query"] = (
        dataset_summary["original_similarity_query_time_sec"]
        / dataset_summary["reduced_similarity_query_time_sec"].clip(lower=1e-12)
    )
    dataset_summary["speedup_ranking"] = (
        dataset_summary["original_ranking_time_sec"]
        / dataset_summary["reduced_ranking_time_sec"].clip(lower=1e-12)
    )
    dataset_summary["speedup_prep"] = (
        dataset_summary["original_prep_time_sec"] / dataset_summary["reduced_prep_time_sec"].clip(lower=1e-12)
    )
    dataset_summary.to_csv(run_dir / "main_structure_query_runtime_dataset_summary.csv", index=False)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "datasets": datasets,
        "methods": methods,
        "repeats": args.repeats,
        "warmup_queries": args.warmup_queries,
        "execution_order": "alternating original-first and reduced-first",
        "query_sample_size": args.query_sample_size,
        "query_seed": args.query_seed,
        "max_half_path_len": args.max_half_path_len,
        "selected_thresholds_csv": args.selected_thresholds_csv.name,
        "semantic_sim": args.semantic_sim,
        "ranking_tie_rule": "similarity descending, original node identifier ascending",
        "ndcg_tie_rule": "mean gain within each tied predicted-score group",
        "note": (
            "Repeated query time measures one-row similarity computation plus top-k ranking. "
            "Reduced retrieval is performed in compact cluster space for every HHIN main type; "
            "expansion from clusters to original node identifiers is excluded from the timed path."
        ),
    }
    (run_dir / "main_structure_query_runtime_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return summary, dataset_summary


def plot_type_speedup(summary: pd.DataFrame, out_base: Path) -> None:
    frame = summary.copy()
    frame["label"] = [
        compact_type_label(dataset, target_name)
        for dataset, target_name in zip(frame["dataset"], frame["target_name"])
    ]
    labels = list(dict.fromkeys(frame["label"].tolist()))
    methods = list(dict.fromkeys(frame["method"].tolist()))
    x = np.arange(len(labels))
    width = 0.30 if len(methods) > 1 else 0.48
    colors = {"pathsim": "#2C7FB8", "hetesim": "#D95F02"}
    method_labels = {"pathsim": "PathSim", "hetesim": "HeteSim"}

    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    max_value = 1.0
    for i, method in enumerate(methods):
        vals = []
        for label in labels:
            row = frame[(frame["label"] == label) & (frame["method"] == method)]
            vals.append(float(row["speedup_query"].iloc[0]) if not row.empty else np.nan)
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            max_value = max(max_value, max(finite_vals))
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=method_labels.get(method, str(method)),
            color=colors.get(method),
            edgecolor="#202020",
            linewidth=0.55,
        )

    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, max(1.22, max_value * 1.12))
    ax.set_ylabel("Retrieval speedup")
    ax.set_xlabel("Dataset-node type")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(frameon=False, ncol=max(1, len(methods)), loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_reduction_vs_speedup(summary: pd.DataFrame, out_base: Path) -> None:
    frame_all = summary.copy()
    frame_all["label"] = [
        compact_type_label(dataset, target_name)
        for dataset, target_name in zip(frame_all["dataset"], frame_all["target_name"])
    ]
    colors = {"pathsim": "#2C7FB8", "hetesim": "#D95F02"}
    method_labels = {"pathsim": "PathSim", "hetesim": "HeteSim"}
    datasets = list(dict.fromkeys(frame_all["dataset"].astype(str).tolist()))
    y_min = max(0.88, float(frame_all["speedup_query"].min()) - 0.05)
    y_max = min(1.32, float(frame_all["speedup_query"].max()) + 0.05)

    fig, axes = plt.subplots(1, len(datasets), figsize=(9.2, 3.25), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    label_offsets = {
        "ACM-P": (9, -13),
        "ACM-A": (-8, 8),
        "ACM-S": (8, 8),
        "DBLP-A": (8, -13),
        "DBLP-P": (8, 8),
        "DBLP-V": (8, 8),
        "IMDB-M": (8, 8),
        "IMDB-D": (8, -13),
        "IMDB-A": (8, 8),
    }

    legend_handles = {}
    for ax, dataset in zip(axes, datasets):
        frame_ds = frame_all[frame_all["dataset"].astype(str) == dataset].copy()
        max_x = max(1.0, float((frame_ds["target_type_reduction_ratio"] * 100.0).max()))

        for label, frame_label in frame_ds.groupby("label", sort=False):
            x_val = float(frame_label["target_type_reduction_ratio"].iloc[0] * 100.0)
            ys = []
            for method in ["pathsim", "hetesim"]:
                row = frame_label[frame_label["method"] == method]
                if row.empty:
                    continue
                y_val = float(row["speedup_query"].iloc[0])
                ys.append(y_val)
                marker = "o" if method == "pathsim" else "s"
                handle = ax.scatter(
                    [x_val],
                    [y_val],
                    s=46,
                    marker=marker,
                    label=method_labels.get(method, str(method)),
                    color=colors.get(method),
                    edgecolor="#202020",
                    linewidth=0.55,
                    zorder=3,
                )
                legend_handles.setdefault(method_labels.get(method, str(method)), handle)
            if len(ys) >= 2:
                ax.plot([x_val, x_val], [min(ys), max(ys)], color="#7a7a7a", linewidth=0.7, zorder=2)
            if ys:
                dx, dy = label_offsets.get(label, (8, 8))
                ax.annotate(
                    label,
                    xy=(x_val, max(ys)),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.5,
                    ha="left" if dx >= 0 else "right",
                    va="center",
                )

        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_title(str(dataset), fontsize=10)
        ax.set_xlim(-max(0.8, max_x * 0.08), max_x * 1.16 + 0.8)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Reduction ratio (%)")
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Retrieval speedup")
    fig.legend(
        [legend_handles[name] for name in legend_handles],
        list(legend_handles.keys()),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.95))
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def copy_figure_pair(src_base: Path, dst_dir: Path, dst_stem: str) -> list[Path]:
    copied: list[Path] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        src = src_base.with_suffix(suffix)
        dst = dst_dir / f"{dst_stem}{suffix}"
        if dst.exists():
            idx = 1
            while (dst_dir / f"{dst_stem}_{idx}{suffix}").exists():
                idx += 1
            dst = dst_dir / f"{dst_stem}_{idx}{suffix}"
        dst.write_bytes(src.read_bytes())
        copied.append(dst)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--datasets", default="ACM,DBLP,IMDB")
    parser.add_argument("--methods", default="pathsim,hetesim")
    parser.add_argument("--selected-thresholds-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--semantic-sim", default="cosine")
    parser.add_argument("--k-values", default="10")
    parser.add_argument("--query-sample-size", type=int, default=1000)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--max-half-path-len", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=Path("main_structure_query_retrieval_out"))
    parser.add_argument("--run-label", default="main_structure_query_retrieval")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--paper-figures-dir", type=Path, default=None)
    args = parser.parse_args()

    args.base_dir = args.base_dir.resolve()
    args.selected_thresholds_csv = args.selected_thresholds_csv.resolve()
    args.out_root = args.out_root.resolve()

    run_dir = rt.fresh_run_dir(args.out_root, args.run_label, args.allow_overwrite)
    print(f"[main-query] Writing new results to: {run_dir}")
    summary, dataset_summary = run_experiment(args, run_dir)

    type_fig = run_dir / "fig_main_structure_query_speedup_by_type"
    scatter_fig = run_dir / "fig_main_structure_reduction_vs_query_speedup"
    plot_type_speedup(summary, type_fig)
    plot_reduction_vs_speedup(summary, scatter_fig)

    copied: list[Path] = []
    if args.paper_figures_dir is not None:
        paper_dir = args.paper_figures_dir.resolve()
        copied.extend(copy_figure_pair(type_fig, paper_dir, "fig_main_structure_query_speedup_by_type"))
        copied.extend(copy_figure_pair(scatter_fig, paper_dir, "fig_main_structure_reduction_vs_query_speedup"))

    print("[main-query] Type-level summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("[main-query] Dataset-level summary:")
    print(dataset_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for path in copied:
        print(f"[main-query] Copied figure to paper: {path}")


if __name__ == "__main__":
    main()

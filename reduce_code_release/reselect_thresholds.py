#!/usr/bin/env python3
"""Select HHIN thresholds from a completed semantic-risk grid."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


METRIC_PREFIXES = ("pathsim", "hetesim")

DEFAULT_SELECTED_COLUMNS = [
    "dataset", "reduction_setting", "selection_source", "semantic_sim",
    "semantic_threshold", "attribute_threshold",
    "fullgraph_reduction_ratio", "main_reduction_ratio", "fullgraph_edge_reduction_ratio",
    "risk_semantic", "risk_pathsim", "risk_hetesim", "normalized_feasible",
    "avg_pathsim_mae", "avg_pathsim_p95e", "avg_pathsim_p99e",
    "avg_pathsim_top10", "avg_pathsim_ndcg10",
    "avg_hetesim_mae", "avg_hetesim_p95e", "avg_hetesim_p99e",
    "avg_hetesim_top10", "avg_hetesim_ndcg10",
    "pathsim_controllable", "hetesim_controllable", "joint_controllable",
]


def parse_float_list(s: str | None) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def adjusted_error_budget(base: float, scale: float) -> float:
    return float(base) * float(scale)


def adjusted_floor_budget(base_floor: float, scale: float) -> float:
    """Scale the allowed loss away from 1.0 for metrics such as nDCG / Top10.

    Example: base_floor=0.95, scale=1.25 -> 1 - 1.25*(1-0.95) = 0.9375.
    scale<1 is stricter; scale>1 is looser.
    """
    base_floor = float(base_floor)
    scale = float(scale)
    val = 1.0 - scale * (1.0 - base_floor)
    return float(np.clip(val, 0.0, 1.0))


def safe_ratio(value: float, denom: float) -> float:
    denom = max(float(denom), 1e-12)
    return float(value) / denom


def compute_normalized_risk(
    df: pd.DataFrame,
    *,
    scale: float = 1.0,
    max_mae: float = 0.10,
    max_p95e: float = 0.30,
    max_p99e: float = 0.50,
    min_ndcg10: float = 0.95,
    min_top10: float = 0.90,
    include_top10_in_risk: bool = False,
) -> pd.DataFrame:
    """Add normalized semantic-risk columns to a threshold-scan DataFrame."""
    out = df.copy()

    eps_mae = adjusted_error_budget(max_mae, scale)
    eps_p95e = adjusted_error_budget(max_p95e, scale)
    eps_p99e = adjusted_error_budget(max_p99e, scale)
    th_ndcg = adjusted_floor_budget(min_ndcg10, scale)
    th_top10 = adjusted_floor_budget(min_top10, scale)

    out["risk_budget_scale"] = float(scale)
    out["budget_max_mae"] = eps_mae
    out["budget_max_p95e"] = eps_p95e
    out["budget_max_p99e"] = eps_p99e
    out["budget_min_ndcg10"] = th_ndcg
    out["budget_min_top10"] = th_top10
    out["top10_used_in_risk"] = int(bool(include_top10_in_risk))

    for prefix in METRIC_PREFIXES:
        mae_col = f"avg_{prefix}_mae"
        p95_col = f"avg_{prefix}_p95e"
        p99_col = f"avg_{prefix}_p99e"
        ndcg_col = f"avg_{prefix}_ndcg10"
        top10_col = f"avg_{prefix}_top10"

        for required in [mae_col, p95_col, p99_col, ndcg_col, top10_col]:
            if required not in out.columns:
                raise ValueError(f"Missing required column in summary CSV: {required}")

        out[f"risk_{prefix}_mae"] = out[mae_col].astype(float) / max(eps_mae, 1e-12)
        out[f"risk_{prefix}_p95e"] = out[p95_col].astype(float) / max(eps_p95e, 1e-12)
        out[f"risk_{prefix}_p99e"] = out[p99_col].astype(float) / max(eps_p99e, 1e-12)
        out[f"risk_{prefix}_ndcg10"] = (1.0 - out[ndcg_col].astype(float)) / max(1.0 - th_ndcg, 1e-12)
        out[f"risk_{prefix}_top10_diagnostic"] = (1.0 - out[top10_col].astype(float)) / max(1.0 - th_top10, 1e-12)

        components = [
            f"risk_{prefix}_mae",
            f"risk_{prefix}_p95e",
            f"risk_{prefix}_p99e",
            f"risk_{prefix}_ndcg10",
        ]
        if include_top10_in_risk:
            components.append(f"risk_{prefix}_top10_diagnostic")
        out[f"risk_{prefix}"] = out[components].max(axis=1)

    out["risk_semantic"] = out[["risk_pathsim", "risk_hetesim"]].max(axis=1)
    out["normalized_feasible"] = (out["risk_semantic"] <= 1.0 + 1e-12).astype(int)
    out["top10_pathsim_diagnostic_ok"] = (out["risk_pathsim_top10_diagnostic"] <= 1.0 + 1e-12).astype(int)
    out["top10_hetesim_diagnostic_ok"] = (out["risk_hetesim_top10_diagnostic"] <= 1.0 + 1e-12).astype(int)
    out["top10_joint_diagnostic_ok"] = (out["top10_pathsim_diagnostic_ok"] & out["top10_hetesim_diagnostic_ok"]).astype(int)
    return out


def select_one_group(
    group: pd.DataFrame,
    *,
    objective_col: str,
    fallback: str = "min_risk",
    risk_tie_tol: float = 1e-12,
) -> pd.Series | None:
    """Select one threshold row from one dataset/reduction-setting group."""
    if objective_col not in group.columns:
        raise ValueError(f"Objective column not found: {objective_col}")
    feasible = group[group["normalized_feasible"].astype(int) == 1].copy()

    if not feasible.empty:
        # Lexicographic tie-break: risk, edge reduction, nDCG, then Top10.
        sort_cols = [
            objective_col,
            "risk_semantic",
            "fullgraph_edge_reduction_ratio",
            "avg_pathsim_ndcg10",
            "avg_hetesim_ndcg10",
            "avg_pathsim_top10",
            "avg_hetesim_top10",
        ]
        ascending = [False, True, False, False, False, False, False]
        row = feasible.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].copy()
        row["selection_source"] = "normalized_risk_feasible_max_node_reduction"
        return row

    if fallback == "none":
        return None

    if fallback == "min_risk":
        sort_cols = [
            "risk_semantic",
            objective_col,
            "fullgraph_edge_reduction_ratio",
            "avg_pathsim_ndcg10",
            "avg_hetesim_ndcg10",
        ]
        ascending = [True, False, False, False, False]
        row = group.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].copy()
        row["selection_source"] = "fallback_min_semantic_risk_no_feasible_threshold"
        return row

    if fallback == "max_node_reduction":
        sort_cols = [objective_col, "risk_semantic", "fullgraph_edge_reduction_ratio"]
        ascending = [False, True, False]
        row = group.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].copy()
        row["selection_source"] = "fallback_max_node_reduction_no_feasible_threshold"
        return row

    raise ValueError("fallback must be one of: none, min_risk, max_node_reduction")


def select_thresholds(
    df_risk: pd.DataFrame,
    *,
    objective_col: str = "fullgraph_reduction_ratio",
    fallback: str = "min_risk",
    group_by_setting: bool = True,
) -> pd.DataFrame:
    group_cols = ["dataset"]
    if group_by_setting and "reduction_setting" in df_risk.columns:
        group_cols.append("reduction_setting")

    rows = []
    for _, group in df_risk.groupby(group_cols, dropna=False):
        row = select_one_group(group, objective_col=objective_col, fallback=fallback)
        if row is not None:
            rows.append(row)
    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected

    for col in DEFAULT_SELECTED_COLUMNS:
        if col not in selected.columns:
            selected[col] = np.nan
    front = [c for c in DEFAULT_SELECTED_COLUMNS if c in selected.columns]
    rest = [c for c in selected.columns if c not in front]
    selected = selected[front + rest]
    return selected


def budget_sensitivity(
    df: pd.DataFrame,
    *,
    scales: Iterable[float],
    objective_col: str,
    fallback: str,
    group_by_setting: bool,
    max_mae: float,
    max_p95e: float,
    max_p99e: float,
    min_ndcg10: float,
    min_top10: float,
    include_top10_in_risk: bool,
) -> pd.DataFrame:
    rows = []
    for scale in scales:
        df_risk = compute_normalized_risk(
            df,
            scale=float(scale),
            max_mae=max_mae,
            max_p95e=max_p95e,
            max_p99e=max_p99e,
            min_ndcg10=min_ndcg10,
            min_top10=min_top10,
            include_top10_in_risk=include_top10_in_risk,
        )
        sel = select_thresholds(
            df_risk,
            objective_col=objective_col,
            fallback=fallback,
            group_by_setting=group_by_setting,
        )
        if sel.empty:
            continue
        keep = [
            "dataset", "reduction_setting", "selection_source", "risk_budget_scale",
            "semantic_threshold", "attribute_threshold", objective_col,
            "fullgraph_reduction_ratio", "main_reduction_ratio", "fullgraph_edge_reduction_ratio",
            "risk_semantic", "risk_pathsim", "risk_hetesim", "normalized_feasible",
            "avg_pathsim_mae", "avg_pathsim_p95e", "avg_pathsim_p99e", "avg_pathsim_top10", "avg_pathsim_ndcg10",
            "avg_hetesim_mae", "avg_hetesim_p95e", "avg_hetesim_p99e", "avg_hetesim_top10", "avg_hetesim_ndcg10",
        ]
        for c in keep:
            if c not in sel.columns:
                sel[c] = np.nan
        rows.extend(sel[keep].to_dict("records"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-select HHIN tau_s/tau_x from semantic_selection_summary.csv using normalized semantic risk."
    )
    ap.add_argument("--summary-csv", type=str, default="semantic_selection_summary.csv",
                    help="Path to the existing 10x10 threshold scan CSV.")
    ap.add_argument("--out-dir", type=str, default="./hhin_reselect_out")
    ap.add_argument("--selected-name", type=str, default="selected_thresholds_for_clustering.csv")
    ap.add_argument("--risk-summary-name", type=str, default="semantic_selection_summary_with_normalized_risk.csv")

    ap.add_argument("--risk-budget-scale", type=float, default=1.0,
                    help="Scale semantic budgets. <1 stricter, >1 looser.")
    ap.add_argument("--max-mae", type=float, default=0.10)
    ap.add_argument("--max-p95e", type=float, default=0.30)
    ap.add_argument("--max-p99e", type=float, default=0.50)
    ap.add_argument("--min-ndcg10", type=float, default=0.95)
    ap.add_argument("--min-top10", type=float, default=0.90,
                    help="Top10 budget used for diagnostics by default; included in hard risk only with --include-top10-in-risk.")
    ap.add_argument("--include-top10-in-risk", action="store_true",
                    help="Use Top10 overlap as a hard normalized-risk component. Default: diagnostic only.")

    ap.add_argument("--objective-col", type=str, default="fullgraph_reduction_ratio",
                    choices=["fullgraph_reduction_ratio", "main_reduction_ratio"],
                    help="Node-reduction objective to maximize under semantic risk budget.")
    ap.add_argument("--fallback", choices=["none", "min_risk", "max_node_reduction"], default="min_risk",
                    help="What to select if a dataset has no feasible threshold.")
    ap.add_argument("--no-group-by-setting", action="store_true",
                    help="Select one row per dataset only, ignoring reduction_setting groups.")

    ap.add_argument("--budget-scales", type=str, default="",
                    help="Optional comma-separated scales, e.g. 0.75,1.0,1.25,1.5.")
    ap.add_argument("--sensitivity-name", type=str, default="budget_sensitivity_summary.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary CSV: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_path}")

    df_risk = compute_normalized_risk(
        df,
        scale=args.risk_budget_scale,
        max_mae=args.max_mae,
        max_p95e=args.max_p95e,
        max_p99e=args.max_p99e,
        min_ndcg10=args.min_ndcg10,
        min_top10=args.min_top10,
        include_top10_in_risk=args.include_top10_in_risk,
    )
    risk_path = out_dir / args.risk_summary_name
    df_risk.to_csv(risk_path, index=False)

    selected = select_thresholds(
        df_risk,
        objective_col=args.objective_col,
        fallback=args.fallback,
        group_by_setting=not args.no_group_by_setting,
    )
    selected_path = out_dir / args.selected_name
    selected.to_csv(selected_path, index=False)

    scales = parse_float_list(args.budget_scales)
    sensitivity_path = None
    if scales:
        sens = budget_sensitivity(
            df,
            scales=scales,
            objective_col=args.objective_col,
            fallback=args.fallback,
            group_by_setting=not args.no_group_by_setting,
            max_mae=args.max_mae,
            max_p95e=args.max_p95e,
            max_p99e=args.max_p99e,
            min_ndcg10=args.min_ndcg10,
            min_top10=args.min_top10,
            include_top10_in_risk=args.include_top10_in_risk,
        )
        sensitivity_path = out_dir / args.sensitivity_name
        sens.to_csv(sensitivity_path, index=False)

    print(f"[OK] Wrote risk-augmented summary: {risk_path}")
    print(f"[OK] Wrote selected thresholds: {selected_path}")
    if sensitivity_path:
        print(f"[OK] Wrote budget sensitivity summary: {sensitivity_path}")
    if not selected.empty:
        cols = [
            "dataset", "reduction_setting", "selection_source",
            "semantic_threshold", "attribute_threshold", args.objective_col,
            "risk_semantic", "risk_pathsim", "risk_hetesim", "normalized_feasible",
            "avg_pathsim_top10", "avg_pathsim_ndcg10", "avg_hetesim_top10", "avg_hetesim_ndcg10",
        ]
        cols = [c for c in cols if c in selected.columns]
        print("\nSelected thresholds:")
        print(selected[cols].to_string(index=False))


if __name__ == "__main__":
    main()

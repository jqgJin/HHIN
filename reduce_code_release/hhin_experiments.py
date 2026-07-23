#!/usr/bin/env python3
"""HHIN reduction, clustering, retrieval, and label-neighborhood experiments."""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    normalized_mutual_info_score,
    precision_score,
    silhouette_score,
    calinski_harabasz_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def _import_base_module():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, os.getcwd())
    try:
        import hhin_core as base
        return base
    except Exception as e:
        raise ImportError(
            "Cannot import hhin_core.py. Keep the release scripts in one directory. "
            "Original error: %r" % e
        )

base = _import_base_module()


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def dataset_zip_path(base_dir: str, dataset: str) -> str:
    cand = [
        Path(base_dir) / f"{dataset}.zip",
        Path(base_dir) / dataset / f"{dataset}.zip",
        Path(base_dir) / dataset.upper() / f"{dataset.upper()}.zip",
        Path(base_dir) / dataset.lower() / f"{dataset.lower()}.zip",
    ]
    for p in cand:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Cannot find zip for {dataset}. Tried: {cand}")


def prepare_dataset(dataset: str, base_dir: str):
    zpath = dataset_zip_path(base_dir, dataset)
    data = base.load_hgb_zip(zpath)
    mats = base.build_link_matrices(data)
    cfg = base.CONFIGS[dataset]
    parts, _history = base.fixed_point_partition(data, mats, cfg['core_link_types'])
    type_names = {int(k): v for k, v in base.TYPE_NAME_CACHE.get(zpath, {}).items()}
    rawX_by_type = {t: base.get_raw_attr_for_type(data, t, cfg['strong_attr_types']) for t in cfg['main_types']}
    termB_by_type = {}
    for t in cfg['main_types']:
        if t in cfg.get('term_paths_by_type', {}):
            termB_by_type[t] = base.derive_term_matrix(mats, cfg['term_paths_by_type'][t])
        else:
            termB_by_type[t] = sp.csr_matrix((len(parts[t]), 0), dtype=np.float32)
    return data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type


def build_clusters_for_mode(cfg, parts, rawX_by_type, termB_by_type, mode: str,
                            tau_sem: Optional[float], tau_attr: Optional[float],
                            semantic_sim: str):
    """Build main-structure cluster mappings for one reduction mode."""
    cluster_of_by_type = {}
    per_type = []
    for t in cfg['main_types']:
        n = len(parts[t])
        if mode == 'exact':
            cluster_of = np.arange(n, dtype=np.int32)
            clusters = [[i] for i in range(n)]
        elif mode == 'semantic_channel_only':
            # tau_attr = 0 preserves only the fixed-point candidate restriction.
            cluster_of, clusters = base.make_layered_threshold_mapping(
                parts[t], rawX_by_type[t], termB_by_type[t],
                tau_term=float(tau_sem), tau_attr=0.0, term_sim=semantic_sim
            )
        elif mode == 'layered_semantic_attribute':
            if tau_attr is None:
                raise ValueError('layered_semantic_attribute requires --attr-taus / attribute_threshold.')
            cluster_of, clusters = base.make_layered_threshold_mapping(
                parts[t], rawX_by_type[t], termB_by_type[t],
                tau_term=float(tau_sem), tau_attr=float(tau_attr), term_sim=semantic_sim
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        cluster_of_by_type[t] = cluster_of
        per_type.append({
            'type_id': t,
            'original_nodes': n,
            'reduced_nodes': len(clusters),
            'reduction_ratio': 1.0 - len(clusters) / max(n, 1),
        })
    return cluster_of_by_type, per_type


def add_semantic_singletons(cluster_of_by_type: dict, data: dict, cfg: dict):
    for t in cfg.get('semantic_types', []):
        n = len(data['ids_by_type'].get(t, []))
        cluster_of_by_type[t] = np.arange(n, dtype=np.int32)
    return cluster_of_by_type


def run_reduction_contrast(args):
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_type_rows = []

    semantic_taus = parse_float_list(args.semantic_taus)
    attr_taus = parse_float_list(args.attr_taus)

    for dataset in args.datasets:
        t0 = time.perf_counter()
        print(f'[reduction] Dataset={dataset}: loading graph and target-node BoW attributes...')
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        full_nodes = int(sum(len(v) for v in data['ids_by_type'].values()))
        main_nodes = int(sum(len(parts[t]) for t in cfg['main_types']))
        semantic_nodes = int(sum(len(data['ids_by_type'].get(t, [])) for t in cfg.get('semantic_types', [])))
        original_edges = base.count_original_edges(data)

        modes_and_pairs = []
        modes_and_pairs.append(('exact', None, None))
        if getattr(args, 'include_term_only_ablation', False):
            for ts in semantic_taus:
                modes_and_pairs.append(('semantic_channel_only', ts, None))
        for ts in semantic_taus:
            for tx in attr_taus:
                modes_and_pairs.append(('layered_semantic_attribute', ts, tx))

        for mode, ts, tx in modes_and_pairs:
            if mode == 'exact':
                label = 'exact'
            elif mode == 'semantic_channel_only':
                label = f'S{ts:g}'
            else:
                label = f'S{ts:g}_X{tx:g}'
            print(f'[reduction] Dataset={dataset} setting={mode} threshold={label} ...')
            cluster_of_by_type, per_type = build_clusters_for_mode(
                cfg, parts, rawX_by_type, termB_by_type, mode, ts, tx, args.semantic_sim
            )
            add_semantic_singletons(cluster_of_by_type, data, cfg)
            reduced_main_nodes = sum(int(cluster_of_by_type[t].max()) + 1 for t in cfg['main_types'])
            reduced_semantic_nodes = semantic_nodes
            reduced_full_nodes = reduced_main_nodes + reduced_semantic_nodes
            reduced_edges = base.count_reduced_edges(data, mats, cluster_of_by_type)
            row = {
                'dataset': dataset,
                'mode': mode,
                'semantic_sim': args.semantic_sim,
                'threshold': label,
                'semantic_threshold': ts,
                'attribute_threshold': tx,
                'full_nodes': full_nodes,
                'main_nodes': main_nodes,
                'semantic_nodes': semantic_nodes,
                'reduced_full_nodes': reduced_full_nodes,
                'reduced_main_nodes': reduced_main_nodes,
                'reduced_edges': reduced_edges,
                'original_edges': original_edges,
                'fullgraph_reduction_ratio': 1.0 - reduced_full_nodes / max(full_nodes, 1),
                'main_reduction_ratio': 1.0 - reduced_main_nodes / max(main_nodes, 1),
                'fullgraph_edge_reduction_ratio': 1.0 - reduced_edges / max(original_edges, 1),
                'runtime_sec_dataset_elapsed': time.perf_counter() - t0,
            }
            rows.append(row)
            print(
                f"[reduction] Dataset={dataset} threshold={label} "
                f"node_reduction={row['fullgraph_reduction_ratio']:.4f} "
                f"edge_reduction={row['fullgraph_edge_reduction_ratio']:.4f} "
                f"elapsed={row['runtime_sec_dataset_elapsed']:.2f}s"
            )
            for pr in per_type:
                pr = dict(pr)
                pr.update({
                    'dataset': dataset,
                    'mode': mode,
                    'semantic_sim': args.semantic_sim,
                    'threshold': label,
                    'semantic_threshold': ts,
                    'attribute_threshold': tx,
                    'type_name': type_names.get(pr['type_id'], str(pr['type_id'])),
                })
                per_type_rows.append(pr)

    df = pd.DataFrame(rows)
    per_type_df = pd.DataFrame(per_type_rows)
    df.to_csv(outdir / 'hhin_semantic_guard_reduction_contrast_summary.csv', index=False)
    per_type_df.to_csv(outdir / 'hhin_semantic_guard_reduction_contrast_per_type.csv', index=False)
    if args.write_plots:
        plot_reduction_contrast(df, outdir)
    return df


def plot_reduction_contrast(df: pd.DataFrame, outdir: Path):
    import matplotlib.pyplot as plt
    for dataset, sub in df[df['mode'] != 'exact'].groupby('dataset'):
        # Optional term-only ablation curve.
        sem = sub[sub['mode'] == 'semantic_channel_only'].sort_values('semantic_threshold')
        if not sem.empty:
            plt.figure(figsize=(7.2, 4.8))
            plt.plot(sem['semantic_threshold'], sem['fullgraph_reduction_ratio'] * 100, marker='o', label='full graph')
            plt.plot(sem['semantic_threshold'], sem['main_reduction_ratio'] * 100, marker='s', label='main nodes')
            plt.xlabel('term/keyword threshold')
            plt.ylabel('reduction ratio (%)')
            plt.title(f'{dataset}: term-only reduction ablation')
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(outdir / f'{dataset.lower()}_term_only_reduction_curve.png', dpi=220)
            plt.close()

        # Layered heatmap for fullgraph reduction
        layered = sub[sub['mode'] == 'layered_semantic_attribute']
        if not layered.empty:
            pivot = layered.pivot_table(index='attribute_threshold', columns='semantic_threshold',
                                        values='fullgraph_reduction_ratio', aggfunc='mean')
            plt.figure(figsize=(7.4, 5.3))
            plt.imshow(pivot.values * 100, aspect='auto', origin='lower')
            plt.xticks(np.arange(len(pivot.columns)), [f'{x:g}' for x in pivot.columns], rotation=45)
            plt.yticks(np.arange(len(pivot.index)), [f'{x:g}' for x in pivot.index])
            plt.xlabel('semantic-channel threshold')
            plt.ylabel('node-attached attribute threshold')
            plt.title(f'{dataset}: layered semantic-attribute Guard full-graph reduction (%)')
            plt.colorbar(label='reduction ratio (%)')
            plt.tight_layout()
        plt.savefig(outdir / f'{dataset.lower()}_layered_semantic_attribute_reduction_heatmap.png', dpi=220)
        plt.close()


SEMANTIC_METRIC_COLUMNS = [
    'avg_pathsim_mae', 'avg_pathsim_p95e', 'avg_pathsim_p99e',
    'avg_pathsim_top10', 'avg_pathsim_ndcg10', 'avg_pathsim_pvalue',
    'avg_hetesim_mae', 'avg_hetesim_p95e', 'avg_hetesim_p99e',
    'avg_hetesim_top10', 'avg_hetesim_ndcg10', 'avg_hetesim_pvalue',
]

SELECTED_THRESHOLD_COLUMNS = [
    'dataset', 'reduction_setting', 'selection_source', 'semantic_sim',
    'semantic_threshold', 'attribute_threshold',
    'fullgraph_reduction_ratio', 'fullgraph_edge_reduction_ratio',
    'reduction_gain',
    'risk_pathsim', 'risk_hetesim', 'risk_semantic', 'normalized_feasible',
    'top10_pathsim_risk_diagnostic', 'top10_hetesim_risk_diagnostic',
    'avg_pathsim_mae', 'avg_pathsim_p95e', 'avg_pathsim_p99e',
    'avg_pathsim_top10', 'avg_pathsim_ndcg10',
    'avg_hetesim_mae', 'avg_hetesim_p95e', 'avg_hetesim_p99e',
    'avg_hetesim_top10', 'avg_hetesim_ndcg10',
    'pathsim_controllable', 'hetesim_controllable', 'joint_controllable',
]


def semantic_quality_score_from_row(row: pd.Series) -> float:
    return float(
        row.get('avg_pathsim_ndcg10', 0.0) + row.get('avg_hetesim_ndcg10', 0.0) +
        row.get('avg_pathsim_top10', 0.0) + row.get('avg_hetesim_top10', 0.0) -
        row.get('avg_pathsim_mae', 0.0) - row.get('avg_hetesim_mae', 0.0) -
        0.50 * row.get('avg_pathsim_p95e', 0.0) - 0.50 * row.get('avg_hetesim_p95e', 0.0) -
        0.25 * row.get('avg_pathsim_p99e', 0.0) - 0.25 * row.get('avg_hetesim_p99e', 0.0)
    )


def add_semantic_controllability(row: dict, args) -> dict:
    pathsim_ok = (
        row['avg_pathsim_mae'] <= args.max_pathsim_mae and
        row['avg_pathsim_p95e'] <= args.max_pathsim_p95e and
        row['avg_pathsim_p99e'] <= args.max_pathsim_p99e and
        row['avg_pathsim_top10'] >= args.min_pathsim_top10 and
        row['avg_pathsim_ndcg10'] >= args.min_pathsim_ndcg10 and
        row['avg_pathsim_pvalue'] >= args.min_pathsim_pvalue
    )
    hetesim_ok = (
        row['avg_hetesim_mae'] <= args.max_hetesim_mae and
        row['avg_hetesim_p95e'] <= args.max_hetesim_p95e and
        row['avg_hetesim_p99e'] <= args.max_hetesim_p99e and
        row['avg_hetesim_top10'] >= args.min_hetesim_top10 and
        row['avg_hetesim_ndcg10'] >= args.min_hetesim_ndcg10 and
        row['avg_hetesim_pvalue'] >= args.min_hetesim_pvalue
    )
    row['pathsim_controllable'] = int(pathsim_ok)
    row['hetesim_controllable'] = int(hetesim_ok)
    row['joint_controllable'] = int(pathsim_ok and hetesim_ok)
    return row




def _safe_ratio(value: float, budget: float) -> float:
    if budget is None or budget <= 0:
        return 0.0
    if value is None or pd.isna(value):
        return np.inf
    return float(value) / float(budget)


def _retention_risk(value: float, min_value: float) -> float:
    """Normalize a retention metric such as nDCG@10 or Top10.

    If a metric must be >= min_value, the normalized violation is
    (1 - value) / (1 - min_value). A value equal to min_value gives risk 1.
    """
    denom = max(1.0 - float(min_value), 1e-12)
    if value is None or pd.isna(value):
        return np.inf
    return float(1.0 - value) / denom


def _scaled_budget(args, name: str, default: float) -> float:
    scale = float(getattr(args, 'risk_budget_scale', 1.0) or 1.0)
    val = float(getattr(args, name, default))
    return val * scale


def _scaled_retention_budget(args, name: str, default: float) -> float:
    """Relax or tighten a retention budget with the same semantic scale.

    scale < 1 is stricter; scale > 1 is looser.
    Example: min_ndcg=0.95, scale=1.5 -> 1 - 1.5*(1-0.95)=0.925.
    """
    scale = float(getattr(args, 'risk_budget_scale', 1.0) or 1.0)
    val = float(getattr(args, name, default))
    return max(0.0, min(1.0, 1.0 - scale * (1.0 - val)))


def normalized_path_risk(row: pd.Series, prefix: str, args) -> Tuple[float, float]:
    """Return (main_risk, top10_diagnostic_risk) for PathSim or HeteSim.

    Main risk uses MAE/P95E/P99E/nDCG@10 by default. Top10 is kept as a
    diagnostic because it is a discrete set-overlap metric and can be unstable
    when many nodes have nearly tied similarity scores. It can be included in
    the hard normalized risk by passing --include-top10-in-risk.
    """
    max_mae = _scaled_budget(args, f'max_{prefix}_mae', 0.10)
    max_p95 = _scaled_budget(args, f'max_{prefix}_p95e', 0.30)
    max_p99 = _scaled_budget(args, f'max_{prefix}_p99e', 0.50)
    min_ndcg = _scaled_retention_budget(args, f'min_{prefix}_ndcg10', 0.95)
    min_top10 = _scaled_retention_budget(args, f'min_{prefix}_top10', 0.90)
    risks = [
        _safe_ratio(row.get(f'avg_{prefix}_mae', np.nan), max_mae),
        _safe_ratio(row.get(f'avg_{prefix}_p95e', np.nan), max_p95),
        _safe_ratio(row.get(f'avg_{prefix}_p99e', np.nan), max_p99),
        _retention_risk(row.get(f'avg_{prefix}_ndcg10', np.nan), min_ndcg),
    ]
    top10_risk = _retention_risk(row.get(f'avg_{prefix}_top10', np.nan), min_top10)
    if bool(getattr(args, 'include_top10_in_risk', False)):
        risks.append(top10_risk)
    return float(np.nanmax(risks)), float(top10_risk)


def add_normalized_risk_columns(df: pd.DataFrame, args) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    risk_p, risk_h, top_p, top_h = [], [], [], []
    for _, row in out.iterrows():
        rp, tp = normalized_path_risk(row, 'pathsim', args)
        rh, th = normalized_path_risk(row, 'hetesim', args)
        risk_p.append(rp)
        risk_h.append(rh)
        top_p.append(tp)
        top_h.append(th)
    out['risk_pathsim'] = risk_p
    out['risk_hetesim'] = risk_h
    out['risk_semantic'] = np.maximum(out['risk_pathsim'].astype(float), out['risk_hetesim'].astype(float))
    out['top10_pathsim_risk_diagnostic'] = top_p
    out['top10_hetesim_risk_diagnostic'] = top_h
    out['normalized_feasible'] = (out['risk_semantic'].astype(float) <= 1.0 + 1e-12).astype(int)
    node_w = float(getattr(args, 'reduction_node_weight', 0.7))
    edge_w = float(getattr(args, 'reduction_edge_weight', 0.3))
    denom = max(node_w + edge_w, 1e-12)
    node_w, edge_w = node_w / denom, edge_w / denom
    out['reduction_gain'] = (
        node_w * out.get('fullgraph_reduction_ratio', 0.0).astype(float) +
        edge_w * out.get('fullgraph_edge_reduction_ratio', 0.0).astype(float)
    )
    return out


def _sort_normalized_candidates(df: pd.DataFrame, args=None) -> pd.DataFrame:
    tmp = df.copy()
    objective_col = getattr(args, 'objective_col', 'fullgraph_reduction_ratio') if args is not None else 'fullgraph_reduction_ratio'
    if objective_col not in tmp.columns:
        raise ValueError(f'Unknown objective column for threshold selection: {objective_col}')
    for c in [objective_col, 'risk_semantic', 'risk_pathsim', 'risk_hetesim', 'fullgraph_edge_reduction_ratio',
              'avg_pathsim_ndcg10', 'avg_hetesim_ndcg10', 'avg_pathsim_top10', 'avg_hetesim_top10']:
        if c not in tmp.columns:
            tmp[c] = np.nan
    # Lexicographic selection: maximize the chosen node-reduction objective under the semantic-risk budget;
    # use risk and edge reduction only as tie-breakers, not as weighted objectives.
    return tmp.sort_values(
        by=[
            objective_col, 'risk_semantic', 'risk_pathsim', 'risk_hetesim',
            'fullgraph_edge_reduction_ratio', 'avg_pathsim_ndcg10', 'avg_hetesim_ndcg10',
            'avg_pathsim_top10', 'avg_hetesim_top10',
        ],
        ascending=[False, True, True, True, False, False, False, False, False],
        na_position='last'
    )


def select_best_normalized_thresholds(summary_df: pd.DataFrame, args) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=SELECTED_THRESHOLD_COLUMNS)
    summary_df = add_normalized_risk_columns(summary_df, args)
    selected_rows = []
    fallback_to = (getattr(args, 'fallback_to', 'best_effort') or 'best_effort').lower()
    for (dataset, setting), group in summary_df.groupby(['dataset', 'reduction_setting'], dropna=False):
        cand = group[group['normalized_feasible'].astype(int) == 1]
        source = 'normalized_risk'
        if cand.empty:
            if fallback_to == 'none':
                continue
            # Scientific fallback: choose the lowest semantic risk first, then the largest reduction.
            source = 'normalized_best_effort'
            cand = group.sort_values(
                by=['risk_semantic', getattr(args, 'objective_col', 'fullgraph_reduction_ratio'), 'fullgraph_edge_reduction_ratio'],
                ascending=[True, False, False],
                na_position='last'
            )
            chosen = cand.iloc[0].to_dict()
        else:
            chosen = _sort_normalized_candidates(cand, args).iloc[0].to_dict()
        chosen['selection_source'] = source
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows)
    for col in SELECTED_THRESHOLD_COLUMNS:
        if col not in selected.columns:
            selected[col] = np.nan
    return selected[SELECTED_THRESHOLD_COLUMNS]


def reselect_from_summary(args):
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(getattr(args, 'summary_csv', None) or (outdir / 'semantic_selection_summary.csv'))
    if not summary_path.exists():
        raise FileNotFoundError(f'Cannot find semantic summary CSV: {summary_path}')
    summary = pd.read_csv(summary_path)
    summary = add_normalized_risk_columns(summary, args)
    risk_summary_path = outdir / 'semantic_selection_summary_with_normalized_risk.csv'
    summary.to_csv(risk_summary_path, index=False)
    if getattr(args, 'selection_rule', 'normalized_risk') == 'normalized_risk':
        selected = select_best_normalized_thresholds(summary, args)
    else:
        selected = select_best_semantic_thresholds(summary, args.selection_mode, args.fallback_to)
    selected_path = selected_thresholds_path(args)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, index=False)
    print(f'[reselect] Read: {summary_path}')
    print(f'[reselect] Wrote normalized-risk summary: {risk_summary_path}')
    print(f'[reselect] Wrote selected thresholds: {selected_path}')
    return summary, selected

def choose_semantic_sample_indices(n: int, args) -> np.ndarray:
    sample_size = int(getattr(args, 'semantic_sample_size', 0) or 0)
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(int(getattr(args, 'semantic_sample_seed', 42)))
    return np.sort(rng.choice(n, size=sample_size, replace=False)).astype(np.int32)


def run_semantic_select(args):
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    semantic_taus = parse_float_list(args.semantic_taus)
    attr_taus = parse_float_list(args.attr_taus)
    rows = []

    for dataset in args.datasets:
        print(f'[semantic_select] Dataset={dataset}: preparing graph, target BoW, and semantic-channel matrices...')
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        target_t = cfg['target_type']
        sample_idx = choose_semantic_sample_indices(len(parts[target_t]), args)
        orig_cache = base.build_original_semantic_cache(data, mats, cfg)
        full_nodes = int(sum(len(v) for v in data['ids_by_type'].values()))
        main_nodes = int(sum(len(parts[t]) for t in cfg['main_types']))
        semantic_nodes = int(sum(len(data['ids_by_type'].get(t, [])) for t in cfg.get('semantic_types', [])))
        original_edges = base.count_original_edges(data)

        settings = []
        for ts in semantic_taus:
            for tx in attr_taus:
                settings.append(('layered_semantic_attribute', ts, tx))

        for mode, ts, tx in settings:
            print(f'[semantic_select] Dataset={dataset} threshold=S{ts:g}_X{tx:g}: reducing and evaluating PathSim/HeteSim...')
            t0 = time.perf_counter()
            cluster_of_by_type, _ = build_clusters_for_mode(
                cfg, parts, rawX_by_type, termB_by_type, mode, ts, tx, args.semantic_sim
            )
            add_semantic_singletons(cluster_of_by_type, data, cfg)
            _, agg = base.evaluate_semantics(
                data, mats, cfg, cluster_of_by_type, sample_idx=sample_idx, orig_cache=orig_cache
            )
            reduced_main_nodes = sum(int(cluster_of_by_type[t].max()) + 1 for t in cfg['main_types'])
            reduced_full_nodes = reduced_main_nodes + semantic_nodes
            reduced_edges = base.count_reduced_edges(data, mats, cluster_of_by_type)
            row = {
                'dataset': dataset,
                'reduction_setting': mode,
                'semantic_sim': args.semantic_sim,
                'semantic_threshold': float(ts),
                'attribute_threshold': np.nan if tx is None else float(tx),
                'full_nodes': full_nodes,
                'main_nodes': main_nodes,
                'semantic_nodes': semantic_nodes,
                'reduced_full_nodes': reduced_full_nodes,
                'reduced_main_nodes': reduced_main_nodes,
                'original_edges': original_edges,
                'reduced_edges': reduced_edges,
                'fullgraph_reduction_ratio': 1.0 - reduced_full_nodes / max(full_nodes, 1),
                'main_reduction_ratio': 1.0 - reduced_main_nodes / max(main_nodes, 1),
                'fullgraph_edge_reduction_ratio': 1.0 - reduced_edges / max(original_edges, 1),
                'semantic_eval_sample_size': int(len(sample_idx)),
                'threshold_runtime_sec': time.perf_counter() - t0,
                'avg_pathsim_mae': agg['pathsim_mae'],
                'avg_pathsim_p95e': agg['pathsim_p95e'],
                'avg_pathsim_p99e': agg['pathsim_p99e'],
                'avg_pathsim_top10': agg['pathsim_top10'],
                'avg_pathsim_ndcg10': agg['pathsim_ndcg10'],
                'avg_pathsim_pvalue': agg['pathsim_pvalue'],
                'avg_hetesim_mae': agg['hetesim_mae'],
                'avg_hetesim_p95e': agg['hetesim_p95e'],
                'avg_hetesim_p99e': agg['hetesim_p99e'],
                'avg_hetesim_top10': agg['hetesim_top10'],
                'avg_hetesim_ndcg10': agg['hetesim_ndcg10'],
                'avg_hetesim_pvalue': agg['hetesim_pvalue'],
            }
            add_semantic_controllability(row, args)
            row['semantic_score'] = semantic_quality_score_from_row(pd.Series(row))
            rows.append(row)
            print(
                f"[semantic_select] Dataset={dataset} S={ts:g} X={tx:g} "
                f"reduction={row['fullgraph_reduction_ratio']:.4f} "
                f"pathsim_ndcg10={row['avg_pathsim_ndcg10']:.4f} "
                f"hetesim_ndcg10={row['avg_hetesim_ndcg10']:.4f} "
                f"joint={row['joint_controllable']} "
                f"time={row['threshold_runtime_sec']:.2f}s"
            )

    summary = pd.DataFrame(rows)
    summary = add_normalized_risk_columns(summary, args)
    summary.to_csv(outdir / 'semantic_selection_summary.csv', index=False)
    if getattr(args, 'selection_rule', 'normalized_risk') == 'normalized_risk':
        selected = select_best_normalized_thresholds(summary, args)
    else:
        selected = select_best_semantic_thresholds(summary, args.selection_mode, args.fallback_to)
    selected.to_csv(outdir / 'selected_thresholds_for_clustering.csv', index=False)
    print(f'[semantic_select] Wrote {outdir / "semantic_selection_summary.csv"}')
    print(f'[semantic_select] Wrote {outdir / "selected_thresholds_for_clustering.csv"} with {len(selected)} selected rows.')
    custom_selected = selected_thresholds_path(args)
    if custom_selected != outdir / 'selected_thresholds_for_clustering.csv':
        custom_selected.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(custom_selected, index=False)
    return summary, selected


def _sort_semantic_candidates(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    for c in ['semantic_score'] + SEMANTIC_METRIC_COLUMNS:
        if c not in tmp.columns:
            tmp[c] = np.nan
    tmp['semantic_score'] = tmp.apply(
        lambda r: semantic_quality_score_from_row(r) if pd.isna(r['semantic_score']) else r['semantic_score'],
        axis=1
    )
    return tmp.sort_values(
        by=[
            'fullgraph_reduction_ratio', 'fullgraph_edge_reduction_ratio',
            'avg_pathsim_ndcg10', 'avg_hetesim_ndcg10',
            'avg_pathsim_top10', 'avg_hetesim_top10',
            'avg_pathsim_mae', 'avg_hetesim_mae',
            'avg_pathsim_p99e', 'avg_hetesim_p99e',
            'semantic_score',
        ],
        ascending=[False, False, False, False, False, False, True, True, True, True, False],
        na_position='last'
    )


def select_best_semantic_thresholds(summary_df: pd.DataFrame, selection_mode: str, fallback_to: str):
    if summary_df.empty:
        return pd.DataFrame(columns=SELECTED_THRESHOLD_COLUMNS)
    selected_rows = []
    selection_mode = (selection_mode or 'joint').lower()
    fallback_to = (fallback_to or 'best_effort').lower()
    mode_to_col = {
        'joint': 'joint_controllable',
        'pathsim': 'pathsim_controllable',
        'hetesim': 'hetesim_controllable',
    }
    if selection_mode not in mode_to_col:
        raise ValueError('selection_mode must be one of: joint, pathsim, hetesim')
    if fallback_to not in ('hetesim', 'pathsim', 'best_effort', 'none'):
        raise ValueError('fallback_to must be one of: hetesim, pathsim, best_effort, none')

    for (dataset, setting), group in summary_df.groupby(['dataset', 'reduction_setting'], dropna=False):
        source = selection_mode
        cand = group[group[mode_to_col[selection_mode]] == 1]
        if cand.empty and fallback_to in ('hetesim', 'pathsim'):
            source = fallback_to
            cand = group[group[mode_to_col[fallback_to]] == 1]
        if cand.empty and fallback_to == 'best_effort':
            source = 'best_effort'
            cand = group.copy()
        if cand.empty:
            continue
        chosen = _sort_semantic_candidates(cand).iloc[0].to_dict()
        chosen['selection_source'] = source
        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows)
    for col in SELECTED_THRESHOLD_COLUMNS:
        if col not in selected.columns:
            selected[col] = np.nan
    return selected[SELECTED_THRESHOLD_COLUMNS]


def extract_target_labels(data: dict, target_type: int, policy: str = 'first', use_test_labels: bool = True):
    labs = []
    if 'label.dat' in data.get('labels', {}):
        labs.extend(data['labels']['label.dat'])
    if use_test_labels and 'label.dat.test' in data.get('labels', {}):
        labs.extend(data['labels']['label.dat.test'])
    id_to_local = {int(nid): i for i, nid in enumerate(data['ids_by_type'][target_type])}
    xs, ys = [], []
    for nid, t, y in labs:
        if int(t) != int(target_type) or int(nid) not in id_to_local:
            continue
        y_str = str(y)
        if ',' in y_str:
            if policy == 'drop':
                continue
            y_str = y_str.split(',')[0]
        xs.append(id_to_local[int(nid)])
        ys.append(y_str)
    if not xs:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), {}
    label_vocab = {lab: i for i, lab in enumerate(sorted(set(ys)))}
    y_int = np.array([label_vocab[v] for v in ys], dtype=np.int32)
    return np.array(xs, dtype=np.int32), y_int, label_vocab


def pathsim_matrix_from_B(B: sp.csr_matrix) -> np.ndarray:
    C = (B @ B.T).astype(np.float32).toarray()
    diag = np.diag(C).astype(np.float32)
    denom = diag[:, None] + diag[None, :]
    sim = np.divide(2.0 * C, denom, out=np.zeros_like(C, dtype=np.float32), where=denom > 0)
    return np.clip(sim, 0.0, 1.0)


def hetesim_matrix_from_factors(BL_norm: sp.csr_matrix, BR_norm: sp.csr_matrix) -> np.ndarray:
    sim = (BL_norm @ BR_norm.T).astype(np.float32).toarray()
    return np.clip(sim, -1.0, 1.0)


def build_similarity_feature(data, mats, cfg, cluster_of_by_type: Optional[dict], method: str,
                             combine_paths: str = 'concat') -> np.ndarray:
    """Build target-node feature matrix from PathSim/HeteSim rows.

    If cluster_of_by_type is None, features are computed on the original graph.
    Otherwise, they are computed on the reduced graph and broadcast back to the
    original target-node index space.
    """
    target_t = cfg['target_type']
    target_cluster_of = None
    G_by_type = {}
    if cluster_of_by_type is not None:
        G_by_type = {t: base.build_group_matrix(cluster_of_by_type[t]) for t in cluster_of_by_type}
        target_cluster_of = cluster_of_by_type[target_t]

    feats = []
    for pname, link_seq in cfg['meta_paths'].items():
        if method == 'pathsim':
            if cluster_of_by_type is None:
                B = base.compose_path(mats, link_seq)
                S = pathsim_matrix_from_B(B)
            else:
                B_red, _ = base.reduced_path_matrix(mats, data['link_defs'], link_seq, G_by_type)
                S_red = pathsim_matrix_from_B(B_red)
                S = S_red[target_cluster_of][:, target_cluster_of]
        elif method == 'hetesim':
            if cluster_of_by_type is None:
                _, _, BLn, BRn = base.compute_hetesim_path_factors(mats, data['link_defs'], link_seq, {})
                S = hetesim_matrix_from_factors(BLn, BRn)
            else:
                _, _, BLn_red, BRn_red = base.compute_hetesim_path_factors(mats, data['link_defs'], link_seq, G_by_type)
                S_red = hetesim_matrix_from_factors(BLn_red, BRn_red)
                S = S_red[target_cluster_of][:, target_cluster_of]
        else:
            raise ValueError(method)
        feats.append(S.astype(np.float32))
    if combine_paths == 'mean':
        return np.mean(feats, axis=0).astype(np.float32)
    return np.concatenate(feats, axis=1).astype(np.float32)


def cluster_acc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    true_map = {v: i for i, v in enumerate(true_labels)}
    pred_map = {v: i for i, v in enumerate(pred_labels)}
    W = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        W[pred_map[yp], true_map[yt]] += 1
    row, col = linear_sum_assignment(W.max() - W)
    return float(W[row, col].sum() / len(y_true))


def map_clusters_to_labels(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mapped = np.zeros_like(y_pred)
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        vals, counts = np.unique(y_true[idx], return_counts=True)
        mapped[idx] = vals[np.argmax(counts)]
    return mapped


def clustering_metrics(X, y_true, y_pred, include_internal=True):
    out = {}
    out['nmi'] = float(normalized_mutual_info_score(y_true, y_pred))
    out['acc'] = cluster_acc(y_true, y_pred)
    mapped = map_clusters_to_labels(y_true, y_pred)
    out['precision_macro'] = float(precision_score(y_true, mapped, average='macro', zero_division=0))
    n_clusters = len(np.unique(y_pred))
    if include_internal and 1 < n_clusters < len(y_pred):
        try:
            out['sc'] = float(silhouette_score(X, y_pred, metric='euclidean'))
        except Exception:
            out['sc'] = np.nan
        try:
            out['chi'] = float(calinski_harabasz_score(X, y_pred))
        except Exception:
            out['chi'] = np.nan
    else:
        out['sc'] = np.nan
        out['chi'] = np.nan
    return out


def run_kmeans(X, n_clusters, seed, n_init=10):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=n_init,
        max_iter=300,
        algorithm="lloyd",
    )
    with threadpool_limits(limits=1):
        return model.fit_predict(X)


def run_fcm(X, n_clusters, seed, m=2.0, max_iter=150, tol=1e-4):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, n_clusters))
    U = U / U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        U_old = U.copy()
        Um = U ** m
        centers = (Um.T @ X) / np.maximum(Um.sum(axis=0)[:, None], 1e-12)
        dist = pairwise_distances(X, centers, metric='euclidean') + 1e-8
        power = 2.0 / (m - 1.0)
        inv = dist ** (-power)
        U = inv / inv.sum(axis=1, keepdims=True)
        if np.linalg.norm(U - U_old) < tol:
            break
    return U.argmax(axis=1)


def run_bsas(X, n_clusters, seed):
    """Basic Sequential Algorithmic Scheme with a data-adaptive threshold.

    The threshold is the median distance from a small random sample to its nearest
    neighbor.  The maximum number of clusters is capped by n_clusters to keep the
    comparison aligned with labeled clustering evaluation.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    order = rng.permutation(n)
    sample = X[order[:min(n, 800)]]
    if len(sample) > 2:
        D = pairwise_distances(sample, metric='euclidean')
        np.fill_diagonal(D, np.inf)
        theta = float(np.median(D.min(axis=1)))
    else:
        theta = 0.0
    centers = []
    labels = np.empty(n, dtype=np.int32)
    counts = []
    for idx in order:
        x = X[idx]
        if not centers:
            centers.append(x.copy())
            counts.append(1)
            labels[idx] = 0
            continue
        d = np.linalg.norm(np.vstack(centers) - x, axis=1)
        j = int(np.argmin(d))
        if d[j] > theta and len(centers) < n_clusters:
            centers.append(x.copy())
            counts.append(1)
            labels[idx] = len(centers) - 1
        else:
            labels[idx] = j
            counts[j] += 1
            centers[j] = centers[j] + (x - centers[j]) / counts[j]
    return labels


def _bic_score(X, labels, centers):
    n, d = X.shape
    clusters = np.unique(labels)
    k = len(clusters)
    var = 0.0
    for c in clusters:
        idx = labels == c
        var += ((X[idx] - centers[c]) ** 2).sum()
    denom = max(n - k, 1) * max(d, 1)
    sigma2 = max(var / denom, 1e-8)
    log_likelihood = -0.5 * n * d * np.log(2 * np.pi * sigma2) - 0.5 * var / sigma2
    p = k * (d + 1)
    return log_likelihood - 0.5 * p * np.log(max(n, 1))


def run_xmeans_simple(X, n_clusters, seed):
    """A lightweight X-means style approximation.

    It recursively tests whether splitting a cluster into two improves a BIC-like
    criterion, until at most n_clusters clusters are obtained.  This is included
    for protocol compatibility with the reference thesis; for strict benchmarking,
    KMeans and FCM are usually more stable.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    changed = True
    while changed and len(np.unique(labels)) < n_clusters:
        changed = False
        new_labels = labels.copy()
        next_label = int(labels.max()) + 1
        for c in list(np.unique(labels)):
            idx = np.where(labels == c)[0]
            if len(idx) < 4 or len(np.unique(labels)) >= n_clusters:
                continue
            Xc = X[idx]
            parent_center = Xc.mean(axis=0, keepdims=True)
            parent_labels = np.zeros(len(idx), dtype=np.int32)
            parent_centers = np.vstack([parent_center])
            parent_bic = _bic_score(Xc, parent_labels, parent_centers)
            km = KMeans(n_clusters=2, random_state=int(rng.integers(1_000_000)), n_init=5).fit(Xc)
            child_bic = _bic_score(Xc, km.labels_, km.cluster_centers_)
            if child_bic > parent_bic:
                # Keep one child as c and assign the other a new label.
                child = km.labels_
                new_labels[idx[child == 1]] = next_label
                next_label += 1
                changed = True
        labels = new_labels
    # Reindex labels
    uniq = {v: i for i, v in enumerate(np.unique(labels))}
    return np.array([uniq[v] for v in labels], dtype=np.int32)


def target_distribution_torch(q):
    import torch
    weight = q ** 2 / torch.clamp(q.sum(0), min=1e-12)
    return (weight.t() / torch.clamp(weight.sum(1), min=1e-12)).t()


def run_deepdec(X, n_clusters, seed, args=None):
    """Autoencoder pretraining + DEC fine tuning, inspired by deep clustering models.

    The input X is the PathSim/HeteSim target-node feature matrix. This keeps
    clustering as a downstream validation stage while giving the clustering
    model a trainable representation learner similar in spirit to B3C/DEC.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError('deepdec requires PyTorch to be installed.') from e

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X_np = np.asarray(X, dtype=np.float32)
    pca_dim = int(getattr(args, 'deep_pca_dim', 256) if args is not None else 256)
    if pca_dim > 0 and X_np.shape[1] > pca_dim:
        n_comp = min(pca_dim, X_np.shape[0] - 1, X_np.shape[1])
        if n_comp >= 2:
            X_np = PCA(n_components=n_comp, random_state=seed).fit_transform(X_np).astype(np.float32)
            X_np = standardize_features(X_np)

    device_arg = getattr(args, 'deep_device', 'auto') if args is not None else 'auto'
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    hidden_dim = int(getattr(args, 'deep_hidden_dim', 128) if args is not None else 128)
    embed_dim = int(getattr(args, 'deep_embed_dim', 32) if args is not None else 32)
    pretrain_epochs = int(getattr(args, 'deep_pretrain_epochs', 80) if args is not None else 80)
    dec_epochs = int(getattr(args, 'deep_dec_epochs', 80) if args is not None else 80)
    lr = float(getattr(args, 'deep_lr', 1e-3) if args is not None else 1e-3)
    verbose = bool(getattr(args, 'verbose_training', False) if args is not None else False)

    class DeepDEC(nn.Module):
        def __init__(self, in_dim, hidden, embed, k):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed))
            self.decoder = nn.Sequential(nn.Linear(embed, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
            self.cluster_layer = nn.Parameter(torch.empty(k, embed))
            nn.init.xavier_uniform_(self.cluster_layer)

        def soft_assign(self, z):
            dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
            q = 1.0 / (1.0 + dist)
            return (q.t() / torch.clamp(q.sum(1), min=1e-12)).t()

        def forward(self, x):
            z = self.encoder(x)
            x_bar = self.decoder(z)
            q = self.soft_assign(z)
            return z, x_bar, q

    x = torch.tensor(X_np, dtype=torch.float32, device=device)
    model = DeepDEC(X_np.shape[1], hidden_dim, embed_dim, n_clusters).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(pretrain_epochs):
        opt.zero_grad()
        _, x_bar, _ = model(x)
        loss = F.mse_loss(x_bar, x)
        loss.backward()
        opt.step()
        if verbose and (epoch == 0 or (epoch + 1) % max(1, pretrain_epochs // 5) == 0):
            print(f'[deepdec] seed={seed} pretrain_epoch={epoch + 1}/{pretrain_epochs} recon_loss={loss.item():.6f}')

    with torch.no_grad():
        z, _, _ = model(x)
    init_labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(z.cpu().numpy())
    centers = np.vstack([z.cpu().numpy()[init_labels == c].mean(axis=0) if np.any(init_labels == c)
                         else rng.normal(size=embed_dim) for c in range(n_clusters)]).astype(np.float32)
    model.cluster_layer.data = torch.tensor(centers, dtype=torch.float32, device=device)

    for epoch in range(dec_epochs):
        with torch.no_grad():
            _, _, q = model(x)
            p = target_distribution_torch(q)
        opt.zero_grad()
        z, x_bar, q = model(x)
        kl_loss = F.kl_div(torch.log(torch.clamp(q, min=1e-12)), p, reduction='batchmean')
        recon_loss = F.mse_loss(x_bar, x)
        loss = kl_loss + 0.1 * recon_loss
        loss.backward()
        opt.step()
        if verbose and (epoch == 0 or (epoch + 1) % max(1, dec_epochs // 5) == 0):
            print(f'[deepdec] seed={seed} dec_epoch={epoch + 1}/{dec_epochs} kl={kl_loss.item():.6f} recon={recon_loss.item():.6f}')

    with torch.no_grad():
        _, _, q = model(x)
    return torch.argmax(q, dim=1).cpu().numpy().astype(np.int32)


def run_clustering_algorithm(name, X, n_clusters, seed, args=None):
    if name == 'kmeans':
        return run_kmeans(X, n_clusters, seed, n_init=getattr(args, 'kmeans_n_init', 10))
    if name == 'fcm':
        return run_fcm(X, n_clusters, seed)
    if name == 'bsas':
        return run_bsas(X, n_clusters, seed)
    if name == 'xmeans':
        return run_xmeans_simple(X, n_clusters, seed)
    if name in ('deepdec', 'dec'):
        return run_deepdec(X, n_clusters, seed, args=args)
    raise ValueError(name)


def standardize_features(X):
    X = np.asarray(X, dtype=np.float32)
    # Similarity-row features are already nonnegative and bounded, but standardizing
    # improves clustering stability across paths/datasets.
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)


def run_clustering_preservation(args):
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    semantic_taus = parse_float_list(args.semantic_taus)
    attr_taus = parse_float_list(args.attr_taus)
    algorithms = [x.strip().lower() for x in args.cluster_algorithms.split(',') if x.strip()]
    methods = [x.strip().lower() for x in args.sim_methods.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]

    for dataset in args.datasets:
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        target_t = cfg['target_type']
        label_idx, y, label_vocab = extract_target_labels(
            data, target_t, policy=args.multi_label_policy, use_test_labels=args.use_test_labels
        )
        if len(label_idx) == 0:
            warnings.warn(f'{dataset}: no usable labels found; skipping clustering evaluation.')
            continue
        n_classes = len(np.unique(y))
        if n_classes < 2:
            warnings.warn(f'{dataset}: fewer than two label classes; skipping clustering evaluation.')
            continue

        # Candidate reduction settings: original plus layered semantic-attribute only.
        settings = [('original', 'original', None, None)]
        for ts in semantic_taus:
            for tx in attr_taus:
                settings.append((f'layered_S{ts:g}_X{tx:g}', 'layered_semantic_attribute', ts, tx))

        # Cache original features per method.
        original_features = {}
        original_scalers = {}
        for method in methods:
            X_full = build_similarity_feature(data, mats, cfg, None, method=method, combine_paths=args.combine_paths)
            scaler = StandardScaler(with_mean=True, with_std=True)
            original_features[method] = scaler.fit_transform(np.asarray(X_full[label_idx], dtype=np.float32))
            original_scalers[method] = scaler

        for setting_name, mode, ts, tx in settings:
            if mode == 'original':
                cluster_of_by_type = None
                fullgraph_reduction_ratio = 0.0
                edge_reduction_ratio = 0.0
            else:
                cluster_of_by_type, _ = build_clusters_for_mode(
                    cfg, parts, rawX_by_type, termB_by_type, mode, ts, tx, args.semantic_sim
                )
                add_semantic_singletons(cluster_of_by_type, data, cfg)
                full_nodes = int(sum(len(v) for v in data['ids_by_type'].values()))
                semantic_nodes = int(sum(len(data['ids_by_type'].get(t, [])) for t in cfg.get('semantic_types', [])))
                reduced_main_nodes = sum(int(cluster_of_by_type[t].max()) + 1 for t in cfg['main_types'])
                reduced_full_nodes = reduced_main_nodes + semantic_nodes
                fullgraph_reduction_ratio = 1.0 - reduced_full_nodes / max(full_nodes, 1)
                edge_reduction_ratio = 1.0 - base.count_reduced_edges(data, mats, cluster_of_by_type) / max(base.count_original_edges(data), 1)

            for method in methods:
                if mode == 'original':
                    X_eval = original_features[method]
                else:
                    X_full = build_similarity_feature(data, mats, cfg, cluster_of_by_type, method=method, combine_paths=args.combine_paths)
                    X_eval = original_scalers[method].transform(np.asarray(X_full[label_idx], dtype=np.float32))

                for alg in algorithms:
                    for seed in seeds:
                        try:
                            pred = run_clustering_algorithm(alg, X_eval, n_classes, seed, args=args)
                            met = clustering_metrics(X_eval, y, pred)
                        except Exception as e:
                            warnings.warn(f'{dataset} {setting_name} {method} {alg} seed={seed} failed: {e}')
                            met = {'nmi': np.nan, 'acc': np.nan, 'precision_macro': np.nan, 'sc': np.nan, 'chi': np.nan}
                        row = {
                            'dataset': dataset,
                            'target_type': target_t,
                            'target_type_name': type_names.get(target_t, str(target_t)),
                            'setting': setting_name,
                            'mode': mode,
                            'semantic_sim': args.semantic_sim,
                            'semantic_threshold': ts,
                            'attribute_threshold': tx,
                            'sim_method': method,
                            'combine_paths': args.combine_paths,
                            'cluster_algorithm': alg,
                            'seed': seed,
                            'num_labeled_nodes': len(y),
                            'num_classes': n_classes,
                            'fullgraph_reduction_ratio': fullgraph_reduction_ratio,
                            'fullgraph_edge_reduction_ratio': edge_reduction_ratio,
                        }
                        row.update(met)
                        rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / 'hhin_clustering_preservation_raw.csv', index=False)
    if raw.empty:
        return raw

    summary = raw.groupby([
        'dataset', 'setting', 'mode', 'semantic_sim', 'semantic_threshold', 'attribute_threshold',
        'sim_method', 'combine_paths', 'cluster_algorithm'
    ], dropna=False, as_index=False).agg(
        num_labeled_nodes=('num_labeled_nodes', 'first'),
        num_classes=('num_classes', 'first'),
        fullgraph_reduction_ratio=('fullgraph_reduction_ratio', 'first'),
        fullgraph_edge_reduction_ratio=('fullgraph_edge_reduction_ratio', 'first'),
        nmi_mean=('nmi', 'mean'), nmi_std=('nmi', 'std'),
        acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
        precision_mean=('precision_macro', 'mean'), precision_std=('precision_macro', 'std'),
        sc_mean=('sc', 'mean'), sc_std=('sc', 'std'),
        chi_mean=('chi', 'mean'), chi_std=('chi', 'std'),
    )

    # Compute preservation deltas relative to original under the same dataset/method/algorithm.
    baseline = summary[summary['mode'] == 'original'][[
        'dataset', 'sim_method', 'combine_paths', 'cluster_algorithm',
        'nmi_mean', 'acc_mean', 'precision_mean', 'sc_mean', 'chi_mean'
    ]].rename(columns={
        'nmi_mean': 'orig_nmi_mean', 'acc_mean': 'orig_acc_mean', 'precision_mean': 'orig_precision_mean',
        'sc_mean': 'orig_sc_mean', 'chi_mean': 'orig_chi_mean'
    })
    summary = summary.merge(baseline, on=['dataset', 'sim_method', 'combine_paths', 'cluster_algorithm'], how='left')
    for m in ['nmi', 'acc', 'precision', 'sc', 'chi']:
        summary[f'delta_{m}'] = summary[f'{m}_mean'] - summary[f'orig_{m}_mean']
        summary[f'abs_delta_{m}'] = np.abs(summary[f'delta_{m}'])

    summary.to_csv(outdir / 'hhin_clustering_preservation_summary.csv', index=False)
    if args.write_plots:
        plot_clustering_summary(summary, outdir)
    return summary


def selected_thresholds_path(args) -> Path:
    if getattr(args, 'selected_thresholds_csv', None):
        return Path(args.selected_thresholds_csv)
    return Path(args.out_dir) / 'selected_thresholds_for_clustering.csv'


def selected_row_to_mode(row: pd.Series) -> str:
    setting = str(row.get('reduction_setting', 'layered_semantic_attribute'))
    if setting == 'semantic_channel_only' or setting.startswith('semantic_only'):
        raise ValueError(
            'Selected thresholds contain semantic_channel_only, but this experiment requires '
            'target-node BoW attributes to participate in HHIN reduction. Regenerate selected '
            'thresholds with the current reduction stage.'
        )
    if setting == 'layered_semantic_attribute':
        return setting
    return 'layered_semantic_attribute'


def run_clustering_selected(args):
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    profile = getattr(args, 'clustering_profile', 'selected')
    if profile == 'pdf':
        raw_name = 'clustering_pdf_aligned_raw.csv'
        summary_name = 'clustering_pdf_aligned_summary.csv'
    elif profile == 'b3c':
        raw_name = 'clustering_b3c_aligned_raw.csv'
        summary_name = 'clustering_b3c_aligned_summary.csv'
    else:
        raw_name = 'clustering_selected_raw.csv'
        summary_name = 'clustering_selected_summary.csv'
    selected_csv = selected_thresholds_path(args)
    if not selected_csv.exists():
        raise FileNotFoundError(
            f'Cannot find selected thresholds CSV: {selected_csv}. '
            'Run --task semantic_select first, or use --task pipeline.'
        )

    selected = pd.read_csv(selected_csv)
    if selected.empty:
        raw = pd.DataFrame()
        raw.to_csv(outdir / raw_name, index=False)
        raw.to_csv(outdir / summary_name, index=False)
        return raw

    algorithms = [x.strip().lower() for x in args.cluster_algorithms.split(',') if x.strip()]
    methods = [x.strip().lower() for x in args.sim_methods.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    rows = []
    print(f'[clustering:{profile}] Reading selected thresholds from {selected_csv}')
    print(f'[clustering:{profile}] Algorithms={algorithms} methods={methods} seeds={seeds}')

    for dataset, selected_ds in selected.groupby('dataset', dropna=False):
        if dataset not in set(args.datasets):
            continue
        print(f'[clustering:{profile}] Dataset={dataset}: loading graph and selected reduced graph settings...')
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        target_t = cfg['target_type']
        label_idx, y, _ = extract_target_labels(
            data, target_t, policy=args.multi_label_policy, use_test_labels=args.use_test_labels
        )
        if len(label_idx) == 0:
            warnings.warn(f'{dataset}: no usable labels found; skipping selected clustering evaluation.')
            continue
        n_classes = len(np.unique(y))
        if n_classes < 2:
            warnings.warn(f'{dataset}: fewer than two label classes; skipping selected clustering evaluation.')
            continue

        original_features = {}
        original_scalers = {}
        for method in methods:
            X_full = build_similarity_feature(data, mats, cfg, None, method=method, combine_paths=args.combine_paths)
            scaler = StandardScaler(with_mean=True, with_std=True)
            original_features[method] = scaler.fit_transform(np.asarray(X_full[label_idx], dtype=np.float32))
            original_scalers[method] = scaler

        original_metrics = {}
        for method in methods:
            for alg in algorithms:
                for seed in seeds:
                    pred = run_clustering_algorithm(
                        alg, original_features[method], n_classes, seed, args=args
                    )
                    original_metrics[(method, alg, seed)] = clustering_metrics(
                        original_features[method], y, pred, include_internal=False
                    )

        for _, sel in selected_ds.iterrows():
            mode = selected_row_to_mode(sel)
            ts = float(sel['semantic_threshold'])
            tx_val = sel.get('attribute_threshold', np.nan)
            tx = None if pd.isna(tx_val) else float(tx_val)
            print(f'[clustering:{profile}] Dataset={dataset} selected threshold=S{ts:g}_X{tx:g}: rebuilding reduced graph...')
            cluster_of_by_type, _ = build_clusters_for_mode(
                cfg, parts, rawX_by_type, termB_by_type, mode, ts, tx, args.semantic_sim
            )
            add_semantic_singletons(cluster_of_by_type, data, cfg)

            selected_meta = {
                'dataset': dataset,
                'reduction_setting': sel.get('reduction_setting', mode),
                'selection_source': sel.get('selection_source', ''),
                'semantic_threshold': ts,
                'attribute_threshold': np.nan if tx is None else tx,
                'clustering_profile': profile,
                'num_labeled_nodes': int(len(y)),
                'num_classes': int(n_classes),
            }

            for method in methods:
                X_by_view = {
                    'original': original_features[method],
                }
                X_red_full = build_similarity_feature(
                    data, mats, cfg, cluster_of_by_type, method=method, combine_paths=args.combine_paths
                )
                X_by_view['reduced'] = original_scalers[method].transform(
                    np.asarray(X_red_full[label_idx], dtype=np.float32)
                )

                for graph_view, X_eval in X_by_view.items():
                    for alg in algorithms:
                        for seed in seeds:
                            try:
                                print(f'[clustering] Dataset={dataset} view={graph_view} method={method} algorithm={alg} seed={seed} ...')
                                if graph_view == 'original':
                                    met = original_metrics[(method, alg, seed)]
                                else:
                                    pred = run_clustering_algorithm(alg, X_eval, n_classes, seed, args=args)
                                    met = clustering_metrics(X_eval, y, pred, include_internal=False)
                                print(
                                    f"[clustering] Dataset={dataset} view={graph_view} method={method} "
                                    f"algorithm={alg} seed={seed} NMI={met['nmi']:.4f} ACC={met['acc']:.4f}"
                                )
                            except Exception as e:
                                warnings.warn(f'{dataset} selected {method} {alg} {graph_view} seed={seed} failed: {e}')
                                met = {'nmi': np.nan, 'acc': np.nan, 'precision_macro': np.nan, 'sc': np.nan, 'chi': np.nan}
                            row = dict(selected_meta)
                            row.update({
                                'sim_method': method,
                                'cluster_algorithm': alg,
                                'seed': seed,
                                'graph_view': graph_view,
                                'NMI': met['nmi'],
                                'ACC': met['acc'],
                                'Precision': met['precision_macro'],
                                'SC': met['sc'],
                                'CHI': met['chi'],
                            })
                            rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / raw_name, index=False)
    if raw.empty:
        raw.to_csv(outdir / summary_name, index=False)
        return raw

    group_cols = [
        'dataset', 'reduction_setting', 'selection_source',
        'semantic_threshold', 'attribute_threshold',
        'sim_method', 'cluster_algorithm', 'clustering_profile',
        'num_labeled_nodes', 'num_classes',
    ]
    means = raw.groupby(group_cols + ['graph_view'], dropna=False, as_index=False).agg(
        NMI_mean=('NMI', 'mean'),
        ACC_mean=('ACC', 'mean'),
        Precision_mean=('Precision', 'mean'),
        SC_mean=('SC', 'mean'),
        CHI_mean=('CHI', 'mean'),
    )
    original = means[means['graph_view'] == 'original'].drop(columns=['graph_view']).rename(columns={
        'NMI_mean': 'NMI_ori_mean',
        'ACC_mean': 'ACC_ori_mean',
        'Precision_mean': 'Precision_ori_mean',
        'SC_mean': 'SC_ori_mean',
        'CHI_mean': 'CHI_ori_mean',
    })
    reduced = means[means['graph_view'] == 'reduced'].drop(columns=['graph_view']).rename(columns={
        'NMI_mean': 'NMI_red_mean',
        'ACC_mean': 'ACC_red_mean',
        'Precision_mean': 'Precision_red_mean',
        'SC_mean': 'SC_red_mean',
        'CHI_mean': 'CHI_red_mean',
    })
    summary = reduced.merge(original, on=group_cols, how='left')
    summary['Delta_NMI'] = summary['NMI_red_mean'] - summary['NMI_ori_mean']
    summary['Delta_ACC'] = summary['ACC_red_mean'] - summary['ACC_ori_mean']
    summary['Delta_Precision'] = summary['Precision_red_mean'] - summary['Precision_ori_mean']
    summary['Delta_SC'] = summary['SC_red_mean'] - summary['SC_ori_mean']
    summary['Delta_CHI'] = summary['CHI_red_mean'] - summary['CHI_ori_mean']
    summary.to_csv(outdir / summary_name, index=False)
    print(f'[clustering:{profile}] Wrote {outdir / raw_name}')
    print(f'[clustering:{profile}] Wrote {outdir / summary_name}')
    return summary


def plot_clustering_summary(summary: pd.DataFrame, outdir: Path):
    import matplotlib.pyplot as plt
    sub = summary[summary['mode'] != 'original'].copy()
    if sub.empty:
        return
    for dataset, df in sub.groupby('dataset'):
        # Plot NMI vs reduction for each mode.
        plt.figure(figsize=(7.2, 4.8))
        for mode, g in df.groupby('mode'):
            gg = g.groupby('setting', as_index=False).agg(
                fullgraph_reduction_ratio=('fullgraph_reduction_ratio', 'mean'),
                nmi_mean=('nmi_mean', 'mean'),
            )
            plt.scatter(gg['fullgraph_reduction_ratio'] * 100, gg['nmi_mean'], label=mode, alpha=0.75)
        plt.xlabel('full-graph node reduction ratio (%)')
        plt.ylabel('NMI')
        plt.title(f'{dataset}: clustering quality vs reduction')
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(outdir / f'{dataset.lower()}_clustering_nmi_vs_reduction.png', dpi=220)
        plt.close()

        # Plot absolute ACC delta vs reduction.
        plt.figure(figsize=(7.2, 4.8))
        for mode, g in df.groupby('mode'):
            gg = g.groupby('setting', as_index=False).agg(
                fullgraph_reduction_ratio=('fullgraph_reduction_ratio', 'mean'),
                abs_delta_acc=('abs_delta_acc', 'mean'),
            )
            plt.scatter(gg['fullgraph_reduction_ratio'] * 100, gg['abs_delta_acc'], label=mode, alpha=0.75)
        plt.xlabel('full-graph node reduction ratio (%)')
        plt.ylabel('|Delta ACC|')
        plt.title(f'{dataset}: clustering preservation vs reduction')
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(outdir / f'{dataset.lower()}_clustering_delta_acc_vs_reduction.png', dpi=220)
        plt.close()





def parse_int_list(s: str):
    return [int(x) for x in str(s).split(',') if str(x).strip()]


def _selected_threshold_rows(args):
    selected_csv = selected_thresholds_path(args)
    if not selected_csv.exists():
        raise FileNotFoundError(
            f'Cannot find selected thresholds CSV: {selected_csv}. '
            'Run --task reselect or --task semantic_select first, or pass --selected-thresholds-csv.'
        )
    selected = pd.read_csv(selected_csv)
    if selected.empty:
        return selected
    allowed = set(args.datasets)
    return selected[selected['dataset'].isin(allowed)].copy()


def _build_selected_cluster_mapping(data, mats, cfg, parts, rawX_by_type, termB_by_type, sel, args):
    mode = selected_row_to_mode(sel)
    ts = float(sel['semantic_threshold'])
    tx_val = sel.get('attribute_threshold', np.nan)
    tx = None if pd.isna(tx_val) else float(tx_val)
    cluster_of_by_type, _ = build_clusters_for_mode(
        cfg, parts, rawX_by_type, termB_by_type, mode, ts, tx, args.semantic_sim
    )
    add_semantic_singletons(cluster_of_by_type, data, cfg)
    return cluster_of_by_type, mode, ts, tx


def _similarity_matrix_for_path(data, mats, cfg, link_seq, method: str, cluster_of_by_type=None):
    target_t = cfg['target_type']
    if cluster_of_by_type is None:
        target_cluster_of = None
        G_by_type = {}
    else:
        G_by_type = {t: base.build_group_matrix(cluster_of_by_type[t]) for t in cluster_of_by_type}
        target_cluster_of = cluster_of_by_type[target_t]

    method = method.lower()
    if method == 'pathsim':
        if cluster_of_by_type is None:
            B = base.compose_path(mats, link_seq)
            return pathsim_matrix_from_B(B)
        B_red, _ = base.reduced_path_matrix(mats, data['link_defs'], link_seq, G_by_type)
        S_red = pathsim_matrix_from_B(B_red)
        return S_red[target_cluster_of][:, target_cluster_of]
    if method == 'hetesim':
        if cluster_of_by_type is None:
            _, _, BLn, BRn = base.compute_hetesim_path_factors(mats, data['link_defs'], link_seq, {})
            return hetesim_matrix_from_factors(BLn, BRn)
        _, _, BLn_red, BRn_red = base.compute_hetesim_path_factors(mats, data['link_defs'], link_seq, G_by_type)
        S_red = hetesim_matrix_from_factors(BLn_red, BRn_red)
        return S_red[target_cluster_of][:, target_cluster_of]
    raise ValueError(f'Unknown similarity method: {method}')


def _ndcg_from_scores(pred_scores, true_scores, k: int):
    return base.ndcg_at_k(pred_scores, true_scores, k=k)


def _retrieval_row_metrics(sim_o, sim_r, query_idx: int, k_values):
    so = np.asarray(sim_o, dtype=np.float32).copy()
    sr = np.asarray(sim_r, dtype=np.float32).copy()
    # Exclude self from retrieval and ranking comparison.
    so[query_idx] = -np.inf
    sr[query_idx] = -np.inf
    finite = np.isfinite(so) & np.isfinite(sr)
    diff = np.abs(so[finite] - sr[finite])
    out = {
        'mae': float(np.mean(diff)) if diff.size else 0.0,
        'p95e': float(np.percentile(diff, 95)) if diff.size else 0.0,
        'p99e': float(np.percentile(diff, 99)) if diff.size else 0.0,
        'maxe': float(np.max(diff)) if diff.size else 0.0,
    }
    n_valid = int(np.sum(np.isfinite(so)))
    candidate_ids = np.flatnonzero(np.isfinite(so) & np.isfinite(sr))
    true_scores = so[candidate_ids]
    pred_scores = sr[candidate_ids]
    effective_ks = [min(int(k), max(n_valid, 0)) for k in k_values]
    max_k = max(effective_ks, default=0)
    order_o = base.deterministic_topk_indices(
        true_scores, max_k, node_ids=candidate_ids
    )
    order_r = base.deterministic_topk_indices(
        pred_scores, max_k, node_ids=candidate_ids
    )
    tie_overlaps = base.tie_aware_overlap_at_ks(
        true_scores, pred_scores, effective_ks
    )
    ndcgs = base.ndcg_at_ks(pred_scores, true_scores, effective_ks)
    for k in k_values:
        kk = min(int(k), max(n_valid, 0))
        if kk <= 0:
            out[f'overlap@{k}'] = 1.0
            out[f'ndcg@{k}'] = 1.0
            continue
        top_o_local = order_o[:kk]
        top_r_local = order_r[:kk]
        top_o = candidate_ids[top_o_local]
        top_r = candidate_ids[top_r_local]
        out[f'overlap@{k}'] = len(set(top_o.tolist()) & set(top_r.tolist())) / kk
        out[f'tie_overlap@{k}'] = tie_overlaps[kk]
        out[f'ndcg@{k}'] = ndcgs[kk]
    return out


def run_retrieval_ranking_selected(args):
    """Evaluate retrieval-set and ranking preservation at selected thresholds.

    Outputs:
      - retrieval_ranking_per_path.csv: per-path summary for PathSim/HeteSim.
      - retrieval_ranking_summary.csv: dataset-level average summary.
      - retrieval_ranking_raw.csv: optional per-query records when --write-retrieval-raw is set.
    """
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected = _selected_threshold_rows(args)
    if selected.empty:
        pd.DataFrame().to_csv(outdir / 'retrieval_ranking_per_path.csv', index=False)
        pd.DataFrame().to_csv(outdir / 'retrieval_ranking_summary.csv', index=False)
        return pd.DataFrame()

    k_values = parse_int_list(getattr(args, 'retrieval_k_values', '1,5,10,20'))
    methods = [x.strip().lower() for x in args.sim_methods.split(',') if x.strip()]
    summary_rows, raw_rows = [], []

    for dataset, selected_ds in selected.groupby('dataset', dropna=False):
        print(f'[retrieval] Dataset={dataset}: loading graph and selected settings...')
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        target_n = len(parts[cfg['target_type']])
        sample_size = int(getattr(args, 'retrieval_sample_size', 0) or 0)
        if sample_size > 0 and sample_size < target_n:
            rng = np.random.default_rng(int(getattr(args, 'semantic_sample_seed', 42)))
            sample_idx = np.sort(rng.choice(target_n, size=sample_size, replace=False)).astype(np.int32)
        else:
            sample_idx = np.arange(target_n, dtype=np.int32)

        # Cache original similarities for all paths/methods.
        orig_sims = {}
        for method in methods:
            for pname, link_seq in cfg['meta_paths'].items():
                print(f'[retrieval] Dataset={dataset} original {method} {pname}...')
                orig_sims[(method, pname)] = _similarity_matrix_for_path(data, mats, cfg, link_seq, method, None)

        for _, sel in selected_ds.iterrows():
            cluster_of_by_type, mode, ts, tx = _build_selected_cluster_mapping(
                data, mats, cfg, parts, rawX_by_type, termB_by_type, sel, args
            )
            selected_meta = {
                'dataset': dataset,
                'reduction_setting': sel.get('reduction_setting', mode),
                'selection_source': sel.get('selection_source', ''),
                'semantic_threshold': ts,
                'attribute_threshold': np.nan if tx is None else tx,
                'sample_size': int(len(sample_idx)),
            }
            for method in methods:
                for pname, link_seq in cfg['meta_paths'].items():
                    print(f'[retrieval] Dataset={dataset} selected S{ts:g}_X{tx:g} {method} {pname}...')
                    S_orig = orig_sims[(method, pname)]
                    S_red = _similarity_matrix_for_path(data, mats, cfg, link_seq, method, cluster_of_by_type)
                    q_metrics = []
                    for qi in sample_idx:
                        m = _retrieval_row_metrics(S_orig[qi], S_red[qi], int(qi), k_values)
                        q_metrics.append(m)
                        if getattr(args, 'write_retrieval_raw', False):
                            row = dict(selected_meta)
                            row.update({'sim_method': method, 'path': pname, 'query_index': int(qi)})
                            row.update(m)
                            raw_rows.append(row)
                    row = dict(selected_meta)
                    row.update({'sim_method': method, 'path': pname})
                    for metric in ['mae', 'p95e', 'p99e', 'maxe']:
                        row[metric] = float(np.mean([m[metric] for m in q_metrics]))
                    for k in k_values:
                        row[f'overlap@{k}'] = float(np.mean([m[f'overlap@{k}'] for m in q_metrics]))
                        row[f'tie_overlap@{k}'] = float(
                            np.mean([m[f'tie_overlap@{k}'] for m in q_metrics])
                        )
                        row[f'ndcg@{k}'] = float(np.mean([m[f'ndcg@{k}'] for m in q_metrics]))
                    summary_rows.append(row)
                    del S_red

    per_path = pd.DataFrame(summary_rows)
    per_path.to_csv(outdir / 'retrieval_ranking_per_path.csv', index=False)
    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(outdir / 'retrieval_ranking_raw.csv', index=False)

    if per_path.empty:
        per_path.to_csv(outdir / 'retrieval_ranking_summary.csv', index=False)
        return per_path
    agg_dict = {m: (m, 'mean') for m in ['mae', 'p95e', 'p99e', 'maxe'] if m in per_path.columns}
    for k in k_values:
        agg_dict[f'overlap@{k}'] = (f'overlap@{k}', 'mean')
        agg_dict[f'tie_overlap@{k}'] = (f'tie_overlap@{k}', 'mean')
        agg_dict[f'ndcg@{k}'] = (f'ndcg@{k}', 'mean')
    summary = per_path.groupby([
        'dataset', 'reduction_setting', 'selection_source', 'semantic_threshold', 'attribute_threshold', 'sim_method'
    ], dropna=False, as_index=False).agg(**agg_dict)
    summary.to_csv(outdir / 'retrieval_ranking_summary.csv', index=False)
    print(f'[retrieval] Wrote {outdir / "retrieval_ranking_per_path.csv"}')
    print(f'[retrieval] Wrote {outdir / "retrieval_ranking_summary.csv"}')
    return summary


def _majority_vote_with_similarity(labels, sims):
    # Sum similarities per label; fall back to frequency under zero similarities.
    scores = {}
    counts = {}
    for lab, sim in zip(labels, sims):
        scores[lab] = scores.get(lab, 0.0) + float(max(sim, 0.0))
        counts[lab] = counts.get(lab, 0) + 1
    if scores and max(scores.values()) > 1e-12:
        return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _label_prediction_metrics(S, label_idx, y, k_values):
    from sklearn.metrics import accuracy_score, f1_score, precision_score
    label_idx = np.asarray(label_idx, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    L = len(label_idx)
    S_lab = S[label_idx][:, label_idx].astype(np.float32)
    rows = []
    for k in k_values:
        preds = []
        for i in range(L):
            scores = S_lab[i].copy()
            scores[i] = -np.inf
            kk = min(int(k), max(L - 1, 1))
            nbr = np.argsort(-scores)[:kk]
            pred = _majority_vote_with_similarity(y[nbr], scores[nbr])
            preds.append(pred)
        preds = np.asarray(preds, dtype=np.int32)
        rows.append({
            'prediction_k': int(k),
            'accuracy': float(accuracy_score(y, preds)),
            'precision_macro': float(precision_score(y, preds, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        })
    return rows


def _combined_similarity_matrix(data, mats, cfg, method: str, cluster_of_by_type, combine_mode='mean'):
    mats_list = []
    for pname, link_seq in cfg['meta_paths'].items():
        mats_list.append(_similarity_matrix_for_path(data, mats, cfg, link_seq, method, cluster_of_by_type))
    if combine_mode == 'mean':
        return np.mean(mats_list, axis=0).astype(np.float32)
    raise ValueError('Prediction currently supports --prediction-combine-paths mean only.')


def run_prediction_selected(args):
    """K-bisimulation-style label-prediction validation using top-k similar labeled nodes.

    This is a lightweight validation protocol: for each labeled target node, its
    label is predicted by weighted majority vote over the top-k most similar
    labeled target nodes. Original and reduced graphs are evaluated under the
    same protocol.
    """
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected = _selected_threshold_rows(args)
    if selected.empty:
        pd.DataFrame().to_csv(outdir / 'prediction_selected_raw.csv', index=False)
        pd.DataFrame().to_csv(outdir / 'prediction_selected_summary.csv', index=False)
        return pd.DataFrame()

    methods = [x.strip().lower() for x in args.sim_methods.split(',') if x.strip()]
    k_values = parse_int_list(getattr(args, 'prediction_k_values', '10'))
    combine_mode = getattr(args, 'prediction_combine_paths', 'mean')
    rows = []

    for dataset, selected_ds in selected.groupby('dataset', dropna=False):
        print(f'[prediction] Dataset={dataset}: loading graph and labels...')
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        target_t = cfg['target_type']
        label_idx, y, label_vocab = extract_target_labels(
            data, target_t, policy=args.multi_label_policy, use_test_labels=args.use_test_labels
        )
        if len(label_idx) == 0 or len(np.unique(y)) < 2:
            warnings.warn(f'{dataset}: no usable labels or fewer than two classes; skipping prediction.')
            continue

        original_sims = {}
        for method in methods:
            print(f'[prediction] Dataset={dataset} original {method}...')
            original_sims[method] = _combined_similarity_matrix(data, mats, cfg, method, None, combine_mode)

        for _, sel in selected_ds.iterrows():
            cluster_of_by_type, mode, ts, tx = _build_selected_cluster_mapping(
                data, mats, cfg, parts, rawX_by_type, termB_by_type, sel, args
            )
            meta = {
                'dataset': dataset,
                'reduction_setting': sel.get('reduction_setting', mode),
                'selection_source': sel.get('selection_source', ''),
                'semantic_threshold': ts,
                'attribute_threshold': np.nan if tx is None else tx,
                'num_labeled_nodes': int(len(y)),
                'num_classes': int(len(np.unique(y))),
                'target_type': int(target_t),
                'target_type_name': type_names.get(target_t, str(target_t)),
                'prediction_combine_paths': combine_mode,
            }
            for method in methods:
                for graph_view, S in [
                    ('original', original_sims[method]),
                    ('reduced', _combined_similarity_matrix(data, mats, cfg, method, cluster_of_by_type, combine_mode))
                ]:
                    mets = _label_prediction_metrics(S, label_idx, y, k_values)
                    for m in mets:
                        row = dict(meta)
                        row.update({'sim_method': method, 'graph_view': graph_view})
                        row.update(m)
                        rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / 'prediction_selected_raw.csv', index=False)
    if raw.empty:
        raw.to_csv(outdir / 'prediction_selected_summary.csv', index=False)
        return raw
    group_cols = [
        'dataset', 'reduction_setting', 'selection_source', 'semantic_threshold', 'attribute_threshold',
        'sim_method', 'prediction_combine_paths', 'prediction_k'
    ]
    original = raw[raw['graph_view'] == 'original'].drop(columns=['graph_view']).rename(columns={
        'accuracy': 'accuracy_original', 'precision_macro': 'precision_original', 'f1_macro': 'f1_original'
    })
    reduced = raw[raw['graph_view'] == 'reduced'].drop(columns=['graph_view']).rename(columns={
        'accuracy': 'accuracy_reduced', 'precision_macro': 'precision_reduced', 'f1_macro': 'f1_reduced'
    })
    keep_cols = group_cols + ['num_labeled_nodes', 'num_classes', 'target_type_name']
    summary = reduced.merge(original, on=keep_cols, how='left')
    summary['delta_accuracy'] = summary['accuracy_reduced'] - summary['accuracy_original']
    summary['delta_precision'] = summary['precision_reduced'] - summary['precision_original']
    summary['delta_f1'] = summary['f1_reduced'] - summary['f1_original']
    summary.to_csv(outdir / 'prediction_selected_summary.csv', index=False)
    print(f'[prediction] Wrote {outdir / "prediction_selected_raw.csv"}')
    print(f'[prediction] Wrote {outdir / "prediction_selected_summary.csv"}')
    return summary


def run_kbisim_aligned_validation(args):
    """Run downstream checks aligned with the K-bisimulation task family.

    The protocol mirrors the task family used in k-bisimulation studies:
    clustering with KMeans/FCM/BSAS/XMeans and label prediction from top-k
    similarity neighbors. It compares original and selected reduced graphs under
    exactly the same input features/similarities.
    """
    configure_clustering_task(args, 'pdf')
    args.cluster_algorithms = 'kmeans,fcm,bsas,xmeans'
    print('[HHIN-KBISIM] Running clustering-preservation evaluation...')
    clustering_summary = run_clustering_selected(args)
    print('[HHIN-KBISIM] Running label-prediction preservation evaluation...')
    prediction_summary = run_prediction_selected(args)
    return clustering_summary, prediction_summary

def export_selected_semantic_matrices(args):
    """Export original and selected-reduced PathSim/HeteSim target-node matrices.

    This is intentionally limited to selected thresholds. Exporting all 100
    threshold pairs can require tens of GB on DBLP/IMDB and is usually not
    necessary for budget reselection. For changing semantic-budget values, use
    --task reselect on semantic_selection_summary.csv instead.
    """
    outdir = Path(args.out_dir) / 'semantic_matrices'
    outdir.mkdir(parents=True, exist_ok=True)
    selected_csv = selected_thresholds_path(args)
    if not selected_csv.exists():
        raise FileNotFoundError(f'Cannot find selected thresholds CSV: {selected_csv}')
    selected = pd.read_csv(selected_csv)
    if selected.empty:
        print('[export_semantic_matrices] Selected threshold CSV is empty.')
        return None

    manifest_rows = []
    for dataset, selected_ds in selected.groupby('dataset', dropna=False):
        if dataset not in set(args.datasets):
            continue
        print(f'[export_semantic_matrices] Dataset={dataset}: loading graph and selected threshold...')
        data, mats, cfg, parts, type_names, rawX_by_type, termB_by_type = prepare_dataset(dataset, args.base_dir)
        target_t = cfg['target_type']
        ds_dir = outdir / str(dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Save original matrices once per dataset/path.
        for pname, link_seq in cfg['meta_paths'].items():
            print(f'[export_semantic_matrices] Dataset={dataset} path={pname}: original matrices...')
            B_orig = base.compose_path(mats, link_seq)
            P_orig = pathsim_matrix_from_B(B_orig)
            BL, BR, BLn, BRn = base.compute_hetesim_path_factors(mats, data['link_defs'], link_seq, {})
            H_orig = hetesim_matrix_from_factors(BLn, BRn)
            p_path = ds_dir / f'{dataset}_{pname}_pathsim_original.npz'
            h_path = ds_dir / f'{dataset}_{pname}_hetesim_original.npz'
            np.savez_compressed(p_path, matrix=P_orig.astype(np.float32))
            np.savez_compressed(h_path, matrix=H_orig.astype(np.float32))
            manifest_rows.extend([
                {'dataset': dataset, 'path': pname, 'matrix_type': 'pathsim_original', 'file': str(p_path), 'shape': str(P_orig.shape)},
                {'dataset': dataset, 'path': pname, 'matrix_type': 'hetesim_original', 'file': str(h_path), 'shape': str(H_orig.shape)},
            ])
            del P_orig, H_orig

        for _, sel in selected_ds.iterrows():
            mode = selected_row_to_mode(sel)
            ts = float(sel['semantic_threshold'])
            tx_val = sel.get('attribute_threshold', np.nan)
            tx = None if pd.isna(tx_val) else float(tx_val)
            label = f'S{ts:g}_X{tx:g}'
            print(f'[export_semantic_matrices] Dataset={dataset} selected={label}: reduced matrices...')
            cluster_of_by_type, _ = build_clusters_for_mode(
                cfg, parts, rawX_by_type, termB_by_type, mode, ts, tx, args.semantic_sim
            )
            add_semantic_singletons(cluster_of_by_type, data, cfg)
            G_by_type = {t: base.build_group_matrix(cluster_of_by_type[t]) for t in cluster_of_by_type}
            target_cluster_of = cluster_of_by_type[target_t]

            for pname, link_seq in cfg['meta_paths'].items():
                B_red, _ = base.reduced_path_matrix(mats, data['link_defs'], link_seq, G_by_type)
                P_red_cluster = pathsim_matrix_from_B(B_red)
                P_red = P_red_cluster[target_cluster_of][:, target_cluster_of]
                BLr, BRr, BLnr, BRnr = base.compute_hetesim_path_factors(mats, data['link_defs'], link_seq, G_by_type)
                H_red_cluster = hetesim_matrix_from_factors(BLnr, BRnr)
                H_red = H_red_cluster[target_cluster_of][:, target_cluster_of]
                p_path = ds_dir / f'{dataset}_{pname}_pathsim_reduced_{label}.npz'
                h_path = ds_dir / f'{dataset}_{pname}_hetesim_reduced_{label}.npz'
                np.savez_compressed(p_path, matrix=P_red.astype(np.float32))
                np.savez_compressed(h_path, matrix=H_red.astype(np.float32))
                manifest_rows.extend([
                    {'dataset': dataset, 'path': pname, 'matrix_type': 'pathsim_reduced', 'threshold': label, 'file': str(p_path), 'shape': str(P_red.shape)},
                    {'dataset': dataset, 'path': pname, 'matrix_type': 'hetesim_reduced', 'threshold': label, 'file': str(h_path), 'shape': str(H_red.shape)},
                ])
                del P_red_cluster, P_red, H_red_cluster, H_red
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / 'semantic_matrix_manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    print(f'[export_semantic_matrices] Wrote manifest: {manifest_path}')
    return manifest

def build_arg_parser():
    ap = argparse.ArgumentParser(description='HHIN semantic-channel Guard and layered semantic-attribute Guard comparison experiments')
    ap.add_argument('--base-dir', type=str, default='.', help='Directory containing ACM.zip/DBLP.zip/IMDB.zip')
    ap.add_argument('--out-dir', type=str, default='./hhin_extra_experiments_out')
    ap.add_argument('--datasets', nargs='+', default=['ACM', 'DBLP', 'IMDB'])
    ap.add_argument('--task', choices=[
        'reduction', 'semantic_select', 'clustering_selected',
        'clustering_pdf', 'clustering_b3c', 'pipeline',
        'clustering', 'both', 'reselect', 'export_semantic_matrices',
        'retrieval_ranking_selected', 'prediction_selected', 'kbisim_aligned'
    ], default='pipeline')
    ap.add_argument('--semantic-sim', '--term-sim', dest='semantic_sim', choices=['cosine', 'jaccard', 'overlap'], default='cosine', help='Similarity for semantic-channel nodes: terms in ACM/DBLP and keywords in IMDB. --term-sim is kept as a backward-compatible alias.')
    ap.add_argument('--semantic-taus', '--term-taus', dest='semantic_taus', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0', help='Thresholds for semantic-channel Guard; --term-taus is kept as a backward-compatible alias.')
    ap.add_argument('--attr-taus', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    ap.add_argument('--include-term-only-ablation', action='store_true', help='Only for reduction contrast/ablation; term-only results are not used by semantic_select.')
    ap.add_argument('--write-plots', action='store_true')

    # semantic threshold selection settings
    ap.add_argument('--selection-mode', choices=['joint', 'pathsim', 'hetesim'], default='joint')
    ap.add_argument('--fallback-to', choices=['hetesim', 'pathsim', 'best_effort', 'none'], default='best_effort')
    ap.add_argument('--selection-rule', choices=['normalized_risk', 'hard'], default='normalized_risk',
                    help='normalized_risk selects max reduction under normalized semantic budget usage <= 1; hard uses legacy boolean constraints.')
    ap.add_argument('--summary-csv', type=str, default=None,
                    help='Existing semantic_selection_summary.csv for --task reselect; avoids rerunning PathSim/HeteSim.')
    ap.add_argument('--risk-budget-scale', type=float, default=1.0,
                    help='Budget multiplier: <1 stricter, >1 looser. Error budgets are multiplied; retention gaps are multiplied.')
    ap.add_argument('--include-top10-in-risk', action='store_true',
                    help='Include Top10 overlap in normalized hard risk. By default Top10 is a diagnostic because it is boundary-sensitive.')
    ap.add_argument('--objective-col', type=str, default='fullgraph_reduction_ratio',
                    help='Primary objective for normalized-risk selection. Default maximizes node reduction without artificial node/edge weights.')
    ap.add_argument('--reduction-node-weight', type=float, default=0.7)
    ap.add_argument('--reduction-edge-weight', type=float, default=0.3)
    ap.add_argument('--semantic-sample-size', type=int, default=0, help='0 means evaluate all target nodes')
    ap.add_argument('--semantic-sample-seed', type=int, default=42)
    ap.add_argument('--max-pathsim-mae', type=float, default=0.10)
    ap.add_argument('--max-pathsim-p95e', type=float, default=0.30)
    ap.add_argument('--max-pathsim-p99e', type=float, default=0.50)
    ap.add_argument('--min-pathsim-top10', type=float, default=0.90)
    ap.add_argument('--min-pathsim-ndcg10', type=float, default=0.95)
    ap.add_argument('--min-pathsim-pvalue', type=float, default=0.05)
    ap.add_argument('--max-hetesim-mae', type=float, default=0.10)
    ap.add_argument('--max-hetesim-p95e', type=float, default=0.30)
    ap.add_argument('--max-hetesim-p99e', type=float, default=0.50)
    ap.add_argument('--min-hetesim-top10', type=float, default=0.90)
    ap.add_argument('--min-hetesim-ndcg10', type=float, default=0.95)
    ap.add_argument('--min-hetesim-pvalue', type=float, default=0.05)

    # clustering settings
    ap.add_argument('--selected-thresholds-csv', type=str, default=None)
    ap.add_argument('--cluster-selected-only', action='store_true', help='For compatibility; selected-only is automatic for clustering_selected/pipeline.')
    ap.add_argument('--clustering-profile', choices=['selected', 'pdf', 'b3c'], default='selected')
    ap.add_argument('--sim-methods', type=str, default='pathsim,hetesim', help='pathsim,hetesim')
    ap.add_argument('--retrieval-k-values', type=str, default='1,5,10,20', help='k values for retrieval overlap/nDCG evaluation')
    ap.add_argument('--retrieval-sample-size', type=int, default=0, help='0 means all target nodes for retrieval/ranking validation')
    ap.add_argument('--write-retrieval-raw', action='store_true', help='Write per-query retrieval/ranking records')
    ap.add_argument('--prediction-k-values', type=str, default='10', help='top-k neighbor sizes for label prediction, e.g. 1,5,10')
    ap.add_argument('--prediction-combine-paths', choices=['mean'], default='mean', help='How to combine meta-path similarity matrices for prediction')
    ap.add_argument('--combine-paths', choices=['concat', 'mean'], default='concat')
    ap.add_argument('--cluster-algorithms', type=str, default='kmeans,fcm,bsas,xmeans')
    ap.add_argument('--kmeans-n-init', type=int, default=10, help='Initializations per seed; five distinct seeds are reported by default.')
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46', help='Five runs by default, following the reference protocol')
    ap.add_argument('--multi-label-policy', choices=['first', 'drop'], default='first')
    label_scope = ap.add_mutually_exclusive_group()
    label_scope.add_argument(
        '--all-labels', '--use-test-labels', dest='use_test_labels', action='store_true',
        help='Use labels from label.dat and label.dat.test (default for unsupervised and transductive evaluation).'
    )
    label_scope.add_argument(
        '--train-labels-only', dest='use_test_labels', action='store_false',
        help='Restrict clustering and transductive label-neighborhood evaluation to label.dat.'
    )
    ap.set_defaults(use_test_labels=True)

    # B3C-aligned deep clustering settings.
    ap.add_argument('--deep-pretrain-epochs', type=int, default=80)
    ap.add_argument('--deep-dec-epochs', type=int, default=80)
    ap.add_argument('--deep-hidden-dim', type=int, default=128)
    ap.add_argument('--deep-embed-dim', type=int, default=32)
    ap.add_argument('--deep-pca-dim', type=int, default=256, help='Reduce similarity-row features before deep training; <=0 disables PCA.')
    ap.add_argument('--deep-lr', type=float, default=1e-3)
    ap.add_argument('--deep-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--verbose-training', action='store_true')
    return ap


def configure_clustering_task(args, profile: str):
    args.clustering_profile = profile
    if profile == 'pdf':
        args.cluster_algorithms = 'kmeans,fcm,bsas,xmeans'
    elif profile == 'b3c':
        args.cluster_algorithms = 'deepdec'
    return args


def main():
    args = build_arg_parser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.task == 'reselect':
        print('[HHIN] Reselecting thresholds from existing semantic summary...')
        reselect_from_summary(args)
        print(f'[HHIN] Done. Outputs written to: {args.out_dir}')
        return
    if args.task == 'export_semantic_matrices':
        print('[HHIN] Exporting selected PathSim/HeteSim matrices...')
        export_selected_semantic_matrices(args)
        print(f'[HHIN] Done. Outputs written to: {args.out_dir}')
        return
    if args.task == 'retrieval_ranking_selected':
        print('[HHIN] Running selected-threshold retrieval/ranking validation...')
        run_retrieval_ranking_selected(args)
        print(f'[HHIN] Done. Outputs written to: {args.out_dir}')
        return
    if args.task == 'prediction_selected':
        print('[HHIN] Running selected-threshold label-prediction validation...')
        run_prediction_selected(args)
        print(f'[HHIN] Done. Outputs written to: {args.out_dir}')
        return
    if args.task == 'kbisim_aligned':
        print('[HHIN] Running K-bisimulation-aligned clustering and prediction validation...')
        run_kbisim_aligned_validation(args)
        print(f'[HHIN] Done. Outputs written to: {args.out_dir}')
        return
    if args.task in ('reduction', 'both'):
        print('[HHIN] Running reduction-effect contrast...')
        run_reduction_contrast(args)
    if args.task in ('semantic_select', 'pipeline'):
        print('[HHIN] Running semantic threshold selection...')
        run_semantic_select(args)
    if args.task == 'clustering_pdf':
        configure_clustering_task(args, 'pdf')
        print('[HHIN] Running PDF-aligned selected-threshold clustering evaluation...')
        run_clustering_selected(args)
    if args.task == 'clustering_b3c':
        configure_clustering_task(args, 'b3c')
        print('[HHIN] Running B3C-aligned deep selected-threshold clustering evaluation...')
        run_clustering_selected(args)
    if args.task == 'pipeline':
        configure_clustering_task(args, 'pdf')
        print('[HHIN] Running PDF-aligned selected-threshold clustering evaluation...')
        run_clustering_selected(args)
    if args.task == 'clustering_selected':
        print('[HHIN] Running selected-threshold clustering evaluation...')
        run_clustering_selected(args)
    if args.task == 'clustering' and args.cluster_selected_only:
        print('[HHIN] Running selected-threshold clustering evaluation...')
        run_clustering_selected(args)
    elif args.task in ('clustering', 'both'):
        print('[HHIN] Running clustering-preservation evaluation...')
        run_clustering_preservation(args)
    print(f'[HHIN] Done. Outputs written to: {args.out_dir}')


if __name__ == '__main__':
    main()

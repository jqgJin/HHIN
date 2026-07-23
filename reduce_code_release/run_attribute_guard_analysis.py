#!/usr/bin/env python3
"""Evaluate the incremental contribution of the DBLP attached-feature Guard."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import hhin_core as core
import hhin_experiments as exp
import run_guard_component_ablation as guard


ATTRIBUTE_SETTINGS = (
    ("term_only", "Term Guard", None),
    ("attribute_0_1", r"Term + Attribute (0.1)", 0.1),
    ("attribute_0_5", r"Term + Attribute (0.5)", 0.5),
    ("attribute_1_0", r"Term + Attribute (1.0)", 1.0),
)


def grouped_members(mapping: np.ndarray) -> list[np.ndarray]:
    members = defaultdict(list)
    for node, cluster in enumerate(mapping.tolist()):
        members[int(cluster)].append(node)
    return [np.asarray(members[key], dtype=np.int32) for key in sorted(members)]


def quantile(values, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if values.size else np.nan


def feature_and_label_metrics(
    features: sp.csr_matrix,
    mapping: np.ndarray,
    labels: np.ndarray,
) -> dict:
    features = features.tocsr().astype(np.float64)
    row_norm = np.sqrt(np.asarray(features.multiply(features).sum(axis=1)).ravel())
    nonzero = row_norm > 0
    normalized = normalize(features, norm="l2", axis=1, copy=True)
    groups = core.build_group_matrix(mapping).astype(np.float64)
    centroids = normalize(groups @ normalized, norm="l2", axis=1, copy=True)
    fidelity = np.asarray(
        normalized.multiply(centroids[mapping]).sum(axis=1)
    ).ravel()
    fidelity = np.clip(fidelity[nonzero], -1.0, 1.0)
    loss = 1.0 - fidelity

    pair_cosines = []
    nonzero_pair_cosines = []
    zero_involved_pairs = 0
    label_disagreement_pairs = 0
    impure_clusters = 0
    authors_in_impure_clusters = 0
    merged_clusters = 0
    merged_authors = 0
    max_cluster_size = 1
    for members in grouped_members(mapping):
        size = len(members)
        max_cluster_size = max(max_cluster_size, size)
        if size < 2:
            continue
        merged_clusters += 1
        merged_authors += size
        similarities = (normalized[members] @ normalized[members].T).toarray()
        lower_index = np.tril_indices(size, k=-1)
        lower = np.clip(similarities[lower_index], -1.0, 1.0)
        pair_cosines.extend(lower.tolist())
        pair_nonzero = nonzero[members][lower_index[0]] & nonzero[members][lower_index[1]]
        nonzero_pair_cosines.extend(lower[pair_nonzero].tolist())
        zero_involved_pairs += int(np.sum(~pair_nonzero))
        cluster_labels = labels[members]
        _, counts = np.unique(cluster_labels, return_counts=True)
        disagreement = size * (size - 1) // 2 - int(np.sum(counts * (counts - 1) // 2))
        label_disagreement_pairs += disagreement
        if len(counts) > 1:
            impure_clusters += 1
            authors_in_impure_clusters += size

    pair_cosines = np.asarray(pair_cosines, dtype=np.float64)
    nonzero_pair_cosines = np.asarray(nonzero_pair_cosines, dtype=np.float64)
    cluster_count = int(mapping.max()) + 1
    return {
        "author_clusters": cluster_count,
        "author_removed": len(mapping) - cluster_count,
        "author_rr": 1.0 - cluster_count / len(mapping),
        "merged_clusters": merged_clusters,
        "merged_authors": merged_authors,
        "max_cluster_size": max_cluster_size,
        "nonzero_feature_authors": int(nonzero.sum()),
        "zero_feature_authors": int((~nonzero).sum()),
        "feature_cosine_mean": float(np.mean(fidelity)),
        "feature_cosine_p05": quantile(fidelity, 0.05),
        "feature_cosine_min": float(np.min(fidelity)),
        "feature_loss_mean": float(np.mean(loss)),
        "feature_loss_p95": quantile(loss, 0.95),
        "feature_loss_max": float(np.max(loss)),
        "merged_pair_count": int(pair_cosines.size),
        "zero_involved_merged_pairs": zero_involved_pairs,
        "nonzero_merged_pair_count": int(nonzero_pair_cosines.size),
        "merged_pair_cosine_mean": float(np.mean(pair_cosines)) if pair_cosines.size else np.nan,
        "merged_pair_cosine_p05": quantile(pair_cosines, 0.05),
        "merged_pair_cosine_min": float(np.min(pair_cosines)) if pair_cosines.size else np.nan,
        "merged_pairs_below_0_1": int(np.sum(pair_cosines < 0.1 - 1e-8)),
        "merged_pairs_below_0_5": int(np.sum(pair_cosines < 0.5 - 1e-8)),
        "merged_pairs_below_0_999999": int(np.sum(pair_cosines < 0.999999)),
        "merged_pairs_below_guard_cutoff": int(np.sum(pair_cosines < 1.0 - 1e-8)),
        "nonzero_pair_cosine_mean": float(np.mean(nonzero_pair_cosines)) if nonzero_pair_cosines.size else np.nan,
        "nonzero_pair_cosine_p05": quantile(nonzero_pair_cosines, 0.05),
        "nonzero_pairs_below_0_999999": int(np.sum(nonzero_pair_cosines < 0.999999)),
        "nonzero_pairs_below_guard_cutoff": int(np.sum(nonzero_pair_cosines < 1.0 - 1e-8)),
        "impure_label_clusters": impure_clusters,
        "authors_in_impure_label_clusters": authors_in_impure_clusters,
        "label_disagreement_pairs": label_disagreement_pairs,
    }


def build_attribute_mappings(config, partitions, attributes, terms, tau_h: float):
    target_type = config["target_type"]
    term_only = {}
    for node_type in config["main_types"]:
        node_count = len(partitions[node_type])
        empty_attributes = sp.csr_matrix((node_count, 0), dtype=np.float32)
        mapping, _ = core.make_layered_threshold_mapping(
            partitions[node_type],
            empty_attributes,
            terms[node_type],
            tau_term=tau_h,
            tau_attr=0.0,
            term_sim="cosine",
        )
        term_only[node_type] = mapping

    mappings = {"term_only": term_only}
    for key, _, tau_x in ATTRIBUTE_SETTINGS[1:]:
        mapping = dict(term_only)
        author_mapping, _ = core.make_layered_threshold_mapping(
            partitions[target_type],
            attributes[target_type],
            terms[target_type],
            tau_term=tau_h,
            tau_attr=tau_x,
            term_sim="cosine",
        )
        mapping[target_type] = author_mapping
        mappings[key] = mapping
    return mappings


def reduction_semantic_rows(data, matrices, config, mappings, tau_h: float):
    original_nodes = sum(len(nodes) for nodes in data["ids_by_type"].values())
    original_main = sum(
        len(data["ids_by_type"][node_type]) for node_type in config["main_types"]
    )
    original_edges = core.count_original_edges(data)
    non_main = original_nodes - original_main
    original_cache = core.build_original_semantic_cache(data, matrices, config)
    rows = []
    paths = []
    labels = {key: label for key, label, _ in ATTRIBUTE_SETTINGS}
    thresholds = {key: tau_x for key, _, tau_x in ATTRIBUTE_SETTINGS}
    for key, mapping in mappings.items():
        print(f"[semantic] {labels[key]}", flush=True)
        reduced_main = sum(int(values.max()) + 1 for values in mapping.values())
        reduced_nodes = reduced_main + non_main
        reduced_edges = core.count_reduced_edges(data, matrices, mapping)
        per_path, metrics = core.evaluate_semantics(
            data, matrices, config, mapping, orig_cache=original_cache
        )
        row = {
            "configuration": key,
            "label": labels[key],
            "tau_h": tau_h,
            "tau_x": thresholds[key],
            "full_node_rr": 1.0 - reduced_nodes / original_nodes,
            "main_node_rr": 1.0 - reduced_main / original_main,
            "edge_rr": 1.0 - reduced_edges / original_edges,
            "r_sem": guard.semantic_risk(metrics),
        }
        row.update(metrics)
        rows.append(row)
        paths.extend({"configuration": key, "label": labels[key], **item} for item in per_path)
    return pd.DataFrame(rows), pd.DataFrame(paths)


def random_refinement(
    coarse_mapping: np.ndarray,
    observed_refinement: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize split locations within coarse-class size strata."""
    coarse_groups = grouped_members(coarse_mapping)
    groups_by_size = defaultdict(list)
    split_patterns_by_size = defaultdict(list)
    for group_id, members in enumerate(coarse_groups):
        size = len(members)
        groups_by_size[size].append(group_id)
        _, block_sizes = np.unique(observed_refinement[members], return_counts=True)
        if len(block_sizes) > 1:
            split_patterns_by_size[size].append(tuple(sorted(block_sizes.tolist())))

    assigned_patterns = {}
    for size, patterns in split_patterns_by_size.items():
        eligible = np.asarray(groups_by_size[size], dtype=np.int32)
        chosen = rng.choice(eligible, size=len(patterns), replace=False)
        shuffled_patterns = [patterns[index] for index in rng.permutation(len(patterns))]
        assigned_patterns.update(
            {int(group_id): pattern for group_id, pattern in zip(chosen, shuffled_patterns)}
        )

    refined = np.empty_like(coarse_mapping)
    next_cluster = 0
    for group_id, members in enumerate(coarse_groups):
        block_sizes = assigned_patterns.get(group_id, (len(members),))
        shuffled = rng.permutation(members)
        start = 0
        for size in block_sizes:
            block = shuffled[start : start + size]
            refined[block] = next_cluster
            next_cluster += 1
            start += size
    target_clusters = int(observed_refinement.max()) + 1
    if next_cluster != target_clusters:
        raise AssertionError("Matched randomization changed the refined cluster count")
    return refined


def refinement_audit(fine: np.ndarray, coarse: np.ndarray) -> dict:
    violations = 0
    for members in grouped_members(fine):
        if len(np.unique(coarse[members])) != 1:
            violations += 1
    return {
        "fine_is_refinement": violations == 0,
        "violating_fine_clusters": violations,
    }


def split_pair_count(coarse: np.ndarray, fine: np.ndarray) -> int:
    total = 0
    for members in grouped_members(coarse):
        size = len(members)
        total += size * (size - 1) // 2
        _, counts = np.unique(fine[members], return_counts=True)
        total -= int(np.sum(counts * (counts - 1) // 2))
    return total


def random_control(
    term_mapping: np.ndarray,
    attribute_mapping: np.ndarray,
    features: sp.csr_matrix,
    labels: np.ndarray,
    runs: int,
    seed: int,
):
    actual = feature_and_label_metrics(features, attribute_mapping, labels)
    rows = []
    for run in range(runs):
        mapping = random_refinement(
            term_mapping, attribute_mapping, np.random.default_rng(seed + run)
        )
        row = feature_and_label_metrics(features, mapping, labels)
        row.update({"run": run, "seed": seed + run})
        rows.append(row)
    raw = pd.DataFrame(rows)
    comparisons = []
    for metric in (
        "feature_loss_mean",
        "feature_loss_p95",
        "merged_pair_cosine_mean",
        "merged_pairs_below_0_999999",
        "impure_label_clusters",
        "label_disagreement_pairs",
    ):
        values = raw[metric].to_numpy(dtype=np.float64)
        actual_value = float(actual[metric])
        lower_is_better = metric != "merged_pair_cosine_mean"
        if lower_is_better:
            empirical_p = (1 + int(np.sum(values <= actual_value + 1e-15))) / (runs + 1)
        else:
            empirical_p = (1 + int(np.sum(values >= actual_value - 1e-15))) / (runs + 1)
        comparisons.append(
            {
                "metric": metric,
                "direction": "lower" if lower_is_better else "higher",
                "attribute_guard": actual_value,
                "random_mean": float(np.mean(values)),
                "random_std": float(np.std(values, ddof=1)),
                "random_min": float(np.min(values)),
                "random_max": float(np.max(values)),
                "empirical_p": empirical_p,
            }
        )
    return raw, pd.DataFrame(comparisons)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau-h", type=float, default=0.4)
    parser.add_argument("--random-runs", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260721)
    parser.add_argument(
        "--skip-semantics",
        action="store_true",
        help="Run mechanism and matched-random diagnostics without recomputing path metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[setup] Loading DBLP and computing the shared fixed point", flush=True)
    data, matrices, config, partitions, attributes, terms, history = guard.load_dblp(
        args.data_dir
    )
    mappings = build_attribute_mappings(
        config, partitions, attributes, terms, args.tau_h
    )
    target_type = config["target_type"]
    label_idx, label_values, _ = exp.extract_target_labels(
        data, target_type, policy="first", use_test_labels=True
    )
    labels = np.empty(len(data["ids_by_type"][target_type]), dtype=np.int32)
    labels[label_idx] = label_values
    features = attributes[target_type]

    print("[mechanism] Computing feature and label diagnostics", flush=True)
    mechanism_rows = []
    setting_labels = {key: label for key, label, _ in ATTRIBUTE_SETTINGS}
    setting_thresholds = {key: threshold for key, _, threshold in ATTRIBUTE_SETTINGS}
    for key, mapping in mappings.items():
        row = feature_and_label_metrics(features, mapping[target_type], labels)
        row.update(
            {
                "configuration": key,
                "label": setting_labels[key],
                "tau_h": args.tau_h,
                "tau_x": setting_thresholds[key],
            }
        )
        mechanism_rows.append(row)
    mechanism = pd.DataFrame(mechanism_rows)

    sensitivity = None
    paths = None
    if not args.skip_semantics:
        semantic, paths = reduction_semantic_rows(
            data, matrices, config, mappings, args.tau_h
        )
        sensitivity = semantic.merge(
            mechanism,
            on=["configuration", "label", "tau_h", "tau_x"],
            how="left",
        )

    term_mapping = mappings["term_only"][target_type]
    attribute_mapping = mappings["attribute_1_0"][target_type]
    audit = refinement_audit(attribute_mapping, term_mapping)
    audit["term_pairs_split_by_attribute_guard"] = split_pair_count(
        term_mapping, attribute_mapping
    )
    audit["term_author_clusters"] = int(term_mapping.max()) + 1
    audit["attribute_author_clusters"] = int(attribute_mapping.max()) + 1
    print("[random control] Running matched refinements", flush=True)
    random_raw, random_summary = random_control(
        term_mapping,
        attribute_mapping,
        features,
        labels,
        args.random_runs,
        args.random_seed,
    )

    mechanism.to_csv(args.output_dir / "attribute_guard_mechanism.csv", index=False)
    if sensitivity is not None:
        sensitivity.to_csv(args.output_dir / "attribute_guard_sensitivity.csv", index=False)
        paths.to_csv(args.output_dir / "attribute_guard_semantics_by_path.csv", index=False)
    random_raw.to_csv(args.output_dir / "attribute_guard_random_control_raw.csv", index=False)
    random_summary.to_csv(
        args.output_dir / "attribute_guard_random_control_summary.csv", index=False
    )
    metadata = {
        "dataset": "DBLP",
        "tau_h": args.tau_h,
        "attribute_thresholds": [item[2] for item in ATTRIBUTE_SETTINGS],
        "random_runs": args.random_runs,
        "random_seed": args.random_seed,
        "fixed_point_iterations": len(history),
        "random_control_rule": (
            "Randomize which Term-Guard candidate classes are split within each "
            "coarse-class size stratum while preserving the observed numbers and "
            "sizes of Attribute-Guard subclusters."
        ),
        "refinement_audit": audit,
    }
    with (args.output_dir / "attribute_guard_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    if sensitivity is not None:
        print(sensitivity.to_string(index=False), flush=True)
    print(random_summary.to_string(index=False), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

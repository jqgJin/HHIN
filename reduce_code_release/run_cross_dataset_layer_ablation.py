#!/usr/bin/env python3
"""Run one cross-dataset architecture and Guard-component ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

import hhin_core as core
import run_reduction_analysis as reduction


@dataclass(frozen=True)
class GuardSetting:
    key: str
    label: str
    use_term: bool
    use_attribute: bool


SETTINGS = (
    GuardSetting("structure", "Structure only", False, False),
    GuardSetting("term", "Term Guard", True, False),
    GuardSetting("attribute", "Attribute Guard", False, True),
    GuardSetting("both", "Both Guards", True, True),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_risk(metrics: dict[str, float]) -> float:
    values = []
    for prefix in ("pathsim", "hetesim"):
        values.extend(
            (
                metrics[f"{prefix}_mae"] / 0.10,
                metrics[f"{prefix}_p95e"] / 0.30,
                metrics[f"{prefix}_p99e"] / 0.50,
                (1.0 - metrics[f"{prefix}_ndcg10"]) / 0.05,
            )
        )
    return float(max(values))


def build_component_mappings(data, matrices, config, partitions, tau_h, tau_x):
    attributes = {}
    terms = {}
    for node_type in config["main_types"]:
        attributes[node_type] = core.get_raw_attr_for_type(
            data, node_type, config["strong_attr_types"]
        )
        terms[node_type] = core.derive_term_matrix(
            matrices, config["term_paths_by_type"][node_type]
        ).tocsr().astype(np.float32)

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
            by_type[node_type], _ = core.make_layered_threshold_mapping(
                partitions[node_type],
                raw_attribute,
                term_matrix,
                tau_term=tau_h,
                tau_attr=tau_x,
                term_sim="cosine",
            )
        mappings[setting.key] = by_type
    return mappings, attributes, terms


def evaluate_mapping(data, matrices, config, mapping, original_cache):
    original_full = sum(len(ids) for ids in data["ids_by_type"].values())
    original_main = sum(
        len(data["ids_by_type"][node_type]) for node_type in config["main_types"]
    )
    target_type = config["target_type"]
    original_target = len(data["ids_by_type"][target_type])
    reduced_main = sum(int(mapping[node_type].max()) + 1 for node_type in config["main_types"])
    reduced_target = int(mapping[target_type].max()) + 1
    reduced_full = reduced_main + original_full - original_main
    original_edges = core.count_original_edges(data)
    reduced_edges = core.count_reduced_edges(data, matrices, mapping)
    _, metrics = core.evaluate_semantics(
        data, matrices, config, mapping, orig_cache=original_cache
    )
    return {
        "main_rc": original_main - reduced_main,
        "target_rc": original_target - reduced_target,
        "full_node_rr": 1.0 - reduced_full / original_full,
        "main_node_rr": 1.0 - reduced_main / original_main,
        "target_node_rr": 1.0 - reduced_target / original_target,
        "edge_rr": 1.0 - reduced_edges / original_edges,
        "pathsim_ndcg10": metrics["pathsim_ndcg10"],
        "hetesim_ndcg10": metrics["hetesim_ndcg10"],
        "r_sem": semantic_risk(metrics),
        **{
            key: metrics[key]
            for key in (
                "pathsim_mae", "pathsim_p95e", "pathsim_p99e",
                "hetesim_mae", "hetesim_p95e", "hetesim_p99e",
            )
        },
    }


def audit_candidate_pairs(config, partitions, attributes, terms, tau_h, tau_x):
    rows = []
    total = Counter()
    for node_type in config["main_types"]:
        classes = defaultdict(list)
        for index, label in enumerate(partitions[node_type]):
            classes[int(label)].append(index)
        counts = Counter()
        for members in classes.values():
            if len(members) < 2:
                continue
            sim_attr, sim_term = core.layered_similarity_matrices(
                attributes[node_type][members], terms[node_type][members], term_sim="cosine"
            )
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    pass_term = (
                        sim_term is None
                        or float(sim_term[left, right]) >= tau_h - 1e-8
                    )
                    pass_attribute = (
                        sim_attr is None
                        or float(sim_attr[left, right]) >= tau_x - 1e-10
                    )
                    if pass_term and pass_attribute:
                        counts["accepted"] += 1
                    elif not pass_term and pass_attribute:
                        counts["term_only_reject"] += 1
                    elif pass_term and not pass_attribute:
                        counts["attribute_only_reject"] += 1
                    else:
                        counts["both_reject"] += 1
        total.update(counts)
        rows.append({"node_type": node_type, **counts})
    rows.append({"node_type": "all_main_types", **total})
    return rows


def plot_summary(architecture, attribution, output_dir):
    for column in (
        "accepted", "term_only_reject", "attribute_only_reject", "both_reject"
    ):
        if column not in attribution:
            attribution[column] = 0
    datasets = architecture["dataset"].drop_duplicates().tolist()
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    colors = {"Mixed HIN": "#8A8F98", "HHIN layered": "#247BA0"}
    width = 0.34
    for offset, method in zip((-width / 2, width / 2), colors):
        values = [
            float(
                architecture[
                    (architecture["dataset"] == dataset)
                    & (architecture["architecture"] == method)
                ]["full_node_rr"].iloc[0]
            )
            * 100
            for dataset in datasets
        ]
        axes[0].bar(x + offset, values, width, label=method, color=colors[method])
    axes[0].set_xticks(x, datasets)
    axes[0].set_ylabel("Full-node reduction ratio (%)")
    axes[0].set_title("Architecture control")
    axes[0].legend(frameon=False)

    aggregate = attribution[attribution["node_type"] == "all_main_types"].set_index("dataset")
    channels = (
        ("term_only_reject", "Term guard", "#E07A5F"),
        ("attribute_only_reject", "Attribute guard", "#3D9970"),
    )
    for offset, (key, label, color) in zip((-width / 2, width / 2), channels):
        values = np.asarray([float(aggregate.loc[d, key]) for d in datasets])
        bars = axes[1].bar(x + offset, values, width, label=label, color=color)
        axes[1].bar_label(
            bars,
            labels=[f"{int(value)}" for value in values],
            padding=2,
            fontsize=8,
        )
    axes[1].set_xticks(x, datasets)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylabel("Uniquely rejected candidate pairs")
    axes[1].set_title("Independent guard decisions")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig_cross_dataset_layer_guard_ablation.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selected-thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="ACM,DBLP,IMDB")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from existing CSV outputs.",
    )
    return parser.parse_args()


def write_metadata(args, datasets):
    hashes = {
        f"{dataset}.zip": file_sha256(args.data_dir / f"{dataset}.zip")
        for dataset in datasets
    }
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "datasets": datasets,
        "selected_thresholds": str(args.selected_thresholds),
        "data_sha256": hashes,
        "ranking_tie_rule": "similarity descending, original node identifier ascending",
        "ndcg_tie_rule": "mean gain within each tied predicted-score group",
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if args.plot_only:
        architecture = pd.read_csv(args.output_dir / "architecture_control.csv")
        attribution = pd.read_csv(args.output_dir / "guard_pair_attribution.csv").fillna(0)
        plot_summary(architecture, attribution, args.output_dir)
        write_metadata(args, datasets)
        print(f"Figure regenerated in {args.output_dir}", flush=True)
        return
    selected = pd.read_csv(args.selected_thresholds)
    architecture_rows = []
    component_rows = []
    attribution_rows = []
    for dataset in datasets:
        print(f"[{dataset}] loading data and fixed-point candidates", flush=True)
        archive = args.data_dir / f"{dataset}.zip"
        loaded = core.load_hgb_zip(str(archive))
        matrices = core.build_link_matrices(loaded)
        data = core._strip_unpickleable_data(loaded)
        config = core.CONFIGS[dataset]
        row = selected[selected["dataset"] == dataset].iloc[0]
        tau_h = float(row["semantic_threshold"])
        tau_x = float(row["attribute_threshold"])
        partitions, history = core.fixed_point_partition(
            data, matrices, config["core_link_types"]
        )
        mappings, attributes, terms = build_component_mappings(
            data, matrices, config, partitions, tau_h, tau_x
        )
        cache = core.build_original_semantic_cache(data, matrices, config)

        for setting in SETTINGS:
            print(f"[{dataset}] evaluating {setting.label}", flush=True)
            metrics = evaluate_mapping(
                data, matrices, config, mappings[setting.key], cache
            )
            component_rows.append(
                {
                    "dataset": dataset,
                    "configuration": setting.key,
                    "label": setting.label,
                    "tau_h": tau_h,
                    "tau_x": tau_x,
                    **metrics,
                }
            )

        print(f"[{dataset}] evaluating single-layer mixed control", flush=True)
        mixed_parts, mixed_history, mixed_mapping = reduction.build_mixed_hin_control(
            data, matrices, config, tau_x
        )
        mixed_metrics = evaluate_mapping(data, matrices, config, mixed_mapping, cache)
        layered_metrics = next(
            item
            for item in component_rows
            if item["dataset"] == dataset and item["configuration"] == "both"
        )
        architecture_rows.extend(
            (
                {
                    "dataset": dataset,
                    "architecture": "Mixed HIN",
                    "tau_h": tau_h,
                    "tau_x": tau_x,
                    "fixed_point_iterations": len(mixed_history),
                    **mixed_metrics,
                },
                {
                    "dataset": dataset,
                    "architecture": "HHIN layered",
                    "tau_h": tau_h,
                    "tau_x": tau_x,
                    "fixed_point_iterations": len(history),
                    **{
                        key: value
                        for key, value in layered_metrics.items()
                        if key not in {"dataset", "configuration", "label", "tau_h", "tau_x"}
                    },
                },
            )
        )

        for pair_row in audit_candidate_pairs(
            config, partitions, attributes, terms, tau_h, tau_x
        ):
            attribution_rows.append(
                {"dataset": dataset, "tau_h": tau_h, "tau_x": tau_x, **pair_row}
            )

    architecture = pd.DataFrame(architecture_rows)
    components = pd.DataFrame(component_rows)
    attribution = pd.DataFrame(attribution_rows).fillna(0)
    architecture.to_csv(args.output_dir / "architecture_control.csv", index=False)
    components.to_csv(args.output_dir / "guard_component_ablation.csv", index=False)
    attribution.to_csv(args.output_dir / "guard_pair_attribution.csv", index=False)
    plot_summary(architecture, attribution, args.output_dir)

    write_metadata(args, datasets)
    print(f"Results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

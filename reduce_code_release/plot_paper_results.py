#!/usr/bin/env python3
"""Generate the paper figures from audited experiment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RED = "#b91f1f"
TEAL = "#2f8585"
BLUE = "#4c78a8"
GRAY = "#6b6b6b"


def read_approx(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[frame["mode"] == "approx"].copy()


def save_figure(fig, output_dir: Path, stem: str):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_trends(args):
    grid = pd.read_csv(args.grid_summary)
    grid = grid.loc[grid["mode"] == "layered_semantic_attribute"].copy()
    grid["semantic_threshold"] = pd.to_numeric(grid["semantic_threshold"])
    grid["attribute_threshold"] = pd.to_numeric(grid["attribute_threshold"])
    diagonal = grid.loc[
        np.isclose(grid["semantic_threshold"], grid["attribute_threshold"])
    ].copy()

    endpoints = read_approx(args.strict_endpoints)
    dblp = read_approx(args.dblp_strict_summary)
    endpoints = pd.concat(
        [endpoints, dblp.loc[np.isclose(dblp["term_threshold"], 1.0)]],
        ignore_index=True,
    )
    for row in endpoints.itertuples():
        mask = (diagonal["dataset"] == row.dataset) & np.isclose(
            diagonal["semantic_threshold"], 1.0
        )
        diagonal.loc[mask, "fullgraph_reduction_ratio"] = row.fullgraph_reduction_ratio

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    styles = {
        "ACM": (RED, "o"),
        "DBLP": (BLUE, "s"),
        "IMDB": (GRAY, "^"),
    }
    for dataset in ("ACM", "DBLP", "IMDB"):
        subset = diagonal.loc[diagonal["dataset"] == dataset].sort_values(
            "semantic_threshold"
        )
        color, marker = styles[dataset]
        ax.plot(
            subset["semantic_threshold"],
            100 * subset["fullgraph_reduction_ratio"],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=dataset,
        )
    ax.set_title("Reduction Trends under Threshold Sweeping", fontsize=18)
    ax.set_xlabel(r"Threshold $T$", fontsize=14)
    ax.set_ylabel("Full-Graph Node Reduction Ratio (%)", fontsize=14)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, args.output_dir, "fig_hhin_three_dataset_reduction_trends")


def profile_rows(args) -> pd.DataFrame:
    aggressive = read_approx(args.aggressive_summary)
    strict = read_approx(args.dblp_strict_summary)
    selected = pd.concat(
        [
            aggressive.loc[
                np.isclose(aggressive["term_threshold"], 0.1)
                & np.isclose(aggressive["attr_threshold"], 0.1)
            ].assign(profile="Aggressive"),
            strict.loc[np.isclose(strict["term_threshold"], 0.4)].assign(
                profile="Balanced"
            ),
            strict.loc[np.isclose(strict["term_threshold"], 0.6)].assign(
                profile="Conservative"
            ),
        ],
        ignore_index=True,
    )
    order = pd.CategoricalDtype(
        ["Aggressive", "Balanced", "Conservative"], ordered=True
    )
    selected["profile"] = selected["profile"].astype(order)
    selected = selected.sort_values("profile")

    retrieval = pd.read_csv(args.retrieval_profile_summary)
    profile_names = selected["profile"].astype(str).tolist()
    for row_index, profile in zip(selected.index, profile_names):
        subset = retrieval.loc[retrieval["selection_source"] == profile]
        pathsim = subset.loc[subset["sim_method"] == "pathsim"].iloc[0]
        hetesim = subset.loc[subset["sim_method"] == "hetesim"].iloc[0]
        selected.loc[row_index, "avg_pathsim_top10"] = pathsim["overlap@10"]
        selected.loc[row_index, "avg_pathsim_ndcg10"] = pathsim["ndcg@10"]
        selected.loc[row_index, "avg_hetesim_top10"] = hetesim["overlap@10"]
        selected.loc[row_index, "avg_hetesim_ndcg10"] = hetesim["ndcg@10"]
    return selected


def plot_profile_tradeoff(args):
    profiles = profile_rows(args)
    labels = profiles["profile"].astype(str).tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    width = 0.35
    axes[0].bar(
        x - width / 2,
        100 * profiles["fullgraph_reduction_ratio"],
        width,
        color=RED,
        label="Node RR",
    )
    axes[0].bar(
        x + width / 2,
        100 * profiles["fullgraph_edge_reduction_ratio"],
        width,
        color=TEAL,
        label="Edge RR",
    )
    axes[0].set_title("Reduction Strength")
    axes[0].set_ylabel("Reduction Ratio (%)")
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylim(0, 10)
    axes[0].legend(frameon=False)

    axes[1].plot(x, profiles["avg_pathsim_top10"], "o-", color=RED, label="P-Top10")
    axes[1].plot(x, profiles["avg_hetesim_top10"], "s-", color=TEAL, label="H-Top10")
    axes[1].plot(
        x,
        profiles["avg_pathsim_ndcg10"],
        "^--",
        color=RED,
        label="P-nDCG@10",
    )
    axes[1].plot(
        x,
        profiles["avg_hetesim_ndcg10"],
        "D--",
        color=TEAL,
        label="H-nDCG@10",
    )
    axes[1].set_title("Retrieval and Ranking")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.64, 1.01)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, args.output_dir, "fig_results_dblp_profile_tradeoff")


def plot_profile_retrieval(args):
    profiles = profile_rows(args)
    labels = profiles["profile"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.30
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(
        x - width / 2,
        profiles["avg_pathsim_top10"],
        width,
        color=RED,
        label="PathSim",
    )
    axes[0].bar(
        x + width / 2,
        profiles["avg_hetesim_top10"],
        width,
        color=TEAL,
        label="HeteSim",
    )
    axes[0].set_title("Top-10 Set Consistency")
    axes[0].set_ylabel("Top10")
    axes[0].set_ylim(0.62, 0.90)
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2,
        profiles["avg_pathsim_ndcg10"],
        width,
        color=RED,
        label="PathSim",
    )
    axes[1].bar(
        x + width / 2,
        profiles["avg_hetesim_ndcg10"],
        width,
        color=TEAL,
        label="HeteSim",
    )
    axes[1].set_title("Ranking Quality")
    axes[1].set_ylabel("nDCG@10")
    axes[1].set_ylim(0.96, 1.002)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, args.output_dir, "fig_results_dblp_profile_retrieval")


def plot_balanced_paths(args):
    paths = pd.read_csv(args.retrieval_profile_per_path)
    paths = paths.loc[paths["selection_source"] == "Balanced"]
    order = ["APA", "APVPA", "APTPA"]
    pathsim = paths.loc[paths["sim_method"] == "pathsim"].set_index("path").loc[order]
    hetesim = paths.loc[paths["sim_method"] == "hetesim"].set_index("path").loc[order]
    x = np.arange(len(order))
    width = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(
        x - width / 2,
        pathsim["overlap@10"],
        width,
        color=RED,
        label="PathSim",
    )
    axes[0].bar(
        x + width / 2,
        hetesim["overlap@10"],
        width,
        color=TEAL,
        label="HeteSim",
    )
    axes[0].set_title("Top-10 Set Consistency")
    axes[0].set_ylabel("Top10")
    axes[0].set_ylim(0.45, 1.02)
    axes[0].legend(frameon=False)

    axes[1].plot(
        x,
        pathsim["ndcg@10"],
        "o-",
        color=RED,
        label="PathSim nDCG@10",
    )
    axes[1].plot(
        x,
        hetesim["ndcg@10"],
        "s-",
        color=TEAL,
        label="HeteSim nDCG@10",
    )
    axes[1].set_title("Ranking Quality")
    axes[1].set_ylabel("nDCG")
    axes[1].set_ylim(0.96, 1.002)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xticks(x, order)
        ax.grid(True, linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, args.output_dir, "fig_dblp_balanced_per_path_retrieval")


def plot_three_dataset_retrieval(args):
    selected = pd.read_csv(args.selected_three_summary).set_index("dataset")
    retrieval = pd.read_csv(args.retrieval_selected_summary)
    for dataset in ("ACM", "DBLP", "IMDB"):
        subset = retrieval.loc[retrieval["dataset"] == dataset]
        for method in ("pathsim", "hetesim"):
            row = subset.loc[subset["sim_method"] == method].iloc[0]
            selected.loc[dataset, f"avg_{method}_top10"] = row["overlap@10"]
            selected.loc[dataset, f"avg_{method}_ndcg10"] = row["ndcg@10"]
    selected = selected.loc[["ACM", "DBLP", "IMDB"]]
    x = np.arange(len(selected))
    width = 0.30
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(
        x - width / 2,
        selected["avg_pathsim_top10"],
        width,
        color=RED,
        label="PathSim",
    )
    axes[0].bar(
        x + width / 2,
        selected["avg_hetesim_top10"],
        width,
        color=TEAL,
        label="HeteSim",
    )
    axes[0].set_title("Top-10 Set Consistency")
    axes[0].set_ylabel("Top10")
    axes[0].set_ylim(0.60, 1.01)
    axes[0].legend(frameon=False)
    axes[1].bar(
        x - width / 2,
        selected["avg_pathsim_ndcg10"],
        width,
        color=RED,
        label="PathSim",
    )
    axes[1].bar(
        x + width / 2,
        selected["avg_hetesim_ndcg10"],
        width,
        color=TEAL,
        label="HeteSim",
    )
    axes[1].set_title("Ranking Quality")
    axes[1].set_ylabel("nDCG@10")
    axes[1].set_ylim(0.96, 1.002)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, selected.index.tolist())
        ax.grid(True, axis="y", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, args.output_dir, "fig_three_dataset_retrieval_ranking")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-summary", type=Path, required=True)
    parser.add_argument("--strict-endpoints", type=Path, required=True)
    parser.add_argument("--aggressive-summary", type=Path, required=True)
    parser.add_argument("--dblp-strict-summary", type=Path, required=True)
    parser.add_argument("--retrieval-profile-summary", type=Path, required=True)
    parser.add_argument("--retrieval-profile-per-path", type=Path, required=True)
    parser.add_argument("--retrieval-selected-summary", type=Path, required=True)
    parser.add_argument("--selected-three-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    if not args.retrieval_only:
        plot_threshold_trends(args)
    plot_profile_tradeoff(args)
    plot_profile_retrieval(args)
    plot_balanced_paths(args)
    plot_three_dataset_retrieval(args)


if __name__ == "__main__":
    main()

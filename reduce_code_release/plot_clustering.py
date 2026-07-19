#!/usr/bin/env python3
"""Plot original and reduced DBLP clustering in a shared PCA space."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hhin_experiments import (
    add_semantic_singletons,
    build_clusters_for_mode,
    build_similarity_feature,
    cluster_acc,
    clustering_metrics,
    extract_target_labels,
    map_clusters_to_labels,
    prepare_dataset,
    run_clustering_algorithm,
    selected_row_to_mode,
)


def _timestamped_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = root / f"clustering_visual_compare_{stamp}"
    suffix = 1
    while out.exists():
        out = root / f"clustering_visual_compare_{stamp}_{suffix:02d}"
        suffix += 1
    out.mkdir(parents=True, exist_ok=False)
    return out


def _copy_without_overwrite(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stem, suffix = src.stem, src.suffix
    counter = 1
    dst = dst_dir / f"{stem}_v{counter}{suffix}"
    while dst.exists():
        counter += 1
        dst = dst_dir / f"{stem}_v{counter}{suffix}"
    shutil.copy2(src, dst)
    return dst


def _read_selected_row(path: Path, dataset: str, profile: str) -> pd.Series:
    selected = pd.read_csv(path)
    selected_ds = selected[selected["dataset"].astype(str).str.upper() == dataset.upper()]
    if "selection_source" in selected_ds.columns and profile:
        selected_ds = selected_ds[
            selected_ds["selection_source"].astype(str).str.lower() == profile.lower()
        ]
    if selected_ds.empty:
        raise ValueError(f"No selected threshold row for dataset={dataset} in {path}")
    return selected_ds.iloc[0]


DBLP_LABEL_NAMES = {
    "0": "Database",
    "1": "Data Mining",
    "2": "AI",
    "3": "Information Retrieval",
}


def _label_names(label_vocab: dict[int | str, int], dataset: str) -> dict[int, str]:
    names = DBLP_LABEL_NAMES if dataset.upper() == "DBLP" else {}
    return {int(encoded): names.get(str(raw), f"Class {raw}") for raw, encoded in label_vocab.items()}


def _metric_text(metrics: dict[str, float]) -> str:
    return (
        f"Mean NMI={metrics['nmi']:.3f}, ACC={metrics['acc']:.3f}, Prec.={metrics['precision_macro']:.3f}"
    )


def _plot_comparison(
    out_png: Path,
    out_pdf: Path,
    coords_original: np.ndarray,
    coords_reduced: np.ndarray,
    y_original_mapped: np.ndarray,
    y_reduced_mapped: np.ndarray,
    label_id_to_name: dict[int, str],
    metrics_original: dict[str, float],
    metrics_reduced: dict[str, float],
    title: str = "",
    show_metrics: bool = False,
):
    labels = sorted(set(y_original_mapped.tolist()) | set(y_reduced_mapped.tolist()))
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85), sharex=True, sharey=True)
    panels = [
        ("Original graph", coords_original, y_original_mapped, metrics_original),
        ("Reduced graph", coords_reduced, y_reduced_mapped, metrics_reduced),
    ]

    for ax, (name, coords, mapped_labels, metrics) in zip(axes, panels):
        for lab in labels:
            idx = mapped_labels == lab
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=9,
                alpha=0.72,
                linewidths=0,
                c=[color_map[lab]],
                label=label_id_to_name.get(lab, str(lab)),
            )
        if show_metrics:
            ax.set_title(f"{name}\n{_metric_text(metrics)}", fontsize=8.8, pad=6)
        else:
            ax.set_title(name, fontsize=9.5, pad=6)
        ax.grid(True, color="#e8e8e8", linewidth=0.6)
        ax.tick_params(labelsize=7.8)
        ax.set_xlabel("PCA-1", fontsize=8.7)
    axes[0].set_ylabel("PCA-2", fontsize=8.7)

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        loc="lower center",
        ncol=min(4, max(1, len(labels_text))),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.015),
    )
    if title:
        fig.suptitle(title, fontsize=9.2, y=0.995)
        fig.tight_layout(rect=[0.0, 0.1, 1.0, 0.94])
    else:
        fig.tight_layout(rect=[0.0, 0.1, 1.0, 1.0])
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create original-vs-reduced clustering visualization.")
    ap.add_argument("--base-dir", default=".", help="Directory containing HGB zip files.")
    ap.add_argument("--dataset", default="DBLP", choices=["DBLP"], help="Dataset to visualize.")
    ap.add_argument(
        "--selected-thresholds-csv",
        default=str(Path(__file__).resolve().parent / "config" / "dblp_profiles.csv"),
        help="Selected threshold CSV used to rebuild the reduced graph.",
    )
    ap.add_argument("--profile", default="Balanced", choices=["Aggressive", "Balanced", "Conservative"])
    ap.add_argument("--out-root", default="./clustering_visualization_out", help="Root output directory.")
    ap.add_argument("--sim-method", default="pathsim", choices=["pathsim", "hetesim"])
    ap.add_argument("--combine-paths", default="concat", choices=["concat", "mean"])
    ap.add_argument("--cluster-algorithm", default="kmeans", choices=["kmeans", "fcm", "bsas", "xmeans"])
    ap.add_argument("--seeds", default="42,43,44,45,46", help="Seeds used to compute mean clustering metrics.")
    ap.add_argument("--semantic-sim", default="cosine", choices=["cosine", "jaccard", "overlap"])
    ap.add_argument("--multi-label-policy", default="first", choices=["first", "drop"])
    label_scope = ap.add_mutually_exclusive_group()
    label_scope.add_argument("--all-labels", "--use-test-labels", dest="use_test_labels", action="store_true")
    label_scope.add_argument("--train-labels-only", dest="use_test_labels", action="store_false")
    ap.set_defaults(use_test_labels=True)
    ap.add_argument("--sample-size", type=int, default=0, help="Optional sample size for scatter plotting only.")
    ap.add_argument(
        "--paper-figures-dir",
        default="",
        help="Optional paper figures directory. Use empty string to skip copying.",
    )
    ap.add_argument("--show-metrics", action="store_true", help="Show mean clustering metrics in panel titles.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    base_dir = Path(args.base_dir)
    outdir = _timestamped_dir(Path(args.out_root))
    selected_csv = Path(args.selected_thresholds_csv)
    selected_row = _read_selected_row(selected_csv, args.dataset, args.profile)

    print(f"[visual] Loading {args.dataset} from {base_dir} ...")
    data, mats, cfg, parts, _type_names, rawX_by_type, termB_by_type = prepare_dataset(args.dataset, str(base_dir))
    target_t = cfg["target_type"]
    label_idx, y_true, label_vocab = extract_target_labels(
        data,
        target_t,
        policy=args.multi_label_policy,
        use_test_labels=args.use_test_labels,
    )
    if len(label_idx) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("No usable labels, or fewer than two label classes.")

    mode = selected_row_to_mode(selected_row)
    tau_sem = float(selected_row["semantic_threshold"])
    tau_attr_value = selected_row.get("attribute_threshold", np.nan)
    tau_attr = None if pd.isna(tau_attr_value) else float(tau_attr_value)
    print(f"[visual] Rebuilding reduced graph with tau_s={tau_sem:g}, tau_x={tau_attr:g} ...")
    cluster_of_by_type, _ = build_clusters_for_mode(
        cfg,
        parts,
        rawX_by_type,
        termB_by_type,
        mode,
        tau_sem,
        tau_attr,
        args.semantic_sim,
    )
    add_semantic_singletons(cluster_of_by_type, data, cfg)

    print(f"[visual] Computing {args.sim_method} features ...")
    X_original_full = build_similarity_feature(data, mats, cfg, None, method=args.sim_method, combine_paths=args.combine_paths)
    X_reduced_full = build_similarity_feature(
        data,
        mats,
        cfg,
        cluster_of_by_type,
        method=args.sim_method,
        combine_paths=args.combine_paths,
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_original = scaler.fit_transform(np.asarray(X_original_full[label_idx], dtype=np.float32))
    X_reduced = scaler.transform(np.asarray(X_reduced_full[label_idx], dtype=np.float32))

    n_classes = len(np.unique(y_true))
    print(f"[visual] Running {args.cluster_algorithm} with k={n_classes}, seeds={seeds} ...")
    per_seed_rows = []
    pred_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        pred_original = run_clustering_algorithm(args.cluster_algorithm, X_original, n_classes, seed)
        pred_reduced = run_clustering_algorithm(args.cluster_algorithm, X_reduced, n_classes, seed)
        pred_by_seed[seed] = {"original": pred_original, "reduced": pred_reduced}
        for graph_view, X_eval, pred in [
            ("original", X_original, pred_original),
            ("reduced", X_reduced, pred_reduced),
        ]:
            met = clustering_metrics(X_eval, y_true, pred)
            met["acc"] = cluster_acc(y_true, pred)
            per_seed_rows.append(
                {
                    "dataset": args.dataset,
                    "sim_method": args.sim_method,
                    "cluster_algorithm": args.cluster_algorithm,
                    "seed": seed,
                    "semantic_threshold": tau_sem,
                    "attribute_threshold": tau_attr,
                    "graph_view": graph_view,
                    "nmi": met["nmi"],
                    "acc": met["acc"],
                    "precision_macro": met["precision_macro"],
                    "sc": met["sc"],
                    "chi": met["chi"],
                }
            )

    per_seed_metrics = pd.DataFrame(per_seed_rows)
    mean_metrics = per_seed_metrics.groupby("graph_view", as_index=False).agg(
        nmi=("nmi", "mean"),
        acc=("acc", "mean"),
        precision_macro=("precision_macro", "mean"),
        sc=("sc", "mean"),
        chi=("chi", "mean"),
    )
    metrics_original = mean_metrics[mean_metrics["graph_view"] == "original"].iloc[0].to_dict()
    metrics_reduced = mean_metrics[mean_metrics["graph_view"] == "reduced"].iloc[0].to_dict()

    def representative_distance(seed: int) -> float:
        rows = per_seed_metrics[per_seed_metrics["seed"] == seed].set_index("graph_view")
        score = 0.0
        for view, means in [("original", metrics_original), ("reduced", metrics_reduced)]:
            for key in ("nmi", "acc", "precision_macro"):
                score += abs(float(rows.loc[view, key]) - float(means[key]))
        return score

    representative_seed = min(seeds, key=representative_distance)
    print(f"[visual] Representative seed for scatter plot: {representative_seed}")

    mapped_original = map_clusters_to_labels(y_true, pred_by_seed[representative_seed]["original"])
    mapped_reduced = map_clusters_to_labels(y_true, pred_by_seed[representative_seed]["reduced"])

    projection_input = np.vstack([X_original, X_reduced])
    coords = PCA(n_components=2, random_state=representative_seed).fit_transform(projection_input)
    coords_original = coords[: len(y_true)]
    coords_reduced = coords[len(y_true) :]

    plot_idx = np.arange(len(y_true))
    if args.sample_size and args.sample_size < len(plot_idx):
        rng = np.random.default_rng(representative_seed)
        plot_idx = np.sort(rng.choice(plot_idx, size=args.sample_size, replace=False))

    label_id_to_name = _label_names(label_vocab, args.dataset)
    seed_label = "seeds" + "-".join(str(s) for s in seeds)
    base_name = (
        f"fig_{args.dataset.lower()}_cluster_ovr_"
        f"{args.sim_method}_{args.cluster_algorithm}_{seed_label}"
        f"_{'metrics' if args.show_metrics else 'nometrics'}"
    )
    out_png = outdir / f"{base_name}.png"
    out_pdf = outdir / f"{base_name}.pdf"
    title = ""
    _plot_comparison(
        out_png,
        out_pdf,
        coords_original[plot_idx],
        coords_reduced[plot_idx],
        mapped_original[plot_idx],
        mapped_reduced[plot_idx],
        label_id_to_name,
        metrics_original,
        metrics_reduced,
        title,
        show_metrics=args.show_metrics,
    )

    node_rows = pd.DataFrame(
        {
            "local_node_index": label_idx,
            "true_label_id": y_true,
            "true_label_name": [label_id_to_name[int(v)] for v in y_true],
            "representative_seed": representative_seed,
            "original_cluster": pred_by_seed[representative_seed]["original"],
            "reduced_cluster": pred_by_seed[representative_seed]["reduced"],
            "original_mapped_label_id": mapped_original,
            "reduced_mapped_label_id": mapped_reduced,
            "original_pca1": coords_original[:, 0],
            "original_pca2": coords_original[:, 1],
            "reduced_pca1": coords_reduced[:, 0],
            "reduced_pca2": coords_reduced[:, 1],
        }
    )
    node_rows.to_csv(outdir / "clustering_visualization_node_projection.csv", index=False)

    per_seed_metrics.to_csv(outdir / "clustering_visualization_metrics_by_seed.csv", index=False)
    mean_metrics.insert(0, "dataset", args.dataset)
    mean_metrics.insert(1, "sim_method", args.sim_method)
    mean_metrics.insert(2, "cluster_algorithm", args.cluster_algorithm)
    mean_metrics.insert(3, "seeds", ",".join(str(s) for s in seeds))
    mean_metrics.insert(4, "representative_seed", representative_seed)
    mean_metrics.insert(5, "semantic_threshold", tau_sem)
    mean_metrics.insert(6, "attribute_threshold", tau_attr)
    mean_metrics.to_csv(outdir / "clustering_visualization_metrics_mean.csv", index=False)

    copied = []
    if args.paper_figures_dir:
        figures_dir = Path(args.paper_figures_dir)
        copied.append(_copy_without_overwrite(out_png, figures_dir))
        copied.append(_copy_without_overwrite(out_pdf, figures_dir))

    print(f"[visual] Wrote figure: {out_png}")
    print(f"[visual] Wrote figure: {out_pdf}")
    print(f"[visual] Wrote node projection CSV: {outdir / 'clustering_visualization_node_projection.csv'}")
    print(f"[visual] Wrote per-seed metrics CSV: {outdir / 'clustering_visualization_metrics_by_seed.csv'}")
    print(f"[visual] Wrote mean metrics CSV: {outdir / 'clustering_visualization_metrics_mean.csv'}")
    for dst in copied:
        print(f"[visual] Copied to paper figures: {dst}")


if __name__ == "__main__":
    main()

"""Validate dataset archives and the result files shipped with this release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hhin_core as core


EXPECTED = {
    "ACM": {
        "sha256": "787766FEF7526310321B8AC94EB220209C0876A536F6603640D812220CA62134",
        "nodes": {0: 3025, 1: 5959, 2: 56, 3: 1902},
        "edges": 547872,
    },
    "DBLP": {
        "sha256": "0D3EA4A74399F9CD3E83AF206E8E0B67E1844FE2C8463B424189884DD58AD7C8",
        "nodes": {0: 4057, 1: 14328, 2: 7723, 3: 20},
        "edges": 239566,
    },
    "IMDB": {
        "sha256": "DC98438C28F738AB1E7BA07AAECA4FC2E035D837B7BB3955FDD4F61F14DD81E1",
        "nodes": {0: 4932, 1: 2393, 2: 6124, 3: 7971},
        "edges": 86642,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_archives(data_dir: Path, failures: list[str]) -> dict:
    report = {}
    for dataset, expected in EXPECTED.items():
        archive = data_dir / f"{dataset}.zip"
        require(archive.is_file(), f"Missing dataset archive: {archive}", failures)
        if not archive.is_file():
            continue

        digest = sha256(archive)
        data = core.load_hgb_zip(str(archive))
        node_counts = {int(t): len(ids) for t, ids in data["ids_by_type"].items()}
        edge_count = core.count_original_edges(data)
        require(digest == expected["sha256"], f"{dataset}: SHA-256 mismatch", failures)
        require(node_counts == expected["nodes"], f"{dataset}: node counts differ: {node_counts}", failures)
        require(edge_count == expected["edges"], f"{dataset}: edge count differs: {edge_count}", failures)
        report[dataset] = {
            "sha256": digest,
            "node_counts": node_counts,
            "total_nodes": int(sum(node_counts.values())),
            "edges": int(edge_count),
        }

        if dataset == "DBLP":
            target = core.CONFIGS["DBLP"]["target_type"]
            label_counts = {}
            for filename in ("label.dat", "label.dat.test"):
                rows = [
                    row
                    for row in data["labels"].get(filename, [])
                    if int(row[1]) == int(target)
                ]
                label_counts[filename] = len(rows)
            label_counts["all"] = label_counts["label.dat"] + label_counts["label.dat.test"]
            require(label_counts["label.dat"] == 1217, "DBLP: expected 1,217 label.dat rows", failures)
            require(label_counts["label.dat.test"] == 2840, "DBLP: expected 2,840 label.dat.test rows", failures)
            require(label_counts["all"] == 4057, "DBLP: expected 4,057 public author labels", failures)
            report[dataset]["labels"] = label_counts
    return report


def validate_results(results_dir: Path, failures: list[str]) -> dict:
    report = {}

    selected_path = results_dir / "semantic_selection" / "selected_thresholds.csv"
    if selected_path.is_file():
        selected = pd.read_csv(selected_path)
        require(len(selected) == 3, "Selected-threshold table must contain three datasets", failures)
        require((selected["risk_semantic"] <= 1.0 + 1e-12).all(), "A selected threshold exceeds the semantic budget", failures)
        report["selected_thresholds"] = selected[
            ["dataset", "semantic_threshold", "attribute_threshold", "risk_semantic"]
        ].to_dict(orient="records")
    else:
        failures.append(f"Missing result file: {selected_path}")

    clustering_path = results_dir / "clustering" / "clustering_summary.csv"
    if clustering_path.is_file():
        clustering = pd.read_csv(clustering_path)
        require((clustering["num_labeled_nodes"] == 4057).all(), "Clustering did not use all 4,057 labels", failures)
        for method, frame in clustering.groupby("sim_method"):
            spread = float(frame["NMI_ori_mean"].max() - frame["NMI_ori_mean"].min())
            require(spread <= 1e-12, f"{method}: original clustering baseline differs across profiles", failures)
        report["clustering_rows"] = int(len(clustering))
    else:
        failures.append(f"Missing result file: {clustering_path}")

    neighbor_path = results_dir / "label_neighborhood" / "label_neighborhood_summary.csv"
    if neighbor_path.is_file():
        neighbor = pd.read_csv(neighbor_path)
        require((neighbor["num_labeled_nodes"] == 4057).all(), "Label-neighborhood evaluation did not use all labels", failures)
        for method, frame in neighbor.groupby("sim_method"):
            for metric in ("accuracy_original", "f1_original"):
                spread = float(frame[metric].max() - frame[metric].min())
                require(spread <= 1e-12, f"{method}: original {metric} differs across profiles", failures)
        report["label_neighborhood_rows"] = int(len(neighbor))
    else:
        failures.append(f"Missing result file: {neighbor_path}")

    case_path = results_dir / "reduction_analysis" / "dblp_case_all_candidate_pairs.csv"
    if case_path.is_file():
        case = pd.read_csv(case_path)
        for column in ("high_value_cosine", "attached_attribute_cosine"):
            if column in case:
                values = pd.to_numeric(case[column], errors="coerce").dropna().to_numpy()
                require(np.all(values >= -1.0 - 1e-7), f"{column}: value below -1", failures)
                require(np.all(values <= 1.0 + 1e-7), f"{column}: value above 1", failures)
        report["case_pairs"] = int(len(case))

    rgcn_path = results_dir / "rgcn" / "hgb_rgcn_raw.csv"
    if rgcn_path.is_file():
        rgcn = pd.read_csv(rgcn_path)
        require(
            set(rgcn["graph_view"]) == {"main_original", "main_reduced"},
            "R-GCN: expected original and reduced main-structure views",
            failures,
        )
        require(
            set(rgcn["seed"].astype(int)) == {42, 43, 44, 45, 46},
            "R-GCN: expected seeds 42 through 46",
            failures,
        )
        require(
            (rgcn["test_examples"].astype(int) == 2840).all(),
            "R-GCN: both graph views must use all 2,840 test authors",
            failures,
        )
        report["rgcn"] = {
            "rows": int(len(rgcn)),
            "seeds": sorted(rgcn["seed"].astype(int).unique().tolist()),
            "test_examples": int(rgcn["test_examples"].iloc[0]),
        }
    else:
        failures.append(f"Missing result file: {rgcn_path}")

    han_path = results_dir / "han" / "han_raw.csv"
    if han_path.is_file():
        han = pd.read_csv(han_path)
        require(
            set(han["graph_view"]) == {"main_original", "main_reduced"},
            "HAN: expected original and reduced main-structure views",
            failures,
        )
        require(
            set(han["seed"].astype(int)) == {42, 43, 44, 45, 46},
            "HAN: expected seeds 42 through 46",
            failures,
        )
        require(
            (han["test_examples"].astype(int) == 2840).all(),
            "HAN: original and reduced views must use all 2,840 test authors",
            failures,
        )
        parameters = han.groupby("graph_view")["parameters"].first()
        require(
            parameters.nunique() == 1,
            "HAN: parameter counts differ between graph views",
            failures,
        )
        report["han"] = {
            "rows": int(len(han)),
            "seeds": sorted(han["seed"].astype(int).unique().tolist()),
            "test_examples": int(han["test_examples"].iloc[0]),
            "parameters": int(parameters.iloc[0]),
        }
    else:
        failures.append(f"Missing result file: {han_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing ACM.zip, DBLP.zip, and IMDB.zip")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    failures: list[str] = []
    report = {
        "datasets": validate_archives(args.data_dir.resolve(), failures),
        "results": validate_results(args.results_dir.resolve(), failures),
        "status": "failed" if failures else "passed",
        "failures": failures,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

"""Shared DBLP reduction and split utilities for the GNN experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import hhin_core as core


def selected_dblp_thresholds(path: Path) -> tuple[float, float]:
    rows = pd.read_csv(path)
    row = rows.loc[rows["dataset"].str.upper() == "DBLP"].iloc[0]
    return float(row["semantic_threshold"]), float(row["attribute_threshold"])


def build_reduction(data, matrices, config, tau_h: float, tau_x: float):
    partitions, history = core.fixed_point_partition(
        data, matrices, config["core_link_types"]
    )
    cluster_of = {}
    for node_type in config["main_types"]:
        raw_attributes = core.get_raw_attr_for_type(
            data, node_type, config["strong_attr_types"]
        )
        high_value_attributes = core.derive_term_matrix(
            matrices, config["term_paths_by_type"][node_type]
        ).tocsr().astype(np.float32)
        mapping, _ = core.make_layered_threshold_mapping(
            partitions[node_type],
            raw_attributes,
            high_value_attributes,
            tau_term=tau_h,
            tau_attr=tau_x,
            term_sim="cosine",
        )
        cluster_of[node_type] = mapping
    return cluster_of, len(history)


def label_records(data, config, offsets, cluster_of=None):
    target_type = config["target_type"]
    global_to_local = {
        int(global_id): local
        for local, global_id in enumerate(data["ids_by_type"][target_type])
    }

    def convert(records):
        converted = []
        for global_id, node_type, label_text in records:
            if int(node_type) != target_type or int(global_id) not in global_to_local:
                continue
            local = global_to_local[int(global_id)]
            if cluster_of is None:
                model_index = offsets[target_type] + local
            else:
                model_index = offsets[target_type] + int(cluster_of[target_type][local])
            converted.append((model_index, int(label_text), local))
        return converted

    train_records = convert(data["labels"]["label.dat"])
    test_records = convert(data["labels"]["label.dat.test"])
    if cluster_of is None:
        return train_records, test_records, 0, 0

    test_clusters = {record[0] for record in test_records}
    labels_by_cluster = defaultdict(set)
    original_records_by_cluster = defaultdict(list)
    for record in train_records:
        labels_by_cluster[record[0]].add(record[1])
        original_records_by_cluster[record[0]].append(record)

    filtered = []
    excluded_overlap = 0
    excluded_conflict = 0
    for cluster, records in original_records_by_cluster.items():
        if cluster in test_clusters:
            excluded_overlap += len(records)
            continue
        if len(labels_by_cluster[cluster]) != 1:
            excluded_conflict += len(records)
            continue
        filtered.append((cluster, records[0][1], records[0][2]))
    return filtered, test_records, excluded_overlap, excluded_conflict


def cluster_split_units(
    original_records,
    reduced_records,
    cluster_mapping: np.ndarray,
    reduced_target_offset: int,
):
    """Build label-consistent cluster units shared by both graph views."""
    reduced_labels = {
        int(model_index - reduced_target_offset): int(label)
        for model_index, label, _ in reduced_records
    }
    original_by_cluster: dict[int, list[tuple[int, int, int]]] = {
        cluster: [] for cluster in reduced_labels
    }
    for record in original_records:
        cluster = int(cluster_mapping[record[2]])
        if cluster in original_by_cluster:
            original_by_cluster[cluster].append(record)

    units = []
    for cluster in sorted(reduced_labels):
        members = original_by_cluster[cluster]
        if not members:
            continue
        label = reduced_labels[cluster]
        if any(int(record[1]) != label for record in members):
            raise ValueError(f"Cluster {cluster} is not label-consistent")
        units.append((cluster, label, members))
    return units


def paired_split(units, seed: int, reduced_target_offset: int):
    unit_ids = np.arange(len(units), dtype=np.int64)
    unit_labels = np.array([unit[1] for unit in units], dtype=np.int64)
    train_units, validation_units = train_test_split(
        unit_ids,
        test_size=0.2,
        random_state=seed,
        stratify=unit_labels,
    )

    def reduced_arrays(selected):
        indices = np.array(
            [reduced_target_offset + units[index][0] for index in selected],
            dtype=np.int64,
        )
        labels = np.array([units[index][1] for index in selected], dtype=np.int64)
        return indices, labels

    def original_arrays(selected):
        records = [record for index in selected for record in units[index][2]]
        indices = np.array([record[0] for record in records], dtype=np.int64)
        labels = np.array([record[1] for record in records], dtype=np.int64)
        return indices, labels

    original_train = original_arrays(train_units)
    original_validation = original_arrays(validation_units)
    reduced_train = reduced_arrays(train_units)
    reduced_validation = reduced_arrays(validation_units)
    return {
        "Original": (*original_train, *original_validation),
        "Reduced": (*reduced_train, *reduced_validation),
        "train_clusters": len(train_units),
        "validation_clusters": len(validation_units),
    }

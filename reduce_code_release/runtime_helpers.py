"""Small utilities shared by the retrieval and ranking benchmark."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

import hhin_core as core


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def choose_query_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=sample_size, replace=False)).astype(np.int32)


def topk_indices(scores: np.ndarray, k: int, self_index: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).copy()
    if 0 <= self_index < len(values):
        values[self_index] = -np.inf
    k = min(max(int(k), 0), max(len(values) - 1, 0))
    if k == 0:
        return np.empty(0, dtype=np.int32)
    return core.deterministic_topk_indices(values, k).astype(np.int32)


def fresh_run_dir(base_dir: Path, label: str, allow_overwrite: bool) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    if allow_overwrite:
        run_dir = base_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{label}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{label}_{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir

#!/usr/bin/env python3
"""Build graph-aware coarse node assignments for the R0723 P-GUTS experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp/results/0723_official_pguts_coarse_imputation"
TSL_CACHE = RESULTS / "tsl_cache"
OUT_DIR = RESULTS / "coarse_assignments"

DIST_FILES = {
    "la": TSL_CACHE / "MetrLA/metr_la_dist.npy",
    "bay": TSL_CACHE / "PemsBay/pems_bay_dist.npy",
}


def symmetrize_distance(dist: np.ndarray) -> np.ndarray:
    dist = dist.astype(np.float64, copy=True)
    np.fill_diagonal(dist, 0.0)
    rev = dist.T
    both = np.stack([dist, rev], axis=0)
    finite = np.isfinite(both)
    out = np.full(dist.shape, np.inf, dtype=np.float64)
    any_finite = finite.any(axis=0)
    out[any_finite] = np.nanmin(np.where(finite, both, np.nan), axis=0)[any_finite]
    np.fill_diagonal(out, 0.0)
    return out


def greedy_distance_assignment(dist: np.ndarray, coarse_factor: int) -> np.ndarray:
    """Group each seed node with its nearest unassigned graph neighbors."""

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"Expected square distance matrix, got {dist.shape}")
    n_nodes = dist.shape[0]
    coarse_factor = max(1, int(coarse_factor))
    dist = symmetrize_distance(dist)
    finite_degree = np.isfinite(dist).sum(axis=1)
    # Prefer well-connected nodes as cluster centers; this avoids isolated nodes
    # becoming centers before their nearby connected components are consumed.
    center_order = np.lexsort((np.arange(n_nodes), -finite_degree))
    unassigned = np.ones(n_nodes, dtype=bool)
    assignment = np.full(n_nodes, -1, dtype=np.int64)
    cluster_id = 0

    for center in center_order:
        if not unassigned[center]:
            continue
        candidates = np.flatnonzero(unassigned)
        candidate_dist = dist[center, candidates]
        finite_candidates = candidates[np.isfinite(candidate_dist)]
        if finite_candidates.size:
            order = np.argsort(dist[center, finite_candidates], kind="stable")
            members = finite_candidates[order[:coarse_factor]]
        else:
            members = candidates[:coarse_factor]
        assignment[members] = cluster_id
        unassigned[members] = False
        cluster_id += 1

    if (assignment < 0).any():
        raise RuntimeError("Some nodes were not assigned to coarse clusters.")
    return assignment


def write_assignment(ds_key: str, coarse_factor: int) -> Path:
    dist_path = DIST_FILES[ds_key]
    assignment = greedy_distance_assignment(np.load(dist_path), coarse_factor)
    out_path = OUT_DIR / f"{ds_key}_distance_greedy_cf{coarse_factor}.npy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_path, assignment)

    counts = np.bincount(assignment)
    print(
        f"{ds_key}: nodes={assignment.size} coarse_nodes={counts.size} "
        f"min_size={counts.min()} max_size={counts.max()} path={out_path}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-factor", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ds_key in ("la", "bay"):
        write_assignment(ds_key, args.coarse_factor)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build METR-LA functional coarse assignments from time-series signals.

This script is for the post-0723 experiment: keep the official P-GUTS
imputation protocol unchanged, but replace distance-greedy spatial grouping
with coarse groups inferred only from variable time-series relationships.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "dataset/metr_la/metr_la.h5"
DEFAULT_OUT = ROOT / "missing_ts_exp/results/0818_metrla_functional_cg/coarse_assignments"


def load_h5_matrix(path: Path) -> np.ndarray:
    """Load a traffic matrix as T x N using pandas first, h5py as fallback."""

    try:
        import pandas as pd  # type: ignore

        df = pd.read_hdf(path)
        values = df.to_numpy(dtype=np.float64)
        if values.ndim == 2:
            return values
    except Exception:
        pass

    try:
        import h5py  # type: ignore

        with h5py.File(path, "r") as h5:
            datasets: list[np.ndarray] = []

            def visitor(_, obj):
                if hasattr(obj, "shape") and len(obj.shape) in {2, 3}:
                    datasets.append(np.asarray(obj))

            h5.visititems(visitor)
        if not datasets:
            raise ValueError(f"No 2D/3D dataset found in {path}")
        values = max(datasets, key=lambda arr: arr.size).astype(np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 2:
            raise ValueError(f"Expected T x N matrix, got {values.shape}")
        return values
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}. Install pandas+tables or h5py in the experiment env."
        ) from exc


def training_slice(x: np.ndarray, train_ratio: float) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected T x N matrix, got {x.shape}")
    n_train = int(round(x.shape[0] * train_ratio))
    n_train = max(1, min(x.shape[0], n_train))
    return x[:n_train].astype(np.float64, copy=True)


def standardize_observed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x)
    means = np.nanmean(np.where(mask, x, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    centered = np.where(mask, x - means.reshape(1, -1), 0.0)
    counts = mask.sum(axis=0).clip(min=1)
    std = np.sqrt((centered**2).sum(axis=0) / counts)
    std = np.where(std > 1e-6, std, 1.0)
    z = np.where(mask, centered / std.reshape(1, -1), 0.0)
    return z, mask


def corr_from_standardized(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.T.astype(np.float64) @ mask.astype(np.float64)
    sim = z.T @ z
    sim = np.divide(sim, counts, out=np.zeros_like(sim), where=counts > 1)
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def residualize(x: np.ndarray, season_period: int) -> np.ndarray:
    mask = np.isfinite(x)
    global_mean = np.nanmean(np.where(mask, x, np.nan), axis=1, keepdims=True)
    global_mean = np.where(np.isfinite(global_mean), global_mean, 0.0)
    residual = np.where(mask, x - global_mean, np.nan)

    if season_period > 1:
        profile = np.zeros((season_period, x.shape[1]), dtype=np.float64)
        for slot in range(season_period):
            block = residual[slot::season_period]
            prof = np.nanmean(block, axis=0)
            profile[slot] = np.where(np.isfinite(prof), prof, 0.0)
        slots = np.arange(x.shape[0]) % season_period
        residual = residual - profile[slots]

    return residual


def lag_corr_similarity(z: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    n = z.shape[1]
    best = np.full((n, n), -1.0, dtype=np.float64)
    max_lag = max(0, int(max_lag))
    mask_f = mask.astype(np.float64)

    for lag in range(max_lag + 1):
        if lag == 0:
            left = z
            right = z
            left_mask = mask_f
            right_mask = mask_f
        else:
            left = z[:-lag]
            right = z[lag:]
            left_mask = mask_f[:-lag]
            right_mask = mask_f[lag:]

        counts = left_mask.T @ right_mask
        corr = left.T @ right
        corr = np.divide(corr, counts, out=np.zeros_like(corr), where=counts > 1)
        corr = np.clip(corr, -1.0, 1.0)
        best = np.maximum(best, corr)
        best = np.maximum(best, corr.T)

    best = np.where(np.isfinite(best), best, 0.0)
    np.fill_diagonal(best, 1.0)
    return best


def greedy_similarity_assignment(sim: np.ndarray, coarse_factor: int) -> np.ndarray:
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got {sim.shape}")

    n_nodes = sim.shape[0]
    coarse_factor = max(1, int(coarse_factor))
    positive = np.where(np.isfinite(sim) & (sim > 0), sim, 0.0)
    center_score = positive.sum(axis=1)
    center_order = np.lexsort((np.arange(n_nodes), -center_score))

    unassigned = np.ones(n_nodes, dtype=bool)
    assignment = np.full(n_nodes, -1, dtype=np.int64)
    cluster_id = 0

    for center in center_order:
        if not unassigned[center]:
            continue
        candidates = np.flatnonzero(unassigned)
        scores = sim[center, candidates]
        score_order = np.nan_to_num(scores, nan=-np.inf)
        order = np.lexsort((candidates, -score_order))
        members = candidates[order[:coarse_factor]]
        assignment[members] = cluster_id
        unassigned[members] = False
        cluster_id += 1

    if (assignment < 0).any():
        raise RuntimeError("Some nodes were not assigned to coarse clusters.")
    return assignment


def random_assignment(n_nodes: int, coarse_factor: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_nodes)
    assignment = np.empty(n_nodes, dtype=np.int64)
    for idx, node in enumerate(order):
        assignment[node] = idx // max(1, coarse_factor)
    return assignment


def group_diagnostics(method: str, assignment: np.ndarray, sim: np.ndarray) -> dict[str, object]:
    counts = np.bincount(assignment)
    within: list[float] = []
    for group in range(counts.size):
        nodes = np.flatnonzero(assignment == group)
        if nodes.size < 2:
            continue
        block = sim[np.ix_(nodes, nodes)]
        tri = block[np.triu_indices(nodes.size, k=1)]
        if tri.size:
            within.append(float(np.nanmean(tri)))
    return {
        "method": method,
        "nodes": int(assignment.size),
        "coarse_nodes": int(counts.size),
        "min_size": int(counts.min()),
        "max_size": int(counts.max()),
        "mean_size": float(counts.mean()),
        "mean_within_similarity": float(np.mean(within)) if within else "",
    }


def build(args: argparse.Namespace) -> None:
    x = load_h5_matrix(args.data_path)
    if args.treat_zero_as_missing:
        x = np.where(x == 0, np.nan, x)
    x_train = training_slice(x, args.train_ratio)

    raw_z, raw_mask = standardize_observed(x_train)
    raw_sim = corr_from_standardized(raw_z, raw_mask)

    residual = residualize(x_train, args.season_period)
    residual_z, residual_mask = standardize_observed(residual)
    residual_sim = corr_from_standardized(residual_z, residual_mask)
    lag_sim = lag_corr_similarity(residual_z, residual_mask, args.max_lag)

    similarities = {
        "raw_corr": raw_sim,
        "residual_corr": residual_sim,
        "lag_corr": lag_sim,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, object]] = []

    for method in args.methods:
        if method == "random":
            assignment = random_assignment(x_train.shape[1], args.coarse_factor, args.random_seed)
            sim_for_diag = residual_sim
        else:
            sim_for_diag = similarities[method]
            assignment = greedy_similarity_assignment(sim_for_diag, args.coarse_factor)
            np.save(args.out_dir / f"la_{method}_similarity.npy", sim_for_diag)

        out_path = args.out_dir / f"la_{method}_cf{args.coarse_factor}.npy"
        np.save(out_path, assignment)
        row = group_diagnostics(method, assignment, sim_for_diag)
        row["path"] = str(out_path)
        diagnostics.append(row)
        print(
            f"{method}: nodes={row['nodes']} coarse_nodes={row['coarse_nodes']} "
            f"min_size={row['min_size']} max_size={row['max_size']} path={out_path}"
        )

    diag_path = args.out_dir / f"la_functional_cf{args.coarse_factor}_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method",
            "nodes",
            "coarse_nodes",
            "min_size",
            "max_size",
            "mean_size",
            "mean_within_similarity",
            "path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"diagnostics: {diag_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["raw_corr", "residual_corr", "lag_corr", "random"],
        default=["raw_corr", "residual_corr", "lag_corr", "random"],
    )
    parser.add_argument("--coarse-factor", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--season-period", type=int, default=288)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260818)
    parser.add_argument("--treat-zero-as-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

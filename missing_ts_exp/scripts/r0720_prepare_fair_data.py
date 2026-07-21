#!/usr/bin/env python3
"""Prepare canonical data/masks for the 0720 strictly fair baseline experiment.

The goal is to make BiTGraph and HD-TTS-AMP consume exactly the same data matrix
and exactly the same missing positions.

Outputs:
  missing_ts_exp/results/0720_fair_b512/fair_data/
    mask_observed_<dataset>_<missing>_rXX_seed2024.npy
    manifest.csv

Mask convention:
  1 = observed / available
  0 = missing
"""

from __future__ import annotations

import csv
import hashlib
import pathlib

import numpy as np
import pandas as pd


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0720_fair_b512"
FAIR_DIR = RESULTS / "fair_data"
SEED = 2024
BLOCK_LEN = 12
RATES = (0.30, 0.50, 0.70)

DATASETS = {
    "Metr": ROOT / "dataset" / "metr_la" / "metr_la.h5",
    "PEMS": ROOT / "dataset" / "pems_bay" / "pems_bay.h5",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def point_mask(shape: tuple[int, int, int], rate: float, rng: np.random.RandomState) -> np.ndarray:
    total = int(np.prod(shape))
    missing = int(round(total * rate))
    flat = np.ones(total, dtype=np.uint8)
    flat[:missing] = 0
    rng.shuffle(flat)
    return flat.reshape(shape)


def temporal_block_mask(shape: tuple[int, int, int], rate: float,
                        rng: np.random.RandomState,
                        block_len: int = BLOCK_LEN) -> np.ndarray:
    # Same high-level pattern as BiTGraph temporal_block: a time block masks all
    # nodes. We generate it once and feed the exact result to both models.
    t_steps, n_nodes, n_channels = shape
    if n_channels != 1:
        raise ValueError(f"Expected single-channel data, got shape={shape}")

    target_missing = int(round(rate * t_steps * n_nodes))
    mask_2d = np.ones((t_steps, n_nodes), dtype=np.uint8)
    missing_count = 0
    max_iters = t_steps * n_nodes * 10
    iters = 0
    while missing_count < target_missing and iters < max_iters:
        start = rng.randint(0, max(1, t_steps - block_len + 1))
        end = min(start + block_len, t_steps)
        newly_zeroed = int(mask_2d[start:end, :].sum())
        mask_2d[start:end, :] = 0
        missing_count += newly_zeroed
        iters += 1
    return mask_2d[:, :, None]


def main() -> int:
    FAIR_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for dataset, data_path in DATASETS.items():
        df = pd.read_hdf(data_path)
        data_shape = (df.shape[0], df.shape[1], 1)
        data_sha = sha256_file(data_path)

        for missing_type in ("point", "block"):
            for rate in RATES:
                rate_tag = f"{int(rate * 100):02d}"
                # Different deterministic stream per dataset/type/rate while
                # keeping one top-level seed recorded in the file name.
                stream_seed = (
                    SEED
                    + {"Metr": 100_000, "PEMS": 200_000}[dataset]
                    + {"point": 10_000, "block": 20_000}[missing_type]
                    + int(rate * 100)
                )
                rng = np.random.RandomState(stream_seed)
                if missing_type == "point":
                    mask = point_mask(data_shape, rate, rng)
                    bitgraph_missing_type = "random_point"
                else:
                    mask = temporal_block_mask(data_shape, rate, rng)
                    bitgraph_missing_type = "temporal_block"

                mask_path = FAIR_DIR / (
                    f"mask_observed_{dataset}_{missing_type}_r{rate_tag}_seed{SEED}.npy"
                )
                np.save(mask_path, mask)
                actual_missing = 1.0 - float(mask.mean())

                rows.append({
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "data_sha256": data_sha,
                    "n_timesteps": str(data_shape[0]),
                    "n_nodes": str(data_shape[1]),
                    "n_channels": str(data_shape[2]),
                    "missing_type": missing_type,
                    "bitgraph_missing_type": bitgraph_missing_type,
                    "target_missing_rate": f"{rate:.2f}",
                    "actual_missing_rate": f"{actual_missing:.6f}",
                    "block_len": str(BLOCK_LEN if missing_type == "block" else ""),
                    "seed": str(SEED),
                    "stream_seed": str(stream_seed),
                    "mask_path": str(mask_path),
                    "mask_sha256": sha256_file(mask_path),
                    "mask_convention": "1=observed,0=missing",
                })

    manifest = FAIR_DIR / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {manifest} rows={len(rows)}")
    for row in rows:
        print(
            row["dataset"],
            row["missing_type"],
            row["target_missing_rate"],
            "actual=" + row["actual_missing_rate"],
            pathlib.Path(row["mask_path"]).name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

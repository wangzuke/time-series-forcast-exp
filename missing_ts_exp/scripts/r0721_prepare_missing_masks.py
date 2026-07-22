#!/usr/bin/env python3
"""Prepare the 0721 missing-mask bundle under dataset/.

This script follows the 0720 fair-mask convention:

  mask_observed_<Dataset>_<missing_type>_rXX_seed2024.npy

Mask convention:
  1 = observed / available
  0 = missing

The bundle is intentionally stored under dataset/ so that two machines can
sync the same canonical missing-data assets before starting model training.

0721 bundle:
  - Reuse 0720 masks for point 50/70 and block_t 50/70.
  - Generate new masks for block_t 90 and block_st 50/70/90.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import shutil
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
OUT_DIR = ROOT / "dataset" / "0721_missing_masks"
SRC_0720 = ROOT / "missing_ts_exp" / "results" / "0720_fair_b512" / "fair_data"
GRAPH_DIR = ROOT / "dataset" / "_archives" / "dcrnn_sensor_graph"

SEED = 2024
BLOCK_LEN = 12
BLOCK_ST_SPATIAL_FRAC = 0.10
RATES_POINT = (0.50, 0.70)
RATES_BLOCK_T = (0.50, 0.70, 0.90)
RATES_BLOCK_ST = (0.50, 0.70, 0.90)

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


def rate_tag(rate: float) -> str:
    return f"{int(round(rate * 100)):02d}"


def mask_name(dataset: str, missing_type: str, rate: float) -> str:
    return f"mask_observed_{dataset}_{missing_type}_r{rate_tag(rate)}_seed{SEED}.npy"


def source_0720_mask_name(dataset: str, missing_type: str, rate: float) -> str:
    # In 0720, temporal block masks were named simply "block".
    src_type = "block" if missing_type == "block_t" else missing_type
    return f"mask_observed_{dataset}_{src_type}_r{rate_tag(rate)}_seed{SEED}.npy"


def copy_0720_mask(dataset: str, missing_type: str, rate: float) -> pathlib.Path:
    src = SRC_0720 / source_0720_mask_name(dataset, missing_type, rate)
    dst = OUT_DIR / mask_name(dataset, missing_type, rate)
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)
    return dst


def temporal_block_mask(shape: tuple[int, int, int], rate: float,
                        rng: np.random.RandomState,
                        block_len: int = BLOCK_LEN) -> np.ndarray:
    """0720-compatible temporal-block mask.

    One event masks all nodes for a continuous time block. This matches the
    0720 "block" implementation and is used here as Block-T for continuity.
    """
    t_steps, n_nodes, n_channels = shape
    if n_channels != 1:
        raise ValueError(f"Expected one channel, got shape={shape}")

    target_missing = int(round(rate * t_steps * n_nodes))
    mask_2d = np.ones((t_steps, n_nodes), dtype=np.uint8)
    missing_count = 0
    max_iters = t_steps * n_nodes * 20
    iters = 0
    while missing_count < target_missing and iters < max_iters:
        start = rng.randint(0, max(1, t_steps - block_len + 1))
        end = min(start + block_len, t_steps)
        newly_zeroed = int(mask_2d[start:end, :].sum())
        mask_2d[start:end, :] = 0
        missing_count += newly_zeroed
        iters += 1
    if missing_count < target_missing:
        raise RuntimeError(
            f"Failed to reach target missing={target_missing}, got={missing_count}"
        )
    return mask_2d[:, :, None]


def load_distance_neighbors(dataset: str, columns: Iterable[object],
                            spatial_size: int) -> list[np.ndarray]:
    """Return nearest-neighbor node indices for each node.

    For METR-LA and PEMS-BAY we use the DCRNN distance files. If a distance
    entry is missing, the fallback still includes the seed node itself.
    """
    col_ids = [str(c) for c in columns]
    id_to_idx = {sid: i for i, sid in enumerate(col_ids)}
    dist_file = {
        "Metr": GRAPH_DIR / "distances_la_2012.csv",
        "PEMS": GRAPH_DIR / "distances_bay_2017.csv",
    }[dataset]

    distances: list[list[tuple[float, int]]] = [[] for _ in col_ids]
    with dist_file.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            src = str(row[0])
            dst = str(row[1])
            if src not in id_to_idx or dst not in id_to_idx:
                continue
            i = id_to_idx[src]
            j = id_to_idx[dst]
            try:
                cost = float(row[2])
            except ValueError:
                continue
            distances[i].append((cost, j))

    neighbors: list[np.ndarray] = []
    all_indices = np.arange(len(col_ids), dtype=np.int64)
    for i, entries in enumerate(distances):
        if entries:
            ordered = [j for _, j in sorted(entries, key=lambda x: x[0])]
            if i not in ordered:
                ordered.insert(0, i)
            neighbors.append(np.array(ordered[:spatial_size], dtype=np.int64))
        else:
            # Deterministic local fallback in column order.
            start = max(0, min(i - spatial_size // 2, len(col_ids) - spatial_size))
            fallback = all_indices[start:start + spatial_size]
            if i not in fallback:
                fallback = np.concatenate([[i], fallback[:-1]])
            neighbors.append(fallback.astype(np.int64))
    return neighbors


def spatio_temporal_block_mask(shape: tuple[int, int, int], rate: float,
                               rng: np.random.RandomState,
                               neighbors: list[np.ndarray],
                               block_len: int = BLOCK_LEN) -> np.ndarray:
    """Generate a Block-ST mask using temporal blocks plus spatial neighborhoods."""
    t_steps, n_nodes, n_channels = shape
    if n_channels != 1:
        raise ValueError(f"Expected one channel, got shape={shape}")

    target_missing = int(round(rate * t_steps * n_nodes))
    mask_2d = np.ones((t_steps, n_nodes), dtype=np.uint8)
    missing_count = 0
    max_iters = t_steps * n_nodes * 50
    iters = 0
    while missing_count < target_missing and iters < max_iters:
        start = rng.randint(0, max(1, t_steps - block_len + 1))
        end = min(start + block_len, t_steps)
        center = rng.randint(0, n_nodes)
        spatial_nodes = neighbors[center]
        newly_zeroed = int(mask_2d[start:end, spatial_nodes].sum())
        mask_2d[start:end, spatial_nodes] = 0
        missing_count += newly_zeroed
        iters += 1
    if missing_count < target_missing:
        raise RuntimeError(
            f"Failed to reach target missing={target_missing}, got={missing_count}"
        )
    return mask_2d[:, :, None]


def write_split_metadata(dataset: str, n_timesteps: int, horizon: int) -> pathlib.Path:
    window = 24
    stride = 1
    total_windows = n_timesteps - window - horizon + 1
    idx = np.arange(total_windows)
    test_len = int(len(idx) * 0.2)
    val_len = int(len(idx) * 0.1)
    test_start = len(idx) - test_len
    val_start = test_start - val_len
    meta = {
        "dataset": dataset,
        "n_timesteps": n_timesteps,
        "window": window,
        "horizon": horizon,
        "stride": stride,
        "total_windows": int(total_windows),
        "train_start": 0,
        "train_len": int(val_start),
        "val_start": int(val_start),
        "val_len": int(val_len),
        "test_start": int(test_start),
        "test_len": int(test_len),
        "split_policy": "slide windows first, then sequential 70/10/20 split",
    }
    out = OUT_DIR / f"split_{dataset}_h{horizon}.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out


def add_row(rows: list[dict[str, str]], *, dataset: str, data_path: pathlib.Path,
            data_sha: str, data_shape: tuple[int, int, int],
            missing_type: str, rate: float, mask_path: pathlib.Path,
            source: str, generator: str, stream_seed: int | str,
            block_len: int | str = "", spatial_size: int | str = "",
            spatial_frac: float | str = "", spatial_graph: str = "") -> None:
    mask = np.load(mask_path)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.shape != data_shape:
        raise ValueError(f"{mask_path} shape {mask.shape} != {data_shape}")
    actual_missing = 1.0 - float(mask.mean())
    rows.append({
        "dataset": dataset,
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "n_timesteps": str(data_shape[0]),
        "n_nodes": str(data_shape[1]),
        "n_channels": str(data_shape[2]),
        "missing_type": missing_type,
        "target_missing_rate": f"{rate:.2f}",
        "actual_missing_rate": f"{actual_missing:.6f}",
        "block_len": str(block_len),
        "spatial_size": str(spatial_size),
        "spatial_frac": str(spatial_frac),
        "spatial_graph": spatial_graph,
        "seed": str(SEED),
        "stream_seed": str(stream_seed),
        "source": source,
        "generator": generator,
        "mask_path": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "mask_convention": "1=observed,0=missing",
    })


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for dataset, data_path in DATASETS.items():
        df = pd.read_hdf(data_path)
        data_shape = (df.shape[0], df.shape[1], 1)
        data_sha = sha256_file(data_path)
        spatial_size = max(1, int(math.ceil(df.shape[1] * BLOCK_ST_SPATIAL_FRAC)))
        neighbors = load_distance_neighbors(dataset, df.columns, spatial_size)

        for horizon in (12, 24):
            write_split_metadata(dataset, df.shape[0], horizon)

        # Reuse 0720 point masks.
        for rate in RATES_POINT:
            dst = copy_0720_mask(dataset, "point", rate)
            add_row(
                rows, dataset=dataset, data_path=data_path, data_sha=data_sha,
                data_shape=data_shape, missing_type="point", rate=rate,
                mask_path=dst, source="0720_reused",
                generator="copy_0720_point_mask", stream_seed="0720",
            )

        # Reuse 0720 temporal block masks for 50/70, generate 90.
        for rate in RATES_BLOCK_T:
            if rate in (0.50, 0.70):
                dst = copy_0720_mask(dataset, "block_t", rate)
                source = "0720_reused"
                generator = "copy_0720_block_mask_as_block_t"
                stream_seed: int | str = "0720"
            else:
                stream_seed = (
                    SEED
                    + {"Metr": 100_000, "PEMS": 200_000}[dataset]
                    + 30_000
                    + int(rate * 100)
                )
                rng = np.random.RandomState(stream_seed)
                mask = temporal_block_mask(data_shape, rate, rng)
                dst = OUT_DIR / mask_name(dataset, "block_t", rate)
                np.save(dst, mask)
                source = "0721_new"
                generator = "temporal_block_mask_0720_compatible"
            add_row(
                rows, dataset=dataset, data_path=data_path, data_sha=data_sha,
                data_shape=data_shape, missing_type="block_t", rate=rate,
                mask_path=dst, source=source, generator=generator,
                stream_seed=stream_seed, block_len=BLOCK_LEN,
            )

        # Generate new Block-ST masks.
        for rate in RATES_BLOCK_ST:
            stream_seed = (
                SEED
                + {"Metr": 100_000, "PEMS": 200_000}[dataset]
                + 40_000
                + int(rate * 100)
            )
            rng = np.random.RandomState(stream_seed)
            mask = spatio_temporal_block_mask(data_shape, rate, rng, neighbors)
            dst = OUT_DIR / mask_name(dataset, "block_st", rate)
            np.save(dst, mask)
            add_row(
                rows, dataset=dataset, data_path=data_path, data_sha=data_sha,
                data_shape=data_shape, missing_type="block_st", rate=rate,
                mask_path=dst, source="0721_new",
                generator="spatio_temporal_block_mask_distance_neighbors",
                stream_seed=stream_seed, block_len=BLOCK_LEN,
                spatial_size=spatial_size,
                spatial_frac=BLOCK_ST_SPATIAL_FRAC,
                spatial_graph=str(GRAPH_DIR),
            )

    manifest = OUT_DIR / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join([
            "# 0721 Missing Mask Bundle",
            "",
            "Mask convention: `1=observed, 0=missing`.",
            "",
            "This bundle reuses 0720 point 50/70 and block_t 50/70 masks,",
            "and adds 0721 block_t 90 plus block_st 50/70/90 masks.",
            "",
            "Use `manifest.csv` as the source of truth for data paths,",
            "actual missing rates, mask paths, and mask sha256 checksums.",
            "",
            "The original data are not modified. Construct missing inputs at runtime",
            "from canonical HDF5 data plus the selected observed-mask `.npy` file.",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"wrote {manifest} rows={len(rows)}")
    for row in rows:
        print(
            row["dataset"],
            row["missing_type"],
            row["target_missing_rate"],
            "actual=" + row["actual_missing_rate"],
            "source=" + row["source"],
            pathlib.Path(row["mask_path"]).name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

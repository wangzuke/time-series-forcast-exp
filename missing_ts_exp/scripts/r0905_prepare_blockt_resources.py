"""Prepare fixed data/mask resources for the R0905 Block-T experiment.

Run with:
    /media/data1/wangzk/miniconda3/envs/hd-tts/bin/python \
        -m missing_ts_exp.scripts.r0905_prepare_blockt_resources
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from missing_ts_exp.src.data.hdtts_missing import (
    HDTTSMaskParams,
    actual_missing_rate,
    calibrate_block_t_p_fault,
    make_observed_mask,
    sha256_file,
)
from missing_ts_exp.src.training.run_hdtts_reliable_spatial import load_spatial_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "resources" / "r0905_blockt"
DEFAULT_SEEDS = (2024, 2025, 2026)
DEFAULT_RATES = (0.30, 0.50, 0.70)


def save_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.float32))
    return sha256_file(path)


def add_manifest_row(rows: list[dict[str, object]], **kwargs) -> None:
    rows.append(dict(kwargs))


def prepare_metr_la(out_dir: Path, seeds: tuple[int, ...], rates: tuple[float, ...], p_noise: float) -> list[dict[str, object]]:
    values, valid_mask, adj, node_ids, data_path = load_spatial_dataset("METR-LA")
    rows: list[dict[str, object]] = []
    n_steps, n_nodes = values.shape
    for seed in seeds:
        for rate in rates:
            params = HDTTSMaskParams("point", 0.0, rate, 1, 1, seed, rate)
            observed = make_observed_mask(n_steps, n_nodes, params=params)
            rel_path = Path("masks") / "METR-LA" / f"point_r{int(rate * 100):02d}_seed{seed}_observed.npy"
            abs_path = out_dir / rel_path
            digest = save_mask(observed, abs_path)
            add_manifest_row(
                rows,
                dataset="METR-LA",
                mode="point",
                target_missing_rate=rate,
                actual_artificial_missing_rate=actual_missing_rate(observed),
                effective_observed_rate=float((observed * valid_mask).mean()),
                seed=seed,
                p_fault=params.p_fault,
                p_noise=params.p_noise,
                min_seq=params.min_seq,
                max_seq=params.max_seq,
                mask_path=str(abs_path),
                mask_path_relative=str(rel_path),
                mask_sha256=digest,
                data_path=data_path,
                n_steps=n_steps,
                n_nodes=n_nodes,
                n_edges=int((adj > 0).sum()),
                valid_value_rate=float(valid_mask.mean()),
            )

            p_fault, observed, actual = calibrate_block_t_p_fault(
                n_steps=n_steps,
                n_nodes=n_nodes,
                target_missing_rate=rate,
                seed=seed,
                p_noise=p_noise,
                min_seq=12,
                max_seq=48,
            )
            rel_path = Path("masks") / "METR-LA" / f"block_t_r{int(rate * 100):02d}_seed{seed}_observed.npy"
            abs_path = out_dir / rel_path
            digest = save_mask(observed, abs_path)
            add_manifest_row(
                rows,
                dataset="METR-LA",
                mode="block_t",
                target_missing_rate=rate,
                actual_artificial_missing_rate=actual,
                effective_observed_rate=float((observed * valid_mask).mean()),
                seed=seed,
                p_fault=p_fault,
                p_noise=p_noise,
                min_seq=12,
                max_seq=48,
                mask_path=str(abs_path),
                mask_path_relative=str(rel_path),
                mask_sha256=digest,
                data_path=data_path,
                n_steps=n_steps,
                n_nodes=n_nodes,
                n_edges=int((adj > 0).sum()),
                valid_value_rate=float(valid_mask.mean()),
            )
    return rows


def prepare_aqi(out_dir: Path, seeds: tuple[int, ...], point_rates: tuple[float, ...]) -> list[dict[str, object]]:
    values, valid_mask, adj, node_ids, data_path = load_spatial_dataset("AQI36")
    rows: list[dict[str, object]] = []
    n_steps, n_nodes = values.shape
    for seed in seeds:
        observed = np.ones_like(valid_mask, dtype=np.float32)
        rel_path = Path("masks") / "AQI36" / f"original_seed{seed}_observed.npy"
        abs_path = out_dir / rel_path
        digest = save_mask(observed, abs_path)
        add_manifest_row(
            rows,
            dataset="AQI36",
            mode="original",
            target_missing_rate=0.0,
            actual_artificial_missing_rate=0.0,
            effective_observed_rate=float((observed * valid_mask).mean()),
            seed=seed,
            p_fault=0.0,
            p_noise=0.0,
            min_seq=1,
            max_seq=1,
            mask_path=str(abs_path),
            mask_path_relative=str(rel_path),
            mask_sha256=digest,
            data_path=data_path,
            n_steps=n_steps,
            n_nodes=n_nodes,
            n_edges=int((adj > 0).sum()),
            valid_value_rate=float(valid_mask.mean()),
        )

        for rate in point_rates:
            params = HDTTSMaskParams("point", 0.0, rate, 1, 1, seed, rate)
            observed = make_observed_mask(n_steps, n_nodes, params=params)
            rel_path = Path("masks") / "AQI36" / f"point_r{int(rate * 100):02d}_seed{seed}_observed.npy"
            abs_path = out_dir / rel_path
            digest = save_mask(observed, abs_path)
            add_manifest_row(
                rows,
                dataset="AQI36",
                mode="point",
                target_missing_rate=rate,
                actual_artificial_missing_rate=actual_missing_rate(observed),
                effective_observed_rate=float((observed * valid_mask).mean()),
                seed=seed,
                p_fault=params.p_fault,
                p_noise=params.p_noise,
                min_seq=params.min_seq,
                max_seq=params.max_seq,
                mask_path=str(abs_path),
                mask_path_relative=str(rel_path),
                mask_sha256=digest,
                data_path=data_path,
                n_steps=n_steps,
                n_nodes=n_nodes,
                n_edges=int((adj > 0).sum()),
                valid_value_rate=float(valid_mask.mean()),
            )
    return rows


def write_manifest(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--rates", default=",".join(str(r) for r in DEFAULT_RATES))
    parser.add_argument("--aqi_point_rates", default="0.25,0.50")
    parser.add_argument("--block_t_p_noise", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    rates = tuple(float(r) for r in args.rates.split(",") if r)
    aqi_point_rates = tuple(float(r) for r in args.aqi_point_rates.split(",") if r)
    rows = []
    rows.extend(prepare_metr_la(out_dir, seeds, rates, args.block_t_p_noise))
    rows.extend(prepare_aqi(out_dir, seeds, aqi_point_rates))
    write_manifest(rows, out_dir)
    print(f"prepared {len(rows)} resources under {out_dir}")
    print(f"manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()

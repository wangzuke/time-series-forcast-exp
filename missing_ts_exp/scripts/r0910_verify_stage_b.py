#!/usr/bin/env python3
"""Verify the complete METR-LA Block-T mechanism-ablation matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OLD = ROOT / "missing_ts_exp/results/r0908_official_baselines/controlled"
NEW = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
OUT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_verification_summary.json"
SEEDS = (2024, 2025, 2026, 2027, 2028)
RATES = (50, 70)
VARIANTS = (
    "official_base",
    "full",
    "phy_only",
    "obs_only",
    "full_no_overlap_reliability",
    "fine",
    "coarse_no_gate",
    "full_no_readout_bias",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite: {value!r}")
    return number


def source_and_tag(variant: str, rate: int, seed: int) -> tuple[Path, str]:
    suffix = f"METR-LA_block_t_r{rate}_seed{seed}"
    if variant == "official_base":
        return (OLD if seed <= 2026 else NEW), f"official_base_{suffix}"
    if variant == "full":
        return (OLD if seed <= 2026 else NEW), f"{'mrmsgf' if seed <= 2026 else 'full'}_{suffix}"
    return NEW, f"{variant}_{suffix}"


def read_row(variant: str, rate: int, seed: int) -> dict:
    directory, tag = source_and_tag(variant, rate, seed)
    path = directory / f"{tag}.json"
    require(path.is_file(), f"missing result JSON: {path}")
    checkpoint_dir = directory / tag
    require(any(checkpoint_dir.glob("*.ckpt")), f"missing best checkpoint under {checkpoint_dir}")
    row = json.loads(path.read_text())
    require(row.get("tag") == tag, f"tag mismatch in {path}")
    expected_model = "official_base" if variant == "official_base" else "mrmsgf"
    require(row.get("model") == expected_model, f"model mismatch in {path}")
    require(row.get("seed") == seed, f"seed mismatch in {path}")
    require(row.get("epochs") == 100 and row.get("train_batches") == 300,
            f"training budget mismatch in {path}")
    require(row.get("official_hdtts_commit") == "46a8717db3802e1684e991ba0a7c0bfe43d22535",
            f"official source commit mismatch in {path}")
    for section, keys in (("masked_test", ("test_mae", "test_mse")),
                          ("unmasked_test", ("test_mae_unmasked", "test_mse_unmasked"))):
        require(section in row, f"missing {section} in {path}")
        for key in keys:
            finite(row[section][key], f"{path.name}:{section}.{key}")
    row["_path"] = str(path)
    return row


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    groups: dict[str, dict] = {}
    all_rows: list[dict] = []
    data_hashes: set[str] = set()
    for rate in RATES:
        by_seed_masks: dict[int, str] = {}
        for variant in VARIANTS:
            rows = [read_row(variant, rate, seed) for seed in SEEDS]
            for row in rows:
                data_hashes.add(row["data_sha256"])
                previous = by_seed_masks.setdefault(row["seed"], row["mask_sha256"])
                require(previous == row["mask_sha256"],
                        f"mask is not paired for r{rate}, seed {row['seed']}: {row['_path']}")
            maes = [row["unmasked_test"]["test_mae_unmasked"] for row in rows]
            rmses = [math.sqrt(row["unmasked_test"]["test_mse_unmasked"]) for row in rows]
            groups[f"r{rate}_{variant}"] = {
                "n": len(rows),
                "mae": stats(maes),
                "rmse": stats(rmses),
                "mask_sha256_by_seed": {str(row["seed"]): row["mask_sha256"] for row in rows},
                "result_paths": [row["_path"] for row in rows],
            }
            all_rows.extend(rows)

        full = [read_row("full", rate, seed) for seed in SEEDS]
        for variant in VARIANTS:
            if variant == "full":
                continue
            compared = [read_row(variant, rate, seed) for seed in SEEDS]
            deltas = [other["unmasked_test"]["test_mae_unmasked"] - base["unmasked_test"]["test_mae_unmasked"]
                      for base, other in zip(full, compared)]
            groups[f"r{rate}_{variant}"]["paired_mae_minus_full"] = stats(deltas)

    require(len(data_hashes) == 1, f"multiple METR-LA data hashes observed: {sorted(data_hashes)}")
    summary = {
        "protocol": "METR-LA, 24-to-12, Block-T, unmasked future MAE/RMSE, 100 epochs, 300 train batches",
        "seeds": list(SEEDS),
        "rates": list(RATES),
        "variants": list(VARIANTS),
        "n_result_rows": len(all_rows),
        "data_sha256": next(iter(data_hashes)),
        "groups": groups,
        "status": "passed",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

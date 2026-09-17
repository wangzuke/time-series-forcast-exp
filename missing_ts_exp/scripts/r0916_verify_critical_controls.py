#!/usr/bin/env python3
"""Verify and summarize the paired critical controls for Block-T 50%."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0916_critical_controls"
OUT = RESULT / "verification_summary.json"
SEEDS = (2024, 2025, 2026, 2027, 2028)
CONTROLS = {
    "full_fallback": ("full", "observed"),
    "feature_only": ("feature_only", "observed"),
    "full_q_shuffled": ("full", "shuffled"),
}


def read_row(label: str, seed: int) -> dict:
    path = RESULT / f"{label}_METR-LA_block_t_r50_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text())
    variant, q_mode = CONTROLS[label]
    if row.get("mr_variant") != variant or row.get("q_mode") != q_mode:
        raise ValueError(f"configuration mismatch in {path}")
    if row.get("zero_observation_physical_fallback") is not True:
        raise ValueError(f"missing zero-observation fallback in {path}")
    if row.get("seed") != seed or row.get("epochs") != 100 or row.get("train_batches") != 300:
        raise ValueError(f"protocol mismatch in {path}")
    metric = float(row["unmasked_test"]["test_mae_unmasked"])
    if not math.isfinite(metric):
        raise ValueError(f"non-finite MAE in {path}")
    row["_path"] = str(path)
    return row


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def main() -> None:
    rows = {label: [read_row(label, seed) for seed in SEEDS] for label in CONTROLS}
    masks = {row["mask_sha256"] for group in rows.values() for row in group}
    if len(masks) != len(SEEDS):
        raise ValueError("each seed must reuse its identical mask across controls")
    full = np.asarray([row["unmasked_test"]["test_mae_unmasked"] for row in rows["full_fallback"]])
    groups: dict[str, dict] = {}
    for label, group in rows.items():
        values = np.asarray([row["unmasked_test"]["test_mae_unmasked"] for row in group])
        item = {"mae": summary(values), "result_paths": [row["_path"] for row in group]}
        if label != "full_fallback":
            delta = values - full
            item["paired_mae_minus_full"] = summary(delta)
            item["paired_t_test_two_sided"] = float(ttest_rel(values, full).pvalue)
        groups[label] = item
    result = {
        "status": "passed",
        "protocol": "METR-LA, Block-T 50%, 24-to-12, five paired mask/training seeds, unmasked future MAE",
        "seeds": list(SEEDS),
        "groups": groups,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

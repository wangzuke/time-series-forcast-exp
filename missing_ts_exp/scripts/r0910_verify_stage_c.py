#!/usr/bin/env python3
"""Verify the formal AQI36 natural-missingness and Point experiment matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_c_aqi36"
OUT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_c_verification_summary.json"
SEEDS = (2024, 2025, 2026, 2027, 2028)
CONDITIONS = (("original", 0), ("point", 25), ("point", 50))
MODELS = ("official_base", "mrmsgf")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def row(model: str, condition: str, rate: int, seed: int) -> dict:
    label = "original_r00" if condition == "original" else f"point_r{rate}"
    tag = f"{model}_AQI36_{label}_seed{seed}"
    path = RESULT / f"{tag}.json"
    require(path.is_file(), f"missing result: {path}")
    value = json.loads(path.read_text())
    require(value.get("model") == model and value.get("config", {}).get("seed") == seed,
            f"identity mismatch in {path}")
    config = value["config"]
    require(config.get("epochs") == 100 and config.get("seq_len") == 24 and config.get("pred_len") == 12,
            f"configuration mismatch in {path}")
    require(value.get("official_hdtts_commit") == "46a8717db3802e1684e991ba0a7c0bfe43d22535",
            f"HD-TTS source mismatch in {path}")
    require(Path(value.get("best_checkpoint", "")).is_file(), f"missing checkpoint in {path}")
    for key in ("mae", "rmse"):
        require(math.isfinite(float(value["test"][key])), f"non-finite {key} in {path}")
    value["_path"] = str(path)
    return value


def main() -> None:
    groups, hashes = {}, set()
    for condition, rate in CONDITIONS:
        key = "original" if condition == "original" else f"point_r{rate}"
        masks: dict[int, str] = {}
        for model in MODELS:
            rows = [row(model, condition, rate, seed) for seed in SEEDS]
            for value in rows:
                hashes.add(value["data_sha256"])
                previous = masks.setdefault(value["config"]["seed"], value["mask_sha256"])
                require(previous == value["mask_sha256"], f"unpaired mask {key}, seed {value['config']['seed']}")
            groups[f"{key}_{model}"] = {
                "n": len(rows), "mae": stats([value["test"]["mae"] for value in rows]),
                "rmse": stats([value["test"]["rmse"] for value in rows]),
                "mask_sha256_by_seed": {str(value["config"]["seed"]): value["mask_sha256"] for value in rows},
            }
        base, full = [row("official_base", condition, rate, seed) for seed in SEEDS], [row("mrmsgf", condition, rate, seed) for seed in SEEDS]
        groups[f"{key}_official_base"]["paired_mae_minus_mrmsgf"] = stats([
            value["test"]["mae"] - ref["test"]["mae"] for ref, value in zip(full, base)
        ])
    require(len(hashes) == 1, f"multiple data hashes: {sorted(hashes)}")
    result = {"status": "passed", "protocol": "AQI36, 24-to-12, natural validity mask plus fixed Point masks, five seeds, raw valid-future MAE/RMSE",
              "seeds": list(SEEDS), "conditions": ["original", "point_r25", "point_r50"], "n_result_rows": 30,
              "data_sha256": next(iter(hashes)), "groups": groups}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the five-seed METR-LA Point comparison matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a2_point"
OUT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a2_verification_summary.json"
SEEDS, RATES, MODELS = (2024, 2025, 2026, 2027, 2028), (30, 50), ("official_base", "mrmsgf", "bitgraph")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def row(model: str, rate: int, seed: int) -> dict:
    stem = "bitgraph_unified" if model == "bitgraph" else model
    tag = f"{stem}_METR-LA_point_r{rate}_seed{seed}"
    path = RESULT / f"{tag}.json"
    require(path.is_file(), f"missing result: {path}")
    checkpoint = RESULT / tag / ("best.pth" if model == "bitgraph" else "")
    require(checkpoint.is_file() if model == "bitgraph" else any((RESULT / tag).glob("*.ckpt")),
            f"missing checkpoint for {path}")
    value = json.loads(path.read_text())
    expected = "bitgraph_official_architecture" if model == "bitgraph" else model
    require(value.get("tag") == tag and value.get("model") == expected and value.get("seed") == seed,
            f"identity mismatch in {path}")
    if model == "bitgraph":
        require(value.get("protocol") == "unified_hdtts_block_t" and value.get("epochs") == 200,
                f"BiTGraph protocol mismatch in {path}")
    else:
        require(value.get("official_hdtts_commit") == "46a8717db3802e1684e991ba0a7c0bfe43d22535"
                and value.get("epochs") == 100 and value.get("train_batches") == 300,
                f"HD-TTS protocol mismatch in {path}")
    mae, mse = float(value["unmasked_test"]["test_mae_unmasked"]), float(value["unmasked_test"]["test_mse_unmasked"])
    require(math.isfinite(mae) and math.isfinite(mse), f"non-finite metric in {path}")
    value["_path"] = str(path)
    return value


def main() -> None:
    groups, data_hashes = {}, set()
    for rate in RATES:
        masks: dict[int, str] = {}
        for model in MODELS:
            rows = [row(model, rate, seed) for seed in SEEDS]
            for value in rows:
                data_hashes.add(value["data_sha256"])
                old = masks.setdefault(value["seed"], value["mask_sha256"])
                require(old == value["mask_sha256"], f"unpaired mask r{rate}, seed {value['seed']}")
            groups[f"r{rate}_{model}"] = {
                "n": len(rows),
                "mae": stats([value["unmasked_test"]["test_mae_unmasked"] for value in rows]),
                "rmse": stats([math.sqrt(value["unmasked_test"]["test_mse_unmasked"]) for value in rows]),
                "mask_sha256_by_seed": {str(value["seed"]): value["mask_sha256"] for value in rows},
            }
        full = [row("mrmsgf", rate, seed) for seed in SEEDS]
        for model in ("official_base", "bitgraph"):
            other = [row(model, rate, seed) for seed in SEEDS]
            groups[f"r{rate}_{model}"]["paired_mae_minus_mrmsgf"] = stats([
                value["unmasked_test"]["test_mae_unmasked"] - ref["unmasked_test"]["test_mae_unmasked"]
                for ref, value in zip(full, other)
            ])
    require(len(data_hashes) == 1, f"multiple data hashes: {sorted(data_hashes)}")
    result = {"status": "passed", "protocol": "METR-LA, 24-to-12, Point, five paired mask seeds, unmasked future MAE/RMSE",
              "seeds": list(SEEDS), "rates": list(RATES), "models": list(MODELS), "n_result_rows": 30,
              "data_sha256": next(iter(data_hashes)), "groups": groups}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

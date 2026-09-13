#!/usr/bin/env python3
"""Verify the complete five-seed METR-LA Block-T main-table matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OLD_CONTROLLED = ROOT / "missing_ts_exp/results/r0908_official_baselines/controlled"
OLD_BITGRAPH = ROOT / "missing_ts_exp/results/r0908_official_baselines/bitgraph_unified_blockt_v2"
STAGE_B = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
STAGE_A1 = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a1_main"
OUT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a1_verification_summary.json"
SEEDS = (2024, 2025, 2026, 2027, 2028)
RATES = (30, 50, 70)
MODELS = ("official_base", "mrmsgf", "bitgraph")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite: {value!r}")
    return number


def location(model: str, rate: int, seed: int) -> tuple[Path, str, str]:
    suffix = f"METR-LA_block_t_r{rate}_seed{seed}"
    if rate == 30:
        stem = "bitgraph_unified" if model == "bitgraph" else model
        return STAGE_A1, f"{stem}_{suffix}", "best.pth" if model == "bitgraph" else "*.ckpt"
    if model == "bitgraph":
        root = OLD_BITGRAPH if seed <= 2026 else STAGE_A1
        return root, f"bitgraph_unified_{suffix}", "best.pth"
    root = OLD_CONTROLLED if seed <= 2026 else STAGE_B
    stem = model if model == "official_base" or seed <= 2026 else "full"
    return root, f"{stem}_{suffix}", "*.ckpt"


def read_row(model: str, rate: int, seed: int) -> dict:
    root, tag, checkpoint = location(model, rate, seed)
    path = root / f"{tag}.json"
    require(path.is_file(), f"missing result JSON: {path}")
    run_dir = root / tag
    has_checkpoint = (run_dir / checkpoint).is_file() if checkpoint == "best.pth" else any(run_dir.glob(checkpoint))
    require(has_checkpoint, f"missing checkpoint for {path}")
    row = json.loads(path.read_text())
    require(row.get("tag") == tag and row.get("seed") == seed, f"identity mismatch in {path}")
    expected = "bitgraph_official_architecture" if model == "bitgraph" else model
    require(row.get("model") == expected, f"model mismatch in {path}")
    if model == "bitgraph":
        require(row.get("protocol") == "unified_hdtts_block_t", f"BiTGraph protocol mismatch in {path}")
        require(row.get("official_bitgraph_commit") == "4f1d05bcc20bb3f5084bd1facc995478f84a40ed",
                f"BiTGraph source mismatch in {path}")
        require(row.get("epochs") == 200, f"BiTGraph training budget mismatch in {path}")
    else:
        require(row.get("official_hdtts_commit") == "46a8717db3802e1684e991ba0a7c0bfe43d22535",
                f"HD-TTS source mismatch in {path}")
        require(row.get("epochs") == 100 and row.get("train_batches") == 300,
                f"controlled training budget mismatch in {path}")
    finite(row["unmasked_test"]["test_mae_unmasked"], f"{path}: MAE")
    finite(row["unmasked_test"]["test_mse_unmasked"], f"{path}: MSE")
    row["_path"] = str(path)
    return row


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def main() -> None:
    groups = {}
    hashes = set()
    for rate in RATES:
        per_seed_masks: dict[int, str] = {}
        for model in MODELS:
            rows = [read_row(model, rate, seed) for seed in SEEDS]
            for row in rows:
                hashes.add(row["data_sha256"])
                previous = per_seed_masks.setdefault(row["seed"], row["mask_sha256"])
                require(previous == row["mask_sha256"],
                        f"unpaired mask at r{rate}, seed {row['seed']}: {row['_path']}")
            maes = [row["unmasked_test"]["test_mae_unmasked"] for row in rows]
            rmses = [math.sqrt(row["unmasked_test"]["test_mse_unmasked"]) for row in rows]
            groups[f"r{rate}_{model}"] = {
                "n": len(rows), "mae": stats(maes), "rmse": stats(rmses),
                "mask_sha256_by_seed": {str(row["seed"]): row["mask_sha256"] for row in rows},
                "result_paths": [row["_path"] for row in rows],
            }
        full = [read_row("mrmsgf", rate, seed) for seed in SEEDS]
        for model in ("official_base", "bitgraph"):
            compared = [read_row(model, rate, seed) for seed in SEEDS]
            delta = [row["unmasked_test"]["test_mae_unmasked"] - ref["unmasked_test"]["test_mae_unmasked"]
                     for ref, row in zip(full, compared)]
            groups[f"r{rate}_{model}"]["paired_mae_minus_mrmsgf"] = stats(delta)
    require(len(hashes) == 1, f"multiple data hashes: {sorted(hashes)}")
    result = {
        "status": "passed",
        "protocol": "METR-LA, 24-to-12, Block-T, five paired mask seeds, unmasked future MAE/RMSE",
        "seeds": list(SEEDS), "rates": list(RATES), "models": list(MODELS),
        "n_result_rows": len(RATES) * len(MODELS) * len(SEEDS),
        "data_sha256": next(iter(hashes)), "groups": groups,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

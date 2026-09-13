#!/usr/bin/env python3
"""Verify and summarize paired horizon metrics for METR-LA Block-T."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_d_horizon"
OUT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_d_horizon_summary.json"
SEEDS, RATES, MODELS, HORIZONS = (2024, 2025, 2026, 2027, 2028), (50, 70), ("official_base", "mrmsgf"), (3, 6, 12)


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def read(model: str, rate: int, seed: int) -> dict:
    path = RESULT / f"{model}_METR-LA_block_t_r{rate}_seed{seed}_horizon.json"
    if not path.is_file():
        raise ValueError(f"missing horizon result: {path}")
    row = json.loads(path.read_text())
    if row.get("model") != model or len(row.get("per_horizon", [])) != 12:
        raise ValueError(f"invalid horizon result: {path}")
    if int(row.get("n_model_parameters", 0)) <= 0 or not math.isfinite(float(row.get("seconds_per_batch", float("nan")))):
        raise ValueError(f"missing efficiency record: {path}")
    for entry in row["per_horizon"]:
        if not math.isfinite(float(entry["mae"])) or not math.isfinite(float(entry["rmse"])):
            raise ValueError(f"non-finite metric: {path}")
    return row


def main() -> None:
    groups = {}
    for rate in RATES:
        for horizon in HORIZONS:
            rows = {model: [read(model, rate, seed) for seed in SEEDS] for model in MODELS}
            for model, values in rows.items():
                metric = [value["per_horizon"][horizon - 1] for value in values]
                groups[f"r{rate}_{model}_h{horizon}"] = {
                    "n": len(metric), "mae": stats([x["mae"] for x in metric]),
                    "rmse": stats([x["rmse"] for x in metric]),
                }
            diff = [a["per_horizon"][horizon - 1]["mae"] - b["per_horizon"][horizon - 1]["mae"]
                    for a, b in zip(rows["official_base"], rows["mrmsgf"])]
            groups[f"r{rate}_official_base_h{horizon}"]["paired_mae_minus_mrmsgf"] = stats(diff)
    efficiency = {}
    for model in MODELS:
        rows = [read(model, rate, seed) for rate in RATES for seed in SEEDS]
        params = {int(row["n_model_parameters"]) for row in rows}
        if len(params) != 1:
            raise ValueError(f"inconsistent parameter count for {model}: {params}")
        efficiency[model] = {"n_model_parameters": next(iter(params)),
                             "seconds_per_batch": stats([float(row["seconds_per_batch"]) for row in rows])}
    result = {"status": "passed", "rates": list(RATES), "seeds": list(SEEDS), "horizons": list(HORIZONS),
              "n_result_rows": len(RATES) * len(SEEDS) * len(MODELS), "groups": groups, "efficiency": efficiency}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

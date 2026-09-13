#!/usr/bin/env python3
"""Validate and summarize the r0908 official-baseline experiment matrix."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0908_official_baselines"
WORKER_LOGS = ROOT / "missing_ts_exp/logs/r0908_official_baselines/formal_workers"


def finite(value: object, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value}")
    return value


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def controlled_summary() -> dict:
    rows = []
    for path in sorted((RESULT / "controlled").glob("*.json")):
        row = json.loads(path.read_text())
        for section in ("masked_test", "unmasked_test"):
            for key, value in row[section].items():
                finite(value, f"{path.name}:{section}.{key}")
        rows.append(row)
    if len(rows) != 12:
        raise ValueError(f"Expected 12 controlled JSON results, found {len(rows)}")
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        rate = re.search(r"_r(\d+)_", row["tag"]).group(1)
        groups.setdefault((row["model"], rate), []).append(row)
    summary = {}
    for (model, rate), items in sorted(groups.items()):
        if len(items) != 3:
            raise ValueError(f"Expected 3 controlled repetitions for {model}, r{rate}; found {len(items)}")
        maes = [finite(item["unmasked_test"]["test_mae_unmasked"], item["tag"]) for item in items]
        rmses = [math.sqrt(finite(item["unmasked_test"]["test_mse_unmasked"], item["tag"])) for item in items]
        summary[f"{model}_r{rate}"] = {"mae": mean_std(maes), "rmse": mean_std(rmses),
                                          "mask_sha256": [item["mask_sha256"] for item in items]}
    return {"n_results": len(rows), "groups": summary}


def hdtts_summary() -> dict:
    commands = (ROOT / "missing_ts_exp/scripts/r0908_formal_cmds.txt").read_text().splitlines()
    rows = []
    for index, command in enumerate(commands):
        if "r0908_run_hdtts_official_single.sh" not in command:
            continue
        log = WORKER_LOGS / f"run_{index}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        masked = re.findall(r"test_mae\s+([0-9.eE+-]+)", text)
        unmasked = re.findall(r"test_mae_unmasked\s+([0-9.eE+-]+)", text)
        if not masked or not unmasked:
            raise ValueError(f"Missing official HD-TTS test metrics in {log}")
        model, seed = re.search(r"single\.sh (\S+) (\d+)", command).groups()
        rows.append(dict(model=model, seed=int(seed), mae=finite(unmasked[-1], str(log))))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 HD-TTS official runs, found {len(rows)}")
    groups = {}
    for model in sorted({row["model"] for row in rows}):
        values = [row["mae"] for row in rows if row["model"] == model]
        if len(values) != 3:
            raise ValueError(f"Expected 3 official HD-TTS repetitions for {model}")
        groups[model] = {"unmasked_mae": mean_std(values)}
    return {"n_runs": len(rows), "groups": groups}


def bitgraph_summary() -> dict:
    groups = {}
    for directory in sorted((RESULT / "official_bitgraph").glob("point_r*")):
        values = []
        for replicate in sorted(directory.glob("replicate*")):
            log = replicate / "test.log"
            text = log.read_text(errors="replace")
            match = re.findall(r"loss,RMSE,MAPE\s+([0-9.eE+-]+)\s+&\s+([0-9.eE+-]+)\s+&\s+([0-9.eE+-]+)", text)
            if not match or not (replicate / "best.pth").is_file():
                raise ValueError(f"Missing BiTGraph result/checkpoint in {replicate}")
            mae, rmse, _ = match[-1]
            values.append((finite(mae, str(log)), finite(rmse, str(log))))
        if len(values) != 3:
            raise ValueError(f"Expected 3 BiTGraph replicas in {directory}, found {len(values)}")
        groups[directory.name] = {"mae": mean_std([x[0] for x in values]),
                                  "rmse": mean_std([x[1] for x in values])}
    if len(groups) != 3:
        raise ValueError(f"Expected 3 BiTGraph rates, found {len(groups)}")
    return {"n_runs": 9, "groups": groups,
            "note": "Official code leaves NumPy/Torch RNG unseeded; rows are independent replicas, not controlled seeds."}


def main() -> None:
    report = dict(controlled=controlled_summary(),
                  official_hdtts=hdtts_summary(),
                  official_bitgraph=bitgraph_summary())
    path = RESULT / "formal_verification_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

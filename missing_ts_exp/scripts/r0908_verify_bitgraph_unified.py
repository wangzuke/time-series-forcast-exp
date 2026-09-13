#!/usr/bin/env python3
"""Validate and summarize BiTGraph under the shared Block-T protocol."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0908_official_baselines"


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def load_rows(directory: Path, expected_model: str) -> list[dict]:
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if len(rows) != 6:
        raise ValueError(f"Expected six result JSON files in {directory}, found {len(rows)}")
    for row in rows:
        if row["model"] != expected_model:
            raise ValueError(f"Unexpected model in {row['tag']}: {row['model']}")
        if not (directory / row["tag"] / "best.pth").is_file():
            raise ValueError(f"Missing best checkpoint for {row['tag']}")
        for value in row["unmasked_test"].values():
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite metric in {row['tag']}")
    return rows


def group_rows(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        rate = re.search(r"_r(\d+)_", row["tag"]).group(1)
        groups.setdefault(rate, []).append(row)
    report = {}
    for rate, items in sorted(groups.items()):
        if len(items) != 3:
            raise ValueError(f"Expected three seeds for r{rate}, found {len(items)}")
        masks = [item["mask_sha256"] for item in items]
        if len(set(masks)) != 3:
            raise ValueError(f"Expected three distinct fixed masks for r{rate}")
        mae = [float(item["unmasked_test"]["test_mae_unmasked"]) for item in items]
        rmse = [math.sqrt(float(item["unmasked_test"]["test_mse_unmasked"])) for item in items]
        report[f"r{rate}"] = {"mae": summary(mae), "rmse": summary(rmse), "mask_sha256": masks}
    return report


def main() -> None:
    bitgraph = load_rows(RESULT / "bitgraph_unified_blockt_v2", "bitgraph_official_architecture")
    controlled = [json.loads(path.read_text()) for path in sorted((RESULT / "controlled").glob("*.json"))]
    report = {
        "protocol": "METR-LA unified Block-T, 24-step history, 12-step horizon, unmasked test targets",
        "bitgraph_official_architecture": group_rows(bitgraph),
        "controlled_result_count": len(controlled),
        "source_commit": "4f1d05bcc20bb3f5084bd1facc995478f84a40ed",
    }
    path = RESULT / "unified_blockt_v2_verification_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

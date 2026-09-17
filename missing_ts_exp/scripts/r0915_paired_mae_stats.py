#!/usr/bin/env python3
"""Report paired MAE improvements, confidence intervals, and p-values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SEEDS = range(2024, 2029)


def load_mae(path: Path) -> float:
    return float(json.loads(path.read_text())["unmasked_test"]["test_mae_unmasked"])


def paired_summary(name: str, baseline_paths: list[Path], mrmsgf_paths: list[Path]) -> dict[str, object]:
    baseline = [load_mae(path) for path in baseline_paths]
    mrmsgf = [load_mae(path) for path in mrmsgf_paths]
    delta = [base - ours for base, ours in zip(baseline, mrmsgf)]
    interval = stats.t.interval(0.95, len(delta) - 1, loc=float(stats.tmean(delta)), scale=float(stats.sem(delta)))
    test = stats.ttest_1samp(delta, popmean=0.0)
    return {
        "condition": name,
        "n_paired_seeds": len(delta),
        "seed_order": list(SEEDS),
        "baseline_minus_mrmsgf_mae": delta,
        "mean_delta": float(stats.tmean(delta)),
        "ci95": [float(interval[0]), float(interval[1])],
        "two_sided_p_value": float(test.pvalue),
    }


def add_holm_bonferroni(rows: list[dict[str, object]]) -> None:
    """Add monotone Holm-adjusted p-values for the declared five tests."""
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["two_sided_p_value"]))
    running_max = 0.0
    n_tests = len(rows)
    for rank, (_, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["two_sided_p_value"]) * (n_tests - rank))
        running_max = max(running_max, adjusted)
        row["holm_bonferroni_p_value"] = running_max
    for row in rows:
        row["holm_family_size"] = n_tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default=str(ROOT / "missing_ts_exp/results/r0910_evidence_closure/paired_mae_inference.json"))
    args = parser.parse_args()
    block_summary = json.loads((ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a1_verification_summary.json").read_text())
    point_root = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a2_point"
    rows = [
        paired_summary(
            f"Block-T {rate}%",
            [Path(path) for path in block_summary["groups"][f"r{rate}_official_base"]["result_paths"]],
            [Path(path) for path in block_summary["groups"][f"r{rate}_mrmsgf"]["result_paths"]],
        )
        for rate in (30, 50, 70)
    ]
    rows.extend(
        paired_summary(
            f"Point {rate}%",
            [point_root / f"official_base_METR-LA_point_r{rate}_seed{seed}.json" for seed in SEEDS],
            [point_root / f"mrmsgf_METR-LA_point_r{rate}_seed{seed}.json" for seed in SEEDS],
        )
        for rate in (30, 50)
    )
    add_holm_bonferroni(rows)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()

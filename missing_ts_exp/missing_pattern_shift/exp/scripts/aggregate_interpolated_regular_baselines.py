#!/usr/bin/env python3
"""Aggregate completed interpolation-baseline test JSONs into an auditable 2x2 report."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

from protocol import EXP_ROOT


SUITE = "r50_b24_l96_h96_per_channel_linear_interpolation_nw10"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out = EXP_ROOT / "summary" / SUITE
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for job in manifest["jobs"]:
        folder = EXP_ROOT / "results" / SUITE / job["id"]
        for test in job["evaluations"]:
            path = folder / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json"
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if record.get("status") != "complete":
                continue
            metric = record["test"]
            rows.append({
                "baseline": record["baseline"], "dataset": record["dataset"],
                "train_family": record["train_family"], "test_family": record["test_family"],
                "train_seed": record["train_seed"], "test_mask_seed": record["test_mask_seed"],
                "mae": metric["mae_raw"], "mse": metric["mse_raw"], "path": str(path),
            })
    fields = ["baseline", "dataset", "train_family", "test_family", "train_seed", "test_mask_seed", "mae", "mse", "path"]
    with (out / "per_mask.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["baseline"], row["dataset"], row["train_family"], row["test_family"]), []).append(row)
    summary: list[dict] = []
    for key, part in sorted(groups.items()):
        mae, mae_sd = mean_sd([row["mae"] for row in part])
        mse, mse_sd = mean_sd([row["mse"] for row in part])
        summary.append({
            "baseline": key[0], "dataset": key[1], "train_family": key[2], "test_family": key[3],
            "repetitions": len(part), "mae_mean": mae, "mae_sd": mae_sd, "mse_mean": mse, "mse_sd": mse_sd,
        })
    with (out / "condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else [
            "baseline", "dataset", "train_family", "test_family", "repetitions", "mae_mean", "mae_sd", "mse_mean", "mse_sd",
        ])
        writer.writeheader(); writer.writerows(summary)
    lookup = {(r["baseline"], r["dataset"], r["train_family"], r["test_family"]): r for r in summary}
    effects: list[dict] = []
    for baseline in manifest["baselines"]:
        for dataset in manifest["datasets"]:
            # Fixed B test: positive value is the target P->B mismatch penalty.
            pb = lookup.get((baseline, dataset, "P", "B")); bb = lookup.get((baseline, dataset, "B", "B"))
            pp = lookup.get((baseline, dataset, "P", "P")); bp = lookup.get((baseline, dataset, "B", "P"))
            for label, left, right in (("P_to_B_minus_B_to_B", pb, bb), ("B_to_P_minus_P_to_P", bp, pp)):
                if left is None or right is None:
                    continue
                effects.append({
                    "baseline": baseline, "dataset": dataset, "effect": label,
                    "mse_delta": left["mse_mean"] - right["mse_mean"],
                    "mse_percent": 100 * (left["mse_mean"] / right["mse_mean"] - 1),
                    "mae_delta": left["mae_mean"] - right["mae_mean"],
                    "mae_percent": 100 * (left["mae_mean"] / right["mae_mean"] - 1),
                })
    with (out / "mode_shift_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["baseline", "dataset", "effect", "mse_delta", "mse_percent", "mae_delta", "mae_percent"])
        writer.writeheader(); writer.writerows(effects)
    lines = [
        "# 线性插补常规预测基线：r=50% 2×2 结果", "",
        f"已完成固定掩码测试：{len(rows)} / {len(manifest['jobs']) * 4}。", "",
        "所有分数为 raw-scale；每格最多包含 2 个训练种子 × 2 个固定测试掩码。", "",
        "| 模型 | 数据集 | 训练→测试 | n | MSE（均值 ± SD） | MAE（均值 ± SD） |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['baseline']} | {item['dataset']} | {item['train_family']}→{item['test_family']} | {item['repetitions']} | "
            f"{item['mse_mean']:.6g} ± {item['mse_sd']:.6g} | {item['mae_mean']:.6g} ± {item['mae_sd']:.6g} |"
        )
    lines += ["", "## 固定测试模式下的训练模式变化", "", "正值表示切换训练缺失模式后误差升高。", "",
              "| 模型 | 数据集 | 对比 | MSE 变化 | MAE 变化 |", "|---|---|---|---:|---:|"]
    for item in effects:
        lines.append(f"| {item['baseline']} | {item['dataset']} | {item['effect']} | {item['mse_percent']:+.2f}% | {item['mae_percent']:+.2f}% |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} per-mask rows and {len(summary)} condition rows to {out}")


if __name__ == "__main__":
    main()

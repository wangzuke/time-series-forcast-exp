#!/usr/bin/env python3
"""Read-only aggregation of registered tasks; smoke artifacts are excluded."""
import csv
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

EXP_ROOT = Path(__file__).resolve().parents[1]
SUITE = "posterior_s4_r50_b24_l96_h96_1119"


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    manifest = read(EXP_ROOT / "configs" / f"{SUITE}.json")
    if not manifest:
        raise RuntimeError("Manifest missing")
    root = EXP_ROOT / "results" / SUITE
    output = EXP_ROOT / "summary" / SUITE
    output.mkdir(parents=True, exist_ok=True)
    rows, trained, finished = [], 0, 0
    for job in manifest["jobs"]:
        directory = root / job["id"]
        trained += read(directory / "training_complete.json").get("status") == "complete"
        finished += read(directory / "progress.json").get("status") == "complete"
        for family in manifest["test_families"]:
            for seed in manifest["test_mask_seeds"]:
                result = read(directory / f"test_{family}_m{seed}.json")
                if result.get("status") == "complete":
                    rows.append({"dataset": job["dataset"], "variant": job["variant"], "train_family": job["family"],
                                 "train_seed": job["seed"], "test_family": family, "mask_seed": seed,
                                 **{k: result[k] for k in ("mse_standardized", "mae_standardized", "mse_raw", "mae_raw")}})
    supervisor = read(EXP_ROOT / "logs" / SUITE / "supervisor_status.json")
    total_jobs = len(manifest["jobs"])
    total_tests = total_jobs * len(manifest["test_families"]) * len(manifest["test_mask_seeds"])
    lines = ["# 观测条件后验与轻量 S4：运行和结果汇总", "",
             f"更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %z')}；状态：{supervisor.get('status', 'not_started')}。", "",
             f"正式训练完成 {trained}/{total_jobs}；含全部测试的任务完成 {finished}/{total_jobs}；测试完成 {len(rows)}/{total_tests}。", "",
             "三个版本为 LI-S4、STAT-S4、PU-S4。Smoke 和性能测试不进入本报告。",
             "以下 MSE 为标准化尺度；先在每个训练种子内平均两个测试掩码，再在已完成的训练种子间平均。",
             "括号为已完成的训练种子数；未完成结果不能用于最终排名。", ""]
    for dataset in sorted({job["dataset"] for job in manifest["jobs"]}):
        lines += [f"## {dataset}", "", "| 版本 | P→P | P→B | B→P | B→B |", "|---|---:|---:|---:|---:|"]
        for variant in ("LI-S4", "STAT-S4", "PU-S4"):
            cells = []
            for train, test in (("P", "P"), ("P", "B"), ("B", "P"), ("B", "B")):
                by_seed = []
                for seed in (2024, 2025):
                    values = [r["mse_standardized"] for r in rows if r["dataset"] == dataset and r["variant"] == variant
                              and r["train_family"] == train and r["test_family"] == test and r["train_seed"] == seed]
                    if len(values) == 2:
                        by_seed.append(mean(values))
                cells.append(f"{mean(by_seed):.6f} ({len(by_seed)})" if by_seed else "—")
            lines.append("| " + " | ".join([variant] + cells) + " |")
        lines += [""]
    lines += ["## 当前活动任务", ""]
    for active in supervisor.get("active", []):
        progress = read(root / active["id"] / "progress.json")
        lines.append(f"- GPU {active['gpu']}，{active['id']}，PID {active['pid']}，"
                     f"{progress.get('status', 'initializing')}，epoch={progress.get('epoch', '—')}，batch={progress.get('batch', '—')}。")
    if supervisor.get("failures"):
        lines += ["", "异常：", "", "```json", json.dumps(supervisor["failures"], ensure_ascii=False, indent=2), "```"]
    with tempfile.NamedTemporaryFile("w", dir=output, encoding="utf-8", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
        temp = handle.name
    os.replace(temp, output / "report.md")
    if rows:
        with tempfile.NamedTemporaryFile("w", dir=output, encoding="utf-8", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            temp = handle.name
        os.replace(temp, output / "results.csv")
    print(json.dumps({"trained": trained, "finished": finished, "tests": len(rows), "report": str(output / "report.md")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=SUITE)
    SUITE = parser.parse_args().suite
    main()

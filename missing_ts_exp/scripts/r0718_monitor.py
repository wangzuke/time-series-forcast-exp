#!/usr/bin/env python3
"""Lightweight monitor for the 0718 block-missing experiments.

This script is intentionally read-only. It reports:
- current wall-clock time;
- nvidia-smi status when available;
- CUDA availability in the two experiment conda envs when requested externally;
- matching experiment processes;
- latest 0718 log files and their tails;
- CSV artifact row counts.

Usage:
  python missing_ts_exp/scripts/r0718_monitor.py
  python missing_ts_exp/scripts/r0718_monitor.py --tail-lines 30
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import pathlib
import subprocess
from typing import Iterable


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0718_block_hmbg"
LOG_DIR = RESULTS / "raw_logs"
CSV_DIR = RESULTS / "csv"


def run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except Exception as exc:  # keep monitor robust
        return 999, f"{type(exc).__name__}: {exc}"


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def newest_files(paths: Iterable[pathlib.Path], n: int) -> list[pathlib.Path]:
    existing = [p for p in paths if p.exists()]
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)[:n]


def count_csv_rows(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(0, len(rows) - 1)


def tail(path: pathlib.Path, n_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument("--latest-logs", type=int, default=8)
    args = parser.parse_args()

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"0718 monitor time: {now}")
    print(f"results dir: {RESULTS}")

    print_section("GPU / NVIDIA")
    code, out = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ])
    if code == 0:
        print(out)
    else:
        print(f"nvidia-smi unavailable (exit={code})")
        print(out)

    print_section("Matching processes")
    code, out = run(["ps", "-eo", "pid,ppid,stat,etime,pcpu,pmem,args"])
    if code != 0:
        print(out)
    else:
        needles = (
            "run_realworld",
            "run_realworld_patched",
            "BiaTCGNet",
            "main.py",
            "test_forecasting.py",
            "block_hmbg",
            "0718_block_hmbg",
        )
        matched = [
            line for line in out.splitlines()
            if any(needle in line for needle in needles)
            and "r0718_monitor.py" not in line
        ]
        print("\n".join(matched) if matched else "No matching experiment processes.")

    print_section("CSV artifacts")
    if CSV_DIR.exists():
        for p in sorted(CSV_DIR.glob("*.csv")):
            mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{p.name:30s} rows={count_csv_rows(p):4d} size={p.stat().st_size:8d} mtime={mtime}")
    else:
        print(f"CSV dir missing: {CSV_DIR}")

    print_section("Latest logs")
    logs = newest_files(LOG_DIR.glob("*") if LOG_DIR.exists() else [], args.latest_logs)
    if not logs:
        print(f"No logs found in {LOG_DIR}")
    for p in logs:
        mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- {p.name} size={p.stat().st_size} mtime={mtime} ---")
        print(tail(p, args.tail_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


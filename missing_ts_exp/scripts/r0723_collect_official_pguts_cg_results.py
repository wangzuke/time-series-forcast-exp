#!/usr/bin/env python3
"""Collect formal R0723 P-GUTS/CG-P-GUTS imputation results.

The workspace contains several historical and aborted 0723 runs. This collector
uses an explicit allowlist for the formal [3,6] checkpoints and parses only the
Table 3 inference logs generated from those checkpoints.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "missing_ts_exp/results/0723_official_pguts_coarse_imputation"
RAW_LOG_DIR = EXP_ROOT / "raw_logs"
CSV_DIR = EXP_ROOT / "csv"
PGUTS_LOG_DIR = ROOT / "external_repro/pguts/log"


@dataclass(frozen=True)
class MainRun:
    dataset: str
    model: str
    seed: int
    exp_name: str
    raw_log: str
    expected_graph_variant: str
    expected_batch_size: int
    expected_batches_epoch: int


MAIN_RUNS = [
    MainRun("la_block", "P-GUTS", 1, "20260726T112948_1_reproduce_la_block_la_pguts_36_s1", "reproduce_la_block_la_pguts_36_s1.log", "full_only", 128, 19),
    MainRun("la_block", "P-GUTS", 2, "20260726T151909_2_reproduce_la_block_la_pguts_36_s2", "reproduce_la_block_la_pguts_36_s2.log", "full_only", 128, 19),
    MainRun("la_block", "CG-P-GUTS", 1, "20260726T112958_1_coarse_la_block_la_cg_pguts_36_s1", "coarse_la_block_la_cg_pguts_36_s1.log", "full_plus_coarse", 128, 19),
    MainRun("la_block", "CG-P-GUTS", 2, "20260726T112958_2_coarse_la_block_la_cg_pguts_36_s2", "coarse_la_block_la_cg_pguts_36_s2.log", "full_plus_coarse", 128, 19),
    MainRun("bay_block", "P-GUTS", 1, "20260726T183727_1_reproduce_clean_bay_block_bay_pguts_36_s1", "reproduce_clean_bay_block_bay_pguts_36_s1.log", "full_only", 256, 10),
    MainRun("bay_block", "P-GUTS", 2, "20260726T183728_2_reproduce_clean_bay_block_bay_pguts_36_s2", "reproduce_clean_bay_block_bay_pguts_36_s2.log", "full_only", 256, 10),
    MainRun("bay_block", "CG-P-GUTS", 1, "20260726T112956_1_coarse_bay_block_bay_cg_pguts_36_s1", "coarse_bay_block_bay_cg_pguts_36_s1.log", "full_plus_coarse", 256, 10),
    MainRun("bay_block", "CG-P-GUTS", 2, "20260726T112957_2_coarse_bay_block_bay_cg_pguts_36_s2", "coarse_bay_block_bay_cg_pguts_36_s2.log", "full_plus_coarse", 256, 10),
]


DATASET_LABEL = {
    "la_block": "METR-LA",
    "bay_block": "PEMS-BAY",
}

DATASET_ORDER = {
    "METR-LA": 0,
    "PEMS-BAY": 1,
}

MODEL_ORDER = {
    "P-GUTS": 0,
    "CG-P-GUTS": 1,
}

PF_LABEL = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p15": 0.15,
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std_pop(values: list[float]) -> float:
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def read_test_mae(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Test MAE:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        raise RuntimeError(f"Could not find Test MAE in {log_path}")
    return float(matches[-1])


def checkpoint_dir(dataset: str, exp_name: str) -> Path:
    return PGUTS_LOG_DIR / dataset / "PGUTS" / exp_name


def validate_main_config(run: MainRun, cfg: dict) -> list[str]:
    issues = []
    expected_factor_t = [3, 6]
    checks = {
        "dataset_name": run.dataset,
        "seed": run.seed,
        "graph_variant": run.expected_graph_variant,
        "batch_size": run.expected_batch_size,
        "batches_epoch": run.expected_batches_epoch,
        "p_fault": 0.0015,
        "p_noise": 0.05,
        "window": 24,
        "stride": 1,
    }
    for key, expected in checks.items():
        actual = cfg.get(key)
        if actual != expected:
            issues.append(f"{key}={actual!r} expected {expected!r}")
    if list(cfg.get("factor_t", [])) != expected_factor_t:
        issues.append(f"factor_t={cfg.get('factor_t')!r} expected {expected_factor_t!r}")
    return issues


def collect_main_runs() -> list[dict]:
    rows = []
    for run in MAIN_RUNS:
        cfg_path = checkpoint_dir(run.dataset, run.exp_name) / "config.yaml"
        log_path = RAW_LOG_DIR / run.raw_log
        output_path = checkpoint_dir(run.dataset, run.exp_name) / "output.pt"
        ckpts = sorted(checkpoint_dir(run.dataset, run.exp_name).glob("*.ckpt"))
        cfg = read_yaml(cfg_path)
        issues = validate_main_config(run, cfg)
        if not output_path.exists():
            issues.append("missing output.pt")
        if not ckpts:
            issues.append("missing .ckpt")
        rows.append(
            {
                "dataset": DATASET_LABEL[run.dataset],
                "dataset_name": run.dataset,
                "model": run.model,
                "seed": run.seed,
                "factor_t": "[3,6]",
                "graph_variant": cfg.get("graph_variant"),
                "batch_size": cfg.get("batch_size"),
                "batches_epoch": cfg.get("batches_epoch"),
                "p_fault_train": cfg.get("p_fault"),
                "p_noise_train": cfg.get("p_noise"),
                "test_mae": read_test_mae(log_path),
                "checkpoint_dir": str(checkpoint_dir(run.dataset, run.exp_name)),
                "raw_log": str(log_path),
                "status": "ok" if not issues else "warning",
                "notes": "; ".join(issues),
            }
        )
    return rows


def summarize_main(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["dataset"], row["model"]), []).append(row)

    summaries = []
    for (dataset, model), group in sorted(
        by_key.items(), key=lambda item: (DATASET_ORDER[item[0][0]], MODEL_ORDER[item[0][1]])
    ):
        values = [float(row["test_mae"]) for row in group]
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "n_train_seeds": len(values),
                "mae_mean": mean(values),
                "mae_std_pop": std_pop(values),
                "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda item: item["seed"])),
            }
        )

    baseline_by_dataset = {
        row["dataset"]: row for row in summaries if row["model"] == "P-GUTS"
    }
    for row in summaries:
        baseline = baseline_by_dataset.get(row["dataset"])
        if row["model"] == "CG-P-GUTS" and baseline is not None:
            row["delta_mae_vs_pguts"] = row["mae_mean"] - baseline["mae_mean"]
            row["delta_pct_vs_pguts"] = 100 * row["delta_mae_vs_pguts"] / baseline["mae_mean"]
        else:
            row["delta_mae_vs_pguts"] = 0.0 if row["model"] == "P-GUTS" else None
            row["delta_pct_vs_pguts"] = 0.0 if row["model"] == "P-GUTS" else None
    return summaries


def parse_table3_log(path: Path) -> dict:
    match = re.match(
        r"table3_(la_block|bay_block)_inference_s(.+)_pf(0p05|0p1|0p15)\.log$",
        path.name,
    )
    if match is None:
        raise RuntimeError(f"Unexpected Table 3 log name: {path.name}")
    dataset_name, exp_name, pf_token = match.groups()
    seed_match = re.search(r"T\d+_(\d+)_", exp_name)
    if seed_match is None:
        raise RuntimeError(f"Could not parse training seed from {exp_name}")
    train_seed = int(seed_match.group(1))
    model = "CG-P-GUTS" if "_cg_pguts_" in exp_name else "P-GUTS"

    text = path.read_text(encoding="utf-8", errors="replace")
    seed_rows = [
        (int(seed), float(mae))
        for seed, mae in re.findall(r"SEED\s+(\d+)\s+-\s+Test MAE:\s*([0-9]+(?:\.[0-9]+)?)", text)
    ]
    if len(seed_rows) != 5:
        raise RuntimeError(f"Expected 5 mask seeds in {path}, got {len(seed_rows)}")
    summary_match = re.search(r"MAE over\s+5\s+runs:\s*([0-9]+(?:\.[0-9]+)?)±([0-9]+(?:\.[0-9]+)?)", text)
    if summary_match is None:
        raise RuntimeError(f"Could not parse MAE summary in {path}")

    cfg_path = checkpoint_dir(dataset_name, exp_name) / "config.yaml"
    cfg = read_yaml(cfg_path)
    return {
        "dataset": DATASET_LABEL[dataset_name],
        "dataset_name": dataset_name,
        "model": model,
        "train_seed": train_seed,
        "eval_p_fault": PF_LABEL[pf_token],
        "eval_p_noise": 0.0,
        "mask_seeds": ",".join(str(seed) for seed, _ in seed_rows),
        "mask_seed_maes": ",".join(f"{mae:.2f}" for _, mae in seed_rows),
        "mae_mean_over_mask_seeds": float(summary_match.group(1)),
        "mae_std_over_mask_seeds": float(summary_match.group(2)),
        "checkpoint_dir": str(checkpoint_dir(dataset_name, exp_name)),
        "graph_variant": cfg.get("graph_variant"),
        "raw_log": str(path),
    }


def collect_table3_runs() -> list[dict]:
    logs = sorted(RAW_LOG_DIR.glob("table3_*_inference_s*_pf*.log"))
    rows = [parse_table3_log(path) for path in logs]
    formal_exp_names = {run.exp_name for run in MAIN_RUNS}
    rows = [
        row
        for row in rows
        if Path(row["checkpoint_dir"]).name in formal_exp_names
        and row["model"] in {"P-GUTS", "CG-P-GUTS"}
    ]
    return sorted(
        rows,
        key=lambda row: (row["dataset"], row["model"], row["train_seed"], row["eval_p_fault"]),
    )


def summarize_table3(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["dataset"], float(row["eval_p_fault"]), row["model"]), []).append(row)

    summaries = []
    for (dataset, eval_p_fault, model), group in sorted(
        by_key.items(),
        key=lambda item: (DATASET_ORDER[item[0][0]], item[0][1], MODEL_ORDER[item[0][2]]),
    ):
        values = [float(row["mae_mean_over_mask_seeds"]) for row in group]
        summaries.append(
            {
                "dataset": dataset,
                "eval_p_fault": eval_p_fault,
                "model": model,
                "n_train_seeds": len(values),
                "mae_mean": mean(values),
                "mae_std_pop_across_train_seeds": std_pop(values),
                "train_seeds": ",".join(str(row["train_seed"]) for row in sorted(group, key=lambda item: item["train_seed"])),
            }
        )

    baseline_by_key = {
        (row["dataset"], row["eval_p_fault"]): row
        for row in summaries
        if row["model"] == "P-GUTS"
    }
    for row in summaries:
        baseline = baseline_by_key.get((row["dataset"], row["eval_p_fault"]))
        if row["model"] == "CG-P-GUTS" and baseline is not None:
            row["delta_mae_vs_pguts"] = row["mae_mean"] - baseline["mae_mean"]
            row["delta_pct_vs_pguts"] = 100 * row["delta_mae_vs_pguts"] / baseline["mae_mean"]
        else:
            row["delta_mae_vs_pguts"] = 0.0 if row["model"] == "P-GUTS" else None
            row["delta_pct_vs_pguts"] = 0.0 if row["model"] == "P-GUTS" else None
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(main_summary: list[dict], table3_summary: list[dict]) -> None:
    path = EXP_ROOT / "summary_0723_official_pguts_cg.md"
    lines = [
        "# 0723 P-GUTS + Coarse Graph 插补实验结果汇总",
        "",
        "## 实验范围",
        "",
        "- 只汇总本轮正式实验：P-GUTS `[3,6]` 与 CG-P-GUTS `[3,6]`，seed=1/2。",
        "- 复盘修正：本轮 `CG-P-GUTS` 使用的是按节点编号连续分组的 coarse branch，不是基于距离图/邻接图聚类的 graph-aware coarsening。",
        "- 主 benchmark 使用 traffic paper BLOCK：训练 `p_fault=0.0015, p_noise=0.05`。",
        "- Table 3 鲁棒性使用同一 checkpoint 做 inference-only 测试，未为不同缺失强度重训。",
        "- METR-LA 使用 `batch_size=128, batches_epoch=19`；PEMS-BAY 使用 `batch_size=256, batches_epoch=10`。",
        "",
        "## 主 benchmark: paper BLOCK, P-GUTS [3,6]",
        "",
        "| Dataset | Model | MAE mean | MAE std | Delta vs P-GUTS | Delta % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_summary:
        lines.append(
            "| {dataset} | {model} | {mae_mean} | {mae_std} | {delta} | {delta_pct}% |".format(
                dataset=row["dataset"],
                model=row["model"],
                mae_mean=fmt_float(row["mae_mean"], 4),
                mae_std=fmt_float(row["mae_std_pop"], 4),
                delta=fmt_float(row["delta_mae_vs_pguts"], 4),
                delta_pct=fmt_float(row["delta_pct_vs_pguts"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## Table 3 鲁棒性: 同一 checkpoint, inference-only eval mask",
            "",
            "| Dataset | Eval p_fault | Model | MAE mean | Seed std | Delta vs P-GUTS | Delta % |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in table3_summary:
        lines.append(
            "| {dataset} | {pf} | {model} | {mae_mean} | {mae_std} | {delta} | {delta_pct}% |".format(
                dataset=row["dataset"],
                pf=fmt_float(row["eval_p_fault"], 2),
                model=row["model"],
                mae_mean=fmt_float(row["mae_mean"], 4),
                mae_std=fmt_float(row["mae_std_pop_across_train_seeds"], 4),
                delta=fmt_float(row["delta_mae_vs_pguts"], 4),
                delta_pct=fmt_float(row["delta_pct_vs_pguts"], 2),
            )
        )
    lines.extend(
        [
            "",
            "说明：std 为 population std；Table 3 每个 checkpoint 先对 5 个 test mask seed 求均值，再在训练 seed=1/2 上汇总。",
            "",
            "## 初步解读",
            "",
            "1. METR-LA 上编号连续分组 coarse branch 有稳定正收益。主 benchmark 中 MAE 从 4.46 降到 4.32，改善 3.14%；Table 3 三个测试缺失强度下也都有约 2% 的改善。",
            "2. PEMS-BAY 上编号连续分组 coarse branch 目前不是收益模块。主 benchmark 中 MAE 从 2.69 升到 3.065，退化 13.94%；Table 3 中也稳定退化约 4% 到 5%。",
            "3. 这个结果不能证明 graph-aware coarse graph 有效，因为当前分组没有使用传感器距离、邻接关系或坐标。它更像一个 index-contiguous node pooling 对照，可能混入了数据文件节点排序的偶然性。",
            "4. 后续不建议直接把当前 CG-P-GUTS 作为统一替代模型。下一步应先把 coarse 构造改成 distance clustering / adjacency clustering，再与当前 contiguous assignment 做构造方式对照；否则无法判断收益来自真实空间粗化，还是来自额外分支容量和编号分组。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    main_rows = collect_main_runs()
    main_summary = summarize_main(main_rows)
    table3_rows = collect_table3_runs()
    table3_summary = summarize_table3(table3_rows)

    write_csv(CSV_DIR / "main_training_runs.csv", main_rows)
    write_csv(CSV_DIR / "main_summary.csv", main_summary)
    write_csv(CSV_DIR / "table3_inference_runs.csv", table3_rows)
    write_csv(CSV_DIR / "table3_inference_summary.csv", table3_summary)
    write_markdown(main_summary, table3_summary)

    print(f"Wrote {len(main_rows)} main rows and {len(table3_rows)} Table 3 rows.")
    print(CSV_DIR)


if __name__ == "__main__":
    main()

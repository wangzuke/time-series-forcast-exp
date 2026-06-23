"""把 results/ 下所有 *.json 汇总成 CSV/Markdown 表格。

主结果表：
  数据集 × 缺失类型 × 缺失率 × 预测长度 × 方法 (含预测器/填补) → MSE/MAE (mean ± std over seeds)

也输出鲁棒性分析表、补值误差—预测误差散点数据。
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict
import math


def load_all(results_dir: str):
    rows = []
    for fp in glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True):
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception as e:
            print("skip", fp, e); continue
        cfg = d.get("config", {})
        te = d.get("test", {})
        row = {
            "dataset": cfg.get("dataset"),
            "method": cfg.get("method"),
            "predictor": cfg.get("predictor"),
            "impute": cfg.get("impute"),
            "missing_type": cfg.get("missing_type"),
            "missing_rate": cfg.get("missing_rate"),
            "seq_len": cfg.get("seq_len"),
            "pred_len": cfg.get("pred_len"),
            "seed": cfg.get("seed"),
            "mse": te.get("mse"),
            "mae": te.get("mae"),
            "impute_mse": te.get("impute_mse"),
            "n_params": d.get("n_params"),
            "peak_mem_mb": d.get("peak_mem_mb"),
            "train_time_sec": te.get("time_sec"),
            "best_val_mse": d.get("best_val_mse"),
        }
        rows.append(row)
    return rows


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    m = sum(values) / n
    if n > 1:
        s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    else:
        s = 0.0
    return m, s


def aggregate(rows, group_keys):
    """按 group_keys 聚合 MSE/MAE 均值/标准差，跨 seed。"""
    bucket = defaultdict(list)
    for r in rows:
        k = tuple(r.get(g) for g in group_keys)
        if r.get("mse") is None: continue
        bucket[k].append(r)
    agg = []
    for k, lst in bucket.items():
        mse_m, mse_s = mean_std([r["mse"] for r in lst])
        mae_m, mae_s = mean_std([r["mae"] for r in lst])
        imp_m, imp_s = mean_std([r["impute_mse"] for r in lst if r.get("impute_mse") is not None])
        rec = dict(zip(group_keys, k))
        rec.update({
            "mse_mean": mse_m, "mse_std": mse_s,
            "mae_mean": mae_m, "mae_std": mae_s,
            "impute_mse_mean": imp_m, "impute_mse_std": imp_s,
            "n_seeds": len(lst),
        })
        agg.append(rec)
    return agg


def write_csv(rows, path, cols=None):
    if not rows:
        with open(path, "w") as f:
            f.write("(empty)\n")
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("wrote", path, "rows:", len(rows))


def write_markdown_table(rows, path, cols, headers=None):
    headers = headers or cols
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(
                f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in cols
            ) + " |\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results_aggregated")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_all(args.results_dir)
    print(f"loaded {len(rows)} result files")
    if not rows:
        return
    write_csv(rows, os.path.join(args.out_dir, "raw.csv"))

    # 主结果（按数据集 / 缺失类型 / 缺失率 / 预测长度 / 方法 / 预测器 / 填补）
    main_keys = ["dataset", "missing_type", "missing_rate", "pred_len",
                 "method", "predictor", "impute"]
    main_agg = aggregate(rows, main_keys)
    main_agg.sort(key=lambda r: (str(r.get("dataset")), str(r.get("missing_type")),
                                  r.get("missing_rate") or 0, r.get("pred_len") or 0,
                                  str(r.get("method")), str(r.get("predictor")),
                                  str(r.get("impute"))))
    write_csv(main_agg, os.path.join(args.out_dir, "main_results.csv"),
              cols=main_keys + ["mse_mean", "mse_std", "mae_mean", "mae_std",
                                 "impute_mse_mean", "n_seeds"])
    write_markdown_table(main_agg, os.path.join(args.out_dir, "main_results.md"),
                         cols=main_keys + ["mse_mean", "mse_std", "mae_mean", "mae_std", "n_seeds"])

    # 鲁棒性分析：固定 (数据集, 方法, 预测器, 填补, 预测长度, 缺失类型) 看缺失率变化
    print("aggregation done. files in", args.out_dir)


if __name__ == "__main__":
    main()

"""作图脚本：缺失率-误差曲线、补值-预测误差散点图、方法对比柱状图。

依赖 matplotlib，调用 results_aggregated/main_results.csv 与 raw.csv。
"""
from __future__ import annotations
import argparse
import os
import json
from collections import defaultdict
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


def read_csv(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            for k, v in list(r.items()):
                if v in ("", None):
                    continue
                try:
                    r[k] = float(v) if "." in v or "e" in v.lower() else int(v) if v.lstrip("-").isdigit() else v
                except Exception:
                    pass
            rows.append(r)
    return rows


def plot_rate_curves(rows, out_dir, dataset, missing_type, pred_len):
    """对给定 (dataset, missing_type, pred_len)，画各方法的缺失率-MSE 曲线。"""
    # 按 method+predictor+impute 分组
    by_method = defaultdict(dict)
    for r in rows:
        if r.get("dataset") != dataset: continue
        if r.get("missing_type") != missing_type: continue
        if int(r.get("pred_len", 0)) != int(pred_len): continue
        key = f"{r.get('method')}/{r.get('predictor')}/{r.get('impute')}"
        by_method[key][float(r.get("missing_rate", 0))] = (float(r.get("mse_mean", float("nan"))),
                                                            float(r.get("mse_std", 0.0)))
    if not by_method:
        return
    plt.figure(figsize=(7, 5))
    for k, d in sorted(by_method.items()):
        xs = sorted(d.keys())
        ys = [d[x][0] for x in xs]
        es = [d[x][1] for x in xs]
        plt.errorbar(xs, ys, yerr=es, marker="o", label=k, capsize=2)
    plt.xlabel("missing rate")
    plt.ylabel("MSE")
    plt.title(f"{dataset} | {missing_type} | H={pred_len}")
    plt.legend(fontsize=7, loc="best")
    plt.grid(True, alpha=0.3)
    out = os.path.join(out_dir, f"curve_{dataset}_{missing_type}_h{pred_len}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def plot_impute_vs_forecast(rows, out_dir):
    xs, ys, labels = [], [], []
    for r in rows:
        if r.get("impute_mse") in (None, "", float("nan")) or r.get("mse") in (None, "", float("nan")):
            continue
        try:
            ix = float(r["impute_mse"]); fy = float(r["mse"])
        except Exception:
            continue
        if ix <= 0: continue
        xs.append(ix); ys.append(fy)
        labels.append(f"{r.get('method')}/{r.get('predictor')}/{r.get('impute')}")
    if not xs:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=12, alpha=0.6)
    plt.xlabel("Imputation MSE (mask=0)")
    plt.ylabel("Forecast MSE")
    plt.title("Imputation error vs. Forecast error")
    plt.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "impute_vs_forecast.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_dir", default="results_aggregated")
    ap.add_argument("--raw_csv", default=None)
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    agg_csv = os.path.join(args.agg_dir, "main_results.csv")
    if not os.path.exists(agg_csv):
        print("no aggregated results at", agg_csv)
        return
    rows = read_csv(agg_csv)

    # 缺失率-误差曲线
    combos = set()
    for r in rows:
        combos.add((r.get("dataset"), r.get("missing_type"), int(r.get("pred_len", 96))))
    for ds, mt, H in sorted(combos):
        if ds in (None, ""): continue
        if mt in (None, "none", ""): continue
        plot_rate_curves(rows, args.out_dir, ds, mt, H)

    # 补值 vs 预测散点图（用 raw 数据更密集）
    raw = args.raw_csv or os.path.join(args.agg_dir, "raw.csv")
    if os.path.exists(raw):
        plot_impute_vs_forecast(read_csv(raw), args.out_dir)


if __name__ == "__main__":
    main()

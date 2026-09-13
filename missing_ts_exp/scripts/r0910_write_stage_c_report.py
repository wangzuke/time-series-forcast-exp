#!/usr/bin/env python3
"""Render the validated AQI36 cross-dataset experiment report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_c_verification_summary.json"
REPORT = ROOT / "missing_ts_exp/docs/0910实验报告_AQI36自然缺失与Point验证.md"


def fmt(value: dict[str, float]) -> str:
    return f"{value['mean']:.4f} +/- {value['std']:.4f}"


def main() -> None:
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Run r0910_verify_stage_c.py first: {SUMMARY}")
    result = json.loads(SUMMARY.read_text())
    if result.get("status") != "passed":
        raise ValueError("Refusing to render an unverified AQI36 report.")
    lines = [
        "# 0910 实验报告：AQI36 自然缺失与 Point 验证", "",
        "## 1. 目的与协议", "",
        "本阶段在具有站点地理坐标的 AQI36 空气质量数据上检验跨数据集可迁移性。输入有效掩码始终保留数据固有的自然缺失；在其基础上分别使用不额外注入、额外 Point 25% 和额外 Point 50% 三种条件。物理图由站点经纬度距离构建，区域分配由该物理图固定生成。", "",
        f"自动校验通过 {result['n_result_rows']} 条结果：{result['protocol']}。", "",
        "## 2. 五 seed 结果", "",
        "| 条件 | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |", "|---|---|---:|---:|---:|",
    ]
    for condition, label in (("original", "自然缺失"), ("point_r25", "自然缺失 + Point 25%"), ("point_r50", "自然缺失 + Point 50%")):
        for model, name in (("official_base", "HD-TTS-AMP"), ("mrmsgf", "MR-MSGF")):
            group = result["groups"][f"{condition}_{model}"]
            delta = "0.0000" if model == "mrmsgf" else fmt(group["paired_mae_minus_mrmsgf"])
            lines.append(f"| {label} | {name} | {fmt(group['mae'])} | {fmt(group['rmse'])} | {delta} |")
    lines += ["", "## 3. 读法与边界", "",
              "正的配对 MAE 差值代表 MR-MSGF 较低。AQI36 的量纲、自然缺失率与交通数据不同，因此本表只检验同一 AQI36 条件内的相对比较，不与 METR-LA 的绝对 MAE/RMSE 横向比较。自然缺失条件的五个 seed 固定同一数据掩码，只改变随机初始化；Point 条件同时使用固定的五份额外掩码。", "",
              "## 4. 产物与下一步", "",
              f"- 自动校验摘要：[stage_c_verification_summary.json]({SUMMARY})。",
              "- 在跨数据集结果稳定后，下一步补充 horizon 分解、配对置信区间与模型效率统计；GinAR 仅在其官方实现完成统一协议审计后才进入可比表。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

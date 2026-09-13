#!/usr/bin/env python3
"""Render the validated METR-LA Point comparison report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a2_verification_summary.json"
REPORT = ROOT / "missing_ts_exp/docs/0910实验报告_METRLA_Point主表对照.md"
MODELS = (("official_base", "HD-TTS-AMP"), ("bitgraph", "BiTGraph"), ("mrmsgf", "MR-MSGF"))


def fmt(value: dict[str, float]) -> str:
    return f"{value['mean']:.4f} +/- {value['std']:.4f}"


def main() -> None:
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Run r0910_verify_stage_a2.py first: {SUMMARY}")
    result = json.loads(SUMMARY.read_text())
    if result.get("status") != "passed":
        raise ValueError("Refusing to render an unverified Point table.")
    lines = [
        "# 0910 实验报告：METR-LA Point 主表对照", "",
        "## 1. 目的与协议", "",
        "本阶段以独立点缺失补充 Block-T 主实验，检验共同观测关系和融合后区域化是否只依赖连续时间块故障。所有方法共享 METR-LA 的时间划分、24 步历史、12 步预测、输入 mask 与未遮蔽未来目标评估。", "",
        f"自动校验通过 {result['n_result_rows']} 条结果：{result['protocol']}。", "",
        "## 2. 五 seed 主结果", "",
        "| Point 缺失率 | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |", "|---:|---|---:|---:|---:|",
    ]
    for rate in result["rates"]:
        for key, label in MODELS:
            group = result["groups"][f"r{rate}_{key}"]
            delta = "0.0000" if key == "mrmsgf" else fmt(group["paired_mae_minus_mrmsgf"])
            lines.append(f"| {rate}% | {label} | {fmt(group['mae'])} | {fmt(group['rmse'])} | {delta} |")
    lines += ["", "## 3. 读法与边界", "",
              "正的配对差值表示 MR-MSGF 在相同 mask seed 下 MAE 更低。Point 结果是关系机制的补充验证，而非替代本文针对连续时间缺失的 Block-T 主场景；报告不将其解释为对 Block-ST 空间同步缺失的证据。", "",
              "## 4. 产物与下一步", "",
              f"- 自动校验摘要：[stage_a2_verification_summary.json]({SUMMARY})。",
              "- 下一阶段使用 AQI36 的自然缺失、附加 Point 25% 和 50%，验证带地理空间信息的环境监测数据上的可迁移性。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

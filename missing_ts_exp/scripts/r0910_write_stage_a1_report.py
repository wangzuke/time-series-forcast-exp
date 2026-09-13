#!/usr/bin/env python3
"""Render the validated METR-LA Block-T main-table expansion report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a1_verification_summary.json"
REPORT = ROOT / "missing_ts_exp/docs/0910实验报告_METRLA_BlockT主表扩展.md"
MODELS = (("official_base", "HD-TTS-AMP"), ("bitgraph", "BiTGraph"), ("mrmsgf", "MR-MSGF"))


def fmt(value: dict[str, float]) -> str:
    return f"{value['mean']:.4f} +/- {value['std']:.4f}"


def main() -> None:
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Run r0910_verify_stage_a1.py first: {SUMMARY}")
    result = json.loads(SUMMARY.read_text())
    if result.get("status") != "passed":
        raise ValueError("Refusing to render an unverified main table.")
    lines = [
        "# 0910 实验报告：METR-LA Block-T 主表扩展",
        "",
        "## 1. 目的与协议",
        "",
        "本阶段完成 METR-LA Block-T 30%、50%、70% 的五 seed 主表。HD-TTS-AMP、MR-MSGF 与 BiTGraph 均使用相同的 24 步历史、12 步预测、时间顺序数据划分和输入 mask；指标为原始有效未来目标上的未遮蔽 MAE/RMSE。BiTGraph 保持作者 `BiaTCGNet.Model` 架构，仅将输入数据接口接到相同协议。",
        "",
        f"自动校验通过 {result['n_result_rows']} 条结果：{result['protocol']}。数据 SHA256：`{result['data_sha256']}`。",
        "",
        "## 2. 五 seed 主结果",
        "",
        "| Block-T 缺失率 | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |",
        "|---:|---|---:|---:|---:|",
    ]
    for rate in result["rates"]:
        for key, label in MODELS:
            group = result["groups"][f"r{rate}_{key}"]
            delta = "0.0000" if key == "mrmsgf" else fmt(group["paired_mae_minus_mrmsgf"])
            lines.append(f"| {rate}% | {label} | {fmt(group['mae'])} | {fmt(group['rmse'])} | {delta} |")
    lines += [
        "",
        "## 3. 读法与边界",
        "",
        "表中配对差值计算为同一 mask seed 下 `baseline - MR-MSGF`；正值表示 MR-MSGF 的 MAE 更低。所有五个 seed 使用配对 mask，但三种方法的随机初始化也固定为对应 seed。由于 BiTGraph 的作者训练日程与 HD-TTS 的 epoch/批次数不同，主结论应以统一数据、mask 与评估口径下的性能比较为限，而不把训练日程差异解释为模型机制本身。",
        "",
        "Block-T 是本文方法更适配的场景：某节点有连续时间缺失，而同一时间的空间邻居和区域上下文通常仍部分可用。该结果不能直接外推到 Block-ST 的空间同步缺失压力场景。",
        "",
        "## 4. 产物与下一步",
        "",
        f"- 自动校验摘要：[stage_a1_verification_summary.json]({SUMMARY})。",
        "- 后续进行 METR-LA Point 30/50% 的五 seed 对照，检验共同观测关系在非连续缺失下的表现；随后开展 AQI36 自然缺失与附加 Point 实验。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

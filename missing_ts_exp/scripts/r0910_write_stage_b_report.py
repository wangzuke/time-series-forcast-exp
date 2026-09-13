#!/usr/bin/env python3
"""Render the validated Stage-B ablation summary as a dated Markdown report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_verification_summary.json"
DIAGNOSTICS = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_diagnostics"
REPORT = ROOT / "missing_ts_exp/docs/0910实验报告_METRLA_BlockT机制消融.md"
ORDER = (
    ("official_base", "HD-TTS-AMP"),
    ("phy_only", "physical-only fine relation"),
    ("obs_only", "observation-only fine relation"),
    ("full_no_overlap_reliability", "full without q_ij overlap penalty"),
    ("fine", "fused fine graph"),
    ("coarse_no_gate", "coarse graph without reliability gate"),
    ("full_no_readout_bias", "full without readout reliability bias"),
    ("full", "full MR-MSGF"),
)


def fmt(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} +/- {stat['std']:.4f}"


def diagnostics_summary() -> dict:
    rows = [json.loads(path.read_text()) for path in sorted(DIAGNOSTICS.glob("*_diagnostics.json"))]
    if len(rows) != 10:
        raise ValueError(f"Expected 10 complete full-model diagnostic files, found {len(rows)}")
    correlations = [row["edge_q_gate_correlation_off_diagonal"]["pearson"] for row in rows]
    q_means = [row["diagnostic_moments"]["edge_common_observation_rate"]["mean"] for row in rows]
    gate_means = [row["diagnostic_moments"]["coarse_gate"]["mean"] for row in rows]
    bucket_totals: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row["mae_by_node_observation_rate"].items():
            total, count = bucket_totals.setdefault(key, [0.0, 0.0])
            bucket_totals[key] = [total + value["mae"] * value["n_nodes_across_windows"],
                                  count + value["n_nodes_across_windows"]]
    return {
        "n": len(rows),
        "q_gate_correlation": {"mean": sum(correlations) / len(correlations),
                                "min": min(correlations), "max": max(correlations)},
        "q_mean": sum(q_means) / len(q_means),
        "coarse_gate_mean": sum(gate_means) / len(gate_means),
        "bucket_mae": {key: total / count for key, (total, count) in bucket_totals.items() if count},
    }


def main() -> None:
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Run r0910_verify_stage_b.py first: {SUMMARY}")
    result = json.loads(SUMMARY.read_text())
    if result.get("status") != "passed":
        raise ValueError("Refusing to report an unverified result matrix.")
    diagnostic = diagnostics_summary()

    lines = [
        "# 0910 实验报告：METR-LA Block-T 机制消融",
        "",
        "## 1. 目的",
        "",
        "本阶段固定 METR-LA、24 步历史、12 步预测、Block-T 输入缺失及未遮蔽未来目标评估，逐项检验 MR-MSGF 的共同观测惩罚、双源细图融合、融合后区域化、区域可靠性门控和 readout reliability bias。该报告只分析通过自动校验的结果。",
        "",
        "## 2. 协议与校验",
        "",
        f"- 协议：{result['protocol']}。",
        f"- 缺失率：{', '.join(str(rate) + '%' for rate in result['rates'])}；mask seed：{', '.join(map(str, result['seeds']))}。",
        f"- 校验结果：{result['n_result_rows']} 条结果记录均包含最佳 checkpoint，HD-TTS 官方提交为 `46a8717db3802e1684e991ba0a7c0bfe43d22535`，数据 SHA256 为 `{result['data_sha256']}`。",
        "- 每个 rate/seed 内的所有变体使用同一份输入 mask；MAE 与 RMSE 均在原始有效未来目标上计算。",
        "",
        "## 3. 结果",
        "",
    ]
    for rate in result["rates"]:
        lines += [
            f"### Block-T {rate}%",
            "",
            "| 变体 | MAE | RMSE | 相对 full 的配对 MAE 差值 |",
            "|---|---:|---:|---:|",
        ]
        for key, label in ORDER:
            group = result["groups"][f"r{rate}_{key}"]
            if key == "full":
                delta = "0.0000"
            else:
                delta = fmt(group["paired_mae_minus_full"])
            lines.append(f"| {label} | {fmt(group['mae'])} | {fmt(group['rmse'])} | {delta} |")
        lines.append("")

    lines += [
        "## 4. 可靠性诊断",
        "",
        f"- 共读取 {diagnostic['n']} 个完整模型 checkpoint 的完整测试集诊断。",
        f"- 非对角边的共同观测率 $q_{{ij}}$ 与边门的 Pearson 相关系数为 {diagnostic['q_gate_correlation']['mean']:.4f}（按运行均值，范围 {diagnostic['q_gate_correlation']['min']:.4f} 至 {diagnostic['q_gate_correlation']['max']:.4f}）。",
        f"- 边共同观测率均值为 {diagnostic['q_mean']:.4f}；coarse gate 均值为 {diagnostic['coarse_gate_mean']:.4f}。",
        "- 按节点历史观测率分桶的未来 MAE："
        + "；".join(f"{key}={value:.4f}" for key, value in sorted(diagnostic['bucket_mae'].items())) + "。",
        "",
        "## 5. 读法与边界",
        "",
        "表中“相对 full 的配对 MAE 差值”为同一 mask seed 下 `variant - full`，正值表示完整模型 MAE 更低，负值表示该变体更低。该统计只描述该固定数据、掩码和训练预算下的观察结果；不以单一 seed 或单一缺失率作因果结论。",
        "",
        "共同观测惩罚消融仅关闭 `softplus(beta) log(q_ij + epsilon)` 对观测图相似度分数的约束，保留 q_ij 作为细图门控和粗图 gate 的输入。因此它测量的是边打分阶段的共同观测证据，而不是把所有 q_ij 相关特征同时删除。",
        "",
        "## 6. 产物与后续",
        "",
        f"- 自动校验摘要：[stage_b_verification_summary.json]({SUMMARY})。",
        f"- 诊断结果目录：[stage_b_diagnostics]({DIAGNOSTICS})。",
        "- 下一阶段将补齐 METR-LA Block-T 30% 主表和 Point 30/50% 对照；AQI36 自然缺失与附加 Point 条件在其独立资源 manifest 完备后执行。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

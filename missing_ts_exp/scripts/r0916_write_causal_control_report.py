#!/usr/bin/env python3
"""Write the fixed-format report after causal-control verification succeeds."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0916_causal_controls/verification_summary.json"
REPORT = ROOT / "missing_ts_exp/docs/0916实验报告_METRLA_BlockT50共观测因果控制.md"


def fmt(metric: dict) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def main() -> None:
    result = json.loads(SUMMARY.read_text())
    groups = result["groups"]
    comparisons = result["paired_variant_minus_full"]
    labels = (("official_base", "HD-TTS-AMP"),
              ("feature_only", "HD-TTS-AMP + Reliability features"),
              ("permuted_q", "MR-MSGF + Permuted-Q"),
              ("full", "Full MR-MSGF"))
    lines = [
        "# 0916 实验报告：METR-LA Block-T 50% 共观测因果控制",
        "",
        "## 协议",
        "",
        "本报告仅汇总预注册的 METR-LA、Block-T 50%、24 步历史至 12 步预测的机制控制实验。每行使用五个配对 mask/训练 seed（2024--2028）、100 epochs、每 epoch 300 个训练 batch、batch size 32，并以未遮蔽有效未来目标上的 MAE/RMSE 评估。HD-TTS-AMP 和 Full 为既有等价协议结果；仅 Feature-only 与 Permuted-Q 是本轮新增训练。",
        "",
        "## 主表",
        "",
        "| Variant | Parameters | MAE | RMSE | Delta MAE vs Full | 95% CI | paired p (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels:
        group = groups[key]
        if key in comparisons:
            test = comparisons[key]["mae"]
            delta = f"{test['mean']:.4f}"
            ci = f"[{test['ci95'][0]:.4f}, {test['ci95'][1]:.4f}]"
            p_value = f"{test['holm_p_value']:.4g}"
        elif key == "full":
            delta, ci, p_value = "--", "--", "--"
        else:
            delta, ci, p_value = "exploratory", "exploratory", "exploratory"
        parameters = group["parameter_count"]
        parameter_text = f"{parameters:,}" if parameters else "existing run not recorded"
        lines.append(f"| {label} | {parameter_text} | {fmt(group['metrics']['mae'])} | {fmt(group['metrics']['rmse'])} | {delta} | {ci} | {p_value} |")
    lines += [
        "",
        "## 配对统计与审计",
        "",
        "正 Delta 表示 Full 的 MAE 更低。置信区间为五个配对差值的双侧 95% paired t interval；两项预注册的 MAE 检验采用 Holm 校正。RMSE 作为辅助终点保留逐 seed 差值，不用于 Holm 主检验。",
        "",
    ]
    for key, label in (("feature_only", "Feature-only"), ("permuted_q", "Permuted-Q")):
        item = comparisons[key]
        lines.append(f"- **{label}**：MAE 逐 seed `variant - Full` = {item['mae']['values']}；RMSE 差值 = {item['rmse']['values']}。")
    audit = result["permutation_invariants"]
    lines += [
        f"- Permuted-Q P0 审计：重复调用一致={audit['repeatable']}；对称={audit['symmetric']}；元素多重集合保持={audit['multiset_preserved']}；范围保持={audit['range_preserved']}；不同窗口使用不同置换={audit['different_window_changes_permutation']}。",
        "- Feature-only 独立继承官方 HD-TTS，不实例化 `A_obs`、top-k、`obs_query/obs_key`、edge gate、fine/coarse 消息传播或区域门控；其仅将四个由历史 mask 和固定物理资源计算的可靠性统计量映射为 readout score bias。",
        "- Permuted-Q 使用 `Q_perm=P Q P^T`，置换由 `mask_seed`、`split_id` 和 `absolute_window_index` 的确定性 BLAKE2b 哈希生成；Full 的其余参数和路径保持不变。",
        "",
        "## 解释",
        "",
        "结果解释必须遵循预注册规则：只有 Full 同时优于这两项控制时，才可把结果表述为与“可靠性统计量本身不足”和“正确节点对共同观测语义有用”一致；若任一比较不支持，则按实验计划第 10 节收缩相应机制主张，而不重选 seed 或改动协议。",
        "",
        f"详细机器可读校验结果：[verification_summary.json]({SUMMARY})。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

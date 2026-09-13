# 0910 实验报告：METR-LA Block-T 机制消融

## 1. 目的

本阶段固定 METR-LA、24 步历史、12 步预测、Block-T 输入缺失及未遮蔽未来目标评估，逐项检验 MR-MSGF 的共同观测惩罚、双源细图融合、融合后区域化、区域可靠性门控和 readout reliability bias。该报告只分析通过自动校验的结果。

## 2. 协议与校验

- 协议：METR-LA, 24-to-12, Block-T, unmasked future MAE/RMSE, 100 epochs, 300 train batches。
- 缺失率：50%, 70%；mask seed：2024, 2025, 2026, 2027, 2028。
- 校验结果：80 条结果记录均包含最佳 checkpoint，HD-TTS 官方提交为 `46a8717db3802e1684e991ba0a7c0bfe43d22535`，数据 SHA256 为 `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`。
- 每个 rate/seed 内的所有变体使用同一份输入 mask；MAE 与 RMSE 均在原始有效未来目标上计算。

## 3. 结果

### Block-T 50%

| 变体 | MAE | RMSE | 相对 full 的配对 MAE 差值 |
|---|---:|---:|---:|
| HD-TTS-AMP | 3.3546 +/- 0.0126 | 6.8780 +/- 0.0215 | 0.0468 +/- 0.0181 |
| physical-only fine relation | 3.3292 +/- 0.0118 | 6.8605 +/- 0.0490 | 0.0213 +/- 0.0204 |
| observation-only fine relation | 3.3348 +/- 0.0161 | 6.8874 +/- 0.0587 | 0.0269 +/- 0.0231 |
| full without q_ij overlap penalty | 3.3149 +/- 0.0200 | 6.8327 +/- 0.0547 | 0.0070 +/- 0.0126 |
| fused fine graph | 3.3176 +/- 0.0111 | 6.8288 +/- 0.0455 | 0.0097 +/- 0.0134 |
| coarse graph without reliability gate | 3.3136 +/- 0.0105 | 6.8395 +/- 0.0420 | 0.0057 +/- 0.0149 |
| full without readout reliability bias | 3.3298 +/- 0.0249 | 6.8512 +/- 0.0375 | 0.0219 +/- 0.0405 |
| full MR-MSGF | 3.3079 +/- 0.0195 | 6.8154 +/- 0.0554 | 0.0000 |

### Block-T 70%

| 变体 | MAE | RMSE | 相对 full 的配对 MAE 差值 |
|---|---:|---:|---:|
| HD-TTS-AMP | 3.4650 +/- 0.0103 | 7.0927 +/- 0.0421 | 0.0191 +/- 0.0121 |
| physical-only fine relation | 3.4481 +/- 0.0146 | 7.1060 +/- 0.0246 | 0.0023 +/- 0.0194 |
| observation-only fine relation | 3.4550 +/- 0.0193 | 7.1184 +/- 0.0286 | 0.0092 +/- 0.0158 |
| full without q_ij overlap penalty | 3.4218 +/- 0.0114 | 7.0338 +/- 0.0193 | -0.0241 +/- 0.0185 |
| fused fine graph | 3.4517 +/- 0.0145 | 7.1199 +/- 0.0225 | 0.0059 +/- 0.0167 |
| coarse graph without reliability gate | 3.4330 +/- 0.0169 | 7.1067 +/- 0.0572 | -0.0128 +/- 0.0221 |
| full without readout reliability bias | 3.4215 +/- 0.0137 | 7.0756 +/- 0.0403 | -0.0243 +/- 0.0190 |
| full MR-MSGF | 3.4458 +/- 0.0102 | 7.1088 +/- 0.0384 | 0.0000 |

## 4. 可靠性诊断

- 共读取 10 个完整模型 checkpoint 的完整测试集诊断。
- 非对角边的共同观测率 $q_{ij}$ 与边门的 Pearson 相关系数为 -0.4051（按运行均值，范围 -0.6882 至 0.0113）。
- 边共同观测率均值为 0.1470；coarse gate 均值为 0.5996。
- 按节点历史观测率分桶的未来 MAE：high_ge_0.8=3.0906；low_lt_0.4=3.0303；medium_0.4_to_0.8=3.2502。

## 5. 读法与边界

表中“相对 full 的配对 MAE 差值”为同一 mask seed 下 `variant - full`，正值表示完整模型 MAE 更低，负值表示该变体更低。该统计只描述该固定数据、掩码和训练预算下的观察结果；不以单一 seed 或单一缺失率作因果结论。

共同观测惩罚消融仅关闭 `softplus(beta) log(q_ij + epsilon)` 对观测图相似度分数的约束，保留 q_ij 作为细图门控和粗图 gate 的输入。因此它测量的是边打分阶段的共同观测证据，而不是把所有 q_ij 相关特征同时删除。

## 6. 产物与后续

- 自动校验摘要：[stage_b_verification_summary.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0910_evidence_closure/stage_b_verification_summary.json)。
- 诊断结果目录：[stage_b_diagnostics](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0910_evidence_closure/stage_b_diagnostics)。
- 下一阶段将补齐 METR-LA Block-T 30% 主表和 Point 30/50% 对照；AQI36 自然缺失与附加 Point 条件在其独立资源 manifest 完备后执行。

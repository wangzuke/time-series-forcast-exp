# 0910 实验报告：METR-LA Block-T 主表扩展

## 1. 目的与协议

本阶段完成 METR-LA Block-T 30%、50%、70% 的五 seed 主表。HD-TTS-AMP、MR-MSGF 与 BiTGraph 均使用相同的 24 步历史、12 步预测、时间顺序数据划分和输入 mask；指标为原始有效未来目标上的未遮蔽 MAE/RMSE。BiTGraph 保持作者 `BiaTCGNet.Model` 架构，仅将输入数据接口接到相同协议。

自动校验通过 45 条结果：METR-LA, 24-to-12, Block-T, five paired mask seeds, unmasked future MAE/RMSE。数据 SHA256：`64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`。

## 2. 五 seed 主结果

| Block-T 缺失率 | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |
|---:|---|---:|---:|---:|
| 30% | HD-TTS-AMP | 3.2281 +/- 0.0108 | 6.6036 +/- 0.0126 | 0.0284 +/- 0.0208 |
| 30% | BiTGraph | 3.8232 +/- 0.0110 | 7.8839 +/- 0.0373 | 0.6235 +/- 0.0235 |
| 30% | MR-MSGF | 3.1997 +/- 0.0159 | 6.5840 +/- 0.0485 | 0.0000 |
| 50% | HD-TTS-AMP | 3.3546 +/- 0.0126 | 6.8780 +/- 0.0215 | 0.0468 +/- 0.0181 |
| 50% | BiTGraph | 3.9729 +/- 0.0180 | 8.1905 +/- 0.0458 | 0.6651 +/- 0.0270 |
| 50% | MR-MSGF | 3.3079 +/- 0.0195 | 6.8154 +/- 0.0554 | 0.0000 |
| 70% | HD-TTS-AMP | 3.4650 +/- 0.0103 | 7.0927 +/- 0.0421 | 0.0191 +/- 0.0121 |
| 70% | BiTGraph | 4.1363 +/- 0.0132 | 8.4812 +/- 0.0306 | 0.6905 +/- 0.0164 |
| 70% | MR-MSGF | 3.4458 +/- 0.0102 | 7.1088 +/- 0.0384 | 0.0000 |

## 3. 读法与边界

表中配对差值计算为同一 mask seed 下 `baseline - MR-MSGF`；正值表示 MR-MSGF 的 MAE 更低。所有五个 seed 使用配对 mask，但三种方法的随机初始化也固定为对应 seed。由于 BiTGraph 的作者训练日程与 HD-TTS 的 epoch/批次数不同，主结论应以统一数据、mask 与评估口径下的性能比较为限，而不把训练日程差异解释为模型机制本身。

Block-T 是本文方法更适配的场景：某节点有连续时间缺失，而同一时间的空间邻居和区域上下文通常仍部分可用。该结果不能直接外推到 Block-ST 的空间同步缺失压力场景。

## 4. 产物与下一步

- 自动校验摘要：[stage_a1_verification_summary.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0910_evidence_closure/stage_a1_verification_summary.json)。
- 后续进行 METR-LA Point 30/50% 的五 seed 对照，检验共同观测关系在非连续缺失下的表现；随后开展 AQI36 自然缺失与附加 Point 实验。

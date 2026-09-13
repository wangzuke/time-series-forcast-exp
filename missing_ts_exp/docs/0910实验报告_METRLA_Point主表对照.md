# 0910 实验报告：METR-LA Point 主表对照

## 1. 目的与协议

本阶段以独立点缺失补充 Block-T 主实验，检验共同观测关系和融合后区域化是否只依赖连续时间块故障。所有方法共享 METR-LA 的时间划分、24 步历史、12 步预测、输入 mask 与未遮蔽未来目标评估。

自动校验通过 30 条结果：METR-LA, 24-to-12, Point, five paired mask seeds, unmasked future MAE/RMSE。

## 2. 五 seed 主结果

| Point 缺失率 | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |
|---:|---|---:|---:|---:|
| 30% | HD-TTS-AMP | 3.1048 +/- 0.0074 | 6.3592 +/- 0.0154 | 0.0284 +/- 0.0113 |
| 30% | BiTGraph | 3.6374 +/- 0.0124 | 7.5260 +/- 0.0377 | 0.5610 +/- 0.0185 |
| 30% | MR-MSGF | 3.0764 +/- 0.0118 | 6.3091 +/- 0.0447 | 0.0000 |
| 50% | HD-TTS-AMP | 3.1598 +/- 0.0112 | 6.4755 +/- 0.0215 | 0.0189 +/- 0.0105 |
| 50% | BiTGraph | 3.6938 +/- 0.0049 | 7.6577 +/- 0.0289 | 0.5529 +/- 0.0053 |
| 50% | MR-MSGF | 3.1409 +/- 0.0068 | 6.4564 +/- 0.0289 | 0.0000 |

## 3. 读法与边界

正的配对差值表示 MR-MSGF 在相同 mask seed 下 MAE 更低。Point 结果是关系机制的补充验证，而非替代本文针对连续时间缺失的 Block-T 主场景；报告不将其解释为对 Block-ST 空间同步缺失的证据。

## 4. 产物与下一步

- 自动校验摘要：[stage_a2_verification_summary.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0910_evidence_closure/stage_a2_verification_summary.json)。
- 下一阶段使用 AQI36 的自然缺失、附加 Point 25% 和 50%，验证带地理空间信息的环境监测数据上的可迁移性。

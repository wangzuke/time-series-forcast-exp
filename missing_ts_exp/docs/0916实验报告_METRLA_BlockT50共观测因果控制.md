# 0916 实验报告：METR-LA Block-T 50% 共观测因果控制

## 协议

本报告仅汇总预注册的 METR-LA、Block-T 50%、24 步历史至 12 步预测的机制控制实验。每行使用五个配对 mask/训练 seed（2024--2028）、100 epochs、每 epoch 300 个训练 batch、batch size 32，并以未遮蔽有效未来目标上的 MAE/RMSE 评估。HD-TTS-AMP 和 Full 为既有等价协议结果；仅 Feature-only 与 Permuted-Q 是本轮新增训练。

## 主表

| Variant | Parameters | MAE | RMSE | Delta MAE vs Full | 95% CI | paired p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| HD-TTS-AMP | existing run not recorded | 3.3546 +/- 0.0126 | 6.8780 +/- 0.0215 | exploratory | exploratory | exploratory |
| HD-TTS-AMP + Reliability features | 378,415 | 3.3537 +/- 0.0119 | 6.9010 +/- 0.0141 | 0.0459 | [0.0286, 0.0632] | 0.003641 |
| MR-MSGF + Permuted-Q | 407,447 | 3.3136 +/- 0.0206 | 6.8444 +/- 0.0572 | 0.0057 | [-0.0070, 0.0184] | 0.2815 |
| Full MR-MSGF | existing run not recorded | 3.3079 +/- 0.0195 | 6.8154 +/- 0.0554 | -- | -- | -- |

## 配对统计与审计

正 Delta 表示 Full 的 MAE 更低。置信区间为五个配对差值的双侧 95% paired t interval；两项预注册的 MAE 检验采用 Holm 校正。RMSE 作为辅助终点保留逐 seed 差值，不用于 Holm 主检验。

- **Feature-only**：MAE 逐 seed `variant - Full` = [0.0317838191986084, 0.03309750556945801, 0.05365610122680664, 0.06473803520202637, 0.04610705375671387]；RMSE 差值 = [0.06522943503318235, 0.10239692786837651, 0.10322066639316585, 0.14570733421871385, 0.011690782570943803]。
- **Permuted-Q**：MAE 逐 seed `variant - Full` = [0.005170345306396484, -0.005857706069946289, -0.0011250972747802734, 0.00964498519897461, 0.020618438720703125]；RMSE 差值 = [0.03663578399217471, -0.021803909432807522, 0.04263431552385466, 0.05781895756531341, 0.029655882642559916]。
- Permuted-Q P0 审计：重复调用一致=True；对称=True；元素多重集合保持=True；范围保持=True；不同窗口使用不同置换=True。
- Feature-only 独立继承官方 HD-TTS，不实例化 `A_obs`、top-k、`obs_query/obs_key`、edge gate、fine/coarse 消息传播或区域门控；其仅将四个由历史 mask 和固定物理资源计算的可靠性统计量映射为 readout score bias。
- Permuted-Q 使用 `Q_perm=P Q P^T`，置换由 `mask_seed`、`split_id` 和 `absolute_window_index` 的确定性 BLAKE2b 哈希生成；Full 的其余参数和路径保持不变。

## 解释

结果解释必须遵循预注册规则：只有 Full 同时优于这两项控制时，才可把结果表述为与“可靠性统计量本身不足”和“正确节点对共同观测语义有用”一致；若任一比较不支持，则按实验计划第 10 节收缩相应机制主张，而不重选 seed 或改动协议。

详细机器可读校验结果：[verification_summary.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0916_causal_controls/verification_summary.json)。

# 0910 审计：GinAR 官方实现与统一协议可比性

## 结论

GinAR 保留为最接近的 related work，不进入本轮 METR-LA Block-T / Point 主表。原因不是模型能力判断，而是其官方缺失协议与本文已锁定的统一协议不等价；为得到可比较结果而修改 GinAR 的数据生成和输入接口，会违反本轮“baseline 使用官方开源实现、不得自制简化或改写模型协议”的约束。

## 核验对象

- 官方源目录：`external_repro/official/ginar`。
- README：`README.md` 明确说明随机生成 $M$ 个缺失变量，并将这 $M$ 个变量的输入特征置零。
- 数据预处理：`data/main_PEM.py` 的 `mask_id_*` 在变量维随机采样，随后执行 `feature_data[:, mask_id_*] = 0`。
- 训练入口：`main_ginar.py` 选择预生成的 `train_x_mask_25/50/75/90`，拼接未来标签后训练；默认 `history_seq_len=12`、`future_seq_len=12`。

## 与本文协议的差异

| 维度 | GinAR 官方代码 | 本轮统一协议 |
|---|---|---|
| 缺失单位 | 被抽中的变量在整个历史时间轴上置零 | 每个节点存在 Point 或连续 Block-T 时间缺失 |
| 历史长度 | 官方 PEMS 脚本为 12 | 固定为 24 |
| 可用性信号 | 被遮蔽变量以零值表示 | 显式输入 `input_mask`，保留真实零值与缺失的区分 |
| 数据管线 | 预生成 NumPy 数组并附带自身划分 | METR-LA 统一时间划分、配对 mask、未遮蔽未来评估 |
| 评估目的 | variable missing 预测 | 时间轴历史观测缺失下的时空预测 |

## 决策

1. 不将 GinAR 数值放入 HD-TTS、BiTGraph、MR-MSGF 的主表或 Point 表。
2. 论文 related work 保留其“预定义图与自适应图在变量缺失下互补”的方法启发，并明确场景差异。
3. 若未来单独开展 variable-missing 附录，应完整复用其官方脚本、12 步历史和变量遮蔽协议；该实验必须单列，不与当前 Block-T 数字横向比较。

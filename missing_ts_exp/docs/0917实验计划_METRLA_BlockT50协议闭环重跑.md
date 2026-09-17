# 0917 实验计划：METR-LA Block-T 50% 协议闭环重跑

## 1. 目的与边界

本计划响应审稿意见，对上一轮 `0916实验报告_METRLA_BlockT50共观测因果控制.md` 做**协议闭环重跑**。上一轮的 Full 来自既有运行，而 Feature-only 与 Permuted-Q 来自新代码；即使数据、mask 和训练预算相同，也不能作为三者严格的同版代码比较。本轮的唯一目标是让 Full-observed、Feature-only 和 Full-permuted-Q 由同一冻结代码、同一 datamodule 输入通路、同一优化器和同一 checkpoint 规则重新训练。

本计划不是为了追求 Permuted-Q 显著性。无论结果是否显著，均按第 8 节的预先规定解释，不调整 seed、训练轮数、置换策略或超参数。HD-TTS-AMP 保留为论文主基线，但不纳入本轮重跑；展示时可与这组三变体同表呈现，并标注其来自既有主基线运行。

本轮只做 METR-LA、Block-T 50%、`24 -> 12`，不新增数据集、缺失模式、基线或超参数搜索。0916 的控制结果保留为先前版本记录，不得写入本轮汇总或替换本轮的 seed 级数据。

## 2. 当前差异与必须修正的实现

当前有效 runner 配置为：

| 变体 | 正式模型选择 | `q_mode` | 核心变化 |
|---|---|---|---|
| Full-observed | `--model mrmsgf --mr_variant full` | `observed` | 使用真实窗口级 `Q` |
| Feature-only | `--model official_base_feature_only` | 不适用 | 仅四项可靠性统计量进入 readout 偏置 |
| Full-permuted-Q | `--model mrmsgf --mr_variant full` | `permuted_pairwise` | 只以置换后的 `Q` 替换所有 `q` 使用位置 |

旧的 `r0916_build_critical_control_cmds.py` 使用了 `feature_only`/`shuffled`，与现 runner 的有效参数不一致；它不作为本轮命令来源。新脚本均以本计划中的三种有效配置生成。

### 2.1 保对角线的 Permuted-Q

在 `hdtts_official_reliable.py` 的 `permuted_pairwise_q` 中，先按每个窗口的确定性节点置换计算

`Q_perm = P Q P^T`，

再执行 `diag(Q_perm) = diag(Q)`。因此每个节点的自身观测率 `q_ii = r_i` 保留，只有非对角的节点对共同观测关系被置换。`Q_perm` 必须继续替换以下全部 q 相关使用：`log(q_ij)` 可靠性惩罚、`edge_gate` 特征、物理邻居 `qbar_i`、coarse gate/区域覆盖/readout 的 q 派生特征，以及诊断中实际进入模型的 q。

保留原有安全规则：历史窗口完全无观测的源节点令 `A_obs[i,:]=0`，并在细图融合时回退到物理图。该规则在 Full-observed 和 Full-permuted-Q 中必须完全相同。

### 2.2 统一窗口元数据输入通路

`make_datamodule` 必须为**三个**变体都把 `window_index` 和 `split_id` 放入 input map，且这些张量、数据切分、`x/u/input_mask/target` 对三个变体逐元素一致。仅 `permuted_pairwise` 在 forward 中读取它们以从 `(mask_seed, split_id, absolute_window_index)` 的稳定哈希获得节点置换；Full-observed 和 Feature-only 接收但不消费这两个可选参数。

此约束排除“是否传递元数据”作为 Permuted-Q 与其他变体之间的混杂因素。Feature-only 仍必须是独立的官方 HD-TTS 加 `Linear(4,L)` readout 可靠性偏置实现，禁止实例化或调用观测图、图融合、细/粗图传播、coarse gate 或区域消息传播。

### 2.3 冻结、manifest 与输出隔离

P0 通过后，将相关代码提交为单一冻结提交；正式任务只从该提交启动。runner 为每个结果写 manifest JSON（并生成汇总 CSV），至少记录：

- Git `HEAD`、完整未提交改动状态和受控文件 diff 的 SHA256；
- 模型类、参数量、`q_mode`、`mr_variant`、mask/train seed；
- 是否传递窗口元数据、是否实际读取元数据、零观测物理图回退、dispersion 数值保护；
- 数据、mask、物理区域分配的 SHA256；
- `24 -> 12`、优化器/学习率/weight decay、训练 budget、batch size、patience、`graph_topk`、`reliability_beta`；
- 最佳 checkpoint 完整路径、最佳 validation MAE、对应 epoch；
- masked test 与 unmasked future test 的 MAE、MSE、RMSE。

正式输出固定为 `missing_ts_exp/results/r0917_protocol_closed/`，日志固定为 `missing_ts_exp/logs/r0917_protocol_closed/`，不得覆盖 0916 目录。Full-observed 和 Full-permuted-Q 的 manifest 除 `q_mode`、置换审计字段及结果值外必须完全相同，且参数量严格相等。

## 3. 固定实验协议

| 项目 | 固定值 |
|---|---|
| 数据 | METR-LA |
| 缺失条件 | Block-T 50% |
| 输入 / 预测长度 | 24 / 12 |
| 配对 seed | 2024, 2025, 2026, 2027, 2028 |
| 每个 seed 的 mask | `resources/r0905_blockt/masks/METR-LA/block_t_r50_seed${SEED}_observed.npy` |
| Full 图配置 | 固定物理图、固定区域划分、`graph_topk=8`、`reliability_beta=1.0` |
| 优化器 | AdamW，`lr=1e-3`，`weight_decay=0.01` |
| 训练 | 100 epochs，每 epoch 300 batches，batch size 32，patience 20 |
| checkpoint | validation MAE 最优 checkpoint |
| 主评估 | 未遮蔽真实未来标签上的 MAE |
| 辅助评估 | 未遮蔽真实未来标签上的 RMSE |

训练初始化 seed 与 mask seed 相同。所有任务使用相同 METR-LA 时间切分、归一化、预处理、评估器与 checkpoint 选择规则。

## 4. P0：实现与轻量验收

正式训练前必须通过独立测试脚本，写入 `results/r0917_protocol_closed/p0_audit.json`：

1. 对同一 `(mask_seed, split_id, window_index)`，两次 `Q_perm` 完全相同。
2. `Q_perm` 对称，`diag(Q_perm) == diag(Q)` 严格成立，非对角元素的排序多重集合保持不变。
3. 不同绝对窗口通常产生不同置换。
4. Full-observed、Feature-only、Full-permuted-Q 的 batch 均含相同的 `window_index/split_id`，并验证其余 input tensor 与切分一致。
5. Full-observed 与 Full-permuted-Q 对人为构造的零观测节点得到相同的物理图回退；输出有限。
6. Feature-only 模块审计不含 `obs_query`、`obs_key`、top-k 观测图、`edge_gate`、fine/coarse message passing 或 `coarse_gate`。
7. 三模型完成 CPU 单 batch forward/backward 与 checkpoint reload；输出、梯度和指标无 NaN/Inf。

P0 不训练正式模型；任何失败均在修复后重新跑 P0，尚未通过不得生成正式命令。

## 5. P1：协议烟测

在 seed 2024 上对三种变体各运行一次 `1 epoch / 2 train batches / 2 eval batches` 的 GPU 烟测。检查 JSON、checkpoint、manifest 字段、统一窗口元数据、Full/Permuted-Q 参数量相等和 SHA 配对。烟测结果写入 `results/r0917_protocol_closed/smoke/`，严禁进入正式统计。

## 6. P2：15 个正式训练任务

每个 seed 运行以下三条任务，共 `3 x 5 = 15` 个。建议标签：

| 组 | 标签 | 数量 |
|---|---|---:|
| Full-observed | `full_observed_METRLA_blockt50_seed${SEED}` | 5 |
| Feature-only | `feature_only_METRLA_blockt50_seed${SEED}` | 5 |
| Full-permuted-Q | `full_permuted_q_METRLA_blockt50_seed${SEED}` | 5 |

Full-observed 命令骨架：

```bash
PY=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
RUNNER=missing_ts_exp.src.training.run_mrmsgf_official_controlled
MASK=missing_ts_exp/resources/r0905_blockt/masks/METR-LA/block_t_r50_seed${SEED}_observed.npy

$PY -m $RUNNER \
  --model mrmsgf --mr_variant full --q_mode observed --mask_seed ${SEED} \
  --mask_path $MASK --seed ${SEED} \
  --epochs 100 --train_batches 300 --patience 20 --batch_size 32 \
  --graph_topk 8 --reliability_beta 1.0 \
  --out_dir missing_ts_exp/results/r0917_protocol_closed \
  --tag full_observed_METRLA_blockt50_seed${SEED}
```

Permuted-Q 仅将 `--q_mode observed` 替换为 `--q_mode permuted_pairwise`。Feature-only 使用 `--model official_base_feature_only`，不传递 `mr_variant`；其他数据、mask、seed、训练参数和输出规范不变。所有正式作业每卡一个进程；失败只能以原始命令重启并保留失败日志。

## 7. P3：验证、统计与报告

验证脚本必须只接受本轮 15 个 JSON，要求每个变体恰有五个 seed、checkpoint 存在、所有指标有限、每个 seed 的 data/mask/region SHA 相同、三模型 Git 状态相同且配置审计通过。它还必须逐字段比较 Full-observed 和 Full-permuted-Q 的 manifest：仅允许 `q_mode`、置换字段、checkpoint 路径、训练轨迹和结果指标不同。

统计以每个 seed 的配对差 `d_s = MAE_variant,s - MAE_Full,s` 为单位。对 Feature-only 和 Permuted-Q 分别报告：

- MAE/RMSE 的 mean +/- sample std；
- 五个原始 `d_s` 与平均 Delta MAE；
- 双侧 paired t-test 的 95% t 置信区间与未校正 p；
- 两项预先规划 MAE 比较的 Holm 校正 p。

正式报告为 `missing_ts_exp/docs/0917实验报告_METRLA_BlockT50协议闭环.md`，附机器可读的 `verification_summary.json`、15 行 manifest CSV 和 P0 审计链接。报告清楚区分“同版闭环重跑三变体”与“既有 HD-TTS-AMP 主基线”。

## 8. 入稿门槛与结果解释

只有同时满足下列条件，才将本轮三行因果对照写入论文：同一冻结 Git 状态；15/15 完整任务；Full 与 Permuted-Q 的逐字段配置审计通过；每个 seed 的 SHA 配对完整；主文同时报告 Permuted-Q 的结果、置信区间和非显著性（若存在）。

- 若 Full 显著优于 Feature-only：可表述为“完整架构的收益不能由直接注入缺失统计特征解释”。
- 若 Full 未显著优于 Permuted-Q：必须表述为“当前控制不足以证明节点对身份是主要收益来源”，不得使用强机制因果措辞。
- 无论 Permuted-Q 的结果如何，均不继续改 seed、训练轮数、置换策略或超参数。

## 9. 完成标准

1. P0 与 P1 全部通过并留存审计产物；
2. 15 个正式训练使用同一冻结提交和统一窗口元数据输入；
3. 验证器通过 15/15 记录、完整 manifest 对比与配对 SHA 检查；
4. 统计和实验报告落盘，逐 seed 数据可复算；
5. 论文只使用通过本计划门槛的闭环结果，并如实呈现 Permuted-Q 的结论边界。

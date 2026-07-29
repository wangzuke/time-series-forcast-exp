# 实验计划0723：P-GUTS + Coarse Graph 插补实验

> 计划日期：2026-07-23  
> 论文依据：`papers/missing-value-forecasting/P-GUTS_kdd2025.pdf`  
> 承接结果：`missing_ts_exp/docs/0721_PGUTS_HDPGUTS实验报告.md`  
> 核心问题：0721 forecasting 实验中，`coarse graph + fixed fusion` 是 HD-PGUTS 里表现最好的变体；本计划检验该模块迁回 P-GUTS 原始任务，也就是 spatiotemporal imputation，是否仍然有效。

---

## 一、实验动机

0721 预测实验显示：

```text
P-GUTS [3,6]          = full graph + temporal3/6 + fixed fusion
w/o adaptive fusion   = full graph + coarse graph + temporal3/6 + fixed fusion
```

在 `T_out=24` 的高块缺失预测中，`w/o adaptive fusion` 在 8/8 个关键条件下优于 P-GUTS `[3,6]`，平均 MAE 改善约 `-0.97%`。这个结果说明 coarse graph 分支可能提供了有用的空间低通先验。

但 forecasting 只是 P-GUTS 论文中提到的 imputation 特例。要判断 coarse graph 是否真是一个可保留模块，必须回到 P-GUTS 原本的插补任务上验证：

> 在窗口内部存在缺失值时，加入 coarse graph 分支是否比原始 P-GUTS 更好？

---

## 二、从 P-GUTS 论文中继承的实验协议

P-GUTS 论文的主任务是 spatiotemporal imputation。论文实验设计中与本计划最相关的设置如下。

| 项目 | 论文设置 | 本计划采用方式 |
|---|---|---|
| 任务 | 输入含缺失的时空序列，预测窗口内缺失位置 | 采用同样的窗口内插补任务 |
| 数据集 | METR-LA、PEMS-BAY、AQI、AQI36 | 首轮只跑 METR-LA、PEMS-BAY；后续再考虑 AQI/AQI36 |
| 窗口 | 交通数据窗口 `T=24`，stride=1 | 首轮采用 `T=24`，stride=1 |
| 缺失模式 | 交通数据使用 BLOCK mask，模拟连续传感器故障；额外 drop 5% uniform values | 必须优先复现该 BLOCK 协议 |
| BLOCK 参数 | 主 benchmark 固定使用官方交通 BLOCK mask；Table 3 鲁棒性复用同一个 checkpoint，只替换测试 `eval_mask` | 主训练固定 `p_fault=0.0015, p_noise=0.05`；鲁棒性不重训，使用 `run_inference.py` 评估 `p_fault=0.05/0.10/0.15, p_noise=0` |
| 指标 | MAE | MAE 主指标，RMSE/MAPE 辅助记录 |
| 随机性 | 论文使用 3 random initializations | 本轮为控制耗时，使用 seed=1/2 两个随机种子 |
| P-GUTS pooling | METR-LA、PEMS-BAY 的 `f*=3`，Table 2 中 `[3,6]` 是交通数据最优/并列最优组合 | 本轮复现和 CG 对照都只保留 `[3,6]` |
| 论文 baseline | GRIN、SPIN、SPIN-H、CSDI、SSSD、MIDM | 本计划主对照是 P-GUTS；外部 baseline 只作为补充引用或后续统一评估 |

论文还有两个重要结论需要继承：

1. P-GUTS 主表中 `[3,6]` 是交通数据上最优或并列最优的尺度组合，因此本轮直接以 `[3,6]` 作为唯一复现基线和 CG 对照基线。
2. 论文 Table 4 的极端缺失下单尺度 `[3]` 或 `[6]` 可能更强，但这属于另一个问题。本轮不再覆盖 Table 4，避免把 coarse graph 主对照做散。

---

## 三、模型命名与对照关系

本计划把加入 coarse graph 但仍使用 fixed fusion 的插补模型称为：

```text
CG-P-GUTS = Coarse-Graph P-GUTS
```

首轮不使用 adaptive gate，因为 0721 forecasting 已经显示 adaptive gate 没有释放 coarse graph 的收益。

### 3.1 主模型

| 名称 | 时间分支 | 空间分支 | 融合 | 作用 |
|---|---|---|---|---|
| P-GUTS `[3,6]` | temporal pool 3 + 6 | full graph / space-time branch | fixed concat/MLP | 多尺度 P-GUTS 基线 |
| CG-P-GUTS `[3,6]` | temporal pool 3 + 6 | full graph + coarse graph | fixed concat/MLP | 主候选模型 |

关键对照是：

```text
CG-P-GUTS [3,6] vs P-GUTS [3,6]
```

这组对照只改变 coarse graph 分支，时间尺度、融合方式、mask、seed、窗口、训练配置均保持一致。执行时，P-GUTS `[3,6]` baseline 由论文复现实验中的对应命令提供，coarse graph 主实验只额外运行 CG-P-GUTS，避免同一 baseline 重复训练。

### 3.2 本轮不做模块消融

本轮只验证一个新增模块：

```text
coarse graph branch
```

因此正式实验只比较：

| 模型 | graph_variant | 含义 |
|---|---|---|
| P-GUTS | `full_only` | 官方 P-GUTS 原始图路径 |
| CG-P-GUTS | `full_plus_coarse` | 在 P-GUTS 上只增加 coarse graph 分支 |

`coarse_only`、`none_graph`、adaptive gate 等消融不放入本轮，避免把问题做散。本轮先判断 coarse graph 这个单模块是否值得继续。

---

## 四、插补任务的数据与 mask 设计

### 4.1 论文复现 mask

为了和 P-GUTS 论文可比，需要沿用官方代码中的 traffic BLOCK mask：

```text
window = 24
stride = 1
block length S ~ U(12, 48)
official training parameter p_fault = 0.0015
extra uniform drop p_noise = 0.05
seed = 1, 2
```

鲁棒性实验不是为不同缺失强度重新训练模型，而是复用同一个 P-GUTS `[3,6]` checkpoint，只替换测试集 `eval_mask`：

```text
python -m experiments.run_inference ...
p_fault = 0.05, 0.10, 0.15
p_noise = 0
test_mask_seed = [6043, 2043, 3043, 4043, 5043]
```

注意：交通数据原始文件可能已有少量真实缺失。训练和评估应区分三种 mask：

| mask | 含义 |
|---|---|
| `true_observed_mask` | 原始数据中真实可观测的位置 |
| `artificial_observed_mask` | 人工 mask 后仍输入模型的位置 |
| `eval_mask` | 原始可观测但被人工遮掉的位置，即 `true_observed_mask & (1 - artificial_observed_mask)` |

MAE 只能在 `eval_mask=1` 的位置计算，不能把原始真实缺失位置计入监督。

### 4.2 0721 mask 兼容实验

为了和上一轮 forecasting 结果形成内部联系，可以补一组 0721 mask bundle 上的插补实验：

```text
dataset/0721_missing_masks/
mask_type = point, block_t, block_st
rate = 50%, 70%, 90% for block_t/block_st; 50%, 70% for point
```

这组结果不能直接对齐 P-GUTS 论文表 2 / 表 3，但有两个价值：

1. 它能检查 coarse graph 在 Block-ST 这种空间-时间联合缺失下是否比纯 Block-T 更有用。
2. 它能和 0721 forecasting 报告使用同一套 mask 语义、同一套数据路径和 sha256 记录。

---

## 五、实验阶段

### Phase 0：官方 P-GUTS 代码固定与改造基线

本轮实验必须基于官方 P-GUTS 代码，不再使用 0721 的 P-GUTS-style 轻量 forecasting 实现作为主实验代码。

官方仓库已经 clone 到：

```text
external_repro/pguts/
```

当前固定的官方 commit：

```text
f8a26162de2e8d775bfcbb9bc714746fb5f8db30
```

本轮改造只在官方代码上进行，主要改动位置：

```text
external_repro/pguts/code/models/pguts.py
external_repro/pguts/experiments/run_imputation.py
external_repro/pguts/config/imputation/r0723/
```

其中 `pguts.py` 新增 `CoarseGraphBranch`，并为官方 `PGUTS` 增加可选参数：

```text
graph_variant = full_only / full_plus_coarse / coarse_only / none_graph
coarse_factor = 4
```

默认 `graph_variant=full_only` 等价于原始 P-GUTS；`graph_variant=full_plus_coarse` 是本轮主候选 CG-P-GUTS。虽然代码预留了 `coarse_only` 和 `none_graph`，但本轮正式实验不使用它们。

### Phase 1：启动 smoke

先做 4 条 smoke，只验证官方代码路径和 coarse graph 改造能否正常训练：

```text
输入: X_window_observed, M_window
输出: Y_hat_window
loss: MAE(Y_hat_window, Y_window) only on eval_mask
```

smoke 矩阵：

| 数据集 | mask | 缺失强度 | pooling | 模型 |
|---|---|---:|---|---|
| METR-LA | paper BLOCK | `p_fault=0.0015` | `[3,6]` | P-GUTS、CG-P-GUTS |
| PEMS-BAY | paper BLOCK | `p_fault=0.0015` | `[3,6]` | P-GUTS、CG-P-GUTS |

共 4 条，跑 5 epoch 即可。

验收：

1. 无 NaN。
2. `eval_mask` 中有效监督点数大于 0。
3. P-GUTS 与 CG-P-GUTS 的 `graph_variant` 不同：`full_only` vs `full_plus_coarse`。
4. 官方 logdir 中保存 `config.yaml`，其中必须能看到 `factor_t`、`graph_variant`、`p_fault`、`p_noise`、`window`、`stride`。

### Phase 2：论文复现实验

先完整跑交通数据上的 P-GUTS 论文复现。这里不加入 coarse graph，只使用 `graph_variant=full_only`。

复现内容：

| 论文表格 | 数据集 | 模型 / pooling | 缺失设置 |
|---|---|---|---|
| Table 2 traffic 部分 | METR-LA、PEMS-BAY | P-GUTS `[3,6]` | `p_fault=0.0015`, `p_noise=0.05` |
| Table 3 traffic 鲁棒性 | METR-LA、PEMS-BAY | 同一个 P-GUTS `[3,6]` checkpoint | 训练固定 `p_fault=0.0015, p_noise=0.05`；测试 inference mask 为 `p_fault=0.05/0.10/0.15, p_noise=0` |
| Table 4 extreme missing | METR-LA、PEMS-BAY | 不纳入本轮 | 后续单独实验 |

去掉重复命令后，复现命令共：

```text
2 datasets × 2 seeds × 1 Table2 best setting = 4 training runs
```

这一步的目标是先确认官方代码在本机上能复现论文趋势：

```text
P-GUTS [3,6]
```

Table 3 鲁棒性在 P-GUTS `[3,6]` checkpoint 完成后单独执行 inference：

```text
2 datasets × 2 seeds × 3 test failure probabilities = 12 inference evaluations
```

这些 inference 评估只替换测试 `eval_mask`，不能重新训练模型。已经用 `run_imputation.py --p-fault ...` 训练出来的鲁棒性结果属于误跑结果，只能归档备注，不能进入正式 Table 3。

### Phase 3：只加入 coarse graph 的主实验

本轮新增模块实验只比较 P-GUTS `[3,6]` 和 CG-P-GUTS `[3,6]`。同一数据集内 P-GUTS / CG-P-GUTS 使用相同 batch size；不同数据集按显存占用分别设置。注意：P-GUTS `[3,6]` baseline 已包含在 Phase 2 复现实验中，因此 Phase 3 命令文件只运行 CG-P-GUTS，报告阶段再把二者配对比较：

```text
2 datasets × 1 new model × 2 seeds = 4 additional runs
```

注意：Phase 2 和 Phase 3 保留官方 `split_batch_in` 机制。`split_batch_in` 用于 micro-batch + gradient accumulation：micro-batch 为 `batch_size // split_batch_in`，等效 batch 仍为命令行 `batch_size`。官方每个 epoch 约处理 `batch_size × batches_epoch = 8 × 300 = 2400` 个窗口样本。本轮按数据集放大 batch size，并同步缩小 `batches_epoch`，让每个 epoch 的样本量接近论文设置：METR-LA 使用 `batch_size=128, batches_epoch=19`，PEMS-BAY 使用 `batch_size=256, batches_epoch=10`。同一数据集内所有 P-GUTS / CG-P-GUTS 对照保持相同设置，避免模型比较时混入训练量变量。

```text
METR-LA  P-GUTS [3,6]      graph_variant=full_only         batch_size=128  batches_epoch=19
METR-LA  CG-P-GUTS [3,6]   graph_variant=full_plus_coarse  batch_size=128  batches_epoch=19
PEMS-BAY P-GUTS [3,6]      graph_variant=full_only         batch_size=256  batches_epoch=10
PEMS-BAY CG-P-GUTS [3,6]   graph_variant=full_plus_coarse  batch_size=256  batches_epoch=10
```

这一步只回答一个问题：

```text
在官方 P-GUTS 插补代码中，只加入 coarse graph 分支是否带来收益？
```

报告格式：

| 数据集 | P-GUTS `[3,6]` | CG-P-GUTS `[3,6]` | ΔMAE |
|---|---:|---:|---:|
| METR-LA | MAE ± std | MAE ± std | % |
| PEMS-BAY | MAE ± std | MAE ± std | % |

### Phase 4：后续可选实验

如果 Phase 3 有正向信号，再考虑下一轮补 0721 mask bundle：

```text
dataset = METR-LA, PEMS-BAY
mask_type = block_t, block_st
rate = 70%, 90%
models = P-GUTS [3,6], CG-P-GUTS [3,6]
seed = 1,2,3
```

重点比较：

- coarse graph 在 Block-ST 上是否比 Block-T 更有效。
- 这是否与 0721 forecasting 报告中 coarse graph 对空间缺失更敏感的分析一致。

---

## 六、实现要求

### 6.1 模型改造

本轮不新增 `missing_ts_exp/src/models/pguts_coarse_imputer.py` 这类本地轻量模型文件。那些文件名属于早期“在 0721 P-GUTS-style 实现上快速验证”的设想，现在已经作废。

本轮必须直接改官方 P-GUTS 代码：

```text
external_repro/pguts/code/models/pguts.py
external_repro/pguts/experiments/run_imputation.py
external_repro/pguts/config/imputation/r0723/
```

已经完成的改造：

```text
PGUTS graph_variant=full_only          原始 P-GUTS
PGUTS graph_variant=full_plus_coarse   CG-P-GUTS 主候选
```

辅助脚本实际采用以下命名：

```text
missing_ts_exp/scripts/r0723_build_official_pguts_cg_cmds.py
missing_ts_exp/scripts/r0723_run_official_pguts_cg.sh
missing_ts_exp/scripts/r0723_check_official_pguts_env.py
missing_ts_exp/scripts/r0723_create_official_pguts_env.sh
```

还需要补的脚本是结果汇总脚本：

```text
missing_ts_exp/scripts/r0723_collect_official_pguts_cg_results.py
```

它负责从官方 P-GUTS 的 Lightning log / checkpoint 目录 / `output.pt` 中提取测试 MAE、MSE、MRE、运行配置，并生成：

```text
missing_ts_exp/results/0723_official_pguts_coarse_imputation/csv/main_results.csv
missing_ts_exp/results/0723_official_pguts_coarse_imputation/csv/reproduce_results.csv
missing_ts_exp/results/0723_official_pguts_coarse_imputation/csv/coarse_results.csv
```

训练目标、窗口内插补 loss、`eval_mask` 逻辑继续使用官方 `experiments/run_imputation.py` 和 `SPINImputer`，不另写一套本地 imputer，以保证和 P-GUTS 原论文代码路径一致。

### 6.2 coarse graph 构造

首版 coarse graph 使用官方代码内新增的 deterministic coarse assignment，也就是按节点顺序分组后在 coarse nodes 上做 Transformer 编码再 unpool 回原节点。报告中必须说明这是第一版工程实现。

复盘修正：本轮已经完成的 CG-P-GUTS 实验严格来说不是“基于交通空间图的 coarse graph”，而是“按节点编号连续分组的 coarse branch”。当前实现没有读取传感器坐标、距离图或邻接矩阵来决定 coarse node membership，因此它只能回答“加入一个编号连续分组的 coarse 聚合分支是否有效”，不能证明真正的 graph-aware coarsening 有效。后续报告和结论必须按这个口径表述。

更推荐做两种 coarse graph：

| 方式 | 说明 | 优先级 |
|---|---|---|
| distance clustering | 用 DCRNN 距离图聚类相邻传感器 | 高 |
| contiguous assignment | 按节点顺序分组，作为快速 smoke | 中 |

distance clustering 不放入本轮正式实验，避免本轮同时引入两个变量。若 coarse graph 有正向信号，下一轮再把 coarse graph 构造方式作为单独实验。

### 6.2.1 复盘后的追加实验：distance-greedy graph-aware coarse

由于首轮 `full_plus_coarse` 实际使用的是编号连续分组，不能证明空间图粗化有效，因此追加一组 graph-aware coarse 实验。追加实验不重跑 P-GUTS baseline，只复用已经完成的 P-GUTS `[3,6]` 复现实验结果，并新增：

```text
CG-P-GUTS-distance [3,6]
graph_variant = full_plus_coarse
coarse_method = distance_greedy
coarse_factor = 4
```

构造方式：

1. 读取官方 TSL cache 中的道路距离矩阵：

```text
MetrLA/metr_la_dist.npy
PemsBay/pems_bay_dist.npy
```

2. 将有向距离矩阵用 `min(D, D^T)` 对称化。
3. 按有限距离连接度优先选择 cluster center。
4. 每个 center 选择最近的未分配传感器，最多组成 `coarse_factor=4` 个节点的 coarse group。
5. 保存 `node_to_coarse` 到：

```text
missing_ts_exp/results/0723_official_pguts_coarse_imputation/coarse_assignments/
```

追加实验矩阵：

| 数据集 | 模型 | coarse_method | seed | batch_size | batches_epoch |
|---|---|---|---|---:|---:|
| METR-LA | CG-P-GUTS-distance `[3,6]` | `distance_greedy` | 1, 2 | 128 | 19 |
| PEMS-BAY | CG-P-GUTS-distance `[3,6]` | `distance_greedy` | 1, 2 | 256 | 10 |

新增命令文件：

```text
missing_ts_exp/scripts/r0723_official_pguts_graph_aware_cg_cmds.txt
```

这一轮的核心比较应写成：

```text
P-GUTS [3,6]
vs CG-P-GUTS-contiguous [3,6]
vs CG-P-GUTS-distance [3,6]
```

解释目标：

1. 如果 `distance_greedy` 优于 `contiguous`，说明真正使用交通空间邻近关系更关键。
2. 如果二者都优于 P-GUTS，说明 coarse 分支本身值得继续，后续再优化构造方式。
3. 如果只有 `contiguous` 有收益，必须警惕收益来自编号排序或额外参数容量，而不是空间图粗化。
4. 如果 `distance_greedy` 在 PEMS-BAY 上缓解退化，说明上一轮 PEMS-BAY 问题很可能来自错误分组。

### 6.3 结果字段

每条结果至少保存：

```text
run_id
model
dataset
task = imputation
mask_protocol = paper_block / r0721_bundle
failure_probability_label
p_fault
extra_uniform_drop
mask_type
target_missing_rate
eval_missing_rate
missing_seed
window
stride
pooling_factors
graph_variant
coarse_graph_method
coarse_ratio
seed
batch_size
learning_rate
epochs
patience
MAE
RMSE
train_time_sec
epoch_time_sec
gpu_peak_mb
checkpoint_path
log_path
```

---

## 七、成功标准

### 7.1 最低成功标准

满足以下条件即可认为 coarse graph 值得继续：

1. `CG-P-GUTS [3,6]` 在 METR-LA 或 PEMS-BAY 至少一个数据集上优于 `P-GUTS [3,6]`。
2. 没有出现某个数据集上明显退化超过 `+2% MAE`。
3. 三种子结果不是由单 seed 偶然驱动。

### 7.2 强成功标准

满足以下条件可以把 CG-P-GUTS 作为新的主线候选：

1. `CG-P-GUTS [3,6]` 在 METR-LA 和 PEMS-BAY 都优于 `P-GUTS [3,6]`。
2. 复现实验中 P-GUTS `[3]`、`[3,6]` 以及极端缺失补充 `[6]` 的相对趋势与论文一致。
3. CG-P-GUTS 的训练时间和显存增幅可接受，最好低于 `+30%`。

### 7.3 失败但有信息量的情况

| 结果 | 解释 |
|---|---|
| 编号连续分组 CG-P-GUTS 出现收益 | 只能说明 coarse 聚合分支可能有信号，不能说明空间图粗化有效；必须补 distance clustering / adjacency clustering 实验 |
| CG-P-GUTS 在插补上无提升，但 forecasting 有提升 | coarse graph 可能更适合未来预测中的空间先验，而不适合窗口内重建 |
| CG-P-GUTS 在插补上无提升，但 forecasting 有提升 | coarse graph 可能更适合未来预测中的空间先验，而不适合窗口内重建 |
| CG-P-GUTS 在 METR-LA / PEMS-BAY 上方向相反 | coarse graph 可能受数据集空间冗余度影响，需要后续再做构造方式实验 |
| CG-P-GUTS 明显退化 | 当前 coarse assignment 可能过平滑，需要下一轮改 distance clustering 或 gate |

---

## 八、建议首轮执行量

为了尽快判断方向，不建议一开始就跑满所有阶段。首轮建议：

```text
Phase 1 smoke: 4 runs
Phase 2 paper reproduction: 4 training runs
Phase 3 coarse graph module: 4 runs
Table 3 inference robustness: 12 P-GUTS evaluations + 12 CG-P-GUTS evaluations after checkpoints finish
```

合计：

```text
8 formal training runs + 24 inference evaluations + 4 smoke runs
```

本机有 8 张 A800-80GB。官方 P-GUTS 论文配置使用 `batch_size=8`、`batches_epoch=300`、`epochs=300`、`patience=40`。本轮为了更好利用显存，Phase 2 论文复现实验和 Phase 3 coarse graph 主实验按数据集放大 batch size，并同步缩小 `batches_epoch`：METR-LA 使用 `batch_size=128, batches_epoch=19`，PEMS-BAY 使用 `batch_size=256, batches_epoch=10`，每个 epoch 处理样本量约为 `2432/2560`，接近论文设置的 `2400`。其余关键设置保持论文口径，并保留官方 `split_batch_in` 梯度累积机制。报告中必须明确标注：这是“官方代码 + 论文协议 + batch 放大且 epoch 样本量近似保持”的复现实验，不是逐字节完全相同的 batch size 复现。

---

## 九、官方代码启动方式

### 9.1 环境准备

官方 repo 自带环境文件：

```text
external_repro/pguts/environment.yml
```

建议创建环境：

```bash
cd /data/wangzuke/time-series-forecast-exp
bash missing_ts_exp/scripts/r0723_create_official_pguts_env.sh
```

当前 base / hzc_agent 环境缺少 `pytorch_lightning`、`tsl`、`torch_geometric`，不能直接启动官方 P-GUTS。

环境检查：

```bash
R0723_PYTHON_CMD='conda run -n spin_env python' \
  bash missing_ts_exp/scripts/r0723_run_official_pguts_cg.sh prepare
```

`prepare` 会同时完成三件事：重新生成命令文件、记录官方代码 commit/patch、准备 TSL 数据缓存。TSL 默认会尝试把数据写入 conda 包目录下的 `.storage`，该目录在本机不可写；启动脚本已经把缓存改到：

```text
missing_ts_exp/results/0723_official_pguts_coarse_imputation/tsl_cache/
```

并将本地已有的 `metr_la.h5`、`pems_bay.h5` 和距离图文件软链接成 TSL 期望的文件名。

### 9.2 命令生成

```bash
cd /data/wangzuke/time-series-forecast-exp
python missing_ts_exp/scripts/r0723_build_official_pguts_cg_cmds.py
```

生成：

```text
missing_ts_exp/scripts/r0723_official_pguts_cg_smoke_cmds.txt          4 runs
missing_ts_exp/scripts/r0723_official_pguts_reproduce_cmds.txt         4 runs
missing_ts_exp/scripts/r0723_official_pguts_cg_coarse_cmds.txt         4 runs
missing_ts_exp/scripts/r0723_official_pguts_cg_table3_inference_cmds.txt  generated after checkpoints finish
```

### 9.3 启动实验

先跑 smoke：

```bash
R0723_PYTHON_CMD='conda run -n spin_env python' \
R0723_GPUS='0 1 2 3' \
  bash missing_ts_exp/scripts/r0723_run_official_pguts_cg.sh smoke
```

smoke 通过后，论文复现实验和 coarse graph 主实验可以并行跑。建议把 8 张 A800 分成两组，复现实验用 0-3，主实验用 4-7：

```bash
R0723_PYTHON_CMD='conda run -n spin_env python' \
R0723_GPUS='0 1 2 3' \
  bash missing_ts_exp/scripts/r0723_run_official_pguts_cg.sh reproduce

R0723_PYTHON_CMD='conda run -n spin_env python' \
R0723_GPUS='4 5 6 7' \
  bash missing_ts_exp/scripts/r0723_run_official_pguts_cg.sh coarse
```

如果希望先确认复现趋势，也可以先跑 `reproduce`，再跑 `coarse`；两种安排不影响每条 run 的配置。

`reproduce` 和 `coarse` 的 `[3,6]` checkpoint 完成后，再生成并启动 Table 3 鲁棒性 inference。该阶段只替换测试 `eval_mask`，不重新训练：

```bash
python missing_ts_exp/scripts/r0723_build_official_pguts_inference_cmds.py

R0723_PYTHON_CMD='conda run -n spin_env python' \
R0723_GPUS='0 1 2 3 4 5 6 7' \
  bash missing_ts_exp/scripts/r0723_run_official_pguts_cg.sh table3
```

如果 checkpoint 尚未全部完成，命令生成脚本会把缺失项写到：

```text
missing_ts_exp/scripts/r0723_official_pguts_cg_table3_missing_checkpoints.txt
```

每次启动会记录：

```text
missing_ts_exp/results/0723_official_pguts_coarse_imputation/notes/official_pguts_commit.txt
missing_ts_exp/results/0723_official_pguts_coarse_imputation/notes/official_pguts_coarse_graph.patch
missing_ts_exp/results/0723_official_pguts_coarse_imputation/raw_logs/
```

---

## 十、最终报告应回答的问题

实验报告不能只列结果，需要回答：

1. coarse graph 在 P-GUTS 原始插补任务上是否有效？
2. 它在 METR-LA 和 PEMS-BAY 上是否一致？
3. 它是否在高 failure probability 下更有优势？
4. 它是 full graph 的补充，还是可以替代 full graph？
5. 它在 Block-ST 这种空间-时间联合缺失上是否比 Block-T 更有价值？
6. 这个模块是否值得并入下一版 P-GUTS / HD-PGUTS 主线？

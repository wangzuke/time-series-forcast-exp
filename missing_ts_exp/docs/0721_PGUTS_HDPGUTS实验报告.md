# 0721 实验报告：P-GUTS-Forecaster 与 HD-PGUTS 主线实验

> **承接**：[`0718实验报告.md`](0718实验报告.md)、[`实验计划0721_PGUTS_HDPGUTS主线.md`](实验计划0721_PGUTS_HDPGUTS主线.md)  
> **日期**：2026-07-21 ~ 2026-07-22  
> **硬件**：8 × NVIDIA A800-SXM4-80GB  
> **实验规模**：168 条汇总结果（P-GUTS 96 条；HD-PGUTS 消融 72 条），另有 Phase 1 smoke 用于启动验收  
> **种子**：P-GUTS 主矩阵 `seed=1`；关键块缺失与 HD-PGUTS 主实验 `seed=1,2,3`  
> **训练设置**：`epochs=100`、`batch_size=512`、`lr=1e-3`、`weight_decay=1e-4`、`patience=20`、`T_in=24`  
> **结果来源**：`missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/`，图表来自 `missing_ts_exp/docs/figures/r0721/`

**前置说明**：本报告覆盖 0721 P-GUTS / HD-PGUTS 主创新线的已完成结果。0720 公平基线目录目前只有少量 smoke 记录，不能作为完整外部 baseline 进入本报告的定量结论。因此，下文的正式结论主要来自本轮内部对比：P-GUTS `[3]` / `[3,6]`、HD-PGUTS 三个消融变体，以及修正后的 adaptive scale weights 诊断。

---

## 一、实验总览

### 1.1 实验背景

0715 与 0718 的复现实验已经给出一个稳定背景：在交通预测任务中，HD-TTS 类多尺度时空建模对连续块缺失更稳，而 BiTGraph 类跨变量图传播在点缺失下相对自然，但在整段时间同时缺失时会失去同时刻观测支撑。基于这个背景，0721 计划尝试回答两个问题：

1. P-GUTS 能否从 imputation 改造成缺失历史条件下的 forecasting 模型。
2. HD-TTS 的时空多尺度思想接入 P-GUTS 后，是否能提升高缺失率 Block-T / Block-ST 预测。

本轮没有拿到可直接纳入对比的 P-GUTS 官方原始复现结果，而是采用本项目内的 P-GUTS-style 最小重写入口：把未来预测构造成 future-value imputation，统一读取 `dataset/0721_missing_masks/` 的 mask bundle。这个实现足以回答本轮的内部架构问题，但不应被解读为对 P-GUTS 官方论文数值的忠实复现。

### 1.2 实验规模

| 阶段 | 内容 | 结果状态 |
|---|---|---|
| Phase 1 | P-GUTS-Forecaster smoke：2 数据集 × 2 块缺失类型 × 2 pooling | 完成，8 条均通过 5 epoch、无 NaN、日志字段齐全 |
| Phase 2 | P-GUTS 全矩阵：seed=1 全条件 + 关键块缺失 seed=2,3 | 完成，`pguts_results.csv` 96 行 |
| Phase 3 | HD-PGUTS 三变体：`no_graph_coarsening`、`no_adaptive_fusion`、`full` | 完成，`hd_pguts_ablation_results.csv` 72 行 |
| Phase 4 | P-GUTS `T_out=12` 补实验 | P-GUTS 完成 32 条；HD-PGUTS full `T_out=12` 未出现在最终汇总中 |
| 修正项 | `no_graph_coarsening` 首版同构 bug 审计与重跑 | 已修正并重跑 24 条，旧结果作废 |

最终汇总文件：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/pguts_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/hd_pguts_ablation_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/hd_pguts_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/pguts_hdpguts_all_results.csv
```

报告新增的派生表：

```text
r0721_pguts_h24_pooling_summary.csv
r0721_pguts_critical_multiseed_summary.csv
r0721_hdpguts_ablation_multiseed_summary.csv
r0721_hdpguts_vs_pguts36_delta.csv
r0721_adaptive_scale_weights_summary.csv
r0721_efficiency_summary.csv
```

### 1.3 数据集与协议

| 数据集 | 节点数 | 时间步 | 维度级别 | 输入/预测窗口 | 缺失类型 |
|---|---:|---:|---|---|---|
| METR-LA | 207 | 34272 | 高维交通图 | `T_in=24`，`T_out=12/24` | Point、Block-T、Block-ST |
| PEMS-BAY | 325 | 52116 | 高维交通图 | `T_in=24`，`T_out=12/24` | Point、Block-T、Block-ST |

所有正式结果遵守统一协议：先按滑窗构造样本，再按窗口顺序 `70% / 10% / 20%` 切分；mask 语义为 `1=observed, 0=missing`；正式 `batch_size=512`；主要指标为 MAE，RMSE 作为辅助指标。

### 1.4 方法命名说明

| 简称 | 全称 | 技术说明 |
|---|---|---|
| P-GUTS `[3]` | P-GUTS-Forecaster 单尺度版本 | 将未来预测构造成 future-value imputation；分支为 identity + temporal pool 3 + full-resolution graph；fixed linear fusion |
| P-GUTS `[3,6]` | P-GUTS-Forecaster 双时间尺度版本 | 在 `[3]` 基础上增加 temporal pool 6；仍只使用 full-resolution graph 与 fixed linear fusion |
| HD-PGUTS w/o graph coarsening | 无粗图分支的 HD 变体 | 分支与 P-GUTS `[3,6]` 相同，但使用 adaptive gate 融合；用于验证 adaptive fusion 在没有空间粗化时是否有效 |
| HD-PGUTS w/o adaptive fusion | 无自适应融合的 HD 变体 | 分支为 identity + temporal pool 3/6 + full graph + coarse graph；使用 fixed linear fusion；用于隔离粗图分支贡献 |
| HD-PGUTS full | 完整 HD-PGUTS | 分支为 identity + temporal pool 3/6 + full graph + coarse graph；使用 adaptive gate 融合 |

### 1.5 实现差异与异常修正

本轮实现包含三类差异：

1. **研究扩展**：新增 `missing_ts_exp/src/training/run_pguts_hdpguts.py`，把 forecasting 表述为 future-value imputation：

```text
X_all = concat(X_hist_observed, zeros_future)
M_all = concat(M_hist, zeros_future_mask)
Y_hat_future = model(X_all, M_all)
loss = MAE(Y_hat_future, Y_future)
```

2. **工程修复**：`distances_bay_2017.csv` 无表头，而 `distances_la_2012.csv` 有 `from,to,cost` 表头。首版 `load_adjacency()` 用 `csv.DictReader` 读取 PEMS-BAY 距离图时触发 `KeyError: 'from'`，现已改为兼容有表头/无表头的 `read_distance_edges()`。
3. **消融 bug 修复**：首版 `no_graph_coarsening` 与 P-GUTS `[3,6]` 完全同构，导致 24 组 MAE/RMSE 与 baseline 逐位相同。旧结果已标记无效，详见 `results/0721_cofill_pguts_forecasting/notes/no_graph_coarsening_invalid_20260722.md`。修正后 `no_graph_coarsening` 保留无粗图分支，但改用 adaptive gate 融合；本报告使用修正后重跑的 24 条结果。

---

## 二、Phase 1：Smoke 与环境问题

### 2.1 Smoke 结果

8 条 smoke 覆盖：

```text
METR-LA / PEMS-BAY
Block-T 70% / Block-ST 70%
pooling [3] / [3,6]
T_out=24
```

验收结果：全部能完成 5 epoch，无 NaN；日志打印 `data_path`、`mask_path`、`mask_sha256`、`actual_missing_rate`、`T_in/T_out`、`batch_size`；正式 batch size 为 512。

### 2.2 环境与数据图修复

启动阶段暴露了两个工程问题：

- 当前 base 环境缺 `tables/pytables`，无法通过 pandas 读取 `.h5`。已新增 `missing_ts_exp/scripts/r0721_check_pguts_env.py` 和 `missing_ts_exp/env_r0721_pguts.yml`，`prepare` 阶段会提前检查环境，避免 GPU 任务启动后集中失败。
- PEMS-BAY 的距离图 CSV 无表头，导致首轮 PEMS smoke 全部失败。修复后直接构建 adjacency 的检查结果为：METR-LA `(207,207)`、PEMS-BAY `(325,325)`，矩阵值有限且行和约等于 1。

这些问题属于实验启动与数据读取层，不影响已经通过修复后重跑并进入最终 CSV 的结果。

---

## 三、Phase 2：P-GUTS-Forecaster 全矩阵

### 3.1 `T_out=24, seed=1` 下 `[3]` 与 `[3,6]` 对比

下表来自 `r0721_pguts_h24_pooling_summary.csv`。负值表示 `[3,6]` 的 MAE 低于 `[3]`。

| 数据集 | 缺失 | 缺失率 | `[3]` MAE | `[3,6]` MAE | Δ% |
|---|---|---:|---:|---:|---:|
| METR-LA | Block-ST | 50% | 8.023 | 7.988 | -0.45% |
| METR-LA | Block-ST | 70% | 8.668 | 8.613 | -0.64% |
| METR-LA | Block-ST | 90% | 9.800 | 9.763 | -0.37% |
| METR-LA | Block-T | 50% | 8.538 | 8.334 | -2.39% |
| METR-LA | Block-T | 70% | 10.008 | 9.644 | **-3.64%** |
| METR-LA | Block-T | 90% | 11.302 | 11.287 | -0.13% |
| METR-LA | Point | 50% | 7.352 | 7.296 | -0.76% |
| METR-LA | Point | 70% | 7.730 | 7.712 | -0.23% |
| PEMS-BAY | Block-ST | 50% | 3.253 | 3.164 | -2.74% |
| PEMS-BAY | Block-ST | 70% | 3.596 | 3.557 | -1.09% |
| PEMS-BAY | Block-ST | 90% | 4.125 | 4.110 | -0.36% |
| PEMS-BAY | Block-T | 50% | 3.253 | 3.189 | -1.99% |
| PEMS-BAY | Block-T | 70% | 3.617 | 3.549 | -1.87% |
| PEMS-BAY | Block-T | 90% | 4.200 | 4.204 | +0.10% |
| PEMS-BAY | Point | 50% | 2.909 | 2.819 | **-3.09%** |
| PEMS-BAY | Point | 70% | 3.077 | 3.043 | -1.08% |

![P-GUTS pooling delta](figures/r0721/pguts_pooling_delta_h24.png)

**关键发现**：

1. `[3,6]` 在 15/16 个 `T_out=24, seed=1` 条件下降低 MAE，说明加入第二个时间池化尺度整体有效，但改善幅度偏小。唯一退化是 PEMS-BAY Block-T 90%，幅度只有 +0.10%。
2. 改善最明显的场景不是所有高缺失率，而是部分中高缺失率：METR-LA Block-T 70% 改善 -3.64%，PEMS-BAY Point 50% 改善 -3.09%。Block-T / Block-ST 90% 反而只有 -0.13% ~ -0.37% 级别，说明粗时间尺度在极端缺失下并没有自动带来更大收益。
3. 对 Point 缺失也有轻微收益，说明 `[3,6]` 的优势不是只来自块缺失补偿，而更像是一般性的时间平滑/上下文扩展。但 Point 缺失下改善有限，不足以把多尺度解释成专门针对缺失模式的强机制。

### 3.2 关键块缺失多种子结果

下表来自 `r0721_pguts_critical_multiseed_summary.csv`，为 `seed=1,2,3` 平均。

| 数据集 | 缺失 | 缺失率 | `[3]` MAE | `[3,6]` MAE | Δ% |
|---|---|---:|---:|---:|---:|
| METR-LA | Block-ST | 70% | 8.685 ± 0.015 | 8.696 ± 0.098 | +0.13% |
| METR-LA | Block-ST | 90% | 9.796 ± 0.005 | 9.778 ± 0.016 | -0.19% |
| METR-LA | Block-T | 70% | 10.018 ± 0.031 | 9.823 ± 0.235 | -1.95% |
| METR-LA | Block-T | 90% | 11.476 ± 0.151 | 11.435 ± 0.225 | -0.36% |
| PEMS-BAY | Block-ST | 70% | 3.600 ± 0.005 | 3.572 ± 0.014 | -0.78% |
| PEMS-BAY | Block-ST | 90% | 4.125 ± 0.001 | 4.119 ± 0.009 | -0.15% |
| PEMS-BAY | Block-T | 70% | 3.607 ± 0.009 | 3.562 ± 0.022 | -1.26% |
| PEMS-BAY | Block-T | 90% | 4.202 ± 0.008 | 4.197 ± 0.006 | -0.13% |

**关键发现**：

1. 多种子后 `[3,6]` 的优势仍存在，但更温和：8 个关键条件中 7 个改善，幅度多在 -0.13% ~ -1.95%。这说明 Phase 2 第三条进入 HD-PGUTS 的判断标准（`[3,6]` 与 `[3]` 有差异）成立，但这个差异不是强到足以单独成为主创新。
2. METR-LA Block-ST 70% 在多种子平均下变成 +0.13%，与 seed=1 的 -0.64% 矛盾，提示部分单点改善会被 seed 方差抹平。报告后续对 HD-PGUTS 的评价因此以三种子均值为主，不使用单 seed 好看的个例。
3. 90% 缺失下没有 NaN，也没有全 batch 无监督导致的系统失败，说明 forecasting-as-imputation 的训练入口在极端块缺失下是稳定的。但 90% 下的多尺度收益普遍偏小，表明仅增加时间池化尺度不能充分解决极端缺失的信息不足。

---

## 四、Phase 3：HD-PGUTS 主创新与消融

### 4.1 三变体与 P-GUTS `[3,6]` 的主表

下表来自 `r0721_hdpguts_ablation_multiseed_summary.csv`，所有数值均为 `seed=1,2,3` 的 MAE 均值 ± 标准差。

| 数据集 | 缺失 | 缺失率 | P-GUTS `[3,6]` | w/o graph coarsening | w/o adaptive fusion | full | 最优 |
|---|---|---:|---:|---:|---:|---:|---|
| METR-LA | Block-ST | 70% | 8.696 ± 0.098 | 9.055 ± 0.104 | **8.605 ± 0.030** | 8.924 ± 0.094 | w/o adaptive fusion |
| METR-LA | Block-ST | 90% | 9.778 ± 0.016 | 10.009 ± 0.045 | **9.641 ± 0.044** | 9.816 ± 0.004 | w/o adaptive fusion |
| METR-LA | Block-T | 70% | 9.823 ± 0.235 | 10.030 ± 0.095 | **9.684 ± 0.063** | 10.064 ± 0.037 | w/o adaptive fusion |
| METR-LA | Block-T | 90% | 11.435 ± 0.225 | 11.526 ± 0.164 | **11.412 ± 0.256** | 11.625 ± 0.049 | w/o adaptive fusion |
| PEMS-BAY | Block-ST | 70% | 3.572 ± 0.014 | 3.656 ± 0.010 | **3.526 ± 0.005** | 3.619 ± 0.009 | w/o adaptive fusion |
| PEMS-BAY | Block-ST | 90% | 4.119 ± 0.009 | 4.117 ± 0.013 | **4.057 ± 0.002** | 4.097 ± 0.022 | w/o adaptive fusion |
| PEMS-BAY | Block-T | 70% | 3.562 ± 0.022 | 3.646 ± 0.012 | **3.533 ± 0.009** | 3.682 ± 0.010 | w/o adaptive fusion |
| PEMS-BAY | Block-T | 90% | 4.197 ± 0.006 | 4.218 ± 0.022 | **4.191 ± 0.005** | 4.241 ± 0.007 | w/o adaptive fusion |

![HD-PGUTS delta vs P-GUTS](figures/r0721/hdpguts_delta_vs_pguts36.png)

相对 P-GUTS `[3,6]` 的平均变化：

| 变体 | 平均 ΔMAE | 最好 | 最差 |
|---|---:|---:|---:|
| w/o graph coarsening | +1.82% | -0.04% | +4.13% |
| w/o adaptive fusion | **-0.97%** | -1.49% | -0.14% |
| full | +1.55% | -0.52% | +3.38% |

**关键发现**：

1. `w/o adaptive fusion` 在 8/8 个关键块缺失条件下都是最优，平均比 P-GUTS `[3,6]` 低 0.97% MAE。这是 HD-PGUTS 主线里最稳定的正向信号。
2. `full` 没有达到计划设定的主创新预期。它只在 PEMS-BAY Block-ST 90% 上优于 P-GUTS `[3,6]`（-0.52%），其余 7 个条件均更差，平均 +1.55%。因此“时间尺度 + 空间粗化 + 自适应融合”的完整组合在本实现里没有形成主线优势。
3. 修正后的 `w/o graph coarsening` 平均 +1.82%，说明只加入 adaptive fusion 而不加入粗图分支，不但不能提升，反而更容易变差。也就是说，自适应融合本身不是收益来源。
4. 与 Phase 2 结合看，真正有贡献的是 coarse graph branch，而不是 adaptive gate。`w/o adaptive fusion` 与 `full` 的差异只在融合方式：前者固定融合、后者自适应融合；结果前者全条件更好，说明 adaptive gate 在当前训练规模和损失下可能引入了优化噪声或过拟合。

### 4.2 `no_graph_coarsening` 异常与修正

本轮中途发现首版 `no_graph_coarsening` 的 MAE/RMSE 与 P-GUTS `[3,6]` 在 24 个匹配条件上逐位相同。进一步检查 checkpoint 后确认，两者 `state_dict` 逐张量完全相等。这不是统计巧合，而是代码实现缺陷：

```text
首版 pguts               = identity + temporal3 + temporal6 + full_graph + fixed fusion
首版 no_graph_coarsening = identity + temporal3 + temporal6 + full_graph + fixed fusion
```

修正后定义为：

```text
P-GUTS [3,6]          = 无 coarse graph + fixed fusion
no_graph_coarsening   = 无 coarse graph + adaptive fusion
no_adaptive_fusion    = 有 coarse graph + fixed fusion
full                  = 有 coarse graph + adaptive fusion
```

修正后的 24 条 `no_graph_coarsening` 已重跑并覆盖 CSV。新结果与 P-GUTS `[3,6]` 不再相等：24 个匹配条件中 MAE 精确相等数为 0，平均相对 P-GUTS `[3,6]` 变差 +1.83%。这批结果才是本报告采用的有效消融。

**解释**：这次异常很有价值。它说明如果只看表格而不审计架构签名，消融实验很容易把“重复 baseline”误当作一个有效变体。后续所有新增消融都应记录 `architecture_signature` 或等价字段。

### 4.3 自适应权重诊断

`full` 与修正后的 `no_graph_coarsening` 都保存了 scale weights。下表来自 `r0721_adaptive_scale_weights_summary.csv`，为跨数据集与 seed 聚合后的均值。

| 模型 | 缺失 | 缺失率 | identity | temporal3 | temporal6 | full_graph | coarse_graph |
|---|---|---:|---:|---:|---:|---:|---:|
| full | Block-ST | 70% | 0.030 | 0.457 | 0.331 | 0.139 | 0.043 |
| full | Block-ST | 90% | 0.000 | 0.481 | 0.364 | 0.126 | 0.029 |
| full | Block-T | 70% | 0.000 | 0.390 | 0.456 | 0.149 | 0.006 |
| full | Block-T | 90% | 0.000 | 0.184 | 0.621 | 0.192 | 0.002 |
| w/o graph coarsening | Block-ST | 70% | 0.026 | 0.356 | 0.563 | 0.055 | - |
| w/o graph coarsening | Block-ST | 90% | 0.002 | 0.358 | 0.607 | 0.033 | - |
| w/o graph coarsening | Block-T | 70% | 0.021 | 0.209 | 0.634 | 0.136 | - |
| w/o graph coarsening | Block-T | 90% | 0.008 | 0.225 | 0.598 | 0.169 | - |

![Adaptive scale weights](figures/r0721/adaptive_scale_weights.png)

**关键发现**：

1. adaptive gate 大部分权重给了 temporal branches，尤其是 `temporal6`。这与块缺失需要粗时间尺度的直觉一致。
2. `full` 中 coarse graph 权重很低：Block-ST 70% 为 0.043，Block-T 90% 只有 0.002。换言之，虽然 `w/o adaptive fusion` 证明 coarse branch 有益，但 `full` 的 gate 并没有充分使用 coarse branch。
3. Block-T 90% 下 `temporal6` 权重最高（full: 0.621；w/o graph coarsening: 0.598），说明 gate 能感知到严重时间块缺失需要更粗时间尺度；但这个“解释性正确”没有转化为 MAE 优势，提示权重诊断与最终性能之间不能直接画等号。

---

## 五、Phase 4：Horizon=12 补实验

P-GUTS 的 `T_out=12` 全矩阵已在 `pguts_results.csv` 中出现，共 32 条（2 数据集 × 8 缺失条件 × 2 pooling × seed=1）。总体趋势与 `T_out=24` 一致：`[3,6]` 大多数条件更好，且 `T_out=12` 作为更短预测任务，MAE 普遍低于 `T_out=24`。

但 HD-PGUTS full 的 `T_out=12` 补实验没有进入最终汇总 CSV。本报告因此不对 “HD-PGUTS 在 12 步预测下是否同样有效” 做结论。后续如果要补齐 Phase 4，应优先跑：

```text
HD-PGUTS full
METR-LA / PEMS-BAY
Block-ST 70% / 90%
T_out=12
seed=1,2,3
```

---

## 六、效率与资源使用

| 方法 | runs | 平均 epoch 秒 | 平均训练分钟 | 峰值显存 GB |
|---|---:|---:|---:|---:|
| P-GUTS | 96 | 19.5 | 34.8 | 13.1 |
| w/o graph coarsening | 24 | 23.9 | 42.8 | 19.0 |
| w/o adaptive fusion | 24 | 27.6 | 49.2 | 19.8 |
| full | 24 | 29.9 | 53.6 | 22.9 |

![Efficiency tradeoff](figures/r0721/efficiency_tradeoff_pguts_hdpguts.png)

**关键发现**：

1. 参数与显存开销随分支增加单调上升：P-GUTS 约 13.1GB，`full` 约 22.9GB。对于 80GB A800，单 run 显存不是瓶颈，采用 `R0721_SLOTS_PER_GPU=2` 的多 run 并发是合理调度。
2. `w/o adaptive fusion` 比 `full` 更快、更省显存、MAE 更好，是本轮性价比最高的 HD 变体。
3. `no_graph_coarsening` 比 `w/o adaptive_fusion` 更快但效果更差，说明节省 coarse graph branch 的代价是主任务性能下降。

---

## 七、对计划成功标准的评估

| 计划标准 | 结论 | 依据 |
|---|---|---|
| P-GUTS 能否改造成 forecasting 模型 | 达成 | 96 条 P-GUTS 结果完整，smoke 无 NaN，90% 块缺失稳定训练 |
| `[3,6]` 与 `[3]` 是否有差异 | 达成但幅度有限 | seed=1 下 15/16 条 `[3,6]` 更好；关键多种子下 7/8 条更好，平均改善多在 2% 以内 |
| HD-PGUTS full 是否稳定优于 P-GUTS `[3,6]` | 未达成 | full 平均相对 P-GUTS `[3,6]` 变差 +1.55%，仅 1/8 个关键条件改善 |
| coarse graph 是否有用 | 有条件达成 | `w/o adaptive fusion` 在 8/8 个关键条件最优，平均 -0.97% |
| adaptive fusion 是否有用 | 未达成 | `no_graph_coarsening` 平均 +1.82%；full 平均 +1.55%；adaptive gate 未带来 MAE 收益 |
| adaptive scale weights 是否可解释缺失模式 | 部分达成 | 权重确实偏向 `temporal6`，但 coarse graph 权重很低，解释性与性能不一致 |

---

## 八、综合结论

1. **P-GUTS-Forecaster 改造是可行的。** 未来插补式输入构造能稳定训练，覆盖 Point / Block-T / Block-ST 以及 90% 极端缺失，没有出现 NaN 或无监督崩溃。
2. **双时间尺度 `[3,6]` 是低成本、小幅正收益改动。** 在 `T_out=24, seed=1` 下 15/16 条改善；关键块缺失多种子下 7/8 条改善。但改善通常低于 2%，不能单独支撑“主创新”。
3. **HD-PGUTS 的有效部分不是 full，而是 coarse graph + fixed fusion。** `w/o adaptive fusion` 在 8/8 个关键条件最优，平均比 P-GUTS `[3,6]` 低 0.97% MAE。
4. **adaptive fusion 是本轮最大负面发现。** 无论没有 coarse branch 的 `no_graph_coarsening`，还是含 coarse branch 的 `full`，adaptive gate 都没有带来稳定收益，反而平均变差。权重诊断显示 gate 偏向 `temporal6`，但没有充分使用 coarse branch。
5. **`no_graph_coarsening` 的同构 bug 已修正，但也暴露了消融设计风险。** 首版消融与 baseline 完全同构，导致 24 条结果逐位相同。后续所有消融必须把架构签名写入结果，避免“名字不同、计算相同”的重复实验。
6. **本轮不支持把 HD-PGUTS full 作为论文主线。** 如果继续沿这条线推进，建议把主线改为 “P-GUTS + coarse graph fixed fusion”，而不是 “coarse graph + adaptive fusion full”。
7. **外部 baseline 与统一 evaluator 仍需补齐。** 0720 公平基线目录不完整，本报告不能给出与 HD-TTS-AMP / BiTGraph 的严格定量优劣结论。最终论文表仍应由统一 evaluator 复算。

---

## 九、后续建议

1. **把 `w/o adaptive fusion` 升级为新的 HD-PGUTS 主候选。** 先不要继续扩展 full gate，而是围绕 coarse graph fixed fusion 做更干净的参数与图粗化消融。
2. **补一组 `coarse graph only` 与 `full graph only` 更细消融。** 当前 `w/o adaptive fusion` 同时含 full graph 与 coarse graph，仍无法分离两者贡献。建议新增：只保留 full graph、只保留 coarse graph、两者都保留。
3. **重新设计 adaptive fusion。** 现有 gate 只用全局 branch summary，可能缺少 mask pattern 的显式条件。后续若保留 adaptive fusion，应把 mask statistics（缺失率、连续块长度、节点覆盖率）直接输入 gate。
4. **补齐 HD-PGUTS full 的 `T_out=12`。** 只有 P-GUTS `T_out=12` 已完成，HD-PGUTS 短 horizon 结论缺失。
5. **接入统一 evaluator。** 目前每条 run 保存了 prediction 文件，但最终跨模型比较应统一读取 `y_true/y_pred/mask` 复算 MAE/RMSE/MAPE，减少模型内部 evaluator 差异。

---

## 十、图表索引

| 图号 | 文件 | 说明 |
|---|---|---|
| 图 1 | `figures/r0721/pguts_pooling_delta_h24.png` | P-GUTS `[3,6]` 相对 `[3]` 的 `T_out=24, seed=1` MAE 变化 |
| 图 2 | `figures/r0721/hdpguts_delta_vs_pguts36.png` | HD-PGUTS 三变体相对 P-GUTS `[3,6]` 的多种子平均 MAE 变化 |
| 图 3 | `figures/r0721/adaptive_scale_weights.png` | adaptive fusion 权重在不同缺失类型与缺失率下的分布 |
| 图 4 | `figures/r0721/efficiency_tradeoff_pguts_hdpguts.png` | P-GUTS / HD-PGUTS 各变体的平均训练时间、显存与误差概览 |

## 十一、产出文件索引

| 文件 | 行数 | 说明 |
|---|---:|---|
| `csv/pguts_results.csv` | 96 | P-GUTS `[3]` / `[3,6]` 全矩阵 |
| `csv/hd_pguts_ablation_results.csv` | 72 | HD-PGUTS 三变体消融 |
| `csv/hd_pguts_results.csv` | 24 | HD-PGUTS full 结果子集 |
| `csv/pguts_hdpguts_all_results.csv` | 168 | 本轮全部汇总结果 |
| `csv/r0721_hdpguts_vs_pguts36_delta.csv` | 24 | HD-PGUTS 相对 P-GUTS `[3,6]` 的改善率表 |
| `csv/r0721_adaptive_scale_weights_summary.csv` | 48 | adaptive fusion 权重诊断 |
| `notes/no_graph_coarsening_invalid_20260722.md` | - | `no_graph_coarsening` 首版失效说明与修正记录 |

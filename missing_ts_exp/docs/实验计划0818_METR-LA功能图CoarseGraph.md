# 实验计划0818：METR-LA 功能图 Coarse Graph 插补实验

> 承接：`0723_PGUTS_CoarseGraph插补实验报告.md`  
> 目标：在不使用道路空间信息的前提下，仅由 METR-LA 时间序列学习变量关系，构造 coarse assignment，并复用 0723 官方 P-GUTS 插补协议验证功能图粗化是否能复现 distance-greedy 的收益。

## 一、核心问题

0723 结果显示，distance-greedy coarse graph 在 METR-LA 上显著改善 P-GUTS `[3,6]`，但该 assignment 依赖道路距离图。本轮只问一个更窄的问题：

```text
如果不知道 METR-LA 空间距离图，仅凭训练集时间序列关系，能否构造同样有效的 coarse graph？
```

本轮参考三篇非时空缺失预测论文：

| 论文 | 借鉴点 | 本轮落地 |
|---|---|---|
| T-PatchGNN | 由 patch / variable embedding 学 time-adaptive variable graph | 先用时序相似度构 functional graph，后续再端到端学习动态图 |
| HyperIMTS | 同时考虑 overall variable similarity 与 time-aware similarity | 同时比较静态相关图与 lag-corr 图 |
| GraFITi | 图可由观测结构生成，不必来自物理拓扑 | 使用 observation/time-series-derived graph 作为空间图替代 |

## 二、实验协议

严格沿用 0723 的 METR-LA 设置：

| 项目 | 设置 |
|---|---|
| 数据集 | METR-LA |
| 任务 | P-GUTS 官方窗口内 spatiotemporal imputation |
| 窗口 | `window=24`, `stride=1` |
| 主 benchmark mask | traffic BLOCK，训练 `p_fault=0.0015`, `p_noise=0.05` |
| Table 3 | 复用 checkpoint，只在 inference 替换测试 mask |
| Table 3 测试强度 | `p_fault=0.05,0.10,0.15`, `p_noise=0` |
| pooling | `[3,6]` |
| 训练 | `seed=1,2`, `batch_size=16`, `batches_epoch=300`, `epochs=300`, `patience=40` |

P-GUTS `[3,6]` baseline 与 distance-greedy CG-P-GUTS 复用 0723 结果，本轮新增功能图 assignment。

## 三、方法矩阵

| 方法 | 构图信息 | 目的 |
|---|---|---|
| `raw_corr` | 原始训练序列 Pearson correlation | 最朴素的功能图 |
| `residual_corr` | 去全局均值与日周期后的 correlation | 减少早晚高峰共同周期伪相关 |
| `lag_corr` | residual 序列最大滞后相关，默认 `max_lag=12` | 捕捉交通传播延迟 |
| `random` | 随机分组 | 负对照，排除额外参数量收益 |

所有方法使用 `coarse_factor=4`，因此 METR-LA 约 `207 -> 52` 个 coarse groups，和 0723 的粗化强度保持同量级。

## 四、执行资产

新增脚本：

```text
missing_ts_exp/scripts/r0818_build_metrla_functional_coarse_assignments.py
missing_ts_exp/scripts/r0818_prepare_metrla_functional_pguts_configs.py
missing_ts_exp/scripts/r0818_build_metrla_functional_pguts_cmds.py
```

推荐执行顺序：

```bash
# 1. 构造功能图 coarse assignment
python missing_ts_exp/scripts/r0818_build_metrla_functional_coarse_assignments.py \
  --data-path dataset/metr_la/metr_la.h5 \
  --coarse-factor 4 \
  --season-period 288 \
  --max-lag 12

# 2. 在官方 P-GUTS 仓库中准备 r0818 configs
python missing_ts_exp/scripts/r0818_prepare_metrla_functional_pguts_configs.py \
  --pguts-dir external_repro/pguts

# 3. 生成 smoke 和正式训练命令
python missing_ts_exp/scripts/r0818_build_metrla_functional_pguts_cmds.py
```

生成命令：

```text
missing_ts_exp/scripts/r0818_metrla_functional_cg_smoke_cmds.txt
missing_ts_exp/scripts/r0818_metrla_functional_cg_train_cmds.txt
```

正式训练完成后，将官方 P-GUTS log 中完成的 exp name 写入：

```text
missing_ts_exp/scripts/r0818_metrla_functional_cg_table3_exp_names_template.txt
```

再生成 Table 3 inference 命令：

```bash
python missing_ts_exp/scripts/r0818_build_metrla_functional_pguts_cmds.py \
  --exp-names-file missing_ts_exp/scripts/r0818_metrla_functional_cg_table3_exp_names_template.txt
```

## 五、预期判定

| 现象 | 解释 |
|---|---|
| `residual_corr` 或 `lag_corr` 接近 / 优于 distance-greedy | METR-LA 粗图收益主要来自可由时序恢复的功能关系 |
| `raw_corr` 好但 `residual_corr` 差 | 收益可能来自全局周期共性，而非局部功能关系 |
| `lag_corr` 最好 | 交通传播延迟对功能图构造关键 |
| 功能图均不优于 random | 当前粗图收益可能依赖空间先验或节点排序，纯时序关系不足 |

本轮成功标准不是立刻超过 0723 distance-greedy，而是至少证明：

```text
functional assignment 明显优于 random，并在 METR-LA 主 benchmark / Table 3 上保持方向一致。
```

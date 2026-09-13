# 面向缺失观测多变量时间序列的双路依赖多尺度区域化图建模的插补方法

> 日期：2026-08-31  
> 方向：观测缺失多变量时间序列插补  
> 承接实验：`0723_PGUTS_CoarseGraph插补实验报告.md`  
> 核心目标：在 P-GUTS + coarse graph 插补实验的基础上，将单一粗图分支深化为“空间依赖路 + 时序依赖路”的多尺度区域化图建模框架。

---

## 1. 问题背景

多变量时间序列插补的目标是在给定部分观测序列和观测掩码的条件下，恢复窗口内部缺失位置的真实值。设输入序列为：

```text
X in R^{T x N x C}
M in {0,1}^{T x N x C}
```

其中 `T` 是时间长度，`N` 是变量或传感器数量，`C` 是特征维度，`M=1` 表示该位置可观测。插补模型学习：

```text
F(X_obs, M) -> X_hat
```

训练和评估只在人工遮蔽且原始真实可观测的位置计算误差：

```text
loss = MAE(X_hat, X_true) on eval_mask
```

此前 0723 实验已经证明，在官方 P-GUTS 插补代码路径上加入 coarse graph 分支后，METR-LA 上 MAE 明显下降，但 PEMS-BAY 上出现退化。这说明区域化粗图有潜力，但简单的单尺度粗图和固定融合并不稳定。

因此，下一步不应只是继续给 P-GUTS 加一个粗图分支，而应把问题推进为：

> 在缺失观测条件下，如何同时从空间依赖和时序依赖中构造多尺度区域化图，并以缺失可靠的方式融合这些图信息，从而提升插补质量并减少错误粗化带来的过平滑。

---

## 2. 核心 Idea

本文方法可以概括为：

> 对缺失观测多变量时间序列，同时构造空间依赖路和时序依赖路；每一路都包含细粒度图和多尺度区域化粗图；再通过缺失可靠性控制两路、多尺度、多区域分支的融合，完成窗口内缺失值插补。

方法暂定名：

```text
DR-MSRGI
Dual-Route Multi-Scale Regional Graph Imputation
```

其中：

- `Dual-Route`：同时建模空间依赖和时序依赖。
- `Multi-Scale`：不是单一 coarse graph，而是多个区域尺度。
- `Regional Graph`：把变量节点或时间片段聚合为区域节点，再进行图传播。
- `Imputation`：主任务是窗口内插补，而不是未来预测。

整体框架：

```text
X_obs, M
  -> mask-aware value encoder
  -> spatial dependency route
       -> fine spatial graph
       -> multi-scale spatial regional graphs
  -> temporal dependency route
       -> fine temporal graph
       -> multi-scale temporal regional graphs
  -> missingness-reliable dual-route fusion
  -> imputation decoder
  -> X_hat
```

---

## 3. 与 0723 实验的继承关系

0723 实验中的模型对照是：

```text
P-GUTS [3,6]
  = temporal pooling [3,6]
  + full graph branch
  + fixed concat/MLP fusion

CG-P-GUTS [3,6]
  = temporal pooling [3,6]
  + full graph branch
  + one coarse graph branch
  + fixed concat/MLP fusion
```

那轮实验给出两个重要结论：

1. METR-LA 上 coarse graph 明确有效，说明空间粗粒度低频关系可以帮助缺失插补。
2. PEMS-BAY 上 coarse graph 退化，说明粗图质量和融合方式会影响结果，简单固定融合可能造成过平滑。

新方案在此基础上做两点深化：

| 0723 实验 | 新方案 |
|---|---|
| 单个 coarse graph | 多尺度 spatial regional graphs |
| 主要利用空间图粗化 | 同时构造 spatial route 和 temporal route |
| fixed concat/MLP fusion | missingness-reliable residual/gated fusion |
| 只验证 coarse graph 是否有效 | 分析哪一路、哪一尺度、哪种缺失条件下有效 |

因此，新方案不是推翻 0723，而是把 0723 的“coarse graph 信号”系统化。

---

## 4. 核心假设

### H1：高缺失率下，细粒度节点关系不稳定

当缺失率升高，变量之间的相关性估计会受到观测重叠不足的影响。特别是在连续块缺失、传感器故障、区域缺失等情况下，细粒度节点之间的边可能变得不可靠。

因此，模型需要更稳定的低频区域关系作为补充。

### H2：空间区域关系和时序区域关系是互补的

空间图描述传感器、变量或实体之间的结构邻近性，例如交通路网距离。它适合捕获局部空间扩散、相邻区域同步变化等模式。

时序图描述时间片段之间的动态相似性、周期性、滞后依赖和缺失条件下的时间上下文。它适合处理长时间缺失、周期恢复和跨时间片段补偿。

仅使用空间图可能无法恢复长时间连续缺失，仅使用时间图又可能忽略空间邻近节点的协同变化。因此，两路依赖应同时存在。

### H3：多尺度区域图比单尺度粗图更稳

单一 coarse graph 的效果强依赖区域划分质量。如果尺度过粗，容易过平滑；如果尺度过细，又不能提供足够低频补偿。

因此需要多个尺度，例如：

```text
spatial scale: fine -> local region -> macro region
temporal scale: time step -> short segment -> long segment
```

模型可以根据缺失模式选择更合适的尺度。

### H4：区域信息必须受缺失可靠性控制

粗图并不总是有益。0723 实验中 PEMS-BAY 的退化表明，如果区域划分不匹配真实动态模式，区域分支可能注入错误低频信息。

因此，区域分支应以残差形式注入，并由观测覆盖率、区域一致性、边可靠性、缺失块长度等统计量控制。

---

## 5. 模型设计

### 5.1 Mask-Aware Input Encoder

输入不仅包含观测值，还包含 mask 信息：

```text
E = Encoder(concat(X_obs, M, time_embedding, optional_node_embedding))
```

其中 `X_obs = X * M`。mask 不是简单辅助特征，而是后续图构造和融合门控的重要可靠性信号。

可选输入特征：

| 特征 | 作用 |
|---|---|
| observed value | 已观测数值 |
| mask | 标识哪些位置可信 |
| time embedding | 捕获小时、星期、周期位置 |
| node embedding | 捕获变量或传感器身份 |
| missing statistics | 给融合门控提供缺失状态 |

---

## 6. 空间依赖路：Spatial Regional Graph Route

空间依赖路以变量或传感器为图节点，主要捕获跨变量关系。

### 6.1 细粒度空间图

如果数据集提供真实空间信息，则构造物理邻接图：

```text
A_phy
```

例如交通数据中可由道路距离、传感器距离或路网连接关系得到。

若空间图不可用，可以退化为数据驱动变量图：

```text
A_var = TopK(Corr_observed(X, M))
```

但本方向优先使用具备空间信息的数据集，因为核心 idea 依赖空间区域化建模。

### 6.2 多尺度空间区域化

构造多个变量到区域的分配矩阵：

```text
S_v^{(1)}, S_v^{(2)}, ..., S_v^{(L_v)}
```

其中每个 `S_v^{(l)} in R^{N x K_l}`，`K_l < N`，表示第 `l` 个空间区域尺度。

候选区域化方式：

| 方法 | 说明 |
|---|---|
| distance clustering | 基于道路距离或传感器距离聚类 |
| graph community | 基于物理图社区发现 |
| spectral clustering | 基于图拉普拉斯谱聚类 |
| temporal-consistency clustering | 用动态相似性修正空间区域 |
| hybrid clustering | 同时考虑空间距离和观测时序相似性 |
| random clustering | 负对照，验证收益不是参数量带来的 |

对于每个空间尺度，区域级表示为：

```text
H_v_reg^{(l)} = normalize(S_v^{(l)T}) H
```

区域级邻接为：

```text
A_v_reg^{(l)} = S_v^{(l)T} A_phy S_v^{(l)}
```

经过区域图传播后再提升回变量级：

```text
Z_v_reg^{(l)} = S_v^{(l)} GNN(H_v_reg^{(l)}, A_v_reg^{(l)})
```

空间依赖路输出：

```text
Z_spatial = Fuse_v(
  GNN(H, A_phy),
  Z_v_reg^{(1)},
  ...,
  Z_v_reg^{(L_v)}
)
```

---

## 7. 时序依赖路：Temporal Regional Graph Route

时序依赖路以时间步或时间片段为图节点，主要捕获跨时间关系。它回答的问题是：在某个时间位置缺失时，哪些历史或邻近时间片段最能帮助恢复它。

### 7.1 细粒度时间图

构造时间步之间的依赖图：

```text
A_time in R^{T x T}
```

可选构造方式：

| 时间边来源 | 含义 |
|---|---|
| local adjacency | 相邻时间步连接 |
| periodic adjacency | 同周期位置连接，例如相同小时 |
| lag correlation | 基于观测值估计滞后相关 |
| mask-overlap reliability | 只信任共同观测充足的时间关系 |
| learnable attention | 由模型学习动态时间边 |

观测重叠可靠性可以定义为：

```text
R_time[i,j] = observed_overlap(time_i, time_j)
```

然后对时间图加权：

```text
A_time_reliable = A_time * R_time
```

### 7.2 多尺度时间区域化

与空间区域化类似，时间依赖路也构造多尺度时间区域：

```text
S_t^{(1)}, S_t^{(2)}, ..., S_t^{(L_t)}
```

其中 `S_t^{(l)} in R^{T x Q_l}`，把时间步聚合为不同长度的时间区域。

可选时间区域包括：

| 时间尺度 | 例子 | 作用 |
|---|---|---|
| short segment | 3、6 个时间步 | 捕获局部连续变化 |
| medium segment | 12 个时间步 | 对应半小时或一小时级上下文 |
| long segment | 24 个时间步 | 捕获日内周期或完整窗口趋势 |
| adaptive segment | 根据缺失块边界划分 | 对连续缺失更敏感 |

时间区域表示：

```text
H_t_reg^{(l)} = normalize(S_t^{(l)T}) H
```

时间区域邻接：

```text
A_t_reg^{(l)} = S_t^{(l)T} A_time_reliable S_t^{(l)}
```

区域传播并提升回时间步：

```text
Z_t_reg^{(l)} = S_t^{(l)} GNN(H_t_reg^{(l)}, A_t_reg^{(l)})
```

时序依赖路输出：

```text
Z_temporal = Fuse_t(
  GNN(H, A_time_reliable),
  Z_t_reg^{(1)},
  ...,
  Z_t_reg^{(L_t)}
)
```

---

## 8. 双路交互：Spatial-Temporal Product Graph View

空间路和时序路不应完全独立，因为插补位置同时由“哪个变量缺失”和“哪个时间缺失”决定。

可以采用三种交互方式，按实现复杂度递增：

### 8.1 Late Fusion

分别得到空间路和时序路输出，再融合：

```text
Z = Fuse(Z_spatial, Z_temporal, H)
```

这是第一版最稳实现。

### 8.2 Alternating Propagation

交替进行空间传播和时间传播：

```text
H_1 = SpatialRoute(H)
H_2 = TemporalRoute(H_1)
H_3 = SpatialRoute(H_2)
```

这更接近时空图网络，但计算更重。

### 8.3 Product Graph Propagation

显式构造空间-时间乘积图：

```text
A_st = A_time kron A_space
```

或者在 P-GUTS 风格中通过 separable graph operation 近似：

```text
TemporalGraph(SpatialGraph(H))
```

这与 P-GUTS 的 product graph 思想最接近，但需要谨慎控制显存。

第一阶段建议采用 `Late Fusion`，因为它最适合验证“两路多尺度区域图是否有收益”这个核心问题。

---

## 9. 缺失可靠融合机制

区域分支采用残差注入，而不是无条件替代细粒度分支：

```text
Z_out = Z_base
      + alpha_spatial * Project(Z_spatial_region)
      + alpha_temporal * Project(Z_temporal_region)
```

其中 `alpha_spatial` 和 `alpha_temporal` 由缺失可靠性统计量产生：

```text
alpha = Gate(mask_stats, region_stats, edge_reliability)
```

候选可靠性统计量：

| 统计量 | 说明 |
|---|---|
| global_observed_ratio | 当前窗口整体观测率 |
| node_observed_ratio | 每个变量的观测率 |
| time_observed_ratio | 每个时间步的观测率 |
| longest_missing_run | 连续缺失长度 |
| spatial_region_coverage | 空间区域内可观测节点比例 |
| temporal_region_coverage | 时间区域内可观测时间步比例 |
| edge_overlap_reliability | 两节点或两时间片段共同观测比例 |
| region_consistency | 区域内部动态是否一致 |

推荐第一版使用简单、稳定的残差门控：

```text
alpha = sigmoid(MLP(mask_stats))
Z_out = Z_fine + alpha * Z_region
```

并将 `alpha` 初始化为接近 0，使模型从 P-GUTS 或 fine-only 基线出发，逐步学习何时使用区域信息。这可以降低 PEMS-BAY 式错误粗化的风险。

---

## 10. 插补解码器与训练目标

最终输出：

```text
X_hat = Decoder(Z_out)
```

主损失：

```text
L_imp = MAE(X_hat, X_true, eval_mask)
```

可选辅助损失：

### 10.1 Observed Reconstruction Loss

在少量 observed 位置上做重构正则，防止表示偏离真实观测：

```text
L_obs = MAE(X_hat, X_true, sampled_observed_mask)
```

### 10.2 Multi-Scale Consistency Loss

约束细尺度预测和区域尺度预测在区域聚合后保持一致：

```text
L_cons = || S^T X_hat_fine - X_hat_region ||_1
```

### 10.3 Region Smoothness Regularization

仅在区域内部动态一致性较高时使用，避免强行平滑：

```text
L_smooth = sum A_reg[i,j] || H_reg[i] - H_reg[j] ||
```

第一版建议只使用：

```text
L = L_imp
```

等主效果明确后，再加入 `L_cons` 或 `L_obs`。

---

## 11. 与相关工作的关系

### 11.1 P-GUTS

P-GUTS 使用 product graph U-networks 和 temporal pooling 做时空插补，证明了图结构和时间池化对插补有效。本文继承其插补任务设定和多尺度思想，但进一步引入：

- 空间区域化多尺度图；
- 时序区域化多尺度图；
- 缺失可靠融合；
- 对粗图收益与失效条件的显式分析。

### 11.2 HD-TTS

HD-TTS 通过时空下采样在含缺失数据的图预测中获得多尺度表示，支持“缺失条件下粗尺度时空信息有价值”这一假设。本文借鉴其多尺度时空视角，但任务回到插补，并采用 P-GUTS 风格的窗口内重构目标。

### 11.3 SPIN / GRIN

SPIN 和 GRIN 是典型的图式时空插补方法，说明空间图消息传递对插补有价值。本文与它们的区别在于，不只在细粒度传感器图上传播，而是同时构造空间区域图和时间区域图。

### 11.4 ImputeFormer

ImputeFormer 强调低秩和全局结构先验对时空插补的重要性。本文的区域图也可以视为一种结构化低频先验，但它显式依赖空间/时间图和缺失可靠性，而不是只依赖 Transformer 全局建模。

### 11.5 Graph Coarsening / Graph U-Nets

图粗化和 pool-unpool 操作为区域图提供基础。本文的关键区别是，粗化不是通用图分类里的结构压缩，而是面向缺失插补的可靠性补偿机制。

---

## 12. 实验设计建议

### 12.1 主数据集

优先选择具有空间信息的数据集：

| 数据集 | 作用 |
|---|---|
| METR-LA | 0723 已证明 coarse graph 有效，是正向验证集 |
| PEMS-BAY | 0723 已暴露 coarse graph 退化，是稳定性挑战集 |
| AQI / AQI36 | 可作为空气质量空间传感器插补扩展 |

### 12.2 主对照

主对照应保持简洁：

| 方法 | 目的 |
|---|---|
| P-GUTS `[3,6]` | 官方强基线 |
| CG-P-GUTS | 0723 单粗图版本 |
| Spatial-MSRG | 只做空间多尺度区域图 |
| Temporal-MSRG | 只做时序多尺度区域图 |
| Dual-MSRG | 空间 + 时序双路多尺度区域图 |
| Dual-MSRG-Gated | 加缺失可靠门控的完整模型 |

### 12.3 必要消融

| 消融 | 验证问题 |
|---|---|
| w/o spatial regional graphs | 空间区域图是否必要 |
| w/o temporal regional graphs | 时序区域图是否必要 |
| w/o gate | 缺失可靠门控是否抑制错误粗化 |
| single-scale region | 多尺度是否优于单尺度 |
| random region | 收益是否来自真实区域结构 |
| physical-only region | 只用空间距离是否足够 |
| temporal-only region | 只用动态相似性是否足够 |
| hybrid region | 空间 + 时序区域是否最稳 |

### 12.4 缺失模式

插补实验应覆盖：

| 缺失模式 | 目的 |
|---|---|
| random point | 检验普通稀疏缺失 |
| temporal block | 检验连续时间缺失 |
| spatial block | 检验局部区域或传感器群缺失 |
| spatiotemporal block | 检验空间-时间联合缺失 |
| mixed missing | 检验真实复杂缺失 |

### 12.5 关键分析

除了 MAE/RMSE，还应分析：

1. 不同缺失率下，区域分支 gate 是否增强。
2. METR-LA 与 PEMS-BAY 的区域质量差异。
3. random region 是否无法复现真实区域收益。
4. temporal regional graph 是否主要改善长连续缺失。
5. spatial regional graph 是否主要改善空间块缺失。
6. 多尺度中哪一层贡献最大。

---

## 13. 第一版实现路线

为了控制复杂度，第一版建议从官方 P-GUTS 代码路径继续，而不是重新写完整模型。

### Stage 1：从 CG-P-GUTS 扩展为空间多尺度粗图

在 0723 的 `CoarseGraphBranch` 基础上支持多个 `S_v`：

```text
coarse_scales = [2, 4, 8]
```

输出多个空间区域分支后融合：

```text
Z_spatial_region = FuseScale(Z_v_reg^{(1)}, Z_v_reg^{(2)}, Z_v_reg^{(3)})
```

验收问题：

> 多尺度空间区域图是否比单尺度 CG-P-GUTS 更稳，尤其是否缓解 PEMS-BAY 退化？

### Stage 2：加入时序区域图分支

在时间维度构造 `S_t`，例如：

```text
temporal_scales = [3, 6, 12]
```

从输入窗口内按时间片段聚合，做时间区域图传播，再提升回时间步。

验收问题：

> 时序区域图是否改善 temporal block 和 long missing run 下的插补？

### Stage 3：加入缺失可靠门控

将空间区域分支和时序区域分支都改成残差门控：

```text
Z = Z_pguts
  + alpha_v * Z_spatial_region
  + alpha_t * Z_temporal_region
```

其中：

```text
alpha_v = Gate_v(mask_stats, spatial_region_coverage)
alpha_t = Gate_t(mask_stats, temporal_region_coverage)
```

验收问题：

> gate 是否能保留 METR-LA 收益，同时减少 PEMS-BAY 上错误粗图导致的退化？

### Stage 4：做区域质量分析

统计每个区域的：

- 组内平均空间距离；
- 组内平均观测相关；
- 组内缺失覆盖率；
- 组内动态一致性；
- gate 平均值。

验收问题：

> 区域质量是否能解释 coarse graph 何时有效、何时有害？

---

## 14. 预期贡献表述

如果实验结果成立，可以将贡献写成：

1. 提出一种面向缺失观测多变量时间序列插补的双路依赖建模框架，同时利用空间依赖和时序依赖。
2. 将单一粗图分支扩展为多尺度区域化图建模，在变量空间和时间维度上分别构造区域级关系。
3. 设计缺失可靠的区域信息融合机制，根据观测覆盖率、缺失块结构和区域一致性调节粗尺度信息注入。
4. 在交通和空气质量等具有空间信息的数据集上验证，分析区域图在不同缺失模式下的有效性与失效条件。

更稳妥的 novelty 表述是：

> 本文不声称区域图总是提升插补性能，而是提出一种缺失可靠的双路多尺度区域图框架，用于识别并利用高缺失条件下仍然稳定的空间和时序低频依赖，同时抑制错误区域划分造成的过平滑。

---

## 15. 风险与应对

| 风险 | 可能原因 | 应对 |
|---|---|---|
| PEMS-BAY 继续退化 | 空间区域划分不匹配动态模式 | 加 hybrid / temporal clustering 和 gate |
| 多尺度分支只带来参数量收益 | 模型容量增加而非区域结构有效 | 加 random region 和 parameter-matched baseline |
| 时序区域图收益不明显 | P-GUTS temporal pooling 已覆盖部分时间尺度 | 在 temporal block / long missing run 下重点评估 |
| 显存开销过大 | product graph 或多尺度分支过多 | 第一版使用 late fusion 和少量尺度 |
| gate 学不会有效选择 | 缺失统计量太粗 | 加区域覆盖率、edge overlap、region consistency |

---

## 16. 当前最小可行实验

为了快速验证 idea，建议首轮只跑：

```text
datasets = METR-LA, PEMS-BAY
mask = paper BLOCK + temporal block + spatiotemporal block
methods = P-GUTS, CG-P-GUTS, Spatial-MSRG, Dual-MSRG-Gated
seeds = 1, 2
```

核心比较：

```text
P-GUTS
  vs CG-P-GUTS
  vs Spatial-MSRG
  vs Dual-MSRG-Gated
```

首轮要回答三个问题：

1. 多尺度空间区域图是否比单尺度 coarse graph 更稳？
2. 加入时序区域图是否改善连续时间缺失？
3. 缺失可靠门控是否能减少 PEMS-BAY 上的过平滑退化？

如果这三个问题中至少两个成立，就可以继续扩展 AQI/AQI36 和更完整的消融矩阵。


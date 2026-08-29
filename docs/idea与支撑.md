# 观测缺失多变量时间序列预测：Idea 框架与外部文献支撑调研

> 日期：2026-08-28  
> 目的：不局限于本地论文集，基于网络检索梳理可支撑本文 idea 的相关工作，并判断是否已有完全同构论文。  
> 研究场景：高缺失率、中低变量维度、多变量时间序列预测，重点关注真实空间图、变量时序图、区域化聚类/粗图、多尺度关系与缺失适配。

## 1. 我们的 Idea 框架

当前 idea 可以规范化为：

> 在观测值缺失的多变量时间序列预测中，缺失不仅破坏单变量时间上下文，也会降低变量间关系估计的可靠性。因此，模型不应只依赖细粒度变量图，而应同时建模细粒度变量关系与区域化变量关系；同时结合真实空间信息和变量时序信息，并用缺失模式控制不同关系尺度的使用强度。

可以拆成 5 个模块：

| 模块 | 核心问题 | 对应设计 |
|---|---|---|
| M1 缺失预测范式 | 是否先插补再预测 | 以 forecasting loss 为主，尽量 direct / imputation-free，必要时只用轻量辅助重构或一致性正则 |
| M2 真实空间关系 | 物理相邻是否有用 | 对 METR-LA、PEMS-BAY 等使用道路距离图或传感器图 `A_phy` |
| M3 变量时序关系 | 物理相近不一定动态相似 | 从 observed-only 历史片段构造变量相关图 `A_temp`，并考虑 overlap reliability |
| M4 区域化/聚类关系 | 高缺失率下细粒度关系不稳定 | 用变量聚类得到区域节点，构造 coarse graph / regional graph |
| M5 缺失模式适配 | 粗图可能有益，也可能过平滑 | 用 mask statistics、group coverage、edge reliability 控制 fine/coarse 融合 |

建议的技术主线是：

```text
X_obs, M
  -> observed-value normalization + mask-aware encoder
  -> physical graph A_phy + observed temporal graph A_temp
  -> variable-to-region assignment S
  -> fine graph branch + regional graph branch
  -> missingness-reliable residual fusion
  -> direct forecasting
```

## 2. 是否已有完全相同思路的论文

网络检索使用了如下关键词组合：

- `"multivariate time series forecasting" "missing values" "graph" "cluster"`
- `"incomplete multivariate time series forecasting" "spatial graph" "temporal graph"`
- `"missing data" "multivariate time series forecasting" "coarse graph"`
- `"missingness" "multi-scale graph" "time series forecasting"`
- `"graph coarsening" "time series forecasting" "missing values"`
- `"graph pooling" "missing data" "time series forecasting"`

检索结论：

**没有发现完全同构论文**，即没有发现同时满足以下条件的工作：

```text
1. 主任务是观测缺失条件下的 multivariate time series forecasting
2. 显式使用真实空间/物理图 A_phy
3. 同时构造 observed-value temporal dependency graph A_temp
4. 通过变量聚类构造 regional / coarse graph
5. 用缺失模式或可靠性统计控制 fine/coarse 融合
6. 以 forecasting-oriented 方式论证区域化关系
```

最接近的工作覆盖的是子问题：

- HD-TTS 覆盖“缺失预测 + 时空多尺度/层级粗化”，但不是“真实空间图 + 时序相关图 + 变量聚类区域图”的统一框架。
- BiTGraph 覆盖“缺失预测 + 图传播 + mask bias”，但不做区域化粗图。
- CRIB / CoIFNet / MissTSM 覆盖“缺失预测范式”，但不显式结合物理图和区域图。
- HyperIMTS / GraFITi / T-PatchGNN 覆盖“不规则/缺失观测的图结构建模”，但不是物理-时序-区域多图融合。
- MTGNN / ForecastGrapher / ESG 覆盖“变量关系图学习”，但默认缺失适配较弱。

因此，当前最稳妥的 novelty 表述是：

> 现有工作分别研究了缺失预测、图式多变量预测、不规则时间序列图建模、多尺度时空下采样和图粗化；但尚缺少一种面向缺失多变量预测的统一框架，同时融合真实空间关系、观测鲁棒的变量时序关系和预测导向的区域化粗粒度关系，并显式考虑缺失模式对关系可靠性的影响。

## 3. 支撑文献图谱

### 3.1 缺失预测范式：支撑 M1

| 文献 | 关键信息 | 支撑点 | 与我们不同 |
|---|---|---|---|
| CRIB: Revisiting Multivariate Time Series Forecasting with Missing Values | 指出 imputation-then-prediction 缺少真实缺失值监督，插补可能破坏数据分布和预测准确性；提出直接从 partially observed time series 预测。来源：https://arxiv.org/abs/2509.23494 | 支撑“不把历史插补作为主目标，而是预测导向表征”；支撑一致性正则和变量相关性保护 | 未引入真实空间图、时序图、区域化粗图 |
| CoIFNet: A Unified Framework for MTSF with Missing Values | 输入 observed values、mask、timestamp embeddings，通过 CTF/CVF 捕获缺失鲁棒时间依赖；报告 point/block missing rate 0.6 下的性能优势。来源：https://arxiv.org/abs/2506.13064 | 支撑 mask 应作为特征进入模型；支撑联合时间/变量融合 | 不构造显式图结构，也无区域图 |
| MissTSM | model-agnostic、imputation-free；面向 irregularly-sampled MTS；高缺失和弱周期场景有优势。来源：https://arxiv.org/abs/2502.15785 | 支撑 adapter / imputation-free 路线；支撑缺失条件下 token 构造要先鲁棒 | 跨变量关系较压缩，不建图 |
| S4M | 针对 block missing，引入 Adaptive Temporal Prototype Mapper 和 Missing-Aware Dual Stream S4。来源：https://arxiv.org/abs/2503.00900 | 支撑高缺失、block missing 下需要专门缺失模式建模 | 重点是状态空间时间建模，不是图/区域关系 |
| Merlin | 多视图表示学习，处理 unfixed missing rates；用蒸馏和对比学习对齐不同缺失率下语义。来源：https://arxiv.org/abs/2506.12459 | 支撑缺失率泛化应单独评估；可借鉴多缺失率训练/一致性学习 | 不关注空间图和区域图 |
| Task-oriented Time Series Imputation Evaluation | 强调插补结果应按下游任务评估，而不是只看重构误差。来源：https://arxiv.org/abs/2410.06652 | 支撑“预测导向”评价口径，避免只证明插补好 | 是评价框架，不是预测模型 |

### 3.2 图式多变量预测：支撑 M2/M3

| 文献 | 关键信息 | 支撑点 | 与我们不同 |
|---|---|---|---|
| MTGNN / Connecting the Dots | 将变量视为图节点，自动学习变量间单向关系，并可融合外部知识。来源：https://arxiv.org/abs/2005.11650 | 支撑变量图建模；支撑外部知识和数据驱动图可以结合 | 默认不是缺失预测；没有缺失可靠边 |
| STFGNN | 构造 data-driven temporal graph 来补充给定 spatial graph 的局限，并融合多种 spatial/temporal graphs。来源：https://arxiv.org/abs/2012.09641 | 直接支撑“真实空间图 + 数据驱动时序图”并非重复，而是互补 | 交通完整预测场景，缺失适配不足 |
| ForecastGrapher | 将 MTSF 重新定义为 node regression，构造 adaptive adjacency 捕获 inter-series correlations。来源：https://arxiv.org/abs/2405.18036 | 支撑 MTSF 可从图节点回归角度建模变量关系 | 不面向缺失，也不做区域 coarse graph |
| MTSF-DG / Multiple Time Series Forecasting with Dynamic Graph Modeling | 学习历史关系图并预测未来关系图，以捕获动态相关。来源：https://dl.acm.org/doi/abs/10.14778/3636218.3636230 | 支撑变量关系不是静态的，时序图可动态变化 | 不以缺失值为核心问题 |
| SAGDFN | 自适应图扩散预测，强调大规模 MTSF 中捕获复杂时空相关。来源：https://arxiv.org/abs/2406.12282 | 支撑图扩散和可扩展图学习 | 主要是规模扩展，不是缺失可靠性 |
| D2STGNN | 将交通信号分解为 diffusion signals 和 inherent signals，并学习动态交通图。来源：https://arxiv.org/abs/2206.09112 | 支撑物理图传播和变量自身动态应区分 | 不是缺失预测 |

### 3.3 缺失/不规则场景下的图结构建模：支撑 M1/M3

| 文献 | 关键信息 | 支撑点 | 与我们不同 |
|---|---|---|---|
| BiTGraph | 面向 time series forecasting with missing values，用 mask-biased temporal convolution graph network 处理缺失。来源：https://github.com/chenxiaodanhit/BiTGraph | 支撑缺失预测中图传播需要 mask-aware bias；重要 baseline | 主要是细粒度图，不显式区域化 |
| GraFITi | 将 irregularly sampled time series with missing values 转为 sparsity structure graph，并把预测看作 edge weight prediction。来源：https://arxiv.org/abs/2305.12932 | 支撑保留原始观测稀疏结构，不必先补齐完整矩阵 | 图是观测事件图，不是变量空间/区域图 |
| T-PatchGNN | ICML 2024；用 transformable patching + GNN 处理 asynchronous IMTS，建模异步变量相关。来源：https://proceedings.mlr.press/v235/zhang24bw.html | 支撑 patch 化和 GNN 可以结合处理缺失/不规则变量关系 | 专用 IMTS 架构，缺少真实空间图和区域图 |
| HyperIMTS | ICML 2025；将观测值转为超图节点，用 temporal 和 variable hyperedges 统一建模依赖。来源：https://arxiv.org/abs/2505.17431 | 支撑不规则/缺失观测可用高阶图表示，且能 time-adaptive 捕获变量依赖 | 不研究物理图和变量聚类粗图 |
| ChannelTokenFormer | ICLR 2026；统一处理 channel dependency、asynchrony、missingness，用 channel token 和 mask-guided attention。来源：https://arxiv.org/abs/2506.08660 | 支撑真实场景需同时考虑通道依赖、异步和缺失；支撑通道级关系显式建模 | 不是图粗化/区域聚类路线 |
| Dynamic Attention Graph Network for incomplete MTSF | 2026 年 Neurocomputing 文章，题名为 Incomplete multivariate time series forecasting with dynamic attention graph network。来源：https://www.sciencedirect.com/science/article/abs/pii/S092523122602093X | 最新相邻工作，说明 incomplete MTSF + dynamic graph attention 已经出现 | 需要进一步读全文确认是否有区域图；从题名和摘要线索看仍不像我们的完整组合 |

### 3.4 多尺度、层级图、区域粗化：支撑 M4/M5

| 文献 | 关键信息 | 支撑点 | 与我们不同 |
|---|---|---|---|
| HD-TTS / Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling | ICML 2024；将输入在时间和空间上逐步 coarsen，得到多尺度表示；条件化观测和缺失模式，用可解释 attention 生成预测；对 contiguous missing blocks 尤其有效。来源：https://arxiv.org/abs/2402.10634 | 最直接支撑“高缺失率下粗时间/粗空间尺度有价值”；支撑缺失模式应控制多尺度融合 | 偏传感器时空图，不研究物理图 + 时序图 + 变量聚类区域图统一 |
| ESG / Learning Evolutionary and Multi-scale Graph Structure | 认为变量交互随时间演化，且不同时间尺度下相关性不同；构造 hierarchical graph structure 捕获 scale-specific correlations。来源：https://arxiv.org/abs/2206.13816 | 支撑“不只看细尺度变量关系”，变量关系应有多尺度结构 | 未针对缺失观测适配 |
| Hierarchical Joint Graph Learning | 将多变量信号表示为图节点，并用 hierarchical signal decomposition 捕获多重空间依赖。来源：https://arxiv.org/abs/2311.12630 | 支撑层级/区域化图关系对 MTSF 有效 | 不是缺失预测 |
| Graph U-Nets | 提出 gPool/gUnpool，在图上做 encoder-decoder 式池化和反池化。来源：https://arxiv.org/abs/1905.05178 | 为 `S^T H` / `S H_reg` 这类 pool-unpool 结构提供通用图学习依据 | 通用图任务，不是 MTSF |
| Graph Coarsening with Neural Networks | 讨论 graph coarsening 的质量度量、coarse graph、projection/lift operator。来源：https://arxiv.org/abs/2102.01350 | 支撑粗图不是随便聚类，需要考虑投影、提升和粗化质量 | 非时间序列预测任务 |
| P-GUTS | 用 temporal pooling 和 product graph U-networks 做 spatiotemporal imputation；强调不同数据集冗余度不同，单一 pooled view 可能受限。来源：https://kdd.org/kdd2025/wp-content/uploads/2025/07/CameraReady-18.pdf | 支撑 pooled/coarse representation 对缺失时空数据有价值；支撑多视图 | 主任务是插补，不是 forecasting 主目标 |
| ImputeFormer | 用低秩先验和 Transformer 做 generalizable spatiotemporal imputation。来源：https://arxiv.org/abs/2312.01728 | 支撑缺失场景下结构先验可稳定信号、抑制噪声 | 是插补和低秩先验，不是区域图预测 |
| SPIN | Graph attention spatiotemporal imputation，指出 GRIN 类 autoregressive imputation 可能误差传播。来源：https://papers.neurips.cc/paper_files/paper/2022/file/cf70320e93c08b39b1b29a348097a376-Paper-Conference.pdf | 支撑 sparse observations 下图结构插补/重建重要，也支撑误差传播问题 | 主任务为 imputation |

## 4. 文献如何支撑我们的具体设计

### 4.1 为什么不能只做先插补再预测

证据来源：

- CRIB：插补缺少真实监督，可能损害数据分布和预测。
- CoIFNet：两阶段目标错位，mask 和 timestamp 应进入预测表征。
- Task-oriented imputation evaluation：插补应按下游任务效果评价。
- S4M：block missing 下两阶段方法容易累积错误。

设计落点：

```text
主任务 = forecasting
历史重构 = 可选辅助
mask = 输入特征 / 可靠性信号
```

### 4.2 为什么需要真实空间图 + 变量时序图

证据来源：

- MTGNN：变量间潜在依赖对 MTSF 关键，且可结合外部知识。
- STFGNN：给定 spatial graph 不够，data-driven temporal graph 可补充真实空间图未覆盖的隐依赖。
- ForecastGrapher / MTSF-DG：变量关系可以动态或自适应学习。

设计落点：

```text
A_phy: 真实空间/物理邻接
A_temp: observed-only temporal dependency
A_fine = fuse(A_phy, A_temp)
```

### 4.3 为什么需要区域化/粗粒度变量关系

证据来源：

- HD-TTS：缺失情况下，通过时空下采样得到多尺度表示，尤其对 contiguous block missing 有优势。
- ESG：变量关系在不同时间尺度下不同，需要 multi-scale graph。
- Hierarchical Joint Graph Learning：层级图分解可捕获多重空间依赖。
- Graph U-Nets / Graph Coarsening：pooling/coarsening 是图上获得粗粒度表示的成熟机制。
- P-GUTS：pooled representations 和多视图对缺失时空建模有价值。

设计落点：

```text
S: variable-to-region assignment
H_region = S^T H_node
A_region = S^T A_fine S
H_node_from_region = S GNN(H_region, A_region)
```

### 4.4 为什么粗图必须缺失适配，而不能直接拼接

证据来源：

- HD-TTS：多尺度表示需要 conditioned on observations and missing data patterns 组合。
- CoIFNet / ChannelTokenFormer：mask-guided 或 mask-aware attention 对真实缺失场景重要。
- 本地已有实验：coarse graph 在 METR-LA 上有正向信号，但在 PEMS-BAY 上会因分组质量或过平滑退化。

设计落点：

```text
Z = Z_fine + alpha(M, S, reliability) * Project(Z_region_to_node)
```

其中 `alpha` 应根据：

- 全局观测率
- 单变量观测率
- 区域内观测覆盖率
- 最长连续缺失长度
- 变量对 observed overlap
- 区域内动态一致性

## 5. 与最接近工作的差异矩阵

| 方法 | 缺失预测 | 真实空间图 | 时序相关图 | 区域/粗图 | 缺失模式控制融合 | 预测导向 |
|---|---|---|---|---|---|---|
| CRIB | 是 | 否 | 隐式 attention | 否 | consistency/IB | 是 |
| CoIFNet | 是 | 否 | 隐式变量融合 | 否 | mask input | 是 |
| MissTSM | 是 | 否 | 同时间步 masked attention | 否 | masked attention | 是/多任务 |
| S4M | 是 | 否 | 隐式 S4 表征 | 否 | mask dual stream | 是 |
| BiTGraph | 是 | 是/图结构 | 可学习图 | 否 | mask-biased graph | 是 |
| HD-TTS | 是 | 是 | 隐式 | 是，时空下采样 | attention conditioned on missing patterns | 是 |
| HyperIMTS | 是，不规则 | 否 | 超图 variable hyperedge | 否 | irregularity-aware | 是 |
| GraFITi | 是，不规则 | 否 | 观测事件图 | 否 | 稀疏图结构 | 是 |
| STFGNN | 否/弱 | 是 | 是 | 否 | 否 | 是 |
| ESG | 否/弱 | 否/可学习 | 是 | hierarchical multi-scale graph | 否 | 是 |
| 我们的方向 | 是 | 是，可选 | 是，observed-only + reliability | 是，变量聚类区域图 | 是，missingness-reliable residual gate | 是 |

结论：我们的区别不是“用了图”或“用了多尺度”，而是 **把缺失可靠性、双源图关系、区域化变量关系和预测目标放在同一个框架中**。

## 6. 现在最可支撑的论文核心假设

### H1：高缺失率下，细粒度变量关系的可靠性下降

支撑：

- CRIB：缺失和插补会破坏变量相关性。
- CoIFNet / S4M：高缺失与 block missing 会显著影响预测，需专门建模。
- 设计推断：`A_temp` 需要 observed overlap reliability。

可验证实验：

- 对不同缺失率下的变量相关矩阵做稳定性分析。
- 统计 `corr_full` vs `corr_observed` 的偏差。
- 看低 overlap 边是否更容易导致预测退化。

### H2：真实空间图和变量时序图互补

支撑：

- MTGNN：外部知识可与图学习结合。
- STFGNN：spatial graph 不能覆盖所有 hidden dependencies，temporal graph 有补充价值。
- MTSF-DG / ForecastGrapher：变量依赖可动态/自适应学习。

可验证实验：

```text
A_phy only
A_temp only
A_phy + A_temp
```

### H3：区域化变量关系能在高缺失率下提供稳定低频补充

支撑：

- HD-TTS：时空下采样对 contiguous missing blocks 有效。
- ESG / Hierarchical Joint Graph Learning：多尺度/层级图捕获不同尺度变量关系。
- Graph U-Nets / Graph Coarsening：图粗化、pool/unpool 有成熟基础。
- P-GUTS：pooled representation 对缺失时空数据有帮助。

可验证实验：

```text
fine-only
region-only
fine + region
random region negative control
```

### H4：区域图必须被缺失模式和分组质量控制

支撑：

- HD-TTS：多尺度组合依赖 observation/missing patterns。
- ChannelTokenFormer：真实场景下 channel dependency、asynchrony、missingness 需统一处理。
- 本地实验：coarse graph 在不同交通数据集上表现不一致。

可验证实验：

```text
fixed fusion
branch-summary gate
mask-statistics gate
mask + reliability gate
```

## 7. 建议继续深入阅读的优先级

第一优先级，直接影响论文定位：

1. CRIB：https://arxiv.org/abs/2509.23494
2. HD-TTS：https://arxiv.org/abs/2402.10634
3. CoIFNet：https://arxiv.org/abs/2506.13064
4. BiTGraph：https://github.com/chenxiaodanhit/BiTGraph
5. STFGNN：https://arxiv.org/abs/2012.09641
6. ESG：https://arxiv.org/abs/2206.13816

第二优先级，支撑不规则/缺失图建模：

1. HyperIMTS：https://arxiv.org/abs/2505.17431
2. GraFITi：https://arxiv.org/abs/2305.12932
3. T-PatchGNN：https://proceedings.mlr.press/v235/zhang24bw.html
4. ChannelTokenFormer：https://arxiv.org/abs/2506.08660
5. S4M：https://arxiv.org/abs/2503.00900

第三优先级，支撑区域/粗图方法合理性：

1. Graph U-Nets：https://arxiv.org/abs/1905.05178
2. Graph Coarsening with Neural Networks：https://arxiv.org/abs/2102.01350
3. Hierarchical Joint Graph Learning：https://arxiv.org/abs/2311.12630
4. P-GUTS：https://kdd.org/kdd2025/wp-content/uploads/2025/07/CameraReady-18.pdf
5. ImputeFormer：https://arxiv.org/abs/2312.01728

需要跟进但暂不作为强证据：

- Incomplete multivariate time series forecasting with dynamic attention graph network：https://www.sciencedirect.com/science/article/abs/pii/S092523122602093X  
  该文是 2026 年非常接近的 incomplete MTSF + dynamic graph attention 工作，需要拿到全文后确认是否涉及区域粗图或物理-时序双图。如果没有，它会成为重要相邻工作；如果有部分重合，需要进一步调整 novelty 表述。

## 8. 下一步建议

为了让 idea 从“可讲”变成“可写论文”，下一步建议做三件事：

1. **全文阅读 6 篇第一优先级论文**  
   输出每篇的 paper reading note：问题、方法、实验、局限、与我们差异。

2. **补一个 novelty table**  
   将 CRIB、CoIFNet、MissTSM、BiTGraph、HD-TTS、HyperIMTS、STFGNN、ESG 与我们的方法逐项对比，作为论文 introduction 和 related work 的基础表。

3. **先设计最小可验证模型，而不是一次性做复杂系统**  
   第一版只保留：

```text
observed-only temporal graph
physical graph optional
regional graph branch
mask/reliability residual gate
direct forecasting loss
```

这样能最直接验证论文的核心假设：

> 在高缺失率下，区域化变量关系是否能作为缺失可靠的多尺度补充，提升多变量时间序列预测？


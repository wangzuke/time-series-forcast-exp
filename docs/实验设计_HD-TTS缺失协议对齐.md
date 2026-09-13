# 实验设计：对齐 HD-TTS 缺失协议

> 日期：2026-09-05
> 目的：重新定义 MR-MSGF 的实验方案，使缺失模式、base 选择和代码改造方式与 HD-TTS 原文及官方代码保持一致。
> 核心原则：主实验不再使用自定义的简单 `time block × random node subset` 矩形缺失，而是采用 HD-TTS 的 `Point / Block-T` 缺失生成机制；Block-ST 只作为 HD-TTS 协议背景说明，不纳入本轮实验。

## 1. 总体实验目标

本轮实验要验证的问题是：

> 以 HD-TTS 官方实现为 base，在其时空层级预测框架上加入缺失可靠的细粒度图、区域粗图和 fine-to-coarse 融合后，是否能在目标节点时间轴缺失、但空间邻域仍可提供补偿信息的场景下稳定提升预测效果。

也就是说，实验不是比较一个从零搭建的新模型和 HD-TTS，而是比较：

```text
HD-TTS official
vs
HD-TTS official + MR-MSGF 增量模块
```

这样可以把结论收束到本文真正关心的改动：关系可靠性、观测驱动图、区域粗图和缺失可靠融合。

## 2. Base 选择

本轮 base 选择 **HD-TTS 官方代码**。

理由：

1. HD-TTS 本身就是 graph-based forecasting with missing data。
2. 它已经包含 temporal downsampling 和 spatial graph hierarchy。
3. 它的实验协议明确包含 Point 和 Block-T，能够支撑本文关注的散点缺失与时间块缺失场景。
4. 本文 idea 正好是在其基础上补充 observed graph、edge reliability、regional coarse graph 和 missingness-aware fusion。

工程上必须保留一个干净的 official baseline：

```text
external_repro/official/hdtts/     # 只用于官方 baseline 和核查
missing_ts_exp/                    # 放我们的实验适配、mask manifest、结果聚合
```

原则上不直接破坏官方仓库。若必须修改 HD-TTS 代码，应复制为单独实验分支或新增模块文件，并记录官方 commit hash、修改文件和配置差异。

## 3. HD-TTS 缺失模式的准确定义

HD-TTS 原文和官方代码中的缺失模式包括 Point、Block-T 和 Block-ST。本轮主实验只采用 Point 和 Block-T，因为二者更直接对应本文希望解决的“时间轴缺失 + 空间信息可补偿”问题。

### 3.1 Point

每个观测点独立随机缺失。

论文参数：

```text
eta: Bernoulli missing probability
```

代码对应：

```text
p_noise
```

典型配置：

```yaml
name: point
p_fault: 0.
p_noise: 0.25
```

### 3.2 Block-T

Block-T 是 temporal block missing，但在 HD-TTS 中它不是“所有节点同一时间段都缺失”，而是：

```text
某个传感器节点以概率 p_fault 发生 fault；
fault 持续 s 个连续时间步；
s 从 [s_min, s_max] 中采样；
同时叠加随机点缺失 p_noise。
```

典型真实世界配置：

```yaml
name: block_t
p_fault: 0.01
p_noise: 0.05
```

其中 `min_seq` 和 `max_seq` 由数据集配置给出。例如 METR-LA / PEMS-BAY 中为：

```yaml
min_seq: 12   # 1 hour
max_seq: 48   # 4 hours
```

### 3.3 Block-ST

Block-ST 是 spatiotemporal block missing，但在 HD-TTS 中它不是简单的矩形块：

```text
先按 Block-T 生成某个节点的连续时间 fault；
再把这个 fault 沿图传播到邻居节点；
传播概率由 p_propagation 控制；
传播范围由 propagation_hops 控制。
```

典型配置：

```yaml
name: block_st
p_fault: 0.003
p_noise: 0.075
p_propagation: [ .33, .15, .05 ]
propagation_hops: 3
```

含义是：fault 可以传播到 1-hop、2-hop、3-hop 邻居，传播概率分别为 0.33、0.15、0.05。最终形成的是由图结构约束的局部时空缺失区域。

本轮不采用 Block-ST 作为主实验。原因是 Block-ST 会主动破坏目标节点周围的空间观测，而本文方法的主要假设是利用仍然可用的空间邻域、相关节点和区域粗图来补偿节点自身的时间缺失。Block-ST 更适合作为空间信息受损时的压力测试，但不是本文当前最适配的验证场景。

## 4. 数据集设计

主实验优先使用带空间图的数据集。

| 数据集  | 用途                              | 缺失协议          |
| ------- | --------------------------------- | ----------------- |
| METR-LA | 主要验证数据集；交通传感器图明确  | Point / Block-T   |
| AQI     | 对齐 HD-TTS 原文真实缺失设置      | Original / +Point |
| EngRAD  | 可选扩展；HD-TTS 原文真实世界数据 | Block-T           |
| PV-US   | 可选扩展；HD-TTS 原文大规模数据   | Block-T           |

重要说明：

1. HD-TTS 原文中 AQI 主结果是 `Original` 和 `+Point`，因为 AQI 本身已有 25.67% 自然缺失，且经常成块出现。
2. 如果我们在 AQI 上额外生成 Block-T，必须标注为本文扩展实验，不能说这是 HD-TTS 原文 AQI 协议。
3. METR-LA 虽然不是 HD-TTS 原文主实验数据集，但官方代码中有 `la.yaml`，且附录有交通 benchmark 的 block setting，因此适合用于本地第一轮验证。
4. 本轮不做 Block-ST，因为它对应空间邻域同时缺失的压力场景，与本文当前要强调的空间补偿机制不完全一致。

## 5. 缺失强度组合

HD-TTS 原文使用参数化缺失分布，并不统一指定目标缺失率。因此本轮实验分成两层。

### 5.1 Paper-faithful 复现层

目的：确认官方协议和本地数据处理没有问题。

| 数据集  | 缺失模式 | 参数                                 |
| ------- | -------- | ------------------------------------ |
| AQI     | Original | 自然缺失                             |
| AQI     | +Point   | `p_noise = 0.25`                   |
| METR-LA | Block-T  | 官方`block_t` 机制，记录实际缺失率 |

### 5.2 Controlled-severity 主实验层

目的：在相同实际缺失率下比较模型。

建议目标实际缺失率：

```text
30%, 50%, 70%
```

对应缺失模式：

```text
Point
Block-T
```

实现方式：

1. 保持 HD-TTS 的 mask 生成机制不变。
2. 保持 Block-T 的连续 fault 定义不变。
3. 通过校准 `p_noise` 或 `p_fault` 让实际缺失率接近目标值。
4. 每个 `(dataset, mode, target_rate, seed)` 保存一份固定 mask。
5. 所有模型读取同一份 mask，而不是各自随机生成。

不建议第一轮加入 90%。90% 会使空间补偿信号也显著变弱，可作为后续极端缺失分析，而不是当前主表。

## 6. Baseline 设计

主 baseline：

| 方法                             | 位置              | 用途                                             |
| -------------------------------- | ----------------- | ------------------------------------------------ |
| HD-TTS-AMP official              | 主 baseline       | 直接 base，最关键对比                            |
| HD-TTS-IMP official              | HD-TTS 内部变体   | 判断 anisotropic / implicit message passing 差异 |
| TTS-AMP official                 | 结构消融 baseline | 判断 hierarchical downsampling 的作用            |
| BiTGraph official / mask-adapted | 补充 baseline     | Point missing 强相关；Block-T 需外部 mask 对齐   |

GinAR 不放主表。原因是 GinAR 主要面向 variable missing，与 HD-TTS 的 Point / Block-T 协议不同。它更适合作为 related work 中的关键支撑工作。

## 7. MR-MSGF 改造方式

改造路径：

```text
HD-TTS official backbone
  -> 保留 temporal downsampling
  -> 保留 spatial hierarchy / graph pooling
  -> 新增 observed graph A_obs
  -> 新增 edge observed overlap q_ij
  -> 构造 A_fine = gate(A_phy, A_obs)
  -> 构造 regional/coarse graph A_coarse
  -> 新增 missingness-aware fine-to-coarse residual gate
  -> 保持 forecasting decoder 和 loss 对齐
```

代码原则：

1. 官方 HD-TTS baseline 必须可独立运行。
2. MR-MSGF 只新增模块，不改动官方 baseline 的默认行为。
3. mask 生成使用 HD-TTS 官方 `sample_mask` 逻辑或严格等价实现。
4. 每个实验保存 `mask.npy`、`manifest.csv`、实际缺失率和 mask hash。
5. 所有模型统一窗口长度、预测长度、数据划分、归一化方式和评价指标。

建议新增模块：

| 模块                       | 功能                                   |
| -------------------------- | -------------------------------------- |
| `mask_manifest_builder`  | 生成/校准 HD-TTS-style mask，保存 hash |
| `observed_graph_builder` | 基于共同可观测片段构造$A^{obs}$      |
| `edge_reliability`       | 计算$q_{ij}$ 和节点观测率            |
| `regional_pooling`       | 构造节点到区域分配$S$                |
| `mrmsgf_fusion`          | 实现 fine-to-coarse reliability gate   |

## 8. 消融矩阵

第一轮消融只在 METR-LA 上做，优先 Block-T 50% / 70%。

| 变体                       | 说明                                         |
| -------------------------- | -------------------------------------------- |
| M0: HD-TTS official        | 不加本文模块                                 |
| M1: +$A^{phy}$ only      | 只保留物理图                                 |
| M2: +$A^{obs}$ only      | 只使用观测驱动图                             |
| M3: + reliable$A^{fine}$ | 物理图 + 观测图 + edge reliability           |
| M4: M3 + coarse graph      | 加粗图，但无 reliability gate                |
| M5: full MR-MSGF           | 粗图 + missingness-aware fine-to-coarse gate |

这套消融回答五个问题：

1. 物理图是否有用？
2. 观测驱动图是否有用？
3. edge reliability 是否能抑制噪声边？
4. coarse graph 是否真的提供高缺失补充？
5. reliability gate 是否避免无条件粗图融合带来的过平滑？

## 9. 第一轮执行矩阵

推荐第一轮不要做太大。

| 阶段      | 数据集  | 模式             | 缺失强度         | 模型                |
| --------- | ------- | ---------------- | ---------------- | ------------------- |
| Smoke     | METR-LA | Block-T          | 50%              | HD-TTS-AMP, MR-MSGF |
| Base 复现 | AQI     | Original, +Point | 原始 / 25% point | HD-TTS-AMP          |
| 主结果 A  | METR-LA | Point, Block-T   | 30%, 50%, 70%    | HD-TTS-AMP, MR-MSGF |
| 主结果 B  | AQI     | Original, +Point | 原始 / 25% point | HD-TTS-AMP, MR-MSGF |
| 消融      | METR-LA | Block-T          | 50%, 70%         | M0-M5               |

如果第一轮结果稳定，再补：

1. HD-TTS-IMP / TTS-AMP。
2. BiTGraph 在 Point 以及可加载外部 mask 的 Block-T。
3. EngRAD 或 PV-US。
4. 3 seeds 完整结果。

## 10. 验收标准

本轮实验的最低验收标准：

1. HD-TTS official 在本地能够按官方协议跑通。
2. MR-MSGF 和 HD-TTS 使用相同 mask、相同数据划分、相同窗口和预测长度。
3. `manifest.csv` 记录每个 mask 的目标缺失率、实际缺失率、参数、seed 和 hash。
4. 在 METR-LA Block-T 50% / 70% 上，full MR-MSGF 相对 HD-TTS-AMP 有稳定收益，或至少通过消融解释为什么没有收益。
5. 若 coarse graph 失败，必须报告是区域划分、过平滑、区域覆盖率不足，还是与 HD-TTS 原空间层级冲突。

## 11. 论文中应如何表述

论文里建议这样写：

> Following HD-TTS, we evaluate forecasting performance under Point and Block-T missing patterns. Point missing randomly removes individual observations, while Block-T simulates temporal sensor faults by masking consecutive observations at a node. We focus on these settings because they preserve neighboring spatial observations, allowing us to evaluate whether spatial and regional graph information can compensate for temporal missingness at target nodes.

对于 AQI：

> For AQI, we follow HD-TTS and evaluate on the original data with natural missing values and an additional Point missing setting. Block-T on AQI, if included, is treated as an extended experiment rather than the original HD-TTS AQI protocol.

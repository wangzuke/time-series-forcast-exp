# MissTSM、CoIFNet、CRIB 多变量缺失时间序列处理报告

## 1. 阅读范围与关注问题

本文整理 `papers/missing-value-forecasting` 目录下三篇论文对多变量时间序列缺失值场景的处理方式：

- `02_MissTSM_Investigating_Model_Agnostic_Imputation_Free_IMTS_Modeling.pdf`
- `05_CoIFNet_Unified_Framework_for_MTS_Forecasting_with_Missing_Values.pdf`
- `01_CRIB_Revisiting_Multivariate_Time_Series_Forecasting_with_Missing_Values.pdf`

重点关注的问题是：在多变量时间序列存在任意缺失时，模型如何表示变量、如何利用变量间关系、是否显式建模 cross-variate correlation，以及缺失模式如何参与多变量建模。

三篇论文都不同程度地反对简单的“先补全再预测”范式，但它们对多变量结构的处理思路明显不同：

- MissTSM 重点解决任意缺失下的 token 化和缺失鲁棒嵌入问题。
- CoIFNet 通过 mask-aware 的时间轴融合和变量轴融合实现一阶段预测。
- CRIB 直接面向缺失导致的变量相关性破坏问题，用统一变量注意力建模全局变量关系。

## 2. 总体对比

| 方法 | 多变量建模核心 | 缺失值处理方式 | 变量间关系建模强度 | 主要取舍 |
| --- | --- | --- | --- | --- |
| MissTSM | 每个时间-变量标量独立成 token，再在同一时间步内聚合已观测变量 | 不显式补值，用 mask attention 忽略缺失变量 | 中等。同一时间步内跨变量聚合，但不是完整变量两两交互 | 插件式、模型无关、轻量，但跨变量关系被压缩成每时刻摘要 |
| CoIFNet | 先 Cross-Timestep Fusion，再 Cross-Variate Fusion | observed value、mask、timestamp 联合输入，同时做历史重构和未来预测 | 中等偏强。显式有跨变量融合模块 | 效率好，结构清晰，但变量关系建模偏轻量 |
| CRIB | 将所有变量的 patch token 展平，做 Unified-Variate Attention | 不显式补值，用信息瓶颈和一致性正则过滤缺失噪声 | 最强。统一建模 intra-variate 和 inter-variate 全局关系 | 变量相关性表达最充分，但 attention 成本更高，需要 patch 降复杂度 |

如果只从多变量关系表达能力看，三者大致排序为：

1. CRIB：最强调全局变量相关性。
2. CoIFNet：显式区分时间融合与变量融合，准确率与效率较均衡。
3. MissTSM：最灵活，适合作为缺失鲁棒 adapter，但变量关系建模较压缩。

## 3. MissTSM：时间-变量独立嵌入与同时间步变量聚合

### 3.1 核心动机

MissTSM 认为传统 Transformer 类时间序列模型在缺失场景下面临一个基础问题：token 的构造依赖完整观测。

传统 Transformer 往往将一个时间步的所有变量 `X_t` 作为一个 token。如果该时间步中任意变量缺失，整个 token 就不完整。iTransformer 类方法则将一个变量的整条历史序列作为 token，如果该变量在任意时间点缺失，也会破坏 token 的完整性。

因此，MissTSM 的出发点不是先设计复杂的变量关系模块，而是先解决“任意时间、任意变量缺失时，如何仍然构造有效输入表示”。

### 3.2 Time-Feature Independent Embedding

MissTSM 使用 Time-Feature Independent Embedding，将每个 `(t, d)` 位置上的单个标量独立映射为一个 embedding：

```text
h_(t,d) = TFIEmbedding(X_(t,d))
```

这样，某个变量在某个时间点缺失，只会影响该位置的 token，不会污染同一时间步的其他变量，也不会污染同一变量的其他时间点。

为了让这些独立 token 仍然保留时间和变量身份，MissTSM 使用二维位置编码，同时编码：

- 时间位置 `t`
- 变量位置 `d`

这种设计使得模型能够在不补值的情况下保留多变量结构。

### 3.3 Missing Feature-Aware Attention

MissTSM 的多变量交互主要发生在 Missing Feature-Aware Attention，也就是 MFAA。

MFAA 在每个时间步内执行 masked cross-attention：

- query 是一个可学习向量，且不依赖具体时间和变量。
- key/value 是该时间步所有变量的 position-aware embedding。
- mask 会将缺失变量对应的 attention score 置零。
- 输出是该时间步的聚合表示 `L_t`。

也就是说，MissTSM 的处理流程是：

```text
每个时间-变量独立嵌入
        ↓
同一时间步内对已观测变量做 masked cross-attention
        ↓
得到每个时间步的 latent summary
        ↓
交给后续 backbone 学习长期时间动态
```

论文将其称为 variate-first：先在同一时间步内处理变量，再由 backbone 处理时间依赖。

### 3.4 多变量处理特点

MissTSM 的优势是缺失鲁棒和模型无关。它可以作为 adapter 插入 MAE、Transformer 或其他 MTS backbone，使原本不能处理任意缺失的模型获得 imputation-free 能力。

但在变量关系建模上，它采取的是压缩式聚合。MFAA 使用的是一个 query 对 `N` 个变量求 `N x 1` attention 权重，而不是完整的 `N x N` 变量 self-attention。因此它可以利用跨变量信息，但不会显式建模所有变量对之间的关系。

论文也承认 full self-attention 可能更好地捕获 cross-variate interactions，但计算更贵。MissTSM 的选择是用较低计算成本获得足够鲁棒的变量聚合。

### 3.5 局限

MissTSM 的 MFAA 本身不负责长期时间建模，也不同时建模非线性时间动态和变量相关性。长期动态交给后续 backbone 完成。

此外，每个时间-变量标量独立成 token，在高维多变量系统中会带来额外计算成本。对于变量数很大的场景，TFI embedding 的 token 数会随 `T x N` 增长。

## 4. CoIFNet：跨时间融合与跨变量融合的一阶段框架

### 4.1 核心动机

CoIFNet 关注传统两阶段方法的信息断裂问题。两阶段方法通常先用 imputer 得到补全序列，再把补全结果交给 forecasting model。论文认为这个过程存在两个问题：

- imputation 优化目标是重构历史缺失值，不一定保留对未来预测最有用的信息。
- mask matrix 在补值后常被丢弃，但缺失模式本身可能包含预测信号。

因此，CoIFNet 将 imputation 和 forecasting 放到一个统一网络中联合优化。

### 4.2 输入表示

CoIFNet 的输入由三部分拼接：

- 归一化后的 observed values
- mask matrix
- timestamp embeddings

输入形式可概括为：

```text
Z_in = [X_bar || M_x || E_tau]
```

其中 mask 不只是用于 loss 或过滤，而是作为显式特征进入模型。这一点对多变量缺失场景很重要，因为不同变量的缺失模式可能携带结构性信息。

### 4.3 RevON：只基于观测值归一化

CoIFNet 提出 Reversible Observed-value Normalization，即 RevON。普通 RevIN 依赖完整序列统计量，而缺失值会污染均值和方差估计。

RevON 只基于 observed values 计算归一化统计量，并将 mask 纳入仿射变换。这使模型在非平稳多变量序列中能更稳定地处理缺失。

### 4.4 Cross-Timestep Fusion

Cross-Timestep Fusion，简称 CTF，沿时间步维度融合信息。它的目标是利用同一变量不同时间点的上下文，弥补缺失造成的信息扭曲。

CTF 使用 sigmoid-gated mechanism，自适应控制每个时间特征保留多少信息。直观上，它让模型更关注观测可靠的时间模式，减弱缺失或噪声特征影响。

### 4.5 Cross-Variate Fusion

Cross-Variate Fusion，简称 CVF，是 CoIFNet 的主要多变量模块。它沿变量维度对 CTF 输出后的表示进行融合，显式建模 cross-variate interactions。

CVF 同样使用 gated mechanism：

- 一支产生 gate，控制变量维度信息通过量。
- 一支产生候选表示。
- 两者逐元素相乘后再投影。

这种结构比全局 self-attention 更轻量，适合计算效率要求较高的 forecasting 场景。

### 4.6 联合重构与预测

CoIFNet 同时输出：

- 历史窗口重构 `X_hat`
- 未来窗口预测 `Y_hat`

损失函数为：

```text
L = (1 - lambda) * L_I + lambda * L_F
```

其中 `L_I` 是 imputation/reconstruction loss，`L_F` 是 forecasting loss。

与传统两阶段方法不同，CoIFNet 的重构任务不是独立训练的前处理器，而是辅助预测任务共同塑造共享表示。这样可以让 observed values、mask 和预测目标在同一表示空间中共同作用。

### 4.7 多变量处理特点

CoIFNet 的多变量处理是“轴向分解式”的：

```text
observed values + mask + timestamp
        ↓
RevON
        ↓
Cross-Timestep Fusion
        ↓
Cross-Variate Fusion
        ↓
同时输出历史重构和未来预测
```

相比 MissTSM，CoIFNet 更显式地设计了变量维度模块。相比 CRIB，CoIFNet 不使用全局 all-to-all attention，而是采用轻量 gated fusion，因此更强调效率和部署友好性。

### 4.8 局限

CoIFNet 的 CVF 是显式变量融合，但表达能力仍然偏轻量。它不构建变量图，也不做所有变量 patch 的全局 self-attention。因此在变量关系非常复杂、长程跨变量依赖很强的场景中，它可能不如 CRIB 充分。

此外，CoIFNet 仍保留了辅助重构任务。虽然它是一阶段联合优化，但如果重构目标与预测目标冲突，`lambda` 的选择会影响最终效果。

## 5. CRIB：统一变量注意力与缺失噪声过滤

### 5.1 核心动机

CRIB 对 imputation-then-prediction 范式的批评最强。论文认为，在真实缺失值没有 ground truth 的情况下，imputation model 很难可靠恢复缺失值。错误补值不仅不能修复数据，反而会：

- 改变原始数据分布。
- 破坏变量间相关性。
- 将错误信号传播给预测模型。

因此，CRIB 选择绕过显式补值，直接从 partially observed input 预测未来。

### 5.2 Patching Embedding

CRIB 首先将每个变量的历史序列切成 patch。设变量数为 `N`，历史长度为 `T`，patch 长度为 `P`，则输入变为：

```text
N x (T / P) x P
```

随后加入 temporal encoding，并通过 TCN 将稀疏 patch 转换为 dense feature representation：

```text
H in R^(N x T/P x D)
```

patching 的作用有两个：

- 增强局部时间语义。
- 降低后续 attention 的 token 数和计算成本。

### 5.3 Unified-Variate Attention

CRIB 最关键的多变量模块是 Unified-Variate Attention。

它将所有变量、所有 patch 展平成一个 token 序列：

```text
H_hat in R^((N * T/P) x D)
```

然后直接做标准 self-attention：

```text
Z = Attention(Q, K, V)
```

这意味着 attention 可以同时学习：

- 同一变量不同时间 patch 之间的 intra-variate temporal correlation
- 不同变量之间的 inter-variate correlation
- 不同变量、不同时间 patch 之间的非局部相关性

CRIB 不人为拆分时间关系和变量关系，也不预设变量图结构，而是把所有 patch token 放入同一个注意力空间中学习全局关系。

### 5.4 Information Bottleneck Guidance

CRIB 使用 Information Bottleneck 指导表示学习。其目标是在 partially observed input `X_o` 和预测目标 `Y` 之间学习一个表示 `Z`，使其：

- 尽量压缩输入中的无关信息和缺失噪声。
- 尽量保留预测未来所需的信息。

论文将 compactness 和 informativeness 写成两个目标：

- compactness：约束 `Z` 不要过度携带由缺失位置引入的噪声。
- informativeness：确保 `Z` 对预测任务仍然有用。

这对多变量场景尤其关键，因为高缺失率下模型可能过度依赖少量观测变量，忽略真正稳定的变量关系。IB 约束鼓励模型保留 task-relevant variate correlation，而不是记住随机缺失模式。

### 5.5 Consistency Regularization

CRIB 还使用一致性正则。它对原始观测输入构造一个更困难的增强视图，例如：

- 对额外 10% 的观测点随机 mask。
- 对观测点加入 Gaussian noise。

然后要求原始输入和增强输入经过同一网络后得到的表示保持一致。

这样做的核心直觉是：模型预测应该对缺失模式扰动保持稳定。换句话说，模型不应过度依赖某几个偶然观测到的点，而应学习更稳定的全局变量关系。

### 5.6 多变量处理特点

CRIB 的多变量处理最直接、最强：

```text
partially observed input
        ↓
patching + temporal encoding + TCN
        ↓
flatten all variate-patch tokens
        ↓
Unified-Variate Attention
        ↓
IB guidance + consistency regularization
        ↓
direct forecasting
```

相比 MissTSM，CRIB 不把同一时间步变量压缩成一个 summary，而是保留变量 patch token 之间的全局交互。相比 CoIFNet，CRIB 不把时间融合和变量融合拆成两个轻量模块，而是在一个统一 attention 空间中同时学习二者。

### 5.7 局限

CRIB 的主要问题是计算复杂度。虽然 patching 将时间 token 数从 `T` 降到 `T/P`，但 attention token 数仍然是 `N * T/P`。当变量数 `N` 很大时，全局 attention 成本依然明显。

此外，CRIB 依赖 IB 和一致性正则来稳定训练，目标函数比 MissTSM 和 CoIFNet 更复杂。实际使用时需要调节多个损失权重。

## 6. 三篇论文的多变量处理差异

### 6.1 变量作为 token 的方式不同

MissTSM 的 token 粒度最细：单个时间-变量标量是 token。它先保证任意缺失下 token 可用，再聚合同一时间步变量。

CoIFNet 不把变量单独展开成 attention token，而是把多变量矩阵与 mask、timestamp 拼接后，通过 CTF/CVF 进行轴向融合。

CRIB 的 token 粒度是变量 patch。它既保留变量身份，又通过 patch 获得局部时间语义，再做全局 attention。

### 6.2 对变量相关性的建模强度不同

MissTSM 使用同一时间步内的 masked cross-attention，能够利用已观测变量生成时间步表示，但不是变量对级别建模。

CoIFNet 使用 CVF 显式建模变量维度交互，强度高于简单线性层，但仍属于轻量融合。

CRIB 使用所有变量 patch 的全局 self-attention，能够学习最完整的 inter-variate 和 intra-variate 关系。

### 6.3 mask 的作用不同

MissTSM 中，mask 主要用于 attention 层屏蔽缺失变量，使缺失 token 不参与聚合。

CoIFNet 中，mask 是输入特征的一部分，并贯穿 CTF、CVF、重构和预测过程。论文强调 mask 本身包含预测信息。

CRIB 中，mask 隐含在 partially observed input 和增强过程中，重点不是直接使用 mask 做特征，而是通过 IB 和一致性正则降低缺失模式噪声。

### 6.4 是否依赖 imputation

MissTSM 是 imputation-free。它不补值，而是只对 observed token 建模。

CoIFNet 是 unified imputation-forecasting。它会输出历史重构，但不是两阶段补值，而是联合优化。

CRIB 是 direct prediction。它明确反对显式 imputation，认为补值会破坏变量相关性。

## 7. 对后续模型设计的启发

### 7.1 缺失场景下不要只关注时间依赖

多变量缺失预测中，变量相关性本身可能被缺失严重破坏。CRIB 的实验和分析强调：错误 imputation 会改变 variate correlation map。因此，模型设计应显式考虑如何保护或恢复变量间结构。

### 7.2 mask 不应只作为 loss 过滤器

CoIFNet 说明 mask matrix 可以作为输入信号参与预测。缺失模式可能反映传感器失效、采样机制、业务行为或变量可靠性，因此在多变量建模中应考虑 mask-aware representation。

### 7.3 token 粒度决定缺失鲁棒性和变量建模能力

不同 token 粒度带来不同取舍：

- 标量级 token：最鲁棒，但 token 数多，变量关系需要后续聚合。
- 变量级 token：适合完整序列，但对时间缺失敏感。
- 变量 patch token：兼顾局部时间语义和变量身份，但 attention 成本较高。

### 7.4 全局变量注意力适合高缺失率，但要控制复杂度

CRIB 的 Unified-Variate Attention 在高缺失率下有优势，因为它允许模型从任意可用变量和任意时间 patch 中寻找有用信息。但当变量数较大时，需要 patching、稀疏 attention、分组 attention 或低秩近似来控制成本。

### 7.5 轻量轴向融合适合工程部署

CoIFNet 的 CTF + CVF 是一个较适合实际系统的折中方案。它明确区分时间维和变量维，计算效率优于全局 attention，也比完全线性方法更能利用多变量关系。

## 8. 结论

三篇论文可以概括为三种不同路线：

- MissTSM：缺失鲁棒 adapter 路线。通过时间-变量独立嵌入和 masked cross-attention，使任意缺失下仍能构造有效时间步表示。
- CoIFNet：一阶段联合建模路线。将 observed value、mask、timestamp 共同输入，通过 CTF 和 CVF 分别融合时间与变量信息，并联合优化重构和预测。
- CRIB：变量相关性保护路线。绕过显式补值，用 Unified-Variate Attention、Information Bottleneck 和一致性正则直接从稀疏观测中学习稳定的全局变量关系。

从多变量处理角度看，CRIB 对变量相关性的建模最充分，CoIFNet 在性能和效率之间最均衡，MissTSM 的通用性和缺失鲁棒性最好。后续如果设计新的多变量缺失预测模型，可以考虑结合三者优势：采用 MissTSM 式细粒度缺失鲁棒 tokenization，引入 CoIFNet 式 mask-aware 轴向融合，再用 CRIB 式全局变量关系建模或一致性约束增强高缺失率下的稳定性。

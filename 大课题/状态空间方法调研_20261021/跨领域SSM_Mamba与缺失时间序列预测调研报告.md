# 状态空间模型 / Mamba 在跨领域与缺失时间序列预测中的应用：现状、优势、问题与可借鉴技术路线

**检索日期：** 2026-10-21  
**服务主题：** 缺失数据条件下的时间序列与时空序列预测  
**范围：** 以深度状态空间模型（S4、S5、Mamba 等）为主；同时纳入与“缺失历史条件下预测未来”直接相关的图时空、插补和不规则采样工作。预印本、期刊/会议论文和出版社早期记录在文中明确区分。

## 一、结论先行

状态空间模型与 Mamba 的核心价值不是自动“补好缺失值”，而是以线性复杂度递推隐状态，并可让状态更新依赖当前输入、观测掩码或采样间隔。对缺失预测而言，这带来三类已被不同论文分别支持的能力：

1. **避免将填充值当作真实观测。** S4M 将观测 mask 作为 S4 的额外输入；MAS-Mamba 进一步用 mask 和距上次有效读数调节选择性 SSM 的离散步长。
2. **在长窗口和长缺口中维持可计算的时间状态。** S4/S5/Mamba 的时间复杂度随序列长度近似线性增长；这为长历史窗、多站点和高频数据提供了计算条件，但不等于一定比 Transformer 更准。
3. **与空间或多尺度模块互补。** TIMBA、STG-Mamba、Graph Mamba 及视觉/医学影像中的 SSM 表明，SSM 通常承担长程传播，局部卷积、图消息或多尺度模块承担局部结构。

但现有证据也表明，缺失时间序列预测中“SSM/Mamba 的优势”尚未成为统一结论。直接缺失预测工作数量有限，许多 Mamba 论文只在完整观测的长时预测中比较；插补、历史表示和因果未来预测常被混用；真实缺失机制、训练—测试缺失机制变化、同步站点故障和可靠不确定性仍缺少系统验证。

因此，合理研究路线不是“将现有编码器替换为 Mamba”，而应先选定一个可检验的缺失问题，再决定状态空间机制是否解决该问题。

## 二、基本概念与任务边界

离散状态空间层可表为：

$$
h_t=ar A h_{t-1}+ar Bx_t,qquad y_t=Ch_t+Dx_t .
$$

S4 使用结构化状态矩阵，使长序列计算有效；S5 将多组 SISO 状态层简化为 MIMO SSM；Mamba 使离散步长及输入/输出投影依赖当前输入，并用选择性 scan 实现高效并行计算。[S4](https://arxiv.org/abs/2111.00396) [S5](https://arxiv.org/abs/2208.04933) [Mamba](https://arxiv.org/abs/2312.00752)

本报告严格区分三件事：

| 任务 | 可访问的信息 | 是否可直接作为缺失预测证据 |
|---|---|---|
| 因果未来预测 | 预测起点以前的历史；历史中可有缺失 | 是，核心任务 |
| 历史插补 | 缺口前后都可使用真实观测 | 否，只能借鉴缺口表征与空间恢复机制 |
| 完整观测预测 | 历史输入完整 | 否，只能说明时间骨干能力和复杂度 |

例如，双向 Mamba 在历史窗内部编码可用于预测；若其读到预测起点之后的真实观测，则只能用于插补，不能作为未来预测器比较。

## 三、跨领域应用：真正可借鉴的设计经验

### 3.1 视觉：处理二维结构时，扫描路径不是实现细节

| 工作 | 已解决的问题 | 核心设计 | 对缺失时空序列的可借鉴点 | 不应直接照搬之处 |
|---|---|---|---|---|
| Vision Mamba，ICML 2024，[论文](https://arxiv.org/abs/2401.09417) | 高分辨率视觉任务的全局上下文与显存/计算限制 | 将图像 token 加位置编码后，以**双向**状态空间块压缩全局表征；在分类、检测、分割中报告效率与性能。 | 需要全历史表征时，可研究历史窗内前/后方向的状态融合。 | 图像 token 的双向扫描不能读取预测起点后的未来观测。 |
| VMamba，NeurIPS 2024 Spotlight，[论文](https://arxiv.org/abs/2401.10166) | 二维数据没有天然一维扫描顺序 | 四条二维扫描路径的 SS2D，从不同方向汇集上下文。 | 交通/空气监测节点也没有由编号决定的自然一维顺序；若沿节点维扫描，应先设计图驱动的多路径/分组，而非按 ID 硬串行。 | 四向网格扫描假设规则二维邻接，不能直接替代道路拓扑或站点图。 |
| S4ND，NeurIPS 2022，[论文](https://arxiv.org/abs/2210.06583) | 图像/视频的多维长程依赖 | 将结构化 SSM 扩展到多维信号。 | 提示“时间维与空间维不必使用同一序列化方式”；可把时间状态与空间消息分开建模。 | 不是针对缺失或未来预测的工作。 |

**得到的原则：** 一维选择性扫描会对输入顺序敏感。对于带物理图的站点数据，状态空间层若扫描节点维，必须说明 token 化和排序依据，并测试节点置换的稳定性。

### 3.2 医学影像：局部证据与全局状态应分工

U-Mamba 在 CT、MR、内窥镜和显微图像分割中将 CNN 的局部特征提取与 SSM 的长程依赖结合，而不是用 SSM 完全替代局部算子。[论文](https://arxiv.org/abs/2401.04722)

这对缺失时空预测的可迁移含义是：

- 最近可观测值、局部趋势、物理邻居是局部证据；
- 跨越长缺口的历史状态、远距离相关站点、区域尺度模式是全局证据；
- 模型应根据观测条件选择二者，而不是假设单一 SSM 状态能够恢复全部信息。

它不支持“CNN+Mamba 必然更好”。其原始任务是分割，损失与预测时的因果性、缺失机制、预测步长均不同。

### 3.3 图学习：节点 token 化、顺序与局部编码是必须显式回答的问题

Graph Mamba 提出将 SSM 用于图时，需显式考虑邻域 token 化、token 排序、双向选择性 SSM 编码和局部编码，并将顺序问题列为图 SSM 的核心设计步骤。[论文](https://arxiv.org/abs/2402.08678)

对时空图预测，这比“图＋Mamba”这一口号更有价值：

1. **时间轴与节点轴职责分离。** 时间扫描适合因果历史状态；节点/邻域聚合应服从图结构。
2. **可观测邻域优先。** 节点缺失时，邻域 token 不应把无效值与真实证据同等处理；可使用 mask、共同观测程度和物理距离描述邻域证据。
3. **避免任意排序结论。** 若采用变量/节点扫描，必须做原始 ID、随机置换、图遍历排序的稳定性实验。

这与 S-Mamba 的提醒一致：S-Mamba 的主要 Mamba 扫描在变量维，而非时间维。[论文](https://doi.org/10.1016/j.neucom.2024.129178)

### 3.4 音频：长序列不能只学低频趋势，应保持多尺度细节

S4 音频生成工作说明，原始音频的采样率高、局部纹理密集，单一长程状态不足以承担全部时频细节。[It's Raw!](https://arxiv.org/abs/2202.09729) 一般时间预测中的 TimeMachine、MambaTS、MOWE-Mamba 也分别使用多分支、多尺度或窗口机制来补偿单尺度 Mamba 的局限。[TimeMachine](https://arxiv.org/abs/2403.09898) [MambaTS](https://doi.org/10.1016/j.patcog.2026.114536)

可迁移结论是：若数据具有 5 分钟交通波动、小时周期和日周期，状态空间分支应与局部/多尺度表示配合。不可迁移的结论是“只要上多尺度，就能处理缺失”；多尺度不能创造缺口内不存在的真实观测。

### 3.5 跨领域经验的汇总

| 跨领域机制 | 原始动机 | 适用于缺失预测的前提 | 当前选题可借鉴形式 |
|---|---|---|---|
| 多方向/双向扫描 | 视觉全局上下文、无自然顺序 | 只读取预测起点前历史 | 历史窗内部双向状态，或图引导多路径节点聚合 |
| 局部算子＋SSM | 医学影像保留局部边界并引入长程依赖 | 局部证据仍可观测 | 近期观测/邻域消息与长程时间状态分支 |
| 图 token 化与排序 | 图没有天然序列顺序 | token 顺序有物理或统计依据 | 物理邻域分组、排序敏感性审计 |
| 多尺度状态 | 音频/视觉同时有局部纹理和长程结构 | 各尺度均不以填充值作为真值 | 短/中/长缺口分层评测与多尺度时间表征 |
| 选择性状态更新 | 输入内容决定保留与遗忘信息 | 状态更新能看到 mask 或时间间隔 | mask/间隔进入输入或离散步长，而非只在前处理插补 |

## 四、缺失数据时间序列预测中的 SSM/Mamba 研究现状

### 4.1 直接缺失预测与紧密相关工作

| 工作 | 年份/状态 | 模型与缺失处理 | 任务和数据 | 证据强度与边界 |
|---|---|---|---|---|
| SSSD | TMLR 2022 | 以 S4 作为条件扩散去噪骨干。 | 插补为主；也报告 Solar、ETTm1 的尾段遮挡预测，插补包含 METR-LA/PEMS-BAY。 | S4＋缺失的重要公开基线；生成式插补不能与因果未来预测直接等价。[论文](https://arxiv.org/abs/2208.09399) [代码](https://github.com/AI4HealthUOL/SSSD) |
| CRU | ICML 2022 | 连续时间 Kalman 型状态演化；真实观测到达时更新并追踪不确定性。 | 不规则序列表示、插值为主。 | 真实时间间隔控制状态演化的坚实参考，非 S4/Mamba 直接预测器。[论文](https://proceedings.mlr.press/v162/schirmer22a.html) |
| S4M | ICLR 2025 | 掩码作为 S4 的额外状态/输出输入；ATPM 原型检索。 | 多变量未来预测；同步时间块和逐变量时间块；ETT、Electricity、Traffic、Weather、USHCN。 | 最直接的 S4 缺失预测论文。掩码输入和原型记忆已不是空白点。[论文](https://arxiv.org/abs/2503.00900) [代码](https://github.com/WINTERWEEL/S4M) |
| TIMBA | 2024 预印本 | 双向 Mamba 替代扩散插补框架中的时间 Transformer；空间端使用图消息和节点 Transformer。 | AQI36、METR-LA、PEMS-BAY 的散点、块、历史掩码插补。 | 对时空缺失表示很相关，但不是因果未来预测。[论文](https://arxiv.org/abs/2410.05916) |
| MAS-Mamba | IET ITS 2026 | mask 与距上次有效读数共同条件化选择性 SSM 离散步长，使断连时状态保持或衰减。 | METR-LA、PEMS-BAY；MCAR、NMAR、传感器断连的短期交通预测。 | 最接近“缺失道路预测＋Mamba”；仅提出间隔门控会直接重合。[DOI](https://doi.org/10.1049/itr2.70304) |
| Mamba-NTP | JNCA 2025 | Mamba、图学习和预测联合处理稀疏测量。 | 通信网络流量矩阵完成与预测。 | 说明端到端完成—预测可行；非道路交通图。[论文](https://doi.org/10.1016/j.jnca.2025.104364) |
| MAP-SSM | 出版社页面标注 2026-12-07 | Missing-Gated Discretization 与 prototype memory。 | 多变量长块缺失预测。 | 首次在线时间、完整代码尚待核验；仍是强选题重合风险。[页面](https://www.sciencedirect.com/science/article/pii/S0925231226023301) |
| EMIRA | 学位论文 2026 | Mamba 的不规则上下文选择性与 query prompting。 | 不规则多变量预测。 | 机制线索有价值，但目前可核验载体不是独立同行评议论文。[记录](https://repository.hkust.edu.hk/ir/bitstream/1783.1-171509/1/1783.1-171509.pdf) |

### 4.2 一般预测与缺失预测之间不能省略的桥梁

一般 Mamba 预测论文已大量探索时间/变量/多尺度设计：TimeMachine、S-Mamba、CMamba、MambaTS、FSMamba 等。它们有助于挑选基础时间骨干，却通常假定历史完整，不能证明对缺失值稳健。[CMamba](https://arxiv.org/abs/2406.05316) [FSMamba](https://doi.org/10.1016/j.knosys.2025.115233)

时空预测方面，STG-Mamba 表明选择性 SSM 可与时空图表示结合，但其主要基准并非面向历史缺失。[论文](https://arxiv.org/abs/2403.12418) HD-TTS 和 BiTGraph 则直接处理缺失时空图，但时间骨干分别是层次下采样和部分时序卷积，而非 Mamba。[HD-TTS](https://arxiv.org/abs/2402.10634) [BiTGraph](https://openreview.net/pdf?id=O9nZCwdGcG)

## 五、已展现的优势：证据、适用条件与限度

| 优势 | 支撑工作 | 在什么条件下成立 | 不能推出什么 |
|---|---|---|---|
| 长序列计算/显存可扩展 | S4、S5、Mamba；Vision Mamba/VMamba 的高分辨率结果 | 使用并行 scan、合适实现和较长输入时 | 不保证任何窗口长度下均更快，也不保证精度必优 |
| 观测状态可直接进入表示 | S4M、MAS-Mamba | mask 被显式输入状态/离散化，且训练覆盖相近缺失情景 | 普通 Mamba 对零填充不会天然识别缺失 |
| 长缺口的隐状态保持或受控衰减 | MAS-Mamba、MAP-SSM | 以间隔/mask 控制离散化，或有可靠潜在先验 | 长缺口内真实动态仍不可从无观测中确定 |
| 多变量/跨节点信息利用 | TIMBA、Mamba-NTP、Graph Mamba | 相关节点或变量在缺失时仍保留有效证据 | 图消息若来自同步缺失邻居，会传播低质量信息 |
| 与多尺度、局部模块兼容 | U-Mamba、TimeMachine、HD-TTS | 局部和长程分支职责清楚、消融可验证 | 复杂模块叠加不自动提升缺失鲁棒性 |
| 不规则采样建模潜力 | CRU、EMIRA | 使用真实时间间隔和不规则查询，不是把空白网格一律视为相同步长 | 固定网格 S4/Mamba 不能自动成为连续时间模型 |

## 六、主要问题与研究空白

### 6.1 方法层面

1. **缺失值与时间间隔并未普遍进入状态转移。** S4M 的 mask 是额外输入流；MAS-Mamba 与 MAP-SSM 才更直接地改变离散化/状态衰减。普通 Mamba 先填零再扫描，仍可能将占位符误当作内容。
2. **长缺口的状态语义不清。** “保持”“衰减”“借助原型”是不同的归纳偏置，适用情形取决于系统动力学；现有论文通常以平均误差比较，较少验证状态在不同缺口长度下是否合理。
3. **空间证据质量缺少统一处理。** 直接缺失 SSM 多面向非图多变量数据；图时空 Mamba 多假设完整输入。同期邻居失效时，简单图传播与强行节点扫描都可能传递错误证据。
4. **变量/节点扫描的顺序偏置。** 视觉和 Graph Mamba 都将路径或排序当作问题；时间序列论文中若沿变量扫描，常未充分检验站点编号置换的影响。
5. **多尺度与状态空间的功能重叠。** 多尺度 Mamba 已很密集。若没有分层缺口长度、频率和效率消融，“多尺度”不能构成解释。

### 6.2 评测层面

1. **插补与未来预测混淆。** 双向插补模型可以使用未来侧历史观测，因而不应作为严格因果预测结果。
2. **合成 mask 过于单一。** 常见 MCAR、固定长度块或固定比例，不能代表 MNAR、末端断连、邻站同步掉线与采样间隔变化。
3. **训练—测试缺失机制转移不足。** 同一种 mask 分布训练并测试，只说明同分布鲁棒性；缺少 source-mask 到 unseen-target-mask 的系统协议。通用机器学习已指出 missingness shift 可使源域选择不可靠。[论文](https://arxiv.org/abs/2406.16484)
4. **真实数据原生缺失利用有限。** AQI36、METR-LA 等具有原生无效/缺失记录，但常又被人工遮挡覆盖；应分别报告原生缺失、额外遮挡和目标有效值处理。
5. **不确定性不足。** 少量工作如 MAS-Mamba、Mamba-ProbTSF 讨论区间预测；多数方法仅报告 MAE/RMSE，未说明在长断连时何时应降低信心。[Mamba-ProbTSF](https://pmc.ncbi.nlm.nih.gov/articles/PMC12281171/)

### 6.3 工程与论证层面

- Mamba 的线性复杂度依赖融合 scan 等实现；在短窗口、少节点的公共基准上，实际速度/显存优势应测量而非假定。
- 强基线不能只用普通 Transformer/RNN；应包含 HD-TTS、BiTGraph、S4M 或其可复现替代、以及相同主干的 mask/interval 消融。
- 近期 MAP-SSM、MAS-Mamba 与 ACSC-GCRN 已使“mask 门控＋原型”“间隔门控”“观测条件空间补偿”分别高度拥挤，不能换一个名字重复做。

## 七、由跨领域证据导出的可行技术路线

下列是研究假设，不是已经被文献证明的创新。

| 优先级 | 问题与可借鉴机制 | 最小实现 | 必须反驳的近邻工作/风险 |
|---|---|---|---|
| 1 | **未见结构化缺失模式下的离线泛化。** 将视觉多视图一致性和通用 missingness-shift 思路用于同一历史窗的多 mask 视图；SSM 仅作为时间骨干。 | 训练使用 Point、Early Block-T、独立 Block-T；测试留出 Terminal Block-T、同步邻站断连。以多环境最坏组预测损失加状态一致性，对比简单多 mask 增广。 | 若简单增强已解决，或未见环境无稳定退化，则停止；不能仅做更多 mask 表格。 |
| 2 | **图约束的节点/变量状态空间聚合。** 借鉴 VMamba 的多路径扫描与 Graph Mamba 的 token 化/排序，解决节点 ID 无序。 | 先做顺序敏感性审计；仅在明显敏感时比较物理图遍历、邻域分组和集合聚合。 | MambaTS 已有变量感知扫描；必须证明缺失节点图任务中的新增问题。 |
| 3 | **按缺口位置与长度分工局部/长程证据。** 借鉴 U-Mamba 的局部—全局混合和 MAS-Mamba 的间隔状态控制。 | 在已有时空主干中仅添加 mask/间隔感知时间状态，并按早期块、末端块、同步块分层评测。 | MAS-Mamba 和 ACSC-GCRN 已分别覆盖间隔门控、可观测性空间补偿；不能再把简单 gate 当贡献。 |
| 4 | **长断连下的可信预测。** 借鉴 CRU 的状态不确定性思路，使不确定性随观测间隔/邻域支持变化。 | 先为点预测主干增加轻量方差或分位数头，评测区间覆盖和宽度。 | 若只报告点预测 MAE 或没有校准检验，不足以宣称可信。 |

第一条与当前已有缺失时空工程的兼容性最高，但需要先做跨 mask 诊断；第二条差异化更强、风险也更高；第三条因近期直接竞争而不宜单独立项；第四条适合作为毕业论文后续章节，而非一周内的主实验。

## 八、建议的验证顺序

1. **先做问题存在性验证。** 在 METR-LA/PEMS-BAY 或 AQI36 保持总体缺失率一致，比较同分布与未见结构 mask；同时记录最长缺口、末次观测距离、节点同步失效比例。
2. **再做最小方法验证。** 先比较固定 mask 训练、简单多 mask 混合、最坏环境训练；只有混合不足时再加入状态一致性。
3. **最后更换时间骨干。** 在相同协议下比较当前可靠时空主干与 mask-aware S4/Mamba。若 SSM 不带来未见 mask 泛化或效率收益，就不要把它写成主创新。
4. **保持因果边界。** 任何双向结构都只处理预测起点以前的窗口；目标未来值不参与历史插补或 mask 生成。
5. **报告失败和代价。** 除 MAE/RMSE 外，报告参数量、训练/推理时间、显存、完整观测退化、分模式结果与置信区间。

## 九、可直接用于论文相关工作的核心参考

1. Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM 2024. [论文](https://arxiv.org/abs/2312.00752)  
2. Peng, J. et al. *S4 for Multivariate Time Series Forecasting with Missing Values*. ICLR 2025. [论文](https://arxiv.org/abs/2503.00900)  
3. López Alcaraz, J. M. & Strodthoff, N. *Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models*. TMLR 2022. [论文](https://arxiv.org/abs/2208.09399)  
4. Schirmer, M. et al. *Modeling Irregular Time Series with Continuous Recurrent Units*. ICML 2022. [论文](https://proceedings.mlr.press/v162/schirmer22a.html)  
5. Cini, A. et al. *Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling*. ICML 2024. [论文](https://arxiv.org/abs/2402.10634)  
6. Chen, X. et al. *Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values*. ICLR 2024. [论文](https://openreview.net/pdf?id=O9nZCwdGcG)  
7. Solís-García, J. et al. *TIMBA: Time Series Imputation with Bi-directional Mamba Blocks and Diffusion Models*. 2024 预印本. [论文](https://arxiv.org/abs/2410.05916)  
8. Huang, C.-I. & Chang, W.-Y. *MAS-Mamba*. IET Intelligent Transport Systems, 2026. [DOI](https://doi.org/10.1049/itr2.70304)  
9. Zhu, L. et al. *Vision Mamba*. ICML 2024. [论文](https://arxiv.org/abs/2401.09417)  
10. Liu, Y. et al. *VMamba*. NeurIPS 2024. [论文](https://arxiv.org/abs/2401.10166)  
11. Ma, J. et al. *U-Mamba*. 2024. [论文](https://arxiv.org/abs/2401.04722)  
12. Behrouz, A. & Hashemi, F. *Graph Mamba*. 2024. [论文](https://arxiv.org/abs/2402.08678)  
13. Rockenschaub, P. et al. *Robust Prediction under Missingness Shifts*. 2024. [论文](https://arxiv.org/abs/2406.16484)

## 十、检索与证据说明

检索覆盖 S4/S5/Mamba 原始论文，视觉、医学影像、图学习、音频的代表性应用，以及缺失/不规则时间序列、图时空预测和缺失机制转移。优先使用原论文、PMLR/OpenReview/arXiv、出版社 DOI 页面和官方代码。对 MAS-Mamba、MAP-SSM、EMIRA 等无法完整取得同行评议正文或代码的工作，文中只使用可核验摘要/记录所支持的结论，并保留其证据状态。

本报告的作用是限定问题与技术路线，不将跨领域结构的成功自动外推为时间序列缺失预测效果；每一条路线仍需要在严格的因果、缺失模式和外部数据协议下验证。


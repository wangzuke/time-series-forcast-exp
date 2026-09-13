# HD-TTS / GinAR 方法核对与改造边界

> 日期：2026-09-02  
> 目的：在基于 HD-TTS 作为 base 进行方法改造前，明确两篇核心支撑工作的模型设计、官方代码位置、可继承部分和新增边界。

## 1. 官方代码来源

| 工作 | 官方代码 | 本地位置 | 当前 commit | 用途 |
|---|---|---|---|---|
| HD-TTS | https://github.com/marshka/hdtts | `external_repro/official/hdtts` | `46a8717db3802e1684e991ba0a7c0bfe43d22535` | base / baseline 源码参照 |
| GinAR | https://github.com/GestaltCogTeam/GinAR | `external_repro/official/ginar` | `089f01135bf38381d17f8265b1c753703955518b` | 缺失下图关系重建机制参照 |

说明：`external_repro/` 已在 `.gitignore` 中，用于保存官方仓库原样副本；我们的方法代码应放在主工程的独立目录中，避免污染官方实现。

## 2. HD-TTS 原文方法设计

HD-TTS 的核心问题是：在图时序预测中，观测缺失会破坏局部时间上下文和局部空间传播；因此模型应在多个时空尺度上计算表示，并根据观测/缺失模式自适应组合这些尺度。

模型主干：

```text
X, mask, exogenous features
  -> mask-aware Encoder
  -> DRNN temporal hierarchy
  -> k-MIS spatial pooling hierarchy
  -> lift pooled representations back to original nodes
  -> attention readout over temporal-scale x spatial-scale states
  -> forecasting output
```

官方代码对应：

| 模块 | 文件 | 关键实现 |
|---|---|---|
| 总模型 | `external_repro/official/hdtts/lib/nn/models/hdtts_model.py` | `HDTTSModel.forward()` 串联 encoder、DRNN、hierarchical pooling、attention readout |
| 缺失感知输入编码 | `external_repro/official/hdtts/lib/nn/layers/encoder.py` | value、node embedding、mask、exogenous feature 分别线性映射后相加 |
| 时间层级 | `external_repro/official/hdtts/lib/nn/layers/drnn.py` | dilated recurrent layers，每层产生一个 temporal scale 的最后状态 |
| 空间粗化 | `external_repro/official/hdtts/lib/nn/layers/pooling.py` | k-MIS pooling，逐层 pool / lift，并在粗化前后做 message passing |
| 多尺度融合 | `external_repro/official/hdtts/lib/nn/layers/readout.py` | `AttentionReadout` 对空间层级和时间层级状态做 attention 加权 |
| 训练入口 | `external_repro/official/hdtts/experiments/run_realworld.py` | 构造 TSL 数据集、mask、covariates、Lightning predictor |

官方默认配置中，HD-TTS 使用 `hidden_size=64`、`dilation=3`、`rnn_layers=4`、`pooling_layers=3`；交通数据配置为 `window=24`、`horizon=12`。本轮新 runner 的正式默认值应尽量向这一配置靠齐，smoke 才使用更小参数。

原文机制拆解：

| 组件 | 原文设计 | 对我们 idea 的含义 |
|---|---|---|
| 输入编码 | 将观测值、外生变量、mask、节点嵌入编码为节点-时间表示；缺失值在数值上用 last observation carried forward，但 mask 同时进入模型 | 不能把填充值当真实观测；关系估计必须使用 mask 区分 observed / filled |
| 时间处理 | 用 GRU + temporal downsampling 形成类似 DRNN 的时间层级，每层对应不同 temporal scale | 可作为高缺失下时间上下文不足时的多尺度补充 |
| 空间处理 | 对每个 temporal scale，再用 SMP + k-MIS pooling 形成 spatial hierarchy；粗节点表示再通过 pseudo-inverse lift 回原节点 | 直接支撑 coarse / regional representation 作为预测信息源 |
| 解码器 | 对 `L * (K + 1)` 个时空尺度表示做 attention reweight，再用 MLP 预测 horizon | 支撑 missingness-aware scale fusion；我们的 reliability gate 应放在 readout score 或 readout 前 |
| 训练目标 | 端到端最小化 forecasting error，不把插补作为必要预处理 | 与我们的 forecasting-oriented 目标一致 |

原文局限对我们的启发：

- k-MIS pooling 是 sparse non-trainable pooling，效率高，但强依赖输入图拓扑；当图拓扑和真实动态相关不一致时会引入偏置。
- HD-TTS 已经能根据状态和缺失模式重加权多尺度表示，但没有显式估计 edge observed overlap，也没有构造 observed-only temporal graph。
- 附录交通实验指出 METR-LA / PEMS-BAY 的图存在 disconnected nodes 和 topology/data mismatch，这正好支持我们引入 `A_temp`、adaptive graph 和 reliability。

可继承点：

- 以 HD-TTS 作为主干 base 是合理的，因为它已经天然包含 temporal hierarchy、spatial hierarchy 和 missing-aware readout。
- `input_mask` 是模型显式输入，适合进一步扩展为 missingness reliability。
- k-MIS pooling 给出了空间 coarse graph 的一条官方路径，可作为默认 coarse branch。

不足边界：

- HD-TTS 的空间粗化主要由原始空间图拓扑决定，没有显式构造 `observed-only temporal graph`。
- 它的多尺度 attention 是 state/mask 条件化的，但没有把 edge observed overlap、region coverage、region dynamic consistency 作为独立可靠性信号。
- 当原始交通图拓扑与真实动态依赖不一致时，附录交通实验也显示非可训练 pooling 可能受影响。

## 3. GinAR 原文方法设计

GinAR 的核心问题是 variable missing：部分变量的整段历史观测缺失，常规 STGNN 会在不完整变量集合上学习错误的 temporal dependency 和 spatial correlation。

模型主干：

```text
zero-filled variable-missing input
  -> Interpolation Attention
  -> predefined graph convolution
  -> adaptive graph convolution
  -> recursive cell update
  -> multi-step forecasting
```

官方代码对应：

| 模块 | 文件 | 关键实现 |
|---|---|---|
| GinAR cell | `external_repro/official/ginar/model1/ginar_cell.py` | `ginarCell.forward()` 中先做 IA，再融合 predefined graph 与 adaptive graph |
| Interpolation Attention | `external_repro/official/ginar/model1/ginar_cell.py` | `InterpositionAttention` 基于节点嵌入学习注意力，用正常变量信息补偿缺失变量表示 |
| Adaptive graph | `external_repro/official/ginar/model1/ginar_cell.py` | 节点可学习嵌入 + 当前表示生成 batch-wise adaptive adjacency |
| 主入口 | `external_repro/official/ginar/main_ginar.py` | 训练与评估入口 |

可继承点：

- predefined graph 与 adaptive graph 互补，直接支撑我们使用 `A_phy + A_temp`。
- 在缺失下不能直接信任固定图关系，图关系需要由观测可用性和当前表示修正。
- variable missing 的极端场景可以作为我们方法的高缺失率压力测试之一。

原文机制拆解：

| 组件 | 原文设计 | 对我们 idea 的含义 |
|---|---|---|
| Variable missing 设置 | 按 25%、50%、75%、90% 随机选择变量，将这些变量的输入历史值置 0，预测所有变量未来值 | 对应我们已有 `variable_channel` 缺失类型，是验证图关系重建能力的好场景 |
| Interpolation Attention | 先学习变量 correspondence，再用正常变量的表示加权恢复缺失变量表示 | 支撑“高缺失下不能直接在残缺变量上做时序建模” |
| Predefined graph | 交通数据用道路距离图；无道路网络数据用 Pearson correlation 构图 | 支撑先验图作为基本空间依赖 |
| Adaptive graph | 用变量嵌入和 IA 后的变量表示生成数据驱动 adjacency | 支撑缺失下需要动态/自适应关系补充固定图 |
| AGCN | 将 predefined graph 和 adaptive graph 两路图卷积结果相加，再做 layer normalization | 可映射为我们的 `A_phy + A_temp/adaptive` fine graph fusion |
| Decoder | 取每层 GinAR 的最后 hidden state 拼接，用直接多步预测避免自回归误差累积 | 与我们 direct forecasting 一致 |

实验与消融结论：

- 数据集：METR-LA、PEMS-BAY、PEMS04、PEMS08、China AQI。
- 历史长度和预测长度均为 12，指标为 12-step 平均预测结果。
- 缺失率为 25%、50%、75%、90%，每个缺失率重复 5 个 seed。
- 消融包括 `w/o ia`、`w/o pg`、`w/o ag`。原文结论是：低缺失率下 predefined graph 更重要；高缺失率下 adaptive graph 更重要。

不足边界：

- GinAR 没有 HD-TTS 式 temporal/spatial hierarchy，也没有 regional/coarse graph。
- GinAR 的缺失模式主要是 whole-variable missing，不覆盖 point/block/spatiotemporal block 的完整组合。
- 它没有显式 edge reliability、region coverage 或 fine/coarse residual gate。

## 4. 我们的方法边界

当前方法不应被表述为“重新发明 HD-TTS”或“重新实现 GinAR”，而应表述为：

```text
HD-TTS base
  + GinAR-style fixed/adaptive graph complementarity
  + observed-only temporal relation graph
  + edge/region reliability
  + missingness-aware fine/coarse fusion
```

最小实现应保留 HD-TTS 的主体结构：

- mask-aware input encoder
- temporal hierarchy
- spatial hierarchy / coarse graph
- multi-scale readout
- forecasting-oriented loss

新增模块只放在三个明确位置：

1. **图构造**：从输入历史的真实观测值和 `input_mask` 构造 `A_temp`，并计算 edge observed overlap reliability。
2. **图融合**：将原始空间图 `A_phy` 与 `A_temp` 融合为 fine graph；高缺失/低 overlap 边需要降权。
3. **尺度融合**：在 HD-TTS readout 前或 readout score 中加入 region coverage、edge reliability、missing statistics，用于控制 fine/coarse 表示的权重。

## 5. 代码管理约定

- 官方代码保持在 `external_repro/official/`，不直接修改。
- 我们的改造代码放在 `missing_ts_exp/src/models/hdtts_reliable.py` 和对应 runner/config 中。
- 代码中需要明确引用官方 HD-TTS 的模块结构；若当前环境不能直接 import 官方依赖，则先实现可运行的 prototype，但不能把 prototype 记作官方 baseline。
- 后续严格 baseline 需要单独创建 HD-TTS 官方环境，并直接调用 `external_repro/official/hdtts/experiments/run_realworld.py`。

## 6. 当前代码进展

已新增：

- `missing_ts_exp/src/models/hdtts_reliable.py`：HD-TTS-based reliable spatiotemporal prototype。
- `missing_ts_exp/src/models/hdtts_official_reliable.py`：直接继承官方 `HDTTSModel` 的 overlay 版本，只改图构造/图输入逻辑。
- `missing_ts_exp/src/training/run_hdtts_reliable_spatial.py`：专门面向带空间图数据的训练入口。
- `missing_ts_exp/src/training/run_official_hdtts_local.py`：用官方 HD-TTS 模块读取本地 METR-LA/AQI36 的 runner，支持 `official_base` 与 `official_reliable` 对照。
- `missing_ts_exp/scripts/r0902_hdtts_reliable_cmds.txt`：smoke 与短实验命令清单。
- `missing_ts_exp/scripts/r0902_official_hdtts_overlay_smoke.py`：官方 HD-TTS overlay 的 forward/backward smoke。

已通过 smoke：

| 数据集 | 命令标签 | 设置 | 结果文件 |
|---|---|---|---|
| METR-LA | `smoke_metrla_cpu_default_dilation` | CPU；`seq_len=12`、`pred_len=3`、`variable_channel=0.5`；每阶段 1 个 batch | `missing_ts_exp/results/r0902_hdtts_reliable/smoke_metrla_cpu_default_dilation.json` |
| AQI36 | `smoke_aqi36_cpu` | CPU；`seq_len=12`、`pred_len=3`、`variable_channel=0.5`；每阶段 1 个 batch | `missing_ts_exp/results/r0902_hdtts_reliable/smoke_aqi36_cpu.json` |
| synthetic graph | `official_hdtts_overlay_smoke` | `hd-tts` 环境；官方 `HDTTSModel` 子类；随机张量 forward/backward | 输出 `official_reliable_forward (2, 3, 10, 1)` |
| METR-LA | `smoke_official_base_metrla_cpu` | `hd-tts` 环境；本地 METR-LA；官方 `HDTTSModel`；每阶段 1 个 batch | `missing_ts_exp/results/r0902_official_hdtts_local/smoke_official_base_metrla_cpu.json` |
| METR-LA | `smoke_official_reliable_metrla_cpu_nodeemb` | `hd-tts` 环境；本地 METR-LA；官方 overlay；每阶段 1 个 batch | `missing_ts_exp/results/r0902_official_hdtts_local/smoke_official_reliable_metrla_cpu_nodeemb.json` |
| METR-LA | `smoke_official_reliable_metrla_gpu` | `hd-tts` 环境；本地 METR-LA；官方 overlay；GPU；每阶段 1 个 batch | `missing_ts_exp/results/r0902_official_hdtts_local/smoke_official_reliable_metrla_gpu.json` |
| AQI36 | `smoke_official_reliable_aqi36_cpu` | `hd-tts` 环境；本地 AQI36；官方 overlay；每阶段 1 个 batch | `missing_ts_exp/results/r0902_official_hdtts_local/smoke_official_reliable_aqi36_cpu.json` |

已通过成对短验证：

| 数据集 | 缺失 | 设置 | official base | official reliable | 说明 |
|---|---|---|---|---|---|
| METR-LA | `variable_channel=0.5` | GPU；`seq_len=24`、`pred_len=12`；2 epoch；每 epoch 20 train batches、10 eval batches | val MSE `0.7535`，test MSE `0.9674` | val MSE `0.7230`，test MSE `0.9503` | 仅验证训练链路和对照入口可用，不作为效果结论 |

注意：当前 `itransformer` 环境没有 `torch_geometric` 和 `tsl`，不能直接运行 HD-TTS 官方 baseline；但本机已有 `hd-tts` 环境，已验证可 import 官方 `HDTTSModel` 并运行 overlay smoke。当前 PyTorch-only prototype 可用于本项目训练栈快速实验；官方 overlay 才是更贴近“基于 HD-TTS base 改造”的代码路径。严格 baseline 仍应直接调用 `external_repro/official/hdtts/experiments/run_realworld.py`。

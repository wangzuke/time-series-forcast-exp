# 0908 实验计划：官方基线与 MR-MSGF 受控验证

## 1. 目标与边界

本轮只补充与研究问题一致的时空缺失预测对照，不纳入 CoIFNet、S4M 或 GinAR。实验回答两个不同问题，结果不得混在同一张表中：

1. **官方复现：** HD-TTS 论文中的内部模型在其原始 METR-LA Block-T 协议下的表现如何；BiTGraph 在其原始 METR-LA 随机 Point missing 协议下的表现如何。
2. **受控方法验证：** 在完全相同的数据划分、预测设定和固定 Block-T mask 下，MR-MSGF 相对于 HD-TTS-AMP 和 BiTGraph 官方架构是否有效。

BiTGraph 的官方训练脚本只提供 `mask_ratio`，不提供外部 mask 文件接口。其原始 Point-missing 运行仅保留为环境与官方复现记录，**不进入论文横向对比**。为得到可比较的 baseline，新增一个外部统一协议适配器：不修改作者仓库中的模型代码，而是直接导入其 `BiaTCGNet.Model`，将数据加载、固定 Block-T mask、窗口长度和评测目标对齐至 C 组。论文中将其准确命名为“BiTGraph official architecture under unified Block-T protocol”，而不是“BiTGraph 原始脚本结果”。

## 2. 忠实实现原则

- HD-TTS 内部基线使用 `external_repro/official/hdtts/experiments/run_realworld.py` 与官方 Hydra 配置；不以自写的 TTS 近似实现替代。
- BiTGraph 使用作者仓库 `chenxiaodanhit/BiTGraph` 的原始 `main.py` 与 `test_forecasting.py`；仓库固定 commit、依赖版本、数据文件哈希和完整命令均须记录。
- MR-MSGF 仅替换 HD-TTS-AMP 的图关系和细到粗尺度传播模块；时间编码、时空下采样、训练损失、数据划分及其他主干超参数均对齐 HD-TTS-AMP 的官方 LA 配置。
- 同一受控比较中的模型复用同一个 mask 文件，记录 SHA-256、实际缺失率和随机种子。评测仅在原始有效观测目标上进行。

## 3. 资源与协议

| 组别 | 数据集 | 缺失模式 | 设置 | 方法 |
|---|---|---|---|---|
| A: HD-TTS 官方复现 | METR-LA | Block-T | 官方 `p_noise=0.05`、`p_fault=0.01`、`min_seq=12`、`max_seq=48` | HD-TTS-AMP、HD-TTS-IMP、TTS-AMP、TTS-IMP |
| B: BiTGraph 官方原始复现 | METR-LA | Point | 官方 `mask_ratio`；0.3/0.5/0.7 | BiTGraph（仅附录复现） |
| C: 受控主比较 | METR-LA | Block-T | 固定实际缺失率约 50%/70%，3 个 seed | HD-TTS-AMP、BiTGraph 官方架构、MR-MSGF |

组 A 和组 C 的 Block-T 都采用 HD-TTS 的 fault-plus-noise 定义，但组 A 使用官方运行器的原生生成掩码，组 C 使用预存固定掩码。因此只在各自组内做数值比较。

## 4. 分阶段执行

### Stage 0：可追溯性审计

1. 记录 HD-TTS 与 BiTGraph 官方仓库 URL、commit、`git status`。
2. 校验 METR-LA 的 shape、值域、SHA-256，并建立对 BiTGraph 所需目录的只读软链接。
3. 记录官方脚本可接受的参数；禁止将历史上为 Block-T 写入的 BiTGraph 改动带入本轮。

### Stage 1：冒烟验证

1. 四个 HD-TTS 官方模型各跑 2 epoch、5 个训练 batch，确认训练、验证、测试闭环和指标输出。
2. BiTGraph 以 `mask_ratio=0.5` 跑 2 epoch，确认原始训练和测试脚本均能运行。
3. MR-MSGF 与 HD-TTS-AMP 在同一固定 Block-T 50% mask 下各跑 2 epoch，验证形状、掩码哈希和数值有限性。

任何 smoke 失败先修复环境或数据适配；不改写官方模型结构，也不启动正式矩阵。

### Stage 2：正式实验

1. **A 组：** 四个 HD-TTS 官方模型在 METR-LA Block-T 的官方参数下各 3 个训练 seed。若官方脚本的 mask seed 与训练 seed 绑定，额外固定并记录两者。
2. **B 组：** BiTGraph 在 METR-LA Point missing 的 `mask_ratio={0.3,0.5,0.7}` 下各 3 次独立官方运行。经源码审计，作者代码虽解析 `--seed`，但未用它初始化 NumPy/Torch RNG；因此这些运行记录为 independent replicates，而不虚称为受控 seed。保持官方窗口、预测长度、模型定义与评价脚本；本表不与 C 组作逐格数值比较。
3. **C 组：** 在既有的固定 Block-T 50%/70% mask（seed 2024/2025/2026）上，使用 24 步历史、12 步预测、同一 HD-TTS 数据划分和同一 unmasked 测试目标，对比 HD-TTS-AMP、full MR-MSGF 与 BiTGraph 官方架构。BiTGraph 仅用外部适配器替换其随机 Point loader；`BiaTCGNet.Model`、默认图网络参数、Adam 优化器、完整 200 epoch 和 best-validation checkpoint 选择均遵循作者代码。新增 6 次 BiTGraph 训练。

### Stage 3：汇总与验收

- 每次运行输出 JSON/原始日志/配置快照，且 `loss`、`MAE`、`RMSE` 必须为有限值。
- C 组按三 seed 报告 mean +/- std，并以配对差值报告 MR-MSGF 相对 HD-TTS-AMP 和 BiTGraph 的改变量。
- A、B、C 分表展示；图注与正文明确说明各自的 mask 和协议。
- `MAPE` 在 METR-LA 接近零速度时不作为主结论，主指标为 MAE 和 RMSE。

## 5. 不做的实验

- 不修改 BiTGraph 作者仓库中的网络源码，也不使用自行重写的 BiTGraph 变体；统一 Block-T 运行只允许通过外部数据协议适配器直接调用作者的 `BiaTCGNet.Model`。
- 不纳入非时空预测的 CoIFNet、S4M。
- 不做 Block-ST 压力测试；它会同时遮蔽空间邻域，与本文“用仍可观测空间邻域补偿目标节点时间缺失”的假设不匹配。

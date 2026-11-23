# 原文对齐与 batch-size 核验

日期：2026-11-06。适用范围：缺失模式漂移主实验（输入长度 96、预测长度 24/96、缺失率 30%、两训练随机种子）。此前启动的 BiTGraph/CTF 队列已中断，其不对齐中间结果已移至 `exp/{results,checkpoints,logs}/archive_pre_alignment_non_s4m/`，不会被新实验复用。

## 对齐结论

| 模型 | 采用设置 | 原文/官方依据 | 与本课题协议的必要差异 |
|---|---|---|---|
| BiTGraph | blocks=3，top-k=10，Adam lr=1e-3，batch=32，200 epochs | ICLR 2024 Sec. 5.1；官方 README 的 `--epochs 200` 命令 | 论文使用 H=F=24、0.6/0.2/0.2 切分；本研究固定 L=96、H=24/96，并保持统一数据切分，以隔离缺失模式漂移。 |
| ChannelTokenFormer | Adam；ETTh1 lr=1e-4，Weather lr=1e-2；e_layers=2；最多 10 epochs、patience=3；ETTh1: d_model/d_ff=256/1024、patch=24、global tokens=2；Weather: 128/128、patch=12、global tokens=1 | ICLR 2026 Appendix A.2；发布的 `CTF_ETTh1.sh` 与 `CTF_weather_practical.sh` | 论文主实用协议用 ETT1/Weather 的异步采样和较长窗口；本研究用完备原始数据人为注入 P/B 缺失，故不能复制其异步数据构造。 |

CTF 的 `keep_prob=1` 保持发布 ETTh1/Weather 脚本一致。论文出现的“随机 patch mask 比例 0.4”是 SolarWind 专门的缺失鲁棒性实验，不能作为 ETTh1/Weather 的默认超参数移植。

CTF 发布的 missing-aware tokenizer 在删除全缺失 patch 时访问 `missing_flag[0, ...]`，故原版面对“每样本、每通道独立注入缺失”的协议只在 batch=1 时正确。该限制已在后续补丁中修复：每个样本独立压缩有效 patch，以 padding 和逐样本 attention mask 组成 batch；B=1 的前向与梯度均与原版严格一致。详见 `1106-CTF批量缺失tokenization核验.md`。

## BiTGraph batch-size 烟测

设备为 RTX 3090 24GB（GPU 6）。统一使用 ETTh1、P 训练、B(24) 测试、种子 2024、完整训练集一个 epoch（8,521 个窗口）；验证和测试各仅一批，故以下是吞吐/显存测试，**不是**可比较的性能实验。

| batch size | 一 epoch 墙钟时间 (s) | 峰值 allocated / reserved (MiB) |
|---:|---:|---:|
| 16 | 23.429 | 156.2 / 192.0 |
| 32 | 12.962 | 182.8 / 310.0 |
| 64 | 17.857 | 222.1 / 360.0 |
| 128 | 10.695 | 297.4 / 440.0 |
| 256 | 8.987 | 452.4 / 486.0 |

结论：硬件并不限制 BiTGraph 的 batch；吞吐上 256 最快，但它会改变优化动态，正式主实验仍固定论文规定的 **batch=32**。这也解释了显存占用低并不表示配置错误：该模型在 7 维 ETTh1 上的图卷积规模本来很小。

本记录中的旧 CTF batch=1 可运行性检查已被后续 batched 实现的等价性与吞吐测试取代；主实验配置使用 batch=32。

原始机器可读结果：`exp/results/batch_size_paper_alignment/`。

## 新任务清单

`exp/configs/primary_paper_aligned_non_s4m_2seeds.json` 已生成：64 个训练任务、320 个固定测试评估。

- BiTGraph：ETTh1、Electricity，各 16 个训练任务；
- ChannelTokenFormer：ETTh1、Weather，各 16 个训练任务；
- 每个模型/数据集组合均含 H=24/96、P 训练及 B(12/24/48) 训练、训练种子 2024/2025；测试掩码种子为 9101/9102。

`run_primary_worker.py` 现支持 `--baselines` 筛选，因而可让 GPU 6 仅跑 BiTGraph、GPU 7 仅跑 CTF，同时仍共享此清单并保持任务 ID 一致。

# CTF 批量缺失 tokenization 核验

## 动机与实现

发布的 `ChannelTokenFormer_missing` 在判定某 patch 是否全缺失时固定读取 `missing_flag[0, ...]`。当一个训练 batch 中不同时间窗口有不同的 P/B 掩码时，后续样本会错误复用第一个样本的 token 布局。

本地分支 `codex/batched-missing-tokenization`（commit `7a938d9`）加入无新增可训练参数的 `PatchlenWiseEmbedding_missing_batched`：

1. 每个样本、每个变量独立保留非全缺失 patch，并压缩到本样本 token 前缀；
2. 将同一变量内较短样本 padding 到该 batch 的最大 token 数；
3. 以逐样本 attention mask 屏蔽 padding token；
4. 对没有任何可用 patch 的变量，保留原实现的“零 patch”回退规则。

## 正确性验证

脚本：`exp/scripts/validate_ctf_batched_missing.py`，GPU 0，随机种子 2024。

| 检查 | 结果 |
|---|---:|
| B=1，旧版与新版前向最大绝对误差 | 0.0 |
| B=1，旧版与新版梯度最大绝对误差 | 0.0 |
| B=4、四个独立缺失模式前向输出 | 全部有限 |
| B=4、反向梯度 | 全部有限 |

因此扩展在 B=1 上严格保持发布代码语义，同时支持独立掩码的 B>1 训练。

## CTF batch-size 吞吐测试

设备：RTX 3090 24GB（GPU 0）。ETTh1，L=96、H=24、P 训练、B(24) 测试、缺失率 30%、训练种子 2024；每个设置恰好处理 256 个训练窗口，验证和测试各一批。该测试只比较吞吐/显存，不比较预测精度。

| batch size | 墙钟时间 (s) | 样本吞吐 (windows/s) | 峰值 allocated / reserved (MiB) |
|---:|---:|---:|---:|
| 1 | 8.656 | 29.6 | 171.6 / 206.0 |
| 8 | 2.520 | 101.6 | 183.2 / 316.0 |
| 16 | 1.980 | 129.3 | 194.8 / 318.0 |
| 32 | 1.925 | 133.0 | 135.7 / 170.0 |

结论：采用 **batch=32**。它与发布运行器的默认值一致，吞吐在已测设置中最高或与最高持平，并且显存远低于 24GB。正式运行仍使用论文对应的 10 epochs、patience=3，以及 ETTh1/Weather 各自学习率。

机器可读验证与基准结果在 `exp/results/validation/ctf_batched_missing.json` 和 `exp/results/batch_size_paper_alignment/`；它们被 `.gitignore` 排除，可由脚本重建。

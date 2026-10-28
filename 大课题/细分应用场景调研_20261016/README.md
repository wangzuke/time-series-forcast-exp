# 细分应用场景调研 · 2026-10-16

先读 [基于公开数据集的场景修订.md](/media/data1/wangzk/time-series-forcast-exp/大课题/细分应用场景调研_20261016/基于公开数据集的场景修订.md)。当前建议以公开基准支持的“缺失条件下多站点空气质量短期预测”为场景，交通短期预测为备选；高浓度等现象降为可选分析，不再限定主任务。前一版[调研结论与场景选择.md](/media/data1/wangzk/time-series-forcast-exp/大课题/细分应用场景调研_20261016/调研结论与场景选择.md)保留文献比较和数据审计。

文件说明：

- `论文场景证据表.md/.csv/.json`：24 个场景比较条目、3 个边界条目；每篇记录数据类型、规模、维度、频率、跨度、缺失作用、方法及核验边界。
- `检索返回记录.json`：保存的部分网络检索与回查结果；不据此声称完整系统检索数量。
- `audit_public_data.py`：只读原始数据，输出统计及图；不训练模型、不填补缺失。
- `数据可用性审计.json`：原始数据摘要、站点缺失率、连续缺口、训练段样本计数和 SHA256。
- `站点原始缺失率.png`：按缺失率排序的站点统计图，不是实验效果图。
- `数据审计输入/`：本轮从 UCI 下载的原始压缩包。AQI36 使用仓库原有文件，不再复制。

数据来源：

- [Beijing Multi-Site Air Quality，UCI](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)，下载入口 `https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip`。按原始仓库要求保留数据来源与许可信息。
- [Parking Birmingham，UCI](https://archive.ics.uci.edu/dataset/482/parking+birmingham)，下载入口 `https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip`。数据使用遵循原数据许可。

在当前环境复现统计：

```bash
MPLCONFIGDIR=/tmp/scenario_mpl_cache /media/data1/wangzk/miniconda3/envs/hd-tts/bin/python /media/data1/wangzk/time-series-forcast-exp/大课题/细分应用场景调研_20261016/audit_public_data.py
```

本轮新增文件均放在本目录，旧文档保留为讨论历史。本轮不启动模型训练；后续研究需要按主报告的验证条件确定是否继续。

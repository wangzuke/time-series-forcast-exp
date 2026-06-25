"""数据集元数据与默认实验设置。"""
from __future__ import annotations
import os

# 数据根目录（CSV 所在）
DATA_ROOT = "/data/wangzuke/time-series-forecast-exp/dataset"

# 每个数据集的相对路径与列信息
DATASETS = {
    "ETTh1": {
        "path": os.path.join(DATA_ROOT, "ETT-small/ETTh1.csv"),
        "date_col": "date",
        "freq": "h",
        # 时间粒度小时
        "split": "ett_hourly",
        "n_features": 7,
    },
    "ETTm1": {
        "path": os.path.join(DATA_ROOT, "ETT-small/ETTm1.csv"),
        "date_col": "date",
        "freq": "t",  # 15 分钟
        "split": "ett_minutely",
        "n_features": 7,
    },
    "Weather": {
        "path": os.path.join(DATA_ROOT, "weather/weather.csv"),
        "date_col": "date",
        "freq": "t",  # 10 分钟
        "split": "ratio",
        "ratio": (0.7, 0.1, 0.2),
        "n_features": 21,
    },
    "Electricity": {
        "path": os.path.join(DATA_ROOT, "electricity/electricity.csv"),
        "date_col": "date",
        "freq": "h",
        "split": "ratio",
        "ratio": (0.7, 0.1, 0.2),
        "n_features": 321,
    },
    "Traffic": {
        "path": os.path.join(DATA_ROOT, "traffic/traffic.csv"),
        "date_col": "date",
        "freq": "h",
        "split": "ratio",
        "ratio": (0.7, 0.1, 0.2),
        "n_features": 862,
    },
    "ExchangeRate": {
        "path": os.path.join(DATA_ROOT, "exchange_rate/exchange_rate.csv"),
        "date_col": "date",
        "freq": "d",
        "split": "ratio",
        "ratio": (0.7, 0.1, 0.2),
        "n_features": 8,
    },
}

# ETT 数据集按论文/iTransformer 仓库标准切分：训练 12 月、验证 4 月、测试 4 月
# 小时粒度 30*24=720 步/月；分钟粒度 30*24*4=2880 步/月
ETT_HOURLY_SPLIT = {
    "train_end": 12 * 30 * 24,
    "val_end": 12 * 30 * 24 + 4 * 30 * 24,
    "test_end": 12 * 30 * 24 + 8 * 30 * 24,
}
ETT_MINUTELY_SPLIT = {
    "train_end": 12 * 30 * 24 * 4,
    "val_end": 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
    "test_end": 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
}

# 缺失率默认列表（精简方案：只保留 10% 与 30%）
MISSING_RATES = [0.1, 0.3]
# 缺失类型
MISSING_TYPES = ["random_point", "continuous_segment", "variable_channel", "mixed"]
# 连续片段缺失长度候选
SEGMENT_LENGTHS = [12, 24, 48]
# 实验种子
SEEDS = [2024, 2025, 2026]

# 主实验默认输入/输出长度
MAIN_SEQ_LEN = 96
MAIN_PRED_LENS = [96, 336]
# 扩展实验
EXT_SEQ_LENS = [96, 336]
EXT_PRED_LENS = [96, 192, 336, 720]

# 第二轮实验设置
R2_DATASETS = ["Weather", "Electricity", "Traffic"]
R2_SEEDS = [2024, 2025]
HIGH_MISSING_RATES = [0.5, 0.7]

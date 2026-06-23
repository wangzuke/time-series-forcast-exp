"""统一的方法注册表：将"缺失处理 + 预测/补值模型"封装为同一接口。

方法分类：
1. baseline:        无缺失（mask 全 1）+ 预测模型
2. simple_impute:   简单填补 + 预测模型 (mean/forward/linear)
3. saits_impute:    SAITS 强补值 + 预测模型
4. missing_aware:   缺失感知模型 (MissTSM / CRIB / CoIFNet)

统一封装类 `Pipeline`：
    fit(train_loader, val_loader, ...)
    predict(batch)
"""
from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..models.dlinear import DLinear
from ..models.patchtst import PatchTST
from ..models.itransformer import iTransformer
from ..models.misstsm import MissTSMModel
from ..models.crib import CRIB
from ..models.coifnet import CoIFNet
from ..imputation.saits import SAITS, random_mit_mask, masked_mae
from ..imputation.simple import apply_simple_imputer
from ..data.timefeatures import time_feature_dim


PREDICTOR_REGISTRY = {
    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
}


def build_predictor(name: str, seq_len: int, pred_len: int, n_channels: int, time_feat_dim: int = 0, **kwargs):
    if name == "DLinear":
        return DLinear(seq_len, pred_len, n_channels, **kwargs)
    if name == "PatchTST":
        return PatchTST(seq_len, pred_len, n_channels, **kwargs)
    if name == "iTransformer":
        return iTransformer(seq_len, pred_len, n_channels, time_feat_dim=time_feat_dim, **kwargs)
    raise ValueError(name)


@dataclass
class PipelineConfig:
    method: str = "baseline"
    predictor: str = "iTransformer"
    impute_strategy: str = "zero"  # zero|mean|forward|linear|saits
    seq_len: int = 96
    pred_len: int = 96
    n_channels: int = 7
    time_feat_dim: int = 4
    # SAITS
    saits_d_model: int = 256
    saits_d_inner: int = 128
    saits_n_groups: int = 2
    saits_inner: int = 1
    saits_head: int = 4
    saits_dk: int = 64
    saits_dv: int = 64
    saits_dropout: float = 0.1
    saits_mit_rate: float = 0.2
    saits_epochs: int = 0  # 若 >0 单独预训练 SAITS；若 =0 则与预测器一起训练（端到端）
    # 缺失感知模型
    aware_q_dim: int = 64
    aware_num_heads: int = 4
    aware_kl_weight: float = 1e-3
    aware_impute_weight: float = 0.5
    # 训练超参
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1


class BasePipeline(nn.Module):
    """所有方法公用的接口。forward 返回 dict：
        forecast : (B, H, C) 预测值
        impute   : (B, L, C) 历史补值（缺失感知/补值方法填，简单填补也填）
        aux_loss : 标量，附加的辅助损失（如 KL、补值损失）；可选
    """

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg
        self.global_mean = nn.Parameter(torch.zeros(cfg.n_channels), requires_grad=False)

    def set_global_mean(self, mean_vec):
        # mean_vec 在标准化后空间应≈0；这里保留接口，标准化后的训练集均值取 0
        self.global_mean.data = torch.as_tensor(mean_vec, dtype=torch.float32)

    def forward(self, batch):
        raise NotImplementedError


# ----------------- baseline + 简单填补 -----------------
class TwoStagePipeline(BasePipeline):
    """两阶段：简单填补 -> 预测器。impute_strategy = none|zero|mean|forward|linear"""

    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        self.predictor = build_predictor(
            cfg.predictor, cfg.seq_len, cfg.pred_len, cfg.n_channels,
            time_feat_dim=cfg.time_feat_dim,
            d_model=cfg.d_model, n_heads=cfg.n_heads, e_layers=cfg.e_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
        ) if cfg.predictor == "iTransformer" else build_predictor(
            cfg.predictor, cfg.seq_len, cfg.pred_len, cfg.n_channels,
            **({"d_model": cfg.d_model, "n_heads": cfg.n_heads, "e_layers": cfg.e_layers,
                "d_ff": cfg.d_ff, "dropout": cfg.dropout} if cfg.predictor == "PatchTST" else {})
        )

    def _impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        st = self.cfg.impute_strategy
        if st in ("none", "no_impute"):
            return x * mask
        return apply_simple_imputer(st, x, mask, global_mean=self.global_mean)

    def forward(self, batch):
        x = batch["x_obs"]
        mask = batch["mask"]
        x_mark = batch.get("x_mark", None)
        x_imp = self._impute(x, mask)
        forecast = self.predictor(x_imp, x_mark, mask=mask)
        return {"forecast": forecast, "impute": x_imp, "aux_loss": torch.tensor(0.0, device=x.device)}


# ----------------- SAITS -> 预测器 -----------------
class SaitsPipeline(BasePipeline):
    """SAITS 强补值 -> 预测器。可端到端或两阶段（先训 SAITS 再训预测器）。"""

    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        self.saits = SAITS(
            d_time=cfg.seq_len, d_feature=cfg.n_channels,
            d_model=cfg.saits_d_model, d_inner=cfg.saits_d_inner,
            n_head=cfg.saits_head, d_k=cfg.saits_dk, d_v=cfg.saits_dv,
            n_groups=cfg.saits_n_groups, n_group_inner_layers=cfg.saits_inner,
            dropout=cfg.saits_dropout,
        )
        self.predictor = build_predictor(
            cfg.predictor, cfg.seq_len, cfg.pred_len, cfg.n_channels,
            time_feat_dim=cfg.time_feat_dim,
            d_model=cfg.d_model, n_heads=cfg.n_heads, e_layers=cfg.e_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
        ) if cfg.predictor == "iTransformer" else build_predictor(
            cfg.predictor, cfg.seq_len, cfg.pred_len, cfg.n_channels,
            **({"d_model": cfg.d_model, "n_heads": cfg.n_heads, "e_layers": cfg.e_layers,
                "d_ff": cfg.d_ff, "dropout": cfg.dropout} if cfg.predictor == "PatchTST" else {})
        )

    def forward(self, batch):
        x = batch["x_obs"]
        mask = batch["mask"]
        x_mark = batch.get("x_mark", None)
        out = self.saits(x, mask)
        imputed = out["imputed"]
        forecast = self.predictor(imputed, x_mark, mask=mask)
        # 补值损失（重建 + 可选 MIT），仅训练时使用
        rec = (masked_mae(out["Xt1"], x, mask)
               + masked_mae(out["Xt2"], x, mask)
               + masked_mae(out["Xt3"], x, mask)) / 3.0
        return {"forecast": forecast, "impute": imputed, "aux_loss": 0.1 * rec,
                "saits_intermediate": out}


# ----------------- 缺失感知方法 -----------------
class MissTSMPipeline(BasePipeline):
    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        self.model = MissTSMModel(
            cfg.seq_len, cfg.pred_len, cfg.n_channels,
            backbone=cfg.predictor,
            q_dim=cfg.aware_q_dim,
            num_heads=cfg.aware_num_heads,
            d_model=cfg.d_model, n_heads=cfg.n_heads, e_layers=cfg.e_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
            time_feat_dim=cfg.time_feat_dim,
        )

    def forward(self, batch):
        forecast = self.model(batch["x_obs"], batch.get("x_mark"), batch["mask"])
        return {"forecast": forecast, "impute": batch["x_obs"] * batch["mask"],
                "aux_loss": torch.tensor(0.0, device=forecast.device)}


class CribPipeline(BasePipeline):
    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        self.model = CRIB(
            cfg.seq_len, cfg.pred_len, cfg.n_channels,
            d_model=cfg.d_model, n_heads=cfg.n_heads, e_layers=cfg.e_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout, kl_weight=cfg.aware_kl_weight,
        )

    def forward(self, batch):
        forecast = self.model(batch["x_obs"], batch.get("x_mark"), batch["mask"])
        aux = self.model.auxiliary_loss()
        if not torch.is_tensor(aux):
            aux = torch.tensor(aux, device=forecast.device)
        return {"forecast": forecast, "impute": batch["x_obs"] * batch["mask"], "aux_loss": aux}


class CoifnetPipeline(BasePipeline):
    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        self.model = CoIFNet(
            cfg.seq_len, cfg.pred_len, cfg.n_channels,
            hidden=cfg.d_model, n_layers=cfg.e_layers, dropout=cfg.dropout,
            impute_weight=cfg.aware_impute_weight,
        )

    def forward(self, batch):
        out = self.model(batch["x_obs"], batch.get("x_mark"), batch["mask"])
        # 把补值loss放进 aux_loss
        x_obs = batch["x_obs"]
        mask = batch["mask"]
        # 用 x_obs 当作真值（在标准化空间），仅在 mask=1 处比较
        rec = ((out["impute"] - x_obs).abs() * mask).sum() / mask.sum().clamp(min=1.0)
        return {"forecast": out["forecast"], "impute": out["impute"],
                "aux_loss": self.cfg.aware_impute_weight * rec}


def build_pipeline(cfg: PipelineConfig) -> BasePipeline:
    method = cfg.method.lower()
    if method in ("baseline", "no_missing", "simple"):
        return TwoStagePipeline(cfg)
    if method == "saits":
        return SaitsPipeline(cfg)
    if method == "misstsm":
        return MissTSMPipeline(cfg)
    if method == "crib":
        return CribPipeline(cfg)
    if method == "coifnet":
        return CoifnetPipeline(cfg)
    raise ValueError(f"unknown method {method}")

"""MissTSM 紧凑实现：在 backbone 之前加入一个"缺失特征感知"的处理层。

核心思想：将每个 (t, c) 标量先嵌入到 q_dim 维向量，再用 mask 控制交叉注意力对缺失变量的可见性，
最终在每个时间步聚合得到一个跨变量表征，再投影回原维度，供下游预测 backbone 使用。

参考：A. Neog et al., arXiv:2502.15785, 2025.
原始代码：external/MissTSM/forecasting/misstsm_itransformer/layers/Transformer_EncDec.py
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .dlinear import DLinear
from .patchtst import PatchTST
from .itransformer import iTransformer


class _LinearEmbed(nn.Module):
    def __init__(self, q_dim: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, q_dim), nn.LayerNorm(q_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, L, C, q_dim)
        return self.embed(x.unsqueeze(-1))


class _PE2D(nn.Module):
    """二维正弦位置编码（沿 L 和 C）。"""

    def __init__(self, q_dim: int):
        super().__init__()
        assert q_dim % 4 == 0, "q_dim must be divisible by 4 for 2D PE"
        self.q_dim = q_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, D)
        B, L, C, D = x.shape
        half = D // 2
        # 沿 L
        pe_l = torch.zeros(L, half, device=x.device)
        pos_l = torch.arange(L, device=x.device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, half, 2, device=x.device).float() * -(math.log(10000.0) / half))
        pe_l[:, 0::2] = torch.sin(pos_l * div)
        pe_l[:, 1::2] = torch.cos(pos_l * div)
        # 沿 C
        pe_c = torch.zeros(C, half, device=x.device)
        pos_c = torch.arange(C, device=x.device).float().unsqueeze(1)
        div2 = torch.exp(torch.arange(0, half, 2, device=x.device).float() * -(math.log(10000.0) / half))
        pe_c[:, 0::2] = torch.sin(pos_c * div2)
        pe_c[:, 1::2] = torch.cos(pos_c * div2)
        # 拼成 (L, C, D)
        pe = torch.cat([pe_l.unsqueeze(1).expand(L, C, half), pe_c.unsqueeze(0).expand(L, C, half)], dim=-1)
        return pe.unsqueeze(0).expand(B, L, C, D)


class MissTSMLayer(nn.Module):
    """MissTSM 核心：跨变量交叉注意力 + 缺失感知 padding mask。

    输入: x (B, L, C), mask (B, L, C) — 1=有观测，0=缺失
    输出: y (B, L, C_out)

    variant:
      - "full": 原始单 query
      - "cond_q": 时间步条件化 query (C1)
      - "multi_q": 多 query K=8 + mean pooling (C2)
      - "grouped_q": 将 C 通道分为 G 组，每组独立 query + cross-attention；
                     若传入 group_order，先按该顺序重排通道再连续切片（方案 A：相关性预分组；
                     方案 D 传入观测数据相关性算出的 group_order 时代码路径完全相同）
      - "grouped_q_soft": 不做硬切分，每组学一个通道路由权重（softmax），软门控后各组独立
                          query + cross-attention（方案 B：可学习软路由）
      - "grouped_q_fuse": 硬切分路径（同 grouped_q）与软路由路径（同 grouped_q_soft）并行
                          计算，各自独立投影后相加融合（方案 E：对应 GinAR/IBN 的
                          A_pre·X·W1 + A_adap·X·W2 预定义图+自适应图融合公式）；
                          gate_mode 可进一步控制 0711 的可靠性门控融合。
    """

    def __init__(self, n_channels: int, q_dim: int = 64, num_heads: int = 4,
                 out_dim: int = None, variant: str = "full", n_queries: int = 8,
                 n_groups: int = 4, group_order: list[int] | None = None,
                 gate_mode: str = "none"):
        super().__init__()
        self.q_dim = q_dim
        self.n_channels = n_channels
        self.out_dim = out_dim if out_dim else n_channels
        self.variant = variant
        self.n_groups = n_groups
        self.gate_mode = gate_mode
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}

        if variant == "multi_q":
            self.var_query = nn.Parameter(torch.zeros(1, n_queries, q_dim))
        elif variant in ("grouped_q", "grouped_q_soft", "grouped_q_fuse"):
            self.var_query = nn.Parameter(torch.zeros(1, n_groups, q_dim))
        else:
            self.var_query = nn.Parameter(torch.zeros(1, 1, q_dim))

        if variant in ("grouped_q", "grouped_q_fuse") and group_order is not None:
            assert len(group_order) == n_channels
            self.register_buffer("group_order", torch.tensor(group_order, dtype=torch.long))
        else:
            self.group_order = None

        if variant in ("grouped_q_soft", "grouped_q_fuse"):
            self.channel_embed = nn.Parameter(torch.randn(n_channels, q_dim) * 0.02)
            self.group_proto = nn.Parameter(torch.randn(n_groups, q_dim) * 0.02)

        self.mask_embed = _LinearEmbed(q_dim)
        self.pos_embed = _PE2D(q_dim)
        if variant in ("grouped_q", "grouped_q_soft"):
            self.mhca_groups = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
                for _ in range(n_groups)
            ])
            self.group_proj = nn.Linear(n_groups * q_dim, q_dim)
        elif variant == "grouped_q_fuse":
            self.mhca_groups_a = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
                for _ in range(n_groups)
            ])
            self.mhca_groups_b = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
                for _ in range(n_groups)
            ])
            if gate_mode == "none":
                self.group_proj_a = nn.Linear(n_groups * q_dim, q_dim)
                self.group_proj_b = nn.Linear(n_groups * q_dim, q_dim)
            else:
                self.group_proj_mix = nn.Linear(n_groups * q_dim, q_dim)
                if gate_mode == "scalar":
                    self.path_gate = nn.Parameter(torch.zeros(()))
                elif gate_mode == "group":
                    self.group_gate = nn.Parameter(torch.zeros(n_groups))
                elif gate_mode == "mask":
                    self.gate_mlp = nn.Sequential(
                        nn.Linear(3, 16),
                        nn.GELU(),
                        nn.Linear(16, 1),
                    )
                else:
                    raise ValueError(f"unknown gate_mode: {gate_mode}")
        else:
            self.mhca = nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(q_dim)
        self.projection = nn.Linear(q_dim, self.out_dim)
        nn.init.trunc_normal_(self.var_query, std=0.02)

        if variant == "cond_q":
            pe_dim = q_dim // 2
            self.time_proj = nn.Linear(pe_dim, q_dim)

    def _revin(self, x: torch.Tensor, mask: torch.Tensor):
        m_sum = mask.sum(dim=1).clamp(min=1.0)
        means = (x * mask).sum(dim=1) / m_sum
        means = means.unsqueeze(1)
        x = x - means
        var = ((x * mask) ** 2).sum(dim=1) / m_sum + 1e-5
        std = torch.sqrt(var).unsqueeze(1)
        x = x / std
        return x, means, std

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_n, means, std = self._revin(x * mask, mask)
        emb = self.mask_embed(x_n)  # (B, L, C, q_dim)
        emb = emb + self.pos_embed(emb)
        B, L, C, D = emb.shape
        emb = emb.reshape(B * L, C, D)

        # Build query based on variant
        if self.variant == "cond_q":
            half = D // 2
            pos = torch.arange(L, device=x.device).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, half, 2, device=x.device).float() * -(math.log(10000.0) / half))
            pe_time = torch.zeros(L, half, device=x.device)
            pe_time[:, 0::2] = torch.sin(pos * div)
            pe_time[:, 1::2] = torch.cos(pos * div)
            time_bias = self.time_proj(pe_time)  # (L, q_dim)
            q = self.var_query + time_bias.unsqueeze(0)  # (1, L, q_dim)
            q = q.expand(B, L, D).reshape(B * L, 1, D)
        elif self.variant == "multi_q":
            K = self.var_query.size(1)
            q = self.var_query.expand(B * L, K, D)
        elif self.variant not in ("grouped_q", "grouped_q_soft", "grouped_q_fuse"):
            q = self.var_query.expand(B * L, 1, D)

        pad_mask = (mask.reshape(B * L, C) < 0.5)
        all_missing = pad_mask.all(dim=1)
        if all_missing.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_missing, 0] = False

        if self.variant == "grouped_q":
            if self.group_order is not None:
                emb = emb.index_select(1, self.group_order)
                pad_mask = pad_mask.index_select(1, self.group_order)
            G = self.n_groups
            group_size = (C + G - 1) // G
            group_outs = []
            for g in range(G):
                c_start = g * group_size
                c_end = min((g + 1) * group_size, C)
                emb_g = emb[:, c_start:c_end, :]
                pad_g = pad_mask[:, c_start:c_end]
                all_miss_g = pad_g.all(dim=1)
                if all_miss_g.any():
                    pad_g = pad_g.clone()
                    pad_g[all_miss_g, 0] = False
                q_g = self.var_query[:, g:g+1, :].expand(B * L, 1, D)
                ao, _ = self.mhca_groups[g](q_g, emb_g, emb_g, key_padding_mask=pad_g)
                group_outs.append(ao.squeeze(1))
            attn_out = self.group_proj(torch.cat(group_outs, dim=-1)).unsqueeze(1)
        elif self.variant == "grouped_q_soft":
            G = self.n_groups
            route_weights = self.route_weights()  # (C, G)
            group_outs = []
            for g in range(G):
                w_g = route_weights[:, g].view(1, C, 1)
                emb_g = emb * w_g
                q_g = self.var_query[:, g:g+1, :].expand(B * L, 1, D)
                ao, _ = self.mhca_groups[g](q_g, emb_g, emb_g, key_padding_mask=pad_mask)
                group_outs.append(ao.squeeze(1))
            attn_out = self.group_proj(torch.cat(group_outs, dim=-1)).unsqueeze(1)
            if self.record_diagnostics:
                self._store_route_diagnostics(route_weights)
        elif self.variant == "grouped_q_fuse":
            G = self.n_groups
            # 路径 A（预定义图，硬切分）：同 grouped_q
            emb_a = emb
            pad_a = pad_mask
            if self.group_order is not None:
                emb_a = emb_a.index_select(1, self.group_order)
                pad_a = pad_a.index_select(1, self.group_order)
            group_size = (C + G - 1) // G
            group_outs_a = []
            for g in range(G):
                c_start = g * group_size
                c_end = min((g + 1) * group_size, C)
                emb_g = emb_a[:, c_start:c_end, :]
                pad_g = pad_a[:, c_start:c_end]
                all_miss_g = pad_g.all(dim=1)
                if all_miss_g.any():
                    pad_g = pad_g.clone()
                    pad_g[all_miss_g, 0] = False
                q_g = self.var_query[:, g:g+1, :].expand(B * L, 1, D)
                ao, _ = self.mhca_groups_a[g](q_g, emb_g, emb_g, key_padding_mask=pad_g)
                group_outs_a.append(ao.squeeze(1))
            # 路径 B（自适应图，软路由）：同 grouped_q_soft，attends over 全部通道
            route_weights = self.route_weights()  # (C, G)
            group_outs_b = []
            for g in range(G):
                w_g = route_weights[:, g].view(1, C, 1)
                emb_g = emb * w_g
                q_g = self.var_query[:, g:g+1, :].expand(B * L, 1, D)
                ao, _ = self.mhca_groups_b[g](q_g, emb_g, emb_g, key_padding_mask=pad_mask)
                group_outs_b.append(ao.squeeze(1))
            # 融合：对应 GinAR/IBN 的 A_pre·X·W1 + A_adap·X·W2
            if self.gate_mode == "none":
                attn_out_a = self.group_proj_a(torch.cat(group_outs_a, dim=-1))
                attn_out_b = self.group_proj_b(torch.cat(group_outs_b, dim=-1))
                attn_out = (attn_out_a + attn_out_b).unsqueeze(1)
                if self.record_diagnostics:
                    self._store_fuse_diagnostics(attn_out_a, attn_out_b, route_weights)
            else:
                mixed_groups = []
                gate_values = []
                if self.gate_mode == "scalar":
                    gate = torch.sigmoid(self.path_gate).view(1, 1)
                    for out_a_g, out_b_g in zip(group_outs_a, group_outs_b):
                        mixed_groups.append(gate * out_a_g + (1.0 - gate) * out_b_g)
                        gate_values.append(gate.expand(out_a_g.size(0), 1))
                elif self.gate_mode == "group":
                    gates = torch.sigmoid(self.group_gate)
                    for g, (out_a_g, out_b_g) in enumerate(zip(group_outs_a, group_outs_b)):
                        gate = gates[g].view(1, 1)
                        mixed_groups.append(gate * out_a_g + (1.0 - gate) * out_b_g)
                        gate_values.append(gate.expand(out_a_g.size(0), 1))
                elif self.gate_mode == "mask":
                    global_obs = 1.0 - pad_mask.float().mean(dim=1, keepdim=True)
                    for g, (out_a_g, out_b_g) in enumerate(zip(group_outs_a, group_outs_b)):
                        c_start = g * group_size
                        c_end = min((g + 1) * group_size, C)
                        pad_g = pad_a[:, c_start:c_end]
                        group_obs = 1.0 - pad_g.float().mean(dim=1, keepdim=True)
                        all_missing_g = pad_g.all(dim=1).float().unsqueeze(1)
                        gate_input = torch.cat([global_obs, group_obs, all_missing_g], dim=-1)
                        gate = torch.sigmoid(self.gate_mlp(gate_input))
                        mixed_groups.append(gate * out_a_g + (1.0 - gate) * out_b_g)
                        gate_values.append(gate)
                else:
                    raise ValueError(f"unknown gate_mode: {self.gate_mode}")
                raw_a = torch.cat(group_outs_a, dim=-1)
                raw_b = torch.cat(group_outs_b, dim=-1)
                attn_out = self.group_proj_mix(torch.cat(mixed_groups, dim=-1)).unsqueeze(1)
                if self.record_diagnostics:
                    gates = torch.cat(gate_values, dim=1)
                    self._store_fuse_diagnostics(raw_a, raw_b, route_weights, gates)
        else:
            attn_out, _ = self.mhca(q, emb, emb, key_padding_mask=pad_mask)

        if self.variant == "multi_q":
            attn_out = attn_out.mean(dim=1, keepdim=True)  # (B*L, 1, D)

        out = attn_out.reshape(B, L, D)
        out = self.layernorm(out)
        out = self.projection(out)  # (B, L, out_dim)
        if self.out_dim == self.n_channels:
            out = out * std + means
        return out

    def route_weights(self) -> torch.Tensor:
        """仅 grouped_q_soft 使用：(C, G) 通道->组路由权重。"""
        logits = self.channel_embed @ self.group_proto.T
        return torch.softmax(logits, dim=1)

    def route_entropy(self) -> torch.Tensor:
        """路由权重的平均熵（越小越"尖锐"，log(G) 为完全均匀退化）。仅 grouped_q_soft 有意义。"""
        w = self.route_weights().clamp(min=1e-8)
        return -(w * w.log()).sum(dim=1).mean()

    def enable_diagnostics(self, enabled: bool = True):
        self.record_diagnostics = enabled
        self.last_diagnostics = {}

    def _store_fuse_diagnostics(
        self,
        out_a: torch.Tensor,
        out_b: torch.Tensor,
        route_weights: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            diag = {
                "out_a_norm": out_a.norm(dim=-1).mean().detach().cpu(),
                "out_b_norm": out_b.norm(dim=-1).mean().detach().cpu(),
                "out_ab_cosine": nn.functional.cosine_similarity(out_a, out_b, dim=-1).mean().detach().cpu(),
            }
            if hasattr(self, "group_proj_a"):
                diag["proj_a_weight_norm"] = self.group_proj_a.weight.norm().detach().cpu()
            if hasattr(self, "group_proj_b"):
                diag["proj_b_weight_norm"] = self.group_proj_b.weight.norm().detach().cpu()
            if route_weights is not None:
                w = route_weights.clamp(min=1e-8)
                entropy = -(w * w.log()).sum(dim=1)
                diag["route_entropy"] = entropy.mean().detach().cpu()
                diag["route_effective_groups"] = (1.0 / (w.pow(2).sum(dim=1).clamp(min=1e-8))).mean().detach().cpu()
                diag["route_top1_counts"] = torch.bincount(
                    route_weights.argmax(dim=1), minlength=self.n_groups
                ).detach().cpu()
            if gates is not None:
                diag["gate_mean"] = gates.mean().detach().cpu()
                diag["gate_std"] = gates.std(unbiased=False).detach().cpu()
                diag["gate_group_mean"] = gates.mean(dim=0).detach().cpu()
            self.last_diagnostics = diag

    def _store_route_diagnostics(self, route_weights: torch.Tensor):
        with torch.no_grad():
            w = route_weights.clamp(min=1e-8)
            entropy = -(w * w.log()).sum(dim=1)
            self.last_diagnostics = {
                "route_entropy": entropy.mean().detach().cpu(),
                "route_effective_groups": (1.0 / (w.pow(2).sum(dim=1).clamp(min=1e-8))).mean().detach().cpu(),
                "route_top1_counts": torch.bincount(
                    route_weights.argmax(dim=1), minlength=self.n_groups
                ).detach().cpu(),
            }


class MissTSMModel(nn.Module):
    """MissTSM 完整预测模型：缺失感知层 + 选定 backbone。"""

    name = "MissTSM"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        backbone: str = "iTransformer",
        q_dim: int = 64,
        num_heads: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        time_feat_dim: int = 0,
        patch_len: int = 16,
        stride: int = 8,
        skip: bool = True,
        variant: str = "full",
        group_order: list[int] | None = None,
        gate_mode: str = "none",
    ):
        super().__init__()
        self.skip = skip
        self.variant = variant

        mtsm_variant = variant if variant in ("cond_q", "multi_q") else "full"
        n_groups = 4
        if variant.startswith("grouped_q"):
            base = variant
            suffix = None
            for s in ("_fuseobs_mgate", "_fuseobs_ggate", "_fuseobs_sgate",
                      "_fuse_mgate", "_fuse_ggate", "_fuse_sgate",
                      "_corrobs", "_corr", "_fuseobs", "_fuse", "_soft"):
                if base.endswith(s):
                    suffix = s[1:]
                    base = base[: -len(s)]
                    break
            parts = base.split("_q")
            if len(parts) == 2 and parts[1].isdigit():
                n_groups = int(parts[1])
            if suffix == "soft":
                mtsm_variant = "grouped_q_soft"
            elif suffix in ("fuse", "fuseobs", "fuse_sgate", "fuse_ggate", "fuse_mgate",
                            "fuseobs_sgate", "fuseobs_ggate", "fuseobs_mgate"):
                mtsm_variant = "grouped_q_fuse"
                if suffix.endswith("sgate"):
                    gate_mode = "scalar"
                elif suffix.endswith("ggate"):
                    gate_mode = "group"
                elif suffix.endswith("mgate"):
                    gate_mode = "mask"
            else:
                mtsm_variant = "grouped_q"
        self.mtsm = MissTSMLayer(n_channels, q_dim=q_dim, num_heads=num_heads,
                                 out_dim=n_channels, variant=mtsm_variant,
                                 n_groups=n_groups,
                                 group_order=group_order if mtsm_variant in ("grouped_q", "grouped_q_fuse") else None,
                                 gate_mode=gate_mode)

        if variant == "soft_skip":
            self.skip_gate = nn.Linear(n_channels, n_channels)
        backbone = backbone.lower()
        if backbone == "itransformer":
            self.backbone = iTransformer(
                seq_len, pred_len, n_channels,
                d_model=d_model, n_heads=n_heads, e_layers=e_layers,
                d_ff=d_ff, dropout=dropout, time_feat_dim=time_feat_dim, use_norm=False,
            )
        elif backbone == "patchtst":
            self.backbone = PatchTST(
                seq_len, pred_len, n_channels,
                patch_len=patch_len, stride=stride, d_model=d_model,
                n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
                use_revin=False,
            )
        elif backbone == "dlinear":
            self.backbone = DLinear(seq_len, pred_len, n_channels)
        else:
            raise ValueError(backbone)

    def forward(self, x: torch.Tensor, x_mark=None, mask=None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        feat = self.mtsm(x, mask)  # (B, L, C)
        if self.variant == "soft_skip":
            alpha = torch.sigmoid(self.skip_gate(feat))
            feat = mask * (alpha * x + (1 - alpha) * feat) + (1 - mask) * feat
        elif self.skip:
            feat = mask * x + (1 - mask) * feat
        return self.backbone(feat, x_mark)

    def route_entropy(self) -> torch.Tensor:
        return self.mtsm.route_entropy()

    def enable_diagnostics(self, enabled: bool = True):
        self.mtsm.enable_diagnostics(enabled)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return self.mtsm.last_diagnostics

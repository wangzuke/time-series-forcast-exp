"""P-GUTS-style and HD-PGUTS-style forecasters for the 0721 traffic runs.

The implementation follows the experiment plan's forecasting-as-imputation
interface: the model receives an observed history followed by a masked future
segment, then predicts the future values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_avg_pool_time(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Average-pool the time axis while preserving shape.

    x: (B, T, N, D)
    """
    if kernel_size <= 1:
        return x
    bsz, steps, nodes, width = x.shape
    y = x.permute(0, 2, 3, 1).reshape(bsz * nodes, width, steps)
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    y = F.pad(y, (left, right), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel_size, stride=1)
    return y.reshape(bsz, nodes, width, steps).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class PGUTSConfig:
    history: int = 24
    horizon: int = 24
    num_nodes: int = 207
    d_model: int = 32
    pooling_factors: tuple[int, ...] = (3,)
    graph_scale: int = 4
    variant: str = "pguts"
    dropout: float = 0.1


class PGUTSHDPGUTSForecaster(nn.Module):
    """Compact graph-temporal forecaster used by the 0721 experiment scripts.

    Variants:
      - pguts: temporal multi-pooling plus one full-resolution graph branch.
      - no_graph_coarsening: temporal/full graph branches with adaptive fusion,
        but without the coarse graph branch.
      - no_adaptive_fusion: HD temporal/full/coarse graph branches with fixed
        fusion.
      - full: HD temporal/full/coarse graph branches with adaptive fusion.
    """

    def __init__(
        self,
        cfg: PGUTSConfig,
        adjacency: torch.Tensor | None = None,
        coarse_assignment: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.total_steps = cfg.history + cfg.horizon
        self.input_proj = nn.Linear(2, cfg.d_model)
        self.temporal_proj = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in cfg.pooling_factors]
        )
        self.graph_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.coarse_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.uses_coarse_branch = cfg.variant in {"no_adaptive_fusion", "full"}
        self.uses_adaptive_fusion = cfg.variant in {"no_graph_coarsening", "full"}
        n_branches = 1 + len(cfg.pooling_factors) + 1 + int(self.uses_coarse_branch)
        self.fixed_fusion = nn.Linear(n_branches * cfg.d_model, cfg.d_model)
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, n_branches),
        )
        self.time_readout = nn.Linear(self.total_steps, cfg.horizon)
        self.value_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )

        if adjacency is None:
            adjacency = torch.eye(cfg.num_nodes, dtype=torch.float32)
        if coarse_assignment is None:
            coarse_assignment = torch.eye(cfg.num_nodes, dtype=torch.float32)
        self.register_buffer("adjacency", adjacency.float())
        self.register_buffer("coarse_assignment", coarse_assignment.float())
        self.last_scale_weights: torch.Tensor | None = None

    def architecture_signature(self) -> str:
        branches = ["identity"]
        branches += [f"temporal_pool_{factor}" for factor in self.cfg.pooling_factors]
        branches.append("full_graph")
        if self.uses_coarse_branch:
            branches.append("coarse_graph")
        fusion = "adaptive_gate" if self.uses_adaptive_fusion else "fixed_linear"
        return f"variant={self.cfg.variant};branches={'+'.join(branches)};fusion={fusion}"

    def _graph_smooth(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,btmd->btnd", self.adjacency, x)

    def _coarse_graph_smooth(self, x: torch.Tensor) -> torch.Tensor:
        assign = self.coarse_assignment
        counts = assign.sum(dim=0).clamp_min(1.0)
        coarse = torch.einsum("nk,btnd->btkd", assign, x) / counts.view(1, 1, -1, 1)
        coarse_adj = assign.t().matmul(self.adjacency).matmul(assign)
        coarse_adj = coarse_adj / coarse_adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        coarse = torch.einsum("kl,btld->btkd", coarse_adj, coarse)
        return torch.einsum("nk,btkd->btnd", assign, coarse)

    def _branches(self, h: torch.Tensor) -> list[torch.Tensor]:
        branches = [h]
        for factor, proj in zip(self.cfg.pooling_factors, self.temporal_proj):
            branches.append(torch.relu(proj(_same_avg_pool_time(h, factor))))
        branches.append(torch.relu(self.graph_proj(self._graph_smooth(h))))
        if self.uses_coarse_branch:
            branches.append(torch.relu(self.coarse_proj(self._coarse_graph_smooth(h))))
        return branches

    def _fuse(self, branches: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(branches, dim=-2)  # (B, T, N, K, D)
        if self.uses_adaptive_fusion:
            temporal_summary = stacked.mean(dim=(1, 2, 3))
            graph_summary = branches[-1].mean(dim=(1, 2))
            logits = self.gate(torch.cat([temporal_summary, graph_summary], dim=-1))
            weights = torch.softmax(logits, dim=-1)
            self.last_scale_weights = weights.detach()
            return (stacked * weights[:, None, None, :, None]).sum(dim=-2)

        self.last_scale_weights = None
        return torch.relu(self.fixed_fusion(torch.cat(branches, dim=-1)))

    def forward(self, x_all: torch.Tensor, mask_all: torch.Tensor) -> torch.Tensor:
        # x_all/mask_all: (B, T_in + T_out, N, 1)
        h = torch.relu(self.input_proj(torch.cat([x_all, mask_all], dim=-1)))
        h = self.dropout(self._fuse(self._branches(h)))
        h = h.permute(0, 2, 3, 1)  # (B, N, D, T)
        h = self.time_readout(h).permute(0, 3, 1, 2)  # (B, H, N, D)
        residual = self.value_head(h)
        base = x_all[:, self.cfg.history - 1 : self.cfg.history, :, :]
        return residual + base.expand(-1, self.cfg.horizon, -1, -1)


def contiguous_coarse_assignment(num_nodes: int, graph_scale: int) -> torch.Tensor:
    """Build a deterministic node-to-coarse-node assignment matrix."""
    graph_scale = max(1, int(graph_scale))
    n_coarse = max(1, (num_nodes + graph_scale - 1) // graph_scale)
    assignment = torch.zeros(num_nodes, n_coarse, dtype=torch.float32)
    for node in range(num_nodes):
        assignment[node, min(n_coarse - 1, node // graph_scale)] = 1.0
    return assignment


def parse_pooling_factors(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part) for part in value.replace(";", ",").split(",") if part.strip())
    return tuple(int(v) for v in value)

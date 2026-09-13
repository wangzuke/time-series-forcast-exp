"""HD-TTS-based reliable spatiotemporal forecaster.

This module keeps the HD-TTS design skeleton used in the official code:
mask-aware input encoding, dilated recurrent temporal hierarchy, spatial
coarsening with pool/lift, and an attention readout over multiscale states.

The additions for our idea are deliberately isolated:
- observed-only temporal relation graph from the current masked history;
- GinAR-style adaptive graph from node embeddings and current states;
- reliability-aware fusion of physical, temporal, and adaptive graphs;
- mask/region-coverage-conditioned scale readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(eps)


def _make_self_loop_adj(n_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.eye(n_nodes, dtype=torch.float32, device=device)


def observed_temporal_graph(
    x: torch.Tensor,
    mask: torch.Tensor,
    min_overlap: float = 2.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch-wise graph from observed-only channel correlations.

    Args:
        x: ``(B, L, N)`` zero-filled observed values.
        mask: ``(B, L, N)`` binary observation mask.

    Returns:
        ``(adj, reliability)`` with shape ``(B, N, N)``.
    """
    bsz, steps, n_nodes = x.shape
    obs = mask.float()
    denom = obs.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * obs).sum(dim=1, keepdim=True) / denom
    centered = (x - mean) * obs

    overlap = torch.bmm(obs.transpose(1, 2), obs)
    cov = torch.bmm(centered.transpose(1, 2), centered) / overlap.clamp_min(1.0)
    var = (centered.square().sum(dim=1) / obs.sum(dim=1).clamp_min(1.0)).clamp_min(eps)
    corr = cov / torch.sqrt(var[:, :, None] * var[:, None, :]).clamp_min(eps)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    reliability = (overlap / float(max(1, steps))).clamp(0.0, 1.0)
    corr = corr.abs() * reliability
    corr = torch.where(overlap >= min_overlap, corr, torch.zeros_like(corr))
    eye = torch.eye(n_nodes, dtype=corr.dtype, device=corr.device).expand(bsz, -1, -1)
    adj = _row_normalize(corr + eye)
    return adj, reliability


def dense_greedy_assignment(adj: torch.Tensor, max_cluster_size: int = 8) -> torch.Tensor:
    """Create a deterministic node-to-region assignment from a dense graph.

    This is a lightweight dense analogue of HD-TTS' k-MIS pooling for the local
    prototype, used when PyG/TSL are unavailable in the active environment.
    """
    if adj.ndim == 3:
        adj = adj.mean(dim=0)
    n_nodes = int(adj.size(0))
    max_cluster_size = max(1, int(max_cluster_size))
    if n_nodes <= max_cluster_size:
        return torch.ones(n_nodes, 1, dtype=torch.float32, device=adj.device)

    scores = adj.sum(dim=-1)
    order = torch.argsort(scores, descending=True)
    assigned = torch.zeros(n_nodes, dtype=torch.bool, device=adj.device)
    clusters: list[list[int]] = []
    for idx in order.tolist():
        if assigned[idx]:
            continue
        weights = adj[idx].clone()
        weights[assigned] = -1.0
        members = torch.argsort(weights, descending=True)[:max_cluster_size]
        members = [int(m) for m in members.tolist() if not bool(assigned[m])]
        if idx not in members:
            members.insert(0, idx)
        for member in members:
            assigned[member] = True
        clusters.append(members)

    assignment = torch.zeros(n_nodes, len(clusters), dtype=torch.float32, device=adj.device)
    for cid, members in enumerate(clusters):
        assignment[members, cid] = 1.0
    empty = assignment.sum(dim=1) == 0
    if empty.any():
        nearest = torch.argmax(adj[empty][:, assignment.sum(dim=0) > 0], dim=1)
        assignment[empty] = F.one_hot(nearest, num_classes=assignment.size(1)).float()
    return assignment


class DenseGraphConv(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(hidden_size, hidden_size)
        self.lin_neigh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.ndim == 2:
            neigh = torch.einsum("nm,bmd->bnd", adj, x)
        else:
            neigh = torch.bmm(adj, x)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        return self.norm(x + self.dropout(F.gelu(out)))


class DilatedTemporalHierarchy(nn.Module):
    """HD-TTS-style dilated recurrent temporal pyramid."""

    def __init__(self, hidden_size: int, n_layers: int, dilation: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dilation = int(dilation)
        self.layers = nn.ModuleList(
            [
                nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, D)
        bsz, steps, n_nodes, hidden = x.shape
        seq = x.permute(1, 0, 2, 3).reshape(steps, bsz * n_nodes, hidden)
        states = []
        for gru in self.layers:
            seq, _ = gru(seq)
            states.append(seq[-1].reshape(bsz, n_nodes, hidden))
            if seq.size(0) > 1:
                offset = (seq.size(0) - 1) % self.dilation
                seq = seq[offset:: self.dilation]
            seq = self.dropout(seq)
        return torch.stack(states, dim=1)  # (B, T_scale, N, D)


class ReliabilityReadout(nn.Module):
    def __init__(self, hidden_size: int, horizon: int, n_scales: int, dropout: float = 0.0):
        super().__init__()
        self.state_score = nn.Linear(hidden_size, 1)
        self.reliability_score = nn.Linear(3, n_scales)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, horizon)
        self.last_alpha: torch.Tensor | None = None

    def forward(
        self,
        states: torch.Tensor,
        node_obs_rate: torch.Tensor,
        node_edge_reliability: torch.Tensor,
        node_region_coverage: torch.Tensor,
    ) -> torch.Tensor:
        # states: (B, S, N, D)
        rel = torch.stack([node_obs_rate, node_edge_reliability, node_region_coverage], dim=-1)
        rel_bias = self.reliability_score(rel).transpose(1, 2).unsqueeze(-1)
        scores = self.state_score(states) + rel_bias
        alpha = torch.softmax(scores, dim=1)
        self.last_alpha = alpha.detach()
        fused = (states * alpha).sum(dim=1)
        out = self.proj(self.dropout(fused)).transpose(1, 2)
        return out


@dataclass(frozen=True)
class ReliableHDTTSConfig:
    seq_len: int
    pred_len: int
    n_channels: int
    hidden_size: int = 64
    temporal_layers: int = 2
    pooling_layers: int = 2
    dilation: int = 2
    max_cluster_size: int = 8
    dropout: float = 0.1
    physical_graph_weight: float = 1.0
    temporal_graph_weight: float = 1.0
    adaptive_graph_weight: float = 0.5


class ReliableHDTTSForecaster(nn.Module):
    """HD-TTS base with GinAR-style graph adaptation and reliability gating."""

    name = "ReliableHDTTS"

    def __init__(
        self,
        cfg: ReliableHDTTSConfig,
        static_adj: torch.Tensor | None = None,
        assignment: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        n_nodes = cfg.n_channels
        self.value_encoder = nn.Linear(2, cfg.hidden_size)
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, cfg.hidden_size) * 0.02)
        self.temporal = DilatedTemporalHierarchy(
            hidden_size=cfg.hidden_size,
            n_layers=cfg.temporal_layers,
            dilation=cfg.dilation,
            dropout=cfg.dropout,
        )
        self.fine_conv = DenseGraphConv(cfg.hidden_size, cfg.dropout)
        self.coarse_convs = nn.ModuleList(
            [DenseGraphConv(cfg.hidden_size, cfg.dropout) for _ in range(cfg.pooling_layers)]
        )
        n_scales = cfg.temporal_layers * (1 + cfg.pooling_layers)
        self.readout = ReliabilityReadout(cfg.hidden_size, cfg.pred_len, n_scales, cfg.dropout)
        self.adaptive_proj = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)

        if static_adj is None:
            static_adj = torch.eye(n_nodes, dtype=torch.float32)
        if assignment is None:
            assignment = dense_greedy_assignment(_row_normalize(static_adj.float()), cfg.max_cluster_size)
        self.register_buffer("static_adj", _row_normalize(static_adj.float()))
        self.register_buffer("assignment", assignment.float())
        self.last_graph_mix: dict[str, torch.Tensor] = {}

    def _adaptive_graph(self, state: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = state.shape
        emb = self.node_embedding.unsqueeze(0).expand(bsz, -1, -1)
        z = self.adaptive_proj(torch.cat([state, emb], dim=-1))
        adj = F.softmax(F.relu(torch.bmm(z, z.transpose(1, 2))), dim=-1)
        eye = torch.eye(n_nodes, dtype=adj.dtype, device=adj.device).expand(bsz, -1, -1)
        return _row_normalize(adj + eye)

    def _fused_fine_graph(self, x: torch.Tensor, mask: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temp_adj, reliability = observed_temporal_graph(x, mask)
        adapt_adj = self._adaptive_graph(state)
        static_adj = self.static_adj.to(x.device)
        graph = (
            self.cfg.physical_graph_weight * static_adj.unsqueeze(0)
            + self.cfg.temporal_graph_weight * temp_adj
            + self.cfg.adaptive_graph_weight * adapt_adj
        )
        graph = _row_normalize(graph)
        self.last_graph_mix = {
            "temporal_adj": temp_adj.detach(),
            "adaptive_adj": adapt_adj.detach(),
            "edge_reliability": reliability.detach(),
        }
        return graph, reliability

    def _coarse_state(
        self,
        state: torch.Tensor,
        fine_adj: torch.Tensor,
        conv: DenseGraphConv,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assign = self.assignment.to(state.device)
        counts = assign.sum(dim=0).clamp_min(1.0)
        coarse = torch.einsum("nk,bnd->bkd", assign, state) / counts.view(1, -1, 1)
        coarse_adj = torch.einsum("nk,bnm,ml->bkl", assign, fine_adj, assign)
        coarse_adj = _row_normalize(coarse_adj)
        coarse = conv(coarse, coarse_adj)
        lifted = torch.einsum("nk,bkd->bnd", assign, coarse)
        cluster_sizes = torch.matmul(assign, counts.view(-1, 1)).squeeze(-1).clamp_min(1.0)
        return lifted, cluster_sizes

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x/mask: (B, L, N)
        bsz, _, n_nodes = x.shape
        h = self.value_encoder(torch.stack([x, mask.float()], dim=-1))
        h = h + self.node_embedding.view(1, 1, n_nodes, -1)
        temporal_states = self.temporal(h)

        all_states = []
        edge_reliabilities = []
        region_coverages = []
        node_obs_rate = mask.float().mean(dim=1)
        for t_idx in range(temporal_states.size(1)):
            state = temporal_states[:, t_idx]
            fine_adj, reliability = self._fused_fine_graph(x, mask, state)
            fine_state = self.fine_conv(state, fine_adj)
            all_states.append(fine_state)
            edge_reliabilities.append(reliability.mean(dim=-1))
            region_coverages.append(node_obs_rate)

            coarse_in = fine_state
            for conv in self.coarse_convs:
                coarse_out, cluster_sizes = self._coarse_state(coarse_in, fine_adj, conv)
                all_states.append(coarse_out)
                assign = self.assignment.to(x.device)
                counts = assign.sum(dim=0).clamp_min(1.0)
                cluster_obs = torch.einsum("bn,nk->bk", node_obs_rate, assign) / counts.view(1, -1)
                node_region_obs = torch.einsum("bk,nk->bn", cluster_obs, assign)
                edge_reliabilities.append(reliability.mean(dim=-1))
                region_coverages.append(node_region_obs)
                coarse_in = coarse_out

        states = torch.stack(all_states, dim=1)
        edge_rel = torch.stack(edge_reliabilities, dim=1).mean(dim=1)
        region_cov = torch.stack(region_coverages, dim=1).mean(dim=1)
        pred = self.readout(states, node_obs_rate, edge_rel, region_cov)
        base = x[:, -1:, :]
        return pred + base

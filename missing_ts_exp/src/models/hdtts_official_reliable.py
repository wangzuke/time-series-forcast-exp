"""MR-MSGF modules layered on top of the official HD-TTS implementation.

The official repository under ``external_repro/official/hdtts`` is intentionally
left untouched. ``OfficialMRMSGFHDTTSModel`` inherits its mask-aware encoder,
temporal hierarchy, spatial pooling hierarchy and direct forecasting readout,
then adds the modules specified in ``docs/论文草稿.md``.

This file deliberately contains no adaptive graph. Adaptive adjacency was an
earlier exploratory overlay and is not part of the current MR-MSGF design.
"""
from __future__ import annotations

import pathlib
import sys
import hashlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


ROOT = pathlib.Path(__file__).resolve().parents[3]
OFFICIAL_HDTTS = ROOT / "external_repro" / "official" / "hdtts"
if str(OFFICIAL_HDTTS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HDTTS))

from lib.nn.models.hdtts_model import HDTTSModel  # noqa: E402


def _row_normalize(adj: Tensor, eps: float = 1e-6) -> Tensor:
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(eps)


def graph_region_assignment(physical_adj: Tensor, max_region_size: int = 8) -> Tensor:
    """Build connected, size-bounded regions from the physical graph only.

    A region grows only through a positive-weight physical edge from an already
    selected member. Nodes that cannot be reached become seeds of later,
    possibly smaller regions. This prevents zero-affinity tie breaking from
    grouping unrelated sensors merely to fill a region.
    """
    if physical_adj.ndim != 2 or physical_adj.size(0) != physical_adj.size(1):
        raise ValueError("physical_adj must be a square dense matrix")
    n_nodes = int(physical_adj.size(0))
    region_size = max(1, int(max_region_size))
    if n_nodes <= region_size:
        return torch.ones(n_nodes, 1, dtype=torch.float32, device=physical_adj.device)

    affinity = 0.5 * (physical_adj.float() + physical_adj.float().transpose(0, 1))
    affinity.fill_diagonal_(0.0)
    remaining = torch.ones(n_nodes, dtype=torch.bool, device=affinity.device)
    groups: list[Tensor] = []
    while bool(remaining.any()):
        candidates = torch.where(remaining)[0]
        degree = affinity[candidates][:, remaining].sum(dim=-1)
        seed = candidates[degree.argmax()]
        members = [int(seed.item())]
        remaining[seed] = False
        while len(members) < region_size:
            member_idx = torch.tensor(members, dtype=torch.long, device=affinity.device)
            candidates = torch.where(remaining)[0]
            if not len(candidates):
                break
            # A candidate must be connected to at least one current member.
            scores = affinity[member_idx][:, candidates].max(dim=0).values
            best_score, best_pos = scores.max(dim=0)
            if float(best_score) <= 0.0:
                break
            next_node = candidates[best_pos]
            members.append(int(next_node.item()))
            remaining[next_node] = False
        groups.append(torch.tensor(members, dtype=torch.long, device=affinity.device))

    assignment = torch.zeros(n_nodes, len(groups), dtype=torch.float32, device=affinity.device)
    for region, members in enumerate(groups):
        assignment[members, region] = 1.0
    return assignment


def _longest_missing_fraction(mask: Tensor) -> Tensor:
    """Longest zero run per node, normalized by the input-window length."""
    missing = mask <= 0.5
    current = torch.zeros_like(mask[:, 0], dtype=torch.float32)
    longest = torch.zeros_like(current)
    for step in range(mask.size(1)):
        current = torch.where(missing[:, step], current + 1.0, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest / float(max(1, mask.size(1)))


def _static_reliability_features(
    mask: Tensor, physical_adjacency: Tensor, region_assignment: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the four Feature-only statistics directly from the input mask."""
    node_rate = mask.mean(dim=1)
    q_ij = torch.bmm(mask.transpose(1, 2), mask) / float(max(1, mask.size(1)))
    physical = physical_adjacency.unsqueeze(0)
    neighbor_overlap = (q_ij * physical).sum(dim=-1) / physical.sum(dim=-1).clamp_min(1e-6)
    counts = region_assignment.sum(dim=0).clamp_min(1.0)
    region_rate = torch.einsum("bn,nk->bk", node_rate, region_assignment) / counts.view(1, -1)
    region_coverage = torch.einsum("bk,nk->bn", region_rate, region_assignment)
    return node_rate, neighbor_overlap, region_coverage, _longest_missing_fraction(mask)


def _stable_permutation_seed(mask_seed: int, split_id: int, absolute_window_index: int) -> int:
    """Map immutable window metadata to a platform-independent 63-bit seed."""
    payload = f"mrmsgf-qperm-v1:{mask_seed}:{split_id}:{absolute_window_index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def permuted_pairwise_q(
    q_ij: Tensor, window_index: Tensor, split_id: Tensor, mask_seed: int
) -> Tensor:
    """Apply one deterministic node permutation to each window's ``Q`` matrix.

    The permutation is keyed by the mask seed, split id and absolute window
    start, never by mini-batch order or the number of forward passes.
    """
    if q_ij.ndim != 3 or q_ij.size(-1) != q_ij.size(-2):
        raise ValueError("q_ij must have shape (batch, nodes, nodes)")

    def sample_values(values: Tensor) -> list[int]:
        if values is None:
            raise ValueError("permuted_pairwise q_mode requires window metadata")
        if values.ndim >= 4:
            values = values[:, 0, 0, 0]
        elif values.ndim >= 2:
            values = values.reshape(values.size(0), -1)[:, 0]
        return [int(value) for value in values.detach().cpu().tolist()]

    windows = sample_values(window_index)
    splits = sample_values(split_id)
    if len(windows) != q_ij.size(0) or len(splits) != q_ij.size(0):
        raise ValueError("window metadata batch size does not match q_ij")
    output = torch.empty_like(q_ij)
    for batch_idx, (window, split) in enumerate(zip(windows, splits)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_permutation_seed(mask_seed, split, window))
        permutation = torch.randperm(q_ij.size(-1), generator=generator, device="cpu").to(q_ij.device)
        output[batch_idx] = q_ij[batch_idx].index_select(0, permutation).index_select(1, permutation)
    # Preserve q_ii=r_i for every node. A simultaneous node permutation never
    # moves a diagonal value off diagonal, so this only restores node-specific
    # self-observation rates while retaining the multiset of pairwise evidence.
    diagonal = torch.arange(q_ij.size(-1), device=q_ij.device)
    output[:, diagonal, diagonal] = q_ij[:, diagonal, diagonal]
    return output


class OfficialFeatureOnlyHDTTSModel(HDTTSModel):
    """Official HD-TTS plus a mask-statistics bias, with no dynamic graph branch.

    This intentionally is not a subclass of ``OfficialMRMSGFHDTTSModel``: it
    must not instantiate observation-graph, fine-graph, or regional-propagation
    parameters in the Feature-only causal control.
    """

    def __init__(
        self,
        *args,
        physical_adjacency: Tensor,
        region_assignment: Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        n_nodes = int(self.encoder_time.n_nodes)
        if physical_adjacency.shape != (n_nodes, n_nodes):
            raise ValueError(f"physical_adjacency has shape {tuple(physical_adjacency.shape)}, expected ({n_nodes}, {n_nodes})")
        physical_adjacency = physical_adjacency.float().clone()
        physical_adjacency.fill_diagonal_(1.0)
        self.register_buffer("physical_adjacency", _row_normalize(physical_adjacency))
        if region_assignment is None:
            region_assignment = graph_region_assignment(self.physical_adjacency)
        if region_assignment.ndim != 2 or region_assignment.size(0) != n_nodes:
            raise ValueError("region_assignment must have shape (n_nodes, n_regions)")
        if not torch.allclose(region_assignment.sum(dim=1), torch.ones(n_nodes, device=region_assignment.device)):
            raise ValueError("region_assignment must assign every node to exactly one region")
        self.register_buffer("region_assignment", region_assignment.float())
        n_layers = (self.pooling_layers + 1) * len(self.drnn.rnns)
        self.scale_reliability = nn.Linear(4, n_layers, bias=False)
        self.last_diagnostics: dict[str, Tensor] = {}

    @staticmethod
    def _mask_3d(mask: Optional[Tensor], x: Tensor) -> Tensor:
        if mask is None:
            return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
        return mask.float().mean(dim=-1).clamp(0.0, 1.0)

    def _readout_with_reliability(self, states: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor]:
        states_for_score = states.movedim(1, -2)
        scores = self.readout.lin_scores_state(states_for_score)
        rel_bias = self.scale_reliability(reliability)
        if self.readout.multi_step_scores:
            scores = self.readout.rearrange(scores)
            alpha = torch.softmax(scores + rel_bias.unsqueeze(1).unsqueeze(-1), dim=-2)
            fused = (states_for_score.unsqueeze(-4) * alpha).sum(dim=-2)
            scale_alpha = alpha.movedim(-2, 1).mean(-1)
        else:
            alpha = torch.softmax(scores + rel_bias.unsqueeze(-1), dim=-2)
            fused = (states_for_score * alpha).sum(dim=-2)
            scale_alpha = alpha.movedim(-2, 1).mean(-1)
        return self.readout.readout(fused), scale_alpha

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        window_index: Optional[Tensor] = None,
        split_id: Optional[Tensor] = None,
    ):
        mask = self._mask_3d(input_mask, x)
        encoded_history = self.encoder_time(x, mask=input_mask, u=u)
        out_time = self.drnn(encoded_history)
        out_space, _ = self.spatial_conv(out_time, edge_index, edge_weight, cached=True)
        states = rearrange(out_space, "b s t n d -> b (s t) n d")
        node_rate, neighbor_overlap, region_coverage, block_fraction = _static_reliability_features(
            mask, self.physical_adjacency, self.region_assignment
        )
        reliability = torch.stack([node_rate, neighbor_overlap, region_coverage, block_fraction], dim=-1)
        out, scale_alpha = self._readout_with_reliability(states, reliability)
        self.last_diagnostics = {
            "node_observation_rate": node_rate.detach(),
            "physical_neighbor_co_observation": neighbor_overlap.detach(),
            "region_coverage": region_coverage.detach(),
            "block_fraction": block_fraction.detach(),
            "scale_attention": scale_alpha.detach(),
        }
        return out, None, scale_alpha, None


class OfficialMRMSGFHDTTSModel(HDTTSModel):
    """Official HD-TTS backbone plus the paper's MR-MSGF increment.

    ``variant`` implements the pre-registered ablations: ``phy_only`` (M1),
    ``obs_only`` (M2), ``fine`` (M3), ``coarse_no_gate`` (M4), and ``full``
    (M5). The Feature-only causal control is deliberately implemented by the
    separate ``OfficialFeatureOnlyHDTTSModel`` above.
    """

    VALID_VARIANTS = {"phy_only", "obs_only", "fine", "coarse_no_gate", "full"}
    VALID_Q_MODES = {"observed", "permuted_pairwise"}

    def __init__(
        self,
        *args,
        physical_adjacency: Tensor,
        region_assignment: Tensor | None = None,
        variant: str = "full",
        relation_dim: int | None = None,
        graph_topk: int = 8,
        reliability_beta: float = 1.0,
        q_mode: str = "observed",
        mask_seed: int | None = None,
        disable_overlap_reliability: bool = False,
        disable_readout_reliability: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown MR-MSGF variant {variant!r}; choose {sorted(self.VALID_VARIANTS)}")
        if q_mode not in self.VALID_Q_MODES:
            raise ValueError(f"Unknown q_mode {q_mode!r}; choose {sorted(self.VALID_Q_MODES)}")
        if q_mode == "permuted_pairwise" and mask_seed is None:
            raise ValueError("permuted_pairwise q_mode requires mask_seed")

        n_nodes = int(self.encoder_time.n_nodes)
        hidden_size = int(self.encoder_time.hidden_size)
        if physical_adjacency.shape != (n_nodes, n_nodes):
            raise ValueError(f"physical_adjacency has shape {tuple(physical_adjacency.shape)}, expected ({n_nodes}, {n_nodes})")
        physical_adjacency = physical_adjacency.float().clone()
        physical_adjacency.fill_diagonal_(1.0)
        self.register_buffer("physical_adjacency", _row_normalize(physical_adjacency))

        if region_assignment is None:
            region_assignment = graph_region_assignment(self.physical_adjacency)
        if region_assignment.ndim != 2 or region_assignment.size(0) != n_nodes:
            raise ValueError("region_assignment must have shape (n_nodes, n_regions)")
        if not torch.allclose(region_assignment.sum(dim=1), torch.ones(n_nodes, device=region_assignment.device)):
            raise ValueError("region_assignment must assign every node to exactly one region")
        self.register_buffer("region_assignment", region_assignment.float())

        self.variant = variant
        self.q_mode = q_mode
        self.mask_seed = None if mask_seed is None else int(mask_seed)
        self.graph_topk = max(1, min(int(graph_topk), n_nodes))
        self.reliability_beta = nn.Parameter(torch.tensor(float(reliability_beta)))
        self.disable_overlap_reliability = bool(disable_overlap_reliability)
        self.disable_readout_reliability = bool(disable_readout_reliability)
        relation_dim = int(relation_dim or hidden_size)
        self.obs_query = nn.Linear(hidden_size, relation_dim, bias=False)
        self.obs_key = nn.Linear(hidden_size, relation_dim, bias=False)
        self.edge_gate = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))

        self.fine_neigh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fine_self = nn.Linear(hidden_size, hidden_size)
        self.coarse_neigh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.coarse_self = nn.Linear(hidden_size, hidden_size)
        self.coarse_project = nn.Linear(hidden_size, hidden_size, bias=False)
        self.coarse_gate = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        self.gate_prior_scale = nn.Parameter(torch.ones(5))
        n_layers = (self.pooling_layers + 1) * len(self.drnn.rnns)
        self.scale_reliability = nn.Linear(4, n_layers, bias=False)
        self.last_diagnostics: dict[str, Tensor] = {}

    @staticmethod
    def _mask_3d(mask: Optional[Tensor], x: Tensor) -> Tensor:
        if mask is None:
            return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
        return mask.float().mean(dim=-1).clamp(0.0, 1.0)

    def _observed_graph(
        self, encoded_history: Tensor, mask: Tensor,
        window_index: Optional[Tensor] = None, split_id: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Construct A_obs from masked temporal pooling and edge overlap q_ij."""
        raw_counts = mask.sum(dim=1)
        counts = raw_counts.clamp_min(1.0)
        node_rate = raw_counts / float(max(1, mask.size(1)))
        zero_nodes = raw_counts <= 0.0
        pooled = (encoded_history * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        query = self.obs_query(pooled)
        key = self.obs_key(pooled)
        score = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        overlap = torch.bmm(mask.transpose(1, 2), mask)
        q_ij = (overlap / float(max(1, mask.size(1)))).clamp(0.0, 1.0)
        if self.q_mode == "permuted_pairwise":
            q_ij = permuted_pairwise_q(q_ij, window_index, split_id, self.mask_seed)
        unavailable_pair = zero_nodes.unsqueeze(-1) | zero_nodes.unsqueeze(-2)
        q_ij = q_ij.masked_fill(unavailable_pair, 0.0)
        if not self.disable_overlap_reliability:
            score = score + F.softplus(self.reliability_beta) * torch.log(q_ij.clamp_min(1e-6))
        top_values, top_index = torch.topk(score, k=self.graph_topk, dim=-1)
        sparse_score = torch.full_like(score, -torch.inf)
        sparse_score.scatter_(dim=-1, index=top_index, src=top_values)
        sparse_score = sparse_score.masked_fill(zero_nodes.unsqueeze(-1), -torch.inf)
        selected = torch.isfinite(sparse_score)
        a_obs = torch.softmax(sparse_score, dim=-1)
        a_obs = torch.nan_to_num(a_obs, nan=0.0)
        eye = torch.eye(score.size(-1), dtype=score.dtype, device=score.device).unsqueeze(0)
        # A fully unobserved source has no learned relation, including no
        # arbitrary top-k tie or learned self-loop. It falls back to A_phy.
        a_obs = _row_normalize(a_obs + eye * (~zero_nodes).unsqueeze(-1))
        return a_obs, q_ij, node_rate, selected, zero_nodes

    def _fine_graph(
        self, a_obs: Tensor, q_ij: Tensor, node_rate: Tensor, zero_nodes: Tensor
    ) -> tuple[Tensor, Tensor]:
        a_phy = self.physical_adjacency.unsqueeze(0).expand_as(a_obs)
        if self.variant == "phy_only":
            return a_phy, torch.ones_like(a_obs)
        if self.variant == "obs_only":
            edge_gate = torch.zeros_like(a_obs)
            a_fine = a_obs
        else:
            rate_i = node_rate.unsqueeze(-1).expand_as(q_ij)
            rate_j = node_rate.unsqueeze(-2).expand_as(q_ij)
            features = torch.stack([rate_i, rate_j, q_ij, a_phy, a_obs], dim=-1)
            edge_gate = torch.sigmoid(self.edge_gate(features).squeeze(-1))
            a_fine = edge_gate * a_phy + (1.0 - edge_gate) * a_obs
        a_fine = torch.where(zero_nodes.unsqueeze(-1), a_phy, a_fine)
        edge_gate = torch.where(zero_nodes.unsqueeze(-1), torch.ones_like(edge_gate), edge_gate)
        return _row_normalize(a_fine), edge_gate

    def _observation_features(
        self, node_rate: Tensor, q_ij: Tensor, block_fraction: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return physical-neighbor overlap and fixed-region coverage."""
        phy = self.physical_adjacency.unsqueeze(0)
        neighbor_overlap = (q_ij * phy).sum(dim=-1) / phy.sum(dim=-1).clamp_min(1e-6)
        assign = self.region_assignment
        counts = assign.sum(dim=0).clamp_min(1.0)
        region_rate = torch.einsum("bn,nk->bk", node_rate, assign) / counts.view(1, -1)
        region_coverage = torch.einsum("bk,nk->bn", region_rate, assign)
        return neighbor_overlap, region_coverage

    def _fine_propagate(self, state: Tensor, a_fine: Tensor) -> Tensor:
        neigh = torch.bmm(a_fine, self.fine_neigh(state))
        return F.elu(neigh + self.fine_self(state))

    def _coarse_propagate(
        self,
        z_fine: Tensor,
        a_fine: Tensor,
        node_rate: Tensor,
        q_ij: Tensor,
        block_fraction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build A_coarse=S^T A_fine S and return lifted coarse context."""
        assign = self.region_assignment
        counts = assign.sum(dim=0).clamp_min(1.0)
        region = torch.einsum("nk,bnd->bkd", assign, z_fine) / counts.view(1, -1, 1)
        a_coarse = _row_normalize(torch.einsum("nk,bij,jl->bkl", assign, a_fine, assign))
        coarse = F.elu(torch.bmm(a_coarse, self.coarse_neigh(region)) + self.coarse_self(region))
        z_coarse = torch.einsum("nk,bkd->bnd", assign, coarse)

        neighbor_overlap, region_coverage = self._observation_features(node_rate, q_ij, block_fraction)
        region_center = torch.einsum("nk,bkd->bnd", assign, region)
        # Singleton or perfectly homogeneous regions have zero dispersion;
        # clamp before sqrt to keep its backward derivative finite.
        dispersion = (z_fine - region_center).square().mean(dim=-1).clamp_min(1e-6).sqrt()
        dispersion = dispersion / dispersion.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        features = torch.stack([node_rate, neighbor_overlap, region_coverage, block_fraction, dispersion], dim=-1)
        learned = self.coarse_gate(features).squeeze(-1)
        prior = torch.stack(
            [1.0 - node_rate, 1.0 - neighbor_overlap, region_coverage - 1.0, block_fraction, -dispersion], dim=-1
        )
        alpha = torch.sigmoid(learned + (prior * self.gate_prior_scale).sum(dim=-1))
        return z_coarse, alpha, region_coverage, dispersion

    def _readout_with_reliability(self, states: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor]:
        states_for_score = states.movedim(1, -2)  # (B, N, layers, D)
        scores = self.readout.lin_scores_state(states_for_score)
        rel_bias = torch.zeros_like(self.scale_reliability(reliability)) if self.disable_readout_reliability else self.scale_reliability(reliability)
        if self.readout.multi_step_scores:
            # Match the official AttentionReadout layout: B, H, N, L, D.
            scores = self.readout.rearrange(scores)
            alpha = torch.softmax(scores + rel_bias.unsqueeze(1).unsqueeze(-1), dim=-2)
            fused = (states_for_score.unsqueeze(-4) * alpha).sum(dim=-2)
            scale_alpha = alpha.movedim(-2, 1).mean(-1)
        else:
            alpha = torch.softmax(scores + rel_bias.unsqueeze(-1), dim=-2)
            fused = (states_for_score * alpha).sum(dim=-2)
            scale_alpha = alpha.movedim(-2, 1).mean(-1)
        return self.readout.readout(fused), scale_alpha

    def diagnostic_summary(self) -> dict[str, float]:
        return {key: float(value.detach().float().mean().cpu()) for key, value in self.last_diagnostics.items()}

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        window_index: Optional[Tensor] = None,
        split_id: Optional[Tensor] = None,
    ):
        mask = self._mask_3d(input_mask, x)
        encoded_history = self.encoder_time(x, mask=input_mask, u=u)
        out_time = self.drnn(encoded_history)
        a_obs, q_ij, node_rate, edge_selected, zero_nodes = self._observed_graph(
            encoded_history, mask, window_index, split_id
        )

        # Keep the official HD-TTS hierarchy strictly static and physical. The
        # per-sample dynamic Afine is used only by the explicit MR-MSGF branch
        # below, so predictions cannot depend on unrelated batch members.
        out_space, _ = self.spatial_conv(out_time, edge_index, edge_weight, cached=True)

        block_fraction = _longest_missing_fraction(mask)
        neighbor_overlap, static_region_coverage = self._observation_features(node_rate, q_ij, block_fraction)
        a_fine, edge_gate = self._fine_graph(a_obs, q_ij, node_rate, zero_nodes)
        fused_fine, alpha_by_time, region_by_time, dispersion_by_time = [], [], [], []
        for time_level in range(out_time.size(1)):
            z_fine = self._fine_propagate(out_time[:, time_level], a_fine)
            if self.variant in {"coarse_no_gate", "full"}:
                z_coarse, alpha, region_coverage, dispersion = self._coarse_propagate(
                    z_fine, a_fine, node_rate, q_ij, block_fraction
                )
                if self.variant == "coarse_no_gate":
                    alpha = torch.ones_like(alpha)
                z_fine = z_fine + alpha.unsqueeze(-1) * self.coarse_project(z_coarse)
            else:
                alpha = torch.zeros_like(node_rate)
                region_coverage = static_region_coverage
                dispersion = torch.zeros_like(node_rate)
            fused_fine.append(z_fine)
            alpha_by_time.append(alpha)
            region_by_time.append(region_coverage)
            dispersion_by_time.append(dispersion)

        out_space = out_space.clone()
        out_space[:, 0] = torch.stack(fused_fine, dim=1)
        alpha_mean = torch.stack(alpha_by_time, dim=1).mean(dim=1)
        region_mean = torch.stack(region_by_time, dim=1).mean(dim=1)
        dispersion_mean = torch.stack(dispersion_by_time, dim=1).mean(dim=1)
        states = rearrange(out_space, "b s t n d -> b (s t) n d")
        reliability = torch.stack([node_rate, neighbor_overlap, region_mean, block_fraction], dim=-1)
        out, scale_alpha = self._readout_with_reliability(states, reliability)

        self.last_diagnostics = {
            "node_observation_rate": node_rate.detach(),
            "edge_common_observation_rate": q_ij.detach(),
            "edge_gate": edge_gate.detach(),
            "edge_selected": edge_selected.detach(),
            "zero_observation_node": zero_nodes.detach(),
            "physical_neighbor_co_observation": neighbor_overlap.detach(),
            "region_coverage": region_mean.detach(),
            "coarse_gate": alpha_mean.detach(),
            "regional_dispersion": dispersion_mean.detach(),
            "block_fraction": block_fraction.detach(),
            "scale_attention": scale_alpha.detach(),
        }
        return out, None, scale_alpha, None


# Compatibility alias for old imports. New experiment commands use ``mrmsgf``.
OfficialReliableHDTTSModel = OfficialMRMSGFHDTTSModel

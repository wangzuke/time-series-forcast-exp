"""Small, isolated ablations around the released S4M implementation.

The released model files are left untouched.  These adapters remove only the
selected information path while preserving the surrounding S4M parameters and
forecasting head.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn


class NoMaskStream(nn.Module):
    """Call the official first MDS-S4 block without its explicit mask input."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.original = original

    def forward(self, x: torch.Tensor, _mask: torch.Tensor | None = None) -> torch.Tensor:
        # The released S4_mask kernel requires a tensor and unconditionally
        # transforms it. A zero tensor is therefore the neutral mask stream:
        # the S4 block and its parameters remain intact, but carry no input
        # information about the actual missingness pattern.
        new_x = self.original.s4_1(x, torch.zeros_like(x))[0]
        residual = x + new_x
        normalized = self.original.norm1(residual)
        feed_forward = self.original.dropout(
            self.original.activation(self.original.conv1(normalized.transpose(-1, 1)))
        )
        feed_forward = self.original.dropout(
            self.original.conv2(feed_forward).transpose(-1, 1)
        )
        return self.original.norm2(normalized + feed_forward)


def _no_prototype_retrieval_forward(self: nn.Module, query: torch.Tensor) -> torch.Tensor:
    """Keep S4M's query encoder and fusion layer, but bypass prototype lookup.

    This tests whether the ATPM memory read contributes to shift stability.
    The query representation replaces the retrieved prototype representation;
    the training-time memory bank is not read or updated.
    """
    batch_size, length, channels = query.shape
    query_4d = query.unsqueeze(0)
    unfolded = self.unfold(query_4d)[:, :, :length]
    sliding_query = unfolded.view(
        batch_size, self.config.short_len, channels, length
    ).permute(0, 3, 1, 2)
    encoded = self.encoder_u(sliding_query)
    hidden = encoded.shape[-1]
    encoded = torch.nn.functional.normalize(encoded.reshape(-1, hidden), dim=1)
    encoded = encoded.reshape(batch_size, length, hidden)
    fused = torch.cat([encoded, encoded, query], dim=-1)
    return torch.matmul(fused, self.pred_w) + self.pred_b + encoded


def apply_s4m_ablation(model: nn.Module, ablation: str) -> None:
    if ablation == "none":
        return
    if ablation == "no_mask_stream":
        model.s4_layers[0] = NoMaskStream(model.s4_layers[0])
        return
    if ablation == "no_prototype_retrieval":
        model.mem_net.forward = MethodType(_no_prototype_retrieval_forward, model.mem_net)
        # The first minibatch is still skipped by the shared runner for a
        # matched training loop, but no memory initialization is necessary.
        model.warmup = MethodType(lambda _self, *_args, **_kwargs: None, model)
        return
    raise ValueError(f"unknown S4M ablation: {ablation}")

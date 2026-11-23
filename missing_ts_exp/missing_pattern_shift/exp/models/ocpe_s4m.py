"""Add OCPE after ATPM without editing the frozen S4M forward implementation."""
from __future__ import annotations

import copy
import torch
from torch import nn
from posterior_s4 import ModelConfig
from ocpe_fusion_s4 import ConditionalEncoder
from run_official_baseline import build_model, s4m_metadata


def tensor_tree(value, device=None):
    if torch.is_tensor(value):
        return value.detach().to(device=device or value.device).clone()
    if isinstance(value, list):
        return [tensor_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree(v, device) for v in value)
    if isinstance(value, dict):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    return copy.deepcopy(value)


class OCPES4M(nn.Module):
    def __init__(self, dataset, channels, enabled, seed, device):
        super().__init__()
        self.cfg = ModelConfig(channels=channels)
        self.enabled = bool(enabled)
        self.dataset = dataset
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.backbone = build_model("S4M", dataset, channels, 96, 96, device, s4m_e_layers=4)
        self.observation = self.encoding_projection = None
        if self.enabled:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + 1000)
                self.observation = ConditionalEncoder(self.cfg).to(device)
                self.encoding_projection = nn.Linear(32, self.backbone.configs.en_conv_hidden_size,
                                                     bias=False, device=device)
                nn.init.zeros_(self.encoding_projection.weight)
        self._encoding = None
        self.backbone.s4_layers[0].register_forward_pre_hook(self._fuse)

    def _fuse(self, module, inputs):
        if self._encoding is None:
            return inputs
        return (inputs[0] + self._encoding, *inputs[1:])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.mem_net.is_training = int(mode)
        return self

    def forward(self, x, mask):
        # Both branches ignore arbitrary hidden placeholders.
        x = torch.where(mask.bool(), x, torch.zeros_like(x))
        nll = x.new_zeros(x.shape[0])
        try:
            if self.enabled:
                features, nll = self.observation(x, mask, mode="F3")
                self._encoding = self.encoding_projection(features)
            out = self.backbone(x, mask, *s4m_metadata(x))[:, -96:, :]
        finally:
            self._encoding = None
        return out, nll

    def memory_state(self, cpu=False):
        memory = self.backbone.mem_net.memory
        state = {k: getattr(memory, k) for k in ("memory", "repre", "num_dic", "count", "idx_arr")}
        state["net_count"] = self.backbone.mem_net.count
        return tensor_tree(state, "cpu" if cpu else None)

    def load_memory(self, state):
        state = tensor_tree(state, next(self.parameters()).device)
        memory = self.backbone.mem_net.memory
        for key in ("memory", "repre", "num_dic", "count", "idx_arr"):
            setattr(memory, key, state[key])
        self.backbone.mem_net.count = state["net_count"]

    def ocpe_parameters(self):
        return [] if not self.enabled else list(self.observation.parameters()) + list(self.encoding_projection.parameters())

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "backbone": sum(p.numel() for p in self.backbone.parameters()),
                "ocpe": sum(p.numel() for p in self.ocpe_parameters())}

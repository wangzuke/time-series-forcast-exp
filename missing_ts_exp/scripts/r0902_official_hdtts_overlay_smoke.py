"""Smoke test for the official-HD-TTS reliable graph overlay.

Run with:
    /media/data1/wangzk/miniconda3/envs/hd-tts/bin/python \
        missing_ts_exp/scripts/r0902_official_hdtts_overlay_smoke.py
"""
from __future__ import annotations

import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missing_ts_exp.src.models.hdtts_official_reliable import OfficialReliableHDTTSModel


def main() -> None:
    torch.manual_seed(2024)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.ones(edge_index.size(1))
    x = torch.randn(2, 12, 10, 1)
    mask = (torch.rand(2, 12, 10, 1) > 0.3).float()
    model = OfficialReliableHDTTSModel(
        input_size=1,
        hidden_size=8,
        n_nodes=10,
        horizon=3,
        rnn_layers=2,
        pooling_layers=1,
        mask_size=1,
        dilation=2,
        mp_method="propconv",
        mp_kernel_size=1,
        recursive_lifting=True,
        fully_connected_readout=False,
        multi_step_scores=True,
        activation="elu",
        dropout=0.0,
        graph_topk=4,
    )
    y_hat, *_ = model(x * mask, edge_index, edge_weight=edge_weight, input_mask=mask)
    loss = y_hat.square().mean()
    loss.backward()
    print(
        "official_reliable_forward",
        tuple(y_hat.shape),
        "params",
        sum(p.numel() for p in model.parameters()),
        "graph",
        tuple(model.last_fused_graph.shape),
    )


if __name__ == "__main__":
    main()

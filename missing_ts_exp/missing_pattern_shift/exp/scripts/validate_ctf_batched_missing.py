#!/usr/bin/env python3
"""Verify the CTF batched-missing extension before any benchmark run.

It proves two properties: exact forward/backward equivalence with the
released missing model at B=1, and finite forward/backward values with a
batch whose members have different missing masks.  It is intentionally a
small synthetic check; performance experiments remain separate.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from types import SimpleNamespace

import torch

from protocol import EXP_ROOT, atomic_json_dump


def config(batched_missing: bool) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast", features="M", seq_len=96, pred_len=24,
        use_norm=1, patch_lens=[24] * 7, sampling_periods=[1.0] * 7,
        num_global_tokens=2, keep_prob=1.0, d_model=256, d_ff=1024,
        n_heads=8, e_layers=2, factor=3, dropout=0.1, activation="gelu",
        batch_size=32, enc_in=7, dec_in=7, c_out=7,
        batched_missing=batched_missing,
    )


def masked_input(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, 96, 7, device=device)
    mask = torch.ones_like(x)
    # Different patterns ensure the B>1 path cannot accidentally share a
    # single token layout across samples.
    patterns = [(0, slice(0, 24), 0), (1, slice(24, 72), 1),
                (2, slice(0, 96), 2), (3, slice(None, None, 3), 4)]
    for sample, steps, channel in patterns[:batch_size]:
        mask[sample, steps, channel] = 0
    return x * mask, mask


def gradient_difference(reference: torch.nn.Module, candidate: torch.nn.Module) -> float:
    maximum = 0.0
    for ref, new in zip(reference.parameters(), candidate.parameters()):
        if ref.grad is None and new.grad is None:
            continue
        maximum = max(maximum, float((ref.grad - new.grad).abs().max()))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out", type=pathlib.Path,
                        default=EXP_ROOT / "results" / "validation" / "ctf_batched_missing.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation")

    repository = EXP_ROOT / "baselines" / "ChannelTokenFormer"
    sys.path.insert(0, str(repository))
    from models.ChannelTokenFormer_missing import Model

    torch.manual_seed(args.seed)
    released = Model(config(False)).to(device)
    batched = Model(config(True)).to(device)
    batched.load_state_dict(copy.deepcopy(released.state_dict()), strict=True)

    x, mask = masked_input(1, device)
    released.eval(); batched.eval()
    with torch.no_grad():
        expected = released(x, None, None, None, mask)
        actual = batched(x, None, None, None, mask)
    forward_max_abs = float((expected - actual).abs().max())

    # Reset dropout RNG before each forward so the B=1 gradient comparison is
    # deterministic as well.
    released.train(); batched.train()
    released.zero_grad(set_to_none=True); batched.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed + 1)
    released(x, None, None, None, mask).square().mean().backward()
    torch.manual_seed(args.seed + 1)
    batched(x, None, None, None, mask).square().mean().backward()
    gradient_max_abs = gradient_difference(released, batched)

    x4, mask4 = masked_input(4, device)
    batched.zero_grad(set_to_none=True)
    output4 = batched(x4, None, None, None, mask4)
    output4.square().mean().backward()
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                           for parameter in batched.parameters())
    result = {
        "status": "complete",
        "seed": args.seed,
        "device": str(device),
        "b1_forward_max_abs": forward_max_abs,
        "b1_gradient_max_abs": gradient_max_abs,
        "b4_output_shape": list(output4.shape),
        "b4_finite_output": bool(torch.isfinite(output4).all()),
        "b4_finite_gradients": finite_gradients,
        "pass": forward_max_abs == 0.0 and gradient_max_abs == 0.0 and finite_gradients,
    }
    atomic_json_dump(result, args.out)
    print(json.dumps(result, ensure_ascii=False))
    if not result["pass"]:
        raise SystemExit("batched CTF validation failed")


if __name__ == "__main__":
    main()

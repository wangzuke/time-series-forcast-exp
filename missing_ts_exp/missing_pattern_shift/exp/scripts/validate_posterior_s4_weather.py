#!/usr/bin/env python3
"""Check the frozen-trainer adaptation and real Weather GPU interfaces."""
import ast
import json
import pathlib

import torch

from protocol import EXP_ROOT, atomic_json_dump
from posterior_s4_common import (ExperimentDataset, ModelConfig, PosteriorS4, mask_audit,
                                 mask_fingerprint, move_batch, runtime_metadata, seed_all)
from run_posterior_s4_weather import SUITE, source_fingerprints
from validate_posterior_s4 import numerical_checks


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Weather preflight requires the assigned CUDA GPU")
    base_source = (EXP_ROOT / "scripts/run_posterior_s4.py").read_text()
    weather_source = (EXP_ROOT / "scripts/run_posterior_s4_weather.py").read_text()
    base_functions = {node.name: node for node in ast.parse(base_source).body if isinstance(node, ast.FunctionDef)}
    weather_functions = {node.name: node for node in ast.parse(weather_source).body if isinstance(node, ast.FunctionDef)}
    checked = ("rng_state", "restore_rng", "evaluate", "train_epoch", "posterior_diagnostic", "input_probe")
    for name in checked:
        assert ast.dump(base_functions[name]) == ast.dump(weather_functions[name]), name
    base_main = ast.dump(base_functions["main"])
    weather_main = ast.dump(weather_functions["main"])
    assert base_main != weather_main  # only dataset-file discovery differs
    # A textual normalization verifies the whole main function, including
    # optimizer, data order, checkpoint selection and test loop, is identical.
    normalized = weather_source.replace('DATASETS[args.dataset]["file"]',
                                        '{"ETTh1":"ETTh1.csv", "Electricity":"electricity.csv"}[args.dataset]')
    normalized_main = {n.name: n for n in ast.parse(normalized).body if isinstance(n, ast.FunctionDef)}["main"]
    assert ast.dump(base_functions["main"]) == ast.dump(normalized_main)
    frozen_check = json.loads((EXP_ROOT / "diagnostics/posterior_s4_r50_b24_l96_h96_1119/correctness.json").read_text())
    sources = source_fingerprints()
    assert frozen_check["status"] == "passed"
    assert all(sources[name] == sha for name, sha in frozen_check["source"].items())
    device = torch.device("cuda:0")
    numerical = numerical_checks(device)
    seed_all(2024)
    dataset = ExperimentDataset("Weather", "train", "P", 2024, True)
    assert dataset.n_features == 21
    assert mask_fingerprint(dataset, 1) != mask_fingerprint(dataset, 2)
    cfg = ModelConfig(channels=21)
    batch = move_batch(torch.utils.data.default_collate([dataset[i] for i in range(8)]), device)
    versions = {}
    common = None
    for variant in ("LI-S4", "STAT-S4", "PU-S4"):
        model = PosteriorS4(variant, cfg, 2024).to(device).eval()
        actual = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(("blocks.", "time_head.", "channel_head."))}
        if common is not None:
            assert all(torch.equal(common[k], v) for k, v in actual.items())
        common = actual
        pred, nll = model(batch["x"], batch["mask"])
        assert pred.shape == (8, 96, 21) and torch.isfinite(pred).all() and torch.isfinite(nll).all()
        ((pred - batch["y"]).square().mean() + .1 * nll.mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if model.observation is not None:
            assert model.observation.loadings.grad.abs().sum() > 0
        ignored = torch.where(batch["mask"].bool(), batch["x"], torch.full_like(batch["x"], 999))
        assert torch.equal(pred.detach(), model(ignored, batch["mask"])[0])
        versions[variant] = {"shape": list(pred.shape), "parameters": model.parameter_counts()}
    audits = {family: mask_audit(ExperimentDataset("Weather", "test", family, 9101)) for family in ("P", "B", "B-uniform")}
    runtime = runtime_metadata()
    assert runtime["has_cuda_extension"]
    result = {"status": "passed", "source": sources, "frozen_training_functions_equal": list(checked),
              "main_function_equal_except_data_lookup": True, "numerical": numerical,
              "weather_versions": versions, "model_config": cfg.to_dict(), "mask_audits": audits,
              "train_windows": len(dataset), "runtime": runtime,
              "original_cuda_and_oracle_preflight": str(EXP_ROOT / "diagnostics/posterior_s4_r50_b24_l96_h96_1119/correctness.json")}
    output = EXP_ROOT / "diagnostics" / SUITE / "correctness.json"
    atomic_json_dump(result, output)
    print(json.dumps({"status": "passed", "output": str(output), "train_windows": len(dataset)}), flush=True)


if __name__ == "__main__":
    main()

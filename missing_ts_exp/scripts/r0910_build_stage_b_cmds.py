#!/usr/bin/env python3
"""Build the pre-registered METR-LA Block-T ablation matrix."""
from pathlib import Path

ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0910_stage_b_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def command(model, rate, seed, variant="full", no_bias=False):
    mask = MASKS / f"block_t_r{rate}_seed{seed}_observed.npy"
    label = model if model == "official_base" else variant
    if no_bias:
        label += "_no_readout_bias"
    tag = f"{label}_METR-LA_block_t_r{rate}_seed{seed}"
    extra = f"--mr_variant {variant}"
    if no_bias:
        extra += " --disable_readout_reliability"
    return (
        f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
        f"--model {model} {extra} --mask_path {mask} --seed {seed} "
        f"--epochs 100 --train_batches 300 --patience 20 --batch_size 32 "
        f"--out_dir {RESULT} --tag {tag}"
    )


def main():
    commands = []
    # Reuse 2024--2026 official-base/full outputs; only fill the two new seeds.
    for rate in (50, 70):
        for seed in (2027, 2028):
            commands += [command("official_base", rate, seed), command("mrmsgf", rate, seed)]
    variants = [("phy_only", False), ("obs_only", False), ("fine", False),
                ("coarse_no_gate", False), ("full", True)]
    for rate in (50, 70):
        for seed in (2024, 2025, 2026, 2027, 2028):
            for variant, no_bias in variants:
                commands.append(command("mrmsgf", rate, seed, variant, no_bias))
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()

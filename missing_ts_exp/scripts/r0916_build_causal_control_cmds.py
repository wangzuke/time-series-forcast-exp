#!/usr/bin/env python3
"""Generate the 10 new, pre-registered 0916 causal-control training jobs."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0916_causal_control_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
REGION = ROOT / "missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
RESULT = ROOT / "missing_ts_exp/results/r0916_causal_controls"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def command(model: str, seed: int) -> str:
    mask = MASKS / f"block_t_r50_seed{seed}_observed.npy"
    if model == "official_base_feature_only":
        tag = f"official_base_feature_only_METR-LA_block_t_r50_seed{seed}"
        extra = ""
    else:
        tag = f"mrmsgf_permuted_q_METR-LA_block_t_r50_seed{seed}"
        extra = f"--mr_variant full --q_mode permuted_pairwise --mask_seed {seed}"
    return (
        f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
        f"--model {model} {extra} --mask_path {mask} --seed {seed} "
        f"--epochs 100 --train_batches 300 --patience 20 --batch_size 32 "
        f"--region_assignment_path {REGION} --out_dir {RESULT} --tag {tag}"
    )


def main() -> None:
    commands = [command(model, seed) for model in ("official_base_feature_only", "mrmsgf") for seed in SEEDS]
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()

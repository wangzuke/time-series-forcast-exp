#!/usr/bin/env python3
"""Build official GRIN baseline commands for the primary Block-T condition."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0916_grin_baseline_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
RESULT = ROOT / "missing_ts_exp/results/r0916_grin_baseline"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def main() -> None:
    commands = []
    for seed in SEEDS:
        mask = MASKS / f"block_t_r50_seed{seed}_observed.npy"
        tag = f"grin_official_METR-LA_block_t_r50_seed{seed}"
        commands.append(
            f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
            f"--model grin --mask_path {mask} --seed {seed} --epochs 100 "
            f"--train_batches 300 --patience 20 --batch_size 32 "
            f"--out_dir {RESULT} --tag {tag}"
        )
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()

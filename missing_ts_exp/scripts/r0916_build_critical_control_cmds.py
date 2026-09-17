#!/usr/bin/env python3
"""Build paired Block-T 50% controls required by the final paper revision."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0916_critical_control_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
RESULT = ROOT / "missing_ts_exp/results/r0916_critical_controls"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def command(label: str, seed: int, variant: str, q_mode: str) -> str:
    mask = MASKS / f"block_t_r50_seed{seed}_observed.npy"
    tag = f"{label}_METR-LA_block_t_r50_seed{seed}"
    return (
        f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
        f"--model mrmsgf --mr_variant {variant} --q_mode {q_mode} "
        f"--mask_path {mask} --seed {seed} --epochs 100 --train_batches 300 "
        f"--patience 20 --batch_size 32 --out_dir {RESULT} --tag {tag}"
    )


def main() -> None:
    controls = (
        ("full_fallback", "full", "observed"),
        ("feature_only", "feature_only", "observed"),
        ("full_q_shuffled", "full", "shuffled"),
    )
    commands = [command(label, seed, variant, q_mode)
                for label, variant, q_mode in controls for seed in SEEDS]
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()

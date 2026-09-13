#!/usr/bin/env python3
"""Build the post-Stage-B ablation for the q_ij overlap penalty only."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0910_stage_b_overlap_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def main() -> None:
    commands = []
    for rate in (50, 70):
        for seed in (2024, 2025, 2026, 2027, 2028):
            mask = MASKS / f"block_t_r{rate}_seed{seed}_observed.npy"
            tag = f"full_no_overlap_reliability_METR-LA_block_t_r{rate}_seed{seed}"
            commands.append(
                f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
                f"--model mrmsgf --mr_variant full --disable_overlap_reliability "
                f"--mask_path {mask} --seed {seed} --epochs 100 --train_batches 300 "
                f"--patience 20 --batch_size 32 --out_dir {RESULT} --tag {tag}"
            )
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()

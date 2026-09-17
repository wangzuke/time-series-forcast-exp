#!/usr/bin/env python3
"""Build the 15 pre-registered same-code protocol-closure commands."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0917_protocol_closed_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
REGION = ROOT / "missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
RESULT = ROOT / "missing_ts_exp/results/r0917_protocol_closed"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def command(variant: str, seed: int) -> str:
    mask = MASKS / f"block_t_r50_seed{seed}_observed.npy"
    common = (
        f"--protocol_id r0917_protocol_closed --mask_path {mask} --seed {seed} --mask_seed {seed} "
        f"--epochs 100 --train_batches 300 --patience 20 --batch_size 32 "
        f"--graph_topk 8 --reliability_beta 1.0 --region_assignment_path {REGION} "
        f"--out_dir {RESULT}"
    )
    if variant == "full_observed":
        model = "--model mrmsgf --mr_variant full --q_mode observed"
    elif variant == "feature_only":
        model = "--model official_base_feature_only"
    elif variant == "full_permuted_q":
        model = "--model mrmsgf --mr_variant full --q_mode permuted_pairwise"
    else:
        raise ValueError(variant)
    return (
        f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled {model} {common} "
        f"--tag {variant}_METRLA_blockt50_seed{seed}"
    )


def main() -> None:
    variants = ("full_observed", "feature_only", "full_permuted_q")
    commands = [command(variant, seed) for variant in variants for seed in SEEDS]
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()

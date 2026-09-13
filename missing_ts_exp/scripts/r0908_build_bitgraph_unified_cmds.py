#!/usr/bin/env python3
"""Build the comparable BiTGraph official-architecture Block-T command matrix."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0908_bitgraph_unified_cmds.txt"
MASK_ROOT = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
RESULT = ROOT / "missing_ts_exp/results/r0908_official_baselines/bitgraph_unified_blockt_v2"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
HDTTS_ROOT = ROOT / "external_repro/official/hdtts"


def main() -> None:
    commands = []
    for rate in (50, 70):
        for seed in (2024, 2025, 2026):
            mask = MASK_ROOT / f"block_t_r{rate}_seed{seed}_observed.npy"
            tag = f"bitgraph_unified_METR-LA_block_t_r{rate}_seed{seed}"
            commands.append(
                f"PYTHONPATH={HDTTS_ROOT} {PYTHON} -m "
                "missing_ts_exp.src.training.run_bitgraph_official_unified "
                f"--mask_path {mask} --seed {seed} --epochs 200 --batch_size 32 "
                f"--out_dir {RESULT} --tag {tag}"
            )
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {OUT} with {len(commands)} commands")


if __name__ == "__main__":
    main()

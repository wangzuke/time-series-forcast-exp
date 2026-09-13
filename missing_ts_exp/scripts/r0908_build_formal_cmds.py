#!/usr/bin/env python3
"""Build the formal r0908 command list for the shared GPU worker pool."""
from __future__ import annotations

from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0908_formal_cmds.txt"
RESULT = ROOT / "missing_ts_exp/results/r0908_official_baselines"
MASK_ROOT = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def main() -> None:
    commands: list[str] = []
    for model in ("hd_tts_amp", "hd_tts_imp", "tts_amp", "tts_imp"):
        for seed in (2024, 2025, 2026):
            out = RESULT / "official_hdtts" / model / f"seed{seed}"
            commands.append(
                f"bash missing_ts_exp/scripts/r0908_run_hdtts_official_single.sh {model} {seed} {out}"
            )
    for rate in (50, 70):
        for seed in (2024, 2025, 2026):
            mask = MASK_ROOT / f"block_t_r{rate}_seed{seed}_observed.npy"
            for model in ("official_base", "mrmsgf"):
                tag = f"{model}_METR-LA_block_t_r{rate}_seed{seed}"
                commands.append(
                    f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
                    f"--model {model} --mask_path {mask} --seed {seed} --epochs 100 "
                    f"--train_batches 300 --patience 20 --batch_size 32 "
                    f"--out_dir {RESULT / 'controlled'} --tag {tag}"
                )
    for rate in ("0.3", "0.5", "0.7"):
        commands.append(
            f"bash missing_ts_exp/scripts/r0908_run_bitgraph_official_replicates.sh {rate} "
            f"{RESULT / 'official_bitgraph' / f'point_r{int(float(rate) * 100):02d}'} 200 3"
        )
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {OUT} with {len(commands)} commands")


if __name__ == "__main__":
    main()

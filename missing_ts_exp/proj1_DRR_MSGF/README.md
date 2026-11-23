# DRR-MSGF Experiment Project

This directory is the self-contained working area for the DRR-MSGF research
line. Run commands from the repository root so that the `missing_ts_exp`
package can be resolved.

## Layout

- `src/`: DRR-MSGF, local HD-TTS adapters, and training entry points.
- `scripts/`: mask preparation, command builders, verification, and report
  scripts from the R0902--R0917 experiment history.
- `resources/`: fixed masks, manifests, and physical-region assignments.
- `results/` and `logs/`: archived outputs from completed experiments.
- `docs/`: historical reports and the current experiment plan.

## Shared Dependencies

Datasets remain in the repository-level `dataset/` directory. The untouched
official HD-TTS and BiTGraph repositories remain under
`external_repro/official/`. The legacy missingness injector remains shared at
`missing_ts_exp/src/data/missing.py` because it is also used by other
experiments.

## Current Entry Points

Prepare fixed masks and region assignments:

```bash
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python -m \
  missing_ts_exp.proj1_DRR_MSGF.scripts.r0905_prepare_blockt_resources
```

Run the controlled model entry point:

```bash
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python -m \
  missing_ts_exp.proj1_DRR_MSGF.src.training.run_mrmsgf_official_controlled --help
```

The archived command lists are historical records. Before launching a new
experiment, use `docs/1115实验计划_DRR-MSGF双图融合与区域化验证.md` to create a fresh command list and output folder.

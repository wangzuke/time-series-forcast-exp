#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
RESULT="$ROOT/missing_ts_exp/results/r0916_causal_controls/smoke"
MASK="$ROOT/missing_ts_exp/resources/r0905_blockt/masks/METR-LA/block_t_r50_seed2024_observed.npy"
REGION="$ROOT/missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"

cd "$ROOT"
"$PYTHON" missing_ts_exp/scripts/r0916_causal_control_checks.py
for model in official_base_feature_only mrmsgf; do
  if [[ "$model" == "official_base_feature_only" ]]; then
    tag=official_base_feature_only_smoke_METR-LA_block_t_r50_seed2024
    extra=()
  else
    tag=mrmsgf_permuted_q_smoke_METR-LA_block_t_r50_seed2024
    extra=(--mr_variant full --q_mode permuted_pairwise --mask_seed 2024)
  fi
  "$PYTHON" -m missing_ts_exp.src.training.run_mrmsgf_official_controlled \
    --model "$model" "${extra[@]}" --mask_path "$MASK" --seed 2024 \
    --epochs 1 --train_batches 2 --eval_batches 2 --patience 1 --batch_size 32 \
    --region_assignment_path "$REGION" --out_dir "$RESULT" --tag "$tag"
done

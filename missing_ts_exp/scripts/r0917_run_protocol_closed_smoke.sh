#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
MASK="$ROOT/missing_ts_exp/resources/r0905_blockt/masks/METR-LA/block_t_r50_seed2024_observed.npy"
REGION="$ROOT/missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
OUT="$ROOT/missing_ts_exp/results/r0917_protocol_closed/smoke"

cd "$ROOT"
"$PYTHON" missing_ts_exp/scripts/r0917_protocol_closed_p0.py
for variant in full_observed feature_only full_permuted_q; do
  case "$variant" in
    full_observed) model=(--model mrmsgf --mr_variant full --q_mode observed) ;;
    feature_only) model=(--model official_base_feature_only) ;;
    full_permuted_q) model=(--model mrmsgf --mr_variant full --q_mode permuted_pairwise) ;;
  esac
  "$PYTHON" -m missing_ts_exp.src.training.run_mrmsgf_official_controlled \
    --protocol_id r0917_protocol_closed_smoke "${model[@]}" \
    --mask_path "$MASK" --seed 2024 --mask_seed 2024 \
    --epochs 1 --train_batches 2 --eval_batches 2 --patience 1 --batch_size 32 \
    --graph_topk 8 --reliability_beta 1.0 --region_assignment_path "$REGION" \
    --out_dir "$OUT" --tag "${variant}_smoke_METRLA_blockt50_seed2024"
done

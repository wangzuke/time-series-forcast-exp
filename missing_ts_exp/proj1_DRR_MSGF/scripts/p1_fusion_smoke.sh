#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
PYTHON="/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
PROJECT="$ROOT/missing_ts_exp/proj1_DRR_MSGF"
MASK="$PROJECT/resources/r0905_blockt/masks/METR-LA/block_t_r50_seed2024_observed.npy"
REGION="$PROJECT/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
OUT="$PROJECT/results/p1_fusion_smoke"
LOG="$PROJECT/logs/p1_fusion_smoke"
mkdir -p "$OUT" "$LOG"
cd "$ROOT"

for mode in physical observation fixed_half global_scalar edgewise; do
  echo "[$(date -Is)] smoke start: $mode" | tee "$LOG/$mode.log"
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" -m missing_ts_exp.proj1_DRR_MSGF.src.training.run_mrmsgf_official_controlled \
    --model mrmsgf --protocol_id p1_fusion_smoke --mr_variant fine \
    --fusion_mode "$mode" --mask_path "$MASK" --mask_seed 2024 --seed 2024 \
    --epochs 1 --train_batches 2 --eval_batches 1 --patience 2 --batch_size 8 \
    --graph_topk 8 --reliability_beta 1.0 --region_assignment_path "$REGION" \
    --out_dir "$OUT" --tag "${mode}_METR-LA_block_t_r50_seed2024" \
    >> "$LOG/$mode.log" 2>&1
  echo "[$(date -Is)] smoke done: $mode" | tee -a "$LOG/$mode.log"
done

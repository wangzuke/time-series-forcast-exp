#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
LOG_DIR="$ROOT/missing_ts_exp/results/0718_block_hmbg/raw_logs"

mkdir -p "$LOG_DIR"

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight.log" 2>&1; then
  echo "[r0718 smoke] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight.log"
  exit 2
fi

cd "$ROOT/external_repro/BiTGraph"
CUDA_VISIBLE_DEVICES=1 conda run -n bitgraph python main.py \
  --epochs 2 \
  --batch_size 128 \
  --seq_len 24 \
  --pred_len 24 \
  --horizon 24 \
  --dataset Metr \
  --dataset-name Metr \
  --mask_ratio 0.5 \
  --missing_type temporal_block \
  --block_len 12 \
  --mask_seed 2024 \
  > "$LOG_DIR/smoke_bitgraph_metr_blockt_b128.log" 2>&1

CUDA_VISIBLE_DEVICES=1 conda run -n bitgraph python test_forecasting.py \
  --batch_size 128 \
  --seq_len 24 \
  --pred_len 24 \
  --horizon 24 \
  --dataset Metr \
  --dataset-name Metr \
  --mask_ratio 0.5 \
  --missing_type temporal_block \
  --block_len 12 \
  --mask_seed 2024 \
  >> "$LOG_DIR/smoke_bitgraph_metr_blockt_b128.log" 2>&1

cd "$ROOT/external_repro/hdtts"
CUDA_VISIBLE_DEVICES=3 conda run -n hd-tts python run_realworld_patched.py \
  model=hd_tts_amp \
  dataset=la \
  dataset/mode=block_t_50 \
  window=24 \
  horizon=24 \
  batch_size=128 \
  epochs=2 \
  train_batches=5 \
  patience=2 \
  > "$LOG_DIR/smoke_hdtts_la_blockt_h24_b128.log" 2>&1

cd "$ROOT"
python missing_ts_exp/scripts/r0718_monitor.py --tail-lines 30

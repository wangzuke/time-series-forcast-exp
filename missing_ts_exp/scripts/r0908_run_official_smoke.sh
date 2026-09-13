#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
HDTTS_ROOT="$ROOT/external_repro/official/hdtts"
BITGRAPH_ROOT="$ROOT/external_repro/official/bitgraph"
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
RESULT_ROOT="$ROOT/missing_ts_exp/results/r0908_official_baselines/smoke"
LOG_ROOT="$ROOT/missing_ts_exp/logs/r0908_official_baselines/smoke"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

run_hdtts() {
  local gpu=$1
  local model=$2
  local run_dir="$RESULT_ROOT/hdtts_${model}"
  (
    cd "$HDTTS_ROOT"
    PYTHONPATH="$HDTTS_ROOT${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/mpl-r0908 "$PYTHON" \
      experiments/run_realworld.py \
      "model=$model" dataset=la dataset/mode=block_t \
      epochs=2 train_batches=5 patience=2 batch_size=32 workers=0 \
      "hydra.run.dir=$run_dir"
  ) >"$LOG_ROOT/hdtts_${model}.log" 2>&1
}

run_bitgraph() {
  local gpu=$1
  (
    cd "$BITGRAPH_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/mpl-r0908 "$PYTHON" main.py \
      --epochs 2 --dataset Metr --dataset-name Metr --mask_ratio 0.5 \
      --seq_len 24 --pred_len 24 --horizon 24 \
      >"$LOG_ROOT/bitgraph_metr_point_r50.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/mpl-r0908 "$PYTHON" test_forecasting.py \
      --epochs 2 --dataset Metr --dataset-name Metr --mask_ratio 0.5 \
      --seq_len 24 --pred_len 24 --horizon 24 \
      >>"$LOG_ROOT/bitgraph_metr_point_r50.log" 2>&1
  )
}

run_hdtts 0 hd_tts_amp &
pid_amp=$!
run_hdtts 1 hd_tts_imp &
pid_imp=$!
run_hdtts 2 tts_amp &
pid_tts_amp=$!
run_hdtts 3 tts_imp &
pid_tts_imp=$!
run_bitgraph 4 &
pid_bitgraph=$!

status=0
for pid in "$pid_amp" "$pid_imp" "$pid_tts_amp" "$pid_tts_imp" "$pid_bitgraph"; do
  wait "$pid" || status=1
done

exit "$status"

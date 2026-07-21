#!/usr/bin/env bash
set -euo pipefail

# Runner for the 0718 block-missing baseline reruns.
#
# This script intentionally refuses to start if nvidia-smi is unavailable,
# because HD-TTS silently falls back to CPU when torch.cuda.is_available() is
# false.  For this experiment we want GPU-only execution on the A800 cards.
#
# Usage:
#   bash missing_ts_exp/scripts/r0718_run_baselines.sh batch_probe
#   bash missing_ts_exp/scripts/r0718_run_baselines.sh full
#
# Optional environment overrides:
#   R0718_BATCH=256
#   R0718_EPOCHS=200
#   R0718_HDTTS_TRAIN_BATCHES=300
#   R0718_HDTTS_PATIENCE=30

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0718_block_hmbg"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
CKPT_DIR="$RESULTS/checkpoints"
MODE="${1:-batch_probe}"

mkdir -p "$LOG_DIR" "$PID_DIR" "$CKPT_DIR"

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight_baselines.log" 2>&1; then
  echo "[r0718 baselines] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight_baselines.log"
  exit 2
fi

if [[ "$MODE" != "batch_probe" && "$MODE" != "full" ]]; then
  echo "Unknown mode: $MODE"
  echo "Expected one of: batch_probe, full"
  exit 64
fi

GPUS=(1 2 3 4 5 6 7)
free_gpus=("${GPUS[@]}")
declare -A pid_gpu
failed_jobs=0

# Waits until at least one GPU is free, reclaiming the GPU held by whichever
# background job exits first (not necessarily the oldest one). This avoids
# double-booking a GPU while another one sits idle, which a naive
# round-robin-by-launch-count scheme can do once jobs finish out of order.
acquire_gpu_wait() {
  while (( ${#free_gpus[@]} == 0 )); do
    local done_pid
    if wait -n -p done_pid; then
      :
    else
      failed_jobs=$((failed_jobs + 1))
    fi
    free_gpus+=("${pid_gpu[$done_pid]}")
    unset 'pid_gpu[$done_pid]'
  done
}

launch_job() {
  local workdir="$1"
  local name="$2"
  shift 2

  acquire_gpu_wait
  local gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")
  local log="$LOG_DIR/${name}.log"

  echo "[r0718 baselines] launch gpu=${gpu} name=${name} cwd=${workdir}"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$gpu" "$@"
  ) > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$PID_DIR/${name}.pid"
  pid_gpu[$pid]="$gpu"
}

run_bitgraph() {
  local dataset="$1"
  local ratio="$2"
  local batch="$3"
  local epochs="$4"
  local ratio_tag="${ratio/./}"
  local name="bitgraph_${dataset}_blockt_r${ratio_tag}_h24_b${batch}_e${epochs}"

  launch_job "$ROOT/external_repro/BiTGraph" "$name" \
    bash -c '
      conda run -n bitgraph python main.py \
        --epochs "$1" --batch_size "$2" --seq_len 24 --pred_len 24 --horizon 24 \
        --dataset "$3" --dataset-name "$3" --mask_ratio "$4" \
        --missing_type temporal_block --block_len 12 --mask_seed 2024 \
      && conda run -n bitgraph python test_forecasting.py \
        --batch_size "$2" --seq_len 24 --pred_len 24 --horizon 24 \
        --dataset "$3" --dataset-name "$3" --mask_ratio "$4" \
        --missing_type temporal_block --block_len 12 --mask_seed 2024
    ' _ "$epochs" "$batch" "$dataset" "$ratio"
}

run_hdtts() {
  local dataset="$1"
  local mode="$2"
  local batch="$3"
  local epochs="$4"
  local train_batches="$5"
  local patience="$6"
  local name="hdtts_${dataset}_${mode}_h24_b${batch}_e${epochs}"
  local run_dir="$CKPT_DIR/$name"

  launch_job "$ROOT/external_repro/hdtts" "$name" \
    conda run -n hd-tts python run_realworld_patched.py \
      model=hd_tts_amp \
      dataset="$dataset" \
      dataset/mode="$mode" \
      window=24 \
      horizon=24 \
      batch_size="$batch" \
      epochs="$epochs" \
      train_batches="$train_batches" \
      patience="$patience" \
      hydra.run.dir="$run_dir"
}

if [[ "$MODE" == "batch_probe" ]]; then
  EPOCHS=${R0718_EPOCHS:-5}
  HDTTS_TRAIN_BATCHES=${R0718_HDTTS_TRAIN_BATCHES:-50}
  HDTTS_PATIENCE=${R0718_HDTTS_PATIENCE:-5}

  for batch in 64 128 256 512; do
    run_bitgraph Metr 0.5 "$batch" "$EPOCHS"
    run_hdtts la block_t_50 "$batch" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
  done
else
  BITGRAPH_BATCH=${R0718_BITGRAPH_BATCH:-256}
  HDTTS_BATCH=${R0718_HDTTS_BATCH:-128}
  EPOCHS=${R0718_EPOCHS:-200}
  HDTTS_TRAIN_BATCHES=${R0718_HDTTS_TRAIN_BATCHES:-300}
  HDTTS_PATIENCE=${R0718_HDTTS_PATIENCE:-30}

  for ratio in 0.5 0.6 0.7 0.8; do
    run_bitgraph Metr "$ratio" "$BITGRAPH_BATCH" "$EPOCHS"
    run_bitgraph PEMS "$ratio" "$BITGRAPH_BATCH" "$EPOCHS"
  done

  for mode in block_t_50 block_t_60 block_t_70 block_t_80; do
    run_hdtts la "$mode" "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
    run_hdtts bay "$mode" "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
  done
fi

while (( ${#pid_gpu[@]} > 0 )); do
  done_pid=""
  if wait -n -p done_pid; then
    :
  else
    failed_jobs=$((failed_jobs + 1))
  fi
  unset 'pid_gpu[$done_pid]'
done

echo "[r0718 baselines] all ${MODE} jobs finished; failed_jobs=${failed_jobs}"
python "$ROOT/missing_ts_exp/scripts/r0718_monitor.py" --tail-lines 30

if (( failed_jobs > 0 )); then
  exit 1
fi

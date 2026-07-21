#!/usr/bin/env bash
set -euo pipefail

# Phase 2b runner: extends baseline coverage (BiTGraph + HD-TTS-AMP only,
# NOT the fusion model -- that is Phase 4) to the rest of the section 7.1/7.2
# main-experiment matrix that Phase 2 (r0718_run_baselines.sh full) didn't
# cover:
#   - block_t   0.3            (BiTGraph + HD-TTS-AMP, both datasets)
#   - block_st  0.3/0.5/0.6/0.7/0.8  (HD-TTS-AMP only; BiTGraph has no
#                                      spatiotemporal_block support -- see
#                                      實驗計劃0718.md 7.1's conditional note)
#   - random_point 0.5/0.7/0.8  (BiTGraph + HD-TTS-AMP, both datasets)
#
# Must be run AFTER r0718_run_baselines.sh full has released GPUs 1-7 --
# this script does its own independent GPU bookkeeping starting from
# "all of 1-7 free", so launching it while Phase 2 jobs are still running
# would double-book GPUs.
#
# Usage:
#   bash missing_ts_exp/scripts/r0718_run_baselines_2b.sh
#
# Optional environment overrides:
#   R0718_BITGRAPH_BATCH=256
#   R0718_HDTTS_BATCH=128
#   R0718_EPOCHS=200
#   R0718_HDTTS_TRAIN_BATCHES=300
#   R0718_HDTTS_PATIENCE=30

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0718_block_hmbg"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
CKPT_DIR="$RESULTS/checkpoints"

mkdir -p "$LOG_DIR" "$PID_DIR" "$CKPT_DIR"

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight_baselines_2b.log" 2>&1; then
  echo "[r0718 baselines_2b] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight_baselines_2b.log"
  exit 2
fi

GPUS=(1 2 3 4 5 6 7)
free_gpus=("${GPUS[@]}")
declare -A pid_gpu
failed_jobs=0

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

  echo "[r0718 baselines_2b] launch gpu=${gpu} name=${name} cwd=${workdir}"
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
  local missing_type="$2"
  local ratio="$3"
  local batch="$4"
  local epochs="$5"
  local ratio_tag="${ratio/./}"
  local type_tag="$6"
  local name="bitgraph_${dataset}_${type_tag}_r${ratio_tag}_h24_b${batch}_e${epochs}"

  launch_job "$ROOT/external_repro/BiTGraph" "$name" \
    bash -c '
      conda run -n bitgraph python main.py \
        --epochs "$1" --batch_size "$2" --seq_len 24 --pred_len 24 --horizon 24 \
        --dataset "$3" --dataset-name "$3" --mask_ratio "$4" \
        --missing_type "$5" --block_len 12 --mask_seed 2024 \
      && conda run -n bitgraph python test_forecasting.py \
        --batch_size "$2" --seq_len 24 --pred_len 24 --horizon 24 \
        --dataset "$3" --dataset-name "$3" --mask_ratio "$4" \
        --missing_type "$5" --block_len 12 --mask_seed 2024
    ' _ "$epochs" "$batch" "$dataset" "$ratio" "$missing_type"
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

BITGRAPH_BATCH=${R0718_BITGRAPH_BATCH:-256}
HDTTS_BATCH=${R0718_HDTTS_BATCH:-128}
EPOCHS=${R0718_EPOCHS:-200}
HDTTS_TRAIN_BATCHES=${R0718_HDTTS_TRAIN_BATCHES:-300}
HDTTS_PATIENCE=${R0718_HDTTS_PATIENCE:-30}

# --- block_t 0.3 (BiTGraph + HD-TTS-AMP, both datasets) ---
run_bitgraph Metr temporal_block 0.3 "$BITGRAPH_BATCH" "$EPOCHS" blockt
run_bitgraph PEMS temporal_block 0.3 "$BITGRAPH_BATCH" "$EPOCHS" blockt
run_hdtts la block_t_30 "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
run_hdtts bay block_t_30 "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"

# --- block_st 0.3/0.5/0.6/0.7/0.8 (HD-TTS-AMP only) ---
for rate in 30 50 60 70 80; do
  run_hdtts la "block_st_${rate}" "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
  run_hdtts bay "block_st_${rate}" "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
done

# --- random_point 0.5/0.7/0.8 (BiTGraph + HD-TTS-AMP, both datasets) ---
for ratio in 0.5 0.7 0.8; do
  run_bitgraph Metr random_point "$ratio" "$BITGRAPH_BATCH" "$EPOCHS" point
  run_bitgraph PEMS random_point "$ratio" "$BITGRAPH_BATCH" "$EPOCHS" point
done
for rate in 50 70 80; do
  run_hdtts la "point_${rate}" "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
  run_hdtts bay "point_${rate}" "$HDTTS_BATCH" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
done

while (( ${#pid_gpu[@]} > 0 )); do
  done_pid=""
  if wait -n -p done_pid; then
    :
  else
    failed_jobs=$((failed_jobs + 1))
  fi
  unset 'pid_gpu[$done_pid]'
done

echo "[r0718 baselines_2b] all jobs finished; failed_jobs=${failed_jobs}"
python "$ROOT/missing_ts_exp/scripts/r0718_monitor.py" --tail-lines 30

if (( failed_jobs > 0 )); then
  exit 1
fi

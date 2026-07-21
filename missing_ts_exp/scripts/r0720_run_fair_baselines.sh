#!/usr/bin/env bash
set -euo pipefail

# 0720 strictly fair baseline runner.
#
# Run from a normal terminal with GPU visibility:
#   cd /data/wangzuke/time-series-forecast-exp
#   bash missing_ts_exp/scripts/r0720_run_fair_baselines.sh prepare
#   bash missing_ts_exp/scripts/r0720_run_fair_baselines.sh smoke
#   bash missing_ts_exp/scripts/r0720_run_fair_baselines.sh full
#
# Matrix in full mode:
#   datasets: Metr, PEMS
#   missing: point, block
#   rates:   30%, 50%, 70%
#   models:  BiTGraph, HD-TTS-AMP
#   batch:   512

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0720_fair_b512"
FAIR_DIR="$RESULTS/fair_data"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
CKPT_DIR="$RESULTS/checkpoints"
MODE="${1:-smoke}"

mkdir -p "$FAIR_DIR" "$LOG_DIR" "$PID_DIR" "$CKPT_DIR" "$RESULTS/csv" "$RESULTS/notes"

run_prepare() {
  conda run -n hd-tts python "$ROOT/missing_ts_exp/scripts/r0720_prepare_fair_data.py"
}

if [[ "$MODE" == "prepare" ]]; then
  run_prepare
  exit 0
fi

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Unknown mode: $MODE"
  echo "Expected one of: prepare, smoke, full"
  exit 64
fi

run_prepare

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight_${MODE}.log" 2>&1; then
  echo "[r0720 fair] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight_${MODE}.log"
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

  echo "[r0720 fair] launch gpu=${gpu} name=${name} cwd=${workdir}"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$gpu" "$@"
  ) > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$PID_DIR/${name}.pid"
  pid_gpu[$pid]="$gpu"
}

data_path() {
  case "$1" in
    Metr) echo "$ROOT/dataset/metr_la/metr_la.h5" ;;
    PEMS) echo "$ROOT/dataset/pems_bay/pems_bay.h5" ;;
    *) echo "unknown dataset $1" >&2; exit 64 ;;
  esac
}

mask_path() {
  local dataset="$1"
  local missing="$2"
  local rate="$3"
  echo "$FAIR_DIR/mask_observed_${dataset}_${missing}_r${rate}_seed2024.npy"
}

bitgraph_missing_type() {
  case "$1" in
    point) echo "random_point" ;;
    block) echo "temporal_block" ;;
    *) echo "unknown missing $1" >&2; exit 64 ;;
  esac
}

run_bitgraph() {
  local dataset="$1"
  local missing="$2"
  local rate="$3"
  local epochs="$4"
  local ratio="0.${rate:0:1}"
  local bg_missing
  bg_missing="$(bitgraph_missing_type "$missing")"
  local name="bitgraph_${dataset}_${missing}_r${rate}_h24_b512_e${epochs}"
  local out_dir="$CKPT_DIR/$name"

  launch_job "$ROOT/external_repro/BiTGraph" "$name" \
    bash -c '
      conda run -n bitgraph python main.py \
        --epochs "$1" --batch_size 512 \
        --seq_len 24 --pred_len 24 --horizon 24 \
        --dataset "$2" --dataset-name "$2" \
        --mask_ratio "$3" --missing_type "$4" --block_len 12 --mask_seed 2024 \
        --data_path "$5" --mask_path "$6" --output_dir "$7" \
      && conda run -n bitgraph python test_forecasting.py \
        --batch_size 512 \
        --seq_len 24 --pred_len 24 --horizon 24 \
        --dataset "$2" --dataset-name "$2" \
        --mask_ratio "$3" --missing_type "$4" --block_len 12 --mask_seed 2024 \
        --data_path "$5" --mask_path "$6" --output_dir "$7"
    ' _ "$epochs" "$dataset" "$ratio" "$bg_missing" \
      "$(data_path "$dataset")" "$(mask_path "$dataset" "$missing" "$rate")" "$out_dir"
}

run_hdtts() {
  local dataset="$1"
  local missing="$2"
  local rate="$3"
  local epochs="$4"
  local train_batches="$5"
  local patience="$6"
  local name="hdtts_amp_${dataset}_${missing}_r${rate}_h24_b512_e${epochs}"
  local run_dir="$CKPT_DIR/$name"

  launch_job "$ROOT/external_repro/hdtts" "$name" \
    conda run -n hd-tts python run_fair_h5_patched.py \
      --dataset "$dataset" \
      --data_path "$(data_path "$dataset")" \
      --mask_path "$(mask_path "$dataset" "$missing" "$rate")" \
      --run_dir "$run_dir" \
      --window 24 \
      --horizon 24 \
      --batch_size 512 \
      --epochs "$epochs" \
      --train_batches "$train_batches" \
      --patience "$patience" \
      --seed 2024
}

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=${R0720_EPOCHS:-2}
  HDTTS_TRAIN_BATCHES=${R0720_HDTTS_TRAIN_BATCHES:-5}
  HDTTS_PATIENCE=${R0720_HDTTS_PATIENCE:-2}
  run_bitgraph Metr point 30 "$EPOCHS"
  run_hdtts Metr point 30 "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
else
  EPOCHS=${R0720_EPOCHS:-200}
  HDTTS_TRAIN_BATCHES=${R0720_HDTTS_TRAIN_BATCHES:-300}
  HDTTS_PATIENCE=${R0720_HDTTS_PATIENCE:-30}
  for dataset in Metr PEMS; do
    for missing in point block; do
      for rate in 30 50 70; do
        run_bitgraph "$dataset" "$missing" "$rate" "$EPOCHS"
        run_hdtts "$dataset" "$missing" "$rate" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
      done
    done
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

echo "[r0720 fair] all ${MODE} jobs finished; failed_jobs=${failed_jobs}"
python "$ROOT/missing_ts_exp/scripts/r0720_collect_results.py"

if (( failed_jobs > 0 )); then
  exit 1
fi

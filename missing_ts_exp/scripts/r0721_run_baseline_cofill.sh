#!/usr/bin/env bash
set -euo pipefail

# 0721 baseline / CoFILL runner.
#
# Run from a normal terminal with GPU visibility:
#   cd /data/wangzuke/time-series-forecast-exp
#   bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh validate
#   bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh smoke
#   bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_full
#   bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_key_seeds
#   bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh collect
#
# Default scheduling:
#   R0721_GPUS="1 2 3 4 5 6 7"  # GPU 0 is intentionally excluded.
#   R0721_SLOTS_PER_GPU=2         # Use two concurrent runs per 80G A800 by default.
#
# CoFILL is optional and requires an actual local implementation:
#   R0721_COFILL_RUNNER=/abs/path/to/cofill_runner.sh \
#     bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh cofill_imputation

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0721_cofill_pguts_forecasting"
MASK_DIR="$ROOT/dataset/0721_missing_masks"
LOG_DIR="$RESULTS/raw_logs/baseline_cofill"
PID_DIR="$RESULTS/pids/baseline_cofill"
CKPT_DIR="$RESULTS/checkpoints/baseline_cofill"
CSV_DIR="$RESULTS/csv"
NOTES_DIR="$RESULTS/notes"
MODE="${1:-smoke}"

mkdir -p "$LOG_DIR" "$PID_DIR" "$CKPT_DIR" "$CSV_DIR" "$NOTES_DIR" "$RESULTS/predictions" "$RESULTS/figures"

validate_assets() {
  conda run -n hd-tts python "$ROOT/missing_ts_exp/scripts/r0721_validate_baseline_cofill_assets.py"
}

collect_results() {
  python "$ROOT/missing_ts_exp/scripts/r0721_collect_baseline_cofill.py"
}

write_task_manifest() {
  python - <<'PY'
import csv
import pathlib

root = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
out = root / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting" / "csv" / "baseline_cofill_task_manifest.csv"
out.parent.mkdir(parents=True, exist_ok=True)
rows = []
mask_types = [("point", [50, 70]), ("block_t", [50, 70, 90]), ("block_st", [50, 70, 90])]
for model in ["BiTGraph", "HD-TTS-AMP"]:
    for dataset in ["Metr", "PEMS"]:
        for mask_type, rates in mask_types:
            for rate in rates:
                for horizon in [12, 24]:
                    rows.append({
                        "phase": "baseline_full",
                        "model": model,
                        "dataset": dataset,
                        "mask_type": mask_type,
                        "target_missing_rate": f"0.{rate:02d}",
                        "T_in": 24,
                        "T_out": horizon,
                        "seed": 1,
                        "batch_size": 512,
                    })
for model in ["BiTGraph", "HD-TTS-AMP"]:
    for dataset in ["Metr", "PEMS"]:
        for mask_type in ["block_t", "block_st"]:
            for rate in [70, 90]:
                for seed in [2, 3]:
                    rows.append({
                        "phase": "baseline_key_seeds",
                        "model": model,
                        "dataset": dataset,
                        "mask_type": mask_type,
                        "target_missing_rate": f"0.{rate:02d}",
                        "T_in": 24,
                        "T_out": 24,
                        "seed": seed,
                        "batch_size": 512,
                    })
for dataset in ["Metr", "PEMS"]:
    for mask_type in ["block_t", "block_st"]:
        rows.append({
            "phase": "cofill_imputation",
            "model": "CoFILL",
            "dataset": dataset,
            "mask_type": mask_type,
            "target_missing_rate": "0.70",
            "T_in": 24,
            "T_out": "",
            "seed": 1,
            "batch_size": 512,
        })
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out} rows={len(rows)}")
PY
}

if [[ "$MODE" == "validate" ]]; then
  validate_assets
  write_task_manifest
  exit 0
fi

if [[ "$MODE" == "collect" ]]; then
  collect_results
  exit 0
fi

case "$MODE" in
  smoke|baseline_full|baseline_key_seeds|cofill_imputation|status)
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Expected: validate, smoke, baseline_full, baseline_key_seeds, cofill_imputation, collect, status"
    exit 64
    ;;
esac

if [[ "$MODE" == "status" ]]; then
  echo "[r0721] logs:"
  find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' | sort
  echo "[r0721] csv:"
  find "$CSV_DIR" -maxdepth 1 -type f -name '*.csv' | sort
  collect_results
  exit 0
fi

validate_assets
write_task_manifest

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight_${MODE}.log" 2>&1; then
  echo "[r0721 baseline_cofill] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight_${MODE}.log"
  exit 2
fi

GPUS_TEXT="${R0721_GPUS:-1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${R0721_SLOTS_PER_GPU:-2}"
read -r -a GPUS <<< "$GPUS_TEXT"
free_slots=()
for gpu in "${GPUS[@]}"; do
  for ((slot=1; slot<=SLOTS_PER_GPU; slot++)); do
    free_slots+=("${gpu}:${slot}")
  done
done
declare -A pid_slot
failed_jobs=0

acquire_gpu_wait() {
  while (( ${#free_slots[@]} == 0 )); do
    local done_pid
    if wait -n -p done_pid; then
      :
    else
      failed_jobs=$((failed_jobs + 1))
    fi
    free_slots+=("${pid_slot[$done_pid]}")
    unset 'pid_slot[$done_pid]'
  done
}

launch_job() {
  local workdir="$1"
  local name="$2"
  shift 2

  acquire_gpu_wait
  local resource="${free_slots[0]}"
  free_slots=("${free_slots[@]:1}")
  local gpu="${resource%%:*}"
  local slot="${resource##*:}"
  local log="$LOG_DIR/${name}.log"

  echo "[r0721 baseline_cofill] launch gpu=${gpu} slot=${slot}/${SLOTS_PER_GPU} name=${name} cwd=${workdir}"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$gpu" "$@"
  ) > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$PID_DIR/${name}.pid"
  pid_slot[$pid]="$resource"
}

wait_all_jobs() {
  while (( ${#pid_slot[@]} > 0 )); do
    local done_pid
    if wait -n -p done_pid; then
      :
    else
      failed_jobs=$((failed_jobs + 1))
    fi
    unset 'pid_slot[$done_pid]'
  done
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
  echo "$MASK_DIR/mask_observed_${dataset}_${missing}_r${rate}_seed2024.npy"
}

bitgraph_missing_type() {
  case "$1" in
    point) echo "random_point" ;;
    block_t|block_st) echo "temporal_block" ;;
    *) echo "unknown missing $1" >&2; exit 64 ;;
  esac
}

run_bitgraph() {
  local dataset="$1"
  local missing="$2"
  local rate="$3"
  local horizon="$4"
  local seed="$5"
  local epochs="$6"
  local ratio
  ratio="0.${rate}"
  local bg_missing
  bg_missing="$(bitgraph_missing_type "$missing")"
  local name="bitgraph_${dataset}_${missing}_r${rate}_h${horizon}_b512_e${epochs}_s${seed}"
  local out_dir="$CKPT_DIR/$name"

  launch_job "$ROOT/external_repro/BiTGraph" "$name" \
    bash -c '
      conda run -n bitgraph python main.py \
        --epochs "$1" --batch_size 512 --seed "$2" \
        --seq_len 24 --pred_len "$3" --horizon "$3" \
        --dataset "$4" --dataset-name "$4" \
        --mask_ratio "$5" --missing_type "$6" --block_len 12 --mask_seed 2024 \
        --data_path "$7" --mask_path "$8" --output_dir "$9" \
      && conda run -n bitgraph python test_forecasting.py \
        --batch_size 512 --seed "$2" \
        --seq_len 24 --pred_len "$3" --horizon "$3" \
        --dataset "$4" --dataset-name "$4" \
        --mask_ratio "$5" --missing_type "$6" --block_len 12 --mask_seed 2024 \
        --data_path "$7" --mask_path "$8" --output_dir "$9"
    ' _ "$epochs" "$seed" "$horizon" "$dataset" "$ratio" "$bg_missing" \
      "$(data_path "$dataset")" "$(mask_path "$dataset" "$missing" "$rate")" "$out_dir"
}

run_hdtts() {
  local dataset="$1"
  local missing="$2"
  local rate="$3"
  local horizon="$4"
  local seed="$5"
  local epochs="$6"
  local train_batches="$7"
  local patience="$8"
  local name="hdtts_amp_${dataset}_${missing}_r${rate}_h${horizon}_b512_e${epochs}_s${seed}"
  local run_dir="$CKPT_DIR/$name"

  launch_job "$ROOT/external_repro/hdtts" "$name" \
    conda run -n hd-tts python run_fair_h5_patched.py \
      --dataset "$dataset" \
      --data_path "$(data_path "$dataset")" \
      --mask_path "$(mask_path "$dataset" "$missing" "$rate")" \
      --run_dir "$run_dir" \
      --window 24 \
      --horizon "$horizon" \
      --batch_size 512 \
      --epochs "$epochs" \
      --train_batches "$train_batches" \
      --patience "$patience" \
      --seed "$seed"
}

run_cofill_imputation() {
  local dataset="$1"
  local missing="$2"
  local rate="$3"
  local seed="$4"
  local runner="${R0721_COFILL_RUNNER:-}"
  local name="cofill_${dataset}_${missing}_r${rate}_h0_b512_s${seed}"
  local run_dir="$CKPT_DIR/$name"

  if [[ -z "$runner" || ! -x "$runner" ]]; then
    local log="$LOG_DIR/${name}.log"
    {
      echo "[r0721 cofill] missing executable CoFILL runner."
      echo "[r0721 cofill] Set R0721_COFILL_RUNNER to an executable script that accepts:"
      echo "  --dataset --data_path --mask_path --run_dir --batch_size --seed --task imputation"
      echo "[r0721 cofill] dataset=${dataset} missing=${missing} rate=${rate} seed=${seed}"
      echo "[r0721 cofill] data_path=$(data_path "$dataset")"
      echo "[r0721 cofill] mask_path=$(mask_path "$dataset" "$missing" "$rate")"
    } > "$log"
    echo "[r0721 cofill] skipped ${name}; see ${log}"
    return 0
  fi

  mkdir -p "$run_dir"
  launch_job "$ROOT" "$name" \
    "$runner" \
      --dataset "$dataset" \
      --data_path "$(data_path "$dataset")" \
      --mask_path "$(mask_path "$dataset" "$missing" "$rate")" \
      --run_dir "$run_dir" \
      --batch_size 512 \
      --seed "$seed" \
      --task imputation
}

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=${R0721_EPOCHS:-2}
  HDTTS_TRAIN_BATCHES=${R0721_HDTTS_TRAIN_BATCHES:-5}
  HDTTS_PATIENCE=${R0721_HDTTS_PATIENCE:-2}
  run_bitgraph Metr block_st 70 12 1 "$EPOCHS"
  run_hdtts Metr block_st 70 12 1 "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
elif [[ "$MODE" == "baseline_full" ]]; then
  EPOCHS=${R0721_EPOCHS:-200}
  HDTTS_TRAIN_BATCHES=${R0721_HDTTS_TRAIN_BATCHES:-300}
  HDTTS_PATIENCE=${R0721_HDTTS_PATIENCE:-30}
  for dataset in Metr PEMS; do
    for missing in point block_t block_st; do
      if [[ "$missing" == "point" ]]; then
        rates=(50 70)
      else
        rates=(50 70 90)
      fi
      for rate in "${rates[@]}"; do
        for horizon in 12 24; do
          run_bitgraph "$dataset" "$missing" "$rate" "$horizon" 1 "$EPOCHS"
          run_hdtts "$dataset" "$missing" "$rate" "$horizon" 1 "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
        done
      done
    done
  done
elif [[ "$MODE" == "baseline_key_seeds" ]]; then
  EPOCHS=${R0721_EPOCHS:-200}
  HDTTS_TRAIN_BATCHES=${R0721_HDTTS_TRAIN_BATCHES:-300}
  HDTTS_PATIENCE=${R0721_HDTTS_PATIENCE:-30}
  for dataset in Metr PEMS; do
    for missing in block_t block_st; do
      for rate in 70 90; do
        for seed in 2 3; do
          run_bitgraph "$dataset" "$missing" "$rate" 24 "$seed" "$EPOCHS"
          run_hdtts "$dataset" "$missing" "$rate" 24 "$seed" "$EPOCHS" "$HDTTS_TRAIN_BATCHES" "$HDTTS_PATIENCE"
        done
      done
    done
  done
elif [[ "$MODE" == "cofill_imputation" ]]; then
  for dataset in Metr PEMS; do
    for missing in block_t block_st; do
      run_cofill_imputation "$dataset" "$missing" 70 1
    done
  done
fi

wait_all_jobs
echo "[r0721 baseline_cofill] all ${MODE} jobs finished; failed_jobs=${failed_jobs}"
collect_results

if (( failed_jobs > 0 )); then
  exit 1
fi

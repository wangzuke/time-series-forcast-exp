#!/usr/bin/env bash
set -euo pipefail

# 0721 P-GUTS / HD-PGUTS runner.
#
# Examples:
#   cd /data/wangzuke/time-series-forecast-exp
#   bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh prepare
#   bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh smoke
#   bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase2
#   bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase3
#   bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh cmdfile missing_ts_exp/scripts/r0721_pguts_phase2_cmds.txt
#
# Resource controls:
#   R0721_GPUS='0 1 2 3 4 5 6 7'     visible GPU ids for this launcher
#   R0721_SLOTS_PER_GPU=2             concurrent runs per GPU
#   R0721_BATCH_SIZE=512              per-run batch size in generated commands
#   R0721_NUM_WORKERS=2               per-run DataLoader workers

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0721_cofill_pguts_forecasting"
LOG_DIR="$RESULTS/raw_logs/pguts_hdpguts"
PID_DIR="$RESULTS/pids/pguts_hdpguts"
MODE="${1:-smoke}"

mkdir -p "$LOG_DIR" "$PID_DIR" "$RESULTS/checkpoints/pguts_hdpguts" \
  "$RESULTS/csv" "$RESULTS/notes" "$RESULTS/predictions" \
  "$RESULTS/metrics/pguts_hdpguts" "$RESULTS/diagnostics/pguts_hdpguts"

cd "$ROOT"

prepare() {
  test -f "$ROOT/dataset/0721_missing_masks/manifest.csv"
  python "$ROOT/missing_ts_exp/scripts/r0721_build_pguts_cmds.py" \
    --smoke_epochs "${R0721_SMOKE_EPOCHS:-5}" \
    --epochs "${R0721_EPOCHS:-100}"
  if [[ "${R0721_SKIP_ENV_CHECK:-0}" == "1" ]]; then
    echo "[r0721 pguts] skip env check because R0721_SKIP_ENV_CHECK=1"
    return 0
  fi
  local python_cmd="${R0721_PYTHON_CMD:-python}"
  local h5_flag=""
  if [[ "${R0721_SKIP_H5_ENV_CHECK:-0}" == "1" ]]; then
    h5_flag="--skip_h5_read"
  fi
  bash -lc "$python_cmd '$ROOT/missing_ts_exp/scripts/r0721_check_pguts_env.py' $h5_flag"
}

if [[ "$MODE" == "prepare" ]]; then
  prepare
  exit 0
fi

prepare

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight_${MODE}.log" 2>&1; then
  echo "[r0721 pguts] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight_${MODE}.log"
  exit 2
fi

case "$MODE" in
  smoke) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0721_pguts_smoke_cmds.txt" ;;
  phase2) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0721_pguts_phase2_cmds.txt" ;;
  phase3) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0721_hdpguts_phase3_cmds.txt" ;;
  phase4) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0721_pguts_phase4_cmds.txt" ;;
  cmdfile)
    if [[ $# -lt 2 ]]; then
      echo "cmdfile mode requires a command-file path"
      exit 64
    fi
    CMD_FILE="$2"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Expected one of: prepare, smoke, phase2, phase3, phase4, cmdfile"
    exit 64
    ;;
esac

GPUS_STR="${R0721_GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${R0721_SLOTS_PER_GPU:-1}"
read -r -a GPUS <<< "$GPUS_STR"
if (( SLOTS_PER_GPU < 1 )); then
  echo "R0721_SLOTS_PER_GPU must be >= 1"
  exit 64
fi

free_gpus=()
for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
  for gpu in "${GPUS[@]}"; do
    free_gpus+=("$gpu")
  done
done
declare -A pid_gpu
failed_jobs=0
echo "[r0721 pguts] scheduler gpus=${GPUS_STR} slots_per_gpu=${SLOTS_PER_GPU} max_concurrent=${#free_gpus[@]}"

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

run_name_from_cmd() {
  python - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1])
def val(flag, default=""):
    try:
        return parts[parts.index(flag) + 1]
    except ValueError:
        return default
dataset = val("--dataset")
mask = val("--mask_type")
rate = int(round(float(val("--missing_rate", "0")) * 100))
h = val("--T_out")
pool = val("--pooling_factors").replace(",", "-")
model = val("--model")
variant = val("--variant")
seed = val("--seed")
tag = "pgutsf" if model == "pgutsf" else "hdpguts"
print(f"pguts_{tag}_{dataset}_{mask}_r{rate}_h{h}_pf{pool}_{variant}_s{seed}")
PY
}

launch_cmd() {
  local cmd="$1"
  acquire_gpu_wait
  local gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")
  local name
  name="$(run_name_from_cmd "$cmd")"
  local log="$LOG_DIR/${name}.log"
  echo "[r0721 pguts] launch gpu=${gpu} name=${name}"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd"
  ) > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_DIR/${name}.pid"
  pid_gpu[$pid]="$gpu"
}

while IFS= read -r cmd; do
  [[ -z "$cmd" || "${cmd:0:1}" == "#" ]] && continue
  launch_cmd "$cmd"
done < "$CMD_FILE"

while (( ${#pid_gpu[@]} > 0 )); do
  done_pid=""
  if wait -n -p done_pid; then
    :
  else
    failed_jobs=$((failed_jobs + 1))
  fi
  unset 'pid_gpu[$done_pid]'
done

python "$ROOT/missing_ts_exp/scripts/r0721_collect_pguts_results.py"
echo "[r0721 pguts] mode=${MODE} finished; failed_jobs=${failed_jobs}"

if (( failed_jobs > 0 )); then
  exit 1
fi

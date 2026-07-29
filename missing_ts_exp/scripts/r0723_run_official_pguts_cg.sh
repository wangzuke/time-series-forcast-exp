#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
PGUTS_DIR="$ROOT/external_repro/pguts"
RESULTS="$ROOT/missing_ts_exp/results/0723_official_pguts_coarse_imputation"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
MODE="${1:-smoke}"
TSL_DATA_DIR="${R0723_TSL_DATA_DIR:-$RESULTS/tsl_cache}"

mkdir -p "$LOG_DIR" "$PID_DIR" "$RESULTS/notes" "$RESULTS/csv"

link_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$dst" && ! -L "$dst" ]]; then
    ln -s "$src" "$dst"
  fi
}

prepare_tsl_data() {
  local metr_dir="$TSL_DATA_DIR/MetrLA"
  local pems_dir="$TSL_DATA_DIR/PemsBay"
  mkdir -p "$metr_dir" "$pems_dir"

  link_if_missing "$ROOT/dataset/metr_la/metr_la.h5" "$metr_dir/metr_la.h5"
  link_if_missing "$ROOT/dataset/_archives/dcrnn_sensor_graph/distances_la_2012.csv" "$metr_dir/distances_la.csv"
  link_if_missing "$ROOT/dataset/_archives/dcrnn_sensor_graph/graph_sensor_ids.txt" "$metr_dir/sensor_ids_la.txt"
  link_if_missing "$ROOT/dataset/_archives/dcrnn_sensor_graph/graph_sensor_locations.csv" "$metr_dir/sensor_locations_la.csv"

  link_if_missing "$ROOT/dataset/pems_bay/pems_bay.h5" "$pems_dir/pems_bay.h5"
  link_if_missing "$ROOT/dataset/_archives/dcrnn_sensor_graph/distances_bay_2017.csv" "$pems_dir/distances_bay.csv"
  link_if_missing "$ROOT/dataset/_archives/dcrnn_sensor_graph/graph_sensor_locations_bay.csv" "$pems_dir/sensor_locations_bay.csv"
}

prepare() {
  test -d "$PGUTS_DIR/.git"
  prepare_tsl_data
  export R0723_TSL_DATA_DIR="$TSL_DATA_DIR"
  python "$ROOT/missing_ts_exp/scripts/r0723_build_official_pguts_cg_cmds.py"
  git -C "$PGUTS_DIR" rev-parse HEAD > "$RESULTS/notes/official_pguts_commit.txt"
  git -C "$PGUTS_DIR" diff > "$RESULTS/notes/official_pguts_coarse_graph.patch"
  if [[ "${R0723_SKIP_ENV_CHECK:-0}" != "1" ]]; then
    local check_cmd="${R0723_PYTHON_CMD:-python}"
    bash -lc "$check_cmd '$ROOT/missing_ts_exp/scripts/r0723_check_official_pguts_env.py'"
  else
    echo "[r0723 official pguts] skip env check because R0723_SKIP_ENV_CHECK=1"
  fi
}

if [[ "$MODE" == "prepare" ]]; then
  prepare
  exit 0
fi

prepare

case "$MODE" in
  smoke) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0723_official_pguts_cg_smoke_cmds.txt" ;;
  reproduce) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0723_official_pguts_reproduce_cmds.txt" ;;
  coarse) CMD_FILE="$ROOT/missing_ts_exp/scripts/r0723_official_pguts_cg_coarse_cmds.txt" ;;
  table3)
    python "$ROOT/missing_ts_exp/scripts/r0723_build_official_pguts_inference_cmds.py"
    CMD_FILE="$ROOT/missing_ts_exp/scripts/r0723_official_pguts_cg_table3_inference_cmds.txt"
    ;;
  cmdfile)
    if [[ $# -lt 2 ]]; then
      echo "cmdfile mode requires a command-file path"
      exit 64
    fi
    CMD_FILE="$2"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Expected one of: prepare, smoke, reproduce, coarse, table3, cmdfile"
    exit 64
    ;;
esac

GPUS_STR="${R0723_GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${R0723_SLOTS_PER_GPU:-1}"
PYTHON_CMD="${R0723_PYTHON_CMD:-python}"
read -r -a GPUS <<< "$GPUS_STR"

free_gpus=()
for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
  for gpu in "${GPUS[@]}"; do
    free_gpus+=("$gpu")
  done
done
declare -A pid_gpu
failed_jobs=0

echo "[r0723 official pguts] mode=${MODE} gpus=${GPUS_STR} slots_per_gpu=${SLOTS_PER_GPU}"

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
import pathlib
import shlex
import sys

parts = shlex.split(sys.argv[1])
def val(flag, default=""):
    try:
        return parts[parts.index(flag) + 1]
    except ValueError:
        return default

config = pathlib.Path(val("--config")).stem
dataset = val("--dataset-name")
seed = val("--seed")
graph = val("--graph-variant", "")
pf = val("--p-fault", "")
exp = pathlib.Path(val("--exp-name", "")).name
suffix = ""
if graph:
    suffix += f"_{graph}"
if pf:
    suffix += f"_pf{pf.replace('.', 'p')}"
if not seed and exp:
    seed = exp
print(f"{dataset}_{config}_s{seed}{suffix}")
PY
}

launch_cmd() {
  local cmd="$1"
  acquire_gpu_wait
  local gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")
  local name
  name="${MODE}_$(run_name_from_cmd "$cmd")"
  local log="$LOG_DIR/${name}.log"
  echo "[r0723 official pguts] launch gpu=${gpu} name=${name}"
  (
    cd "$PGUTS_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" R0723_RUN_TAG="$name" bash -lc "$PYTHON_CMD ${cmd#python }"
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

echo "[r0723 official pguts] mode=${MODE} finished; failed_jobs=${failed_jobs}"

if (( failed_jobs > 0 )); then
  exit 1
fi

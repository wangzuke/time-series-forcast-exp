#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
PGUTS_DIR="$ROOT/external_repro/pguts"
RESULTS="$ROOT/missing_ts_exp/results/0723_official_pguts_coarse_imputation"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
PYTHON_CMD_STR="${R0723_PYTHON_CMD:-/data/miniconda3/bin/conda run -n spin_env python}"
TSL_DATA_DIR="${R0723_TSL_DATA_DIR:-$RESULTS/tsl_cache}"

read -r -a PYTHON_CMD <<< "$PYTHON_CMD_STR"

mkdir -p "$LOG_DIR" "$PID_DIR"

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"
  echo "[r0723 bs16/be300] launch gpu=${gpu} name=${name}"
  (
    cd "$PGUTS_DIR"
    nohup env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      R0723_RUN_TAG="$name" \
      R0723_TSL_DATA_DIR="$TSL_DATA_DIR" \
      "${PYTHON_CMD[@]}" "$@" > "$log" 2>&1 &
    echo "$!" > "$PID_DIR/${name}.pid"
  )
}

launch 0 bs16be300_repro_la_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/la_pguts_36.yaml --dataset-name la_block --seed 1 --batch-size 16 --batches-epoch 300
launch 1 bs16be300_repro_la_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/la_pguts_36.yaml --dataset-name la_block --seed 2 --batch-size 16 --batches-epoch 300
launch 2 bs16be300_repro_bay_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/bay_pguts_36.yaml --dataset-name bay_block --seed 1 --batch-size 16 --batches-epoch 300
launch 3 bs16be300_repro_bay_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/bay_pguts_36.yaml --dataset-name bay_block --seed 2 --batch-size 16 --batches-epoch 300
launch 4 bs16be300_cgdist_la_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/la_cgdist_pguts_36.yaml --dataset-name la_block --seed 1 --batch-size 16 --batches-epoch 300
launch 5 bs16be300_cgdist_la_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/la_cgdist_pguts_36.yaml --dataset-name la_block --seed 2 --batch-size 16 --batches-epoch 300
launch 6 bs16be300_cgdist_bay_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/bay_cgdist_pguts_36.yaml --dataset-name bay_block --seed 1 --batch-size 16 --batches-epoch 300
launch 7 bs16be300_cgdist_bay_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/bay_cgdist_pguts_36.yaml --dataset-name bay_block --seed 2 --batch-size 16 --batches-epoch 300

echo "[r0723 bs16/be300] all launches submitted"

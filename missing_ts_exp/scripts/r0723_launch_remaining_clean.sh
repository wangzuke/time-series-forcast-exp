#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0723_official_pguts_coarse_imputation"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
PGUTS_DIR="$ROOT/external_repro/pguts"
TSL_DATA_DIR="$RESULTS/tsl_cache"
CONDA=/data/miniconda3/bin/conda

mkdir -p "$LOG_DIR" "$PID_DIR"

launch() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local dataset="$4"
  local seed="$5"
  local log="$LOG_DIR/${name}.log"
  local pidfile="$PID_DIR/${name}.pid"

  (
    cd "$PGUTS_DIR"
    exec setsid env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      R0723_TSL_DATA_DIR="$TSL_DATA_DIR" \
      R0723_RUN_TAG="$name" \
      "$CONDA" run -n spin_env python -m experiments.run_imputation \
        --config "$config" \
        --dataset-name "$dataset" \
        --seed "$seed" \
        --batch-size 256 \
        --batches-epoch 10
  ) > "$log" 2>&1 < /dev/null &
  echo "$!" > "$pidfile"
  echo "launch gpu=$gpu name=$name pid=$(cat "$pidfile")"
}

launch 2 reproduce_clean_bay_block_bay_pguts_36_s1 imputation/r0723/bay_pguts_36.yaml bay_block 1
launch 5 reproduce_clean_bay_block_bay_pguts_36_s2 imputation/r0723/bay_pguts_36.yaml bay_block 2

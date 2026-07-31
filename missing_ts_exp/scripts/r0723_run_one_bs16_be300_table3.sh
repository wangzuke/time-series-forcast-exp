#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
PGUTS_DIR="$ROOT/external_repro/pguts"
RESULTS="$ROOT/missing_ts_exp/results/0723_official_pguts_coarse_imputation"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
TSL_DATA_DIR="${R0723_TSL_DATA_DIR:-$RESULTS/tsl_cache}"
PYTHON_CMD="${R0723_PYTHON_CMD:-/data/miniconda3/bin/conda run -n spin_env python}"
MASK_SEEDS="${R0723_TABLE3_MASK_SEEDS:-6043,2043,3043,4043,5043}"

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <gpu> <name> <dataset_name> <exp_name> <batch_size>"
  exit 64
fi

GPU="$1"
NAME="$2"
DATASET_NAME="$3"
EXP_NAME="$4"
BATCH_SIZE="$5"

mkdir -p "$LOG_DIR" "$PID_DIR"
echo "$$" > "$PID_DIR/${NAME}.pid"

cd "$PGUTS_DIR"

for PF in 0.05 0.1 0.15; do
  PF_TOKEN="${PF/./p}"
  LOG="$LOG_DIR/bs16be300_table3_${NAME}_pf${PF_TOKEN}.log"
  echo "[table3] start name=${NAME} gpu=${GPU} p_fault=${PF} exp=${EXP_NAME}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  R0723_TSL_DATA_DIR="$TSL_DATA_DIR" \
  $PYTHON_CMD -m experiments.run_inference \
    --config inference.yaml \
    --model-name PGUTS \
    --dataset-name "$DATASET_NAME" \
    --exp-name "$EXP_NAME" \
    --p-fault "$PF" \
    --p-noise 0 \
    --test-mask-seed "$MASK_SEEDS" \
    --batch-size "$BATCH_SIZE" \
    > "$LOG" 2>&1
  echo "[table3] done name=${NAME} p_fault=${PF}"
done

echo "[table3] all done name=${NAME}"

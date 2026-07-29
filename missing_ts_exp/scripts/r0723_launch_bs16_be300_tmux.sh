#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
PGUTS_DIR="$ROOT/external_repro/pguts"
RESULTS="$ROOT/missing_ts_exp/results/0723_official_pguts_coarse_imputation"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
TSL_DATA_DIR="${R0723_TSL_DATA_DIR:-$RESULTS/tsl_cache}"
SESSION="${R0723_TMUX_SESSION:-r0723_bs16be300}"
PYTHON_CMD="${R0723_PYTHON_CMD:-/data/miniconda3/bin/conda run -n spin_env python}"

mkdir -p "$LOG_DIR" "$PID_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

window_cmd() {
  local gpu="$1"
  local name="$2"
  shift 2
  printf 'echo $$ > %q; cd %q; exec env CUDA_VISIBLE_DEVICES=%q R0723_RUN_TAG=%q R0723_TSL_DATA_DIR=%q %s %s > %q 2>&1' \
    "$PID_DIR/${name}.pid" \
    "$PGUTS_DIR" \
    "$gpu" \
    "$name" \
    "$TSL_DATA_DIR" \
    "$PYTHON_CMD" \
    "$*" \
    "$LOG_DIR/${name}.log"
}

new_window() {
  local gpu="$1"
  local name="$2"
  shift 2
  local cmd
  cmd="$(window_cmd "$gpu" "$name" "$@")"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$name" "bash -lc '$cmd'"
  else
    tmux new-window -t "$SESSION" -n "$name" "bash -lc '$cmd'"
  fi
  echo "[r0723 bs16/be300 tmux] launch gpu=${gpu} window=${name}"
}

new_window 0 bs16be300_repro_la_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/la_pguts_36.yaml --dataset-name la_block --seed 1 --batch-size 16 --batches-epoch 300
new_window 1 bs16be300_repro_la_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/la_pguts_36.yaml --dataset-name la_block --seed 2 --batch-size 16 --batches-epoch 300
new_window 2 bs16be300_repro_bay_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/bay_pguts_36.yaml --dataset-name bay_block --seed 1 --batch-size 16 --batches-epoch 300
new_window 3 bs16be300_repro_bay_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/bay_pguts_36.yaml --dataset-name bay_block --seed 2 --batch-size 16 --batches-epoch 300
new_window 4 bs16be300_cgdist_la_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/la_cgdist_pguts_36.yaml --dataset-name la_block --seed 1 --batch-size 16 --batches-epoch 300
new_window 5 bs16be300_cgdist_la_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/la_cgdist_pguts_36.yaml --dataset-name la_block --seed 2 --batch-size 16 --batches-epoch 300
new_window 6 bs16be300_cgdist_bay_pguts36_s1 -m experiments.run_imputation --config imputation/r0723/bay_cgdist_pguts_36.yaml --dataset-name bay_block --seed 1 --batch-size 16 --batches-epoch 300
new_window 7 bs16be300_cgdist_bay_pguts36_s2 -m experiments.run_imputation --config imputation/r0723/bay_cgdist_pguts_36.yaml --dataset-name bay_block --seed 2 --batch-size 16 --batches-epoch 300

echo "[r0723 bs16/be300 tmux] session=${SESSION}"

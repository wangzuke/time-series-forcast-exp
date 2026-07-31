#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
SESSION="${R0723_TABLE3_TMUX_SESSION:-r0723_bs16be300_table3}"
HELPER="$ROOT/missing_ts_exp/scripts/r0723_run_one_bs16_be300_table3.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

new_window() {
  local gpu="$1"
  local name="$2"
  local dataset_name="$3"
  local exp_name="$4"
  local batch_size="$5"
  local cmd
  cmd="bash '$HELPER' '$gpu' '$name' '$dataset_name' '$exp_name' '$batch_size'"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$name" "bash -lc \"$cmd\""
  else
    tmux new-window -t "$SESSION" -n "$name" "bash -lc \"$cmd\""
  fi
  echo "[r0723 table3] launch gpu=${gpu} window=${name}"
}

new_window 0 table3_repro_la_s1 la_block 20260728T153814_1_bs16be300_repro_la_pguts36_s1 64
new_window 1 table3_repro_la_s2 la_block 20260728T153813_2_bs16be300_repro_la_pguts36_s2 64
new_window 2 table3_repro_bay_s1 bay_block 20260728T153813_1_bs16be300_repro_bay_pguts36_s1 48
new_window 3 table3_repro_bay_s2 bay_block 20260728T153812_2_bs16be300_repro_bay_pguts36_s2 48
new_window 4 table3_cgdist_la_s1 la_block 20260728T153814_1_bs16be300_cgdist_la_pguts36_s1 64
new_window 5 table3_cgdist_la_s2 la_block 20260728T153814_2_bs16be300_cgdist_la_pguts36_s2 64
new_window 6 table3_cgdist_bay_s1 bay_block 20260728T153812_1_bs16be300_cgdist_bay_pguts36_s1 48
new_window 7 table3_cgdist_bay_s2 bay_block 20260728T153812_2_bs16be300_cgdist_bay_pguts36_s2 48

echo "[r0723 table3] session=${SESSION}"

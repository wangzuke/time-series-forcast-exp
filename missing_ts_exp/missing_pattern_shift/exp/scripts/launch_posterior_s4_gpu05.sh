#!/usr/bin/env bash
# Launch only after correctness and throughput checks have passed.
set -euo pipefail
EXP_ROOT="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
PY_BIN="/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
SUITE="posterior_s4_r50_b24_l96_h96_1119"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/posterior_s4_mpl
mkdir -p "$EXP_ROOT/logs/$SUITE"
TMUX_SOCKET="posterior-s4-1119"
TMUX_SESSION="experiments"
if tmux -L "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "Experiment session already exists: tmux -L $TMUX_SOCKET attach -t $TMUX_SESSION"
  exit 1
fi
# The command executor reaps ordinary background jobs. A dedicated tmux
# server gives the queue a persistent session independent of this command.
printf -v task_command '%q ' "$PY_BIN" -u "$EXP_ROOT/scripts/run_posterior_s4_suite.py" \
  --gpus 0 5 --phase all --slots-per-gpu "${SLOTS_PER_GPU:-3}" \
  --batch-size "${MICROBATCH:-64}" --workers "${DATA_WORKERS:-10}"
tmux -L "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" \
  "exec $task_command >> '$EXP_ROOT/logs/$SUITE/supervisor.log' 2>&1"
task_pid=$(tmux -L "$TMUX_SOCKET" display-message -p -t "$TMUX_SESSION" '#{pane_pid}')
echo "$task_pid" > "$EXP_ROOT/logs/$SUITE/supervisor.pid"
echo "Supervisor PID: $task_pid"
echo "Session: tmux -L $TMUX_SOCKET attach -t $TMUX_SESSION"
echo "Log: $EXP_ROOT/logs/$SUITE/supervisor.log"

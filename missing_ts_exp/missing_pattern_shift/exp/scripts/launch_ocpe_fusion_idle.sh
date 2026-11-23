#!/usr/bin/env bash
# Run correctness and synchronized concurrency benchmarks before launching.
set -euo pipefail
EXP_ROOT="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
PY_BIN="/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
SUITE="ocpe_fusion_uniform_r50_l96_h96_1120"
TMUX_SOCKET="ocpe-fusion-1120"
TMUX_SESSION="core"
if (($#)); then GPU_IDS=("$@"); else GPU_IDS=(0 1 7); fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$EXP_ROOT/logs/$SUITE"
if tmux -L "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Supervisor already exists: tmux -L $TMUX_SOCKET attach -t $TMUX_SESSION"
    exit 1
fi
MEASURED_SLOTS=$("$PY_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen_slots_per_gpu"])' "$EXP_ROOT/diagnostics/$SUITE/parallel/summary.json")
printf -v task_command '%q ' "$PY_BIN" -u "$EXP_ROOT/scripts/run_ocpe_fusion_suite.py" \
    --gpus "${GPU_IDS[@]}" --slots-per-gpu "$MEASURED_SLOTS" --workers 6
tmux -L "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" \
    "exec $task_command >> '$EXP_ROOT/logs/$SUITE/supervisor.log' 2>&1"
echo "Launched supervisor: GPUs=${GPU_IDS[*]}, slots_per_gpu=$MEASURED_SLOTS"
echo "Status: $EXP_ROOT/logs/$SUITE/supervisor_status.json"

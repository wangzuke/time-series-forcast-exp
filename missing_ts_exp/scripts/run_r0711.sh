#!/usr/bin/env bash
# Run 0711 experiments on GPUs 1-7 only.
#
# Usage from missing_ts_exp:
#   bash scripts/run_r0711.sh
#   bash scripts/run_r0711.sh scripts/r0711_phase1_cmds.txt

set -uo pipefail

CMD_FILE=${1:-scripts/r0711_cmds.txt}
GPU_LIST=${GPU_LIST:-1,2,3,4,5,6,7}
PER_GPU=${PER_GPU:-1}
LOG_DIR=${LOG_DIR:-logs/r0711}
PYTHON_ENV_BIN=${PYTHON_ENV_BIN:-/data/miniconda3/envs/itransformer/bin}

if [[ -d "$PYTHON_ENV_BIN" ]]; then
    export PATH="$PYTHON_ENV_BIN:$PATH"
fi

bash scripts/run_experiments.sh "$CMD_FILE" "$GPU_LIST" "$PER_GPU" "$LOG_DIR"

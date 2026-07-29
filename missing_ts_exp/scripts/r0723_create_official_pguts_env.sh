#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
PGUTS_DIR="$ROOT/external_repro/pguts"
ENV_NAME="${R0723_ENV_NAME:-spin_env}"
PYG_WHEEL_URL="${R0723_PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.3.0+cu118.html}"
RETRIES="${R0723_ENV_RETRIES:-3}"

cd "$PGUTS_DIR"

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      echo "[r0723 env] command failed after ${attempts} attempts: $*"
      return 1
    fi
    echo "[r0723 env] attempt ${n}/${attempts} failed; cleaning conda caches and retrying"
    conda clean --index-cache --tarballs -y || true
    sleep $((n * 10))
    n=$((n + 1))
  done
}

install_or_update_env() {
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[r0723 env] conda env '$ENV_NAME' already exists; updating from environment.yml"
    conda env update -n "$ENV_NAME" -f environment.yml --prune
  else
    echo "[r0723 env] creating conda env '$ENV_NAME' from official environment.yml"
    conda env create -n "$ENV_NAME" -f environment.yml
  fi
}

retry "$RETRIES" install_or_update_env

echo "[r0723 env] installing PyG sparse dependencies"
retry "$RETRIES" conda run -n "$ENV_NAME" python -m pip install \
  --retries 10 \
  --timeout 120 \
  --no-cache-dir \
  torch-scatter torch-sparse \
  -f "$PYG_WHEEL_URL"

echo "[r0723 env] running preflight"
R0723_PYTHON_CMD="conda run -n $ENV_NAME python" \
  python "$ROOT/missing_ts_exp/scripts/r0723_check_official_pguts_env.py"

echo "[r0723 env] ready: use R0723_PYTHON_CMD='conda run -n $ENV_NAME python'"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
COFILL="$ROOT/external_repro/CoFILL"
COFILL_ENV="${R0721_COFILL_ENV:-hd-tts}"

DATASET=""
DATA_PATH=""
MASK_PATH=""
RUN_DIR=""
BATCH_SIZE=512
SEED=1
TASK="imputation"
EPOCHS="${R0721_COFILL_EPOCHS:-200}"
NSAMPLE="${R0721_COFILL_NSAMPLE:-5}"
DIFFUSION_STEPS="${R0721_COFILL_DIFFUSION_STEPS:-50}"
NUM_WORKERS="${R0721_COFILL_NUM_WORKERS:-4}"
MISSING_TYPE=""
TARGET_MISSING_RATE=""
DEVICE="${CUDA_VISIBLE_DEVICES:+cuda:0}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data_path) DATA_PATH="$2"; shift 2 ;;
    --mask_path) MASK_PATH="$2"; shift 2 ;;
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --nsample) NSAMPLE="$2"; shift 2 ;;
    --diffusion_steps) DIFFUSION_STEPS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --missing_type) MISSING_TYPE="$2"; shift 2 ;;
    --target_missing_rate) TARGET_MISSING_RATE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$DATASET" || -z "$DATA_PATH" || -z "$MASK_PATH" || -z "$RUN_DIR" ]]; then
  echo "Required args: --dataset --data_path --mask_path --run_dir" >&2
  exit 64
fi

PYTHON_CMD=(conda run -n "$COFILL_ENV" python)

"${PYTHON_CMD[@]}" "$ROOT/missing_ts_exp/scripts/r0721_prepare_cofill_data.py"

cd "$COFILL"
CMD=(
  "${PYTHON_CMD[@]}" "$ROOT/missing_ts_exp/scripts/r0721_cofill_runner.py"
  --dataset "$DATASET"
  --data_path "$DATA_PATH"
  --mask_path "$MASK_PATH"
  --run_dir "$RUN_DIR"
  --batch_size "$BATCH_SIZE"
  --seed "$SEED"
  --task "$TASK"
  --epochs "$EPOCHS"
  --nsample "$NSAMPLE"
  --diffusion_steps "$DIFFUSION_STEPS"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
)

if [[ -n "$MISSING_TYPE" ]]; then
  CMD+=(--missing_type "$MISSING_TYPE")
fi
if [[ -n "$TARGET_MISSING_RATE" ]]; then
  CMD+=(--target_missing_rate "$TARGET_MISSING_RATE")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry_run)
fi

exec "${CMD[@]}"

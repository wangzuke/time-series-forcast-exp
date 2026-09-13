#!/usr/bin/env bash
set -euo pipefail

# Calls the unmodified HD-TTS official entry point from its repository root.
ROOT=/media/data1/wangzk/time-series-forcast-exp
HDTTS_ROOT="$ROOT/external_repro/official/hdtts"
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python

MODELS_CSV=${1:-hd_tts_amp,hd_tts_imp,tts_amp,tts_imp}
GPUS_CSV=${2:-0,1,2,3}
EPOCHS=${3:-2}
TRAIN_BATCHES=${4:-5}
PATIENCE=${5:-2}
RUN_SEED=${6:-2024}
RESULT_ROOT=${7:-"$ROOT/missing_ts_exp/results/r0908_official_baselines"}
LOG_ROOT=${8:-"$ROOT/missing_ts_exp/logs/r0908_official_baselines"}

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if (( ${#MODELS[@]} > ${#GPUS[@]} )); then
  echo "Need at least one GPU per requested model." >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
pids=()
for i in "${!MODELS[@]}"; do
  model=${MODELS[$i]}
  gpu=${GPUS[$i]}
  run_dir="$RESULT_ROOT/$model/seed${RUN_SEED}"
  log="$LOG_ROOT/${model}_seed${RUN_SEED}.log"
  (
    cd "$HDTTS_ROOT"
    PYTHONPATH="$HDTTS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/mpl-r0908 \
      "$PYTHON" experiments/run_realworld.py \
      "model=$model" dataset=la dataset/mode=block_t \
      "epochs=$EPOCHS" "train_batches=$TRAIN_BATCHES" "patience=$PATIENCE" \
      batch_size=32 workers=0 "+run.seed=$RUN_SEED" \
      "hydra.run.dir=$run_dir"
  ) >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

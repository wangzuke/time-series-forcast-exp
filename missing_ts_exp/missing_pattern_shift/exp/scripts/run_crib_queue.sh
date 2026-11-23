#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID TRAIN_FAMILY TRAIN_SEED" >&2
  exit 2
fi

gpu="$1"
family="$2"
seed="$3"
root="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
python="/media/data1/wangzk/miniconda3/envs/itransformer/bin/python"
worker_log="$root/logs/s4m_robustness_1114/worker_gpu${gpu}_${family}_s${seed}.log"
mkdir -p "$(dirname "$worker_log")"

for dataset in ETTh1 Weather Electricity; do
  batch_size=16
  if [[ "$dataset" == "Electricity" ]]; then
    batch_size=4
  fi
  echo "[$(date -Is)] START gpu=$gpu dataset=$dataset train=$family seed=$seed batch=$batch_size" | tee -a "$worker_log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" \
    "$root/scripts/run_crib_shift.py" \
    --dataset "$dataset" \
    --train-family "$family" \
    --seed "$seed" \
    --epochs 10 \
    --batch-size "$batch_size" \
    --patch-len 8 \
    --workers 0 \
    --tag s4m_robustness_1114 >> "$worker_log" 2>&1
  code=$?
  echo "[$(date -Is)] DONE gpu=$gpu dataset=$dataset train=$family seed=$seed exit=$code" | tee -a "$worker_log"
done

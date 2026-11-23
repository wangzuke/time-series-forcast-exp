#!/usr/bin/env bash
set +e

EXP_ROOT="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
PY_BIN="/media/data1/wangzk/miniconda3/envs/itransformer/bin/python"
WORKER="$EXP_ROOT/scripts/run_traffic_stability_worker.py"
MANIFEST="$EXP_ROOT/configs/s4m_traffic_electricity_stability_1115.json"

"$PY_BIN" -u "$WORKER" --manifest "$MANIFEST" --gpu 4 --baseline S4M \
    --phase electricity_ablation \
    --job-ids Electricity_S4M_no_prototype_retrieval_B_b24_s2025 \
    --status-tag queue_gpu4_S4M_no_prototype_B2025_1119
rc=$?

# Let the GPU 7 queue controller continue after this distinct job is complete.
kill -CONT 760048 2>/dev/null || true
exit "$rc"

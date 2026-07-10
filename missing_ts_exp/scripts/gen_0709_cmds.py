#!/usr/bin/env python3
"""Generate 0709 experiment commands: MissTSM grouped-query 自动分组
(Approach A: grouped_q4_corr, Approach B: grouped_q4_soft).

对齐 R4 P2-B 的实验矩阵，便于直接与已有 grouped_q4/full 结果对比。
详见 docs/实验计划0709.md。
"""
import os
import itertools

RESULTS_DIR = "results"
SEEDS = [2024, 2025]
DATASETS = ["Weather", "Electricity", "Traffic"]
MISS_TYPES = ["random_point", "continuous_segment"]
MISS_RATES = [0.3, 0.5, 0.7]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 32
LR = "1e-3"
PRED_LEN = 96


def result_exists(tag):
    return os.path.exists(os.path.join(RESULTS_DIR, f"{tag}.json"))


def cmd(tag, dataset, mt, mr, seed, variant, extra_args=""):
    args = (
        f"python -m src.training.run_forecast "
        f"--dataset {dataset} --method misstsm --predictor iTransformer "
        f"--impute none --missing_type {mt} --missing_rate {mr} "
        f"--seq_len 96 --pred_len {PRED_LEN} --seed {seed} "
        f"--epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} "
        f"--patience {PATIENCE} "
        f"--tag {tag} --out_dir results --misstsm_variant {variant}"
    )
    if extra_args:
        args += " " + extra_args
    return args


def gen_phase1_a():
    """方案 A：grouped_q4_corr（相关性重排 + 连续切分）。"""
    cmds = []
    for ds, mt, mr, seed in itertools.product(DATASETS, MISS_TYPES, MISS_RATES, SEEDS):
        rate_int = int(mr * 100)
        tag = f"r0709_P1A__misstsm_gq4corr__{ds}__{mt}_{rate_int}__h{PRED_LEN}_s{seed}"
        if result_exists(tag):
            continue
        cmds.append(cmd(tag, ds, mt, mr, seed, "grouped_q4_corr"))
    return cmds


def gen_phase1_b():
    """方案 B：grouped_q4_soft（可学习软路由，不加熵正则）。"""
    cmds = []
    for ds, mt, mr, seed in itertools.product(DATASETS, MISS_TYPES, MISS_RATES, SEEDS):
        rate_int = int(mr * 100)
        tag = f"r0709_P1B__misstsm_gq4soft__{ds}__{mt}_{rate_int}__h{PRED_LEN}_s{seed}"
        if result_exists(tag):
            continue
        cmds.append(cmd(tag, ds, mt, mr, seed, "grouped_q4_soft"))
    return cmds


def main():
    cmds = gen_phase1_a() + gen_phase1_b()
    out_path = "scripts/r0709_cmds.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(cmds) + "\n")
    print(f"wrote {len(cmds)} commands to {out_path}")


if __name__ == "__main__":
    main()

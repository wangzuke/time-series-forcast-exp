#!/usr/bin/env python3
"""Generate R4 experiment commands for P0 + P1 + P2.

Checks results/ for existing files and only outputs missing experiments.
"""
import os
import glob
import itertools

RESULTS_DIR = "results"
SEEDS = [2024, 2025]
MISS_TYPES = ["random_point", "continuous_segment"]
MISS_RATES = [0.1, 0.3, 0.5, 0.7]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 32
LR = "1e-3"

def result_exists(tag):
    return os.path.exists(os.path.join(RESULTS_DIR, f"{tag}.json"))

def cmd(tag, dataset, method, predictor, impute, mt, mr, pl, seed,
        extra_args=""):
    rate_int = int(mr * 100)
    args = (
        f"python -m src.training.run_forecast "
        f"--dataset {dataset} --method {method} --predictor {predictor} "
        f"--impute {impute} --missing_type {mt} --missing_rate {mr} "
        f"--seq_len 96 --pred_len {pl} --seed {seed} "
        f"--epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} "
        f"--patience {PATIENCE} "
        f"--tag {tag} --out_dir results"
    )
    if extra_args:
        args += " " + extra_args
    return args


def gen_p0a():
    """P0-A: ExchangeRate two-stage baselines."""
    cmds = []
    ds = "ExchangeRate"
    predictors = ["DLinear", "PatchTST", "iTransformer"]
    for pred, mt, mr, pl, seed in itertools.product(
        predictors, MISS_TYPES, MISS_RATES, [96, 336], SEEDS
    ):
        rate_int = int(mr * 100)
        tag = f"r4_P0A__{ds}__linear_{pred}__{mt}_{rate_int}__h{pl}_s{seed}"
        if result_exists(tag):
            continue
        cmds.append(cmd(tag, ds, "simple", pred, "linear", mt, mr, pl, seed))
    return cmds


def gen_p0b():
    """P0-B: Expand Mask-Aware iTransformer(add) to more datasets/rates."""
    cmds = []
    p0b_patchtst_keys = set()
    # Phase 1: Electricity, ExchangeRate full rates + ETTh1/Weather high rates
    configs = []
    for ds in ["Electricity", "ExchangeRate"]:
        for mr in MISS_RATES:
            configs.append((ds, mr))
    for ds in ["ETTh1", "Weather"]:
        for mr in [0.5, 0.7]:
            configs.append((ds, mr))

    for (ds, mr), mt, seed in itertools.product(configs, MISS_TYPES, SEEDS):
        rate_int = int(mr * 100)
        # mask_aware=add + iTransformer
        tag = f"r4_P0B__mask_aware_add_iTrans__{ds}__{mt}_{rate_int}__h96_s{seed}"
        if not result_exists(tag):
            cmds.append(cmd(tag, ds, "simple", "iTransformer", "linear", mt, mr, 96, seed,
                           "--mask_aware add"))
        # Also PatchTST add
        tag2 = f"r4_P0B__mask_aware_add_PatchTST__{ds}__{mt}_{rate_int}__h96_s{seed}"
        p0b_patchtst_keys.add((ds, mt, rate_int, seed))
        if not result_exists(tag2):
            cmds.append(cmd(tag2, ds, "simple", "PatchTST", "linear", mt, mr, 96, seed,
                           "--mask_aware add"))
    return cmds, p0b_patchtst_keys


def gen_p1():
    """P1: Condition matrix — fill gaps for 6 methods × 5 datasets × 4 rates."""
    cmds = []
    datasets = ["ETTh1", "ExchangeRate", "Weather", "Electricity", "Traffic"]

    # Method configs: (method, predictor, impute, extra_args, tag_method)
    methods = [
        ("simple", "iTransformer", "linear", "", "interp_iTrans"),
        ("simple", "PatchTST", "linear", "", "interp_PatchTST"),
        ("simple", "iTransformer", "linear", "--mask_aware add", "mask_add_iTrans"),
        ("misstsm", "iTransformer", "none", "", "misstsm_full"),
        ("coifnet", "iTransformer", "none", "", "coifnet_faithful"),
    ]

    # CoIFNet R2 variant (independent + xmask)
    methods.append(
        ("coifnet", "iTransformer", "none",
         "--coifnet_embed_type independent --coifnet_input_form xmask_cat_mask",
         "coifnet_R2var")
    )

    # DLinear only for ExchangeRate
    for mt, mr, seed in itertools.product(MISS_TYPES, MISS_RATES, SEEDS):
        rate_int = int(mr * 100)
        tag = f"r4_P1__interp_DLinear__ExchangeRate__{mt}_{rate_int}__h96_s{seed}"
        if not result_exists(tag):
            cmds.append(cmd(tag, "ExchangeRate", "simple", "DLinear", "linear",
                           mt, mr, 96, seed))

    # Traffic: only 4 methods (skip PatchTST, skip coifnet_R2var)
    traffic_methods = ["interp_iTrans", "mask_add_iTrans", "misstsm_full", "coifnet_faithful"]

    for (method, pred, imp, extra, tag_method), ds, mt, mr, seed in itertools.product(
        methods, datasets, MISS_TYPES, MISS_RATES, SEEDS
    ):
        # Skip Traffic for methods not in traffic set
        if ds == "Traffic" and tag_method not in traffic_methods:
            continue

        rate_int = int(mr * 100)
        tag = f"r4_P1__{tag_method}__{ds}__{mt}_{rate_int}__h96_s{seed}"
        if result_exists(tag):
            continue
        cmds.append(cmd(tag, ds, method, pred, imp, mt, mr, 96, seed, extra))
    return cmds


def gen_p2a():
    """P2-A: CoIFNet ablation — 5 variants on Weather/Electricity."""
    cmds = []
    variants = [
        ("A0", 128, "independent", "xmask_cat_mask"),
        ("A1", 256, "independent", "xmask_cat_mask"),
        ("A2", 128, "independent", "x_cat_mask"),
        ("A3", 256, "shared", "x_cat_mask"),
        ("A4", 256, "independent", "x_cat_mask"),
    ]
    datasets = ["Weather", "Electricity"]
    rates = [0.3, 0.5]

    for (vname, hidden, embed, inpform), ds, mt, mr, seed in itertools.product(
        variants, datasets, MISS_TYPES, rates, SEEDS
    ):
        rate_int = int(mr * 100)
        tag = f"r4_P2A__{vname}__{ds}__{mt}_{rate_int}__h96_s{seed}"
        if result_exists(tag):
            continue
        extra = (f"--coifnet_hidden {hidden} "
                 f"--coifnet_embed_type {embed} "
                 f"--coifnet_input_form {inpform}")
        cmds.append(cmd(tag, ds, "coifnet", "iTransformer", "none",
                       mt, mr, 96, seed, extra))
    return cmds


def gen_p2b():
    """P2-B: MissTSM grouped query (G=4)."""
    cmds = []
    datasets = ["Weather", "Electricity", "Traffic"]
    rates = [0.3, 0.5, 0.7]

    for ds, mt, mr, seed in itertools.product(datasets, MISS_TYPES, rates, SEEDS):
        rate_int = int(mr * 100)
        tag = f"r4_P2B__misstsm_gq4__{ds}__{mt}_{rate_int}__h96_s{seed}"
        if result_exists(tag):
            continue
        cmds.append(cmd(tag, ds, "misstsm", "iTransformer", "none",
                       mt, mr, 96, seed, "--misstsm_variant grouped_q4"))
    return cmds


def gen_p2c(p0b_patchtst_keys):
    """P2-C: Mask-Aware PatchTST(add) — 4 datasets, 3 rates, pred_len 96.
    Skip conditions already covered by P0-B."""
    cmds = []
    datasets = ["ETTh1", "Weather", "Electricity", "ExchangeRate"]
    rates = [0.1, 0.3, 0.5]

    for ds, mt, mr, seed in itertools.product(datasets, MISS_TYPES, rates, SEEDS):
        rate_int = int(mr * 100)
        key = (ds, mt, rate_int, seed)
        if key in p0b_patchtst_keys:
            continue
        tag = f"r4_P2C__mask_add_PatchTST__{ds}__{mt}_{rate_int}__h96_s{seed}"
        if result_exists(tag):
            continue
        cmds.append(cmd(tag, ds, "simple", "PatchTST", "linear",
                       mt, mr, 96, seed, "--mask_aware add"))
    return cmds


if __name__ == "__main__":
    all_cmds = []

    p0a = gen_p0a()
    print(f"P0-A (ExchangeRate two-stage): {len(p0a)} cmds")
    all_cmds.extend(p0a)

    p0b, p0b_ptst_keys = gen_p0b()
    print(f"P0-B (Mask-Aware expand):       {len(p0b)} cmds")
    all_cmds.extend(p0b)

    p1 = gen_p1()
    print(f"P1  (Condition matrix):          {len(p1)} cmds")
    all_cmds.extend(p1)

    p2a = gen_p2a()
    print(f"P2-A (CoIFNet ablation):         {len(p2a)} cmds")
    all_cmds.extend(p2a)

    p2b = gen_p2b()
    print(f"P2-B (MissTSM grouped_q4):       {len(p2b)} cmds")
    all_cmds.extend(p2b)

    p2c = gen_p2c(p0b_ptst_keys)
    print(f"P2-C (PatchTST mask add):        {len(p2c)} cmds")
    all_cmds.extend(p2c)

    print(f"\nTotal: {len(all_cmds)} commands")

    out_path = "scripts/r4_cmds.txt"
    with open(out_path, "w") as f:
        for c in all_cmds:
            f.write(c + "\n")
    print(f"Written to {out_path}")

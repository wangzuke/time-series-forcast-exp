"""生成方向2 PMA 实验命令清单。

本脚本只写命令，不直接执行。执行方式见 docs/实验计划0729_方向2_缺失模式自适应Adapter.md。
"""
from __future__ import annotations

import argparse
import itertools
import os


DATASETS_MAIN = ["ETTh1", "Weather", "Electricity", "Traffic"]
DATASETS_PILOT = ["Weather", "Electricity"]
DATASETS_STRESS = ["Weather", "Traffic"]
MISSING_TYPES_MAIN = ["random_point", "continuous_segment", "variable_channel", "mixed"]
MISSING_TYPES_PILOT = ["continuous_segment", "variable_channel"]
RATES_MAIN = [0.1, 0.3, 0.5]
RATES_STRESS = [0.7]
PRED_LENS_MAIN = [96, 336]
SEEDS_MAIN = [2024, 2025, 2026]
SEEDS_FAST = [2024]

BASELINES = [
    ("simple", "iTransformer", "linear", "full"),
    ("saits", "iTransformer", "none", "full"),
    ("crib", "iTransformer", "none", "full"),
    ("coifnet", "iTransformer", "none", "full"),
    ("misstsm", "iTransformer", "none", "full"),
]

PMA_VARIANTS = [
    "grouped_q4_corr",
    "grouped_q4_corrobs",
    "grouped_q4_soft",
    "grouped_q4_fuse",
    "grouped_q4_fuseobs_mgate",
]

BACKBONES = ["DLinear", "PatchTST", "iTransformer"]


def common_args(ds, mt, rate, pred_len, seed, epochs, batch_size, lr, patience):
    return [
        f"--dataset {ds}",
        "--seq_len 96",
        f"--pred_len {pred_len}",
        f"--missing_type {mt}",
        f"--missing_rate {rate}",
        f"--seed {seed}",
        f"--epochs {epochs}",
        f"--batch_size {batch_size}",
        f"--lr {lr}",
        f"--patience {patience}",
        "--num_workers 0",
    ]


def make_cmd(
    ds,
    mt,
    rate,
    pred_len,
    seed,
    method,
    predictor,
    impute,
    variant,
    tag_prefix,
    epochs,
    batch_size,
    lr,
    patience,
    base_out,
):
    args = common_args(ds, mt, rate, pred_len, seed, epochs, batch_size, lr, patience)
    args += [f"--method {method}", f"--predictor {predictor}", f"--impute {impute}"]
    if method == "misstsm":
        args += [f"--misstsm_variant {variant}"]
    if method == "saits":
        args += ["--saits_pretrain_epochs 2"]
    tag = (
        f"{tag_prefix}/{method}_{predictor}_{impute}_{variant}/"
        f"{ds}/{mt}_{int(rate * 100)}/h{pred_len}_s{seed}"
    ).replace("/", "__")
    args += [f"--tag {tag}"]
    return "python -m src.training.run_forecast " + " ".join(args) + f" --out_dir {base_out}"


def build_smoke(base_out):
    cmds = []
    settings = [
        ("simple", "iTransformer", "linear", "full"),
        ("misstsm", "iTransformer", "none", "full"),
        ("misstsm", "iTransformer", "none", "grouped_q4_fuseobs_mgate"),
        ("crib", "iTransformer", "none", "full"),
        ("coifnet", "iTransformer", "none", "full"),
    ]
    for method, predictor, impute, variant in settings:
        cmds.append(
            make_cmd(
                "ETTh1", "continuous_segment", 0.3, 96, 2024,
                method, predictor, impute, variant,
                "r0729_smoke", 1, 64, 1e-3, 1, base_out,
            )
        )
    return cmds


def build_pilot(base_out):
    cmds = []
    methods = BASELINES + [
        ("misstsm", "iTransformer", "none", "grouped_q4_fuseobs_mgate"),
    ]
    for ds, mt, rate, seed in itertools.product(DATASETS_PILOT, MISSING_TYPES_PILOT, [0.3, 0.5], SEEDS_FAST):
        for method, predictor, impute, variant in methods:
            cmds.append(
                make_cmd(
                    ds, mt, rate, 96, seed,
                    method, predictor, impute, variant,
                    "r0729_pilot", 3, 64, 1e-3, 2, base_out,
                )
            )
    return cmds


def build_main(base_out):
    cmds = []
    methods = BASELINES + [
        ("misstsm", "iTransformer", "none", "grouped_q4_fuseobs_mgate"),
    ]
    for ds, mt, rate, pred_len, seed in itertools.product(
        DATASETS_MAIN, MISSING_TYPES_MAIN, RATES_MAIN, PRED_LENS_MAIN, SEEDS_MAIN
    ):
        for method, predictor, impute, variant in methods:
            cmds.append(
                make_cmd(
                    ds, mt, rate, pred_len, seed,
                    method, predictor, impute, variant,
                    "r0729_main", 20, 32, 1e-3, 5, base_out,
                )
            )
    return cmds


def build_ablation(base_out):
    cmds = []
    for ds, mt, rate, seed, variant in itertools.product(
        DATASETS_PILOT, MISSING_TYPES_PILOT, [0.3, 0.5, 0.7], SEEDS_MAIN, ["full"] + PMA_VARIANTS
    ):
        cmds.append(
            make_cmd(
                ds, mt, rate, 96, seed,
                "misstsm", "iTransformer", "none", variant,
                "r0729_ablation", 20, 32, 1e-3, 5, base_out,
            )
        )
    return cmds


def build_backbone(base_out):
    cmds = []
    for ds, mt, rate, seed, backbone, variant in itertools.product(
        DATASETS_PILOT, MISSING_TYPES_PILOT, [0.3, 0.5], SEEDS_MAIN, BACKBONES,
        ["full", "grouped_q4_fuseobs_mgate"],
    ):
        cmds.append(
            make_cmd(
                ds, mt, rate, 96, seed,
                "misstsm", backbone, "none", variant,
                "r0729_backbone", 20, 32, 1e-3, 5, base_out,
            )
        )
    return cmds


def build_stress(base_out):
    cmds = []
    methods = [
        ("simple", "iTransformer", "linear", "full"),
        ("misstsm", "iTransformer", "none", "full"),
        ("misstsm", "iTransformer", "none", "grouped_q4_fuseobs_mgate"),
        ("crib", "iTransformer", "none", "full"),
        ("coifnet", "iTransformer", "none", "full"),
    ]
    for ds, mt, seed in itertools.product(DATASETS_STRESS, MISSING_TYPES_MAIN, SEEDS_MAIN):
        for method, predictor, impute, variant in methods:
            cmds.append(
                make_cmd(
                    ds, mt, 0.7, 96, seed,
                    method, predictor, impute, variant,
                    "r0729_stress", 20, 32, 1e-3, 5, base_out,
                )
            )
    return cmds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        choices=["smoke", "pilot", "main", "ablation", "backbone", "stress", "all"],
        default="pilot",
    )
    ap.add_argument("--out", default="results/cmds/r0729_direction2_cmds.txt")
    ap.add_argument("--base_out", default="results/0729_direction2")
    return ap.parse_args()


def main():
    args = parse_args()
    builders = {
        "smoke": build_smoke,
        "pilot": build_pilot,
        "main": build_main,
        "ablation": build_ablation,
        "backbone": build_backbone,
        "stress": build_stress,
    }
    if args.suite == "all":
        cmds = []
        for name in ["pilot", "main", "ablation", "backbone", "stress"]:
            cmds.extend(builders[name](args.base_out))
    else:
        cmds = builders[args.suite](args.base_out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for cmd in cmds:
            f.write(cmd + "\n")
    print(f"wrote {len(cmds)} commands -> {args.out}")


if __name__ == "__main__":
    main()

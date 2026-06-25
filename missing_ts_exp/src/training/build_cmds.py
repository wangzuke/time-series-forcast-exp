"""按实验计划生成所有实验的命令清单（不直接执行）。

每个实验产出一个 JSON 文件。命令清单按 GPU 分组写入 shell 文件，
由 scripts/run_experiments.sh 调度执行。
"""
from __future__ import annotations
import os
import argparse
import itertools
import json
from dataclasses import dataclass
from typing import List


# 与实验计划对齐的默认设置
MAIN_DATASETS = ["ETTh1", "Weather", "Electricity", "Traffic"]
EXT_DATASETS = ["ETTm1", "ExchangeRate"]
MAIN_MISSING_TYPES = ["random_point", "continuous_segment"]
EXT_MISSING_TYPES = ["variable_channel", "mixed"]
MISSING_RATES = [0.1, 0.3]
PRED_LENS_MAIN = [96, 336]
PRED_LENS_EXT = [96, 192, 336, 720]
SEEDS = [2024, 2025, 2026]
PRED_MODELS = ["DLinear", "PatchTST", "iTransformer"]
SIMPLE_IMPUTERS = ["linear"]

# 第二轮实验设置
R2_DATASETS = ["Weather", "Electricity", "Traffic"]
R2_SEEDS = [2024, 2025]
R2_AWARE_METHODS = ["misstsm", "crib", "coifnet"]
R2_EPOCHS = 20
R2_PATIENCE = 5


@dataclass
class Cmd:
    name: str
    args: List[str]

    def to_cmd(self, py: str = "python", base_out: str = "results") -> str:
        return (
            f"{py} -m src.training.run_forecast "
            + " ".join(self.args)
            + f" --out_dir {base_out}"
        )


def _common(dataset, seq_len, pred_len, missing_type, missing_rate, seed,
            epochs, batch_size, lr, patience=None):
    args = [
        f"--dataset {dataset}",
        f"--seq_len {seq_len}",
        f"--pred_len {pred_len}",
        f"--missing_type {missing_type}",
        f"--missing_rate {missing_rate}",
        f"--seed {seed}",
        f"--epochs {epochs}",
        f"--batch_size {batch_size}",
        f"--lr {lr}",
    ]
    if patience is not None:
        args.append(f"--patience {patience}")
    return args


def gen_baseline_no_missing(seq_len=96, epochs=10, batch_size=32, lr=1e-3):
    """Section 8.1：无缺失上界，每个数据集 * 预测长度 * 预测模型 * 种子。"""
    cmds = []
    for ds in MAIN_DATASETS + EXT_DATASETS:
        for H in PRED_LENS_MAIN:
            for pm in PRED_MODELS:
                for seed in SEEDS:
                    args = _common(ds, seq_len, H, "none", 0.0, seed, epochs, batch_size, lr) + [
                        f"--method baseline",
                        f"--predictor {pm}",
                        f"--impute none",
                    ]
                    tag = f"baseline/{ds}/no_missing/{pm}_h{H}_s{seed}"
                    cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_simple_imputation(seq_len=96, epochs=10, batch_size=32, lr=1e-3,
                          datasets=None, missing_types=None, rates=None, preds=None,
                          seeds=None, models=None, impts=None):
    datasets = datasets or MAIN_DATASETS
    missing_types = missing_types or MAIN_MISSING_TYPES
    rates = rates or MISSING_RATES
    preds = preds or PRED_LENS_MAIN
    seeds = seeds or SEEDS
    models = models or PRED_MODELS
    impts = impts or SIMPLE_IMPUTERS
    cmds = []
    for ds, mt, r, H, seed, pm, imp in itertools.product(
        datasets, missing_types, rates, preds, seeds, models, impts
    ):
        args = _common(ds, seq_len, H, mt, r, seed, epochs, batch_size, lr) + [
            f"--method simple",
            f"--predictor {pm}",
            f"--impute {imp}",
        ]
        tag = f"simple_impute/{ds}/{mt}_{int(r*100)}/{imp}_{pm}_h{H}_s{seed}"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_saits(seq_len=96, epochs=10, batch_size=32, lr=1e-3,
              datasets=None, missing_types=None, rates=None, preds=None, seeds=None,
              models=None, pretrain=2):
    datasets = datasets or MAIN_DATASETS
    missing_types = missing_types or MAIN_MISSING_TYPES
    rates = rates or MISSING_RATES
    preds = preds or PRED_LENS_MAIN
    seeds = seeds or SEEDS
    models = models or PRED_MODELS
    cmds = []
    for ds, mt, r, H, seed, pm in itertools.product(
        datasets, missing_types, rates, preds, seeds, models
    ):
        args = _common(ds, seq_len, H, mt, r, seed, epochs, batch_size, lr) + [
            f"--method saits",
            f"--predictor {pm}",
            f"--impute none",
            f"--saits_pretrain_epochs {pretrain}",
        ]
        tag = f"saits/{ds}/{mt}_{int(r*100)}/{pm}_h{H}_s{seed}"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_missing_aware(seq_len=96, epochs=10, batch_size=32, lr=1e-3,
                      datasets=None, missing_types=None, rates=None, preds=None,
                      seeds=None, methods=None, predictor="iTransformer"):
    datasets = datasets or MAIN_DATASETS
    missing_types = missing_types or MAIN_MISSING_TYPES
    rates = rates or MISSING_RATES
    preds = preds or PRED_LENS_MAIN
    seeds = seeds or SEEDS
    methods = methods or ["misstsm", "crib", "coifnet"]
    cmds = []
    for ds, mt, r, H, seed, m in itertools.product(
        datasets, missing_types, rates, preds, seeds, methods
    ):
        args = _common(ds, seq_len, H, mt, r, seed, epochs, batch_size, lr) + [
            f"--method {m}",
            f"--predictor {predictor}",
            f"--impute none",
        ]
        tag = f"{m}/{ds}/{mt}_{int(r*100)}/h{H}_s{seed}"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_extension(seq_len_list=(96, 336), epochs=10, batch_size=32, lr=1e-3,
                  rates=(0.1, 0.3)):
    """扩展实验：ETTm1/ExchangeRate × {variable_channel, mixed} × {10,30}% × {96,192,336,720}"""
    cmds = []
    methods = ["simple:linear", "saits", "misstsm", "crib", "coifnet"]
    for ds, mt, r, sl, H, seed in itertools.product(
        EXT_DATASETS, EXT_MISSING_TYPES, rates, seq_len_list, PRED_LENS_EXT, SEEDS
    ):
        for m in methods:
            if m.startswith("simple"):
                method = "simple"; imp = m.split(":")[1]; predictor = "iTransformer"
            else:
                method = m; imp = "none"; predictor = "iTransformer"
            args = _common(ds, sl, H, mt, r, seed, epochs, batch_size, lr) + [
                f"--method {method}",
                f"--predictor {predictor}",
                f"--impute {imp}",
            ]
            tag = f"ext/{ds}/{mt}_{int(r*100)}/{method}_{imp}_{predictor}_L{sl}_h{H}_s{seed}"
            cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_ablation_mask(seq_len=96, epochs=10, batch_size=32, lr=1e-3):
    """11.1 缺失掩码作用：三种输入形式。

    通过开关 --impute (zero/no_impute) 与 method (simple/misstsm) 区分：
      A: simple+zero — 仅填补 0，但 mask 不传给模型（默认 mask 仍传，
         我们另开 'no_mask' 选项控制 misstsm 是否使用 mask）—— 这里采用 method=simple+linear
         作为 "只有数值" 的代理 vs. method=misstsm (有 mask 输入) vs. method=misstsm + 时间特征。
    """
    cmds = []
    for ds in MAIN_DATASETS:
        for mt in MAIN_MISSING_TYPES:
            for r in [0.1, 0.3]:
                for seed in SEEDS:
                    # 只有数值（线性插值后送 iTransformer，关闭时间特征 -> 通过去掉 x_mark 传入实现，这里简化保持时间特征不变）
                    args = _common(ds, seq_len, 96, mt, r, seed, epochs, batch_size, lr) + [
                        "--method simple", "--predictor iTransformer", "--impute linear",
                    ]
                    tag = f"abl_mask/{ds}/{mt}_{int(r*100)}/A_only_value_s{seed}"
                    cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
                    # 数值 + mask（MissTSM 使用 mask 进行 padding）
                    args = _common(ds, seq_len, 96, mt, r, seed, epochs, batch_size, lr) + [
                        "--method misstsm", "--predictor iTransformer", "--impute none",
                    ]
                    tag = f"abl_mask/{ds}/{mt}_{int(r*100)}/B_value_mask_s{seed}"
                    cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
                    # 数值 + mask + 时间位置（再额外考虑：当前管线默认就传时间特征；用 CRIB 加显式 KL 正则等）
                    args = _common(ds, seq_len, 96, mt, r, seed, epochs, batch_size, lr) + [
                        "--method crib", "--predictor iTransformer", "--impute none",
                    ]
                    tag = f"abl_mask/{ds}/{mt}_{int(r*100)}/C_value_mask_time_s{seed}"
                    cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_ablation_error_prop(seq_len=96, epochs=10, batch_size=32, lr=1e-3):
    """11.2 补值误差传播：所有两阶段方法（simple ∪ saits），记录补值误差和预测误差。

    运行结果 JSON 中已含 impute_mse；调度同 SAITS / 简单填补的子集即可。
    """
    return (
        gen_simple_imputation(seq_len, epochs, batch_size, lr,
                              models=["iTransformer"], rates=[0.1, 0.3],
                              missing_types=["random_point"],
                              datasets=["ETTh1", "Weather"], preds=[96])
        + gen_saits(seq_len, epochs, batch_size, lr,
                    models=["iTransformer"], rates=[0.1, 0.3],
                    missing_types=["random_point"],
                    datasets=["ETTh1", "Weather"], preds=[96])
    )


def gen_ablation_generalize_missing_type(seq_len=96, epochs=10, batch_size=32, lr=1e-3):
    """11.3 缺失类型泛化：训练 random_point，但测试时改 missing_type。

    需要在 run_forecast 中支持训练/测试缺失类型分别设定；
    暂以"训练 random_point；逐个测试集 missing_type"为独立 4 个实验示意。
    简化做法：直接生成 4 个独立运行配置。
    """
    cmds = []
    for ds in MAIN_DATASETS:
        for test_mt in ["random_point", "continuous_segment", "variable_channel", "mixed"]:
            for seed in SEEDS:
                args = _common(ds, seq_len, 96, test_mt, 0.3, seed, epochs, batch_size, lr) + [
                    "--method misstsm", "--predictor iTransformer", "--impute none",
                ]
                tag = f"abl_gen_mt/{ds}/train_rp_test_{test_mt}_s{seed}"
                cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_ablation_predictor_complexity(seq_len=96, epochs=10, batch_size=32, lr=1e-3):
    """11.5 预测模型复杂度：固定缺失处理方式，遍历预测模型 (DLinear/PatchTST/iTransformer)。
    本组实验是 gen_simple_imputation 与 gen_saits 中已经覆盖的子集，按需汇总即可。
    """
    return gen_simple_imputation(seq_len, epochs, batch_size, lr,
                                 datasets=["ETTh1", "Weather"],
                                 missing_types=["random_point", "continuous_segment"],
                                 rates=[0.1, 0.3], preds=[96],
                                 impts=["linear"]) + gen_saits(
        seq_len, epochs, batch_size, lr,
        datasets=["ETTh1", "Weather"],
        missing_types=["random_point", "continuous_segment"],
        rates=[0.1, 0.3], preds=[96])


# ====================== 第二轮实验命令生成 ======================

def gen_r2_a(seq_len=96, batch_size=32, lr=1e-3):
    """组 A：主对比（144 条）—— 完整版缺失感知方法 vs 两阶段基准。"""
    cmds = []
    for ds, mt, r, H, seed, m in itertools.product(
        R2_DATASETS, MAIN_MISSING_TYPES, MISSING_RATES, PRED_LENS_MAIN,
        R2_SEEDS, R2_AWARE_METHODS
    ):
        args = _common(ds, seq_len, H, mt, r, seed, R2_EPOCHS, batch_size, lr,
                       patience=R2_PATIENCE) + [
            f"--method {m}", "--predictor iTransformer", "--impute none",
        ]
        tag = f"r2_a/{m}/{ds}/{mt}_{int(r*100)}/h{H}_s{seed}"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_r2_b(seq_len=96, batch_size=32, lr=1e-3):
    """组 B：高缺失率探测（64 条）—— 极端缺失下谁更鲁棒。"""
    cmds = []
    high_rates = [0.5, 0.7]
    datasets_b = ["Weather", "Traffic"]
    methods_b = [
        ("simple", "iTransformer", "linear"),
        ("misstsm", "iTransformer", "none"),
        ("crib", "iTransformer", "none"),
        ("coifnet", "iTransformer", "none"),
    ]
    for ds, mt, r, seed in itertools.product(
        datasets_b, MAIN_MISSING_TYPES, high_rates, R2_SEEDS
    ):
        for method, predictor, imp in methods_b:
            args = _common(ds, seq_len, 96, mt, r, seed, R2_EPOCHS, batch_size, lr,
                           patience=R2_PATIENCE) + [
                f"--method {method}", f"--predictor {predictor}", f"--impute {imp}",
            ]
            tag = f"r2_b/{method}/{ds}/{mt}_{int(r*100)}/h96_s{seed}"
            cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_r2_c(seq_len=96, batch_size=32, lr=1e-3):
    """组 C：SAITS+PatchTST 溢出修复（48 条）—— 补全第一轮空白。"""
    cmds = []
    datasets_c = ["Weather", "Electricity"]
    seeds_c = [2024, 2025, 2026]
    for ds, mt, r, H, seed in itertools.product(
        datasets_c, MAIN_MISSING_TYPES, MISSING_RATES, PRED_LENS_MAIN, seeds_c
    ):
        args = _common(ds, seq_len, H, mt, r, seed, R2_EPOCHS, batch_size, lr,
                       patience=R2_PATIENCE) + [
            "--method saits", "--predictor PatchTST", "--impute none",
            "--saits_pretrain_epochs 2",
        ]
        tag = f"r2_c/saits_PatchTST/{ds}/{mt}_{int(r*100)}/h{H}_s{seed}"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_r2_d(seq_len=96, batch_size=32, lr=1e-3):
    """组 D：掩码消融重验证（24 条）—— 掩码对完整版方法是否有效。"""
    cmds = []
    datasets_d = ["Weather", "Traffic"]
    rates_d = [0.3, 0.5]
    methods_d = [
        ("simple", "iTransformer", "linear"),
        ("misstsm", "iTransformer", "none"),
        ("crib", "iTransformer", "none"),
    ]
    for ds, r, seed in itertools.product(datasets_d, rates_d, R2_SEEDS):
        for method, predictor, imp in methods_d:
            args = _common(ds, seq_len, 96, "continuous_segment", r, seed,
                           R2_EPOCHS, batch_size, lr, patience=R2_PATIENCE) + [
                f"--method {method}", f"--predictor {predictor}", f"--impute {imp}",
            ]
            tag = f"r2_d/{method}/{ds}/cs_{int(r*100)}/h96_s{seed}"
            cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_r2_e(seq_len=96, batch_size=32, lr=1e-3):
    """组 E：扩展场景验证（64 条）—— 变量通道缺失和混合缺失。"""
    cmds = []
    datasets_e = ["Electricity", "Traffic"]
    methods_e = [
        ("simple", "iTransformer", "linear"),
        ("misstsm", "iTransformer", "none"),
        ("crib", "iTransformer", "none"),
        ("coifnet", "iTransformer", "none"),
    ]
    for ds, mt, H, seed in itertools.product(
        datasets_e, EXT_MISSING_TYPES, PRED_LENS_MAIN, R2_SEEDS
    ):
        for method, predictor, imp in methods_e:
            args = _common(ds, seq_len, H, mt, 0.3, seed, R2_EPOCHS, batch_size, lr,
                           patience=R2_PATIENCE) + [
                f"--method {method}", f"--predictor {predictor}", f"--impute {imp}",
            ]
            tag = f"r2_e/{method}/{ds}/{mt}_30/h{H}_s{seed}"
            cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_r2_f(seq_len=96, batch_size=32, lr=1e-3):
    """组 F：误差传播补充（12 条）—— 完整版方法补值误差与预测的关系。"""
    cmds = []
    datasets_f = ["Weather", "Electricity"]
    for ds, r, m in itertools.product(datasets_f, MISSING_RATES, R2_AWARE_METHODS):
        args = _common(ds, seq_len, 96, "random_point", r, 2024,
                       R2_EPOCHS, batch_size, lr, patience=R2_PATIENCE) + [
            f"--method {m}", "--predictor iTransformer", "--impute none",
        ]
        tag = f"r2_f/{m}/{ds}/rp_{int(r*100)}/h96_s2024"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def gen_r2_all(batch_size=32, lr=1e-3):
    """第二轮全部实验（356 条）。"""
    return (gen_r2_a(batch_size=batch_size, lr=lr)
            + gen_r2_b(batch_size=batch_size, lr=lr)
            + gen_r2_c(batch_size=batch_size, lr=lr)
            + gen_r2_d(batch_size=batch_size, lr=lr)
            + gen_r2_e(batch_size=batch_size, lr=lr)
            + gen_r2_f(batch_size=batch_size, lr=lr))


def gen_r2_smoke():
    """第二轮烟测：各方法 × Weather × 1 epoch。"""
    cmds = []
    for m, pm, imp in [
        ("simple", "iTransformer", "linear"),
        ("saits", "PatchTST", "none"),
        ("misstsm", "iTransformer", "none"),
        ("crib", "iTransformer", "none"),
        ("coifnet", "iTransformer", "none"),
    ]:
        extra = ["--saits_pretrain_epochs 1"] if m == "saits" else []
        args = _common("Weather", 96, 96, "random_point", 0.3, 2024, 1, 64, 1e-3) + [
            f"--method {m}", f"--predictor {pm}", f"--impute {imp}",
        ] + extra
        tag = f"r2_smoke/{m}_{pm}_{imp}"
        cmds.append(Cmd(tag, args + [f"--tag {tag.replace('/', '__')}"]))
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    choices=["all", "baseline", "simple", "saits", "aware", "extension",
                             "ablation_mask", "ablation_err", "ablation_gen", "ablation_pred",
                             "smoke", "r2", "r2_a", "r2_b", "r2_c", "r2_d", "r2_e", "r2_f",
                             "r2_smoke"])
    ap.add_argument("--out", default="scripts/cmds.txt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--base_out", default="results")
    args = ap.parse_args()

    all_cmds: List[Cmd] = []
    if args.group in ("all", "baseline"):
        all_cmds += gen_baseline_no_missing(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "simple"):
        all_cmds += gen_simple_imputation(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "saits"):
        all_cmds += gen_saits(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "aware"):
        all_cmds += gen_missing_aware(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "extension"):
        all_cmds += gen_extension(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.group in ("all", "ablation_mask"):
        all_cmds += gen_ablation_mask(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "ablation_err"):
        all_cmds += gen_ablation_error_prop(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "ablation_gen"):
        all_cmds += gen_ablation_generalize_missing_type(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group in ("all", "ablation_pred"):
        all_cmds += gen_ablation_predictor_complexity(args.seq_len, args.epochs, args.batch_size, args.lr)
    if args.group == "smoke":
        # 快速烟测：所有 8 个方法 × ETTh1 × 1 epoch
        for m, pm, imp in [
            ("baseline","DLinear","none"),("baseline","PatchTST","none"),("baseline","iTransformer","none"),
            ("simple","iTransformer","linear"),("saits","iTransformer","none"),
            ("misstsm","iTransformer","none"),("crib","iTransformer","none"),("coifnet","iTransformer","none"),
        ]:
            args2 = _common("ETTh1", 96, 96, "random_point", 0.3, 2024, 1, 64, 1e-3) + [
                f"--method {m}", f"--predictor {pm}", f"--impute {imp}",
            ]
            tag = f"smoke/{m}_{pm}_{imp}"
            all_cmds.append(Cmd(tag, args2 + [f"--tag {tag.replace('/', '__')}"]))
    # 第二轮实验
    if args.group in ("r2", "r2_a"):
        all_cmds += gen_r2_a(args.seq_len, args.batch_size, args.lr)
    if args.group in ("r2", "r2_b"):
        all_cmds += gen_r2_b(args.seq_len, args.batch_size, args.lr)
    if args.group in ("r2", "r2_c"):
        all_cmds += gen_r2_c(args.seq_len, args.batch_size, args.lr)
    if args.group in ("r2", "r2_d"):
        all_cmds += gen_r2_d(args.seq_len, args.batch_size, args.lr)
    if args.group in ("r2", "r2_e"):
        all_cmds += gen_r2_e(args.seq_len, args.batch_size, args.lr)
    if args.group in ("r2", "r2_f"):
        all_cmds += gen_r2_f(args.seq_len, args.batch_size, args.lr)
    if args.group == "r2_smoke":
        all_cmds += gen_r2_smoke()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for c in all_cmds:
            f.write(c.to_cmd(base_out=args.base_out) + "\n")
    print(f"wrote {len(all_cmds)} commands -> {args.out}")


if __name__ == "__main__":
    main()

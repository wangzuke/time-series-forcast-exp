"""Build, schedule, and summarize the METR-LA P2 and P3 experiment matrix."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
PACKAGE = "missing_ts_exp.proj1_DRR_MSGF"
PYTHON = sys.executable
PREPARE = PROJECT / "scripts/prepare_region_assignments.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(message: str) -> None:
    print(f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


@dataclass(frozen=True)
class Job:
    tag: str
    variant: str
    order: str
    membership: str
    assignment: Path
    seed: int
    mask: Path
    command: list[str]


class RegionalRuns:
    def __init__(self, args):
        self.args = args
        self.config_path = args.config.resolve()
        self.config = json.loads(self.config_path.read_text())
        self.name = args.run_id
        self.results = PROJECT / "results" / self.name
        self.logs = PROJECT / "logs" / self.name
        self.resources = PROJECT / "resources" / self.name
        self.physical = PROJECT / self.config["physical_assignment"]
        self.order = self.config["p3_reference_order"]
        self.environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.environment[key] = "2"
        self.environment.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED="0", MPLBACKEND="Agg",
                                CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.frozen = {
            path: digest(path)
            for path in (self.config_path, *sorted((PROJECT / "src").rglob("*.py")),
                         Path(__file__), PREPARE, self.physical)
        }

    def check_frozen(self):
        if any(digest(path) != sha for path, sha in self.frozen.items()):
            raise RuntimeError("A frozen source, configuration or physical assignment changed during the campaign.")

    def phase_directory(self, stage, phase) -> Path:
        return self.results / stage / phase

    def jobs(self, stage, phase, dynamic=True) -> list[Job]:
        conf = self.config
        full = "full"
        if stage == "p2":
            definitions = {
                name: (full, name, "physical_static", self.physical) for name in conf["p2_orders"]
            }
        else:
            definitions = {
                "physical_static": (full, self.order, "physical_static", self.physical),
                "functional_static": (full, self.order, "functional_static",
                                      self.resources / "functional_static.npy"),
                "no_region": ("fine", self.order, "physical_static", self.physical),
                "fixed_soft": (full, self.order, "fixed_soft", self.resources / "fixed_soft.npy"),
                "physical_anchored_dynamic": (full, self.order, "physical_anchored_dynamic", self.physical),
            }
            if not dynamic:
                definitions = {k: v for k, v in definitions.items()
                               if k not in {"fixed_soft", "physical_anchored_dynamic"}}
        seeds = conf["confirmation_seeds"] if phase == "confirm" else [conf["development_seed"]]
        budget = conf["smoke" if phase == "smoke" else "formal"]
        out = self.phase_directory(stage, phase)
        jobs = []
        for seed in seeds:
            for name, (variant, order, membership, assignment) in definitions.items():
                tag = f"{name}_seed{seed}"
                mask = PROJECT / conf["mask_template"].format(seed=seed)
                command = [PYTHON, "-m", f"{PACKAGE}.src.training.run_mrmsgf_official_controlled",
                           "--model", "mrmsgf", "--protocol_id", f"{conf['protocol_id']}_{stage}_{phase}",
                           "--seed", str(seed), "--mask_seed", str(seed), "--mask_path", str(mask),
                           "--mr_variant", variant, "--regional_order", order, "--region_mode", membership,
                           "--region_assignment_path", str(assignment), "--fusion_mode", conf["fusion_mode"],
                           "--graph_topk", str(conf["graph_topk"]), "--reliability_beta", str(conf["reliability_beta"]),
                           "--physical_anchor_lambda", str(conf["physical_anchor_lambda"]),
                           "--dynamic_region_temperature", str(conf["dynamic_region_temperature"]),
                           "--protocol_config_path", str(self.config_path), "--q_mode", "observed",
                           "--out_dir", str(out), "--tag", tag]
                for key, value in budget.items():
                    command.extend([f"--{key}", str(value)])
                jobs.append(Job(tag, variant, order, membership, assignment, seed, mask, command))
        return jobs

    def execute(self, module, *arguments, gpu=None, name=None):
        env = self.environment | {"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}
        path = self.logs / f"{name or module}.log"
        command = [PYTHON, "-m", f"{PACKAGE}.scripts.{module}", *map(str, arguments)]
        with path.open("a") as output:
            subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)

    def gpu_status(self):
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        return {gpu: (total, free, util) for gpu, total, free, util in
                (tuple(map(int, line.split(","))) for line in output.strip().splitlines())
                if gpu in self.args.gpus}

    def schedule(self, stage, phase, jobs):
        pending, active = list(jobs), []
        log_dir = self.logs / stage / phase
        log_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            if not job.assignment.is_file() or not job.mask.is_file():
                raise FileNotFoundError(f"{job.tag}: missing {job.assignment} or {job.mask}")
            if (self.phase_directory(stage, phase) / job.tag).exists():
                raise FileExistsError(f"{job.tag}: checkpoint directory exists. Use a new run ID for a retry.")
        last_notice = 0.0
        while pending or active:
            for item in active[:]:
                gpu, job, proc, output, started = item
                code = proc.poll()
                if code is not None:
                    output.close()
                    active.remove(item)
                    log(f"{stage}/{phase} {job.tag} GPU {gpu} exited {code} in {(time.monotonic()-started)/60:.1f} min")
                    if code != 0:
                        # Allow the independent sibling jobs to finish and retain their outputs.
                        for _, _, sibling, handle, _ in active:
                            sibling.wait()
                            handle.close()
                        raise RuntimeError(f"{job.tag} failed. See {log_dir / (job.tag + '.log')}")
            if pending:
                status = self.gpu_status()
                occupied = {gpu: sum(item[0] == gpu for item in active) for gpu in self.args.gpus}
                candidates = []
                for gpu, (total, free, utilization) in status.items():
                    slots = occupied[gpu]
                    if (slots >= self.args.slots_per_gpu or utilization > (78 if slots else 55)
                            or free < self.config["planning_mib_per_job"] + self.config["minimum_free_mib"]):
                        continue
                    if total - free + self.config["planning_mib_per_job"] > 0.82 * total:
                        continue
                    candidates.append((slots, utilization, -free, gpu))
                for _, _, _, gpu in sorted(candidates):
                    if not pending:
                        break
                    self.check_frozen()
                    job = pending.pop(0)
                    output = (log_dir / f"{job.tag}.log").open("w")
                    proc = subprocess.Popen(job.command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                            env=self.environment | {"CUDA_VISIBLE_DEVICES": str(gpu)})
                    active.append((gpu, job, proc, output, time.monotonic()))
                    log(f"{stage}/{phase} {job.tag} started on GPU {gpu}, PID {proc.pid}")
                if not candidates and time.monotonic() - last_notice > 60:
                    log(f"{stage}/{phase} waiting for GPU capacity: {len(pending)} queued, {len(active)} running")
                    last_notice = time.monotonic()
            if active or pending:
                time.sleep(5)

    def summarize(self, stage, phase, jobs):
        from scipy import stats
        import numpy as np

        fields = ("data_sha256", "parameter_count", "optimizer", "model_hparams", "window_metadata",
                  "graph_topk", "fusion_mode", "epochs", "train_batches", "eval_batches", "batch_size", "patience")
        out = self.phase_directory(stage, phase)
        rows = []
        reference = self.config["p2_orders"][0] if stage == "p2" else "physical_static"
        for job in jobs:
            row = json.loads((out / f"{job.tag}.json").read_text())
            expect = {
                "regional_order": job.order, "region_mode": job.membership, "mr_variant": job.variant,
                "seed": job.seed, "mask_seed": job.seed, "mask_sha256": digest(job.mask),
                "region_assignment_sha256": digest(job.assignment),
                "protocol_config_sha256": self.frozen[self.config_path],
            }
            if any(row[key] != value for key, value in expect.items()):
                raise RuntimeError(f"Result protocol mismatch: {job.tag}")
            if not Path(row["best_checkpoint"]).is_file():
                raise RuntimeError(f"Missing checkpoint: {job.tag}")
            test = row["unmasked_test"]
            mae, rmse = test["test_mae_unmasked"], math.sqrt(test["test_mse_unmasked"])
            if not all(map(math.isfinite, (mae, rmse, row["best_val_mae"]))):
                raise RuntimeError(f"Nonfinite metric: {job.tag}")
            if rows and any(row[key] != rows[0]["raw"][key] for key in fields):
                raise RuntimeError(f"{job.tag} differs in a shared protocol field.")
            if rows and row["source_manifest"]["controlled_source_sha256"] != rows[0]["raw"]["source_manifest"]["controlled_source_sha256"]:
                raise RuntimeError(f"{job.tag} used different controlled code.")
            rows.append(dict(tag=job.tag, variant=job.tag.rsplit("_seed", 1)[0], seed=job.seed,
                             mae=mae, rmse=rmse, best_val_mae=row["best_val_mae"], raw=row))
        references = {row["seed"]: row["mae"] for row in rows if row["variant"] == reference}
        comparisons = {}
        for name in dict.fromkeys(row["variant"] for row in rows):
            subset = [row for row in rows if row["variant"] == name]
            diff = [row["mae"] - references[row["seed"]] for row in subset]
            result = dict(mean_mae=float(np.mean([r["mae"] for r in subset])),
                          mean_rmse=float(np.mean([r["rmse"] for r in subset])),
                          paired_mae_differences=diff)
            if len(diff) > 1:
                result["sample_sd_mae"] = float(np.std([r["mae"] for r in subset], ddof=1))
                result["sample_sd_rmse"] = float(np.std([r["rmse"] for r in subset], ddof=1))
                if name != reference:
                    mean, sem = float(np.mean(diff)), float(stats.sem(diff))
                    half_width = float(stats.t.ppf(0.975, len(diff) - 1)) * sem
                    result["paired_ci95"] = [mean - half_width, mean + half_width]
                    result["paired_p"] = (float(stats.ttest_1samp(diff, popmean=0).pvalue)
                                          if sem > 0 else (0.0 if mean != 0 else 1.0))
            comparisons[name] = result
        running = 0.0
        tests = sorted((row for row in comparisons.values() if "paired_p" in row), key=lambda row: row["paired_p"])
        for index, row in enumerate(tests):
            running = max(running, min(1.0, (len(tests) - index) * row["paired_p"]))
            row["holm_p"] = running
        clean = [{k: v for k, v in row.items() if k != "raw"} for row in rows]
        (out / "summary.json").write_text(json.dumps(dict(stage=stage, phase=phase, reference=reference,
                                                          rows=clean, comparisons=comparisons),
                                                    indent=2, allow_nan=False) + "\n")
        with (out / "paired_metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(clean[0]))
            writer.writeheader()
            writer.writerows(clean)
        if phase != "smoke":
            report = PROJECT / "docs" / f"{self.name}_{stage}_{phase}_实验报告.md"
            title = "融合顺序" if stage == "p2" else "区域成员归属"
            lines = [f"# {stage.upper()} {title} · {phase}", "",
                     "METR-LA，Block-T 50%。测试指标基于原始有效未来标签。正 ΔMAE 表示该变体误差高于参照。",
                     f"参照：`{reference}`。试运行只比较当前单 seed，确认阶段报告配对统计。", "",
                     "| 变体 | MAE | RMSE | 各 seed ΔMAE | 配对 95% CI | Holm p |",
                     "|---|---:|---:|---|---|---:|"]
            for name, result in comparisons.items():
                delta = ", ".join(f"{value:+.4f}" for value in result["paired_mae_differences"])
                ci = ", ".join(f"{value:+.4f}" for value in result.get("paired_ci95", [])) or "—"
                p = f'{result["holm_p"]:.4g}' if "holm_p" in result else "—"
                mae, rmse = (f'{result[key]:.4f}' for key in ("mean_mae", "mean_rmse"))
                if "sample_sd_mae" in result:
                    mae += f' ± {result["sample_sd_mae"]:.4f}'
                    rmse += f' ± {result["sample_sd_rmse"]:.4f}'
                lines.append(f"| {name} | {mae} | {rmse} | {delta} | {ci} | {p} |")
            if stage == "p3":
                lines.extend(["", f"区域构造顺序：`{self.order}`。动态分区采用物理锚定。",
                              f"[分区诊断]({self.resources / 'regional_diagnostics.json'})"])
            lines.extend(["", f"[逐任务和汇总结果]({out})"])
            report.write_text("\n".join(lines) + "\n")
        return rows

    def run_phase(self, stage, phase, dynamic=True):
        jobs = self.jobs(stage, phase, dynamic=dynamic)
        self.schedule(stage, phase, jobs)
        return self.summarize(stage, phase, jobs)

    def begin(self):
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / "frozen_inputs.json").write_text(
            json.dumps({str(k): v for k, v in self.frozen.items()}, indent=2) + "\n")
        if self.args.stage == "pilot":
            self.execute("audit_regional_batch", "--config", self.config_path, "--stage", "p2",
                         "--out", self.results / "p2" / "cpu_audit.json", name="p2_cpu_audit")
            self.execute("prepare_region_assignments", "--config", self.config_path, "--out", self.resources)
            self.run_phase("p2", "smoke")
            self.run_phase("p2", "pilot")
            self.check_frozen()
            # Keep the paper's order fixed for P3. This makes it independent of the P2 comparison.
            tag = f"{self.order}_seed{self.config['development_seed']}"
            p2_result = self.phase_directory("p2", "pilot") / f"{tag}.json"
            while True:
                available = [gpu for gpu, (_, free, util) in self.gpu_status().items()
                             if free > self.config["planning_mib_per_job"] + self.config["minimum_free_mib"] and util <= 55]
                if available:
                    break
                time.sleep(15)
            log(f"P2 regional-graph and P3 membership diagnostics started on GPU {available[-1]}")
            self.execute("diagnose_dynamic_regions", "--config", self.config_path, "--result", p2_result,
                         "--out", self.resources, gpu=available[-1])
            diagnostic = json.loads((self.resources / "regional_diagnostics.json").read_text())
            dynamic = diagnostic["passed"]
            log(f"P3 dynamic membership diagnostic passed: {dynamic}")
            if dynamic:
                self.execute("audit_regional_batch", "--config", self.config_path, "--stage", "p3",
                             "--out", self.results / "p3" / "cpu_audit.json", name="p3_cpu_audit")
            self.run_phase("p3", "smoke", dynamic=dynamic)
            self.run_phase("p3", "pilot", dynamic=dynamic)
        elif self.args.stage == "p2-confirm":
            self.run_phase("p2", "confirm")
        else:
            diagnostic = json.loads((self.resources / "regional_diagnostics.json").read_text())
            self.run_phase("p3", "confirm", dynamic=diagnostic["passed"])
        log(f"Campaign phase {self.args.stage} complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/p2_p3_regions.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=["pilot", "p2-confirm", "p3-confirm"], default="pilot")
    parser.add_argument("--gpus", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--slots-per-gpu", type=int, choices=[1, 2], default=2)
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    runs = RegionalRuns(args)
    if args.print_commands:
        import shlex
        stages = [("p2", "smoke"), ("p2", "pilot"), ("p3", "smoke"), ("p3", "pilot")]
        if args.stage.endswith("-confirm"):
            stages = [(args.stage.split("-")[0], "confirm")]
        for stage, phase in stages:
            for job in runs.jobs(stage, phase):
                print(shlex.join(job.command))
    else:
        runs.begin()


if __name__ == "__main__":
    main()

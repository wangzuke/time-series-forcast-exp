"""Run the six joint-region jobs on GPU 6 and test the locked capacity pair."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = "missing_ts_exp.proj1_DRR_MSGF"
GPU = "6"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(message: str) -> None:
    print(f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class Campaign:
    def __init__(self, config: Path, run_id: str):
        self.config_path = config.resolve()
        self.cfg = read(self.config_path)
        self.results = PROJECT / "results" / run_id
        self.logs = PROJECT / "logs" / run_id
        self.resources = PROJECT / "resources" / run_id
        self.run_id = run_id
        self.environment = os.environ | {key: "2" for key in
                                         ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        self.environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED="1")
        monitored = (self.config_path, Path(__file__), PROJECT / "scripts/train_joint_regions.py",
                     PROJECT / "src/models/hdtts_official_reliable.py",
                     PROJECT / "src/training/run_mrmsgf_official_controlled.py",
                     PROJECT / "src/data/joint_regions.py")
        self.sources = {str(path): digest(path) for path in monitored}

    def wait_for_cache(self):
        pid = int((self.logs / "prepare.pid").read_text())
        manifest = self.resources / "manifest.json"
        while not manifest.is_file():
            stat = Path(f"/proc/{pid}/stat")
            if not stat.is_file() or stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z":
                raise RuntimeError("Partition preparation exited. See prepare.log.")
            time.sleep(5)
        return read(manifest)

    def available(self, active: int) -> bool:
        line = subprocess.check_output([
            "nvidia-smi", "--id=" + GPU, "--query-gpu=memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        total, free, utilization = map(int, line.strip().split(","))
        expected = self.cfg["planning_mib_per_job"]
        utilization_limit = self.cfg["max_utilization_to_add_job"] if active else 65
        return (
            active < self.cfg["gpu_slots"] and utilization <= utilization_limit
            and free >= expected + self.cfg["minimum_free_mib"] and total - free + expected <= total * 0.82
        )

    def phase(self, stage: str, caps: list[int]) -> dict:
        out = self.results / stage
        logs = self.logs / stage
        logs.mkdir(parents=True, exist_ok=True)
        queue = [(mode, cap) for cap in caps for mode in ("static", "dynamic")]
        tags = [f"{mode}_cap{cap}" for mode, cap in queue]
        active, errors = {}, []
        last_start = last_wait = 0.0
        while queue or active:
            for tag, (process, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[tag]
                log(f"{stage} {tag} finished with code {code}")
                if code != 0:
                    errors.append(tag)
            if queue and not errors and time.monotonic() - last_start >= 8 and self.available(len(active)):
                if any(digest(Path(p)) != h for p, h in self.sources.items()):
                    raise RuntimeError("Frozen experiment sources changed.")
                mode, cap = queue.pop(0)
                tag = f"{mode}_cap{cap}"
                command = [sys.executable, "-m", f"{PACKAGE}.scripts.train_joint_regions",
                           "--config", str(self.config_path), "--resources", str(self.resources),
                           "--cap", str(cap), "--mode", mode, "--stage", stage, "--out", str(out)]
                if stage == "test":
                    command.extend(("--selection", str(self.results / "selection.json")))
                path = logs / f"{tag}.log"
                handle = path.open("w")
                process = subprocess.Popen(command, env=self.environment, cwd=PROJECT.parents[1],
                                           stdout=handle, stderr=subprocess.STDOUT)
                active[tag] = (process, handle)
                last_start = time.monotonic()
                log(f"{stage} {tag} started on GPU {GPU}, PID {process.pid}, active {len(active)}")
            if errors and not active:
                raise RuntimeError(f"{stage} failed for {', '.join(errors)}. See {logs}.")
            if queue and time.monotonic() - last_wait >= 60:
                log(f"{stage}: {len(queue)} queued, {len(active)} running on GPU {GPU}")
                last_wait = time.monotonic()
            if queue or active:
                time.sleep(4)
        rows = {tag: read(out / f"{tag}.json") for tag in tags}
        for tag, row in rows.items():
            if row["config_sha256"] != self.sources[str(self.config_path)] or row["tag"] != tag:
                raise RuntimeError(f"Result protocol mismatch for {tag}.")
        save(out / "summary.json", rows)
        return rows

    def write_report(self, validation: dict, test: dict, chosen: int, cache: dict):
        report = PROJECT / "docs" / f"{self.run_id}_实验报告.md"
        lines = [
            "# 融合图联合分区与区域规模试运行", "",
            "METR-LA，Block-T 50%，seed 2024。所有对照使用同一冻结融合图生成器。",
            "区域数由分区产生。测试按验证集锁定的上限，仅在两个选中模型上评估。", "",
            "| 容量上限 | 分区 | 验证 MAE | 验证 RMSE | 验证实际区域数范围 | 最大区域 | 大于 8 节点的组占比 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in validation.values():
            kind = row["mode"]
            stats = cache["validation_partitions"][str(row["cap"])]["static" if kind == "static" else "validation_dynamic"]
            metrics = row["validation"]
            lines.append(f"| {row['cap']} | {kind} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
                         f"{stats['region_count_min']}–{stats['region_count_max']} | {stats['max_size']} | "
                         f"{stats['fraction_over_eight']:.2%} |")
        lines.extend(["", f"按两组平均验证 MAE，锁定容量 **{chosen}**。", "",
                      "| 分区 | 测试样本分组 | MAE | RMSE | 有效标签数 |",
                      "|---|---|---:|---:|---:|"])
        for row in test.values():
            for group, metrics in row["test"].items():
                mae, rmse = ("—", "—") if not metrics["n"] else (f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}")
                lines.append(f"| {row['mode']} | {group} | {mae} | {rmse} | {metrics['n']} |")
        static = test[f"static_cap{chosen}"]["test"]["all"]
        dynamic = test[f"dynamic_cap{chosen}"]["test"]["all"]
        summary = cache["validation_partitions"][str(chosen)]["validation_dynamic"]
        delta = dynamic["mae"] - static["mae"]
        larger = [str(c) for c in self.cfg["caps"] if
                  max(cache["validation_partitions"][str(c)][k]["max_size"]
                      for k in ("static", "validation_dynamic")) > 8]
        larger_text = ("、".join(larger) + " 的划分在验证中出现了超过 8 个节点的区域。"
                       if larger else "所有上限的划分在验证中都未出现超过 8 个节点的区域。")
        lines.extend(["", "## 分析与结论", "",
                      larger_text,
                      f"选定上限下，动态相对静态的测试 ΔMAE 为 {delta:+.4f}，"
                      f"ΔRMSE 为 {dynamic['rmse']-static['rmse']:+.4f}。"
                      f"验证窗中，动态分区平均有 {summary['mean_changed_nodes']:.2%} 的节点同组成员有变化。"
                      "分组结果采用相同的未来标签，反映实际发生成员变化的位置是否受益。",
                      "本轮为单 seed 对照，预测误差的方向与量级需要在独立种子上确认。", "",
                      f"[原始结果]({self.results}) · [分区诊断]({self.resources / 'manifest.json'})"])
        report.write_text("\n".join(lines) + "\n")

    def run(self):
        self.results.mkdir(parents=True, exist_ok=False)
        save(self.results / "frozen_sources.json", self.sources)
        log(f"Waiting for partition cache on GPU {GPU}")
        manifest = self.wait_for_cache()
        for cap, pair in manifest["validation_partitions"].items():
            stats = pair["validation_dynamic"]
            log(f"cap {cap}: validation regions {stats['region_count_min']}–{stats['region_count_max']}, "
                f"maximum size {stats['max_size']}, fraction above 8: {stats['fraction_over_eight']:.2%}")
        self.phase("smoke", self.cfg["caps"])
        validation = self.phase("pilot", self.cfg["caps"])
        scores = {cap: sum(validation[f"{mode}_cap{cap}"]["validation"]["mae"]
                           for mode in ("static", "dynamic")) / 2 for cap in self.cfg["caps"]}
        chosen = min(scores, key=lambda cap: (scores[cap], cap))
        checkpoints = {f"{mode}_cap{chosen}": validation[f"{mode}_cap{chosen}"]["checkpoint_sha256"]
                       for mode in ("static", "dynamic")}
        save(self.results / "selection.json",
             dict(cap=chosen, validation_mean_mae=scores, checkpoints=checkpoints,
                  config_sha256=self.sources[str(self.config_path)]))
        log(f"Locked capacity {chosen}; evaluating only the matched test pair")
        test = self.phase("test", [chosen])
        self.write_report(validation, test, chosen, manifest)
        log("Campaign complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/joint_regions.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    Campaign(args.config, args.run_id).run()


if __name__ == "__main__":
    main()

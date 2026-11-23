#!/usr/bin/env python3
"""GPU 0/5 experiment supervisor with a pilot barrier and durable status.

Each task trains once, evaluates all fixed masks and writes resumable outputs.
Failures stop new scheduling, leaving independent running tasks intact.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time

from protocol import EXP_ROOT, atomic_json_dump
from posterior_s4_common import SUITE, VARIANTS, source_fingerprints

LOG_ROOT = EXP_ROOT / "logs" / SUITE
MANIFEST = EXP_ROOT / "configs" / f"{SUITE}.json"
RUNNER = EXP_ROOT / "scripts/run_posterior_s4.py"
DATASETS = ("Electricity", "ETTh1")


def build_manifest(args):
    jobs = []
    for dataset in DATASETS:
        for seed in (2024, 2025):
            for family in ("P", "B"):
                for variant in VARIANTS:
                    job_id = f"{dataset}_{variant}_{family}_s{seed}"
                    jobs.append(dict(id=job_id, dataset=dataset, variant=variant, family=family, seed=seed,
                                     phase="pilot" if dataset == DATASETS[0] and seed == 2024 else "core"))
    manifest = {"suite": SUITE, "jobs": jobs, "batch_size": args.batch_size, "effective_batch": 64,
            "workers": args.workers, "threads": args.threads, "slots_per_gpu": args.slots_per_gpu,
            "gpus": args.gpus, "epochs": 30, "patience": 5, "source": source_fingerprints(),
            "test_families": ["P", "B", "B-uniform"], "test_mask_seeds": [9101, 9102],
            "total_training_jobs": len(jobs), "pilot_training_jobs": sum(j["phase"] == "pilot" for j in jobs),
            "selection_note": "frozen configuration; no automatic test-driven changes"}
    if args.weather:
        manifest.update(extension_of="posterior_s4_r50_b24_l96_h96_1119", require_idle_gpus=True,
                        trainer_adapter_changes=["dataset lookup", "suite namespace", "source provenance"])
    return manifest


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def complete(job):
    root = EXP_ROOT / "results" / SUITE / job["id"]
    record = read_json(root / "progress.json")
    if record.get("status") != "complete":
        return False
    return all(read_json(root / f"test_{family}_m{seed}.json").get("status") == "complete"
               for family in ("P", "B", "B-uniform") for seed in (9101, 9102))


def command(job, manifest):
    return [sys.executable, "-u", str(RUNNER),
            "--variant", job["variant"], "--dataset", job["dataset"], "--family", job["family"],
            "--seed", str(job["seed"]), "--job-id", job["id"], "--batch-size", str(manifest["batch_size"]),
            "--workers", str(manifest["workers"]), "--threads", str(manifest["threads"]),
            "--epochs", str(manifest["epochs"]), "--patience", str(manifest["patience"])]


def gpu_status(gpus, own_pids=()):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    records = {}
    uuid_to_index = {}
    for line in output.strip().splitlines():
        index, uuid, used, free, util = [x.strip() for x in line.split(",")]
        uuid_to_index[uuid] = index
        if index in gpus:
            records[index] = {"used_mib": int(used), "free_mib": int(free), "utilization_percent": int(util), "foreign_pids": []}
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader,nounits"], text=True, timeout=15)
    for line in processes.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [x.strip() for x in line.split(",")]
        index = uuid_to_index.get(uuid)
        if index in records and int(pid) not in own_pids:
            records[index]["foreign_pids"].append(int(pid))
    return records


def main(args):
    global SUITE, LOG_ROOT, MANIFEST, RUNNER, DATASETS, source_fingerprints
    if args.weather:
        from run_posterior_s4_weather import SUITE as weather_suite, source_fingerprints as weather_sources
        SUITE, RUNNER, DATASETS = weather_suite, EXP_ROOT / "scripts/run_posterior_s4_weather.py", ("Weather",)
        LOG_ROOT = EXP_ROOT / "logs" / SUITE
        MANIFEST = EXP_ROOT / "configs" / f"{SUITE}.json"
        source_fingerprints = weather_sources
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    proposed = build_manifest(args)
    if MANIFEST.exists():
        if read_json(MANIFEST) != proposed:
            raise RuntimeError("Existing manifest differs; do not overwrite a launched configuration")
    else:
        atomic_json_dump(proposed, MANIFEST)
    if args.build_only:
        print(json.dumps({"manifest": str(MANIFEST), "jobs": len(proposed["jobs"])}), flush=True)
        return
    manifest = proposed
    correctness = read_json(EXP_ROOT / "diagnostics" / SUITE / "correctness.json")
    if correctness.get("status") != "passed":
        raise RuntimeError("Missing successful correctness/CUDA preflight")
    if correctness.get("source") != manifest["source"]:
        raise RuntimeError("Correctness preflight does not match current source fingerprints")
    with (LOG_ROOT / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        running, failures = {}, []
        selected = [job for job in manifest["jobs"] if args.phase == "all" or job["phase"] == "pilot"]
        status_path = LOG_ROOT / "supervisor_status.json"
        last_snapshot = 0
        while True:
            for key, info in list(running.items()):
                code = info["process"].poll()
                if code is None:
                    continue
                info["log"].close()
                if code != 0 or not complete(info["job"]):
                    failures.append({"id": info["job"]["id"], "returncode": code,
                                     "gpu": info["gpu"], "timestamp": time.time()})
                print(json.dumps({"finished": info["job"]["id"], "returncode": code}), flush=True)
                del running[key]
            completed = [job["id"] for job in selected if complete(job)]
            active = {info["job"]["id"] for info in running.values()}
            pending = [job for job in selected if job["id"] not in set(completed) | active]
            pilot_done = all(complete(job) for job in selected if job["phase"] == "pilot")
            if pilot_done and not (LOG_ROOT / "pilot_summary.json").exists():
                atomic_json_dump(summarize(manifest, pilot_only=True), LOG_ROOT / "pilot_summary.json")
            available = [job for job in pending if job["phase"] == "pilot" or pilot_done]
            devices = gpu_status(args.gpus, {info["process"].pid for info in running.values()})
            reserved = {gpu: 0 for gpu in args.gpus}
            if not failures:
                for gpu in args.gpus:
                    if args.weather and devices[gpu]["foreign_pids"]:
                        continue
                    for slot in range(args.slots_per_gpu):
                        key = (gpu, slot)
                        if key in running or not available:
                            continue
                        if devices[gpu]["free_mib"] - reserved[gpu] < args.min_free_mib:
                            continue
                        job = available.pop(0)
                        stream = (LOG_ROOT / f"{job['id']}.log").open("a")
                        env = os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(args.threads),
                                   MKL_NUM_THREADS=str(args.threads), MPLCONFIGDIR="/tmp/posterior_s4_mpl")
                        process = subprocess.Popen(command(job, manifest), stdout=stream, stderr=subprocess.STDOUT,
                                                   env=env, cwd=str(EXP_ROOT.parents[2]))
                        running[key] = dict(job=job, gpu=gpu, process=process, log=stream, start=time.time())
                        print(json.dumps({"started": job["id"], "gpu": gpu, "slot": slot, "pid": process.pid}), flush=True)
                        # Reserve headroom for processes still constructing
                        # their CUDA context. The exact peak is preflighted.
                        reserved[gpu] += args.reserve_mib
            status = {"status": "failed" if failures else ("complete" if len(completed) == len(selected) else "running"),
                      "pid": os.getpid(), "updated_at": time.time(), "jobs_total": len(selected),
                      "jobs_complete": completed, "pilot_complete": pilot_done, "failures": failures,
                      "active": [{"id": i["job"]["id"], "gpu": i["gpu"], "pid": i["process"].pid,
                                  "start": i["start"]} for i in running.values()], "devices": devices}
            atomic_json_dump(status, status_path)
            if time.time() - last_snapshot >= 60:
                with (LOG_ROOT / "monitor.jsonl").open("a") as handle:
                    handle.write(json.dumps(status) + "\n")
                last_snapshot = time.time()
                subprocess.run([sys.executable, str(EXP_ROOT / "scripts/report_posterior_s4.py"), "--suite", SUITE], check=True)
            if not running and (failures or len(completed) == len(selected)):
                summary_root = EXP_ROOT / "summary" / SUITE
                atomic_json_dump(summarize(manifest), summary_root / "results.json")
                subprocess.run([sys.executable, str(EXP_ROOT / "scripts/report_posterior_s4.py"), "--suite", SUITE], check=True)
                if not failures and args.weather:
                    subprocess.run([sys.executable, str(EXP_ROOT / "scripts/analyze_posterior_s4_results.py"), "--suite", SUITE], check=True)
                break
            time.sleep(10)
        if failures:
            raise SystemExit("Experiment failure recorded; new scheduling stopped. See supervisor_status.json")


def summarize(manifest, pilot_only=False):
    rows = []
    for job in manifest["jobs"]:
        if pilot_only and job["phase"] != "pilot":
            continue
        for family in ("P", "B", "B-uniform"):
            for seed in (9101, 9102):
                path = EXP_ROOT / "results" / SUITE / job["id"] / f"test_{family}_m{seed}.json"
                result = read_json(path)
                if result.get("status") == "complete":
                    rows.append({**job, "test_family": family, "mask_seed": seed,
                                 **{k: result[k] for k in ("mse_standardized", "mae_standardized", "mse_raw", "mae_raw")}})
    cells = []
    for dataset in sorted({j["dataset"] for j in manifest["jobs"]}):
        for variant in VARIANTS:
            for train_seed in (2024, 2025):
                grid = {}
                for tf in ("P", "B"):
                    for ef in ("P", "B"):
                        values = [r["mse_standardized"] for r in rows if r["dataset"] == dataset and r["variant"] == variant
                                  and r["seed"] == train_seed and r["family"] == tf and r["test_family"] == ef]
                        if len(values) == 2:
                            grid[f"{tf}->{ef}"] = sum(values) / 2
                if len(grid) == 4:
                    cells.append(dict(dataset=dataset, variant=variant, train_seed=train_seed, **grid,
                                      B_mismatch_cost=grid["P->B"] / grid["B->B"] - 1,
                                      P_mismatch_cost=grid["B->P"] / grid["P->P"] - 1))
    return {"rows": rows, "paired_seed_matrices": cells, "updated_at": time.time(),
            "interpretation": "two mask seeds are repeated evaluation, not independent training"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", choices=("0", "5"), default=["0", "5"])
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--phase", choices=("pilot", "all"), default="all")
    parser.add_argument("--reserve-mib", type=int, default=5000)
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--weather", action="store_true", help="isolated Weather extension with frozen training logic")
    args = parser.parse_args()
    main(args)

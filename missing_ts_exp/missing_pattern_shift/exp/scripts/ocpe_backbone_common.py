"""Frozen identity, paired S4M warm starts and audited legacy S4 reuse."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import default_collate
from protocol import EXP_ROOT, atomic_torch_save
from ocpe_fusion_common import (FusionDataset, clean_dataset, mask_fingerprint, mask_audit,
    ModelConfig, seed_all, digest, move_batch, memory_stats, runtime_metadata, sha_file)
from ocpe_s4m import OCPES4M
from run_posterior_s4 import rng_state, restore_rng

SUITE = "ocpe_backbone_uniform_r50_l96_h96_1123"
LEGACY = "ocpe_fusion_uniform_r50_l96_h96_1120"
VARIANTS = ("S4", "OCPE-S4", "S4M", "OCPE-S4M")
LABELS = {v: v for v in VARIANTS}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def source_fingerprints():
    paths = [EXP_ROOT / "models" / f for f in ("posterior_s4.py", "ocpe_fusion_s4.py", "ocpe_s4m.py")]
    paths += [EXP_ROOT / "scripts" / f for f in (
        "protocol.py", "posterior_s4_common.py", "ocpe_fusion_common.py", "run_ocpe_fusion_s4.py",
        "run_posterior_s4.py", "run_official_baseline.py", "s4m_ablation.py", "ocpe_backbone_common.py",
        "run_ocpe_s4m.py", "validate_ocpe_backbone.py", "run_ocpe_backbone_suite.py", "report_ocpe_backbone.py")]
    paths += sorted((EXP_ROOT / "baselines/S4M/s4m/model").rglob("*.py"))
    paths += sorted((EXP_ROOT / "baselines/S4M/s4m/layers").rglob("*.py"))
    return {str(p.relative_to(EXP_ROOT)): sha_file(p) for p in paths}


def build_jobs():
    jobs = []
    for seed in (2024, 2025):
        for family in ("P", "Bu"):
            for dataset in ("ETTh1", "Weather", "Electricity"):
                for variant in VARIANTS:
                    reuse = dataset != "ETTh1" and variant in ("S4", "OCPE-S4")
                    oldvariant = "F3" if variant == "OCPE-S4" else "F0"
                    job_id = f"{dataset}_{variant}_{family}_s{seed}"
                    jobs.append({"id": job_id, "dataset": dataset, "variant": variant,
                                 "family": family, "seed": seed, "reused": reuse,
                                 "result_suite": LEGACY if reuse else SUITE,
                                 "result_job_id": f"{dataset}_{oldvariant}_{family}_s{seed}" if reuse else job_id})
    return jobs


def directory(job):
    return EXP_ROOT / "results" / job["result_suite"] / job["result_job_id"]


def complete(job):
    root = directory(job)
    protocol = read(root / "protocol.json")
    progress = read(root / "progress.json")
    if progress.get("status") != "complete" or progress.get("fingerprint") != protocol.get("fingerprint"):
        return False
    return bool(protocol) and all(read(root / f"test_{f}_m{s}.json").get("fingerprint") == protocol["fingerprint"]
        and read(root / f"test_{f}_m{s}.json").get("status") == "complete"
        for f in ("P", "Bu") for s in (9101, 9102))


def audit_reuse(jobs):
    from ocpe_fusion_common import source_fingerprints as legacy_sources
    old_manifest = read(EXP_ROOT / "configs" / f"{LEGACY}.json")
    if old_manifest["source"] != legacy_sources():
        raise RuntimeError("Legacy dependencies changed; cannot silently reuse S4 results")
    records = []
    for job in jobs:
        if not job["reused"]:
            continue
        root = directory(job)
        p = read(root / "protocol.json")
        assert complete(job), job["id"]
        assert p["source"] == old_manifest["source"]
        expected = "F3" if job["variant"] == "OCPE-S4" else "F0"
        assert (p["variant"], p["epochs"], p["patience"], p["batch_size"], p["effective_batch"], p["aux_weight"]) == (expected, 30, 5, 64, 64, .1)
        train = FusionDataset(job["dataset"], "train", job["family"], job["seed"], True)
        val = FusionDataset(job["dataset"], "val", job["family"], job["seed"] + 10000)
        for key in ("data_sha256", "validity_sha256", "mean_sha256", "std_sha256"):
            assert p["data_quality"][key] == train.quality[key]
        assert p["train_mask_epoch1"] == mask_fingerprint(train, 1)
        assert p["train_mask_epoch2"] == mask_fingerprint(train, 2)
        assert p["validation_effective_mask"] == mask_fingerprint(val, effective=True)
        for family in ("P", "Bu"):
            for seed in (9101, 9102):
                ds = FusionDataset(job["dataset"], "test", family, seed)
                assert read(root / f"test_{family}_m{seed}.json")["mask_fingerprint"] == mask_fingerprint(ds)
        paths = [root / "protocol.json", root / "training_complete.json"]
        paths += [root / f"test_{f}_m{s}.json" for f in ("P", "Bu") for s in (9101, 9102)]
        paths += [EXP_ROOT / "checkpoints" / LEGACY / f"{job['result_job_id']}.best.pt"]
        records.append({"id": job["id"], "fingerprint": p["fingerprint"],
                        "files": {str(f.relative_to(EXP_ROOT)): sha_file(f) for f in paths}})
    return records


def memory_digest(state):
    h = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            h.update(str((array.shape, array.dtype)).encode()); h.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(str(len(value)).encode())
            for item in value: visit(item)
        else: h.update(repr(value).encode())
    visit(state)
    return h.hexdigest()


def initialize_memory(model, ds, seed, suite=SUITE):
    key = f"{ds.dataset}_{ds.family}_s{seed}"
    root = EXP_ROOT / "checkpoints" / suite / "warm"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pt"
    identity = digest({"source": source_fingerprints(), "quality": ds.quality,
                       "seed": seed, "family": ds.family, "warm_batch": 16, "eval_mode": True})
    saved_rng, training, epoch = rng_state(), model.training, ds.epoch
    with (root / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["identity"] != identity:
                    raise RuntimeError("Warm-start identity differs")
            else:
                ds.set_epoch(1)
                order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed + 100001))[:16].tolist()
                batch = move_batch(default_collate([ds[i] for i in order]), next(model.parameters()).device)
                model.eval()
                with torch.no_grad(): model.backbone.mem_net.warm(batch["x"])
                state = model.memory_state(cpu=True)
                saved = {"identity": identity, "memory": state, "memory_sha256": memory_digest(state),
                         "indices": order, "mask_sha256": hashlib.sha256(batch["mask"].cpu().numpy().tobytes()).hexdigest()}
                atomic_torch_save(saved, path)
            model.load_memory(saved["memory"])
        finally:
            ds.set_epoch(epoch); model.train(training); restore_rng(saved_rng)
    return {"path": str(path), "file_sha256": sha_file(path), "identity": identity,
            "memory_sha256": saved["memory_sha256"]}


def make_s4m_optimizers(model):
    lr = {"ETTh1": .005, "Weather": .001, "Electricity": .01}[model.dataset]
    optimizers = {"backbone": torch.optim.SGD([p for p in model.backbone.parameters() if p.requires_grad],
                                             lr=lr, momentum=.9, weight_decay=1e-5)}
    if model.enabled:
        optimizers["ocpe"] = torch.optim.AdamW(model.ocpe_parameters(), lr=.001, weight_decay=.0001)
    return optimizers

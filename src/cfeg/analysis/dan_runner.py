"""Single-shot DAN programme. No resume, tuning, raw retry or held60 switches."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.dan_data import SOURCE_IDS, extract_human, load_cache, save_cache
from cfeg.analysis.dan_report import summarize
from cfeg.analysis.dan_roles import FoldSupport, FreezeGate
from cfeg.analysis.dan_runtime import Meter, now, save_npz, save_torch, sha_file, write_json
from cfeg.analysis.dan_signal import EnsembleTRCA, fbcca_scores, filter_prefix, fit_ensemble_trca
from cfeg.analysis.dan_training import align_sources

ROOT = Path(__file__).resolve().parents[3]
PLAN_SHA = "ca734672afc122d9fc4f36a662fa73bbfd82a14c68b32c25bdc38f9bdd9f84f6"
CONFIG_SHA = "af3c075b483fbb447748bdb8b9784904cacecf9fee19d0fa8a5c601878f3b922"
ARTIFACT_PATHS = [
    "src/cfeg/models/dan_alignment.py", "src/cfeg/analysis/dan_teacher.py",
    "src/cfeg/analysis/dan_signal.py", "src/cfeg/analysis/dan_training.py",
    "src/cfeg/analysis/dan_roles.py", "src/cfeg/analysis/dan_runtime.py",
    "src/cfeg/analysis/dan_data.py", "src/cfeg/analysis/dan_report.py",
    "src/cfeg/analysis/dan_runner.py", "src/cfeg/analysis/dan_audit.py",
    "src/cfeg/analysis/joint_harmonic_data.py", "src/cfeg/analysis/joint_harmonic.py",
    "src/cfeg/data/prepare_mat.py", "src/cfeg/data/prepare_wearable.py",
    "tests/test_dan_full_flow.py", "tests/test_dan_cuda_qualification.py"]


def _save_decoder(output, name, calibration):
    decoder = fit_ensemble_trca(calibration)
    save_npz(output / name, filters=decoder.filters, templates=decoder.templates)
    return sha_file(output / name)


def execute(output: Path, config: dict, meter: Meter, *, device: str,
            generated_targets: tuple[int, ...] | None = None) -> dict:
    if generated_targets is not None and not meter.generated:
        raise ValueError("No human target subset override.")
    data, vault = load_cache(output, meter)
    targets = tuple(data.ids if generated_targets is None else generated_targets)
    interfaces = range(1 if meter.generated else 2)
    if not set(targets) <= set(data.ids):
        raise ValueError("Target subset absent from cache.")
    cells = set(itertools.product(targets, interfaces, config["budgets"], config["arms"],
                                  config["training"]["seeds"], range(3)))
    gate = FreezeGate(cells)
    artifacts, models, decoders, matched, roles = {}, [], [], {}, {}
    for fold in range(3):
        role = FoldSupport(data, fold, sham_seed=config["splits"]["sham_seed"])
        roles[str(fold)] = {"source": list(role.source_ids), "fit": list(role.fit_ids),
                            "target": list(role.target_ids), "donors": role.donors,
                            "scalers": {str(key): {"mean": value.mean.tolist(),
                                                   "scale": value.scale.tolist()}
                                        for key, value in role.scalers.items()}}
        for target in role.target_ids:
            if target not in targets:
                continue
            row_index = data.ids.index(target)
            for interface in interfaces:
                for k in config["budgets"]:
                    name = f"decoder_{target}_{interface}_{k}_TRCA.npz"
                    artifacts[name] = _save_decoder(output, name, data.bands[row_index, interface, :, :k])
                    decoders.append({"target": target, "interface": interface, "k": k,
                                     "arm": "TRCA", "seed": -1, "file": name})
                    for seed in config["training"]["seeds"]:
                        for arm in config["arms"]:
                            augmented, band_files = [], []
                            for band in range(3):
                                source, support, weights = role.packet(target, interface, band, k, arm)
                                meter.consume("adaptation_cells")
                                meter.active_training_since = time.monotonic()
                                fitted = align_sources(source, support, weights, config["training"],
                                                       seed=seed + band, device=device,
                                                       consume=meter.consume, check=meter.check)
                                meter.training_seconds += time.monotonic() - meter.active_training_since
                                meter.active_training_since = None
                                signature = (fitted.initial_sha256,
                                             tuple(r["batch_order_sha256"] for r in fitted.records))
                                pair = (target, interface, k, seed, band)
                                if pair in matched and matched[pair] != signature:
                                    raise ValueError("Paired arms differ in initialization or exposure.")
                                matched[pair] = signature
                                name = f"align_{target}_{interface}_{k}_{arm}_{seed}_{band}.pt"
                                key = (target, interface, k, arm, seed, band)
                                save_torch(output / name, {
                                    "key": key, "fold": fold, "source_ids": list(role.source_ids),
                                    "fit_ids": list(role.fit_ids), "states": fitted.states,
                                    "records": fitted.records, "teacher": torch.from_numpy(fitted.teacher),
                                    "weights": torch.from_numpy(weights), "initial_sha256": fitted.initial_sha256})
                                artifacts[name] = sha_file(output / name)
                                gate.register(key, artifacts[name])
                                models.append({"key": key, "file": name, "fold": fold})
                                band_files.append(name)
                                augmented.append(np.concatenate((support, fitted.transformed)))
                            name = f"decoder_{target}_{interface}_{k}_{arm}_{seed}.npz"
                            artifacts[name] = _save_decoder(output, name, np.stack(augmented))
                            decoders.append({"target": target, "interface": interface, "k": k,
                                             "arm": arm, "seed": seed, "file": name,
                                             "alignment_files": band_files})
                            meter.event("decoder_frozen", target=target, interface=interface,
                                        k=k, arm=arm, seed=seed,
                                        completed_cells=meter.counts["adaptation_cells"])
    gate.seal()
    freeze = {"at": now(), "generated": meter.generated, "artifacts": artifacts,
              "models": models, "decoders": decoders, "roles": roles, "targets": list(targets),
              "interfaces": list(interfaces), "cache_sha256": sha_file(output / "cache.npz"),
              "config_sha256": sha_file(output / "config.json"), "counts": meter.counts.copy()}
    write_json(output / "freeze.json", freeze)
    meter.event("all_models_frozen", cells=len(cells), freeze_sha256=sha_file(output / "freeze.json"))
    meter.consume("outer_reveal_batches")
    prefixes = vault.reveal(gate)
    scores, rows, query_bands = [], [], {}
    classes = len(config["frequencies"])
    for target in targets:
        for interface in interfaces:
            query = filter_prefix(prefixes[data.ids.index(target), interface], config)
            query = query.reshape(3, -1, len(config["channels"]), config["n_samples"])
            query_bands[target, interface] = query
            score = fbcca_scores(query, np.array(config["frequencies"]), sfreq=config["sfreq"])
            scores.append(score)
            rows.append({"target": target, "interface": interface, "k": 0,
                         "arm": "CCA", "seed": -1, "score_index": len(scores) - 1})
    for entry in decoders:
        with np.load(output / entry["file"], allow_pickle=False) as packet:
            model = EnsembleTRCA(packet["filters"], packet["templates"])
        scores.append(model.scores(query_bands[entry["target"], entry["interface"]]))
        rows.append({**entry, "score_index": len(scores) - 1})
        meter.check()
    labels = np.tile(np.arange(classes), scores[0].shape[0] // classes)
    for row, score in zip(rows, scores):
        row["correct"] = int(np.count_nonzero(score.argmax(1) == labels))
        row["trials"] = len(labels)
        row["accuracy"] = row["correct"] / row["trials"]
    save_npz(output / "scores.npz", scores=np.stack(scores), labels=labels)
    report = summarize(rows, config, generated=meter.generated)
    report.update(freeze_sha256=sha_file(output / "freeze.json"),
                  scores_sha256=sha_file(output / "scores.npz"), meter=meter.snapshot())
    write_json(output / "result.json", report)
    meter.event("final_query_batch_complete", score_rows=len(rows))
    return report


def validate_qualification(root: Path, qualification: Path) -> tuple[dict, dict]:
    if sha_file(root / "docs/dan_teacher_v1_plan.md") != PLAN_SHA:
        raise ValueError("Science plan changed.")
    path = root / "configs/analysis/dan_teacher_v1.json"
    if sha_file(path) != CONFIG_SHA:
        raise ValueError("Science config changed.")
    config = json.loads(path.read_text())
    receipt = json.loads(qualification.read_text())
    if (receipt.get("status") != "PASS_FULL_GENERATED_QUALIFICATION"
            or receipt.get("plan_sha256") != PLAN_SHA or receipt.get("config_sha256") != CONFIG_SHA
            or receipt.get("human_entry_qualified") is not True
            or set(receipt.get("artifact_sha256", {})) != set(ARTIFACT_PATHS)):
        raise ValueError("Full generated qualification missing/incomplete.")
    for relative, digest in receipt["artifact_sha256"].items():
        if sha_file(root / relative) != digest:
            raise ValueError(f"Qualified artifact changed: {relative}")
    for relative, digest in receipt["evidence_sha256"].items():
        if sha_file(root / relative) != digest:
            raise ValueError("Qualification evidence changed.")
    if tuple(config["source_subject_ids"]) != SOURCE_IDS:
        raise ValueError("Participant allowlist changed.")
    return config, receipt


def human_entry(qualification: Path) -> Path:
    config, receipt = validate_qualification(ROOT, qualification)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for qualified human run.")
    if shutil.disk_usage("/home/whwovy/eeg-data").free < config["limits"]["new_output_bytes"] + (6 << 30):
        raise RuntimeError("Insufficient disk headroom for the bounded run.")
    utilization = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    if int(utilization.splitlines()[torch.cuda.current_device()]) > 10:
        raise RuntimeError("GPU is busy; do not interfere with another workload.")
    free, total = torch.cuda.mem_get_info()
    if free < config["limits"]["gpu_allocated_bytes"] + (2 << 30):
        raise RuntimeError("Insufficient free GPU memory; do not evict other workloads.")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(config["limits"]["gpu_allocated_bytes"] / total)
    torch.cuda.reset_peak_memory_stats()
    output = Path(tempfile.mkdtemp(prefix="dan-teacher-human-v1-", dir="/home/whwovy/eeg-data"))
    start_path = ROOT / "docs/reports/dan_teacher_v1_human_start.json"
    write_json(start_path, {"at": now(), "output": str(output),
                            "qualification_sha256": sha_file(qualification), "no_retry": True})
    write_json(output / "config.json", config)
    write_json(output / "qualification.json", receipt)
    meter = Meter(output, config["limits"], generated=False)

    def deadline(signum, frame):
        raise TimeoutError("DAN human wall-time ceiling reached.")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config["limits"]["human_wall_seconds"])
    try:
        meter.event("human_start", output=str(output))
        arrays, provenance = extract_human(config, meter)
        digest = save_cache(output, arrays)
        del arrays
        write_json(output / "provenance.json", {**provenance, "cache_sha256": digest})
        execute(output, config, meter, device="cuda")
        from cfeg.analysis.dan_audit import audit_saved
        meter.consume("saved_audit_batches")
        meter.consume("human_cache_loads")
        audit = audit_saved(output)
        write_json(output / "audit.json", audit)
        meter.event("audit_complete", status=audit["status"])
        write_json(output / "terminal.json", {"status": "COMPLETE_AUDITED", "at": now(),
                                              "meter": meter.snapshot()})
    except BaseException as error:
        write_json(output / "terminal.json", {"status": "FAILED_NO_RETRY", "at": now(),
                                              "error": str(error), "traceback": traceback.format_exc(),
                                              "meter": meter.snapshot()})
        raise
    finally:
        signal.alarm(0)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    print(human_entry(args.qualification))

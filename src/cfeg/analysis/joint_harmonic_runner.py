"""Finite source39 joint-learning runner, with a separate generated-data entry.

There is no retry, resume, threshold tuning or new candidate flag. The human
entry requires a pinned qualification receipt and an exclusive one-shot start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.joint_harmonic import (
    ARMS,
    ContextScaler,
    direct_impedance_gain,
    participant_splits,
    raw_prototype_scores,
    regularized_cca_scores,
    select_learning_rate,
    select_policy,
)
from cfeg.analysis.joint_harmonic_data import (
    SOURCE_IDS,
    RoleData,
    prefix_contexts,
    read_human_cache,
    select_role,
)
from cfeg.models.joint_harmonic import JointHarmonicPrototype

PLAN_SHA = "ef32b8c245a8e4f314bd18965b63e5106dd1435fc223e63fc21691e1bac4f28e"
CONFIG_SHA = "0d302b418ff9da5a338a59a63e2b471615f0b7388fe244c70570055c987c8627"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict, *, exclusive: bool = True) -> None:
    """Durable fresh artifacts; only the latest meter snapshot may be replaced."""
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if exclusive:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".next")
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def save_npz(path: Path, **arrays) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


class Meter:
    def __init__(self, output: Path, limits: dict, *, generated: bool):
        self.output, self.limits, self.generated = output, limits, generated
        self.started = time.monotonic()
        self.counts: dict[str, int] = {}
        self.training_seconds = 0.0
        self.peak_gpu_bytes = 0
        self.events: list[dict] = []
        self.flush()

    def consume(self, key: str, amount: int = 1) -> None:
        value = self.counts.get(key, 0) + amount
        if amount < 0 or key not in self.limits or value > self.limits[key]:
            raise RuntimeError(f"Budget exceeded or unknown counter: {key}={value}")
        self.counts[key] = value
        if key not in {"human_updates", "human_episodes"}:
            self.flush()

    def snapshot(self) -> dict:
        return {"generated": self.generated, "counts": self.counts,
                "wall_seconds": time.monotonic() - self.started,
                "training_seconds": self.training_seconds,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_gpu_allocated_bytes": self.peak_gpu_bytes}

    def flush(self) -> None:
        write_json(self.output / "meter.json", self.snapshot(), exclusive=False)

    def event(self, name: str, **fields) -> None:
        row = {"event": name, "sequence": len(self.events), "at": utc_now(), **fields}
        self.events.append(row)
        with (self.output / "journal.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.flush()
        if not self.generated:
            print(json.dumps(row), flush=True)

    def check(self) -> None:
        if torch.cuda.is_available():
            self.peak_gpu_bytes = max(self.peak_gpu_bytes, torch.cuda.max_memory_allocated())
        if self.generated:
            return
        checks = {
            "human_wall_seconds": time.monotonic() - self.started,
            "training_seconds": self.training_seconds,
            "ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpu_allocated_bytes": self.peak_gpu_bytes,
            "output_bytes": sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()),
        }
        for key, value in checks.items():
            if value > self.limits[key]:
                raise RuntimeError(f"Resource ceiling exceeded: {key}={value}")
        self.flush()


@dataclass
class Fitted:
    model: JointHarmonicPrototype
    scaler: ContextScaler
    record: dict


def model_from_config(config: dict) -> JointHarmonicPrototype:
    return JointHarmonicPrototype(n_channels=len(config["channels"]),
                                  n_classes=len(config["frequencies"]),
                                  n_harmonics=config["n_harmonics"], **config["model"])


def state_digest(weights: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_one(role: RoleData, config: dict, *, arm: str, lr: float, seed: int,
              fold: int, kind: str, device: str, meter: Meter) -> Fitted:
    meter.consume("human_fits")
    fit_id = f"{kind}-fold{fold}-{arm}-lr{lr:g}-seed{seed}"
    meter.event("fit_start", fit_id=fit_id, fit_subject_ids=list(role.ids))
    torch.manual_seed(seed + 100 * fold)
    model = model_from_config(config).to(device)
    initial_sha = state_digest(model.state_dict())
    raw_contexts, donors = prefix_contexts(role, arm, sham_seed=config["splits"]["sham_seed"] + fold)
    source_ids = np.repeat(role.ids, 6).tolist()
    scaler = ContextScaler.fit(raw_contexts.reshape(-1, len(config["channels"]), 10),
                               subject_ids=source_ids, allowed_fit_ids=role.ids)
    contexts = torch.tensor(scaler.transform(raw_contexts), dtype=torch.float32, device=device)
    spectra = torch.tensor(role.spectra, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=config["training"]["weight_decay"])
    rng = np.random.default_rng(seed + 100 * fold + 17000)
    classes, channels, features = spectra.shape[3:]
    batch = config["training"]["episodes_per_step"]
    targets = torch.arange(classes, device=device).repeat(batch)
    losses, schedule_hash = [], hashlib.sha256()
    fit_start = time.monotonic()
    last_resource_check = fit_start
    for step in range(config["training"]["steps"]):
        if not meter.generated:
            now = time.monotonic()
            if now - meter.started > config["limits"]["human_wall_seconds"]:
                raise TimeoutError("Human program wall-time budget exhausted.")
            if meter.training_seconds + now - last_resource_check > config["limits"]["training_seconds"]:
                raise TimeoutError("Training wall-time budget exhausted.")
        ki, k = step % 3, config["budgets"][step % 3]
        people = rng.integers(len(role.ids), size=batch)
        interfaces = rng.integers(2, size=batch)
        blocks = rng.integers(5, 10, size=batch)
        schedule_hash.update(np.stack((people, interfaces, blocks)).astype("<i8").tobytes())
        p, e, b = [torch.as_tensor(v, device=device) for v in (people, interfaces, blocks)]
        support = spectra[p, e, :k].reshape(batch, k * classes, channels, features)
        query = spectra[p, e, b]
        labels = torch.arange(classes, device=device).repeat(batch, k)
        optimizer.zero_grad(set_to_none=True)
        logits = model(support, labels, query, contexts[p, e, ki])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite source training loss.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad_norm"],
                                              error_if_nonfinite=True)
        meter.consume("human_updates")
        meter.consume("human_episodes", batch)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == config["training"]["steps"]:
            losses.append({"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm)})
            now = time.monotonic()
            meter.training_seconds += now - last_resource_check
            last_resource_check = now
            meter.check()
    if device == "cuda":
        torch.cuda.synchronize()
    meter.training_seconds += time.monotonic() - last_resource_check
    model.eval()
    actuation = None
    if arm == "QM":
        with torch.no_grad():
            c = contexts[:1, 0, 0].clone()
            changed = c.clone()
            changed[..., 2:6] = contexts[1:2, 0, 0, :, 2:6]
            s = spectra[:1, 0, :1].reshape(1, classes, channels, features)
            q = spectra[:1, 0, 5]
            labels = torch.arange(classes, device=device)[None]
            before, after = model(s, labels, q, c), model(s, labels, q, changed)
            delta = after - before
            centered = delta - delta.mean(dim=-1, keepdim=True)
            actuation = {"scope": "fit-participant M-only standardized auxiliary swap diagnostic",
                         "max_centered_logit_delta": float(centered.abs().max()),
                         "argmax_changes": int((before.argmax(-1) != after.argmax(-1)).sum()),
                         "not_an_efficacy_test": True}
    record = {"fit_id": fit_id, "kind": kind, "fold": fold, "arm": arm, "lr": lr,
              "seed": seed, "fit_subject_ids": list(role.ids),
              "normalizer_fit_subject_ids": list(scaler.fit_subject_ids),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "initial_state_sha256": initial_sha, "episode_schedule_sha256": schedule_hash.hexdigest(),
              "steps": config["training"]["steps"], "loss_samples": losses,
              "sham_donors": {str(k): v for k, v in donors.items()}, "actuation": actuation,
              "normalizer_mean": scaler.mean.tolist(), "normalizer_scale": scaler.scale.tolist(),
              "elapsed_seconds": time.monotonic() - fit_start}
    checkpoint = meter.output / f"{fit_id}.pt"
    with checkpoint.open("xb") as stream:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    record["checkpoint_sha256"] = file_sha(checkpoint)
    write_json(meter.output / f"{fit_id}.json", record)
    meter.event("fit_complete", fit_id=fit_id, checkpoint_sha256=record["checkpoint_sha256"])
    return Fitted(model, scaler, record)


def evaluate_model(fitted: Fitted, role: RoleData, config: dict, device: str) -> tuple[np.ndarray, dict]:
    if set(role.ids) & set(fitted.scaler.fit_subject_ids):
        raise ValueError("Evaluation participant overlaps normalizer/model fit role.")
    arm, fold = fitted.record["arm"], fitted.record["fold"]
    raw, donors = prefix_contexts(role, arm, sham_seed=config["splits"]["sham_seed"] + fold)
    normalized = fitted.scaler.transform(raw)
    contexts = torch.tensor(normalized, dtype=torch.float32, device=device)
    spectra = torch.tensor(role.spectra, dtype=torch.float32, device=device)
    _, _, _, classes, channels, features = spectra.shape
    output = np.empty((len(role.ids), 2, 3, 5 * classes, classes), np.float32)
    with torch.no_grad():
        for i in range(len(role.ids)):
            query = spectra[i, :, 5:10].reshape(2, 5 * classes, channels, features)
            for ki, k in enumerate(config["budgets"]):
                support = spectra[i, :, :k].reshape(2, k * classes, channels, features)
                labels = torch.arange(classes, device=device).repeat(2, k)
                output[i, :, ki] = fitted.model(support, labels, query, contexts[i, :, ki]).cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite evaluation logits.")
    return output, {"normalized_contexts": normalized.astype(np.float32),
                    "donors": {str(k): v for k, v in donors.items()}}


def cca_scores(role: RoleData, config: dict) -> np.ndarray:
    shape = role.query_crop.shape
    scores = regularized_cca_scores(role.query_crop.reshape(-1, *shape[-2:]),
                                    frequencies=config["frequencies"], sfreq=config["sfreq"],
                                    n_harmonics=config["n_harmonics"])
    return scores.reshape(shape[0], 2, -1, len(config["frequencies"]))


def accuracy(scores: np.ndarray) -> np.ndarray:
    classes = scores.shape[-1]
    if scores.shape[-2] != 5 * classes or not np.isfinite(scores).all():
        raise ValueError("All five balanced late-query blocks are required.")
    return (scores.argmax(-1) == np.tile(np.arange(classes), 5)).mean(axis=-1)


def execute(cache: dict, config: dict, meter: Meter, *, device: str) -> dict:
    """Shared generated/human path; all training finishes before outer scoring."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ids = cache["ids"].tolist()
    if ids != config["source_subject_ids"]:
        raise ValueError("Cache participant order differs from execution contract.")
    splits = participant_splits(ids)
    write_json(meter.output / "roles.json", {"splits": splits, "generated": meter.generated})
    final_records, choices, all_fit_records = [], {}, []
    for split in splits:
        fold = split["fold"]
        inner_fit = select_role(cache, split["inner_fit"])
        inner_validation = select_role(cache, split["inner_validation"])
        # Validates role/nuisance coverage before any fit for this split.
        for role in (inner_fit, inner_validation, select_role(cache, split["outer_fit"]),
                     select_role(cache, split["outer_query"])):
            prefix_contexts(role, "SHAM", sham_seed=config["splits"]["sham_seed"] + fold)
        common_inner = cca_scores(inner_validation, config)
        save_npz(meter.output / f"inner-cca-fold{fold}.npz", scores=common_inner,
                 subject_ids=np.asarray(inner_validation.ids))
        cca_accuracy = float(accuracy(common_inner).mean())
        choices[str(fold)] = {}
        for arm in ARMS:
            validation_means, validation_curves = {}, {}
            for lr in config["training"]["learning_rates"]:
                fitted = train_one(inner_fit, config, arm=arm, lr=lr,
                                   seed=config["training"]["seeds"][0], fold=fold,
                                   kind="inner", device=device, meter=meter)
                meter.consume("inner_evaluations")
                scores, context_receipt = evaluate_model(fitted, inner_validation, config, device)
                save_npz(meter.output / f"{fitted.record['fit_id']}-validation.npz", scores=scores,
                         contexts=context_receipt["normalized_contexts"],
                         subject_ids=np.asarray(inner_validation.ids))
                curve = accuracy(scores).mean(axis=(0, 1))
                validation_means[lr] = float(curve.mean())
                validation_curves[lr] = {0: cca_accuracy, **dict(zip(config["budgets"], curve.tolist()))}
                all_fit_records.append(fitted.record)
            chosen = select_learning_rate(validation_means)
            policy = select_policy(validation_curves[chosen], config["evaluation"]["target_ba"])
            choices[str(fold)][arm] = {"lr": chosen, "policy": policy,
                                       "validation_curve": validation_curves[chosen],
                                       "lr_validation_means": validation_means}
        outer_fit = select_role(cache, split["outer_fit"])
        for arm in ARMS:
            for seed in config["training"]["seeds"]:
                fitted = train_one(outer_fit, config, arm=arm, lr=choices[str(fold)][arm]["lr"],
                                   seed=seed, fold=fold, kind="outer", device=device, meter=meter)
                final_records.append(fitted.record)
                all_fit_records.append(fitted.record)
        meter.check()
    del fitted
    if len(final_records) != 24 or len(all_fit_records) != 48:
        raise RuntimeError("Unexpected number of frozen models.")
    if len({r["parameter_count"] for r in all_fit_records}) != 1:
        raise RuntimeError("Capacity mismatch across arms.")
    for kind in ("inner", "outer"):
        for fold in range(3):
            for seed in config["training"]["seeds"]:
                group = [r for r in all_fit_records if (r["kind"], r["fold"], r["seed"])
                         == (kind, fold, seed)]
                if group and (len({r["initial_state_sha256"] for r in group}) != 1
                              or len({r["episode_schedule_sha256"] for r in group}) != 1):
                    raise RuntimeError("Initial state or episode opportunity differs across arms/LRs.")
    freeze = {"at": utc_now(), "choices": choices, "final_models": final_records,
              "fit_records": [r["fit_id"] for r in all_fit_records], "generated": meter.generated,
              "fit_record_sha256": {r["fit_id"]: file_sha(meter.output / f"{r['fit_id']}.json")
                                     for r in all_fit_records}}
    write_json(meter.output / "all_models_frozen.json", freeze)
    freeze_sha = file_sha(meter.output / "all_models_frozen.json")
    meter.event("all_models_frozen", models=24, sha256=freeze_sha)
    meter.consume("outer_reveal_batches")
    meter.event("outer_reveal_start", freeze_sha256=freeze_sha)
    classes, n = len(config["frequencies"]), len(ids)
    learned = np.empty((4, 2, n, 2, 3, 5 * classes, classes), np.float32)
    common = np.empty((n, 2, 5 * classes, classes), np.float64)
    baselines = np.empty((2, n, 2, 3, 5 * classes, classes), np.float64)
    for split in splits:
        fold = split["fold"]
        role = select_role(cache, split["outer_query"])
        positions = [ids.index(s) for s in role.ids]
        common[positions] = cca_scores(role, config)
        for record in (r for r in final_records if r["fold"] == fold):
            path = meter.output / f"{record['fit_id']}.pt"
            if file_sha(path) != record["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed before query scoring.")
            model = model_from_config(config).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            scaler = ContextScaler(np.asarray(record["normalizer_mean"]),
                                   np.asarray(record["normalizer_scale"]),
                                   tuple(record["normalizer_fit_subject_ids"]))
            scores, context_receipt = evaluate_model(Fitted(model, scaler, record), role, config, device)
            ai, si = ARMS.index(record["arm"]), config["training"]["seeds"].index(record["seed"])
            learned[ai, si, positions] = scores
            save_npz(meter.output / f"{record['fit_id']}-query.npz", scores=scores,
                     contexts=context_receipt["normalized_contexts"], subject_ids=np.asarray(role.ids))
            write_json(meter.output / f"{record['fit_id']}-query-role.json",
                       {"subject_ids": list(role.ids), "donors": context_receipt["donors"],
                        "freeze_sha256": freeze_sha})
        for i, position in enumerate(positions):
            for e in range(2):
                for ki, k in enumerate(config["budgets"]):
                    s = role.spectra[i, e, :k].reshape(k * classes, *role.spectra.shape[-2:])
                    q = role.spectra[i, e, 5:].reshape(5 * classes, *role.spectra.shape[-2:])
                    labels = np.tile(np.arange(classes), k)
                    baselines[0, position, e, ki] = raw_prototype_scores(s, labels, q, n_classes=classes)
                    baselines[1, position, e, ki] = raw_prototype_scores(
                        s, labels, q, n_classes=classes,
                        channel_gain=direct_impedance_gain(role.impedance[i, e, :k]))
        meter.check()
    save_npz(meter.output / "all_query_scores.npz", learned=learned, common=common,
             baselines=baselines, subject_ids=np.asarray(ids))
    meter.event("outer_scores_saved", sha256=file_sha(meter.output / "all_query_scores.npz"))
    # No metric summary is exposed until every query score is durably saved.
    from cfeg.analysis.joint_harmonic_report import summarize

    report = summarize(learned, common, baselines, choices, config)
    report.update(generated=meter.generated, freeze_sha256=freeze_sha,
                  scores_sha256=file_sha(meter.output / "all_query_scores.npz"),
                  cache_sha256=file_sha(meter.output / "features.npz"),
                  counters=meter.snapshot(), audit_status="PENDING",
                  metadata_actuation_source_fits=[r["actuation"] for r in all_fit_records
                                                   if r["arm"] == "QM"])
    write_json(meter.output / "result.json", report)
    meter.event("execution_complete", result_sha256=file_sha(meter.output / "result.json"))
    return report


def human_entry(root: Path) -> Path:
    config_path = root / "configs/analysis/joint_harmonic_program_v1.json"
    plan_path = root / "docs/joint_harmonic_program_v1_plan.md"
    state_path = root / "docs/reports/joint_harmonic_program_v1_state.json"
    if file_sha(config_path) != CONFIG_SHA or file_sha(plan_path) != PLAN_SHA:
        raise ValueError("Frozen science specification changed.")
    config, state = json.loads(config_path.read_text()), json.loads(state_path.read_text())
    if state["status"] != "ENGINEERING_QUALIFIED_HUMAN_NOT_STARTED":
        raise RuntimeError("Human execution requires complete generated qualification.")
    if state["consumed"]["human_extraction_attempts"] or tuple(config["source_subject_ids"]) != SOURCE_IDS:
        raise RuntimeError("Human attempt already consumed or allowlist changed.")
    for name, expected in {**state["qualified_core_artifacts"], **state["qualified_runner_artifacts"]}.items():
        if file_sha(root / name) != expected:
            raise ValueError(f"Qualified implementation changed: {name}")
    proof = json.loads((root / state["full_flow_receipt"]).read_text())
    if proof["status"] != "PASS_GENERATED_FULL_FLOW" or proof["failed"] != 0:
        raise RuntimeError("No passing full-flow qualification receipt.")
    output = Path(tempfile.mkdtemp(prefix="joint-harmonic-human-v1-", dir="/home/whwovy/eeg-data"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, config["limits"]["gpu_allocated_bytes"] / total))
    start = {"at": utc_now(), "output": str(output), "pid": os.getpid(), "device": device,
             "plan_sha256": PLAN_SHA, "config_sha256": CONFIG_SHA,
             "state_sha256": file_sha(state_path), "qualified_artifacts": state["qualified_runner_artifacts"]}
    # Fixed canonical one-shot reservation: never unlink or overwrite after failure.
    write_json(root / "docs/reports/joint_harmonic_program_v1_start.json", start)
    write_json(output / "start.json", start)
    write_json(output / "config.json", config)
    meter = Meter(output, config["limits"], generated=False)
    try:
        cache, provenance = read_human_cache(config, meter, file_sha)
        save_npz(output / "features.npz", **cache)
        write_json(output / "provenance.json", provenance)
        write_json(output / "cache_receipt.json", {"sha256": file_sha(output / "features.npz"),
                                                   "bytes": (output / "features.npz").stat().st_size})
        execute(cache, config, meter, device=device)
        meter.check()
        write_json(output / "terminal.json", {"status": "COMPLETE_AUDIT_PENDING", "at": utc_now(),
                                               **meter.snapshot()})
    except BaseException as exc:
        meter.flush()
        write_json(output / "failure.json", {"status": "VALIDITY_OR_RUNTIME_FAILURE", "at": utc_now(),
                                              "error": str(exc), "traceback": traceback.format_exc(),
                                              **meter.snapshot()})
        raise
    return output


def main() -> None:
    def terminated(signum, frame):
        raise TimeoutError(f"Supervisor termination signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-human", action="store_true")
    args = parser.parse_args()
    if not args.execute_human:
        parser.error("No implicit data access; explicit --execute-human is required.")
    print(human_entry(Path(__file__).resolve().parents[3]), flush=True)


if __name__ == "__main__":
    main()

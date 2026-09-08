"""One pinned task-shape source39 development attempt, failure preserving."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_archive as archive
from cfeg.analysis import task_trca_shape_artifact_audit as audit
from cfeg.analysis import task_trca_shape_evaluation as evaluation
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_shape_learning as learning
from cfeg.analysis.task_trca_shape_inputs import RolePartition

DESIGN_SHA = "5e17b2368a84174f008ce31e4c7a3de1836e69b4e5c1570e8722aa6b2c56d696"
NATIVE_MANIFEST_SHA = "15c088048f3435a500d3758138c3cd6a1616e869d568bdf68d10bcc5e978d83a"
OLD_PLAN_SHA = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, expected=None):
    encoded = Path(path).read_bytes()
    require(
        expected is None or hashlib.sha256(encoded).hexdigest() == expected,
        "JSON pin differs: " + str(path),
    )
    return json.loads(encoded)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": audit.sha(path), "bytes": Path(path).stat().st_size}


def write_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": audit.sha(path), "bytes": Path(path).stat().st_size}


def read_saved_arrays(artifact):
    path = Path(artifact["path"])
    require(
        path.stat().st_size == artifact["bytes"] and audit.sha(path) == artifact["sha256"],
        "Saved array pin differs",
    )
    with np.load(path, allow_pickle=False) as archive_file:
        result = {name: archive_file[name] for name in archive_file.files}
    require(audit.sha(path) == artifact["sha256"], "Saved array changed during audit read")
    return result


def install_guard(readable, output):
    readable = {Path(v).absolute() for v in readable}
    data_suffixes = {
        ".json",
        ".jsonl",
        ".mat",
        ".npz",
        ".npy",
        ".h5",
        ".hdf5",
        ".parquet",
        ".pkl",
        ".db",
        ".sqlite",
        ".sqlite3",
    }
    environments = (ROOT / ".venv", Path("/usr/local/lib"), Path("/usr/lib"))

    def guard(event, args):
        if event in ("subprocess.Popen", "socket.connect", "socket.getaddrinfo"):
            raise PermissionError("No external processes/network during study")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        own_output = path.parent == output and path.resolve() == path
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC) and not own_output:
            raise PermissionError("Write outside new study artifacts denied: " + str(path))
        if (
            path.suffix.lower() in data_suffixes
            and path not in readable
            and not own_output
            and not any(path.is_relative_to(p) for p in environments)
        ):
            raise PermissionError("Unapproved dataset/old outcome/held input denied: " + str(path))

    sys.addaudithook(guard)


def run(manifest_path, manifest_sha):
    manifest_path = manifest_path.absolute()
    manifest = read_json(manifest_path, manifest_sha)
    require(
        ROOT == Path("/home/whwovy/califreeEEG"), "Only integrated canonical repository may execute"
    )
    require(
        manifest["schema"] == "cfeg.task_trca_shape.source39_execution.v1"
        and manifest["status"] == "EXECUTION_FROZEN",
        "Executable frozen manifest required",
    )
    require(
        manifest["source_ids"] == list(archive.SOURCE_IDS)
        and manifest["held60_authorized"] is False,
        "Source-only authority required",
    )
    require(manifest["design"]["sha256"] == DESIGN_SHA, "Historical design pin differs")
    read_json(manifest["design"]["path"], DESIGN_SHA)
    old = read_json(manifest["native_plan"]["path"], OLD_PLAN_SHA)
    require(manifest["native_plan"]["sha256"] == OLD_PLAN_SHA, "Native compatibility plan pin")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "Require committed clean integration",
    )
    require(
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        == "main",
        "Main only",
    )
    for path, digest in manifest["code_pins"].items():
        require(audit.sha(ROOT / path) == digest, "Implementation pin differs: " + path)
    require(
        sys.version_info[:2] == (3, 10)
        and np.__version__ == "1.26.4"
        and scipy.__version__ == "1.15.3"
        and torch.__version__ == "2.2.2+cu121",
        "Pinned existing environment required",
    )
    require(Path(sys.executable) == ROOT / ".venv/bin/python", "Pinned executable required")
    require(
        all(
            os.environ.get(v) == "1"
            for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "BLAS1 required",
    )
    require(
        manifest["backend"] == "batch" and manifest["device"] == "cuda",
        "Artificially validated fixed runtime required",
    )
    require(torch.cuda.is_available(), "Frozen CUDA device unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.16)
    torch.cuda.reset_peak_memory_stats()
    # Warm optimizer imports/CUDA before the data boundary; no task data.
    warm = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device="cuda"))
    optimizer = torch.optim.Adam([warm], lr=0.01)
    warm.square().sum().backward()
    optimizer.step()
    del warm, optimizer
    output = Path(manifest["output_root"])
    require(
        output.is_absolute() and output.resolve() == output and not output.exists(),
        "New exact output directory required",
    )
    require(
        output.parent == Path("/home/whwovy")
        and output.name.startswith("task-trca-shape-source39-v1-"),
        "Bounded new output path",
    )
    require(
        os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize >= 20 * 1024**3,
        "Free disk reserve20GiB required",
    )
    output.mkdir()
    started = time.perf_counter()
    state = {"stage": "START", "outer_fold": None, "query_access_count": 0, "models_frozen": False}
    artifacts = {}
    access_records = []

    def event(value):
        record = {"time": now(), "elapsed_seconds": time.perf_counter() - started, **value}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        require(size <= manifest["output_budget_bytes"], "Frozen output budget exceeded")

    def record_access(records):
        for item in records:
            access_records.append(item)
            if item["kind"] in ("query", "full_k3", "full_k5", "a0"):
                state["query_access_count"] += 1
            with (output / "access.jsonl").open("a") as stream:
                stream.write(json.dumps({"outer_fold": state["outer_fold"], **item}) + "\n")

    @contextmanager
    def tracked(reader):
        try:
            with reader:
                yield reader
        finally:
            record_access(reader.access_log)

    def alarm(*unused):
        raise TimeoutError("Frozen whole-attempt time limit exceeded")

    write_json(
        output / "start.json",
        {
            "status": "STARTED",
            "time": now(),
            "manifest_sha256": manifest_sha,
            "manifest": manifest,
            "source_revision": revision,
            "held60_access": False,
            "old_attempts_reopened": False,
            "device_name": torch.cuda.get_device_name(),
        },
    )
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(manifest["max_seconds"])
    try:
        native_root = Path(manifest["native_root"])
        readable = [
            manifest_path,
            manifest["design"]["path"],
            manifest["native_plan"]["path"],
            native_root / "result.json",
            native_root / "start.json",
            old["source_projection"]["path"],
        ]
        readable += [ROOT / path for path in manifest["code_pins"]]
        readable += [native_root / f"S{pid:03d}.npz" for pid in archive.SOURCE_IDS]
        install_guard(readable, output)
        native = read_json(native_root / "result.json", NATIVE_MANIFEST_SHA)
        require(
            native["status"] == "COMPLETE"
            and native["plan_sha256"] == OLD_PLAN_SHA
            and native["source_subject_ids"] == list(archive.SOURCE_IDS),
            "Native manifest identity",
        )
        require(audit.sha(native_root / "start.json") == native["start_sha256"], "Native start pin")
        require(
            [r["subject"] for r in native["files"]] == list(archive.SOURCE_IDS),
            "Exact native source39 files",
        )
        specs = {}
        for row in native["files"]:
            require(row["filename"] == f"S{row['subject']:03d}.npz", "Native filename identity")
            specs[row["subject"]] = archive.ArchiveSpec(
                native_root / row["filename"], row["subject"], row["sha256"], row["bytes"]
            )
        spec = old["source_projection"]
        envelope = {
            **spec["envelope_provenance"],
            "manifest_sha256": spec["manifest_sha256"],
            "returned_rows": spec["returned_rows"],
            "returned_packets": spec["packets"],
            "returned_subject_ids": list(archive.SOURCE_IDS),
            "columns": spec["columns"],
        }
        weights = [old["native_weights"]["ETRCA"][name] for name in old["interfaces"]]
        a0_weights = np.array(
            [old["native_weights"]["A0_author"][name] for name in old["interfaces"]]
        )
        frozen = []
        for fold in range(3):
            state.update(stage="SOURCE_PREPARATION", outer_fold=fold)
            evaluation_ids = archive.SOURCE_IDS[fold::3]
            fit_ids = tuple(pid for pid in archive.SOURCE_IDS if pid not in evaluation_ids)
            partition = RolePartition(fit_ids, (), evaluation_ids)
            cases = []
            with tracked(
                archive.SupportMetadata(spec["path"], spec["sha256"], envelope, partition)
            ) as metadata:
                for pid in fit_ids:
                    with tracked(archive.NativeArchive(specs[pid], partition)) as source:
                        for interface in range(2):
                            packets = {k: metadata.support(pid, interface, k) for k in (3, 5)}
                            for n in archive.SAMPLE_COUNTS:
                                supervision = source.supervision(interface, n)
                                for k in (3, 5):
                                    packet, order = packets[k]
                                    support = source.support(interface, n, k)
                                    cases.append(
                                        learning.make_task_case(
                                            pid,
                                            interface,
                                            order,
                                            support,
                                            packet,
                                            old["frequencies"],
                                            supervision,
                                            weights=weights[interface],
                                            device="cuda",
                                        )
                                    )
                    event(
                        {"event": "source_participant_ready", "fold": fold, "participant_id": pid}
                    )
            cases = tuple(sorted(cases, key=lambda c: c.key))
            packed = evaluation.pack_cases(cases)
            artifacts[f"source{fold}"] = write_npz(output / f"source{fold}.npz", packed)
            packed = read_saved_arrays(artifacts[f"source{fold}"])
            state["stage"] = "NESTED_TRAINING"
            pipeline, selection = learning.nested_fit(
                cases,
                outer_evaluation_ids=evaluation_ids,
                backend="batch",
                progress=lambda row, fold=fold: event({"outer_fold": fold, **row}),
            )
            model = {
                "pipeline": pipeline.record(),
                "selection": selection,
                "source_artifact": artifacts[f"source{fold}"],
                "manifest_sha256": manifest_sha,
            }
            state["stage"] = "SOURCE_AUDIT"
            receipt = audit.audit_training(packed, model, fit_ids, evaluation_ids)
            model["independent_source_audit"] = receipt
            artifact = write_json(output / f"model{fold}.json", model)
            artifacts[f"model{fold}"] = artifact
            frozen.append(
                {
                    "fold_id": fold,
                    "fit_ids": list(fit_ids),
                    "evaluation_ids": list(evaluation_ids),
                    **artifact,
                }
            )
            event({"event": "outer_model_frozen", "fold": fold, "source_audit": receipt})
            del cases, packed, pipeline, model
            gc.collect()
            torch.cuda.empty_cache()
        require(state["query_access_count"] == 0, "Query already accessed before global freeze")
        freeze_doc = {
            "schema": "cfeg.task_trca_shape.all_models_frozen.v1",
            "status": "ALL_MODELS_FROZEN",
            "source_ids": list(archive.SOURCE_IDS),
            "manifest_sha256": manifest_sha,
            "query_access_count": 0,
            "models": frozen,
        }
        artifacts["freeze"] = write_json(output / "globalfreeze.json", freeze_doc)
        token = archive.verify_freeze(
            output / "globalfreeze.json", artifacts["freeze"]["sha256"], manifest_sha
        )
        state.update(stage="FINAL_QUERY_EVALUATION", models_frozen=True)
        event({"event": "all_models_frozen", "freeze_sha256": artifacts["freeze"]["sha256"]})
        all_scores = np.full((39, 2, 4, 2, 9, 48, 12), np.nan)
        all_a0 = np.full((39, 2, 4, 48, 12), np.nan)
        coverage_m = np.zeros((39, 2, 8, 2))
        coverage_donor = np.zeros_like(coverage_m)
        coverage_available = np.zeros((39, 2, 8), dtype=bool)
        r_changes = np.zeros((39, 2, 4, 2))
        w_changes = np.zeros_like(r_changes)
        qm_coefficients = []
        for fold, frozen_model in enumerate(frozen):
            state["outer_fold"] = fold
            state["stage"] = "FINAL_QUERY_EVALUATION"
            model = read_json(frozen_model["path"], frozen_model["sha256"])
            pipeline = evaluation.pipeline_from_record(model["pipeline"])
            qm_coefficients.append(pipeline.residuals["QM"].coefficients.tolist())
            fit_ids, evaluation_ids = (
                tuple(frozen_model["fit_ids"]),
                tuple(frozen_model["evaluation_ids"]),
            )
            partition = RolePartition(fit_ids, (), evaluation_ids)
            states = {}
            with tracked(
                archive.SupportMetadata(spec["path"], spec["sha256"], envelope, partition)
            ) as metadata:
                for pid in evaluation_ids:
                    with tracked(archive.NativeArchive(specs[pid], partition)) as source:
                        for interface in range(2):
                            packets = {k: metadata.support(pid, interface, k) for k in (3, 5)}
                            for n in archive.SAMPLE_COUNTS:
                                for k in (3, 5):
                                    packet, order = packets[k]
                                    states[(pid, interface, n, k)] = evaluation.support_state(
                                        pid,
                                        interface,
                                        order,
                                        source.support(interface, n, k),
                                        packet,
                                        old["frequencies"],
                                        weights[interface],
                                    )
            donors = {}
            for interface in range(2):
                for n in archive.SAMPLE_COUNTS:
                    for k in (3, 5):
                        rows = [states[(pid, interface, n, k)] for pid in evaluation_ids]
                        mapping = features.donor_map(
                            list(evaluation_ids),
                            np.stack([c.mask for c in rows]),
                            [c.order for c in rows],
                            interface,
                        )
                        donors.update(
                            {c.key: (mapping[c.participant_id], *c.condition) for c in rows}
                        )
            fold_records = []
            for pid in evaluation_ids:
                p = archive.SOURCE_IDS.index(pid)
                records = []
                with tracked(archive.NativeArchive(specs[pid], partition)) as source:
                    for interface in range(2):
                        for ni, n in enumerate(archive.SAMPLE_COUNTS):
                            query = source.query(interface, n, token).reshape(48, 5, 8, n)
                            a0_corr = source.a0_correlations(interface, n, token).reshape(48, 5, 12)
                            all_a0[p, interface, ni] = np.einsum(
                                "nbc,b->nc", a0_corr, a0_weights[interface]
                            )
                            for ki, k in enumerate((3, 5)):
                                current = states[(pid, interface, n, k)]
                                donor = states[donors[current.key]]
                                result = evaluation.evaluate(pipeline, current, query, donor=donor)
                                result.update(
                                    {
                                        "keys": np.array(current.key),
                                        "orders": current.order,
                                        "q": current.q,
                                        "m": current.m,
                                        "available": current.available,
                                        "packet5": np.pad(
                                            current.packet,
                                            ((0, 5 - k), (0, 0)),
                                            constant_values=np.nan,
                                        ),
                                        "s": current.s.numpy(),
                                        "c": current.c.numpy(),
                                        "anchors": current.anchors.numpy(),
                                        "weights": current.weights.numpy(),
                                        "donor_id": donor.participant_id,
                                        "a0_correlations": a0_corr,
                                        "cached_full_correlations": source.full_correlations(
                                            interface, n, k, token
                                        ).reshape(48, 5, 12),
                                    }
                                )
                                all_scores[p, interface, ni, ki] = result["scores"]
                                r_changes[p, interface, ni, ki] = np.max(
                                    np.abs(result["r"][3] - result["r"][1])
                                )
                                w_changes[p, interface, ni, ki] = np.max(
                                    np.abs(result["filters"][4] - result["filters"][2])
                                )
                                if k == 3:
                                    coverage_m[p, interface] = current.m
                                    coverage_donor[p, interface] = donor.m
                                    coverage_available[p, interface] = current.available
                                records.append(result)
                packed_eval = {name: np.stack([r[name] for r in records]) for name in records[0]}
                artifact = write_npz(output / f"evaluation_S{pid:03d}.npz", packed_eval)
                artifacts[f"evaluation{pid}"] = artifact
                # Independent donor scoring needs the full outer13 partition;
                # raw per-participant files are audited together below.
                fold_records.append(read_saved_arrays(artifact))
                event(
                    {"event": "evaluation_participant_saved", "fold": fold, "participant_id": pid}
                )
            packed_fold = {
                name: np.concatenate([r[name] for r in fold_records]) for name in fold_records[0]
            }
            state["stage"] = "EVALUATION_AUDIT"
            receipt = audit.audit_evaluation(packed_fold, model["pipeline"])
            artifacts[f"evaluation_audit{fold}"] = write_json(
                output / f"evaluation_audit{fold}.json", receipt
            )
            event({"event": "evaluation_fold_audit_pass", "fold": fold, "receipt": receipt})
            del states, packed_fold, fold_records, packed_eval
            gc.collect()
        state["stage"] = "TERMINAL_ASSESSMENT"
        require(
            np.isfinite(all_scores).all() and np.isfinite(all_a0).all(), "Incomplete score grid"
        )
        aggregates = {
            "scores": all_scores,
            "a0": all_a0,
            "coverage_m": coverage_m,
            "coverage_donor_m": coverage_donor,
            "coverage_available": coverage_available,
            "r_changes": r_changes,
            "filter_changes": w_changes,
            "qm_coefficients": np.array(qm_coefficients),
            "weights": np.array(weights),
            "a0_weights": a0_weights,
        }
        artifacts["scores"] = write_npz(output / "scores.npz", aggregates)
        from cfeg.analysis.task_trca_shape_audit import summarize

        summary = summarize(
            all_scores,
            all_a0,
            {"m": coverage_m, "donor_m": coverage_donor, "available": coverage_available},
            {
                "band_weights": weights,
                "a0_band_weights": a0_weights,
                "r_max_abs_qm_minus_q": r_changes,
                "filter_max_abs_qm_minus_q": w_changes,
                "qm_coefficients": qm_coefficients,
            },
        )
        result = {
            "status": "COMPLETE",
            "summary": summary,
            "artifacts": artifacts,
            "manifest_sha256": manifest_sha,
            "source_revision": revision,
            "time": now(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "state": state,
            "held60_access": False,
            "query_metadata_decoded": False,
            "evidence_scope": "repeatedly exposed source39 development; no independent confirmation",
        }
        require(
            state["query_access_count"] == 1248, "Exact query-bearing decode call total required"
        )
        event(
            {
                "event": "attempt_verified_before_publication",
                "result_path": str(output / "result.json"),
                "terminal": summary.get("terminal"),
            }
        )
        require(
            sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            + len(json.dumps(result)) * 2
            + 1024**2
            < manifest["output_budget_bytes"],
            "Final receipt budget reserve",
        )
        signal.alarm(0)
        write_json(output / "result.json", result)
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "status": "VALIDITY_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "state": state,
                "artifacts": artifacts,
                "time": now(),
                "elapsed_seconds": time.perf_counter() - started,
                "held60_access": False,
                "manifest_sha256": manifest_sha,
                "interpretation": "Candidate attempt stopped; no scientific tuning, sample exclusion, or efficacy interpretation of partial results",
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    run(args.manifest, args.manifest_sha256)

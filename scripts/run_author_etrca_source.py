"""Execute the frozen source39 author-implementation compatibility assessment.

The only classifiers are unchanged public toolbox APIs. The CLI has no subject,
raw-file, tuning, retry, or development override. Pure helpers support fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import stat
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import io, stats

PLAN_SHA256 = "d36886d4dfd80961d1d8b7ceac685d468ca12211ff6149ecbf993cc0b4502462"
SOURCE_IDS = (
    4,
    6,
    8,
    11,
    14,
    21,
    22,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    37,
    41,
    42,
    43,
    44,
    46,
    54,
    55,
    56,
    61,
    63,
    65,
    67,
    73,
    74,
    77,
    80,
    82,
    83,
    84,
    89,
    92,
    97,
    100,
    102,
)
SOURCE_ROOT = Path("/home/whwovy/califreeEEG")
PLAN_PATH = SOURCE_ROOT / "configs/analysis/author_etrca_source39_v1.json"
RAW_ROOT = Path("/home/whwovy/eeg-data/raw/wearable")
OUTPUT_ROOT = Path("/home/whwovy/author-etrca-artifacts/source39-v1")
UPSTREAM_ROOT = Path("/home/whwovy/ssvep-author-compatibility-20260907/upstream")
PYTHON_PATH = Path("/home/whwovy/ssvep-author-compatibility-20260907/.venv/bin/python")
REVISION = "3344bd199daf78888e364d9db00ae7d8128d2b5f"
VERSIONS = {
    "numpy": "1.23.4",
    "scipy": "1.13.0",
    "joblib": "1.4.2",
    "scikit-learn": "1.3.0",
    "mat73": "0.63",
    "h5py": "3.11.0",
    "threadpoolctl": "3.5.0",
}
ROW_KEY = ("stage", "view", "interface", "n_samples", "method", "k")
CACHE_KEYS = ("a0_r", "etrca_lobo_r", "etrca_chrono_r")
_NATIVE = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return stream_sha256(stream)


def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def read_plan(path):
    encoded = Path(path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != PLAN_SHA256:
        raise ValueError("Frozen plan hash mismatch")
    plan = json.loads(encoded)
    if (
        tuple(plan["source_subject_ids"]) != SOURCE_IDS
        or plan["raw_root"] != str(RAW_ROOT)
        or plan["raw_filename"] != "S{subject:03d}.mat"
        or plan["execution"]["output_root"] != str(OUTPUT_ROOT)
        or plan["upstream"]["root"] != str(UPSTREAM_ROOT)
        or plan["upstream"]["revision"] != REVISION
        or any(
            plan["authority"][key]
            for key in (
                "metadata_access",
                "held_access",
                "retired_access",
                "old_runner_or_seal_access",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("Source-only authority does not match the immutable allowlist")
    return plan


def preflight(plan_path, output, upstream):
    # Do not resolve a supplied symlink into an apparently valid authorized path.
    for supplied, expected in (
        (plan_path, PLAN_PATH),
        (output, OUTPUT_ROOT),
        (upstream, UPSTREAM_ROOT),
    ):
        if Path(supplied) != expected or Path(supplied).resolve() != expected:
            raise ValueError("Only the exact non-symlink frozen paths are allowed")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Existing output consumes this attempt; no retry or overwrite")
    if Path(__file__).resolve().parents[1] != SOURCE_ROOT:
        raise RuntimeError("Real execution is restricted to the integrated main checkout")
    plan = read_plan(plan_path)
    if git(SOURCE_ROOT, "status", "--porcelain"):
        raise RuntimeError("Integrated source must be clean and committed")
    if git(SOURCE_ROOT, "rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise RuntimeError("Execute only the integrated main branch")
    if git(upstream, "rev-parse", "HEAD") != REVISION or git(upstream, "status", "--porcelain"):
        raise RuntimeError("Upstream must be clean at the exact revision")
    for relative, expected in plan["upstream"]["pins"].items():
        if file_sha256(upstream / relative) != expected:
            raise RuntimeError("Upstream numerical source hash mismatch: " + relative)
    if Path(sys.executable) != PYTHON_PATH or sys.version.split()[0] != "3.9.21":
        raise RuntimeError("Use the frozen external Python 3.9.21 interpreter")
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Dependency mismatch: " + name)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError("Require " + name + "=1 before interpreter startup")
    return plan, {
        "plan_sha256": PLAN_SHA256,
        "source_commit": git(SOURCE_ROOT, "rev-parse", "HEAD"),
        "source_tree": git(SOURCE_ROOT, "rev-parse", "HEAD^{tree}"),
        "upstream_revision": REVISION,
        "python": sys.version.split()[0],
        "dependencies": dict(VERSIONS),
    }


def import_native(upstream):
    global _NATIVE
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from SSVEPAnalysisToolbox.algorithms.cca import SCCA_canoncorr
    from SSVEPAnalysisToolbox.algorithms.trca import ETRCA
    from SSVEPAnalysisToolbox.utils import wearablepreprocess
    from SSVEPAnalysisToolbox.utils.algsupport import gen_ref_sin

    hashes = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(
            ("SSVEPAnalysisToolbox.datasets", "SSVEPAnalysisToolbox.evaluator", "cfeg")
        ):
            raise RuntimeError("Forbidden runtime import: " + name)
        if name.startswith("SSVEPAnalysisToolbox"):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(upstream)
            hashes[str(relative)] = file_sha256(path)
    _NATIVE = SimpleNamespace(
        SCCA=SCCA_canoncorr,
        ETRCA=ETRCA,
        preprocess=wearablepreprocess.preprocess,
        filterbank=wearablepreprocess.filterbank,
        gen_ref_sin=gen_ref_sin,
        suggested_weights_filterbank=wearablepreprocess.suggested_weights_filterbank,
    )
    return hashes


def runtime_guard(event, args):
    """Reviewed Python defense in depth; not an OS isolation/security claim."""
    if event == "import" and str(args[0]).startswith(
        ("SSVEPAnalysisToolbox.datasets", "SSVEPAnalysisToolbox.evaluator", "cfeg")
    ):
        raise RuntimeError("Forbidden data/evaluator/project import")
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
        raise RuntimeError("Network/subprocess access is forbidden after preflight")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0]))
    suffix = path.suffix.lower()
    if suffix == ".mat" and path not in {RAW_ROOT / f"S{s:03d}.mat" for s in SOURCE_IDS}:
        raise RuntimeError("MAT path is not source39 data")
    if suffix in (".npy", ".h5", ".hdf5", ".parquet", ".pkl"):
        raise RuntimeError("Processed or metadata data access is forbidden")
    if suffix == ".npz" and path != OUTPUT_ROOT / "correlations.npz":
        raise RuntimeError("Only the frozen output cache is permitted")
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC) and path not in {
        OUTPUT_ROOT / name for name in ("start.json", "correlations.npz", "result.json")
    }:
        raise RuntimeError("Only the three frozen artifacts may be written")


def finite_real(values, shape):
    if (
        not isinstance(values, np.ndarray)
        or values.shape != tuple(shape)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected a finite real numeric array with the exact native shape")


def preprocess_trial(raw, n_samples, native):
    finite_real(raw, (8, 710))
    if n_samples not in (125, 188, 250, 500):
        raise ValueError("Undeclared window")
    descriptor = SimpleNamespace(srate=250)
    segment = np.asarray(raw[:, 125 : 160 + n_samples], dtype=np.float64).copy()
    filtered = native.filterbank(descriptor, native.preprocess(descriptor, segment))
    result = np.asarray(filtered[:, :, 35 : 35 + n_samples], dtype=np.float64).copy()
    finite_real(result, (5, 8, n_samples))
    return result


def references_for_window(n_samples, plan, gen_ref_sin):
    return [
        gen_ref_sin(freq, 250, n_samples, 5, phase * np.pi)
        for freq, phase in zip(plan["frequencies"], plan["phases_pi"])
    ]


def scores_from_correlations(correlations, weights):
    # Match the author's per-trial row-matrix product, including tie rounding.
    leading = correlations.shape[:-2]
    w = np.asarray(weights, dtype=np.float64)[None, :]
    return np.asarray([w @ r for r in correlations.reshape(-1, 5, 12)]).reshape(*leading, 12)


def predict_correlations(model, query, weights):
    predicted, correlations = model.predict(query)
    correlations = np.asarray(correlations, dtype=np.float64)
    finite_real(correlations, (len(query), 5, 12))
    if np.max(np.abs(correlations)) > 1 + 1e-10:
        raise ValueError("Native correlations exceed their numerical domain")
    scores = scores_from_correlations(correlations, weights)
    if not np.array_equal(np.asarray(predicted), scores.argmax(axis=-1)):
        raise ValueError("Native labels disagree with native score arithmetic")
    return correlations


def compute_participant(raw, plan, native):
    """Pure numerical adapter, callable with artificial arrays and native/stub APIs."""
    finite_real(raw, (8, 710, 2, 10, 12))
    a0 = np.empty((2, 4, 10, 12, 5, 12), dtype=np.float64)
    lobo = np.empty((2, 10, 12, 5, 12), dtype=np.float64)
    chrono = np.empty((2, 4, 2, 5, 12, 5, 12), dtype=np.float64)
    for w, n_samples in enumerate(plan["sample_counts"]):
        references = references_for_window(n_samples, plan, native.gen_ref_sin)
        for interface_index, interface in enumerate(plan["interfaces"]):
            trials = [
                preprocess_trial(raw[:, :, interface_index, block, label], n_samples, native)
                for block in range(10)
                for label in range(12)
            ]
            wa = plan["methods"]["weights"]["A0_author"][interface]
            we = plan["methods"]["weights"]["ETRCA"][interface]
            for weights, method in ((wa, "cca"), (we, "trca")):
                if not np.array_equal(
                    weights, native.suggested_weights_filterbank(5, interface, method)
                ):
                    raise ValueError("Native band weights differ from the contract")
            anchor = native.SCCA(
                n_component=1,
                n_jobs=None,
                weights_filterbank=wa,
                force_output_UV=False,
                update_UV=True,
            )
            anchor.fit(ref_sig=references)
            a0[interface_index, w] = predict_correlations(anchor, trials, wa).reshape(10, 12, 5, 12)
            if anchor.model["U"] is not None or anchor.model["V"] is not None:
                raise ValueError("A0 unexpectedly cached query-specific filters")
            for ki, k in enumerate(plan["budgets"]):
                model = native.ETRCA(n_jobs=None, weights_filterbank=we)
                model.fit(X=trials[: 12 * k], Y=list(range(12)) * k)
                chrono[interface_index, w, ki] = predict_correlations(
                    model, trials[60:], we
                ).reshape(5, 12, 5, 12)
            if n_samples == 500:
                for block in range(10):
                    train = trials[: 12 * block] + trials[12 * (block + 1) :]
                    model = native.ETRCA(n_jobs=None, weights_filterbank=we)
                    model.fit(X=train, Y=list(range(12)) * 9)
                    lobo[interface_index, block] = predict_correlations(
                        model,
                        trials[12 * block : 12 * (block + 1)],
                        we,
                    )
    return dict(zip(CACHE_KEYS, (a0, lobo, chrono)))


def load_participant(subject, plan):
    if subject not in SOURCE_IDS or tuple(plan["source_subject_ids"]) != SOURCE_IDS:
        raise ValueError("Participant is outside source39")
    if plan["raw_root"] != str(RAW_ROOT) or RAW_ROOT.resolve() != RAW_ROOT:
        raise ValueError("Invalid or symlinked raw root")
    path = RAW_ROOT / f"S{subject:03d}.mat"
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Symlinked raw files are forbidden")
    # O_NOFOLLOW and a single descriptor bind hashing and parsing to this file.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Raw input must be a regular file")
        digest = stream_sha256(stream)
        stream.seek(0)
        loaded = io.loadmat(
            stream,
            variable_names=["data"],
            squeeze_me=False,
            mat_dtype=False,
            verify_compressed_data_integrity=True,
        )
        raw = loaded["data"]
        finite_real(raw, plan["raw_shape"])
        stored_dtype = str(raw.dtype)
        stream.seek(0)
        if stream_sha256(stream) != digest:
            raise ValueError("Raw file changed during parsing")
    record = {
        "subject": subject,
        "path": str(path),
        "sha256": digest,
        "stored_dtype": stored_dtype,
        "shape": list(raw.shape),
    }
    arrays = compute_participant(np.asarray(raw, dtype=np.float64), plan, _NATIVE)
    return subject, arrays, record


def balanced_accuracy(correlations, weights, rotated=False):
    scores = scores_from_correlations(correlations, weights)
    labels = np.broadcast_to(np.arange(12), scores.shape[:-1])
    shifts = range(1, 12) if rotated else (0,)
    return float(
        np.mean(
            [
                np.mean(np.argmax(np.roll(scores, shift, axis=-1), axis=-1) == labels)
                for shift in shifts
            ]
        )
    )


def rows_from_cache(cache, plan):
    if set(cache) != set(CACHE_KEYS):
        raise ValueError("Cache must contain exactly the three native correlation arrays")
    for key in CACHE_KEYS:
        shape = (len(plan["source_subject_ids"]), *plan["cache"][key]["shape"][1:])
        finite_real(cache[key], shape)
        if cache[key].dtype != np.float64 or np.max(np.abs(cache[key])) > 1 + 1e-10:
            raise ValueError("Cache dtype/domain mismatch")
    rows = []
    for p, participant in enumerate(plan["source_subject_ids"]):
        for i, interface in enumerate(plan["interfaces"]):

            def append(
                stage, view, n_samples, method, k, r, participant=participant, interface=interface
            ):
                base_method = "ETRCA" if method == "ETRCA_rotated" else method
                rows.append(
                    {
                        "participant": participant,
                        "stage": stage,
                        "view": view,
                        "interface": interface,
                        "n_samples": n_samples,
                        "method": method,
                        "k": k,
                        "ba": balanced_accuracy(
                            r,
                            plan["methods"]["weights"][base_method][interface],
                            method == "ETRCA_rotated",
                        ),
                        "query_count": int(np.prod(r.shape[:-2])),
                        "label_count": 12 * k,
                    }
                )

            for view, selected in (("all10", slice(None)), ("last5", slice(5, 10))):
                append(
                    "native_lobo_2s", view, 500, "A0_author", 0, cache["a0_r"][p, i, 3, selected]
                )
                for method in ("ETRCA", "ETRCA_rotated"):
                    append(
                        "native_lobo_2s",
                        view,
                        500,
                        method,
                        9,
                        cache["etrca_lobo_r"][p, i, selected],
                    )
            for w, n_samples in enumerate(plan["sample_counts"]):
                stage = "bridge_chronological_2s" if n_samples == 500 else "bridge_short_windows"
                append(stage, "last5", n_samples, "A0_author", 0, cache["a0_r"][p, i, w, 5:])
                for ki, k in enumerate(plan["budgets"]):
                    for method in ("ETRCA", "ETRCA_rotated"):
                        append(
                            stage,
                            "last5",
                            n_samples,
                            method,
                            k,
                            cache["etrca_chrono_r"][p, i, w, ki],
                        )
    return sorted(rows, key=lambda row: tuple(row[key] for key in ("participant", *ROW_KEY)))


def summarize_rows(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ROW_KEY)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        if len({row["participant"] for row in group}) != len(group):
            raise ValueError("Repeated participant in one summary cell")
        values = np.asarray([row["ba"] for row in group], dtype=np.float64)
        mean = float(values.mean())
        half = (
            float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 and not np.all(values == values[0])
            else 0.0
        )
        summaries.append(
            {
                **dict(zip(ROW_KEY, key)),
                "mean_ba": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "n_participants": len(values),
            }
        )
    return summaries


def publish(path, payload, *, compressed=False):
    """Create once, fsync, retain read-only partial artifact on any exception."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o400), "wb") as stream:
        if compressed:
            np.savez_compressed(stream, **payload)
        else:
            stream.write(
                (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            )
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_sha256(path)


def gather_participants(plan):
    result = {}
    pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork"))
    try:
        futures = [pool.submit(load_participant, subject, plan) for subject in SOURCE_IDS]
        for future in as_completed(
            futures, timeout=plan["execution"]["expected_runtime_ceiling_seconds"]
        ):
            subject, arrays, record = future.result()
            if subject in result:
                raise ValueError("Duplicate participant result")
            result[subject] = (arrays, record)
            print(
                f"completed source participant {subject}: {len(result)}/{len(SOURCE_IDS)}",
                flush=True,
            )
    except BaseException:
        # Python 3.9 has no public terminate_workers; terminate only this pool's
        # own processes so shutdown cannot wait past the frozen timeout.
        for process in list(pool._processes.values()):
            process.terminate()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    cache = {key: np.stack([result[s][0][key] for s in SOURCE_IDS]) for key in CACHE_KEYS}
    return cache, [result[s][1] for s in SOURCE_IDS]


def run(plan_path, output, upstream):
    start_clock = time.monotonic()
    plan, provenance = preflight(plan_path, output, upstream)
    sys.dont_write_bytecode = True
    sys.addaudithook(runtime_guard)
    imported = import_native(upstream)
    started_at = utc_now()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # Existing directory, including a concurrent start, is fatal.
    start = {
        "schema": "cfeg.author-etrca-source.start.v1",
        "study_id": plan["study_id"],
        **provenance,
        "started_at": started_at,
        "source_subject_ids": list(SOURCE_IDS),
        "raw_root": str(RAW_ROOT),
        "output_root": str(output),
        "imported_source_hashes": imported,
        "metadata_access": False,
        "held_access": False,
        "retired_access": False,
    }
    publish(output / "start.json", start)
    cache, records = gather_participants(plan)
    if time.monotonic() - start_clock > plan["execution"]["expected_runtime_ceiling_seconds"]:
        raise TimeoutError("Frozen runtime ceiling exceeded")
    rows = rows_from_cache(cache, plan)
    summaries = summarize_rows(rows)
    if len(rows) != 2028 or len(summaries) != 52:
        raise ValueError("Incomplete row/summary grid")
    if sum(arr.nbytes for arr in cache.values()) > plan["execution"]["resource_budget_bytes"]:
        raise ValueError("Uncompressed cache exceeds artifact quota")
    cache_hash = publish(output / "correlations.npz", cache, compressed=True)
    report = {
        "schema": plan["reporting"]["schema"],
        "study_id": plan["study_id"],
        "status": "COMPATIBILITY_ASSESSMENT_COMPLETE",
        **{
            key: provenance[key]
            for key in ("plan_sha256", "source_commit", "source_tree", "upstream_revision")
        },
        "started_at": started_at,
        "completed_at": utc_now(),
        "cache_sha256": cache_hash,
        "raw_files": records,
        "rows": rows,
        "summary": summaries,
    }
    total = sum(path.stat().st_size for path in output.iterdir())
    encoded_size = len(
        (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )
    if total + encoded_size > plan["execution"]["resource_budget_bytes"]:
        raise ValueError("Total artifact quota exceeded")
    publish(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM_ROOT)
    args = parser.parse_args()
    try:
        result = run(args.plan, args.output, args.upstream)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "infrastructure_or_data_inconclusive",
                    "error": str(error),
                    "retry_allowed": False,
                }
            ),
            file=sys.stderr,
        )
        raise
    print(
        json.dumps(
            {"status": result["status"], "rows": len(result["rows"]), "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()

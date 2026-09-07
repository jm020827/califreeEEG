"""Frozen source39 native subset cache producer and one-time study lifecycle.

No selector is implemented here. The separately owned core fits/evaluates from
the explicit cache. Pure functions are usable with artificial arrays only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from scipy import io

PLAN_SHA256 = "ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a"
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
PLAN_PATH = SOURCE_ROOT / "configs/analysis/native_subset_m_source39_v1.json"
RAW_ROOT = Path("/home/whwovy/eeg-data/raw/wearable")
PROJECTION_PATH = Path("/home/whwovy/context-template-artifacts/source39-v1/source-projection.json")
PROJECTION_SHA256 = "1082a81d23ebffe26e8451a9e99f1a7d4c84d0ba2d9d00c6695b3a5c6a5114d4"
BASELINE_PATH = Path("/home/whwovy/author-etrca-artifacts/source39-v1/correlations.npz")
BASELINE_SHA256 = "5e4197d2de7b05130a36a2bd980d395db1adc62dcec9beca2c1dc7f7cca8b21f"
OUTPUT_ROOT = Path("/home/whwovy/native-subset-m-artifacts/source39-v1")
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
ARTIFACTS = (
    "start.json",
    "source-projection.json",
    "features.npz",
    "fold-freezes.json",
    "result.json",
)
CACHE_KEYS = ("a0_r", "expert_r", "q_features")
_NATIVE = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return stream_sha256(stream)


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
        or plan["source_projection"]["path"] != str(PROJECTION_PATH)
        or plan["source_projection"]["sha256"] != PROJECTION_SHA256
        or plan["baseline_reference"]["path"] != str(BASELINE_PATH)
        or plan["baseline_reference"]["sha256"] != BASELINE_SHA256
        or plan["execution"]["output_root"] != str(OUTPUT_ROOT)
        or plan["upstream"]["root"] != str(UPSTREAM_ROOT)
        or plan["upstream"]["revision"] != REVISION
        or not all(
            plan["authority"][key]
            for key in ("metadata_access", "source_projection_reuse", "baseline_cache_read")
        )
        or any(
            plan["authority"][key]
            for key in (
                "held_access",
                "retired_access",
                "manifest_or_full_impedance_access",
                "old_runner_or_seal_access",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("Immutable source39 input authority differs")
    return plan


def preflight(plan_path, output, upstream):
    for supplied, expected in (
        (plan_path, PLAN_PATH),
        (output, OUTPUT_ROOT),
        (upstream, UPSTREAM_ROOT),
    ):
        if Path(supplied) != expected or Path(supplied).resolve() != expected:
            raise ValueError("Only exact non-symlink execution paths are permitted")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Existing output consumes the attempt; no resume or overwrite")
    if Path(__file__).resolve().parents[1] != SOURCE_ROOT:
        raise RuntimeError("Human execution requires the integrated main checkout")
    plan = read_plan(plan_path)
    if (
        git(SOURCE_ROOT, "status", "--porcelain")
        or git(SOURCE_ROOT, "rev-parse", "--abbrev-ref", "HEAD") != "main"
    ):
        raise RuntimeError("Integrated main must be clean and committed")
    if git(upstream, "rev-parse", "HEAD") != REVISION or git(upstream, "status", "--porcelain"):
        raise RuntimeError("Upstream must be clean at the pinned revision")
    for relative, expected in plan["upstream"]["pins"].items():
        if file_sha256(upstream / relative) != expected:
            raise RuntimeError("Upstream numerical source hash differs: " + relative)
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
            hashes[str(path.relative_to(upstream))] = file_sha256(path)
    _NATIVE = SimpleNamespace(
        SCCA=SCCA_canoncorr,
        ETRCA=ETRCA,
        preprocess=wearablepreprocess.preprocess,
        filterbank=wearablepreprocess.filterbank,
        gen_ref_sin=gen_ref_sin,
        suggested_weights_filterbank=wearablepreprocess.suggested_weights_filterbank,
    )
    return hashes


def import_core():
    # Main owns this module; importing the producer for a fixture does not need it.
    return importlib.import_module("native_subset_m_core")


def runtime_guard(event, args):
    """Python defense in depth after library import, not an OS security sandbox."""
    if event == "import" and str(args[0]).startswith(
        ("SSVEPAnalysisToolbox.datasets", "SSVEPAnalysisToolbox.evaluator", "cfeg")
    ):
        raise RuntimeError("Forbidden dataset/evaluator/closed project import")
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
        raise RuntimeError("Network/process access forbidden after preflight")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0]))
    suffix = path.suffix.lower()
    if suffix == ".mat" and path not in {RAW_ROOT / f"S{s:03d}.mat" for s in SOURCE_IDS}:
        raise RuntimeError("Only allowlisted source39 raw files are permitted")
    if suffix in (".h5", ".hdf5", ".npy", ".parquet", ".pkl"):
        raise RuntimeError("Shared/processed data containers are forbidden")
    if suffix == ".npz" and path not in (BASELINE_PATH, OUTPUT_ROOT / "features.npz"):
        raise RuntimeError("Only pinned baseline and new feature caches are permitted")
    if (
        suffix == ".json"
        and path != PROJECTION_PATH
        and path not in {OUTPUT_ROOT / name for name in ARTIFACTS}
    ):
        raise RuntimeError("Only the source-only projection and new artifacts may be read")
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC) and path not in {
        OUTPUT_ROOT / name for name in ARTIFACTS
    }:
        raise RuntimeError("Only the five new artifacts may be written")


def open_regular(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Input must have an exact non-symlink path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise ValueError("Input must be a regular file")
    return stream


def load_projection(plan):
    if (
        plan["source_projection"]["path"] != str(PROJECTION_PATH)
        or plan["source_projection"]["sha256"] != PROJECTION_SHA256
    ):
        raise ValueError("Projection input authority differs")
    with open_regular(PROJECTION_PATH) as stream:
        encoded = stream.read()
    if hashlib.sha256(encoded).hexdigest() != PROJECTION_SHA256:
        raise ValueError("Source-only projection hash mismatch")
    payload = json.loads(encoded)
    expected = plan["source_projection"]
    if (
        payload["manifest_sha256"] != expected["manifest_sha256"]
        or payload["returned_subject_ids"] != list(SOURCE_IDS)
        or payload["returned_packets"] != expected["packets"]
        or payload["returned_rows"] != expected["returned_rows"]
        or payload["columns"] != expected["columns"]
        or len(payload["packets"]) != expected["packets"]
    ):
        raise ValueError("Source-only projection receipt differs")
    # Per-packet vector/order/missingness/key validation is core.projection_arrays.
    return payload


def finite_real(values, shape):
    if (
        not isinstance(values, np.ndarray)
        or values.shape != tuple(shape)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected a finite real numeric array of the exact shape")


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


def scores_from_correlations(correlations, weights):
    row = np.asarray(weights, dtype=np.float64)[None, :]
    return np.asarray([row @ r for r in correlations.reshape(-1, 5, 12)]).reshape(
        *correlations.shape[:-2], 12
    )


def predict_correlations(model, query, weights):
    predicted, r = model.predict(query)
    r = np.asarray(r, dtype=np.float64)
    finite_real(r, (len(query), 5, 12))
    if np.max(np.abs(r)) > 1 + 1e-10:
        raise ValueError("Native correlations exceed their domain")
    if not np.array_equal(predicted, scores_from_correlations(r, weights).argmax(axis=-1)):
        raise ValueError("Native predictions disagree with their score arithmetic")
    return r


def covariance_logs(trials):
    """Current-trial first-band statistic only; does not alter native inputs."""
    x = np.asarray(trials, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    covariance = x @ x.swapaxes(-1, -2) / (x.shape[-1] - 1)
    mean_variance = np.trace(covariance, axis1=-2, axis2=-1) / 8
    shrunk = 0.8 * covariance + (0.2 * mean_variance + np.maximum(mean_variance, 1e-12) * 1e-8)[
        ..., None, None
    ] * np.eye(8)
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("Nonpositive shrinkage covariance")
    return (eigenvectors * np.log(eigenvalues)[..., None, :]) @ eigenvectors.swapaxes(-1, -2)


def score_statistics(scores):
    ordered = np.sort(scores, axis=-1)
    return ordered[..., -1], ordered[..., -1] - ordered[..., -2], scores.std(axis=-1, ddof=0)


def support_consistency(trials, omitted, retained, weights):
    # [class,band,channel,time] -> scalar-centered flattened vectors per band.
    a = trials[omitted].reshape(12, 5, -1).copy()
    b = trials[retained].mean(axis=0).reshape(12, 5, -1)
    a -= a.mean(axis=-1, keepdims=True)
    b -= b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    correlation = np.divide(
        np.sum(a * b, axis=-1), denominator, out=np.zeros((12, 5)), where=denominator > 0
    )
    correlation = np.clip(correlation, -1, 1)
    per_class = correlation @ (np.asarray(weights) / np.sum(weights))
    return float(per_class.mean()), float(per_class.min())


def make_q_features(
    trials,
    a0_all_r,
    expert_r,
    budget,
    interface,
    n_samples,
    headband_order_wet,
    weights_a0,
    weights_etrca,
):
    """33 frozen EEG/shared-context features; no impedance or query-label API."""
    if budget not in (3, 5) or interface not in (0, 1) or headband_order_wet not in (0, 1):
        raise ValueError("Invalid support/context boundary")
    a0_scores = scores_from_correlations(a0_all_r, weights_a0) / np.sum(weights_a0)
    expert_scores = scores_from_correlations(expert_r[: budget + 1], weights_etrca) / np.sum(
        weights_etrca
    )
    full = expert_scores[0]
    a0_query = a0_scores[5:]
    full_max, full_margin, full_std = score_statistics(full)
    a0_max, a0_margin, _ = score_statistics(a0_query)
    full_label, a0_label = full.argmax(axis=-1), a0_query.argmax(axis=-1)
    # Only known SUPPORT labels enter true-class diagnostics.
    support_scores = a0_scores[:budget]
    true_scores = np.diagonal(support_scores, axis1=-2, axis2=-1)
    max_other = np.max(np.where(np.eye(12, dtype=bool), -np.inf, support_scores), axis=-1)
    block_corr = true_scores.mean(axis=-1)
    block_margin = (true_scores - max_other).mean(axis=-1)
    logs = covariance_logs(trials[:, :, 0])
    support_logs = logs[:budget].mean(axis=1)
    distance = np.log1p(
        np.sum((logs[5:, :, None] - support_logs[None, None]) ** 2, axis=(-1, -2)) / 8
    )
    qblock = np.broadcast_to(np.arange(5, 10)[:, None], (5, 12))
    result = np.empty((budget, 5, 12, 33), dtype=np.float64)

    def constant(value):
        return np.full((5, 12), value, dtype=np.float64)

    for omitted in range(budget):
        retained = [b for b in range(budget) if b != omitted]
        omit = expert_scores[omitted + 1]
        omit_max, omit_margin, _ = score_statistics(omit)
        omit_label = omit.argmax(axis=-1)
        max_delta, margin_delta = omit_max - full_max, omit_margin - full_margin
        dq_omitted = distance[..., omitted]
        dq_retained = distance[..., retained].mean(axis=-1)
        dq_delta = dq_omitted - dq_retained
        consistency_mean, consistency_min = support_consistency(
            trials, omitted, retained, weights_etrca
        )
        retained_margin = float(block_margin[retained].mean())
        columns = [
            constant(interface),
            constant(n_samples / 500),
            constant(budget / 5),
            qblock / 9,
            constant(omitted / 4),
            (qblock - omitted) / 9,
            constant(headband_order_wet),
            constant(interface != headband_order_wet),
            full_max,
            full_margin,
            full_std,
            a0_max,
            a0_margin,
            full_label == a0_label,
            omit_max,
            omit_margin,
            np.sqrt(np.mean((omit - full) ** 2, axis=-1)),
            omit_label == full_label,
            omit_label == a0_label,
            max_delta,
            margin_delta,
            constant(block_corr[omitted]),
            constant(block_corr[retained].mean()),
            constant(block_margin[omitted]),
            constant(retained_margin),
            dq_omitted,
            dq_retained,
            dq_delta,
            constant(consistency_mean),
            constant(consistency_min),
            margin_delta**2,
            dq_delta * consistency_mean,
            (block_margin[omitted] - retained_margin) * margin_delta,
        ]
        result[omitted] = np.stack(columns, axis=-1)
    finite_real(result, (budget, 5, 12, 33))
    return result


def compute_participant(raw, plan, native, headband_order_wet):
    finite_real(raw, (8, 710, 2, 10, 12))
    if headband_order_wet not in (0, 1):
        raise ValueError("One shared headband-order bit is required")
    a0 = np.zeros((2, 4, 5, 12, 5, 12), dtype=np.float64)
    experts = np.zeros((2, 4, 2, 6, 5, 12, 5, 12), dtype=np.float64)
    features = np.zeros((2, 4, 2, 5, 5, 12, 33), dtype=np.float64)
    for w, n_samples in enumerate(plan["sample_counts"]):
        references = [
            native.gen_ref_sin(f, 250, n_samples, 5, phase * np.pi)
            for f, phase in zip(plan["frequencies"], plan["phases_pi"])
        ]
        for i, interface in enumerate(plan["interfaces"]):
            trials = np.stack(
                [
                    preprocess_trial(raw[:, :, i, block, label], n_samples, native)
                    for block in range(10)
                    for label in range(12)
                ]
            ).reshape(10, 12, 5, 8, n_samples)
            wa = plan["native_weights"]["A0_author"][interface]
            we = plan["native_weights"]["ETRCA"][interface]
            for weights, method in ((wa, "cca"), (we, "trca")):
                if not np.array_equal(
                    weights, native.suggested_weights_filterbank(5, interface, method)
                ):
                    raise ValueError("Native weights differ from the contract")
            anchor = native.SCCA(
                n_component=1,
                n_jobs=None,
                weights_filterbank=wa,
                force_output_UV=False,
                update_UV=True,
            )
            anchor.fit(ref_sig=references)
            all_a0 = predict_correlations(
                anchor, list(trials.reshape(120, 5, 8, n_samples)), wa
            ).reshape(10, 12, 5, 12)
            if anchor.model["U"] is not None or anchor.model["V"] is not None:
                raise ValueError("A0 cached query-specific filters")
            a0[i, w] = all_a0[5:]
            query = list(trials[5:].reshape(60, 5, 8, n_samples))
            for ki, budget in enumerate(plan["budgets"]):
                subsets = [list(range(budget))] + [
                    [b for b in range(budget) if b != omitted] for omitted in range(budget)
                ]
                for expert, blocks in enumerate(subsets):
                    model = native.ETRCA(n_jobs=None, weights_filterbank=we)
                    model.fit(
                        X=list(trials[blocks].reshape(12 * len(blocks), 5, 8, n_samples)),
                        Y=list(range(12)) * len(blocks),
                    )
                    experts[i, w, ki, expert] = predict_correlations(model, query, we).reshape(
                        5, 12, 5, 12
                    )
                features[i, w, ki, :budget] = make_q_features(
                    trials,
                    all_a0,
                    experts[i, w, ki],
                    budget,
                    i,
                    n_samples,
                    headband_order_wet,
                    wa,
                    we,
                )
    return {"a0_r": a0, "expert_r": experts, "q_features": features}


def load_participant(subject, plan, headband_order_wet):
    if (
        type(subject) is not int
        or subject not in SOURCE_IDS
        or tuple(plan["source_subject_ids"]) != SOURCE_IDS
    ):
        raise ValueError("Participant outside source39")
    if plan["raw_root"] != str(RAW_ROOT) or RAW_ROOT.resolve() != RAW_ROOT:
        raise ValueError("Invalid or symlinked raw root")
    path = RAW_ROOT / f"S{subject:03d}.mat"
    with open_regular(path) as stream:
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
            raise ValueError("Raw file changed while loading")
    record = {
        "subject": subject,
        "path": str(path),
        "sha256": digest,
        "stored_dtype": stored_dtype,
        "shape": list(raw.shape),
    }
    return (
        subject,
        compute_participant(np.asarray(raw, dtype=np.float64), plan, _NATIVE, headband_order_wet),
        record,
    )


def validate_cache(cache, plan):
    if set(cache) != set(CACHE_KEYS):
        raise ValueError("Cache must contain exactly the three declared arrays")
    for key in CACHE_KEYS:
        shape = (len(plan["source_subject_ids"]), *plan["cache"][key]["shape"][1:])
        finite_real(cache[key], shape)
        if cache[key].dtype != np.float64:
            raise ValueError("All cache arrays must be float64")
        if key != "q_features" and np.max(np.abs(cache[key])) > 1 + 1e-10:
            raise ValueError("Correlation domain differs")
    if np.any(cache["expert_r"][:, :, :, 0, 4:] != 0) or np.any(
        cache["q_features"][:, :, :, 0, 3:] != 0
    ):
        raise ValueError("Padded experts and omission features must be exactly zero")


def compare_baseline(cache, old, plan):
    pairs = {
        "a0_r": (cache["a0_r"], old["a0_r"][:, :, :, 5:10], "A0_author"),
        "full_expert_r": (cache["expert_r"][:, :, :, :, 0], old["etrca_chrono_r"], "ETRCA"),
    }
    result = {}
    for key, (new, reference, method) in pairs.items():
        if (
            new.shape != reference.shape
            or new.dtype != np.float64
            or reference.dtype != np.float64
            or not np.isfinite(reference).all()
        ):
            raise ValueError("Baseline array schema differs")
        maximum = float(np.max(np.abs(new - reference)))
        if maximum > 1e-12:
            raise ValueError("Pinned native baseline correlation mismatch: " + key)
        for i, interface in enumerate(plan["interfaces"]):
            weights = plan["native_weights"][method][interface]
            if not np.array_equal(
                scores_from_correlations(new[:, i], weights).argmax(axis=-1),
                scores_from_correlations(reference[:, i], weights).argmax(axis=-1),
            ):
                raise ValueError("Pinned native baseline argmax mismatch: " + key)
        result[key] = {"max_abs_error": maximum, "argmax_exact": True}
    return result


def check_baseline(cache, plan):
    if (
        plan["baseline_reference"]["path"] != str(BASELINE_PATH)
        or plan["baseline_reference"]["sha256"] != BASELINE_SHA256
    ):
        raise ValueError("Baseline cache input authority differs")
    with open_regular(BASELINE_PATH) as stream:
        if stream_sha256(stream) != BASELINE_SHA256:
            raise ValueError("Pinned baseline cache hash mismatch")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as loaded:
            if set(loaded.files) != {"a0_r", "etrca_chrono_r", "etrca_lobo_r"}:
                raise ValueError("Prior baseline cache keys differ")
            old = {key: loaded[key] for key in ("a0_r", "etrca_chrono_r")}
        stream.seek(0)
        if stream_sha256(stream) != BASELINE_SHA256:
            raise ValueError("Baseline cache changed while reading")
    return {
        "input_path": str(BASELINE_PATH),
        "input_sha256": BASELINE_SHA256,
        **compare_baseline(cache, old, plan),
    }


def gather_participants(plan, order):
    if np.shape(order) != (len(SOURCE_IDS),) or not np.isin(order, (0, 1)).all():
        raise ValueError("Expected one source-only shared order bit per participant")
    result = {}
    pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork"))
    try:
        futures = [
            pool.submit(load_participant, subject, plan, int(order[p]))
            for p, subject in enumerate(SOURCE_IDS)
        ]
        for future in as_completed(
            futures, timeout=plan["execution"]["expected_runtime_ceiling_seconds"]
        ):
            subject, arrays, record = future.result()
            if subject in result:
                raise ValueError("Duplicate source participant result")
            result[subject] = (arrays, record)
            print(f"completed participant {subject}: {len(result)}/{len(SOURCE_IDS)}", flush=True)
    except BaseException:
        for process in list(pool._processes.values()):
            process.terminate()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return (
        {key: np.stack([result[s][0][key] for s in SOURCE_IDS]) for key in CACHE_KEYS},
        [result[s][1] for s in SOURCE_IDS],
    )


def json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def publish(path, payload, *, compressed=False, quota=1073741824):
    existing = sum(p.stat().st_size for p in path.parent.iterdir())
    required = (
        sum(arr.nbytes for arr in payload.values()) + 16384
        if compressed
        else len(json_bytes(payload))
    )
    if existing + required > quota:
        raise ValueError("Artifact quota exceeded before publication")
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400), "wb"
    ) as stream:
        if compressed:
            np.savez_compressed(stream, **payload)
        else:
            stream.write(json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_sha256(path)


def run(plan_path, output, upstream):
    clock_start = time.monotonic()
    plan, provenance = preflight(plan_path, output, upstream)
    sys.dont_write_bytecode = True
    # Library import-only temporary files precede the strict artifact guard;
    # no dataset constructors or human inputs are used by either import.
    imported = import_native(upstream)
    core = import_core()
    sys.addaudithook(runtime_guard)
    started_at = utc_now()
    common = {
        key: provenance[key]
        for key in ("plan_sha256", "source_commit", "source_tree", "upstream_revision")
    }
    common.update(study_id=plan["study_id"], started_at=started_at)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    quota = plan["execution"]["resource_budget_bytes"]
    start = {
        "schema": "cfeg.native-subset-m.start.v1",
        **provenance,
        **common,
        "source_subject_ids": list(SOURCE_IDS),
        "raw_root": str(RAW_ROOT),
        "output_root": str(output),
        "imported_source_hashes": imported,
        "core_sha256": file_sha256(Path(core.__file__)),
        "metadata_access": True,
        "source_projection_reuse": True,
        "baseline_cache_read": True,
        "held_access": False,
        "retired_access": False,
        "manifest_or_full_impedance_access": False,
    }
    publish(output / "start.json", start, quota=quota)
    projection = load_projection(plan)
    _, order = core.projection_arrays(projection, plan)
    projection_hash = publish(
        output / "source-projection.json",
        {
            "schema": "cfeg.native-subset-m.projection.v1",
            "input_path": str(PROJECTION_PATH),
            "input_sha256": PROJECTION_SHA256,
            "projection": projection,
        },
        quota=quota,
    )
    cache, raw_files = gather_participants(plan, order)
    validate_cache(cache, plan)
    agreement = check_baseline(cache, plan)
    features_hash = publish(output / "features.npz", cache, compressed=True, quota=quota)
    freeze = core.fit_all(cache, projection, plan)
    if freeze["plan_sha256"] != PLAN_SHA256 or len(freeze["folds"]) != 3:
        raise ValueError("Core did not freeze all three declared folds")
    freeze.update(common, source_projection_sha256=projection_hash, features_sha256=features_hash)
    freeze_hash = publish(output / "fold-freezes.json", freeze, quota=quota)
    evaluated = core.evaluate_all(cache, projection, freeze, plan)
    if len(evaluated["rows"]) != 4680 or len(evaluated["summary"]) != 120:
        raise ValueError("Incomplete evaluation row/summary grid")
    if time.monotonic() - clock_start > plan["execution"]["expected_runtime_ceiling_seconds"]:
        raise TimeoutError("Application-level runtime budget exceeded")
    result = {
        **evaluated,
        **common,
        "schema": plan["reporting"]["schema"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "completed_at": utc_now(),
        "source_projection_sha256": projection_hash,
        "features_sha256": features_hash,
        "fold_freezes_sha256": freeze_hash,
        "raw_files": raw_files,
        "baseline_agreement": agreement,
    }
    publish(output / "result.json", result, quota=quota)
    return result


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

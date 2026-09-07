"""Fixture-only integration check of unchanged toolbox-author eTRCA.

No dataset argument, human data, dataset constructors, fitting grid or efficacy gate.
Run with the separate Python 3.9 environment documented in author_etrca_compatibility.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REVISION = "3344bd199daf78888e364d9db00ae7d8128d2b5f"
PINS = {
    "SSVEPAnalysisToolbox/algorithms/trca.py": "49f67828ff5da4156c0a86d829e9f0dc80d74ba6a4b1c7dcba1e23890dbe342b",
    "SSVEPAnalysisToolbox/utils/wearablepreprocess.py": "248b40b6a5e8e2e6b55b890a40263549f593ca46ad8bec6accc7b3f1ea2260ac",
    "README.md": "9c29b15d84182a096be220786e1d74e5b40f9fddd3402b829274f6073fca0be8",
}
VERSIONS = {
    "numpy": "1.23.4",
    "scipy": "1.13.0",
    "joblib": "1.4.2",
    "scikit-learn": "1.3.0",
    "mat73": "0.63",
    "h5py": "3.11.0",
    "threadpoolctl": "3.5.0",
}


def fixture_guard(event, args):
    """Defense-in-depth for this reviewed fixture, not an OS security sandbox."""
    if event == "import" and str(args[0]).startswith(
        ("SSVEPAnalysisToolbox.datasets", "SSVEPAnalysisToolbox.evaluator")
    ):
        raise RuntimeError("Dataset/evaluator imports are forbidden in fixture mode")
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
        raise RuntimeError("Network/process execution is forbidden in fixture mode")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if Path(path).suffix.lower() in (".mat", ".npy", ".npz", ".parquet", ".pkl"):
            raise RuntimeError("Data-file access is forbidden in fixture mode")


def verify_checkout(upstream):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()

    if git("rev-parse", "HEAD") != REVISION or git("status", "--porcelain"):
        raise RuntimeError("Expected a clean, exact pinned upstream checkout")
    for relative, expected in PINS.items():
        if hashlib.sha256((upstream / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Upstream hash mismatch: {relative}")
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("Use the isolated Python 3.9 environment")
    for name, expected in VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Dependency mismatch: {name}")


def preprocess_trial(raw, n_samples, native):
    """Mirror BaseDataset.get_data_single_trial's slicing, without constructing it."""
    if raw.shape != (8, 710) or n_samples not in (125, 188, 250):
        raise ValueError("Fixture requires 8x710 and one declared window")
    descriptor = SimpleNamespace(srate=250)
    segment = raw[:, 125 : 160 + n_samples].copy()
    filtered = native.filterbank(descriptor, native.preprocess(descriptor, segment))
    return filtered[:, :, 35 : 35 + n_samples].copy()


def run_fixture(upstream):
    verify_checkout(upstream)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    sys.addaudithook(fixture_guard)
    import numpy as np
    from SSVEPAnalysisToolbox.algorithms.trca import ETRCA
    from SSVEPAnalysisToolbox.utils import wearablepreprocess as native

    # Deterministic engineering signals, not a DGP for scientific efficacy.
    rng = np.random.default_rng(20260907)
    frequencies = np.array(
        [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
    )
    t = (np.arange(710) - 160) / 250
    mixing = rng.normal(size=(12, 8, 5))
    raw = []
    labels = []
    for block in range(6):
        for label, freq in enumerate(frequencies):
            harmonics = np.array([np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 6)])
            signal = mixing[label] @ harmonics + 0.4 * rng.normal(size=(8, 710))
            signal += 2 * np.sin(2 * np.pi * 50 * t) + 0.1 * t
            raw.append(signal)
            labels.append(label)
    rows = []
    for n_samples in (125, 188, 250):
        trials = [preprocess_trial(x, n_samples, native) for x in raw]
        # Changing everything outside the permitted slice must not change inputs.
        changed = raw[0].copy()
        changed[:, :125] += 12345
        changed[:, 160 + n_samples :] -= 12345
        np.testing.assert_array_equal(preprocess_trial(changed, n_samples, native), trials[0])
        for interface in ("dry", "wet"):
            weights = native.suggested_weights_filterbank(5, interface, "trca")
            for k in (3, 5):
                train = trials[: 12 * k]
                query = trials[60:72]
                model = ETRCA(n_jobs=None, weights_filterbank=weights)
                model.fit(X=train, Y=labels[: 12 * k])
                before = model.model["U"].copy()
                predicted, r_list = model.predict(query)
                correlations = np.asarray(r_list)
                if correlations.shape != (12, 5, 12) or not np.isfinite(correlations).all():
                    raise AssertionError("Invalid native correlation output")
                scores = np.einsum("f,qfc->qc", weights, correlations)
                np.testing.assert_array_equal(predicted, scores.argmax(axis=1))
                np.testing.assert_array_equal(before, model.model["U"])
                scalar = np.empty_like(correlations)
                for c in range(12):
                    expected = np.mean([train[i] for i in range(c, len(train), 12)], axis=0)
                    np.testing.assert_allclose(model.model["template_sig"][c], expected, atol=1e-12)
                    for q in range(12):
                        for f in range(5):
                            w = model.model["U"][f, c]
                            a = (w.T @ query[q][f]).ravel()
                            b = (w.T @ expected[f]).ravel()
                            a, b = a - a.mean(), b - b.mean()
                            scalar[q, f, c] = (a @ b) / np.sqrt((a @ a) * (b @ b))
                np.testing.assert_allclose(correlations, scalar, atol=5e-12, rtol=0)
                single_pred, single_r = model.predict([query[0]])
                np.testing.assert_array_equal(single_pred, predicted[:1])
                np.testing.assert_allclose(single_r[0], correlations[0], atol=1e-12, rtol=0)
                rotated = ETRCA(n_jobs=None, weights_filterbank=weights)
                rotated.fit(X=train, Y=[(y + 1) % 12 for y in labels[: 12 * k]])
                rotated_pred, rotated_r = rotated.predict(query)
                np.testing.assert_array_equal(rotated_pred, (np.array(predicted) + 1) % 12)
                np.testing.assert_allclose(
                    rotated_r, np.roll(correlations, 1, axis=2), atol=5e-12, rtol=0
                )
                rows.append(
                    {
                        "interface_preset": interface,
                        "n_samples": n_samples,
                        "k": k,
                        "max_scalar_error": float(np.max(np.abs(scalar - correlations))),
                        "synthetic_correct_of_12": int(
                            np.sum(np.array(predicted) == labels[60:72])
                        ),
                    }
                )
    for event, args in [
        ("open", ("/nonexistent/fixture.mat", "r", 0)),
        ("import", ("SSVEPAnalysisToolbox.datasets",)),
        ("socket.connect", (None,)),
        ("subprocess.Popen", (None,)),
    ]:
        try:
            fixture_guard(event, args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Fixture guard negative test failed")
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("SSVEPAnalysisToolbox"):
            continue
        path = Path(module.__file__).resolve()
        relative = path.relative_to(upstream)
        imported[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if "/datasets/" in str(path) or "/evaluator/" in str(path):
            raise AssertionError("Forbidden package imported")
    return {
        "status": "FIXTURE_COMPATIBILITY_PASS",
        "upstream_revision": REVISION,
        "scope": "synthetic engineering only; no human performance reproduction",
        "python": sys.version.split()[0],
        "dependencies": VERSIONS,
        "seed": 20260907,
        "cases": rows,
        "imported_source_hashes": imported,
        "human_data_access": False,
        "metadata_access": False,
        "dataset_constructor_used": False,
        "upstream_numeric_code_modified": False,
        "guard_scope": "Python audit defense-in-depth, not a security sandbox",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    report = run_fixture(args.upstream.resolve())
    report["elapsed_seconds"] = time.perf_counter() - start
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

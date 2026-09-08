"""Direct cross-environment native-zero check on identical artificial bytes via pipes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PYTHON = Path("/home/whwovy/ssvep-author-compatibility-20260907/.venv/bin/python")
UPSTREAM = Path("/home/whwovy/ssvep-author-compatibility-20260907/upstream")


def encode(array):
    data = array.astype("<f8").tobytes()
    return {
        "shape": list(array.shape),
        "data": base64.b64encode(data).decode(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def native_child():
    # This mode imports neither project Torch nor its analysis package.
    from check_author_etrca import fixture_guard, verify_checkout

    verify_checkout(UPSTREAM)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(UPSTREAM))
    sys.addaudithook(fixture_guard)
    import numpy as np
    from SSVEPAnalysisToolbox.algorithms.trca import ETRCA

    packet = json.load(sys.stdin)
    if set(packet) != {"support", "query", "weights"}:
        raise ValueError("Invalid artificial bridge schema")
    values = {}
    for name, record in packet.items():
        data = base64.b64decode(record["data"], validate=True)
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Artificial input byte hash mismatch")
        values[name] = np.frombuffer(data, dtype="<f8").reshape(record["shape"])
        if not np.isfinite(values[name]).all():
            raise ValueError("Nonfinite artificial array")
    x, query, weights = (values[n] for n in ("support", "query", "weights"))
    if x.shape not in ((3, 12, 2, 8, 125), (5, 12, 2, 8, 125)):
        raise ValueError("Only the declared artificial geometry is accepted")
    if query.shape != (24, 2, 8, 125) or weights.shape != (2,):
        raise ValueError("Invalid artificial query/weight geometry")
    model = ETRCA(n_jobs=None, weights_filterbank=weights.tolist())
    model.fit(X=list(x.reshape(-1, 2, 8, 125)), Y=list(np.tile(np.arange(12), len(x))))
    prediction, correlation = model.predict(list(query))
    print(
        json.dumps(
            {
                "prediction": np.asarray(prediction).tolist(),
                "correlation": np.asarray(correlation).tolist(),
                "input_shas": {n: r["sha256"] for n, r in packet.items()},
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            }
        )
    )


def run():
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from cfeg.analysis.task_trca_shape_operator import native_zero_scores

    rng = np.random.default_rng(66473)
    rows = []
    for offset in (0.0, 0.25):
        raw = rng.normal(size=(7, 12, 2, 8, 125))
        t = np.arange(125) / 250
        for label in range(12):
            raw[:, label] += np.sin(2 * np.pi * (9 + 0.5 * label) * t)
        raw -= raw.mean(-1, keepdims=True)
        raw += offset * rng.normal(size=(1, 12, 2, 8, 1))
        query = raw[5:].reshape(24, 2, 8, 125)
        weights = np.array([1.0, 0.4])
        for k in (3, 5):
            support = raw[:k]
            packet = {
                n: encode(v)
                for n, v in (("support", support), ("query", query), ("weights", weights))
            }
            environment = dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            child = subprocess.run(
                [str(NATIVE_PYTHON), "-B", str(Path(__file__).resolve()), "--native-child"],
                input=json.dumps(packet),
                text=True,
                capture_output=True,
                check=True,
                timeout=45,
                env=environment,
            )
            reference = json.loads(child.stdout)
            if reference["input_shas"] != {n: r["sha256"] for n, r in packet.items()}:
                raise AssertionError("Parent/child array bytes differ")
            scores, corr = native_zero_scores(support, query, weights)
            np.testing.assert_allclose(corr, reference["correlation"], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(scores.argmax(-1), reference["prediction"])
            rows.append(
                {
                    "k": k,
                    "offset": offset,
                    "identical_input_shas": reference["input_shas"],
                    "correlation_max_abs_error": float(
                        np.max(abs(corr - reference["correlation"]))
                    ),
                    "predictions_exact": True,
                    "native_numpy": reference["numpy"],
                    "native_python": reference["python"],
                }
            )
    return {
        "status": "ARTIFICIAL_NATIVE_BRIDGE_PASS",
        "seed": 66473,
        "cases": rows,
        "upstream_revision": "3344bd199daf78888e364d9db00ae7d8128d2b5f",
        "human_data_access": False,
        "new_preprocessing_validation": False,
        "scope": "new eta0 wrapper versus actual pinned toolbox on identical pipe-transferred bytes",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    if parser.parse_args().native_child:
        native_child()
    else:
        print(json.dumps(run(), indent=2, allow_nan=False))

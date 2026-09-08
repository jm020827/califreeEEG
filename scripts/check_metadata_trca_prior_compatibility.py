"""Artificial gamma0 comparison against pinned native toolbox, no dataset construction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from check_author_etrca import fixture_guard, verify_checkout


def run(upstream):
    verify_checkout(upstream)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(upstream))
    sys.dont_write_bytecode = True
    sys.addaudithook(fixture_guard)
    import numpy as np
    from SSVEPAnalysisToolbox.algorithms.trca import ETRCA

    # Load only this pure module: cfeg.analysis.__init__ exports unrelated pandas
    # analyses that are intentionally absent from the isolated native environment.
    module_path = Path(__file__).resolve().parents[1] / "src/cfeg/analysis/metadata_trca_prior.py"
    spec = importlib.util.spec_from_file_location("prior_operator_fixture", module_path)
    op = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = op
    spec.loader.exec_module(op)

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
            native = ETRCA(n_jobs=None, weights_filterbank=weights.tolist())
            native.fit(
                X=list(support.reshape(k * 12, 2, 8, 125)), Y=list(np.tile(np.arange(12), k))
            )
            prediction, corr = native.predict(list(query))
            ours = op.fit_trca(support, np.ones((2, 8)), 0.0)
            scores, actual = op.score_trca(ours, query, weights)
            np.testing.assert_allclose(actual, np.asarray(corr), atol=5e-10, rtol=0)
            np.testing.assert_array_equal(scores.argmax(-1), prediction)
            np.testing.assert_allclose(
                ours.templates, np.asarray(native.model["template_sig"]), atol=1e-13, rtol=0
            )
            rows.append(
                {
                    "k": k,
                    "offset": offset,
                    "max_correlation_error": float(np.max(np.abs(actual - np.asarray(corr)))),
                    "predictions_match": True,
                }
            )
    return {
        "status": "ARTIFICIAL_NATIVE_GAMMA0_COMPATIBILITY_PASS",
        "cases": rows,
        "seed": 66473,
        "human_data_access": False,
        "native_preprocessing_reproduced": False,
        "upstream_revision": "3344bd199daf78888e364d9db00ae7d8128d2b5f",
        "scope": "signed ensemble scoring and training only; no efficacy experiment",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().upstream.resolve()), indent=2, sort_keys=True))

"""Full artificial cache integration; no raw EEG, projection, or artifact I/O.

Random Q is intentionally not a signal-extraction oracle. Separate producer
fixtures verify Q33; this checks nine fits, controls, and complete reporting.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_full_artificial_fit_and_independent_reporting_replay():
    core, audit = module("native_subset_m_core"), module("audit_native_subset_m_source")
    plan = json.loads((ROOT / "configs/analysis/native_subset_m_source39_v1.json").read_text())
    rng = np.random.default_rng(337)
    cache = {
        key: rng.uniform(-0.2, 0.2, size=value["shape"]) for key, value in plan["cache"].items()
    }
    cache["expert_r"][:, :, :, 0, 4:] = 0
    cache["q_features"][:, :, :, 0, 3:] = 0
    packets = []
    for j, subject in enumerate(plan["source_subject_ids"]):
        for i, interface in enumerate(plan["interfaces"]):
            for bi, k in enumerate((3, 5)):
                cache["q_features"][j, i, :, bi, :k, ..., 6] = j % 2
                cache["q_features"][j, i, :, bi, :k, ..., 7] = int(j % 2 != i)
            for b in range(10):
                values = rng.uniform(0, 100, 8).tolist()
                if b % 3 == 0:
                    values[b % 8] = None
                packets.append(
                    {
                        "subject_id": subject,
                        "interface": interface,
                        "block_id": b,
                        "impedance_kohm": values,
                        "headband_order": plan["interfaces"][j % 2],
                        "condition_period": "first" if j % 2 == i else "second",
                    }
                )
    projection = {
        "manifest_sha256": plan["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": plan["source_subject_ids"],
        "columns": plan["source_projection"]["columns"],
    }
    freeze = core.fit_all(cache, projection, plan)
    result = core.evaluate_all(cache, projection, freeze, plan)
    json.dumps(freeze, allow_nan=False)
    json.dumps(result, allow_nan=False)
    z, _ = core.projection_arrays(projection, plan)
    scores = audit.cache_scores(cache, plan)
    assert audit.verify_fits(cache, scores, z, plan, freeze) < 2e-10
    rows, diagnostics = audit.replay_evaluation(cache, scores, z, plan, freeze)
    audit.equal(result["rows"], rows, "artificial rows")
    audit.equal(result["diagnostics"], diagnostics, "artificial diagnostics")
    for name, values in zip(
        ("summary", "contrasts", "attainment"), audit.replay_reporting(rows, plan)
    ):
        audit.equal(result[name], values, "artificial " + name)

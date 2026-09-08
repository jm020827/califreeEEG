"""Cold audit of the failed attempt and its completed first source fold only."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_artifact_audit as audit

OUTPUT = Path("/home/whwovy/task-trca-shape-source39-v1-attempt1")
MANIFEST = ROOT / "configs/analysis/task_trca_shape_source39_execution_v1.json"
MANIFEST_SHA = "4440bd74192f877b66e1ac05bee04a120eb3da67ae49e30aedf4a3d28ea406a7"
FAILURE_SHA = "78e540eb45e93378219127c626437a05acecac46633ae8e1fa37924a079f37ae"
START_SHA = "8958839225978ce864f536e9abe2233de2fe8dec04430d5229e5977618c3e871"
ACCESS_SHA = "292035b38ec39262a450e5ceca7ae5b65590c5eda8d9ded37ae62e69e4adcc8b"
EVENTS_SHA = "d4eaba05a8a09a7762592a14ede0727983f6185af6e781606e6a05fb02ab21ec"


def json_pin(path, digest):
    info = path.stat(follow_symlinks=False)
    audit.require(
        path.resolve() == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
        "regular evidence file",
    )
    encoded = path.read_bytes()
    audit.require(hashlib.sha256(encoded).hexdigest() == digest, "pinned evidence bytes")
    return json.loads(encoded)


def check_access(records, ids):
    expected = Counter()
    for fold in (0, 1):
        fit = [p for rank, p in enumerate(ids) if rank % 3 != fold]
        for pid in fit:
            for interface in (0, 1):
                for k in (3, 5):
                    expected[
                        (fold, pid, "fit", "metadata_support", interface, None, tuple(range(k)))
                    ] += 1
                for n in audit.SAMPLES:
                    expected[(fold, pid, "fit", "source_supervision", interface, n, (5,))] += 1
                    for k in (3, 5):
                        expected[(fold, pid, "fit", "support", interface, n, tuple(range(k)))] += 1
    actual = Counter(
        (
            r["outer_fold"],
            r["participant_id"],
            r["role"],
            r["kind"],
            r["interface"],
            r.get("samples"),
            tuple(r["blocks"]),
        )
        for r in records
    )
    audit.require(
        actual == expected, "exact role/participant/interface/window/block access multiset"
    )
    audit.require(not any("freeze_sha256" in r for r in records), "no post-freeze accesses")


def run():
    start = time.perf_counter()
    audit.require(not (OUTPUT / "failure_audit.json").exists(), "exclusive failure audit")
    manifest = json_pin(MANIFEST, MANIFEST_SHA)
    initial = json_pin(OUTPUT / "start.json", START_SHA)
    failure = json_pin(OUTPUT / "failure.json", FAILURE_SHA)
    audit.require(
        initial["manifest"] == manifest
        and initial["manifest_sha256"] == failure["manifest_sha256"] == MANIFEST_SHA,
        "manifest byte/body chain",
    )
    audit.require(
        manifest["output_root"] == str(OUTPUT) and manifest["held60_authorized"] is False, "scope"
    )
    for path, digest in manifest["code_pins"].items():
        audit.require(audit.sha(ROOT / path) == digest, "unchanged science/execution dependencies")
    audit.require(
        failure["status"] == "VALIDITY_FAILURE"
        and failure["state"]
        == {
            "stage": "NESTED_TRAINING",
            "outer_fold": 1,
            "query_access_count": 0,
            "models_frozen": False,
        },
        "terminal boundary",
    )
    expected_names = {
        "start.json",
        "failure.json",
        "access.jsonl",
        "events.jsonl",
        "source0.npz",
        "source1.npz",
        "model0.json",
    }
    audit.require(
        {p.name for p in OUTPUT.iterdir()} == expected_names,
        "no freeze/query/score/result artifacts",
    )
    audit.require(
        audit.sha(OUTPUT / "access.jsonl") == ACCESS_SHA
        and audit.sha(OUTPUT / "events.jsonl") == EVENTS_SHA,
        "event pins",
    )
    records = [json.loads(v) for v in (OUTPUT / "access.jsonl").read_text().splitlines()]
    check_access(records, manifest["source_ids"])
    audit.require(
        set(failure["artifacts"]) == {"source0", "source1", "model0"}, "exact partial artifact set"
    )
    for name, artifact in failure["artifacts"].items():
        path = OUTPUT / (name + (".json" if name.startswith("model") else ".npz"))
        audit.require(
            artifact["path"] == str(path)
            and path.stat().st_size == artifact["bytes"]
            and audit.sha(path) == artifact["sha256"],
            "partial artifact key/path/pin chain",
        )
    model = json_pin(OUTPUT / "model0.json", failure["artifacts"]["model0"]["sha256"])
    audit.require(
        model["source_artifact"] == failure["artifacts"]["source0"]
        and model["manifest_sha256"] == MANIFEST_SHA,
        "first model binding",
    )
    ids = tuple(manifest["source_ids"])
    evaluation_ids = ids[::3]
    fit_ids = tuple(p for p in ids if p not in evaluation_ids)
    with np.load(OUTPUT / "source0.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    np.testing.assert_array_equal(
        data["weights"], np.array(manifest["native_weights"]["ETRCA"])[data["keys"][:, 1]]
    )
    first_fold = audit.audit_training(data, model, fit_ids, evaluation_ids)
    audit.require(
        first_fold == model["independent_source_audit"], "cold first-fold exact audit receipt"
    )
    return {
        "status": "FAILED_ATTEMPT_BOUNDARY_AND_PARTIAL_SOURCE_AUDIT_PASS",
        "scientific_terminal": "VALIDITY_FAILURE",
        "manifest_sha256": MANIFEST_SHA,
        "start_sha256": START_SHA,
        "failure_sha256": FAILURE_SHA,
        "access_sha256": ACCESS_SHA,
        "events_sha256": EVENTS_SHA,
        "source_revision": initial["source_revision"],
        "first_fold": first_fold,
        "access_events": len(records),
        "distinct_source_participants": len({r["participant_id"] for r in records}),
        "completed_outer_folds": 1,
        "completed_pipelines": 10,
        "completed_heads_at_least": 40,
        "completed_optimizer_steps_at_least": 8000,
        "query_or_outcome_access": False,
        "held60_access": False,
        "full_cold_evaluation_audit_executed": False,
        "scope": "exact failure/manifest/partial artifact/access chain and cold first source-fold replay; no final efficacy scores exist",
        "seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    receipt = run()
    with (OUTPUT / "failure_audit.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, indent=2))

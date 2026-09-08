"""Bounded phase2: saved frozen priors on one prefix, no query scoring or refit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_archive as pinned
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_shape_operator as operator
from cfeg.analysis import task_trca_temporal_evaluation as evaluation
from cfeg.analysis import task_trca_temporal_learning as learn

spec = importlib.util.spec_from_file_location(
    "frozen_symmetry_phase1", ROOT / "scripts/diagnose_task_trca_temporal_symmetry.py"
)
phase1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase1)
FIELDS = ("keys", "orders", "q", "m", "available", "packet5", "s", "c")
IDS = pinned.SOURCE_IDS[::3]
KEY = phase1.KEY


def read_json(item):
    with pinned._PinnedFile(item["path"], item["sha256"], item.get("bytes")) as source:
        return json.loads(pinned.pread_exact(source.fd, source.before.st_size, 0))


def load_rows(item):
    with (
        pinned._PinnedFile(item["path"], item["sha256"], item["bytes"]) as source,
        os.fdopen(os.dup(source.fd), "rb") as stream,
        np.load(stream, allow_pickle=False) as arrays,
    ):
        keys = arrays["keys"]
        wanted = np.array([(pid, *KEY[1:]) for pid in IDS])
        positions = []
        for key in wanted:
            found = np.flatnonzero(np.all(keys == key, axis=1))
            phase1.require(len(found) == 1, "Exact full donor-partition diagnostic rows")
            positions.append(found[0])
        result = {name: arrays[name][positions].copy() for name in FIELDS if name != "keys"}
        result["keys"] = keys[positions].copy()
    phase1.require(np.array_equal(result["keys"], wanted), "Canonical fixed thirteen prefixes")
    return result


def frozen_priors(data, pipeline, target):
    """Same CPU operations as frozen_prior, without constructing query/model state."""
    ids = data["keys"][:, 0].tolist()
    k, interface = int(data["keys"][target, 3]), int(data["keys"][target, 1])
    mapping = features.donor_map(
        ids, np.isfinite(data["packet5"][:, :k]), data["orders"].tolist(), interface
    )
    donor = ids.index(mapping[ids[target]])
    qx = pipeline.q_scaler.transform(data["q"][target])
    qlog = learn._head(evaluation._tensor(qx), evaluation._tensor(pipeline.q.coefficients), 0.8)
    result = {}
    for arm in evaluation.POSITIVE_ARMS:
        active_q = torch.zeros_like(qlog) if arm == "ISO" else qlog
        rlog = torch.zeros_like(qlog)
        if arm not in ("ISO", "Q", "MISSING"):
            available = data["available"][target]
            if arm == "Q2":
                values = pipeline.q2_scaler.transform(
                    data["q"][target, ..., 1:3], np.broadcast_to(available, (5, 8))
                )
                coefficients = pipeline.residuals["Q2"].coefficients
            else:
                position = donor if arm in ("SHAM_REFIT", "PERMUTED") else target
                raw = data["m"][position]
                if arm == "STALE":
                    raw, available = features.metadata_features(
                        np.repeat(data["packet5"][target, :1], k, 0)
                    )
                values = pipeline.m_scaler.transform(raw, available)
                coefficients = pipeline.residuals[
                    "SHAM_REFIT" if arm == "SHAM_REFIT" else "QM"
                ].coefficients
            rlog = learn._head(
                evaluation._tensor(values), evaluation._tensor(coefficients), 0.2
            ) * evaluation._tensor(available)
        result[arm] = operator.shape_prior(active_q + rlog)
    phase1.require(torch.equal(result["Q"], result["MISSING"]), "Q/MISSING exact prior")
    return result, ids[donor]


def measure(data, pipeline):
    target = np.flatnonzero(np.all(data["keys"] == np.array(KEY), axis=1))
    phase1.require(len(target) == 1, "Single target prefix")
    target = int(target[0])
    priors, donor = frozen_priors(data, pipeline, target)
    s, c = (
        operator._symmetric(evaluation._tensor(data[name][target]), name, constant=True)
        for name in ("s", "c")
    )
    minimum = torch.linalg.eigvalsh(c)[..., 0]
    phase1.require(bool(torch.all(minimum > 0)), "Strict positive C")
    tau = operator.ETA * minimum / operator.R_MAX
    result = {}
    for arm, r in priors.items():
        b = c + torch.diag_embed(tau[..., None] * r[:, None, :])
        lower = torch.linalg.cholesky(b)
        left = torch.linalg.solve_triangular(lower, s, upper=False)
        h = torch.linalg.solve_triangular(lower, left.mT, upper=False).mT
        row = phase1.symmetry(h)
        row["r"] = r.tolist()
        try:
            operator.leading_projector(h)
            row.update(strict_projector_status="PASS", strict_projector_error=None)
        except ValueError as error:
            row.update(strict_projector_status="REJECTED", strict_projector_error=str(error))
        result[arm] = row
    return {
        "key": list(KEY),
        "donor_id": donor,
        "arms": result,
        "first_rejected_arm_in_execution_order": next(
            (arm for arm, value in result.items() if value["strict_projector_status"] != "PASS"),
            None,
        ),
    }


def run(output):
    output = Path(output).absolute()
    phase1.require(
        output.resolve() == output and not output.exists() and output.parent.is_dir(),
        "New diagnostic output",
    )
    phase1.require(
        all(
            os.environ.get(v) == "1"
            for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "CPU1 only",
    )
    torch.set_num_threads(1)
    began = time.perf_counter()

    def expired(*unused):
        raise TimeoutError("Phase2 sixty-second diagnostic limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        failure = read_json(
            {"path": str(phase1.ATTEMPT / "failure.json"), "sha256": phase1.FAILURE_SHA}
        )
        items = [failure["artifacts"][name] for name in ("source1", "source2")]
        for index, item in enumerate(items, 1):
            phase1.require(
                item["path"] == str(phase1.ATTEMPT / f"source{index}.npz"), "Pinned source path"
            )
        left, right = (load_rows(item) for item in items)
        phase1.require(
            all(np.array_equal(left[name], right[name], equal_nan=True) for name in FIELDS),
            "Duplicate prefix snapshots differ",
        )
        model = read_json(failure["artifacts"]["model0"])
        phase1.require(
            model["selection"]["outer_evaluation_ids"] == list(IDS), "Exact frozen donor partition"
        )
        pipeline = evaluation.pipeline_from_record(model["pipeline"])
        result = {
            "status": "BOUNDED_FROZEN_PRIOR_DIAGNOSIS_COMPLETE",
            "failure_sha256": phase1.FAILURE_SHA,
            "source_descriptors": items,
            "model_descriptor": failure["artifacts"]["model0"],
            "numeric_fields_decoded": list(FIELDS),
            "source_snapshot_equality": True,
            "measurement": measure(left, pipeline),
            "new_query_reads_or_scores": 0,
            "human_optimizer_updates": 0,
            "numerical_policy_changed": False,
            "held60_access": False,
            "elapsed_seconds": time.perf_counter() - began,
            "code_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/diagnose_task_trca_temporal_frozen_prior.py",
                    "scripts/diagnose_task_trca_temporal_symmetry.py",
                    "src/cfeg/analysis/task_trca_shape_operator.py",
                    "src/cfeg/analysis/task_trca_temporal_evaluation.py",
                    "src/cfeg/analysis/task_trca_shape_features.py",
                    "src/cfeg/analysis/task_trca_temporal_learning.py",
                )
            },
            "scope": "One saved-prefix frozen-prior arithmetic replay. No original raw/M archive, query, scoring, refit, policy repair, efficacy or C2 stability claim.",
        }
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        phase1.require(len(encoded.encode()) < 1024**2, "Phase2 one-MiB output cap")
        with output.open("x") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        output.chmod(0o400)
        return result
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "first_rejected_arm": result["measurement"][
                    "first_rejected_arm_in_execution_order"
                ],
                "arms": {
                    arm: {
                        name: row[name]
                        for name in (
                            "maximum_ratio",
                            "failed_band_class_indices_zero_based",
                            "strict_projector_status",
                            "strict_projector_error",
                        )
                    }
                    for arm, row in result["measurement"]["arms"].items()
                },
            },
            indent=2,
        )
    )

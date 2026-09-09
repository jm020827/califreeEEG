"""Freeze new N1 generated inputs or a source39 human execution manifest.

No human numerical artifact is opened. This prepares immutable inputs and/or a
manifest; the actual runtime and independent cold CLI execute separately so
their process-local data guards cannot be bypassed by this fixture builder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis.task_trca_shape_archive import SOURCE_IDS as ROUTING_IDS

SEED = 20260914
IDS = (4, 6, 8, 11, 14, 21, 22, 25, 28)
PROFILE = {
    "source_ids": list(IDS),
    "interfaces": [0, 1],
    "samples": [17],
    "budgets": [3, 5],
    "generated": True,
    "seed": SEED,
}
DESIGN = ROOT / "configs/analysis/task_trca_n1_source39_v1.json"
NATIVE_CONFIGURATION = ROOT / "configs/analysis/metadata_prior_source39_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o400)


def artifact(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def generated_values():
    """One fixed seed; M has no constructed efficacy relation to EEG.

    All ten EEG blocks have independent noise, including four query blocks.
    Unlike the earlier integration fixture, 48 query rows are not tiled12.
    Routing identifiers are labels only; none of these are human observations.
    """
    rng = np.random.default_rng(SEED)
    eeg = {}
    for pid in IDS:
        prototype = rng.normal(size=(2, 12, 5, 8, 17))
        eeg[pid] = prototype[:, None] + rng.normal(size=(2, 10, 12, 5, 8, 17))
        eeg[pid] += rng.normal(size=(2, 10, 12, 5, 8, 1))
    packets = []
    for rank, pid in enumerate(ROUTING_IDS):
        order = ("dry", "wet")[rank % 2]
        for interface in ("dry", "wet"):
            values = rng.uniform(0, 40, size=(10, 8))
            values[0, 1] = 0.0
            values[0, 3] = np.nan
            values[1, 0] = np.nan
            values[3, 2] = np.nan
            for block in range(10):
                packets.append(
                    {
                        "subject_id": pid,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [
                            float(v) if np.isfinite(v) else None for v in values[block]
                        ],
                        "headband_order": order,
                        "condition_period": "first" if interface == order else "second",
                    }
                )
    a0 = {pid: rng.uniform(-0.5, 0.5, size=(2, 4, 12, 5, 12)) for pid in IDS}
    return eeg, packets, a0


def build_inputs(root):
    """New exact fixture directory, tiny N17 stored archives and metadata projection."""
    root = Path(root).absolute()
    if root.resolve() != root or root.exists():
        raise ValueError("Generated input root must be a new unaliased directory")
    root.mkdir()
    # This is a tracked numerical-free configuration, never its referenced data.
    configuration = json.loads(NATIVE_CONFIGURATION.read_text())
    eeg, packets, a0 = generated_values()
    envelope = {
        "schema": "cfeg.GENERATED.task_trca_n1_source39.projection.v1",
        "study_id": "GENERATED-task-trca-n1-source39-v1",
        "plan_sha256": "0" * 64,
        "source_commit": "GENERATED",
        "start_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": list(ROUTING_IDS),
        "columns": configuration["source_projection"]["columns"],
    }
    projection = root / "projection.json"
    write_json(projection, {**envelope, "packets": packets})
    plan = {
        "generated": True,
        "seed": SEED,
        "interfaces": ["dry", "wet"],
        "frequencies": configuration["frequencies"],
        "native_weights": configuration["native_weights"],
        "source_projection": {
            "path": str(projection),
            "sha256": sha(projection),
            "manifest_sha256": envelope["manifest_sha256"],
            "packets": 780,
            "returned_rows": 9360,
            "columns": envelope["columns"],
            "envelope_provenance": {
                key: envelope[key]
                for key in ("schema", "study_id", "plan_sha256", "source_commit", "start_sha256")
            },
        },
    }
    plan_path = root / "native_plan.json"
    write_json(plan_path, plan)
    write_json(root / "start.json", {"generated": True, "seed": SEED, "profile": PROFILE})
    files = []
    for pid in IDS:
        full = np.empty((2, 2, 4, 12, 5, 12))
        for interface in (0, 1):
            weights = configuration["native_weights"]["ETRCA"][("dry", "wet")[interface]]
            query = eeg[pid][interface, 6:10].reshape(48, 5, 8, 17)
            for budget_index, k in enumerate((3, 5)):
                model = native.fit_trca(eeg[pid][interface, :k], np.ones((5, 8)), 0)
                _, corr = native.score_trca(model, query, weights)
                full[interface, budget_index] = corr.reshape(4, 12, 5, 12)
        path = root / f"S{pid:03d}.npz"
        with path.open("xb") as stream:
            np.savez(stream, x_17=eeg[pid], full_17=full, a0_17=a0[pid])
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o400)
        files.append(
            {
                "subject": pid,
                "filename": path.name,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
        )
    result = {
        "status": "GENERATED_COMPLETE",
        "generated": True,
        "seed": SEED,
        "plan_sha256": sha(plan_path),
        "source_subject_ids": list(IDS),
        "start_sha256": sha(root / "start.json"),
        "files": files,
    }
    write_json(root / "result.json", result)
    write_json(
        root / "fixture.json",
        {
            "status": "GENERATED_INPUTS_ONLY",
            "seed": SEED,
            "profile": PROFILE,
            "generator_sha256": sha(__file__),
            "native_function_sha256": sha(ROOT / "src/cfeg/analysis/metadata_trca_prior.py"),
            "native_configuration_sha256": sha(NATIVE_CONFIGURATION),
            "result": artifact(root / "result.json"),
            "plan": artifact(plan_path),
            "projection": artifact(projection),
            "human_artifact_reads": False,
            "a0_scope": "frozen random generated correlations, not an actual k0 efficacy baseline",
            "metadata_effect": "NOT_EVALUATED",
        },
    )
    return plan_path, plan


def prepare(
    root, resource_path, resource_sha, *, human=False, preflight_path=None, preflight_sha=None
):
    """One new append-only manifest/fixture, never a retry of an earlier candidate."""
    root = Path(root).absolute()
    if (
        root.resolve() != root
        or not root.is_dir()
        or any(root.iterdir())
        or root.parent != Path("/home/whwovy")
        or not root.name.startswith("task-trca-n1-source39-")
    ):
        raise ValueError("Fresh empty exact task-trca-n1-source39 parent required")
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONDONTWRITEBYTECODE",
        )
    ):
        raise ValueError("CPU1/no-bytecode launch required")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        subprocess.run(command, cwd=ROOT, check=True)
    expected = "bb90cc361a28c3e0a53861564ec007f38602cf5d16554e0923a6550eb1be7c95"
    if sha(DESIGN) != expected:
        raise ValueError("Frozen source39 science changed")
    spec = importlib.util.spec_from_file_location(
        "new_source39_cold", ROOT / "scripts/audit_task_trca_n1_source39.py"
    )
    cold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cold)
    pins = {path: sha(ROOT / path) for path in cold.CODE_PATHS}
    design = json.loads(DESIGN.read_text())
    resource_path = Path(resource_path).absolute()
    if (
        resource_path.resolve() != resource_path
        or resource_path.name != "resource.json"
        or sha(resource_path) != resource_sha
    ):
        raise ValueError("Exact generated resource proof")
    resource_proof = json.loads(resource_path.read_text())
    if (
        resource_proof.get("status") != "GENERATED_RESOURCE_PASS"
        or resource_proof.get("code_pins") != pins
        or resource_proof.get("design_sha256") != expected
        or resource_proof.get("updates_completed") != 800
        or resource_proof.get("human_reads") is not False
    ):
        raise ValueError("Same-code completed full-shape resource proof required")
    # Complete prerequisite verification precedes registration and the only RNG.
    # These pure checks do not follow human input paths.
    cold.prior_binding(design, pins)
    prerequisite_manifest = {
        "resource_preflight": artifact(resource_path),
        "code_pins": pins,
        "design": {"path": str(DESIGN), "sha256": expected},
    }
    cold.resource_binding(prerequisite_manifest)
    preflight = None
    if human:
        preflight_path = Path(preflight_path).absolute()
        if (
            preflight_path.resolve() != preflight_path
            or preflight_path.name != "cold_audit.json"
            or sha(preflight_path) != preflight_sha
        ):
            raise ValueError("Exact new generated cold proof")
        previous = json.loads(preflight_path.read_text())
        if (
            previous.get("status") != "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
            or previous.get("code_pins") != pins
            or previous.get("profile") != PROFILE
            or previous.get("design_sha256") != expected
        ):
            raise ValueError("Same-code new generated completion required")
        preflight = artifact(preflight_path)
        cold.generated_preflight_binding({**prerequisite_manifest, "preflight": preflight})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    profile = design["human_profile"] if human else PROFILE
    started = time.monotonic()
    write_json(
        root / "registration.json",
        {
            "schema": "cfeg.task_trca_n1_source39.registration.v1",
            "status": "BEFORE_HUMAN_ATTEMPT" if human else "BEFORE_GENERATION",
            "study_id": design["study_id"],
            "generated": not human,
            "profile": profile,
            "source_revision": revision,
            "design_sha256": expected,
            "code_pins": pins,
            "human_numeric_reads": False,
            "gpu_used": False,
        },
    )
    try:
        if human:
            inputs = design["human_inputs"]
            native_root = Path(inputs["native_root"])
            plan_path = Path(inputs["native_plan_path"])
            # Numeric-free tracked configuration only. No native result/EEG/M read here.
            plan = json.loads(plan_path.read_text())
            plan_sha = inputs["native_plan_sha256"]
            native_sha = inputs["native_manifest_sha256"]
            fixture = None
            input_bytes = 0
            output_budget = 12 * 1024**3
        else:
            native_root = root / "inputs"
            plan_path, plan = build_inputs(native_root)
            plan_sha = sha(plan_path)
            native_sha = sha(native_root / "result.json")
            fixture = artifact(native_root / "fixture.json")
            input_bytes = sum(p.stat().st_size for p in native_root.iterdir())
            output_budget = 2 * 1024**3 - input_bytes - 2 * 1024**2
            if output_budget < 64 * 1024**2:
                raise ValueError("Generated input/output reserve")
        output = root / (
            "task-trca-n1-source39-primary1" if human else "task-trca-n1-source39-generated1"
        )
        manifest_path = root / ("human_manifest.json" if human else "generated_manifest.json")
        manifest = {
            "schema": "cfeg.task_trca_n1_source39.execution.v1"
            if human
            else "cfeg.task_trca_n1_source39.generated.v1",
            "study_id": design["study_id"],
            "algorithm_schema": "task-trca-n1-integration-v1",
            "score_schema": "component-time-centered-ensemble-pearson-v1",
            "attempt_id": output.name,
            "status": "EXECUTION_FROZEN" if human else "GENERATED_FROZEN",
            "generated": not human,
            "seed": profile["seed"],
            "profile": profile,
            "source_ids": profile["source_ids"],
            "held60_authorized": False,
            "retired1_3_authorized": False,
            "old_attempts_reopened": False,
            "design": {"path": str(DESIGN), "sha256": expected},
            "native_plan": {"path": str(plan_path), "sha256": plan_sha},
            "native_root": str(native_root),
            "native_manifest_sha256": native_sha,
            "fixture": fixture,
            "native_weights": {
                kind: [plan["native_weights"][kind][interface] for interface in ("dry", "wet")]
                for kind in ("ETRCA", "A0_author")
            },
            "code_pins": pins,
            "device": "cpu",
            "backend": "batch",
            "precision": "float64",
            "cpu_threads": 1,
            "output_root": str(output),
            "output_budget_bytes": output_budget,
            "max_seconds": 21600 if human else 1800,
            "optimizer_updates_max": 24000,
            "preflight": preflight,
            "resource_preflight": artifact(resource_path),
        }
        write_json(manifest_path, manifest)
        if not human:
            cold.validate_manifest(manifest, sha(manifest_path))
        return {
            "status": "HUMAN_MANIFEST_PREPARED_NO_NUMERIC_READ"
            if human
            else "GENERATED_INPUTS_AND_MANIFEST_PREPARED",
            "manifest": artifact(manifest_path),
            "fixture": fixture,
            "input_bytes": input_bytes,
            "elapsed_seconds": time.monotonic() - started,
            "human_numeric_reads": False,
            "optimizer_updates": 0,
        }
    except BaseException as error:
        write_json(
            root / "prepare_failure.json",
            {
                "status": "PREPARATION_FAILURE",
                "generated": not human,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "human_numeric_reads": False,
                "registered_generations_charged": 0 if human else 1,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resource-preflight", type=Path, required=True)
    parser.add_argument("--resource-sha256", required=True)
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.root,
                args.resource_preflight,
                args.resource_sha256,
                human=args.human,
                preflight_path=args.preflight,
                preflight_sha=args.preflight_sha256,
            )
        )
    )

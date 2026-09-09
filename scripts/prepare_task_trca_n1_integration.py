"""Build fixed GENERATED native archives for the real temporal completed-path rehearsal.

No human artifact is opened. This prepares immutable fixture inputs and a
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

SEED = 20260913
IDS = (4, 6, 8, 11, 14, 21, 22, 25, 28)
PROFILE = {
    "source_ids": list(IDS),
    "interfaces": [0, 1],
    "samples": [17],
    "budgets": [3, 5],
    "generated": True,
    "seed": SEED,
}
DESIGN = ROOT / "configs/analysis/task_trca_n1_integration_v1.json"
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
        "schema": "cfeg.GENERATED.task_trca_n1.projection.v1",
        "study_id": "GENERATED-task-trca-n1-integration-v1",
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


def prepare(root):
    """One append-only generation after all implementation pins are committed."""
    root = Path(root).absolute()
    if (
        root.resolve() != root
        or not root.is_dir()
        or any(root.iterdir())
        or not root.name.startswith("task-trca-n1-integration-")
    ):
        raise ValueError("Fresh empty unaliased task-trca-n1-integration-* parent required")
    if any(
        os.environ.get(k) != "1"
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("Launch with BLAS/OpenMP1 before imports")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        subprocess.run(command, cwd=ROOT, check=True)
    expected = "15d5ee385bc0f30d5dff780ae582034597db06793d34c9c8029c44f42adf25cd"
    if sha(DESIGN) != expected:
        raise ValueError("Frozen N1 integration design changed")
    spec = importlib.util.spec_from_file_location(
        "n1_cold_contract", ROOT / "scripts/audit_task_trca_n1_integration.py"
    )
    cold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cold)
    pins = {path: sha(ROOT / path) for path in cold.CODE_PATHS}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    started = time.monotonic()
    write_json(
        root / "registration.json",
        {
            "schema": "cfeg.task_trca_n1.registration.v1",
            "status": "BEFORE_GENERATION",
            "study_id": "task-trca-n1-integration-v1",
            "seed": SEED,
            "profile": PROFILE,
            "source_revision": revision,
            "design_sha256": expected,
            "code_pins": pins,
            "human_reads": False,
            "gpu_used": False,
        },
    )
    try:
        input_root = root / "inputs"
        output_root = root / "task-trca-n1-integration-primary1"
        manifest_path = root / "manifest.json"
        plan_path, plan = build_inputs(input_root)
        input_bytes = sum(p.stat().st_size for p in input_root.iterdir())
        if input_bytes >= 2 * 1024**3 - 64 * 1024**2:
            raise ValueError("Generated input leaves insufficient output reservation")
        manifest = {
            "schema": "cfeg.task_trca_n1.generated_execution.v1",
            "study_id": "task-trca-n1-integration-v1",
            "score_schema": "component-time-centered-ensemble-pearson-v1",
            "attempt_id": output_root.name,
            "status": "GENERATED_FROZEN",
            "generated": True,
            "seed": SEED,
            "profile": PROFILE,
            "source_ids": list(IDS),
            "held60_authorized": False,
            "retired1_3_authorized": False,
            "old_attempts_reopened": False,
            "design": {"path": str(DESIGN), "sha256": expected},
            "native_plan": {"path": str(plan_path), "sha256": sha(plan_path)},
            "native_root": str(input_root),
            "native_manifest_sha256": sha(input_root / "result.json"),
            "fixture": artifact(input_root / "fixture.json"),
            "native_weights": {
                kind: [plan["native_weights"][kind][interface] for interface in ("dry", "wet")]
                for kind in ("ETRCA", "A0_author")
            },
            "code_pins": pins,
            "device": "cpu",
            "backend": "batch",
            "precision": "float64",
            "cpu_threads": 1,
            "output_root": str(output_root),
            "output_budget_bytes": 2 * 1024**3 - input_bytes - 2 * 1024**2,
            "max_seconds": 7200,
            "optimizer_updates_max": 24000,
            "preflight": None,
        }
        write_json(manifest_path, manifest)
        cold.validate_manifest(manifest, sha(manifest_path))
        return {
            "status": "GENERATED_INPUTS_AND_MANIFEST_PREPARED",
            "manifest": artifact(manifest_path),
            "fixture": manifest["fixture"],
            "elapsed_seconds": time.monotonic() - started,
            "input_bytes": input_bytes,
            "human_reads": False,
            "optimizer_updates": 0,
        }
    except BaseException as error:
        write_json(
            root / "prepare_failure.json",
            {
                "status": "GENERATED_PREPARATION_FAILURE",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "human_reads": False,
                "registered_generation_attempts": 1,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root)))

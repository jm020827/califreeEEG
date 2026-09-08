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
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis.task_trca_shape_archive import SOURCE_IDS as ROUTING_IDS

SEED = 20260909
IDS = (4, 6, 8, 11, 14, 21, 22, 25, 28)
PROFILE = {
    "source_ids": list(IDS),
    "interfaces": [0, 1],
    "samples": [17],
    "budgets": [3, 5],
    "generated": True,
    "seed": SEED,
}
DESIGN = ROOT / "configs/analysis/task_trca_temporal_v1_design.json"
PROGRAM = ROOT / "configs/analysis/metadata_learning_program_v1.json"
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
        "schema": "cfeg.GENERATED.temporal.projection.v1",
        "study_id": "GENERATED-temporal-completed-path-20260909",
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


def prepare(input_root, output_root, manifest_path, *, device):
    output_root, manifest_path = Path(output_root).absolute(), Path(manifest_path).absolute()
    if output_root.exists() or output_root.resolve() != output_root:
        raise ValueError("Generated runtime output must not exist or alias another path")
    if manifest_path.exists():
        raise ValueError("Generated manifest is append-only")
    spec = importlib.util.spec_from_file_location(
        "temporal_cold_contract", ROOT / "scripts/audit_task_trca_temporal_source39.py"
    )
    cold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cold)
    plan_path, plan = build_inputs(input_root)
    manifest = {
        "schema": "cfeg.task_trca_temporal.source39_generated.v1",
        "study_id": "task-trca-temporal-v1",
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
        "program_path": str(PROGRAM),
        "program_sha256": sha(PROGRAM),
        "design": {"path": str(DESIGN), "sha256": sha(DESIGN)},
        "native_plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "native_root": str(Path(input_root).absolute()),
        "native_manifest_sha256": sha(Path(input_root) / "result.json"),
        "native_weights": {
            kind: [plan["native_weights"][kind][interface] for interface in ("dry", "wet")]
            for kind in ("ETRCA", "A0_author")
        },
        "code_pins": {path: sha(ROOT / path) for path in cold.CODE_PATHS},
        "device": device,
        "backend": "batch",
        "precision": "float64",
        "cpu_threads": 1,
        "gpu_memory_fraction": 0.16,
        "output_root": str(output_root),
        "output_budget_bytes": 4 * 1024**3,
        "max_seconds": 7200,
        "optimizer_updates_max": 24000,
        "preflight": None,
    }
    write_json(manifest_path, manifest)
    return {
        "manifest": artifact(manifest_path),
        "fixture": artifact(Path(input_root) / "fixture.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_root, args.output_root, args.manifest, device=args.device)))

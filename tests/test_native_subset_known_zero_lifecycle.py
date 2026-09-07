"""Serialized artificial parent→real guarded IO→policy→independent full audit.

Only input-authority/preflight/path declarations are substituted for temporary
artificial bundles. All parsing, parent replay, new policy and audit math run.
"""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def serialized(tmp_path_factory):
    io_owner = load("scripts/native_subset_known_zero_io.py")
    data = load("tests/test_native_subset_known_zero_source.py").synthetic_parent()
    plan = copy.deepcopy(io_owner.load_plan())
    science = data["science"]
    root = tmp_path_factory.mktemp("kz-artificial")
    parent = root / "parent"
    parent.mkdir()
    plan["parent"]["root"] = str(parent)
    plan["output_root"] = str(root / "revised")
    amendment = json.loads((ROOT / plan["parent"]["execution_plan_path"]).read_text())
    amendment["output_root"] = str(parent)
    common = {
        key: plan["parent"][key]
        for key in ("attempt_id", "study_id", "source_commit", "source_tree")
    }
    common.update(
        execution_plan_sha256=plan["parent"]["execution_plan_sha256"],
        plan_sha256=plan["science"]["sha256"],
        upstream_revision=science["upstream"]["revision"],
        started_at="2026-09-07T14:22:00+00:00",
    )
    start = {
        **common,
        "schema": "cfeg.native-subset-m.start.v1",
        "core_sha256": amendment["source_helpers"]["scripts/native_subset_m_core.py"],
        "source_helper_hashes": amendment["source_helpers"],
        "imported_source_hashes": science["upstream"]["pins"],
        "source_subject_ids": science["source_subject_ids"],
        "raw_root": science["raw_root"],
        "output_root": str(parent),
        "python": science["execution"]["python_version"],
        "dependencies": science["execution"]["dependencies"],
        "metadata_access": True,
        "source_projection_reuse": True,
        "baseline_cache_read": True,
        "held_access": False,
        "retired_access": False,
        "manifest_or_full_impedance_access": False,
    }
    budget, pins = plan["runtime"]["parent_budget_bytes"], {}
    pins["start.json"] = io_owner.publish(parent / "start.json", start, budget)
    wrapper = {
        "schema": "cfeg.native-subset-m.projection.v1",
        "input_path": science["source_projection"]["path"],
        "input_sha256": science["source_projection"]["sha256"],
        "projection": {**data["content_projection"], **amendment["envelope_provenance"]},
    }
    pins["source-projection.json"] = io_owner.publish(
        parent / "source-projection.json", wrapper, budget
    )
    with os.fdopen(
        os.open(parent / "features.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400), "wb"
    ) as stream:
        np.savez_compressed(stream, **data["cache"])
        stream.flush()
        os.fsync(stream.fileno())
    pins["features.npz"] = io_owner.digest((parent / "features.npz").read_bytes())
    freeze = {
        **data["freezes"],
        **common,
        "features_sha256": pins["features.npz"],
        "source_projection_sha256": pins["source-projection.json"],
    }
    pins["fold-freezes.json"] = io_owner.publish(parent / "fold-freezes.json", freeze, budget)
    result = {
        **data["parent_result"],
        **common,
        "schema": science["reporting"]["schema"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "completed_at": "2026-09-07T14:22:01+00:00",
        "features_sha256": pins["features.npz"],
        "source_projection_sha256": pins["source-projection.json"],
        "fold_freezes_sha256": pins["fold-freezes.json"],
        "raw_files": [
            {
                "subject": p,
                "path": f"{science['raw_root']}/S{p:03d}.mat",
                "sha256": "b" * 64,
                "stored_dtype": "float64",
                "shape": science["raw_shape"],
            }
            for p in science["source_subject_ids"]
        ],
        "baseline_agreement": {
            "input_path": science["baseline_reference"]["path"],
            "input_sha256": science["baseline_reference"]["sha256"],
            "a0_r": {"max_abs_error": 0.0, "argmax_exact": True},
            "full_expert_r": {"max_abs_error": 0.0, "argmax_exact": True},
        },
    }
    pins["result.json"] = io_owner.publish(parent / "result.json", result, budget)
    plan["parent"]["artifacts"] = pins
    plan["parent"]["total_bytes"] = sum(path.stat().st_size for path in parent.iterdir())
    hashes = {
        **plan["pinned_helpers"],
        str(io_owner.PLAN_PATH.relative_to(ROOT)): io_owner.PLAN_SHA256,
        plan["mechanism"]["path"]: plan["mechanism"]["sha256"],
        plan["science"]["path"]: plan["science"]["sha256"],
        plan["parent"]["execution_plan_path"]: plan["parent"]["execution_plan_sha256"],
    }
    hashes.update(
        {path: io_owner.digest((ROOT / path).read_bytes()) for path in plan["code_paths"]}
    )
    state = {
        "plan": plan,
        "science": science,
        "amendment": amendment,
        "source_hashes": hashes,
        "commit": io_owner.git("rev-parse", "HEAD").decode().strip(),
        "tree": io_owner.git("rev-parse", "HEAD^{tree}").decode().strip(),
    }
    return root, state


def launch(root, state, name):
    settings = root / (name + ".json")
    settings.write_text(json.dumps(state))
    command = """
import json, sys
import native_subset_known_zero_io as io_owner
import run_native_subset_known_zero_source as runner
state=json.loads(open(sys.argv[1]).read())
plan=state['plan']
modules={__import__('pathlib').Path(p).stem:io_owner.compile_helper(p,h)
         for p,h in plan['pinned_helpers'].items()}
modules['native_subset_m_core'].interval([0.0,0.1,0.2])
def artificial_preflight(path):
    if __import__('pathlib').Path(plan['output_root']).exists():
        raise FileExistsError('Artificial attempt already consumed')
    return (plan,state['science'],state['amendment'],modules,state['commit'],state['tree'],state['source_hashes'])
io_owner.preflight=artificial_preflight
io_owner.execute(runner.evaluate)
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "scripts"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    return subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-c", command, str(settings)],
        capture_output=True,
        check=False,
        text=True,
        env=env,
        timeout=180,
        cwd=ROOT,
    )


def test_serialized_parent_actual_guard_to_policy_and_full_independent_audit(
    serialized, monkeypatch
):
    root, state = serialized
    process = launch(root, state, "valid")
    assert process.returncode == 0, process.stdout + process.stderr
    assert "DEVELOPMENT_ASSESSMENT_COMPLETE" in process.stdout
    auditor = load("scripts/audit_native_subset_known_zero_source.py")
    # Artificial parent/output identity only; all source hashes, receipts,
    # original envelope/provenance APIs and independent mathematics remain real.
    monkeypatch.setattr(auditor, "load_plan", lambda path: state["plan"])
    configurations = auditor.load_configurations

    def artificial_parent_paths(repository, plan):
        configs = configurations(repository, plan)
        key = plan["parent"]["execution_plan_path"]
        configs[key] = {**configs[key], "output_root": plan["parent"]["root"]}
        return configs

    monkeypatch.setattr(auditor, "load_configurations", artificial_parent_paths)
    audited = auditor.audit(state["plan"]["output_root"], repository=ROOT)
    assert audited["status"] == "PASS" and audited["revision_diagnostics"] == 624
    assert audited["independently_replayed_parent_routers"] == 9
    assert not audited["policy_refit"]
    retry = launch(root, state, "retry")
    assert retry.returncode != 0 and "already consumed" in retry.stderr


def test_bad_parent_hash_leaves_start_only_and_no_revised_outcome(serialized):
    root, original = serialized
    state = copy.deepcopy(original)
    state["plan"]["output_root"] = str(root / "rejected")
    state["plan"]["parent"]["artifacts"]["start.json"] = "0" * 64
    result = launch(root, state, "bad-hash")
    assert result.returncode != 0 and "Parent artifact hash" in result.stderr
    output = Path(state["plan"]["output_root"])
    assert {p.name for p in output.iterdir()} == {"start.json"}

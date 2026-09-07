"""Artifact-only envelope-r1 audit with pinned, independent scientific replay.

This entrypoint owns the new execution/projection transport contract. It imports
only the explicitly authorized, hash-pinned independent auditor definitions and
never calls that module's old audit/main entrypoints or changes its globals.
No original human input, old cache or previous-attempt artifact is opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np

EXECUTION_PLAN_SHA256 = "93108bd8d0cce8b0572e6cbc200208c3990be725ffdbdf2fdcc0bc8daec82115"
SCIENTIFIC_PLAN_SHA256 = "ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a"
STUDY_ID = "native-subset-m-source39-v1"
ATTEMPT_ID = "native-subset-m-source39-v1-envelope-r1"
AMENDMENT_RELATIVE = "configs/analysis/native_subset_m_source39_envelope_r1.json"
AUDIT_HELPER = "scripts/audit_native_subset_m_source.py"
CONTENT_KEYS = (
    "manifest_sha256",
    "packets",
    "returned_rows",
    "returned_packets",
    "returned_subject_ids",
    "columns",
)
HELPERS = {
    "scripts/run_native_subset_m_source.py": "4516f7d68cf19adebc484c27e77e2afc0a66871f19dcc5e0fb40243458f5fb59",
    "scripts/native_subset_m_core.py": "20a3e3e505f00c2c43af51a5166457031838e4140cf7af15cdb2f427df904014",
    "scripts/audit_native_subset_m_source.py": "937368d639944184d71083fa67829640435e02cfeb3bd5f0f7f573f4b5224245",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_amendment(amendment):
    require(
        amendment["schema"] == "cfeg.native-subset-m.execution-amendment.v1", "Execution schema"
    )
    require(amendment["attempt_id"] == ATTEMPT_ID, "Execution attempt identity")
    require(
        amendment["study_id"] == amendment["randomization_namespace"] == STUDY_ID,
        "Scientific identity / SHAM namespace changed",
    )
    require(
        amendment["scientific_plan"]["sha256"] == SCIENTIFIC_PLAN_SHA256,
        "Scientific plan binding changed",
    )
    require(amendment["scientific_changes"] == [], "Execution amendment cannot change science")
    require(amendment["content_keys"] == list(CONTENT_KEYS), "Six declared content keys")
    require(amendment["source_helpers"] == HELPERS, "Exact three frozen helper definitions")
    require(
        set(amendment["envelope_provenance"])
        == {"schema", "study_id", "plan_sha256", "source_commit", "start_sha256"},
        "Five envelope provenance fields",
    )
    require(
        amendment["previous_attempt"]["preserve"] is True
        and amendment["previous_attempt"]["read_contents"] is False
        and amendment["output_root"] != amendment["previous_attempt"]["path"],
        "Previous attempt preservation / distinct destination",
    )
    execution = amendment["execution"]
    require(
        execution["cpu_workers"] == 4 and execution["blas_threads"] == 1,
        "Frozen resource concurrency",
    )
    for flag in (
        "retry_same_attempt",
        "held_access",
        "retired_access",
        "manifest_or_full_impedance_access",
        "new_dependencies",
    ):
        require(execution[flag] is False, "Forbidden execution authority: " + flag)


def load_amendment(path):
    raw = Path(path).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == EXECUTION_PLAN_SHA256, "Execution amendment byte hash"
    )
    amendment = json.loads(raw)
    check_amendment(amendment)
    return amendment


def content_projection(envelope, amendment):
    """Validate the exact original 11-key envelope and return an isolated six-key view."""
    check_amendment(amendment)
    provenance = amendment["envelope_provenance"]
    require(
        type(envelope) is dict and set(envelope) == set(CONTENT_KEYS) | set(provenance),
        "Projection must retain exact eleven-key envelope",
    )
    for key, expected in provenance.items():
        require(
            type(envelope[key]) is type(expected) and envelope[key] == expected,
            "Pinned envelope provenance: " + key,
        )
    return copy.deepcopy({key: envelope[key] for key in CONTENT_KEYS})


def load_helper(repository, amendment):
    """Compile authenticated definitions without writing an import bytecode cache."""
    check_amendment(amendment)
    path = Path(repository) / AUDIT_HELPER
    require(
        path.resolve() == path.absolute() and path.is_file() and not path.is_symlink(),
        "Independent helper must be the regular non-symlink source",
    )
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == HELPERS[AUDIT_HELPER], "Independent helper byte hash"
    )
    module = ModuleType("pinned_native_subset_m_independent_audit_r1")
    module.__file__ = str(path)
    # Authenticated local definitions only; avoids importlib's optional .pyc write.
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    require(module.PLAN_SHA256 == SCIENTIFIC_PLAN_SHA256, "Independent helper scientific plan")
    return module


def verify_execution_provenance(root, plan, amendment, payloads, hashes, repository, helper):
    check_amendment(amendment)
    require(str(root) == amendment["output_root"], "Exact amendment output destination")
    require(plan["study_id"] == amendment["randomization_namespace"], "Unchanged SHAM namespace")
    for name in ("start.json", "fold-freezes.json", "result.json"):
        receipt = payloads[name]
        helper.equal(receipt["attempt_id"], ATTEMPT_ID, name + " attempt_id")
        helper.equal(
            receipt["execution_plan_sha256"],
            EXECUTION_PLAN_SHA256,
            name + " execution amendment receipt",
        )
        helper.equal(receipt["plan_sha256"], SCIENTIFIC_PLAN_SHA256, name + " science receipt")
        helper.equal(receipt["study_id"], STUDY_ID, name + " study identity")
    start = payloads["start.json"]
    helper.equal(start["source_helper_hashes"], HELPERS, "Start frozen helper hashes")
    helper.equal(start["core_sha256"], HELPERS["scripts/native_subset_m_core.py"], "Start core pin")
    # Only the execution destination is adapted for the original provenance
    # checker. No scientific fields, helper constants, or caller plan are changed.
    runtime_plan = copy.deepcopy(plan)
    runtime_plan["execution"]["output_root"] = amendment["output_root"]
    helper.verify_provenance(root, runtime_plan, payloads, hashes, repository)
    for relative, expected in [(AMENDMENT_RELATIVE, EXECUTION_PLAN_SHA256), *HELPERS.items()]:
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", start["source_commit"] + ":" + relative]
        )
        helper.equal(
            hashlib.sha256(blob).hexdigest(), expected, "Committed execution source " + relative
        )


def audit(output_root, execution_plan_path, repository=None):
    root = Path(output_root)
    repository = Path(repository) if repository is not None else Path(__file__).resolve().parents[1]
    amendment = load_amendment(execution_plan_path)
    require(str(root) == amendment["output_root"], "Frozen amendment output root")
    helper = load_helper(repository, amendment)
    plan = helper.load_plan(amendment["scientific_plan"]["path"])
    require(plan["study_id"] == amendment["study_id"], "Scientific study unchanged")
    require(
        root.is_dir() and not root.is_symlink() and root.resolve() == root,
        "Output must be an absolute non-symlink directory",
    )
    names = plan["execution"]["artifacts"]
    require({path.name for path in root.iterdir()} == set(names), "Exact five-artifact inventory")
    hashes, payloads, times, size = {}, {}, [], 0
    for name in names:
        path = root / name
        info = path.lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            "Immutable regular artifact " + name,
        )
        times.append(info.st_mtime_ns)
        size += info.st_size
        hashes[name] = helper.digest(path)
        if name.endswith(".json"):
            payloads[name] = json.loads(path.read_bytes())
    require(times == sorted(times), "Publication order by artifact mtimes")
    require(
        size <= amendment["execution"]["resource_budget_bytes"], "Execution artifact byte budget"
    )
    verify_execution_provenance(root, plan, amendment, payloads, hashes, repository, helper)
    wrapper = payloads["source-projection.json"]
    require(
        set(wrapper) == {"schema", "input_path", "input_sha256", "projection"},
        "Exact four-key projection wrapper",
    )
    content = content_projection(wrapper["projection"], amendment)
    # Preserve the complete envelope in the saved artifact and local payloads;
    # adapt only a new in-memory wrapper for the immutable independent reader.
    numeric_wrapper = {**wrapper, "projection": content}
    z, order = helper.load_projection(numeric_wrapper, plan)
    with np.load(root / "features.npz", allow_pickle=False) as stored:
        require(len(stored.files) == len(set(stored.files)), "Duplicate NPZ members")
        cache = {key: stored[key] for key in stored.files}
    scores = helper.cache_scores(cache, plan)
    q_error = helper.verify_q(cache, scores, order, plan)
    fit_error = helper.verify_fits(cache, scores, z, plan, payloads["fold-freezes.json"])
    rows, diagnostics = helper.replay_evaluation(
        cache, scores, z, plan, payloads["fold-freezes.json"]
    )
    summary, contrasts, attainment = helper.replay_reporting(rows, plan)
    require(
        (len(rows), len(summary), len(contrasts), len(attainment), len(diagnostics))
        == (4680, 120, 153, 1248, 624),
        "Complete unchanged scientific reporting grid",
    )
    result = payloads["result.json"]
    for key, expected in (
        ("rows", rows),
        ("summary", summary),
        ("contrasts", contrasts),
        ("attainment", attainment),
        ("diagnostics", diagnostics),
    ):
        helper.equal(result[key], expected, key)
    helper.equal(
        result["evidence_scope"],
        "adaptively exposed development, descriptive intervals, no automatic promotion",
        "Evidence scope",
    )
    for name, expected in hashes.items():
        helper.equal(helper.digest(root / name), expected, "Artifact stable during audit " + name)
    return {
        "status": "PASS",
        "study_id": STUDY_ID,
        "attempt_id": ATTEMPT_ID,
        "plan_sha256": SCIENTIFIC_PLAN_SHA256,
        "execution_plan_sha256": EXECUTION_PLAN_SHA256,
        "randomization_namespace": amendment["randomization_namespace"],
        "scientific_changes": [],
        "artifact_sha256": hashes,
        "artifact_bytes": size,
        "source_helper_hashes": HELPERS.copy(),
        "projection_envelope_keys": 11,
        "validated_projection_provenance": amendment["envelope_provenance"].copy(),
        "rows": len(rows),
        "summaries": len(summary),
        "contrasts": len(contrasts),
        "attainment": len(attainment),
        "diagnostics": len(diagnostics),
        "independently_replayed_routers": 9,
        "q_cache_derived_max_abs_error": q_error,
        "ridge_coef_max_abs_error": fit_error,
        "scope": "New execution receipts and strict preserved envelope; pinned independent cache/Q/M/ridge/routes/controls/report replay.",
        "runtime_binding_adaptation": "Only a copy of scientific_plan.execution.output_root is changed to the authenticated amendment destination for the original provenance checker.",
        "not_replayed": [
            "raw EEG preprocessing/native fits",
            "raw-derived Q support-quality/covariance/consistency",
            "original projection byte comparison/raw impedance precision",
            "old baseline cache comparison",
            "previous failed attempt contents",
            "runtime access or lifecycle ordering beyond recorded provenance/mtimes",
        ],
        "evidence_role": "adaptively_exposed_development_not_confirmation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.output_root, args.execution_plan)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

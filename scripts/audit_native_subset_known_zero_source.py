"""Independent artifact-only audit for the single known-zero source39 revision.

Imports authenticated independent audit definitions only, never production/core
or IO implementations and never old audit/main/run entrypoints. Fit replay is an
audit calculation: routing always uses the exact saved parent policy freezes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

PLAN_SHA256 = "7b3e478892574d3edb78f7b6659935a2215cf369c619c7bd1da5fd6b68105258"
PLAN_RELATIVE = "configs/analysis/native_subset_known_zero_source39_v1.json"
PLAN_PATH = Path("/home/whwovy/califreeEEG") / PLAN_RELATIVE
AUDIT_DEFINITIONS = (
    "scripts/audit_native_subset_m_source.py",
    "scripts/audit_native_subset_m_envelope_r1.py",
    "scripts/audit_native_subset_known_zero.py",
)
MODES = ("Q", "QM", "SHAM_REFIT", "M_STALE")
SCIENTIFIC_FIELDS = ("rows", "summary", "contrasts", "diagnostics", "attainment", "evidence_scope")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_plan(path):
    encoded = Path(path).read_bytes()
    require(hashlib.sha256(encoded).hexdigest() == PLAN_SHA256, "Frozen execution plan hash")
    return json.loads(encoded)


def load_definition(repository, relative, expected):
    require(
        relative in AUDIT_DEFINITIONS, "Only three independent auditor definitions may be imported"
    )
    path = Path(repository) / relative
    require(
        path.resolve() == path.absolute() and path.is_file() and not path.is_symlink(),
        "Regular non-symlink audit definition required",
    )
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Independent definition hash: " + relative)
    module = ModuleType("known_zero_replay_" + path.stem)
    module.__file__ = str(path)
    # Hash-authenticated local definitions; no optional import bytecode write.
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def rounded_gains(inputs, model):
    standardized = np.clip((inputs - model["feature_mean"]) / model["feature_scale"], -10, 10)
    coefficients = np.asarray(model["coef"])
    return np.round(standardized @ coefficients[1:] + coefficients[0], 10)


def old_choices(gains):
    return np.vstack((np.zeros((1, gains.shape[1])), gains)).argmax(axis=0)


def revision_metrics(predictions, gains, old_before, corrected_before, old_final, corrected_final):
    queries = predictions.shape[1]
    index = np.arange(queries)
    labels = np.tile(np.arange(12), queries // 12)
    require(labels.shape == (queries,), "Revision queries must be complete 12-class blocks")
    aliases = predictions[1:] == predictions[0]
    previous = predictions[old_final, index]
    revised = predictions[corrected_final, index]
    old_full = old_final == 0
    old_alias = ~old_full & (previous == predictions[0])
    return {
        "query_count": queries,
        "old_full_selected": int(old_full.sum()),
        "old_alias_selected": int(old_alias.sum()),
        "old_nonalias_selected": int((~old_full & ~old_alias).sum()),
        "projected_gain_entries_changed": int(np.count_nonzero(aliases & (gains != 0))),
        "prefallback_action_changes": int(np.count_nonzero(old_before != corrected_before)),
        "prefallback_prediction_changes": int(
            np.count_nonzero(predictions[old_before, index] != predictions[corrected_before, index])
        ),
        "action_changes": int(np.count_nonzero(old_final != corrected_final)),
        "prediction_changes": int(np.count_nonzero(previous != revised)),
        "repaired_previous": int(np.count_nonzero((revised == labels) & (previous != labels))),
        "damaged_previous": int(np.count_nonzero((previous == labels) & (revised != labels))),
        "old_ba": float(np.mean(previous == labels)),
        "new_ba": float(np.mean(revised == labels)),
    }


def verify_parent_replay(cache, projectioncontent, freezes, parent_result, science, helper):
    z, order = helper.load_projection(
        {
            "schema": "cfeg.native-subset-m.projection.v1",
            "input_path": science["source_projection"]["path"],
            "input_sha256": science["source_projection"]["sha256"],
            "projection": projectioncontent,
        },
        science,
    )
    scores = helper.cache_scores(cache, science)
    q_error = helper.verify_q(cache, scores, order, science)
    fit_error = helper.verify_fits(cache, scores, z, science, freezes)
    parent_rows, parent_diagnostics = helper.replay_evaluation(cache, scores, z, science, freezes)
    parent_summary, parent_contrasts, parent_attainment = helper.replay_reporting(
        parent_rows, science
    )
    expected = {
        "rows": parent_rows,
        "diagnostics": parent_diagnostics,
        "summary": parent_summary,
        "contrasts": parent_contrasts,
        "attainment": parent_attainment,
        "evidence_scope": "adaptively exposed development, descriptive intervals, no automatic promotion",
    }
    for name, values in expected.items():
        helper.equal(parent_result[name], values, "Unchanged parent " + name)
    return z, scores, q_error, fit_error


def independent_replay(cache, projectioncontent, freezes, parent_result, science, helper, oracle):
    """Verify the entire parent first, then route the unchanged gains by scalar policy."""
    z, scores, _, _ = verify_parent_replay(
        cache, projectioncontent, freezes, parent_result, science, helper
    )
    rows = copy.deepcopy(parent_result["rows"])
    row_lookup = {tuple(row[key] for key in helper.ROW_KEYS): row for row in rows}
    diagnostics, revision = [], []
    parent_diagnostics = {
        tuple(row[key] for key in ("participant", "interface", "n_samples", "k")): row
        for row in parent_result["diagnostics"]
    }
    truth = np.tile(np.arange(12), 5)
    query_indices = np.arange(60)
    for p, participant in enumerate(science["source_subject_ids"]):
        fold = p % 3
        frozen = freezes["folds"][fold]
        for i, interface in enumerate(science["interfaces"]):
            for w, n in enumerate(science["sample_counts"]):
                for bi, k in enumerate(science["budgets"]):
                    key = participant, interface, n, k
                    previous = parent_diagnostics[key]
                    diagnostic = copy.deepcopy(previous)
                    predictions = (
                        scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12).argmax(-1)
                    )
                    gains, fallback = {}, {}
                    for mode in MODES:
                        inputs, missing = helper.cell_inputs(
                            cache, z, science, frozen["metadata_scaler"], p, i, w, bi, mode, fold
                        )
                        model = frozen["routers"]["QM" if mode == "M_STALE" else mode]
                        gains[mode] = rounded_gains(inputs, model)
                        if mode != "Q":
                            fallback[mode] = missing
                    shuffle_gains, shuffle_fallbacks = [], []
                    for intervention in previous["shuffle_interventions"]:
                        inputs, missing = helper.cell_inputs(
                            cache,
                            z,
                            science,
                            frozen["metadata_scaler"],
                            p,
                            i,
                            w,
                            bi,
                            "QM",
                            fold,
                            tuple(intervention["mapping"]),
                        )
                        shuffle_gains.append(rounded_gains(inputs, frozen["routers"]["QM"]))
                        shuffle_fallbacks.append(missing)
                    corrected = oracle.route_family_scalar(
                        predictions, gains, fallback, shuffle_gains, shuffle_fallbacks
                    )
                    actions, shuffled_actions = corrected["actions"], corrected["shuffle_actions"]
                    decisions = {
                        mode: predictions[action, query_indices] for mode, action in actions.items()
                    }
                    method_metrics = {}
                    for mode in (*MODES, "M_MISSING"):
                        original_mode = "Q" if mode == "M_MISSING" else mode
                        old_final = np.asarray(previous["actions"][original_mode], dtype=int)
                        old_before = old_choices(gains[original_mode])
                        corrected_before = oracle.choose_known_zero_scalar(
                            gains[original_mode], predictions
                        )
                        method_metrics[mode] = revision_metrics(
                            predictions,
                            gains[original_mode],
                            old_before,
                            corrected_before,
                            old_final,
                            actions[mode],
                        )
                        row_lookup[participant, interface, n, mode, k]["ba"] = float(
                            np.mean(decisions[mode] == truth)
                        )
                    qright, mright = decisions["Q"] == truth, decisions["QM"] == truth
                    diagnostic["qm_vs_q_selector_changes"] = int(
                        np.count_nonzero(actions["QM"] != actions["Q"])
                    )
                    diagnostic["qm_vs_q_prediction_changes"] = int(
                        np.count_nonzero(decisions["QM"] != decisions["Q"])
                    )
                    diagnostic["qm_repaired_q"] = int(np.count_nonzero(mright & ~qright))
                    diagnostic["qm_damaged_q"] = int(np.count_nonzero(qright & ~mright))
                    diagnostic["actions"] = {mode: actions[mode].tolist() for mode in MODES}
                    shuffle_metrics = []
                    for index, intervention in enumerate(diagnostic["shuffle_interventions"]):
                        action = shuffled_actions[index]
                        prediction = predictions[action, query_indices]
                        old_final = np.asarray(
                            previous["shuffle_interventions"][index]["actions"], dtype=int
                        )
                        old_before = old_choices(shuffle_gains[index])
                        corrected_before = oracle.choose_known_zero_scalar(
                            shuffle_gains[index], predictions
                        )
                        shuffle_metrics.append(
                            {
                                "mapping": intervention["mapping"].copy(),
                                **revision_metrics(
                                    predictions,
                                    shuffle_gains[index],
                                    old_before,
                                    corrected_before,
                                    old_final,
                                    action,
                                ),
                            }
                        )
                        intervention.update(
                            {
                                "selector_changes": int(np.count_nonzero(action != actions["QM"])),
                                "prediction_changes": int(
                                    np.count_nonzero(prediction != decisions["QM"])
                                ),
                                "ba": float(np.mean(prediction == truth)),
                                "actions": action.tolist(),
                            }
                        )
                    diagnostic["mean_shuffle_selector_changes"] = float(
                        np.mean(
                            [
                                item["selector_changes"]
                                for item in diagnostic["shuffle_interventions"]
                            ]
                        )
                    )
                    diagnostic["mean_shuffle_prediction_changes"] = float(
                        np.mean(
                            [
                                item["prediction_changes"]
                                for item in diagnostic["shuffle_interventions"]
                            ]
                        )
                    )
                    row_lookup[participant, interface, n, "M_SHUFFLE", k]["ba"] = float(
                        np.mean([item["ba"] for item in diagnostic["shuffle_interventions"]])
                    )
                    diagnostics.append(diagnostic)
                    revision.append(
                        {
                            "participant": participant,
                            "interface": interface,
                            "n_samples": n,
                            "k": k,
                            "methods": method_metrics,
                            "shuffle": shuffle_metrics,
                        }
                    )
    summary, contrasts, attainment = helper.replay_reporting(rows, science)
    return {
        "rows": rows,
        "summary": summary,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
        "attainment": attainment,
        "revision_diagnostics": revision,
        "evidence_scope": parent_result["evidence_scope"],
        "parent_replay_verified": True,
    }


def read_bundle(root, names, budget, expected_hashes=None):
    """Read only the exact regular-file inventory, with a stable descriptor per file."""
    root = Path(root)
    require(
        root.is_dir() and root.resolve() == root and not root.is_symlink(),
        "Exact regular bundle root",
    )
    require({path.name for path in root.iterdir()} == set(names), "Exact artifact inventory")
    raw, hashes, mtimes, size = {}, {}, [], 0
    for name in names:
        path = root / name
        before = path.lstat()
        require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o400
            and before.st_nlink == 1,
            "Immutable single-link regular artifact: " + name,
        )
        size += before.st_size
        require(size <= budget, "Artifact bundle byte budget")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            current = os.fstat(stream.fileno())
            require(
                (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                "Artifact identity changed before reading",
            )
            data = stream.read()
            after = os.fstat(stream.fileno())
            require(
                (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns),
                "Artifact changed during reading",
            )
        raw[name] = data
        hashes[name] = hashlib.sha256(data).hexdigest()
        if expected_hashes is not None:
            require(hashes[name] == expected_hashes[name], "Pinned parent artifact hash: " + name)
        mtimes.append(before.st_mtime_ns)
    require(mtimes == sorted(mtimes), "Artifact publication order")
    return raw, hashes, size


def configuration_hashes(plan):
    return {
        PLAN_RELATIVE: PLAN_SHA256,
        plan["mechanism"]["path"]: plan["mechanism"]["sha256"],
        plan["science"]["path"]: plan["science"]["sha256"],
        plan["parent"]["execution_plan_path"]: plan["parent"]["execution_plan_sha256"],
    }


def load_configurations(repository, plan):
    configs = {}
    for relative, expected in configuration_hashes(plan).items():
        encoded = (Path(repository) / relative).read_bytes()
        require(
            hashlib.sha256(encoded).hexdigest() == expected,
            "Configuration byte binding: " + relative,
        )
        configs[relative] = json.loads(encoded)
    return configs


def git_bytes(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args])


def verify_new_provenance(root, plan, payloads, hashes, repository, helper):
    start, result = payloads["start.json"], payloads["result.json"]
    require(set(start) == {"schema", *plan["common_fields"]}, "Exact new start receipt fields")
    require(
        set(result) == {"schema", *plan["common_fields"], *plan["result_extra_fields"]},
        "Exact revised result fields",
    )
    helper.equal(start["schema"], plan["schemas"]["start"], "New start schema")
    helper.equal(result["schema"], plan["schemas"]["result"], "New result schema")
    for field in plan["common_fields"]:
        helper.equal(result[field], start[field], "Common receipt " + field)
    expected = {
        "attempt_id": plan["attempt_id"],
        "study_id": plan["study_id"],
        "randomization_namespace": plan["randomization_namespace"],
        "execution_plan_sha256": PLAN_SHA256,
        "mechanism_plan_sha256": plan["mechanism"]["sha256"],
        "scientific_plan_sha256": plan["science"]["sha256"],
        "parent_root": plan["parent"]["root"],
        "parent_artifact_sha256": plan["parent"]["artifacts"],
        "runtime": plan["runtime"],
        "output_root": plan["output_root"],
        "policy_refit": False,
        "held_access": False,
        "retired_access": False,
        "raw_eeg_access": False,
    }
    require(str(root) == plan["output_root"], "Frozen revised output destination")
    for field, value in expected.items():
        helper.equal(start[field], value, "New authority " + field)
    helper.equal(result["start_sha256"], hashes["start.json"], "New start byte binding")
    helper.equal(result["status"], "DEVELOPMENT_ASSESSMENT_COMPLETE", "Revised completion status")
    helper.equal(result["parent_replay_verified"], True, "Parent replay prerequisite receipt")
    for field in ("source_commit", "source_tree"):
        require(re.fullmatch(r"[0-9a-f]{40}", start[field]) is not None, "New git receipt format")
    tree = git_bytes(repository, "rev-parse", start["source_commit"] + "^{tree}").decode().strip()
    helper.equal(start["source_tree"], tree, "New committed source tree")
    pinned = {**plan["pinned_helpers"], **configuration_hashes(plan)}
    require(
        set(start["source_hashes"]) == set(pinned) | set(plan["code_paths"]),
        "Exact source hash inventory",
    )
    for relative, declared in start["source_hashes"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", declared) is not None, "Source digest format")
        if relative in pinned:
            helper.equal(declared, pinned[relative], "Frozen dependency " + relative)
        blob = git_bytes(repository, "show", start["source_commit"] + ":" + relative)
        helper.equal(hashlib.sha256(blob).hexdigest(), declared, "Committed source " + relative)
        local = Path(repository) / relative
        require(
            local.resolve() == local.absolute() and local.is_file() and not local.is_symlink(),
            "Non-symlink local execution source",
        )
        helper.equal(
            hashlib.sha256(local.read_bytes()).hexdigest(),
            declared,
            "Local audit/source consistency " + relative,
        )
    before, after = (
        datetime.fromisoformat(start["started_at"]),
        datetime.fromisoformat(result["completed_at"]),
    )
    require(
        before.tzinfo is not None
        and after.tzinfo is not None
        and after >= before
        and (after - before).total_seconds() <= plan["runtime"]["ceiling_seconds"],
        "New execution chronology/budget",
    )


def verify_parent_provenance(
    parent_payloads, parent_hashes, plan, science, amendment, repository, helper, envelope_helper
):
    for name in ("start.json", "fold-freezes.json", "result.json"):
        item = parent_payloads[name]
        for field in ("attempt_id", "study_id", "source_commit", "source_tree"):
            helper.equal(item[field], plan["parent"][field], "Pinned parent " + name + " " + field)
    envelope_helper.verify_execution_provenance(
        Path(plan["parent"]["root"]),
        science,
        amendment,
        parent_payloads,
        parent_hashes,
        repository,
        helper,
    )
    wrapper = parent_payloads["source-projection.json"]
    require(
        set(wrapper) == {"schema", "input_path", "input_sha256", "projection"},
        "Preserved parent wrapper",
    )
    content = envelope_helper.content_projection(wrapper["projection"], amendment)
    helper.load_projection({**wrapper, "projection": content}, science)
    return content


def audit(output_root, execution_plan_path=PLAN_PATH, repository=None):
    root = Path(output_root)
    repository = Path(repository) if repository is not None else Path(__file__).resolve().parents[1]
    plan = load_plan(execution_plan_path)
    require(str(root) == plan["output_root"], "Exact revised artifact root")
    configs = load_configurations(repository, plan)
    science = configs[plan["science"]["path"]]
    amendment = configs[plan["parent"]["execution_plan_path"]]
    helper, envelope_helper, oracle = [
        load_definition(repository, relative, plan["pinned_helpers"][relative])
        for relative in AUDIT_DEFINITIONS
    ]
    new_raw, new_hashes, new_size = read_bundle(
        root, plan["artifacts"], plan["runtime"]["output_budget_bytes"]
    )
    payloads = {name: json.loads(raw) for name, raw in new_raw.items()}
    verify_new_provenance(root, plan, payloads, new_hashes, repository, helper)
    parent_root = Path(plan["parent"]["root"])
    parent_raw, parent_hashes, parent_size = read_bundle(
        parent_root,
        list(plan["parent"]["artifacts"]),
        plan["runtime"]["parent_budget_bytes"],
        plan["parent"]["artifacts"],
    )
    require(parent_size == plan["parent"]["total_bytes"], "Pinned parent total bytes")
    parent_payloads = {
        name: json.loads(raw) for name, raw in parent_raw.items() if name.endswith(".json")
    }
    content = verify_parent_provenance(
        parent_payloads,
        parent_hashes,
        plan,
        science,
        amendment,
        repository,
        helper,
        envelope_helper,
    )
    parent_finished = datetime.fromisoformat(parent_payloads["result.json"]["completed_at"])
    new_started = datetime.fromisoformat(payloads["start.json"]["started_at"])
    require(
        parent_finished.tzinfo is not None and parent_finished <= new_started,
        "Parent completion must precede new attempt",
    )
    with np.load(io.BytesIO(parent_raw["features.npz"]), allow_pickle=False) as stored:
        require(len(stored.files) == len(set(stored.files)), "Duplicate cache array members")
        cache = {key: stored[key] for key in stored.files}
    reconstructed = independent_replay(
        cache,
        content,
        parent_payloads["fold-freezes.json"],
        parent_payloads["result.json"],
        science,
        helper,
        oracle,
    )
    for key, expected in reconstructed.items():
        helper.equal(payloads["result.json"][key], expected, "Revised " + key)
    for key, expected_count in plan["evaluation"].items():
        if key in ("rows", "summary", "contrasts", "diagnostics", "attainment"):
            require(len(reconstructed[key]) == expected_count, "Complete revised " + key)
    require(
        len(reconstructed["revision_diagnostics"]) == plan["revision_diagnostics"]["rows"],
        "Complete revision diagnostics",
    )
    for bundle, hashes in ((root, new_hashes), (parent_root, parent_hashes)):
        for name, expected in hashes.items():
            helper.equal(helper.digest(bundle / name), expected, "Stable audit input " + name)
    return {
        "status": "PASS",
        "attempt_id": plan["attempt_id"],
        "execution_plan_sha256": PLAN_SHA256,
        "mechanism_plan_sha256": plan["mechanism"]["sha256"],
        "scientific_plan_sha256": plan["science"]["sha256"],
        "artifact_sha256": new_hashes,
        "artifact_bytes": new_size,
        "parent_artifact_sha256": parent_hashes,
        "parent_artifact_bytes": parent_size,
        "parent_replay_verified": True,
        "independently_replayed_parent_routers": 9,
        "rows": len(reconstructed["rows"]),
        "summaries": len(reconstructed["summary"]),
        "contrasts": len(reconstructed["contrasts"]),
        "diagnostics": len(reconstructed["diagnostics"]),
        "attainment": len(reconstructed["attainment"]),
        "revision_diagnostics": len(reconstructed["revision_diagnostics"]),
        "policy_refit": False,
        "held_access": False,
        "retired_access": False,
        "raw_eeg_access": False,
        "scope": "Pinned parent provenance/5 artifacts, independent cache/Q/M/scaler/9 ridge and parent report replay; scalar corrected policy and all revised/control reports.",
        "not_replayed": [
            "raw EEG/native fits",
            "raw-derived Q support/covariance/consistency",
            "original projection byte/raw impedance precision extraction",
            "parent old-baseline cache comparison",
            "runtime access beyond recorded receipts",
        ],
        "evidence_role": "repeatedly_exposed_development_not_independent_confirmation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, default=PLAN_PATH)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                audit(args.output_root, args.execution_plan), sort_keys=True, allow_nan=False
            )
        )
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

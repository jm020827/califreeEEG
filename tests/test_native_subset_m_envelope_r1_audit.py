"""Synthetic-only tests of new envelope transport and execution audit bindings."""

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "subset_envelope_audit", ROOT / "scripts/audit_native_subset_m_envelope_r1.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def amendment():
    return AUDIT.load_amendment(ROOT / AUDIT.AMENDMENT_RELATIVE)


@pytest.fixture
def helper(amendment):
    return AUDIT.load_helper(ROOT, amendment)


@pytest.fixture
def science(helper, amendment):
    return helper.load_plan(amendment["scientific_plan"]["path"])


def envelope_fixture(amendment, science):
    packets = []
    for p, participant in enumerate(science["source_subject_ids"]):
        first = ("dry", "wet")[p % 2]
        for interface in ("dry", "wet"):
            for block in range(10):
                packets.append(
                    {
                        "subject_id": participant,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [0.0, None, 2.0, 3.0, 4.0, 5.0, 6.0, float(block)],
                        "headband_order": first,
                        "condition_period": "first" if interface == first else "second",
                    }
                )
    return {
        **amendment["envelope_provenance"],
        "manifest_sha256": science["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": science["source_subject_ids"].copy(),
        "columns": science["source_projection"]["columns"].copy(),
    }


def wrapper_fixture(envelope, science):
    return {
        "schema": "cfeg.native-subset-m.projection.v1",
        "input_path": science["source_projection"]["path"],
        "input_sha256": science["source_projection"]["sha256"],
        "projection": envelope,
    }


def receipts_fixture(amendment, science):
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    common = {
        "attempt_id": amendment["attempt_id"],
        "execution_plan_sha256": AUDIT.EXECUTION_PLAN_SHA256,
        "plan_sha256": AUDIT.SCIENTIFIC_PLAN_SHA256,
        "study_id": science["study_id"],
        "source_commit": commit,
        "source_tree": tree,
        "upstream_revision": science["upstream"]["revision"],
        "started_at": "2026-09-07T12:00:00+00:00",
    }
    hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in science["execution"]["artifacts"]
    }
    start = {
        **common,
        "schema": "cfeg.native-subset-m.start.v1",
        "core_sha256": amendment["source_helpers"]["scripts/native_subset_m_core.py"],
        "source_helper_hashes": amendment["source_helpers"].copy(),
        "imported_source_hashes": science["upstream"]["pins"].copy(),
        "source_subject_ids": science["source_subject_ids"].copy(),
        "raw_root": science["raw_root"],
        "output_root": amendment["output_root"],
        "python": science["execution"]["python_version"],
        "dependencies": science["execution"]["dependencies"].copy(),
        "metadata_access": True,
        "source_projection_reuse": True,
        "baseline_cache_read": True,
        "held_access": False,
        "retired_access": False,
        "manifest_or_full_impedance_access": False,
    }
    freeze = {
        **common,
        "source_projection_sha256": hashes["source-projection.json"],
        "features_sha256": hashes["features.npz"],
    }
    result = {
        **freeze,
        "schema": science["reporting"]["schema"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "completed_at": "2026-09-07T12:00:01+00:00",
        "fold_freezes_sha256": hashes["fold-freezes.json"],
        "raw_files": [
            {
                "subject": subject,
                "path": f"{science['raw_root']}/S{subject:03d}.mat",
                "sha256": "c" * 64,
                "stored_dtype": "float64",
                "shape": science["raw_shape"].copy(),
            }
            for subject in science["source_subject_ids"]
        ],
        "baseline_agreement": {
            "input_path": science["baseline_reference"]["path"],
            "input_sha256": science["baseline_reference"]["sha256"],
            "a0_r": {"max_abs_error": 0.0, "argmax_exact": True},
            "full_expert_r": {"max_abs_error": 0.0, "argmax_exact": True},
        },
    }
    return {"start.json": start, "fold-freezes.json": freeze, "result.json": result}, hashes


def test_amendment_authenticated_bytes_and_exact_no_science_delta(amendment, tmp_path):
    assert amendment["scientific_changes"] == []
    assert amendment["randomization_namespace"] == "native-subset-m-source39-v1"
    damaged = tmp_path / "amendment.json"
    damaged.write_bytes((ROOT / AUDIT.AMENDMENT_RELATIVE).read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte hash"):
        AUDIT.load_amendment(damaged)


@pytest.mark.parametrize(
    "damage",
    [
        "namespace",
        "study",
        "scientific_delta",
        "science_hash",
        "helper_hash",
        "content_key",
        "held",
        "retry",
        "same_destination",
    ],
)
def test_amendment_scientific_and_authority_corruption(amendment, damage):
    if damage == "namespace":
        amendment["randomization_namespace"] = AUDIT.ATTEMPT_ID
    elif damage == "study":
        amendment["study_id"] = AUDIT.ATTEMPT_ID
    elif damage == "scientific_delta":
        amendment["scientific_changes"] = ["alter ridge"]
    elif damage == "science_hash":
        amendment["scientific_plan"]["sha256"] = "0" * 64
    elif damage == "helper_hash":
        amendment["source_helpers"][AUDIT.AUDIT_HELPER] = "0" * 64
    elif damage == "content_key":
        amendment["content_keys"].append("schema")
    elif damage == "same_destination":
        amendment["output_root"] = amendment["previous_attempt"]["path"]
    else:
        amendment["execution"]["held_access" if damage == "held" else "retry_same_attempt"] = True
    with pytest.raises(ValueError):
        AUDIT.check_amendment(amendment)


def test_exact_eleven_to_six_connects_to_frozen_independent_reader(
    amendment, science, helper, tmp_path
):
    original = envelope_fixture(amendment, science)
    path = tmp_path / "artificial-original-envelope.json"
    path.write_text(json.dumps(original, allow_nan=False))
    loaded = json.loads(path.read_bytes())
    preserved = copy.deepcopy(loaded)
    content = AUDIT.content_projection(loaded, amendment)
    assert list(content) == list(AUDIT.CONTENT_KEYS) and len(loaded) == 11
    z, order = helper.load_projection(wrapper_fixture(content, science), science)
    assert z.shape == (39, 2, 10, 8)
    assert np.isnan(z[..., 1]).all() and np.all(z[..., 0] == 0)
    np.testing.assert_array_equal(order, np.arange(39) % 2)
    assert loaded == preserved
    content["packets"][0]["impedance_kohm"][0] = 999.0
    assert loaded == preserved
    with pytest.raises(ValueError, match="Projection fields"):
        helper.load_projection(wrapper_fixture(loaded, science), science)


@pytest.mark.parametrize(
    "damage",
    [
        "six_only",
        "extra",
        "missing",
        "schema",
        "study_id",
        "plan_sha256",
        "source_commit",
        "start_sha256",
    ],
)
def test_envelope_unknown_missing_and_each_provenance_pin(amendment, science, damage):
    envelope = envelope_fixture(amendment, science)
    if damage == "six_only":
        envelope = {key: envelope[key] for key in AUDIT.CONTENT_KEYS}
    elif damage == "extra":
        envelope["unexpected"] = True
    elif damage == "missing":
        del envelope["returned_rows"]
    else:
        envelope[damage] = "wrong"
    with pytest.raises(ValueError):
        AUDIT.content_projection(envelope, amendment)


@pytest.mark.parametrize("damage", ["packet_key", "participant", "negative", "order"])
def test_extracted_content_still_under_exact_independent_packet_contract(
    amendment, science, helper, damage
):
    envelope = envelope_fixture(amendment, science)
    packet = envelope["packets"][0]
    if damage == "packet_key":
        packet["future_label"] = 3
    elif damage == "participant":
        packet["subject_id"] = 1
    elif damage == "negative":
        packet["impedance_kohm"][0] = -1
    else:
        packet["condition_period"] = "second"
    extracted = AUDIT.content_projection(envelope, amendment)
    with pytest.raises(ValueError):
        helper.load_projection(wrapper_fixture(extracted, science), science)


def test_helper_authentication_before_definition_execution(amendment, tmp_path):
    path = tmp_path / AUDIT.AUDIT_HELPER
    path.parent.mkdir()
    path.write_text("raise RuntimeError('must never execute unauthenticated code')")
    with pytest.raises(ValueError, match="helper byte hash"):
        AUDIT.load_helper(tmp_path, amendment)


def test_unchanged_helper_math_and_namespace_after_transport(amendment, science, helper):
    envelope = envelope_fixture(amendment, science)
    content = AUDIT.content_projection(envelope, amendment)
    z, _ = helper.load_projection(wrapper_fixture(content, science), science)
    globals_before = {key: id(value) for key, value in vars(helper).items()}
    scaler = helper.impedance_scaler(z, list(range(1, 39)))
    prefix, query = z[0, 0, :3], z[0, 0, 5:10]
    q = np.zeros((3, 60, 33))
    values, missing = helper.m_features(prefix, query, q, scaler, 0)
    assert values.shape == (3, 60, 36) and not missing.any()
    mapping = helper.select_sham(prefix, science["study_id"], 0, 4, "dry")
    family = helper.permutation_family(prefix)
    expected = family[
        int(hashlib.sha256(b"native-subset-m-source39-v1|0|4|dry|3").hexdigest(), 16) % len(family)
    ]
    assert mapping == expected
    model = helper.ridge_replay(np.zeros((12, 69)), np.zeros(12), np.full(12, 1 / 12))
    np.testing.assert_array_equal(helper.route(np.zeros((3, 60, 69)), model), np.zeros(60, int))
    assert vars(helper)["PLAN_SHA256"] == AUDIT.SCIENTIFIC_PLAN_SHA256
    assert globals_before == {key: id(value) for key, value in vars(helper).items()}


def test_new_execution_provenance_uses_real_git_objects_without_plan_mutation(
    amendment, science, helper
):
    payloads, hashes = receipts_fixture(amendment, science)
    untouched = copy.deepcopy(science)
    AUDIT.verify_execution_provenance(
        Path(amendment["output_root"]), science, amendment, payloads, hashes, ROOT, helper
    )
    assert science == untouched
    assert science["execution"]["output_root"] != amendment["output_root"]


@pytest.mark.parametrize(
    "damage",
    [
        "attempt",
        "amendment_hash",
        "namespace",
        "source_hash",
        "helper_hash",
        "core_hash",
        "cache_binding",
        "freeze_binding",
        "destination",
    ],
)
def test_execution_receipt_hash_namespace_helper_and_destination_sensitivity(
    amendment, science, helper, damage
):
    payloads, hashes = receipts_fixture(amendment, science)
    start, freeze, result = (
        payloads[name] for name in ("start.json", "fold-freezes.json", "result.json")
    )
    root = Path(amendment["output_root"])
    if damage == "attempt":
        freeze["attempt_id"] = AUDIT.STUDY_ID
    elif damage == "amendment_hash":
        result["execution_plan_sha256"] = "0" * 64
    elif damage == "namespace":
        for receipt in (start, freeze, result):
            receipt["study_id"] = AUDIT.ATTEMPT_ID
    elif damage == "source_hash":
        result["plan_sha256"] = AUDIT.EXECUTION_PLAN_SHA256
    elif damage == "helper_hash":
        start["source_helper_hashes"][AUDIT.AUDIT_HELPER] = "0" * 64
    elif damage == "core_hash":
        start["core_sha256"] = "0" * 64
    elif damage == "cache_binding":
        result["features_sha256"] = "0" * 64
    elif damage == "freeze_binding":
        result["fold_freezes_sha256"] = "0" * 64
    else:
        root = Path("/synthetic/copied-artifacts")
        start["output_root"] = str(root)
    with pytest.raises(ValueError):
        AUDIT.verify_execution_provenance(root, science, amendment, payloads, hashes, ROOT, helper)


def test_old_destination_rejected_before_artifact_reads(amendment):
    with pytest.raises(ValueError, match="Frozen amendment output root"):
        AUDIT.audit(amendment["previous_attempt"]["path"], ROOT / AUDIT.AMENDMENT_RELATIVE)


def test_new_wrapper_never_calls_old_entrypoints_or_mutates_helper_globals():
    tree = ast.parse(Path(AUDIT.__file__).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "helper"
        ):
            assert node.func.attr not in ("audit", "main", "run")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                assert not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "helper"
                )
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("native_subset_m_core", "run_native_subset_m_source")

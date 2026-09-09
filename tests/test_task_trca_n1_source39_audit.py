"""New study identity, unchanged endpoint branches and human descriptive coverage."""

import ast
import inspect
import json

import numpy as np
import pytest
from test_task_trca_n1_audit import coverage_models, evaluation_fixture

from cfeg.analysis import task_trca_n1_audit as n1
from cfeg.analysis import task_trca_n1_source39_artifact_audit as artifact
from cfeg.analysis import task_trca_n1_source39_audit as audit
from cfeg.analysis import task_trca_temporal_audit as old_endpoint


@pytest.fixture(autouse=True)
def only_toy_seed(monkeypatch):
    original = np.random.default_rng

    def guarded(seed=None):
        assert seed == 73, "Only nonregistered toy seed73"
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)


def endpoint_fixture(*, native3=40, centered3=40, q3=38, qm3=40, q5=40):
    scores = np.zeros((39, 2, 4, 2, 10, 48, 12))
    a0 = np.zeros((39, 2, 4, 48, 12))
    truth = np.tile(np.arange(12), 4)
    for position, arm in enumerate(audit.ARMS):
        for budget in range(2):
            count = (
                (
                    qm3
                    if arm == "QM"
                    else native3
                    if arm == "FULL_NATIVE"
                    else centered3
                    if arm == "FULL_CENTERED"
                    else q3
                )
                if budget == 0
                else q5
            )
            prediction = np.where(np.arange(48) < count, truth, (truth + 1) % 12)
            scores[..., budget, position, np.arange(48), prediction] = 1
    a0[..., np.arange(48), (truth + 1) % 12] = 1
    coverage = {
        "m": np.zeros((39, 2, 8, 2)),
        "donor_m": np.ones((39, 2, 8, 2)),
        "available": np.ones((39, 2, 8), bool),
    }
    actuation = {
        "band_weights": np.ones((2, 5)),
        "a0_band_weights": np.ones((2, 5)),
        "qm_coefficients": np.ones((3, 3)),
        **{
            field: np.full((39, 2, 4, 2), 0.01)
            for field in (
                "r_max_abs_qm_minus_q",
                "projector_max_abs_qm_minus_q",
                "j_max_abs_qm_minus_q",
            )
        },
    }
    return scores, a0, coverage, actuation


def test_ten_arm_positive_endpoint_and_original_costs_preserved():
    result = audit.summarize(*endpoint_fixture())
    assert result["terminal"] == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    assert result["arms"] == list(audit.ARMS)
    assert np.asarray(result["correct_counts"]).shape == (39, 2, 4, 2, 10)
    assert np.asarray(result["nll_by_budget_arm"]).shape == (2, 10)
    assert np.asarray(result["conditions"][0]["accuracy"]).shape == (2, 10)
    assert result["grid_attainment"]["pooled_label_savings_mean"] == 24
    assert result["comparisons"]["QM3_minus_FULL_NATIVE3"]["mean_pp"] == 0
    assert result["comparisons"]["QM3_minus_FULL_CENTERED3"]["mean_pp"] == 0
    assert result["comparisons"]["FULL_CENTERED3_minus_FULL_NATIVE3"]["mean_pp"] == 0
    assert "projector_max_abs_qm_minus_q" in result["actuation"]["by_k"]["3"]
    assert "j_max_abs_qm_minus_q" in result["actuation"]["by_k"]["3"]
    assert "filter_max_abs_qm_minus_q" not in result["actuation"]["by_k"]["3"]
    json.dumps(result, allow_nan=False)


def test_added_centered_guard_only_demotes_in_correct_direction():
    result = audit.summarize(*endpoint_fixture(centered3=41))
    assert result["terminal_before_centered_guard"] == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    assert result["terminal"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert result["comparisons"]["QM3_minus_FULL_CENTERED3"]["ci_low_pp"] < -1
    assert not result["calibration_additional_checks"][
        "qm3_minus_full_centered3_ci_low_gt_minus_1pp"
    ]


def test_strong_native_guard_cannot_be_replaced_by_easy_centered_control():
    result = audit.summarize(*endpoint_fixture(native3=42, centered3=0))
    assert result["terminal_before_centered_guard"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert result["terminal"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert not result["calibration_additional_checks"]["qm3_minus_full_native3_ci_low_gt_minus_1pp"]
    assert result["calibration_additional_checks"]["qm3_minus_full_centered3_ci_low_gt_minus_1pp"]


@pytest.mark.parametrize(
    "terminal",
    ["METADATA_INCREMENT_NOT_ESTABLISHED", "STRUCTURAL_NO_ACTUATION", "VALIDITY_FAILURE"],
)
def test_centered_control_never_promotes_old_failure_branches(terminal):
    scores, a0, coverage, actuation = endpoint_fixture(centered3=0, qm3=38)
    if terminal == "STRUCTURAL_NO_ACTUATION":
        actuation["qm_coefficients"][:] = 0
    elif terminal == "VALIDITY_FAILURE":
        actuation["validity_errors"] = ["artificial saved-artifact mismatch"]
    result = audit.summarize(scores, a0, coverage, actuation)
    assert result["terminal"] == terminal
    assert result["terminal_before_centered_guard"] == terminal


def test_nonattainment_costs_remain_null_and_argmax_static_not_structural():
    scores, a0, coverage, actuation = endpoint_fixture(q3=20, qm3=20, q5=20, centered3=20)
    scores[..., 5, :, :] += 0.0001
    actuation["j_max_abs_qm_minus_q"][:] = 0
    result = audit.summarize(scores, a0, coverage, actuation)
    assert result["grid_attainment"]["pooled_label_savings_mean"] is None
    assert result["grid_attainment"]["q_labels"][0][0][0] is None
    assert result["terminal"] == "METADATA_INCREMENT_NOT_ESTABLISHED"
    assert result["actuation"]["by_k"]["3"]["argmax_changed_queries"] == 0


@pytest.mark.parametrize("defect", ["old_shape", "missing_diff", "projector", "j", "native_nan"])
def test_summary_rejects_mixed_or_bad_evidence(defect):
    scores, a0, coverage, actuation = endpoint_fixture()
    if defect == "old_shape":
        scores = scores[..., 1:, :, :]
    elif defect == "missing_diff":
        scores[..., 9, :, :] += 1e-15
    elif defect == "projector":
        actuation["projector_max_abs_qm_minus_q"].flat[0] = -1
    elif defect == "j":
        actuation["j_max_abs_qm_minus_q"].flat[0] = np.nan
    else:
        scores[..., 0, :, :] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        audit.summarize(scores, a0, coverage, actuation)


def test_transparent_math_and_endpoint_reuse():
    assert artifact.independent is n1
    assert audit.SCHEMA == n1.SCHEMA == "task-trca-n1-integration-v1"
    args = endpoint_fixture()
    old = old_endpoint.summarize(*args)
    new = audit.summarize(*args)
    assert new["study_id"] == "task-trca-n1-source39-v1"
    assert new["algorithm_schema"] == n1.SCHEMA
    assert new["schema"] == audit.SUMMARY_SCHEMA != old["schema"]
    for name in old.keys() - {"schema"}:
        assert new[name] == old[name]


@pytest.mark.parametrize(
    "profile,count,n,seed",
    [
        ("human", 39, [125, 188, 250, 500], None),
        ("generated", 9, [17], 20260914),
    ],
)
def test_exact_profiles_and_grid(profile, count, n, seed):
    contract = artifact.profile_record(profile)
    assert len(contract["source_ids"]) == count
    assert contract["samples"] == n and contract["seed"] == seed
    assert contract["generated"] is (profile == "generated")
    keys = np.array(
        [
            [pid, i, window, k]
            for pid in contract["source_ids"]
            for i in (0, 1)
            for window in n
            for k in (3, 5)
        ]
    )
    artifact._grid({"keys": keys}, tuple(contract["source_ids"]), profile)
    assert len(keys) == (624 if profile == "human" else 36)
    with pytest.raises(ValueError):
        artifact._grid({"keys": keys[::-1]}, tuple(contract["source_ids"]), profile)
    with pytest.raises(ValueError):
        artifact.profile_record("auto")


def test_all_ten_arms_keep_independent_n1_math_and_generated_grid():
    data, pipeline, ids = evaluation_fixture()
    receipt = artifact.audit_evaluation(data, pipeline, ids, profile="generated")
    assert receipt["status"] == "TEMPORAL_EVALUATION_AUDIT_PASS"
    with pytest.raises(ValueError):
        artifact.audit_evaluation(data, pipeline, ids, profile="human")


@pytest.mark.parametrize("family", ["Q", "Q2", "QM", "SHAM_REFIT"])
def test_human_nonactuation_is_descriptive_not_generated_pass(family):
    models = coverage_models()
    head = (
        models[-1]["pipeline"]["Q"]
        if family == "Q"
        else models[-1]["pipeline"]["residuals"][family]
    )
    head["coefficients"][0] = 0.0
    head["trace"][0]["gradient_norm"] = 0.0
    result = audit.learning_path_coverage(models, profile="human")
    assert result["descriptive_only"] and not result["exercised"]
    assert not result["families"][family]["exercised"]
    assert result["registered_updates"] == 24000
    with pytest.raises(ValueError, match="PATH_NOT_EXERCISED"):
        audit.learning_path_coverage(models, profile="generated")


def test_human_descriptive_coverage_still_rejects_invalid_trace():
    models = coverage_models()
    models[0]["pipeline"]["Q"]["trace"][0]["gradient_norm"] = float("nan")
    with pytest.raises(ValueError):
        audit.learning_path_coverage(models, profile="human")


def test_new_auditors_import_only_independent_modules():
    allowed = {"task_trca_n1_audit", "task_trca_shape_audit", "task_trca_temporal_audit"}
    for module in (audit, artifact):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.ImportFrom) and node.module == "cfeg.analysis":
                assert {alias.name for alias in node.names} <= allowed

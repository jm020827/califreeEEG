from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    MetadataCalibrationBundleSpec,
    evaluate_development_gate,
    evaluate_held_claim_sequence,
    metadata_calibration_bundle_spec_for_phase,
    one_sided_paired_t,
    reduce_metadata_calibration_predictions,
    validate_metadata_calibration_private_staging,
)
from cfeg.metadata_calibration_execution import cell_execution_contract_sha256


def _digest(values) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(value) for value in values):
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _spec() -> MetadataCalibrationBundleSpec:
    return MetadataCalibrationBundleSpec(
        n_classes=3,
        seeds=(1, 2),
        sensitivity_resamples=200,
    )


def test_production_bundle_spec_is_phase_derived_and_cannot_omit_diagnostics() -> None:
    held = metadata_calibration_bundle_spec_for_phase(
        phase="held_participant_evaluation", sensitivity_resamples=200
    )
    development = metadata_calibration_bundle_spec_for_phase(
        phase="source_development", sensitivity_resamples=200
    )
    assert held.checkpoint_groups == ("held",)
    assert len(held.contexts) == 8
    assert development.checkpoint_groups == ("fold0", "fold1", "fold2")
    assert len(development.contexts) == 12
    assert ("A_QM", "support_metadata_only") in development.contexts


def _query(n_subjects: int = 2) -> pd.DataFrame:
    rows = []
    for subject in range(n_subjects):
        for condition in ("dry", "wet"):
            for block in range(2):
                for label in range(3):
                    raw_identity = f"s{subject}_{condition}_b{block}_y{label}"
                    rows.append(
                        {
                            "query_token": "q_" + hashlib.sha256(
                                f"secret:{raw_identity}".encode()
                            ).hexdigest(),
                            "_raw_identity": raw_identity,
                            "subject_id": f"s{subject}",
                            "electrode_type": condition,
                            "checkpoint_group": "held",
                            "label": label,
                        }
                    )
    frame = pd.DataFrame(rows)
    digest_by_cell = {
        key: _digest(group["_raw_identity"])
        for key, group in frame.groupby(["subject_id", "electrode_type"])
    }
    frame["query_identity_sha256"] = [
        digest_by_cell[(row.subject_id, row.electrode_type)]
        for row in frame.itertuples(index=False)
    ]
    return frame.drop(columns="_raw_identity")


def _support(query: pd.DataFrame, spec: MetadataCalibrationBundleSpec) -> pd.DataFrame:
    rows = []
    for cell in query[
        ["subject_id", "electrode_type", "checkpoint_group"]
    ].drop_duplicates().to_dict(orient="records"):
        for budget in spec.budgets:
            rows.append(
                {
                    **cell,
                    "budget": budget,
                    "support_identity_sha256": hashlib.sha256(
                        f"{cell['subject_id']}:{cell['electrode_type']}:{budget}".encode()
                    ).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def _bindings() -> MetadataCalibrationBundleBindings:
    return MetadataCalibrationBundleBindings(
        plan_sha256="a" * 64,
        execution_manifest_sha256="b" * 64,
        decision_receipt_sha256="c" * 64,
        asset_receipt_sha256="d" * 64,
        asset_fingerprint_bundle_sha256="e" * 64,
        class_map_sha256="f" * 64,
        source_tree_sha256="1" * 64,
    )


def _predictions(query: pd.DataFrame, spec: MetadataCalibrationBundleSpec) -> pd.DataFrame:
    rows = []
    for role, context in spec.contexts:
        for seed in spec.seeds:
            for budget in spec.budgets:
                for row in query.to_dict(orient="records"):
                    probability = np.full(spec.n_classes, 0.05)
                    strength = 0.75
                    if role == "A_QM" and context == "correct":
                        strength = min(0.95, 0.78 + 0.03 * budget)
                    elif role == "A_QM" and context in {"block_shuffle", "stale"}:
                        strength = 0.70
                    elif role == "A_QM" and context == "opposite_interface":
                        strength = 0.72
                    if role == "A_QM" and context == "all_missing":
                        strength = 0.75
                    probability[:] = (1.0 - strength) / (spec.n_classes - 1)
                    probability[int(row["label"])] = strength
                    unlabeled = {key: value for key, value in row.items() if key != "label"}
                    rows.append(
                        {
                            **unlabeled,
                            "candidate_id": spec.candidate_id,
                            "phase": spec.phase,
                            **_bindings().as_dict(),
                            "checkpoint_sha256": hashlib.sha256(
                                f"{seed}:{row['checkpoint_group']}".encode()
                            ).hexdigest(),
                            "role": role,
                            "context": context,
                            "seed": seed,
                            "budget": budget,
                            "support_identity_sha256": hashlib.sha256(
                                f"{row['subject_id']}:{row['electrode_type']}:{budget}".encode()
                            ).hexdigest(),
                            "query_identity_sha256": row["query_identity_sha256"],
                            "intervention_mapping_sha256": hashlib.sha256(
                                f"{role}:{context}".encode()
                            ).hexdigest(),
                            "resolved_context_usage_sha256": hashlib.sha256(
                                f"{role}:{context}:{budget}:{row['checkpoint_group']}".encode()
                            ).hexdigest(),
                            "cell_execution_contract_sha256": (
                                cell_execution_contract_sha256(phase=spec.phase)
                            ),
                            **{
                                f"prob_{label:03d}": float(probability[label])
                                for label in range(spec.n_classes)
                            },
                        }
                    )
    return pd.DataFrame(rows)


def test_complete_bundle_reduces_seed_probabilities_before_balanced_accuracy() -> None:
    spec = _spec()
    query = _query()
    result = reduce_metadata_calibration_predictions(
        _predictions(query, spec),
        expected_query=query,
        expected_support=_support(query, spec),
        bindings=_bindings(),
        spec=spec,
    )

    assert result["missing_fallback_probability_exact"] is True
    assert result["seed_reduction"] == "mean_probabilities_per_sample_before_argmax"
    contrasts = result["participant_contrasts"]
    assert len(contrasts) == 2
    assert np.allclose(contrasts["eauc_increment"], 0.0)
    assert "electrode_type" not in result["participant_balanced_accuracy"]


def test_complete_private_staging_is_validated_before_any_label_join() -> None:
    spec = _spec()
    query = _query()
    predictions = _predictions(query, spec)
    result = validate_metadata_calibration_private_staging(
        predictions,
        expected_query_unlabeled=query.drop(columns="label"),
        expected_support=_support(query, spec),
        bindings=_bindings(),
        spec=spec,
    )
    assert result["query_outcomes_loaded"] is False
    assert result["complete_atomic_grid"] is True
    assert result["missing_fallback_probability_exact"] is True
    assert len(result["staging_content_sha256"]) == 64

    drift = predictions.copy()
    mask = drift["role"].eq("A_QM") & drift["context"].eq("all_missing")
    row = drift.index[mask][0]
    drift.loc[row, ["prob_000", "prob_001"]] = [0.74, 0.135]
    with pytest.raises(ValueError, match="not exactly equal"):
        validate_metadata_calibration_private_staging(
            drift,
            expected_query_unlabeled=query.drop(columns="label"),
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )


def test_bundle_rejects_missing_job_probability_error_and_missing_fallback_drift() -> None:
    spec = _spec()
    query = _query()
    predictions = _predictions(query, spec)
    with pytest.raises(ValueError, match="complete atomic grid"):
        reduce_metadata_calibration_predictions(
            predictions.iloc[:-1],
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    broken = predictions.copy()
    broken.loc[0, "prob_000"] = 2.0
    with pytest.raises(ValueError, match="normalized probability"):
        reduce_metadata_calibration_predictions(
            broken,
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    drift = predictions.copy()
    mask = drift["role"].eq("A_QM") & drift["context"].eq("all_missing")
    row = drift.index[mask][0]
    drift.loc[row, ["prob_000", "prob_001"]] = [0.74, 0.135]
    with pytest.raises(ValueError, match="not exactly equal"):
        reduce_metadata_calibration_predictions(
            drift,
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )


def test_private_staging_rejects_outcomes_and_binding_or_support_drift() -> None:
    spec = _spec()
    query = _query()
    predictions = _predictions(query, spec)

    leaked = predictions.assign(label=0)
    with pytest.raises(ValueError, match="outcome-free"):
        reduce_metadata_calibration_predictions(
            leaked,
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    label_encoded_identity = predictions.assign(sample_id="wearable_sub_target07")
    with pytest.raises(ValueError, match="outcome-free"):
        validate_metadata_calibration_private_staging(
            label_encoded_identity,
            expected_query_unlabeled=query.drop(columns="label"),
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    undeclared = predictions.assign(trial_id="trial01")
    with pytest.raises(ValueError, match="undeclared staging"):
        validate_metadata_calibration_private_staging(
            undeclared,
            expected_query_unlabeled=query.drop(columns="label"),
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    wrong_binding = predictions.copy()
    wrong_binding["plan_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="frozen plan_sha256"):
        reduce_metadata_calibration_predictions(
            wrong_binding,
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    wrong_support = predictions.copy()
    wrong_support.loc[wrong_support.index[0], "support_identity_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="sealed support"):
        reduce_metadata_calibration_predictions(
            wrong_support,
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    wrong_dispatch = predictions.copy()
    wrong_dispatch["cell_execution_contract_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="phase dispatch hash"):
        validate_metadata_calibration_private_staging(
            wrong_dispatch,
            expected_query_unlabeled=query.drop(columns="label"),
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )

    usage_drift = predictions.copy()
    target = usage_drift.index[
        usage_drift["role"].eq("A_QM")
        & usage_drift["context"].eq("correct")
        & usage_drift["budget"].eq(1)
        & usage_drift["seed"].eq(spec.seeds[0])
    ][0]
    usage_drift.loc[target, "resolved_context_usage_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="Resolved context usage differs"):
        validate_metadata_calibration_private_staging(
            usage_drift,
            expected_query_unlabeled=query.drop(columns="label"),
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )


def test_development_bundle_binds_one_checkpoint_per_seed_and_outer_fold() -> None:
    spec = MetadataCalibrationBundleSpec(
        n_classes=3,
        seeds=(1, 2),
        phase="source_development",
        checkpoint_groups=("fold0", "fold1"),
        sensitivity_resamples=200,
    )
    query = _query()
    query.loc[query["subject_id"].eq("s0"), "checkpoint_group"] = "fold0"
    query.loc[query["subject_id"].eq("s1"), "checkpoint_group"] = "fold1"
    predictions = _predictions(query, spec)
    result = reduce_metadata_calibration_predictions(
        predictions,
        expected_query=query,
        expected_support=_support(query, spec),
        bindings=_bindings(),
        spec=spec,
    )
    assert result["phase"] == "source_development"

    reused = predictions.copy()
    reused["checkpoint_sha256"] = reused["seed"].map(
        lambda seed: hashlib.sha256(f"{seed}:reused".encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="must not reuse"):
        reduce_metadata_calibration_predictions(
            reused,
            expected_query=query,
            expected_support=_support(query, spec),
            bindings=_bindings(),
            spec=spec,
        )


def test_one_sided_t_uses_t_p_value_and_margin() -> None:
    positive = one_sided_paired_t(np.full(12, 0.03), alpha=0.05, null_margin=0.0)
    boundary = one_sided_paired_t(np.full(12, -1.0 / 60.0), alpha=0.025, null_margin=-1.0 / 60.0)
    assert positive["reject_null"] is True
    assert positive["p_value"] == 0.0
    assert positive["statistic"] is None
    assert positive["degenerate_standard_error"] is True
    assert positive["degenerate_mean_relation_to_null"] == "above_null"
    assert boundary["reject_null"] is False


def _contrast_table(n: int, *, strong: bool) -> pd.DataFrame:
    effect = 0.03 if strong else 0.0
    mechanism = 0.02 if strong else 0.0
    values = np.linspace(-0.002, 0.002, n)
    frame = pd.DataFrame({"subject_id": [f"s{index:03d}" for index in range(n)]})
    for budget, baseline in ((0, 0.60), (1, 0.64), (3, 0.68), (5, 0.72)):
        frame[f"aq_k{budget}"] = baseline
        frame[f"aqm_k{budget}"] = baseline + effect + values
        if budget in (0, 1, 3):
            frame[f"shuffle_k{budget}"] = frame[f"aqm_k{budget}"] - (
                mechanism + values
            )
            frame[f"stale_k{budget}"] = frame[f"aqm_k{budget}"] - (
                mechanism + values
            )
            frame[f"wrong_k{budget}"] = baseline - 0.002 + values / 10.0
    return _recalculate_derived(frame)


def _recalculate_derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    weights = {0: 1.0 / 6.0, 1: 1.0 / 2.0, 3: 1.0 / 3.0}
    result["eauc_increment"] = sum(
        weight * (result[f"aqm_k{budget}"] - result[f"aq_k{budget}"])
        for budget, weight in weights.items()
    )
    result["correct_minus_shuffle_eauc"] = sum(
        weight * (result[f"aqm_k{budget}"] - result[f"shuffle_k{budget}"])
        for budget, weight in weights.items()
    )
    result["correct_minus_stale_eauc"] = sum(
        weight * (result[f"aqm_k{budget}"] - result[f"stale_k{budget}"])
        for budget, weight in weights.items()
    )
    result["wrong_minus_aq_eauc"] = sum(
        weight * (result[f"wrong_k{budget}"] - result[f"aq_k{budget}"])
        for budget, weight in weights.items()
    )
    result["calibration_value_k1_to_k3"] = result["aq_k3"] - result["aq_k1"]
    result["save_k0_vs_k1"] = result["aqm_k0"] - result["aq_k1"]
    result["save_k1_vs_k3"] = result["aqm_k1"] - result["aq_k3"]
    result["save_k3_vs_k5"] = result["aqm_k3"] - result["aq_k5"]
    return result


def test_development_gate_is_all_or_nothing() -> None:
    frame = _contrast_table(39, strong=True)
    folds = {subject: index % 3 for index, subject in enumerate(frame["subject_id"])}
    passed = evaluate_development_gate(
        frame,
        outer_fold_by_subject=folds,
        missing_fallback_probability_exact=True,
        metadata_only_structural_chance_exact=True,
    )
    assert passed["passed"] is True
    assert passed["held_evaluation_access_allowed"] is True
    observed = passed["observed_statistics"]
    assert observed["eauc_increment_mean"] == pytest.approx(0.03)
    assert observed["correct_minus_shuffle_eauc_mean"] == pytest.approx(0.02)
    assert observed["eauc_increment_positive_participants"] == 39
    assert set(observed["outer_fold_means"]["eauc_increment"]) == {"0", "1", "2"}

    failed = evaluate_development_gate(
        frame,
        outer_fold_by_subject=folds,
        missing_fallback_probability_exact=False,
        metadata_only_structural_chance_exact=True,
    )
    assert failed["passed"] is False
    assert failed["decision"] == "development_no_go"


def test_metadata_only_screen_is_a_deterministic_structural_proof() -> None:
    frame = _contrast_table(39, strong=True)
    folds = {subject: index % 3 for index, subject in enumerate(frame["subject_id"])}
    failed = evaluate_development_gate(
        frame,
        outer_fold_by_subject=folds,
        missing_fallback_probability_exact=True,
        metadata_only_structural_chance_exact=False,
    )
    assert failed["passed"] is False
    assert failed["metadata_only_screen"]["method"] == (
        "deterministic_information_structure_proof"
    )
    assert failed["metadata_only_screen"]["implied_balanced_accuracy"] == pytest.approx(
        1.0 / 12.0
    )

    unbalanced = {
        subject: (0 if index < 14 else (1 if index < 27 else 2))
        for index, subject in enumerate(frame["subject_id"])
    }
    with pytest.raises(ValueError, match="equally sized"):
        evaluate_development_gate(
            frame,
            outer_fold_by_subject=unbalanced,
            missing_fallback_probability_exact=True,
            metadata_only_structural_chance_exact=True,
        )


def test_held_claims_use_two_stage_core_and_separate_safety() -> None:
    spec = MetadataCalibrationBundleSpec(sensitivity_resamples=200)
    passed = evaluate_held_claim_sequence(_contrast_table(60, strong=True), spec=spec)
    assert passed["primary_claim_passed"] is True
    assert passed["calibration_saving_claim_passed"] is True
    assert passed["confirmed_saving"] == (
        "two_complete_blocks_per_interface_schedule_in_equal_wet_dry_estimand"
    )
    assert passed["confirmed_labeled_trial_reduction_per_interface"] == 24
    assert all(test["component_passed"] for test in passed["confirmatory_tests"][1:])
    assert all(test["claim_passed"] for test in passed["mechanism_family"]["tests"])
    assert passed["deployment_safety"]["deployment_qualified"] is True
    assert passed["k1_consistency"]["identity_exact"] is True

    weak = _contrast_table(60, strong=False)
    failed = evaluate_held_claim_sequence(weak, spec=spec)
    assert failed["primary_claim_passed"] is False
    assert failed["confirmatory_tests"][1]["tested_confirmatorily"] is False
    assert failed["calibration_saving_claim_passed"] is False
    assert failed["confirmed_saving"] is None


def test_savings_claim_requires_calibration_value_and_endpoint_utility() -> None:
    spec = MetadataCalibrationBundleSpec(sensitivity_resamples=200)
    no_calibration_value = _contrast_table(60, strong=True)
    no_calibration_value["aq_k3"] = no_calibration_value["aq_k1"] + 0.005
    no_calibration_value = _recalculate_derived(no_calibration_value)
    result = evaluate_held_claim_sequence(no_calibration_value, spec=spec)
    assert result["primary_claim_passed"] is True
    assert result["confirmatory_tests"][1]["practical_threshold_met"] is False
    assert result["calibration_saving_claim_passed"] is False

    below_floor = _contrast_table(60, strong=True)
    below_floor["aq_k1"] = 0.40
    below_floor["aq_k3"] = 0.46
    below_floor["aqm_k1"] = 0.45
    below_floor = _recalculate_derived(below_floor)
    result = evaluate_held_claim_sequence(below_floor, spec=spec)
    assert result["confirmatory_tests"][2]["utility_floor_met"] is False
    assert result["calibration_saving_claim_passed"] is False


def test_held_claims_reject_raw_ba_or_derived_contrast_drift() -> None:
    invalid_ba = _contrast_table(60, strong=True)
    invalid_ba.loc[0, "aq_k0"] = 1.01
    with pytest.raises(ValueError, match="must lie in"):
        evaluate_held_claim_sequence(invalid_ba)

    inconsistent = _contrast_table(60, strong=True)
    inconsistent.loc[0, "eauc_increment"] += 0.001
    with pytest.raises(ValueError, match="algebraically inconsistent"):
        evaluate_held_claim_sequence(inconsistent)

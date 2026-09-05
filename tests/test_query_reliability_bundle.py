from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cfeg.analysis.query_reliability_bundle import (
    QUERY_RELIABILITY_AFFECTED_MASK_HASH_SCHEMA,
    QUERY_RELIABILITY_EVALUATION_INPUT_HASH_SCHEMA,
    QueryReliabilityBundleBindings,
    QueryReliabilityBundleSpec,
    _bundle_spec_from_plan,
    _validate_and_reduce_query_reliability_bundle,
    query_reliability_affected_mask_sha256,
    query_reliability_evaluation_input_sha256,
    validate_and_reduce_query_reliability_bundle,
)
from cfeg.governance import GovernanceError
from cfeg.query_reliability_contract import QUERY_RELIABILITY_ROLES
from cfeg.utils.config import load_config

FAMILY_SHA = "a" * 64
SOURCE_SHA = "b" * 64
PLAN_SHA = "c" * 64
ASSET_RECEIPT_SHA = "d" * 64
ASSET_BUNDLE_SHA = "e" * 64
SPLIT_SHA = "f" * 64


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _bindings(samples: pd.DataFrame) -> QueryReliabilityBundleBindings:
    records = (
        samples.loc[:, ["sample_id", "subject_id", "label"]]
        .sort_values("sample_id", kind="mergesort")
        .to_dict(orient="records")
    )
    sample_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return QueryReliabilityBundleBindings(
        family_sha256=FAMILY_SHA,
        source_sha256=SOURCE_SHA,
        plan_sha256=PLAN_SHA,
        asset_receipt_sha256=ASSET_RECEIPT_SHA,
        asset_fingerprint_bundle_sha256=ASSET_BUNDLE_SHA,
        sample_identity_sha256=sample_sha,
        split_sha256=SPLIT_SHA,
    )


def _fixture() -> tuple[QueryReliabilityBundleSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = QueryReliabilityBundleSpec(
        roles=tuple(QUERY_RELIABILITY_ROLES),
        seeds=(11, 29),
        corruption_cells=("noise",),
        draw_indices=(0, 1),
        n_classes=2,
        trials_per_participant_class=1,
        n_participants=2,
        n_channels=2,
    )
    samples = pd.DataFrame(
        [
            {"sample_id": f"{subject}-{label}", "subject_id": subject, "label": label}
            for subject in ("s1", "s2")
            for label in (0, 1)
        ]
    )
    bindings = _bindings(samples)
    jobs = pd.DataFrame(
        [
            {
                "candidate_id": "query-reliability-spatial-v1",
                "plan_sha256": bindings.plan_sha256,
                "family_sha256": FAMILY_SHA,
                "asset_receipt_sha256": bindings.asset_receipt_sha256,
                "asset_fingerprint_bundle_sha256": (bindings.asset_fingerprint_bundle_sha256),
                "sample_identity_sha256": bindings.sample_identity_sha256,
                "role": role,
                "seed": seed,
                "status": "completed_fixed_epoch",
                "epochs_completed": 10,
                "checkpoint_epoch": 10,
                "checkpoint_sha256": _digest(("checkpoint", role, seed)),
                "split_sha256": bindings.split_sha256,
                "parameter_schema_sha256": _digest("parameters"),
                "initial_state_sha256": _digest(("initial", seed)),
                "source_sha256": SOURCE_SHA,
                "fairness_recipe_sha256": _digest("fairness"),
                "runtime_recipe_sha256": _digest(("runtime", role, seed)),
            }
            for role in spec.roles
            for seed in spec.seeds
        ]
    )
    checkpoint = jobs.set_index(["role", "seed"])["checkpoint_sha256"].to_dict()
    rows: list[dict] = []
    for role in spec.roles:
        for seed in spec.seeds:
            for cell, draw in spec.evaluation_scenarios:
                for record in samples.to_dict(orient="records"):
                    label = int(record["label"])
                    prediction = label
                    if role == "Q0_AUG" and cell != "clean":
                        prediction = 1 - label
                    probabilities = np.full(2, 0.1, dtype=float)
                    probabilities[prediction] = 0.9
                    rows.append(
                        {
                            "candidate_id": "query-reliability-spatial-v1",
                            "family_sha256": FAMILY_SHA,
                            "role": role,
                            "seed": seed,
                            "eval_cell": cell,
                            "draw_index": draw,
                            **record,
                            "checkpoint_sha256": checkpoint[(role, seed)],
                            "evaluation_input_hash_schema": (
                                QUERY_RELIABILITY_EVALUATION_INPUT_HASH_SCHEMA
                            ),
                            "evaluation_input_sha256": _digest(
                                (cell, draw, record["sample_id"], "input")
                            ),
                            "affected_mask_hash_schema": (
                                QUERY_RELIABILITY_AFFECTED_MASK_HASH_SCHEMA
                            ),
                            "affected_mask_sha256": (
                                query_reliability_affected_mask_sha256(
                                    np.zeros(spec.n_channels, dtype=bool), clean=True
                                )
                                if cell == "clean"
                                else _digest((cell, draw, record["sample_id"], "mask"))
                            ),
                            "prob_000": probabilities[0],
                            "prob_001": probabilities[1],
                        }
                    )
    return spec, jobs, pd.DataFrame(rows), samples


def _reduce(spec, jobs, predictions, samples):
    return _validate_and_reduce_query_reliability_bundle(
        jobs=jobs,
        predictions=predictions,
        expected_samples=samples,
        bindings=_bindings(samples),
        spec=spec,
    )


def test_public_reducer_rejects_draft_before_predictions_or_asset_access() -> None:
    with pytest.raises(GovernanceError, match="still a draft"):
        validate_and_reduce_query_reliability_bundle(
            jobs=pd.DataFrame(),
            predictions=pd.DataFrame(),
            processed_root="/path-that-must-not-be-read",
        )


def test_bundle_spec_is_resolved_from_the_hash_bound_plan_values() -> None:
    plan = load_config("configs/analysis/query_reliability_spatial_v1.yaml")
    spec = _bundle_spec_from_plan(plan)
    assert spec.n_participants == 20
    assert spec.n_channels == 64
    assert spec.practical_effect_min == 0.02
    assert spec.clean_noninferiority_margin == -0.01

    changed = json.loads(json.dumps(plan))
    changed["estimands"]["provisional_gates_not_yet_frozen"]["primary_mean_delta_min"] = 0.03
    assert _bundle_spec_from_plan(changed).practical_effect_min == 0.03


def test_canonical_input_hash_covers_every_query_field_and_clean_mask() -> None:
    kwargs = {
        "sample_id": "sample-1",
        "x": np.zeros((2, 4), dtype=np.float32),
        "channel_mask": np.array([True, True]),
        "canonical_channel_ids": np.array([61, 62], dtype=np.int64),
        "channel_query_qc": np.array([1.0, 2.0], dtype=np.float32),
        "channel_query_qc_missing": np.array([False, False]),
        "sfreq_processed": 200.0,
    }
    reference = query_reliability_evaluation_input_sha256(**kwargs)
    for field, changed in (
        ("sample_id", "sample-2"),
        ("x", np.ones((2, 4), dtype=np.float32)),
        ("channel_mask", np.array([True, False])),
        ("canonical_channel_ids", np.array([61, 63], dtype=np.int64)),
        ("channel_query_qc", np.array([1.0, 3.0], dtype=np.float32)),
        ("channel_query_qc_missing", np.array([False, True])),
        ("sfreq_processed", 250.0),
    ):
        mutated = dict(kwargs)
        mutated[field] = changed
        assert query_reliability_evaluation_input_sha256(**mutated) != reference

    clean = np.array([False, False])
    assert len(query_reliability_affected_mask_sha256(clean, clean=True)) == 64
    with pytest.raises(ValueError, match="clean evaluation row"):
        query_reliability_affected_mask_sha256(np.array([True, False]), clean=True)


def test_complete_bundle_averages_seed_probabilities_before_participant_ba() -> None:
    spec, jobs, predictions, samples = _fixture()
    result = _reduce(spec, jobs, predictions, samples)
    assert result["seed_reduction"] == ("mean_probabilities_per_sample_before_argmax_and_ba")
    assert result["tie_break"] == "lowest_class_index"
    assert result["atomic_publication_authorized"] is False
    assert len(result["verified_jobs_sha256"]) == 64
    assert len(result["verified_predictions_sha256"]) == 64
    assert len(result["analysis_input_contract_sha256"]) == 64
    assert result["promotion_gates"]["promotion_pass"] is True
    contrasts = result["participant_contrasts"]
    assert contrasts["corrupted_q1_aug_minus_q0_aug"].eq(1.0).all()
    assert contrasts["corrupted_q1_aug_minus_q0_clean"].eq(0.0).all()
    assert contrasts["corrupted_query_by_augmentation_interaction"].eq(1.0).all()
    assert set(result["participant_cell_transitions"]["transition"]) == {"gained"}
    assert set(result["participant_composite_lower_tail"]["participant_quantile"]) == {
        0.10,
        0.25,
        0.50,
    }
    assert result["class_by_corruption_cell"]["q1_aug_minus_q0_aug"].eq(1.0).all()
    assert not result["class_by_corruption_cell"]["fifty_hz_harmonic_overlap"].any()


def test_bundle_contract_hashes_every_verified_job_and_prediction_field() -> None:
    spec, jobs, predictions, samples = _fixture()
    reference = _reduce(spec, jobs, predictions, samples)
    reordered = _reduce(
        spec,
        jobs.sample(frac=1.0, random_state=5),
        predictions.sample(frac=1.0, random_state=7),
        samples,
    )
    assert reordered["verified_jobs_sha256"] == reference["verified_jobs_sha256"]
    assert reordered["verified_predictions_sha256"] == reference["verified_predictions_sha256"]

    changed_jobs = jobs.copy()
    changed_jobs.loc[0, "runtime_recipe_sha256"] = _digest("another-runtime")
    job_result = _reduce(spec, changed_jobs, predictions, samples)
    assert job_result["verified_jobs_sha256"] != reference["verified_jobs_sha256"]
    assert (
        job_result["analysis_input_contract_sha256"] != reference["analysis_input_contract_sha256"]
    )

    changed_predictions = predictions.copy()
    changed_predictions.loc[0, ["prob_000", "prob_001"]] = [0.8, 0.2]
    prediction_result = _reduce(spec, jobs, changed_predictions, samples)
    assert (
        prediction_result["verified_predictions_sha256"] != reference["verified_predictions_sha256"]
    )
    assert (
        prediction_result["analysis_input_contract_sha256"]
        != reference["analysis_input_contract_sha256"]
    )


def test_bundle_rejects_incomplete_grid_and_role_specific_corruption() -> None:
    spec, jobs, predictions, samples = _fixture()
    with pytest.raises(ValueError, match="complete role/seed/scenario grid"):
        _reduce(spec, jobs, predictions.iloc[:-1].copy(), samples)

    mismatched = predictions.copy()
    mismatched.loc[0, "evaluation_input_sha256"] = _digest("role-specific-input")
    with pytest.raises(ValueError, match="differs across role/seed"):
        _reduce(spec, jobs, mismatched, samples)

    false_clean_mask = predictions.copy()
    false_clean_mask.loc[false_clean_mask["eval_cell"].eq("clean"), "affected_mask_sha256"] = (
        _digest("not-the-canonical-clean-mask")
    )
    with pytest.raises(ValueError, match="canonical all-false mask"):
        _reduce(spec, jobs, false_clean_mask, samples)


def test_bundle_rejects_incomplete_jobs_nonfinite_probabilities_and_state_drift() -> None:
    spec, jobs, predictions, samples = _fixture()
    with pytest.raises(ValueError, match="complete four-role by seed"):
        _reduce(spec, jobs.iloc[:-1].copy(), predictions, samples)

    nonfinite = predictions.copy()
    nonfinite.loc[0, "prob_000"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        _reduce(spec, jobs, nonfinite, samples)

    drift = jobs.copy()
    drift.loc[(drift["role"] == "Q1_AUG") & (drift["seed"] == 11), "initial_state_sha256"] = (
        _digest("drift")
    )
    with pytest.raises(ValueError, match="initial trainable state"):
        _reduce(spec, drift, predictions, samples)

    repeated_seed_state = jobs.copy()
    repeated_seed_state["initial_state_sha256"] = _digest("one-seed-only")
    with pytest.raises(ValueError, match="distinct initial state"):
        _reduce(spec, repeated_seed_state, predictions, samples)

    repeated_checkpoint = jobs.copy()
    repeated_checkpoint.loc[0, "checkpoint_sha256"] = repeated_checkpoint.loc[
        1, "checkpoint_sha256"
    ]
    role = repeated_checkpoint.loc[0, "role"]
    seed = repeated_checkpoint.loc[0, "seed"]
    predictions_for_job = predictions.copy()
    predictions_for_job.loc[
        predictions_for_job["role"].eq(role) & predictions_for_job["seed"].eq(seed),
        "checkpoint_sha256",
    ] = repeated_checkpoint.loc[0, "checkpoint_sha256"]
    with pytest.raises(ValueError, match="distinct final checkpoint"):
        _reduce(spec, repeated_checkpoint, predictions_for_job, samples)

    plan_drift = jobs.copy()
    plan_drift.loc[0, "plan_sha256"] = _digest("another-plan")
    with pytest.raises(ValueError, match="plan binding"):
        _reduce(spec, plan_drift, predictions, samples)

    fractional_seed = jobs.copy()
    fractional_seed["seed"] = fractional_seed["seed"].astype(float)
    fractional_seed.loc[0, "seed"] = 11.9
    with pytest.raises(ValueError, match="job seed must contain exact finite integers"):
        _reduce(spec, fractional_seed, predictions, samples)

    fractional_draw = predictions.copy()
    fractional_draw["draw_index"] = fractional_draw["draw_index"].astype(float)
    fractional_draw.loc[0, "draw_index"] = 0.9
    with pytest.raises(
        ValueError, match="prediction draw_index must contain exact finite integers"
    ):
        _reduce(spec, jobs, fractional_draw, samples)

    sample_drift = samples.copy()
    sample_drift.loc[0, "sample_id"] = "different-sample"
    with pytest.raises(ValueError, match="Expected-sample identity"):
        _validate_and_reduce_query_reliability_bundle(
            jobs=jobs,
            predictions=predictions,
            expected_samples=sample_drift,
            bindings=_bindings(samples),
            spec=spec,
        )

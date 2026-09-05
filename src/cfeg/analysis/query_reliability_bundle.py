from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cfeg.analysis.query_reliability_statistics import (
    evaluate_query_reliability_promotion_gates,
)
from cfeg.governance import GovernanceError, current_source_revision_contract
from cfeg.query_reliability_contract import (
    QUERY_RELIABILITY_ASSET_RECEIPT_SHA256,
    QUERY_RELIABILITY_BASE_CONFIG,
    QUERY_RELIABILITY_CANDIDATE_PLAN,
    QUERY_RELIABILITY_CORRUPTION_CELLS,
    QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256,
    QUERY_RELIABILITY_ROLES,
    QUERY_RELIABILITY_SUITE_CONFIG,
    load_query_reliability_beta_lockbox_manifest,
    validate_query_reliability_family,
)
from cfeg.utils.config import load_config

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
QUERY_RELIABILITY_EVALUATION_INPUT_HASH_SCHEMA = "cfeg.query-reliability-evaluation-input-sha256.v1"
QUERY_RELIABILITY_AFFECTED_MASK_HASH_SCHEMA = "cfeg.query-reliability-affected-mask-sha256.v1"


@dataclass(frozen=True)
class QueryReliabilityBundleSpec:
    roles: tuple[str, ...]
    seeds: tuple[int, ...]
    corruption_cells: tuple[str, ...]
    draw_indices: tuple[int, ...]
    n_classes: int
    trials_per_participant_class: int
    n_participants: int
    n_channels: int = 64
    alpha: float = 0.05
    practical_effect_min: float = 0.02
    clean_noninferiority_margin: float = -0.01
    deployment_clean_noninferiority_margin: float = -0.01
    positive_fraction_min: float = 0.60
    checkpoint_epoch: int = 10

    @property
    def evaluation_scenarios(self) -> tuple[tuple[str, int], ...]:
        return (("clean", 0),) + tuple(
            (cell, draw) for cell in self.corruption_cells for draw in self.draw_indices
        )

    @property
    def probability_columns(self) -> tuple[str, ...]:
        return tuple(f"prob_{index:03d}" for index in range(self.n_classes))


@dataclass(frozen=True)
class QueryReliabilityBundleBindings:
    family_sha256: str
    source_sha256: str
    plan_sha256: str
    asset_receipt_sha256: str
    asset_fingerprint_bundle_sha256: str
    sample_identity_sha256: str
    split_sha256: str


DRAFT_QUERY_RELIABILITY_BUNDLE_SPEC = QueryReliabilityBundleSpec(
    roles=tuple(QUERY_RELIABILITY_ROLES),
    seeds=(11, 29, 47),
    corruption_cells=tuple(QUERY_RELIABILITY_CORRUPTION_CELLS),
    draw_indices=(0, 1, 2),
    n_classes=40,
    trials_per_participant_class=4,
    n_participants=20,
    n_channels=64,
    alpha=0.05,
    practical_effect_min=0.02,
    clean_noninferiority_margin=-0.01,
    deployment_clean_noninferiority_margin=-0.01,
    positive_fraction_min=0.60,
    checkpoint_epoch=10,
)


def validate_and_reduce_query_reliability_bundle(
    *,
    jobs: pd.DataFrame,
    predictions: pd.DataFrame,
    processed_root: str | Path,
) -> dict:
    """Validate and reduce only an owner-approved, canonically bound BETA bundle.

    Callers cannot supply their own sample table or expected digests. The exact
    plan, family, clean source tree, asset receipt, six-file asset fingerprint,
    split, and 20-person/3,200-row sample identity are resolved internally. The
    current draft plan is intentionally rejected before predictions are reduced.
    Private staging and atomic publication remain the future runner's separate
    responsibility.
    """

    samples, bindings, spec = _load_canonical_bundle_inputs(processed_root)
    return _validate_and_reduce_query_reliability_bundle(
        jobs=jobs,
        predictions=predictions,
        expected_samples=samples,
        bindings=bindings,
        spec=spec,
    )


def _validate_and_reduce_query_reliability_bundle(
    *,
    jobs: pd.DataFrame,
    predictions: pd.DataFrame,
    expected_samples: pd.DataFrame,
    bindings: QueryReliabilityBundleBindings,
    spec: QueryReliabilityBundleSpec,
) -> dict:
    """Parameterized implementation used by small, outcome-free test fixtures."""

    _validate_bindings(bindings)
    _validate_spec(spec)
    samples = _validate_expected_samples(expected_samples, spec)
    if _sample_identity_sha256(samples) != bindings.sample_identity_sha256:
        raise ValueError("Expected-sample identity differs from its canonical binding.")
    verified_jobs = _validate_jobs(
        jobs,
        bindings=bindings,
        spec=spec,
    )
    verified_predictions = _validate_predictions(
        predictions,
        expected_samples=samples,
        jobs=verified_jobs,
        expected_family_sha256=bindings.family_sha256,
        spec=spec,
    )
    return _reduce_predictions(
        verified_predictions,
        jobs=verified_jobs,
        expected_samples=samples,
        bindings=bindings,
        spec=spec,
    )


def _load_canonical_bundle_inputs(
    processed_root: str | Path,
) -> tuple[
    pd.DataFrame,
    QueryReliabilityBundleBindings,
    QueryReliabilityBundleSpec,
]:
    """Resolve future production bindings and fail closed while the plan is a draft."""

    repository = Path(__file__).resolve().parents[3]
    plan_path = repository / QUERY_RELIABILITY_CANDIDATE_PLAN
    plan = load_config(plan_path, strict_env=False)
    execution = plan.get("execution_authority", {}) or {}
    bundle = plan.get("outcome_bundle_contract", {}) or {}
    if (
        plan.get("status") != "frozen_exact_outcome_bundle"
        or plan.get("exact_outcome_bundle_approved") is not True
        or execution.get("human_training_allowed") is not True
        or execution.get("lockbox_outcomes_allowed") is not True
        or bundle.get("status") != "frozen_owner_approved"
        or bundle.get("grants_human_outcome_execution_authority") is not True
    ):
        raise GovernanceError(
            "The query-reliability outcome bundle is still a draft; canonical reduction "
            "is blocked before any prediction outcome is inspected."
        )

    source_freeze_tag = execution.get("source_freeze_tag")
    current_source = current_source_revision_contract()
    if (
        not isinstance(source_freeze_tag, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", source_freeze_tag)
        or current_source.get("source_dirty") is not False
    ):
        raise GovernanceError("The frozen bundle requires one safe clean-source tag.")
    tag_commit = _resolve_git_tag_commit(repository, source_freeze_tag)
    if current_source.get("source_commit_sha") != tag_commit:
        raise GovernanceError("HEAD differs from the exact source-freeze tag target.")
    source_sha256 = current_source.get("source_tree_sha256")
    if not _is_sha256(source_sha256):
        raise GovernanceError("The clean source-tree digest is unavailable.")

    base = load_config(repository / QUERY_RELIABILITY_BASE_CONFIG, strict_env=False)
    suite = load_config(repository / QUERY_RELIABILITY_SUITE_CONFIG, strict_env=False)
    family_sha256 = validate_query_reliability_family(base, suite)
    samples = load_query_reliability_beta_lockbox_manifest(processed_root)
    asset_fingerprint_bundle_sha256 = _sha256_json(
        plan["data_contract"]["primary_dataset"]["asset_fingerprints"]
    )
    split_sha256 = bundle.get("canonical_runtime_split_sha256")
    if (
        bundle.get("canonical_lockbox_n_subjects") != 20
        or bundle.get("canonical_lockbox_n_samples") != 3200
        or bundle.get("canonical_lockbox_sample_identity_sha256")
        != QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256
        or bundle.get("canonical_asset_fingerprint_bundle_sha256")
        != asset_fingerprint_bundle_sha256
        or not _is_sha256(split_sha256)
    ):
        raise GovernanceError("The frozen outcome bundle lacks canonical data bindings.")
    bindings = QueryReliabilityBundleBindings(
        family_sha256=family_sha256,
        source_sha256=str(source_sha256),
        plan_sha256=_sha256_file(plan_path),
        asset_receipt_sha256=QUERY_RELIABILITY_ASSET_RECEIPT_SHA256,
        asset_fingerprint_bundle_sha256=asset_fingerprint_bundle_sha256,
        sample_identity_sha256=QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256,
        split_sha256=str(split_sha256),
    )
    return samples, bindings, _bundle_spec_from_plan(plan)


def _bundle_spec_from_plan(plan: dict) -> QueryReliabilityBundleSpec:
    """Resolve statistical and grid values from the same hash-bound plan."""

    primary = plan.get("data_contract", {}).get("primary_dataset", {}) or {}
    experiment = plan.get("experiment", {}) or {}
    gates = plan.get("estimands", {}).get("provisional_gates_not_yet_frozen", {}) or {}
    alpha_values = {
        float(gates.get("primary_one_sided_alpha", np.nan)),
        float(gates.get("clean_one_sided_alpha", np.nan)),
        float(gates.get("deployment_clean_one_sided_alpha", np.nan)),
    }
    if len(alpha_values) != 1:
        raise GovernanceError("The current reducer requires one common frozen alpha.")
    roles = tuple(str(value) for value in experiment.get("roles", []))
    seeds = tuple(int(value) for value in experiment.get("seeds", []))
    corruption_cells = tuple(
        str(value) for value in experiment.get("corruption_cells_equal_weight", [])
    )
    draw_indices = tuple(int(value) for value in experiment.get("evaluation_draw_indices", []))
    if (
        roles != tuple(QUERY_RELIABILITY_ROLES)
        or seeds != (11, 29, 47)
        or corruption_cells != tuple(QUERY_RELIABILITY_CORRUPTION_CELLS)
        or draw_indices != (0, 1, 2)
        or int(experiment.get("epochs", 0)) != 10
    ):
        raise GovernanceError("The hash-bound plan grid differs from the canonical family.")
    return QueryReliabilityBundleSpec(
        roles=roles,
        seeds=seeds,
        corruption_cells=corruption_cells,
        draw_indices=draw_indices,
        n_classes=int(primary.get("n_classes", 0)),
        trials_per_participant_class=int(primary.get("trials_per_participant_class", 0)),
        n_participants=int(primary.get("lockbox_n_subjects", 0)),
        n_channels=int(primary.get("n_channels", 0)),
        alpha=alpha_values.pop(),
        practical_effect_min=float(gates.get("primary_mean_delta_min", np.nan)),
        clean_noninferiority_margin=float(gates.get("clean_noninferiority_margin", np.nan)),
        deployment_clean_noninferiority_margin=float(
            gates.get("deployment_clean_noninferiority_margin", np.nan)
        ),
        positive_fraction_min=float(gates.get("primary_positive_participant_fraction_min", np.nan)),
        checkpoint_epoch=int(experiment.get("epochs", 0)),
    )


def _validate_bindings(bindings: QueryReliabilityBundleBindings) -> None:
    for field, value in bindings.__dict__.items():
        if not _is_sha256(value):
            raise ValueError(f"Bundle binding {field} must be a SHA-256 value.")


def _validate_spec(spec: QueryReliabilityBundleSpec) -> None:
    if tuple(spec.roles) != tuple(QUERY_RELIABILITY_ROLES):
        raise ValueError("The bundle must retain the exact four treatment roles.")
    if len(spec.seeds) < 2 or len(set(spec.seeds)) != len(spec.seeds):
        raise ValueError("The bundle needs at least two unique optimization seeds.")
    if not spec.corruption_cells or len(set(spec.corruption_cells)) != len(spec.corruption_cells):
        raise ValueError("Corruption cells must be nonempty and unique.")
    if not spec.draw_indices or len(set(spec.draw_indices)) != len(spec.draw_indices):
        raise ValueError("Evaluation draw indices must be nonempty and unique.")
    if (
        min(
            spec.n_classes,
            spec.trials_per_participant_class,
            spec.n_participants,
            spec.n_channels,
        )
        <= 0
    ):
        raise ValueError("Bundle dimensions must all be positive.")
    numeric = np.asarray(
        [
            spec.alpha,
            spec.practical_effect_min,
            spec.clean_noninferiority_margin,
            spec.deployment_clean_noninferiority_margin,
            spec.positive_fraction_min,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Bundle statistical thresholds must be finite.")
    if not 0.0 < spec.alpha < 0.5:
        raise ValueError("Bundle alpha must lie within (0,0.5).")
    if not 0.0 < spec.practical_effect_min <= 1.0:
        raise ValueError("Bundle practical effect threshold must lie within (0,1].")
    if not -1.0 <= spec.clean_noninferiority_margin < 0.0:
        raise ValueError("Bundle clean margin must lie within [-1,0).")
    if not -1.0 <= spec.deployment_clean_noninferiority_margin < 0.0:
        raise ValueError("Bundle deployment clean margin must lie within [-1,0).")
    if not 0.0 < spec.positive_fraction_min <= 1.0:
        raise ValueError("Bundle positive fraction must lie within (0,1].")


def _validate_expected_samples(
    expected_samples: pd.DataFrame, spec: QueryReliabilityBundleSpec
) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "label"}
    missing = sorted(required - set(expected_samples.columns))
    if missing:
        raise ValueError(f"Expected-sample manifest is missing columns: {missing}.")
    samples = expected_samples.loc[:, sorted(required)].copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["subject_id"] = samples["subject_id"].astype(str)
    samples["label"] = _strict_integer_series(samples["label"], field="sample label")
    if samples["sample_id"].duplicated().any():
        raise ValueError("Expected-sample manifest contains duplicate sample IDs.")
    if samples["subject_id"].nunique() != spec.n_participants:
        raise ValueError("Expected-sample participant count differs from the bundle spec.")
    if set(samples["label"]) != set(range(spec.n_classes)):
        raise ValueError("Expected-sample labels must be exactly 0..n_classes-1.")
    counts = samples.groupby(["subject_id", "label"]).size()
    if (
        len(counts) != spec.n_participants * spec.n_classes
        or not counts.eq(spec.trials_per_participant_class).all()
    ):
        raise ValueError(
            "Every participant/class must contain the exact preregistered trial count."
        )
    return samples.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _validate_jobs(
    jobs: pd.DataFrame,
    *,
    bindings: QueryReliabilityBundleBindings,
    spec: QueryReliabilityBundleSpec,
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "plan_sha256",
        "family_sha256",
        "asset_receipt_sha256",
        "asset_fingerprint_bundle_sha256",
        "sample_identity_sha256",
        "role",
        "seed",
        "status",
        "epochs_completed",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "split_sha256",
        "parameter_schema_sha256",
        "initial_state_sha256",
        "source_sha256",
        "fairness_recipe_sha256",
        "runtime_recipe_sha256",
    }
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"Job table is missing columns: {missing}.")
    frame = jobs.loc[:, sorted(required)].copy()
    frame["seed"] = _strict_integer_series(frame["seed"], field="job seed")
    frame["epochs_completed"] = _strict_integer_series(
        frame["epochs_completed"], field="epochs_completed"
    )
    frame["checkpoint_epoch"] = _strict_integer_series(
        frame["checkpoint_epoch"], field="checkpoint_epoch"
    )
    expected_grid = {(role, seed) for role in spec.roles for seed in spec.seeds}
    observed_grid = set(frame[["role", "seed"]].itertuples(index=False, name=None))
    if len(frame) != len(expected_grid) or observed_grid != expected_grid:
        raise ValueError("Outcome bundle requires the complete four-role by seed job grid.")
    if frame.duplicated(["role", "seed"]).any():
        raise ValueError("Outcome bundle contains a duplicate role/seed job.")
    if set(frame["candidate_id"].astype(str)) != {"query-reliability-spatial-v1"}:
        raise ValueError("Job candidate binding is wrong.")
    if set(frame["plan_sha256"].astype(str)) != {bindings.plan_sha256}:
        raise ValueError("Job plan binding is incomplete or mixed.")
    if set(frame["family_sha256"].astype(str)) != {bindings.family_sha256}:
        raise ValueError("Job family binding is incomplete or mixed.")
    if set(frame["source_sha256"].astype(str)) != {bindings.source_sha256}:
        raise ValueError("Job source binding is incomplete or mixed.")
    if set(frame["asset_receipt_sha256"].astype(str)) != {bindings.asset_receipt_sha256}:
        raise ValueError("Job asset-receipt binding is incomplete or mixed.")
    if set(frame["asset_fingerprint_bundle_sha256"].astype(str)) != {
        bindings.asset_fingerprint_bundle_sha256
    }:
        raise ValueError("Job asset-fingerprint binding is incomplete or mixed.")
    if set(frame["sample_identity_sha256"].astype(str)) != {bindings.sample_identity_sha256}:
        raise ValueError("Job sample-identity binding is incomplete or mixed.")
    if set(frame["split_sha256"].astype(str)) != {bindings.split_sha256}:
        raise ValueError("Job split binding differs from the canonical partition.")
    if set(frame["status"].astype(str)) != {"completed_fixed_epoch"}:
        raise ValueError("Every job must be terminal at the fixed epoch.")
    if (
        not frame["epochs_completed"].eq(spec.checkpoint_epoch).all()
        or not frame["checkpoint_epoch"].eq(spec.checkpoint_epoch).all()
    ):
        raise ValueError("Every job must bind the exact plan-declared final checkpoint epoch.")
    digest_columns = [
        "plan_sha256",
        "family_sha256",
        "asset_receipt_sha256",
        "asset_fingerprint_bundle_sha256",
        "sample_identity_sha256",
        "checkpoint_sha256",
        "split_sha256",
        "parameter_schema_sha256",
        "initial_state_sha256",
        "source_sha256",
        "fairness_recipe_sha256",
        "runtime_recipe_sha256",
    ]
    for column in digest_columns:
        if not frame[column].astype(str).map(_is_sha256).all():
            raise ValueError(f"Job column {column} contains a non-SHA-256 value.")
    for column in ("split_sha256", "parameter_schema_sha256", "fairness_recipe_sha256"):
        if frame[column].astype(str).nunique() != 1:
            raise ValueError(f"Job fairness binding {column} differs across the grid.")
    initial_counts = frame.groupby("seed")["initial_state_sha256"].nunique()
    if not initial_counts.eq(1).all():
        raise ValueError("The four roles must share one initial trainable state per seed.")
    seed_states = frame.groupby("seed", sort=True)["initial_state_sha256"].first()
    if seed_states.nunique() != len(spec.seeds):
        raise ValueError("Every optimization seed must produce a distinct initial state.")
    if frame["checkpoint_sha256"].nunique() != len(frame):
        raise ValueError("Every role/seed job must bind a distinct final checkpoint.")
    return frame.sort_values(["role", "seed"], kind="mergesort").reset_index(drop=True)


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected_samples: pd.DataFrame,
    jobs: pd.DataFrame,
    expected_family_sha256: str,
    spec: QueryReliabilityBundleSpec,
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "family_sha256",
        "role",
        "seed",
        "eval_cell",
        "draw_index",
        "sample_id",
        "subject_id",
        "label",
        "checkpoint_sha256",
        "evaluation_input_hash_schema",
        "evaluation_input_sha256",
        "affected_mask_hash_schema",
        "affected_mask_sha256",
        *spec.probability_columns,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}.")
    frame = predictions.loc[:, sorted(required)].copy()
    frame["seed"] = _strict_integer_series(frame["seed"], field="prediction seed")
    frame["draw_index"] = _strict_integer_series(frame["draw_index"], field="prediction draw_index")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["label"] = _strict_integer_series(frame["label"], field="prediction label")
    if set(frame["candidate_id"].astype(str)) != {"query-reliability-spatial-v1"}:
        raise ValueError("Prediction candidate binding is wrong.")
    if set(frame["family_sha256"].astype(str)) != {expected_family_sha256}:
        raise ValueError("Prediction family binding is incomplete or mixed.")
    if set(frame["evaluation_input_hash_schema"].astype(str)) != {
        QUERY_RELIABILITY_EVALUATION_INPUT_HASH_SCHEMA
    }:
        raise ValueError("Prediction evaluation-input hash schema is invalid.")
    if set(frame["affected_mask_hash_schema"].astype(str)) != {
        QUERY_RELIABILITY_AFFECTED_MASK_HASH_SCHEMA
    }:
        raise ValueError("Prediction affected-mask hash schema is invalid.")

    observed_scenarios = set(frame[["eval_cell", "draw_index"]].itertuples(index=False, name=None))
    if observed_scenarios != set(spec.evaluation_scenarios):
        raise ValueError("Prediction scenarios differ from clean plus the fixed cell/draw grid.")
    expected_groups = {
        (role, seed, cell, draw)
        for role in spec.roles
        for seed in spec.seeds
        for cell, draw in spec.evaluation_scenarios
    }
    observed_groups = set(
        frame[["role", "seed", "eval_cell", "draw_index"]].itertuples(index=False, name=None)
    )
    expected_rows = len(expected_groups) * len(expected_samples)
    if len(frame) != expected_rows or observed_groups != expected_groups:
        raise ValueError("Prediction table is not the complete role/seed/scenario grid.")
    identity_key = ["role", "seed", "eval_cell", "draw_index", "sample_id"]
    if frame.duplicated(identity_key).any():
        raise ValueError("Prediction table contains duplicate evaluation rows.")
    group_sizes = frame.groupby(identity_key[:-1], sort=False).size()
    if not group_sizes.eq(len(expected_samples)).all():
        raise ValueError("At least one prediction group omits an expected sample.")

    merged = frame.merge(
        expected_samples,
        on="sample_id",
        how="left",
        suffixes=("", "_expected"),
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Prediction table contains a sample outside the sealed manifest.")
    if (
        not merged["subject_id"].eq(merged["subject_id_expected"]).all()
        or not merged["label"].eq(merged["label_expected"]).all()
    ):
        raise ValueError("Prediction sample identity, participant, or label changed.")
    merged = merged.drop(columns=["subject_id_expected", "label_expected", "_merge"])

    checkpoint_map = jobs.set_index(["role", "seed"])["checkpoint_sha256"]
    expected_checkpoints = pd.MultiIndex.from_frame(merged[["role", "seed"]]).map(checkpoint_map)
    if not merged["checkpoint_sha256"].astype(str).eq(expected_checkpoints).all():
        raise ValueError("Prediction rows are not bound to their declared final checkpoint.")
    for column in (
        "checkpoint_sha256",
        "evaluation_input_sha256",
        "affected_mask_sha256",
    ):
        if not merged[column].astype(str).map(_is_sha256).all():
            raise ValueError(f"Prediction column {column} contains a non-SHA-256 value.")
    equality_key = ["eval_cell", "draw_index", "sample_id"]
    for column in ("evaluation_input_sha256", "affected_mask_sha256"):
        if not merged.groupby(equality_key, sort=False)[column].nunique().eq(1).all():
            raise ValueError(f"{column} differs across role/seed for the same evaluation query.")
    clean_mask_sha256 = query_reliability_affected_mask_sha256(
        np.zeros(spec.n_channels, dtype=np.bool_), clean=True
    )
    clean_rows = merged["eval_cell"].eq("clean")
    if not merged.loc[clean_rows, "affected_mask_sha256"].astype(str).eq(clean_mask_sha256).all():
        raise ValueError("Clean prediction rows do not bind the canonical all-false mask.")

    probabilities = merged.loc[:, spec.probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain NaN or infinity.")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("Prediction probabilities must lie within [0,1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Prediction probabilities must sum to one for every row.")
    return merged.sort_values(identity_key, kind="mergesort").reset_index(drop=True)


def _reduce_predictions(
    predictions: pd.DataFrame,
    *,
    jobs: pd.DataFrame,
    expected_samples: pd.DataFrame,
    bindings: QueryReliabilityBundleBindings,
    spec: QueryReliabilityBundleSpec,
) -> dict:
    verified_jobs_sha256 = _sha256_dataframe_contract(
        jobs,
        sort_by=["role", "seed"],
        schema="cfeg.query-reliability-verified-jobs.v1",
    )
    verified_predictions_sha256 = _sha256_dataframe_contract(
        predictions,
        sort_by=["role", "seed", "eval_cell", "draw_index", "sample_id"],
        schema="cfeg.query-reliability-verified-predictions.v1",
    )
    probability_columns = list(spec.probability_columns)
    ensemble_key = [
        "role",
        "eval_cell",
        "draw_index",
        "sample_id",
        "subject_id",
        "label",
    ]
    seed_counts = predictions.groupby(ensemble_key, sort=False)["seed"].nunique()
    if not seed_counts.eq(len(spec.seeds)).all():
        raise ValueError("Every ensemble query must contain all optimization seeds.")
    ensemble = (
        predictions.groupby(ensemble_key, as_index=False, sort=False)[probability_columns]
        .mean()
        .sort_values(ensemble_key, kind="mergesort")
        .reset_index(drop=True)
    )
    # NumPy argmax returns the first maximum, which freezes the lowest-class tie rule.
    ensemble["prediction"] = np.argmax(
        ensemble.loc[:, probability_columns].to_numpy(dtype=np.float64), axis=1
    )
    ensemble["correct"] = ensemble["prediction"].eq(ensemble["label"]).astype(float)

    class_accuracy = (
        ensemble.groupby(
            ["role", "eval_cell", "draw_index", "subject_id", "label"],
            as_index=False,
            sort=False,
        )["correct"]
        .mean()
        .rename(columns={"correct": "class_accuracy"})
    )
    balanced_accuracy = (
        class_accuracy.groupby(
            ["role", "eval_cell", "draw_index", "subject_id"],
            as_index=False,
            sort=False,
        )["class_accuracy"]
        .mean()
        .rename(columns={"class_accuracy": "balanced_accuracy"})
    )
    clean = balanced_accuracy.loc[balanced_accuracy["eval_cell"].eq("clean")].copy()
    corrupted = balanced_accuracy.loc[~balanced_accuracy["eval_cell"].eq("clean")].copy()
    cell_ba = (
        corrupted.groupby(["role", "eval_cell", "subject_id"], as_index=False)["balanced_accuracy"]
        .mean()
        .rename(columns={"balanced_accuracy": "draw_averaged_balanced_accuracy"})
    )
    corrupted_participant = (
        cell_ba.groupby(["role", "subject_id"], as_index=False)["draw_averaged_balanced_accuracy"]
        .mean()
        .rename(columns={"draw_averaged_balanced_accuracy": "corrupted_balanced_accuracy"})
    )

    corrupted_wide = corrupted_participant.pivot(
        index="subject_id", columns="role", values="corrupted_balanced_accuracy"
    )
    clean_wide = clean.pivot(index="subject_id", columns="role", values="balanced_accuracy")
    for role in spec.roles:
        if role not in corrupted_wide or role not in clean_wide:
            raise ValueError(f"Reduction lost the required role {role}.")
    contrasts = pd.DataFrame(index=corrupted_wide.index)
    contrasts["corrupted_q1_aug_minus_q0_aug"] = corrupted_wide["Q1_AUG"] - corrupted_wide["Q0_AUG"]
    contrasts["corrupted_q1_aug_minus_q0_clean"] = (
        corrupted_wide["Q1_AUG"] - corrupted_wide["Q0_CLEAN"]
    )
    corrupted_q_effect_clean_training = corrupted_wide["Q1_CLEAN"] - corrupted_wide["Q0_CLEAN"]
    contrasts["corrupted_q1_clean_minus_q0_clean"] = corrupted_q_effect_clean_training
    contrasts["corrupted_query_by_augmentation_interaction"] = (
        contrasts["corrupted_q1_aug_minus_q0_aug"] - corrupted_q_effect_clean_training
    )
    contrasts["clean_q1_aug_minus_q0_aug"] = clean_wide["Q1_AUG"] - clean_wide["Q0_AUG"]
    contrasts["clean_q1_aug_minus_q0_clean"] = clean_wide["Q1_AUG"] - clean_wide["Q0_CLEAN"]
    clean_q_effect_clean_training = clean_wide["Q1_CLEAN"] - clean_wide["Q0_CLEAN"]
    contrasts["clean_q1_clean_minus_q0_clean"] = clean_q_effect_clean_training
    contrasts["clean_q1_aug_minus_q1_clean"] = clean_wide["Q1_AUG"] - clean_wide["Q1_CLEAN"]
    contrasts["clean_query_by_augmentation_interaction"] = (
        contrasts["clean_q1_aug_minus_q0_aug"] - clean_q_effect_clean_training
    )
    contrasts = contrasts.reset_index()
    gates = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=contrasts["corrupted_q1_aug_minus_q0_aug"],
        clean_q1_aug_minus_q0_aug=contrasts["clean_q1_aug_minus_q0_aug"],
        clean_q1_aug_minus_q0_clean=contrasts["clean_q1_aug_minus_q0_clean"],
        expected_n=spec.n_participants,
        alpha=spec.alpha,
        practical_effect_min=spec.practical_effect_min,
        clean_noninferiority_margin=spec.clean_noninferiority_margin,
        deployment_clean_noninferiority_margin=(spec.deployment_clean_noninferiority_margin),
        positive_fraction_min=spec.positive_fraction_min,
    )

    cell_wide = cell_ba.pivot(
        index=["subject_id", "eval_cell"],
        columns="role",
        values="draw_averaged_balanced_accuracy",
    )
    cell_wide["q1_aug_minus_q0_aug"] = cell_wide["Q1_AUG"] - cell_wide["Q0_AUG"]
    cell_wide["q1_aug_minus_q0_clean"] = cell_wide["Q1_AUG"] - cell_wide["Q0_CLEAN"]
    cell_wide["query_by_augmentation_interaction"] = (cell_wide["Q1_AUG"] - cell_wide["Q0_AUG"]) - (
        cell_wide["Q1_CLEAN"] - cell_wide["Q0_CLEAN"]
    )
    cell_transitions = cell_wide.reset_index()
    cell_transitions["transition"] = np.select(
        [
            cell_transitions["q1_aug_minus_q0_aug"].ge(spec.practical_effect_min),
            cell_transitions["q1_aug_minus_q0_aug"].le(-spec.practical_effect_min),
        ],
        ["gained", "lost"],
        default="retained_within_margin",
    )

    class_cell = (
        class_accuracy.loc[~class_accuracy["eval_cell"].eq("clean")]
        .groupby(["role", "eval_cell", "label"], as_index=False)["class_accuracy"]
        .mean()
        .pivot(index=["eval_cell", "label"], columns="role", values="class_accuracy")
        .reset_index()
    )
    class_cell["q1_aug_minus_q0_aug"] = class_cell["Q1_AUG"] - class_cell["Q0_AUG"]
    class_cell["q1_aug_minus_q0_clean"] = class_cell["Q1_AUG"] - class_cell["Q0_CLEAN"]
    class_cell["query_by_augmentation_interaction"] = (
        class_cell["Q1_AUG"] - class_cell["Q0_AUG"]
    ) - (class_cell["Q1_CLEAN"] - class_cell["Q0_CLEAN"])
    class_cell["stimulus_frequency_hz"] = 8.0 + 0.2 * class_cell["label"]
    harmonic_order = 50.0 / class_cell["stimulus_frequency_hz"]
    class_cell["fifty_hz_harmonic_order"] = np.where(
        np.isclose(harmonic_order, np.rint(harmonic_order), rtol=0.0, atol=1e-12),
        np.rint(harmonic_order).astype(int),
        0,
    )
    class_cell["fifty_hz_harmonic_overlap"] = class_cell["fifty_hz_harmonic_order"].gt(0)

    worst_cell = (
        cell_ba.sort_values(
            ["role", "subject_id", "draw_averaged_balanced_accuracy", "eval_cell"],
            kind="mergesort",
        )
        .groupby(["role", "subject_id"], as_index=False, sort=False)
        .first()
        .rename(
            columns={
                "eval_cell": "worst_eval_cell",
                "draw_averaged_balanced_accuracy": "worst_cell_balanced_accuracy",
            }
        )
    )
    lower_tail = (
        corrupted_participant.groupby("role", sort=False)["corrupted_balanced_accuracy"]
        .quantile([0.10, 0.25, 0.50])
        .rename("balanced_accuracy")
        .reset_index()
        .rename(columns={"level_1": "participant_quantile"})
    )

    binding_payload = {
        **bindings.__dict__,
        "verified_jobs_sha256": verified_jobs_sha256,
        "verified_predictions_sha256": verified_predictions_sha256,
        "sample_identity": expected_samples.sort_values("sample_id").to_dict(orient="records"),
        "evaluation_scenarios": list(spec.evaluation_scenarios),
        "statistical_contract": {
            "alpha": spec.alpha,
            "practical_effect_min": spec.practical_effect_min,
            "clean_noninferiority_margin": spec.clean_noninferiority_margin,
            "deployment_clean_noninferiority_margin": (spec.deployment_clean_noninferiority_margin),
            "positive_fraction_min": spec.positive_fraction_min,
            "checkpoint_epoch": spec.checkpoint_epoch,
        },
    }
    return {
        "schema": "cfeg.query-reliability-analysis-bundle.v1",
        "candidate_id": "query-reliability-spatial-v1",
        **bindings.__dict__,
        "verified_jobs_hash_schema": "cfeg.query-reliability-verified-jobs.v1",
        "verified_jobs_sha256": verified_jobs_sha256,
        "verified_predictions_hash_schema": ("cfeg.query-reliability-verified-predictions.v1"),
        "verified_predictions_sha256": verified_predictions_sha256,
        "analysis_input_contract_sha256": _sha256_json(binding_payload),
        "seed_reduction": "mean_probabilities_per_sample_before_argmax_and_ba",
        "tie_break": "lowest_class_index",
        "participant_contrasts": contrasts,
        "participant_cell_transitions": cell_transitions,
        "participant_worst_corruption_cell": worst_cell,
        "participant_composite_lower_tail": lower_tail,
        "class_by_corruption_cell": class_cell,
        "balanced_accuracy": balanced_accuracy,
        "promotion_gates": gates,
        "atomic_publication_authorized": False,
    }


def _is_sha256(value: object) -> bool:
    return bool(_HEX_SHA256.fullmatch(str(value)))


def query_reliability_evaluation_input_sha256(
    *,
    sample_id: str,
    x: np.ndarray,
    channel_mask: np.ndarray,
    canonical_channel_ids: np.ndarray,
    channel_query_qc: np.ndarray,
    channel_query_qc_missing: np.ndarray,
    sfreq_processed: float,
) -> str:
    """Hash the complete prediction-time input domain with canonical dtypes."""

    waveform = np.ascontiguousarray(x, dtype="<f4")
    mask = np.ascontiguousarray(channel_mask, dtype=np.bool_)
    channel_ids = np.ascontiguousarray(canonical_channel_ids, dtype="<i8")
    query_qc = np.ascontiguousarray(channel_query_qc, dtype="<f4")
    query_missing = np.ascontiguousarray(channel_query_qc_missing, dtype=np.bool_)
    if waveform.ndim != 2:
        raise ValueError("Evaluation input waveform must have shape [channel,time].")
    channels = waveform.shape[0]
    if any(value.shape != (channels,) for value in (mask, channel_ids, query_qc, query_missing)):
        raise ValueError("Evaluation input channel fields must share one channel axis.")
    if (
        not str(sample_id)
        or not np.isfinite(float(sfreq_processed))
        or float(sfreq_processed) <= 0.0
    ):
        raise ValueError("Evaluation input requires a sample ID and positive sample rate.")
    if not np.isfinite(waveform[mask]).all() or not np.isfinite(query_qc).all():
        raise ValueError("Evaluation input requires finite active waveform and query QC.")

    digest = hashlib.sha256()
    _update_hash_field(
        digest,
        "schema",
        QUERY_RELIABILITY_EVALUATION_INPUT_HASH_SCHEMA.encode("utf-8"),
        dtype="utf8",
        shape=(),
    )
    _update_hash_field(
        digest,
        "sample_id",
        str(sample_id).encode("utf-8"),
        dtype="utf8",
        shape=(),
    )
    for name, value in (
        ("x", waveform),
        ("channel_mask", mask),
        ("canonical_channel_ids", channel_ids),
        ("channel_query_qc", query_qc),
        ("channel_query_qc_missing", query_missing),
        ("sfreq_processed", np.asarray([sfreq_processed], dtype="<f8")),
    ):
        _update_hash_field(
            digest,
            name,
            value.tobytes(order="C"),
            dtype=value.dtype.str,
            shape=value.shape,
        )
    return digest.hexdigest()


def query_reliability_affected_mask_sha256(
    affected_mask: np.ndarray,
    *,
    clean: bool,
) -> str:
    """Hash a canonical channel-order affected mask and enforce clean all-false."""

    mask = np.ascontiguousarray(affected_mask, dtype=np.bool_)
    if mask.ndim != 1 or not len(mask):
        raise ValueError("Affected mask must be a nonempty channel vector.")
    if clean and mask.any():
        raise ValueError("A clean evaluation row must have an all-false affected mask.")
    digest = hashlib.sha256()
    _update_hash_field(
        digest,
        "schema",
        QUERY_RELIABILITY_AFFECTED_MASK_HASH_SCHEMA.encode("utf-8"),
        dtype="utf8",
        shape=(),
    )
    _update_hash_field(
        digest,
        "affected_mask",
        mask.tobytes(order="C"),
        dtype=mask.dtype.str,
        shape=mask.shape,
    )
    return digest.hexdigest()


def _update_hash_field(
    digest,
    name: str,
    payload: bytes,
    *,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    header = json.dumps(
        {"name": name, "dtype": dtype, "shape": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    for value in (header, payload):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)


def _strict_integer_series(values: pd.Series, *, field: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain exact integers.") from exc
    array = numeric.to_numpy()
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{field} must contain exact finite integers.")
    return numeric.astype(np.int64)


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256_dataframe_contract(
    frame: pd.DataFrame,
    *,
    sort_by: list[str],
    schema: str,
) -> str:
    """Stream-hash validated table semantics with bounded auxiliary memory."""

    missing = sorted(set(sort_by) - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot hash table without ordering columns: {missing}.")
    columns = sorted(str(column) for column in frame.columns)
    ordered = frame.sort_values(sort_by, kind="mergesort").loc[:, columns]
    digest = hashlib.sha256()
    _update_length_prefixed_hash(digest, b"table-schema", schema.encode("utf-8"))
    for column in columns:
        _update_length_prefixed_hash(digest, b"column", column.encode("utf-8"))
    for row in ordered.itertuples(index=False, name=None):
        digest.update(b"\x1e")
        for value in row:
            _update_length_prefixed_hash(
                digest,
                b"scalar",
                _canonical_hash_scalar(value),
            )
    return digest.hexdigest()


def _canonical_hash_scalar(value: object) -> bytes:
    if value is None:
        canonical: list[object] = ["null", None]
    elif isinstance(value, (bool, np.bool_)):
        canonical = ["bool", bool(value)]
    elif isinstance(value, (int, np.integer)):
        canonical = ["int", int(value)]
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Cannot hash a nonfinite table value.")
        canonical = ["float64_hex", number.hex()]
    else:
        canonical = ["utf8", str(value)]
    return json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _update_length_prefixed_hash(
    digest,
    field: bytes,
    payload: bytes,
) -> None:
    digest.update(len(field).to_bytes(8, "big"))
    digest.update(field)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _sample_identity_sha256(samples: pd.DataFrame) -> str:
    return _sha256_json(
        samples.loc[:, ["sample_id", "subject_id", "label"]]
        .sort_values("sample_id", kind="mergesort")
        .to_dict(orient="records")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_git_tag_commit(repository: Path, tag: str) -> str:
    try:
        tag_type = subprocess.run(
            ["git", "cat-file", "-t", f"refs/tags/{tag}"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        if tag_type != "tag":
            raise GovernanceError(
                "The source-freeze ref must be an annotated tag, not a lightweight tag."
            )
        return subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GovernanceError("The source-freeze tag is missing or invalid.") from exc

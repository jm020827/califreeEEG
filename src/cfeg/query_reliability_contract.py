from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cfeg.data.query_reliability_corruption import (
    QUERY_RELIABILITY_CORRUPTION_CELLS_V1,
)
from cfeg.governance import GovernanceError
from cfeg.utils.config import load_config, merge_overrides

QUERY_RELIABILITY_ROLES = ("Q0_CLEAN", "Q1_CLEAN", "Q0_AUG", "Q1_AUG")
QUERY_RELIABILITY_TREATMENTS = {
    "Q0_CLEAN": ("null", False),
    "Q1_CLEAN": ("observed", False),
    "Q0_AUG": ("null", True),
    "Q1_AUG": ("observed", True),
}
QUERY_RELIABILITY_CANDIDATE_PLAN = "configs/analysis/query_reliability_spatial_v1.yaml"
QUERY_RELIABILITY_BASE_CONFIG = "configs/train/query_reliability_candidate.yaml"
QUERY_RELIABILITY_SUITE_CONFIG = "configs/train/query_reliability_suite.yaml"
QUERY_RELIABILITY_PREDECESSOR_RECEIPT = (
    "outputs/engineering/reliability-spatial-v1/stage0/receipt.json"
)
QUERY_RELIABILITY_ASSET_RECEIPT = "configs/governance/query_reliability_beta_asset_receipt.json"
QUERY_RELIABILITY_FEATURE_SCHEMA = (
    "query_window_reliability_features_v3_waveform_6_60_qc_scale_6_90"
)
QUERY_RELIABILITY_CORRUPTION_CELLS = (
    "zero_0125",
    "zero_0250",
    "white_snr_0",
    "white_snr_m5",
    "line50_snr_0",
    "line50_snr_m5",
)
QUERY_RELIABILITY_BETA_LOCKBOX_SUBJECTS = frozenset(
    {
        "sub021",
        "sub025",
        "sub026",
        "sub027",
        "sub032",
        "sub034",
        "sub039",
        "sub040",
        "sub043",
        "sub048",
        "sub049",
        "sub050",
        "sub054",
        "sub057",
        "sub058",
        "sub061",
        "sub062",
        "sub064",
        "sub069",
        "sub070",
    }
)
_PREDECESSOR_RECEIPT_SHA256 = "e8f3f87bbe47a8c8d5825f7d0525c318e87d42c6ecdd2840de21b1c657eefc35"
_PREDECESSOR_SOURCE_SNAPSHOT_SHA256 = (
    "136b1a5fdb96f2bd66c6302d6562409eb3f08918f4739d6d7d81f2321f86c3c0"
)
QUERY_RELIABILITY_ASSET_RECEIPT_SHA256 = (
    "b09a0db3f43b21a966af1b5578dfe67c9b29903cc4944127b83540c54cf6f333"
)
QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256 = (
    "06304b78dc19829e857865fe6e3f168abafdaf0c3d371763ee645ecdca617532"
)
QUERY_RELIABILITY_WRONG_QUERY_DONOR_MAPPING_SHA256 = (
    "e1d7be88b55a1ad35d975787b95183ff65909507eace648109e069164344027f"
)


def validate_query_reliability_candidate_bindings(base: dict) -> dict:
    """Validate the draft plan without granting human-outcome authority."""

    protocol = base.get("protocol", {})
    if protocol.get("candidate_id") != "query-reliability-spatial-v1":
        raise ValueError("Query-reliability candidate ID is missing or invalid.")
    plan_path = _canonical_project_path(
        protocol.get("candidate_plan"), QUERY_RELIABILITY_CANDIDATE_PLAN
    )
    plan = load_config(plan_path, strict_env=False)
    expected_plan = {
        "schema": "cfeg.query-reliability-candidate-plan.v1",
        "decision_id": "DEC-20260904-001",
        "status": "implementation_draft_outcome_blocked",
        "candidate_id": "query-reliability-spatial-v1",
        "owner_direction_approved": True,
        "exact_outcome_bundle_approved": False,
    }
    observed_plan = {key: plan.get(key) for key in expected_plan}
    if observed_plan != expected_plan:
        raise ValueError(f"Query-reliability plan header is invalid: {observed_plan}.")
    model_plan = plan.get("model_contract", {}) or {}
    expected_model_plan = {
        "architecture": "query_reliability_spatial_v1",
        "backbone": "spectral_transformer",
        "query_feature_schema": QUERY_RELIABILITY_FEATURE_SCHEMA,
        "operator_family_identifier": "diagonal_low_rank",
        "operator_structure": "symmetric_diagonal_plus_low_rank",
        "maximum_frobenius_norm": 0.20,
        "identity_initialized": True,
        "external_metadata_parameters_allowed": False,
    }
    if {key: model_plan.get(key) for key in expected_model_plan} != expected_model_plan:
        raise ValueError("The query-reliability plan/model structural contract drifted.")
    execution = plan.get("execution_authority", {})
    for forbidden in (
        "human_training_allowed",
        "validation_outcomes_allowed",
        "lockbox_outcomes_allowed",
        "dong_replication_allowed",
        "wearable_reuse_allowed",
    ):
        if execution.get(forbidden) is not False:
            raise ValueError(f"Draft query-reliability plan requires {forbidden}=false.")

    tombstone = plan.get("predecessor_tombstone", {})
    if (
        tombstone.get("receipt") != QUERY_RELIABILITY_PREDECESSOR_RECEIPT
        or tombstone.get("receipt_sha256") != _PREDECESSOR_RECEIPT_SHA256
        or tombstone.get("historical_source_snapshot_sha256") != _PREDECESSOR_SOURCE_SNAPSHOT_SHA256
        or tombstone.get("required_receipt_status") != "failed"
        or tombstone.get("grants_execution_authority") is not False
    ):
        raise ValueError("The failed predecessor tombstone is incomplete or mutable.")
    receipt_path = _canonical_project_path(
        tombstone.get("receipt"), QUERY_RELIABILITY_PREDECESSOR_RECEIPT
    )
    if _sha256_file(receipt_path) != _PREDECESSOR_RECEIPT_SHA256:
        raise ValueError("The historical failed Stage-0 receipt changed.")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if (
        receipt.get("candidate_id") != "reliability-spatial-v1"
        or receipt.get("status") != "failed"
        or receipt.get("gate", {}).get("status") != "failed"
    ):
        raise ValueError("The predecessor receipt is not the terminal failed result.")

    _validate_query_reliability_model_contract(base)
    _validate_data_contract(base, plan)
    _validate_asset_receipt_binding(base, plan)
    _validate_draft_statistical_contract(plan)
    _validate_draft_outcome_bundle_contract(plan)
    return {
        "schema": plan["schema"],
        "decision_id": plan["decision_id"],
        "status": plan["status"],
        "candidate_id": plan["candidate_id"],
        "plan_sha256": _sha256_file(plan_path),
        "predecessor_receipt_sha256": _PREDECESSOR_RECEIPT_SHA256,
        "outcome_execution_authorized": False,
    }


def validate_query_reliability_family(base: dict, suite: dict) -> str:
    """Validate exact graph parity across the query-access × augmentation 2x2."""

    candidate = validate_query_reliability_candidate_bindings(base)
    family = suite.get("query_reliability_family") or {}
    expected_family_header = {
        "schema": "cfeg.query-reliability-family.v1",
        "candidate_id": "query-reliability-spatial-v1",
        "status": "implementation_draft_outcome_blocked",
        "roles": list(QUERY_RELIABILITY_ROLES),
        "treatment_axes": [
            "query_qc_access",
            "preregistered_corruption_training",
        ],
        "primary_contrast": "Q1_AUG_minus_Q0_AUG",
        "clean_safety_contrast": "Q1_AUG_minus_Q0_AUG_on_clean_queries",
        "deployment_clean_safety_contrast": ("Q1_AUG_minus_Q0_CLEAN_on_clean_queries"),
        "seeds": [11, 29, 47],
        "all_roles_run_together": True,
        "partial_role_execution_allowed": False,
        "population_inference_allowed": False,
        "outcome_execution_authorized": False,
        "wearable_reuse_allowed": False,
    }
    observed_family = {key: family.get(key) for key in expected_family_header}
    if observed_family != expected_family_header:
        raise ValueError("Query-reliability family header differs from the exact draft contract.")
    if suite.get("base_config") != QUERY_RELIABILITY_BASE_CONFIG:
        raise ValueError("Query-reliability suite points to a noncanonical base config.")

    canonical_base = load_config(
        _repository_root() / QUERY_RELIABILITY_BASE_CONFIG, strict_env=False
    )
    if _base_contract_sha256(base) != _base_contract_sha256(canonical_base):
        raise ValueError("Query-reliability base differs from the canonical draft file.")

    variants = suite.get("variants") or {}
    if tuple(variants) != QUERY_RELIABILITY_ROLES:
        raise ValueError(f"Query-reliability variants must be exactly {QUERY_RELIABILITY_ROLES}.")
    expected_override_keys = {
        "model.conditioning.spatial_reliability.query_qc_mode",
        "augment.query_reliability_corruption.enabled",
    }
    signatures: dict[str, str] = {}
    for role in QUERY_RELIABILITY_ROLES:
        overrides = variants[role]
        if set(overrides) != expected_override_keys:
            raise ValueError(f"{role} changes fields outside the two declared treatment axes.")
        cfg = merge_overrides(
            copy.deepcopy(base),
            [f"{key}={value!r}" for key, value in overrides.items()],
        )
        _validate_query_reliability_model_contract(cfg)
        observed = _treatment(cfg)
        if observed != QUERY_RELIABILITY_TREATMENTS[role]:
            raise ValueError(
                f"Query-reliability treatment mismatch for {role}: "
                f"expected {QUERY_RELIABILITY_TREATMENTS[role]}, got {observed}."
            )
        normalized = copy.deepcopy(cfg)
        normalized["model"]["conditioning"]["spatial_reliability"]["query_qc_mode"] = (
            "<query-access>"
        )
        normalized["augment"]["query_reliability_corruption"]["enabled"] = "<corruption-training>"
        signatures[role] = _sha256_json(normalized)
    if len(set(signatures.values())) != 1:
        raise ValueError(
            f"Query-reliability arms differ outside the two declared treatment axes: {signatures}."
        )

    canonical_suite = load_config(
        _repository_root() / QUERY_RELIABILITY_SUITE_CONFIG, strict_env=False
    )
    if _sha256_json(suite) != _sha256_json(canonical_suite):
        raise ValueError("Query-reliability suite differs from the canonical draft file.")

    payload = {
        "candidate": candidate,
        "base_contract_sha256": _base_contract_sha256(base),
        "family": family,
        "variants": variants,
    }
    return _sha256_json(payload)


def validate_query_reliability_training_preflight(
    cfg: dict, *, dry_run: bool, verify_assets: bool = False
) -> str | None:
    """Allow exact dry-run graph checks and fail closed on every outcome route."""

    if not query_reliability_contract_marker_present(cfg):
        return None
    architecture = cfg.get("model", {}).get("conditioning", {}).get("architecture")
    if architecture != "query_reliability_spatial_v1":
        raise GovernanceError(
            "A query-reliability candidate marker cannot be retained while changing "
            "or removing its governed architecture."
        )
    protocol = cfg.get("protocol", {}) or {}
    data = cfg.get("data", {}) or {}
    expected_markers = {
        "candidate_id": "query-reliability-spatial-v1",
        "candidate_plan": QUERY_RELIABILITY_CANDIDATE_PLAN,
        "asset_audit_receipt": QUERY_RELIABILITY_ASSET_RECEIPT,
    }
    observed_markers = {key: protocol.get(key) for key in expected_markers}
    if observed_markers != expected_markers:
        raise ValueError(
            "Query-reliability runtime candidate ID/plan/asset markers are incomplete."
        )
    if data.get("cohort_role") != "public_beta_preregistered_partition":
        raise ValueError("Query-reliability runtime cohort marker is missing.")
    _validate_query_reliability_model_contract(cfg)

    canonical_base = load_config(
        _repository_root() / QUERY_RELIABILITY_BASE_CONFIG, strict_env=False
    )
    canonical_suite = load_config(
        _repository_root() / QUERY_RELIABILITY_SUITE_CONFIG, strict_env=False
    )
    family_sha256 = validate_query_reliability_family(canonical_base, canonical_suite)
    protocol = cfg.get("protocol", {})
    role = protocol.get("query_reliability_family_role")
    if role not in QUERY_RELIABILITY_ROLES:
        raise ValueError("query_reliability_spatial_v1 dry-runs require one exact 2x2 family role.")
    if _treatment(cfg) != QUERY_RELIABILITY_TREATMENTS[str(role)]:
        raise ValueError("Query-reliability runtime role and treatment do not match.")
    if protocol.get("query_reliability_family_sha256") != family_sha256:
        raise ValueError("Query-reliability runtime lacks the canonical family binding.")
    seed = int(cfg.get("seed", -1))
    if seed not in canonical_suite["query_reliability_family"]["seeds"]:
        raise ValueError("Query-reliability runtime uses an undeclared optimization seed.")

    expected = merge_overrides(
        copy.deepcopy(canonical_base),
        [f"{key}={value!r}" for key, value in canonical_suite["variants"][str(role)].items()],
    )
    expected["seed"] = seed
    expected["protocol"].update(
        {
            "query_reliability_family_role": role,
            "query_reliability_family_sha256": family_sha256,
        }
    )
    if _runtime_recipe_sha256(cfg) != _runtime_recipe_sha256(expected):
        raise ValueError("Query-reliability runtime recipe differs from its canonical draft arm.")
    if not dry_run:
        raise GovernanceError(
            "Query-reliability human outcomes remain blocked until the exact rendered "
            "bundle, margins, baselines, atomic runner, and owner approval are frozen."
        )
    if verify_assets:
        validate_query_reliability_asset_fingerprints(cfg)
    return family_sha256


def query_reliability_contract_marker_present(cfg: Mapping) -> bool:
    """Detect any durable marker for the blocked successor-study namespace."""

    protocol = cfg.get("protocol", {}) or {}
    data = cfg.get("data", {}) or {}
    conditioning = cfg.get("model", {}).get("conditioning", {}) or {}
    return any(
        (
            protocol.get("candidate_id") == "query-reliability-spatial-v1",
            protocol.get("candidate_plan") == QUERY_RELIABILITY_CANDIDATE_PLAN,
            protocol.get("asset_audit_receipt") == QUERY_RELIABILITY_ASSET_RECEIPT,
            data.get("cohort_role") == "public_beta_preregistered_partition",
            conditioning.get("architecture") == "query_reliability_spatial_v1",
        )
    )


def reject_query_reliability_beta_lockbox_access(
    manifest: pd.DataFrame,
    *,
    action: str,
) -> None:
    """Deny generic model/baseline outcome access to the reserved BETA subjects."""

    if not {"dataset_id", "subject_id"}.issubset(manifest.columns):
        raise ValueError("Lockbox access audit requires dataset_id and subject_id.")
    beta = manifest.loc[manifest["dataset_id"].astype(str).eq("beta")]
    overlap = sorted(set(beta["subject_id"].astype(str)) & QUERY_RELIABILITY_BETA_LOCKBOX_SUBJECTS)
    if overlap:
        raise GovernanceError(
            "The query-reliability BETA lockbox is reserved from generic outcome "
            f"access ({action}); protected participants include {overlap[:5]}"
            f"{'...' if len(overlap) > 5 else ''}. A future dedicated atomic runner "
            "requires a new exact owner-approved execution contract."
        )


def derive_query_reliability_beta_subject_partition() -> tuple[list[str], list[str]]:
    """Recompute the outcome-free deterministic S16--S70 subject partition."""

    salt = "query-reliability-spatial-v1|beta-long-window|subject"
    eligible = [f"sub{index:03d}" for index in range(16, 71)]
    ranked = sorted(
        eligible,
        key=lambda subject: (
            hashlib.sha256(f"{salt}|{subject}".encode()).hexdigest(),
            subject,
        ),
    )
    forced_training = {"sub016"}
    lockbox_candidates = [subject for subject in ranked if subject not in forced_training]
    lockbox = sorted(lockbox_candidates[-20:])
    training = sorted(set(eligible) - set(lockbox))
    if not forced_training.issubset(training):
        raise GovernanceError("The previously exposed sub016 must remain training-only.")
    return training, lockbox


def derive_query_reliability_wrong_query_donor_map(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Build the outcome-free, within-participant/class nonself donor-Q mapping."""

    required = {"sample_id", "subject_id", "label"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Wrong-query donor manifest is missing columns: {missing}.")
    frame = samples.loc[:, ["sample_id", "subject_id", "label"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    numeric_labels = pd.to_numeric(frame["label"], errors="raise")
    if (
        not np.isfinite(numeric_labels.astype(float)).all()
        or not np.equal(numeric_labels.astype(float), np.floor(numeric_labels.astype(float))).all()
    ):
        raise ValueError("Wrong-query donor labels must be exact finite integers.")
    frame["label"] = numeric_labels.astype(np.int64)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Wrong-query donor manifest contains duplicate sample IDs.")

    salt = "query-reliability-spatial-v1|wrong-query-within-participant-class|v2"

    def rank(value: str) -> tuple[str, str]:
        return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(), value

    subjects = sorted(frame["subject_id"].unique().tolist(), key=rank)
    labels = sorted(frame["label"].unique().tolist())
    counts = frame.groupby(["subject_id", "label"], sort=False).size()
    expected_pairs = len(subjects) * len(labels)
    if len(counts) != expected_pairs or counts.nunique() != 1 or int(counts.iloc[0]) < 2:
        raise ValueError(
            "Wrong-query mapping requires at least two balanced trials in every "
            "participant-by-class cell."
        )

    rows: list[dict[str, object]] = []
    for subject in subjects:
        for label in labels:
            sample_ids = sorted(
                frame.loc[
                    frame["subject_id"].eq(subject) & frame["label"].eq(label),
                    "sample_id",
                ].tolist(),
                key=rank,
            )
            donor_ids = sample_ids[1:] + sample_ids[:1]
            for target_sample, donor_sample in zip(sample_ids, donor_ids, strict=True):
                rows.append(
                    {
                        "target_sample_id": target_sample,
                        "donor_sample_id": donor_sample,
                        "target_subject_id": subject,
                        "donor_subject_id": subject,
                        "label": int(label),
                    }
                )
    mapping = (
        pd.DataFrame(rows).sort_values("target_sample_id", kind="mergesort").reset_index(drop=True)
    )
    if (
        len(mapping) != len(frame)
        or mapping["target_sample_id"].duplicated().any()
        or mapping["donor_sample_id"].duplicated().any()
        or set(mapping["target_sample_id"]) != set(frame["sample_id"])
        or set(mapping["donor_sample_id"]) != set(frame["sample_id"])
        or mapping["target_sample_id"].eq(mapping["donor_sample_id"]).any()
        or not mapping["target_subject_id"].eq(mapping["donor_subject_id"]).all()
    ):
        raise RuntimeError("Wrong-query donor mapping failed its bijection contract.")
    return mapping


def query_reliability_wrong_query_donor_mapping_sha256(
    mapping: pd.DataFrame,
) -> str:
    columns = [
        "target_sample_id",
        "donor_sample_id",
        "target_subject_id",
        "donor_subject_id",
        "label",
    ]
    if list(mapping.columns) != columns:
        raise ValueError("Wrong-query donor mapping schema or column order changed.")
    records = mapping.sort_values("target_sample_id", kind="mergesort").to_dict(orient="records")
    return _sha256_json(
        {
            "schema": "cfeg.query-reliability-wrong-query-donor-map.v2",
            "records": records,
        }
    )


def validate_query_reliability_asset_fingerprints(cfg: dict) -> str:
    """Hash-bind all six BETA files before even an outcome-free human dry-run."""

    plan_path = _canonical_project_path(
        cfg.get("protocol", {}).get("candidate_plan"),
        QUERY_RELIABILITY_CANDIDATE_PLAN,
    )
    plan = load_config(plan_path, strict_env=False)
    expected = plan["data_contract"]["primary_dataset"]["asset_fingerprints"]
    roots = list(cfg.get("data", {}).get("processed_dirs") or [])
    if len(roots) != 1 or "${env:" in str(roots[0]):
        raise GovernanceError("Set EEG_DATA_ROOT so the exact beta_v1 asset can be hash-verified.")
    root = Path(str(roots[0]))
    if not root.is_dir() or root.is_symlink() or root.name != "beta_v1":
        raise GovernanceError("The bound beta_v1 processed root is missing or unsafe.")
    observed: dict[str, str] = {}
    for name, digest in expected.items():
        path = root / str(name)
        if not path.is_file() or path.is_symlink():
            raise GovernanceError(f"Bound BETA asset file is missing or unsafe: {path}.")
        observed[str(name)] = _sha256_file(path)
        if observed[str(name)] != str(digest):
            raise GovernanceError(f"Bound BETA asset fingerprint changed: {name}.")
    _validate_beta_manifest_contract(root, plan)
    return _sha256_json(observed)


def load_query_reliability_beta_lockbox_manifest(
    processed_root: str | Path,
) -> pd.DataFrame:
    """Load the exact outcome-free sample identity bound to the draft lockbox.

    Signal bytes are hash-verified, but this function returns manifest identity
    only.  It neither reads predictions nor grants outcome execution authority.
    """

    root = Path(processed_root).expanduser().resolve()
    base = load_config(_repository_root() / QUERY_RELIABILITY_BASE_CONFIG, strict_env=False)
    base["data"]["processed_dirs"] = [str(root)]
    validate_query_reliability_asset_fingerprints(base)
    plan = load_config(_repository_root() / QUERY_RELIABILITY_CANDIDATE_PLAN, strict_env=False)
    manifest = pd.read_parquet(root / "manifest.parquet")
    samples = manifest.loc[
        manifest["subject_id"].astype(str).isin(QUERY_RELIABILITY_BETA_LOCKBOX_SUBJECTS),
        ["sample_id", "subject_id", "label"],
    ].copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["subject_id"] = samples["subject_id"].astype(str)
    samples["label"] = samples["label"].astype(int)
    samples = samples.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    primary = plan["data_contract"]["primary_dataset"]
    declared_digest = primary.get("lockbox_sample_identity_sha256")
    observed_digest = _sha256_json(samples.to_dict(orient="records"))
    if (
        len(samples) != 3200
        or set(samples["subject_id"]) != QUERY_RELIABILITY_BETA_LOCKBOX_SUBJECTS
        or samples["sample_id"].duplicated().any()
        or declared_digest != QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256
        or observed_digest != QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256
    ):
        raise GovernanceError("The canonical BETA lockbox sample-identity binding changed.")
    return samples


def _validate_beta_manifest_contract(root: Path, plan: Mapping) -> None:
    """Check identity/balance only; never compute or inspect model outcomes."""

    manifest = pd.read_parquet(root / "manifest.parquet")
    required = {
        "sample_id",
        "h5_index",
        "dataset_id",
        "subject_id",
        "label",
        "sfreq_processed",
        "window_start_sec",
        "window_duration_sec",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise GovernanceError(f"Bound BETA manifest is missing columns: {missing}.")
    if len(manifest) != 11200 or manifest["sample_id"].astype(str).duplicated().any():
        raise GovernanceError("Bound BETA manifest row/sample identity contract failed.")
    h5_indices = manifest["h5_index"].astype(int).to_numpy()
    if not np.array_equal(h5_indices, np.arange(11200)):
        raise GovernanceError("Bound BETA h5_index must be unique, contiguous, and ordered.")
    if set(manifest["dataset_id"].astype(str)) != {"beta"}:
        raise GovernanceError("Bound BETA manifest contains another dataset identity.")
    observed_subjects = set(manifest["subject_id"].astype(str))
    expected_subjects = {f"sub{index:03d}" for index in range(1, 71)}
    if observed_subjects != expected_subjects:
        raise GovernanceError("Bound BETA manifest subject set changed.")
    if set(manifest["label"].astype(int)) != set(range(40)):
        raise GovernanceError("Bound BETA manifest no longer has exact labels 0..39.")
    counts = manifest.groupby(
        [manifest["subject_id"].astype(str), manifest["label"].astype(int)]
    ).size()
    if len(counts) != 70 * 40 or not counts.eq(4).all():
        raise GovernanceError(
            "Every bound BETA participant must have exactly four trials for every class."
        )
    if (
        not manifest["sfreq_processed"].astype(float).eq(200.0).all()
        or not manifest["window_start_sec"].astype(float).eq(0.63).all()
        or not manifest["window_duration_sec"].astype(float).eq(2.0).all()
    ):
        raise GovernanceError("Bound BETA time-grid contract changed.")
    if (
        not manifest["n_channels_used"].astype(int).eq(64).all()
        or not manifest["query_signal_std"].astype(float).map(np.isfinite).all()
        or not manifest["query_signal_std"].astype(float).gt(0.0).all()
    ):
        raise GovernanceError("Bound BETA active-channel or query-QC contract changed.")
    for value in manifest["query_signal_std_by_channel"]:
        channel_qc = np.asarray(value, dtype=np.float64)
        if (
            channel_qc.shape != (64,)
            or not np.isfinite(channel_qc).all()
            or not (channel_qc > 0.0).all()
        ):
            raise GovernanceError("Bound BETA channel-QC vector is invalid.")
    _validate_beta_jsonl_parquet_parity(root, manifest)
    _validate_beta_hdf_contract(root, manifest)
    primary = plan["data_contract"]["primary_dataset"]
    included = set(primary["training_subject_ids"]) | set(primary["lockbox_subject_ids"])
    excluded = set(primary["excluded_subject_ids"])
    if (
        len(included) != 55
        or len(excluded) != 15
        or included & excluded
        or included | excluded != expected_subjects
    ):
        raise GovernanceError("BETA eligibility roles no longer cover S1-S70 exactly.")
    lockbox_samples = manifest.loc[
        manifest["subject_id"].astype(str).isin(QUERY_RELIABILITY_BETA_LOCKBOX_SUBJECTS),
        ["sample_id", "subject_id", "label"],
    ]
    donor_mapping = derive_query_reliability_wrong_query_donor_map(lockbox_samples)
    if query_reliability_wrong_query_donor_mapping_sha256(donor_mapping) != primary.get(
        "wrong_query_donor_mapping_sha256"
    ):
        raise GovernanceError("The BETA wrong-query donor mapping binding changed.")


def _validate_beta_jsonl_parquet_parity(root: Path, manifest: pd.DataFrame) -> None:
    jsonl_path = root / "manifest.jsonl"
    with jsonl_path.open("r", encoding="utf-8") as handle:
        json_rows = [json.loads(line) for line in handle if line.strip()]
    if len(json_rows) != len(manifest):
        raise GovernanceError("BETA JSONL and Parquet manifest row counts differ.")
    columns = list(manifest.columns)
    if any(set(row) != set(columns) for row in json_rows):
        raise GovernanceError("BETA JSONL and Parquet manifest schemas differ.")
    for index, (json_row, parquet_row) in enumerate(
        zip(json_rows, manifest.to_dict(orient="records"), strict=True)
    ):
        for column in columns:
            if not _semantically_equal_manifest_value(json_row[column], parquet_row[column]):
                raise GovernanceError(
                    f"BETA JSONL and Parquet manifests differ at row {index}, column {column}."
                )


def _semantically_equal_manifest_value(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(right, (list, tuple, np.ndarray)):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape != right_array.shape:
            return False
        if left_array.dtype.kind in "f" or right_array.dtype.kind in "f":
            return bool(
                np.allclose(
                    left_array.astype(float),
                    right_array.astype(float),
                    rtol=1e-7,
                    atol=1e-9,
                    equal_nan=True,
                )
            )
        return bool(np.array_equal(left_array, right_array))
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=1e-7, atol=1e-9))
    return left == right


def _validate_beta_hdf_contract(root: Path, manifest: pd.DataFrame) -> None:
    with h5py.File(root / "signals.h5", "r") as handle:
        expected = {
            "x": ((11200, 64, 400), np.dtype("float32")),
            "channel_mask": ((11200, 64), np.dtype("bool")),
            "y": ((11200,), np.dtype("int64")),
        }
        if set(handle.keys()) != set(expected):
            raise GovernanceError("Bound BETA HDF5 dataset keys changed.")
        for name, (shape, dtype) in expected.items():
            if handle[name].shape != shape or handle[name].dtype != dtype:
                raise GovernanceError(f"Bound BETA HDF5 {name} shape or dtype changed.")
        labels = manifest.sort_values("h5_index")["label"].astype(int).to_numpy()
        if not np.array_equal(handle["y"][:], labels):
            raise GovernanceError("Bound BETA HDF5 labels differ from the manifest.")
        if not bool(handle["channel_mask"][:].all()):
            raise GovernanceError("Bound BETA HDF5 no longer has 64 active channels.")


def _validate_query_reliability_model_contract(cfg: dict) -> None:
    protocol = cfg.get("protocol", {})
    model = cfg.get("model", {})
    conditioning = model.get("conditioning", {})
    spatial = conditioning.get("spatial_reliability", {})
    encoder = model.get("condition_encoder", {})
    expected = {
        "metadata_contract_version": "0.4-dev",
        "feature_schema": QUERY_RELIABILITY_FEATURE_SCHEMA,
        "phase_features_allowed": False,
        "architecture": "query_reliability_spatial_v1",
        "backbone": "spectral_transformer",
        "placement": "pre_backbone_waveform",
        "operator_family": "diagonal_low_rank",
        "operator_rank": 4,
        "max_operator_norm": 0.20,
        "n_prompt_tokens": 0,
        "fields": [],
        "include_continuous": False,
        "include_channels": False,
        "force_missing": False,
        "external_metadata_mode": "null",
        "adapter_enabled": False,
        "latent_enabled": False,
    }
    observed = {
        "metadata_contract_version": protocol.get("metadata_contract_version"),
        "feature_schema": protocol.get("reliability_query_feature_schema"),
        "phase_features_allowed": protocol.get("phase_features_allowed"),
        "architecture": conditioning.get("architecture"),
        "backbone": model.get("backbone", {}).get("name"),
        "placement": spatial.get("placement"),
        "operator_family": spatial.get("operator_family"),
        "operator_rank": int(spatial.get("operator_rank", 0)),
        "max_operator_norm": float(spatial.get("max_operator_norm", -1.0)),
        "n_prompt_tokens": int(encoder.get("n_prompt_tokens", -1)),
        "fields": list(encoder.get("fields") or []),
        "include_continuous": bool(encoder.get("include_continuous", False)),
        "include_channels": bool(encoder.get("include_channels", False)),
        "force_missing": bool(encoder.get("force_missing", False)),
        "external_metadata_mode": encoder.get("external_metadata_mode"),
        "adapter_enabled": bool(model.get("adapter", {}).get("enabled", True)),
        "latent_enabled": bool(model.get("latent", {}).get("enabled", True)),
    }
    if observed != expected:
        raise ValueError(
            f"Query-reliability model contract mismatch: expected={expected}, observed={observed}."
        )
    forbidden_spatial = {
        "metadata_operator_norm",
        "metadata_alpha_limit",
        "metadata_residual_enabled",
    }.intersection(spatial)
    if forbidden_spatial:
        raise ValueError(
            f"Query-only spatial config contains metadata branch keys: {sorted(forbidden_spatial)}."
        )

    augment = cfg.get("augment", {})
    if bool(augment.get("make_two_views", False)):
        raise ValueError("Query-reliability study forbids legacy two-view augmentation.")
    corruption = augment.get("query_reliability_corruption", {})
    if tuple(corruption.get("cell_ids") or ()) != QUERY_RELIABILITY_CORRUPTION_CELLS:
        raise ValueError("Query-reliability corruption cell order/content is not canonical.")
    if set(QUERY_RELIABILITY_CORRUPTION_CELLS) != set(QUERY_RELIABILITY_CORRUPTION_CELLS_V1):
        raise RuntimeError("Code/config query-corruption registries disagree.")
    if int(corruption.get("seed", -1)) != 20260904:
        raise ValueError("Query-reliability corruption seed is not frozen.")
    if float(corruption.get("clean_probability", -1.0)) != 0.25:
        raise ValueError("Query-reliability augmented training must retain 25% clean queries.")


def _validate_data_contract(base: dict, plan: Mapping) -> None:
    data = base.get("data", {})
    roots = list(data.get("processed_dirs") or [])
    if len(roots) != 1 or Path(str(roots[0])).name != "beta_v1":
        raise ValueError("Query-reliability primary dry-run may access only beta_v1.")
    if data.get("expected_revisions") not in (None, {}):
        raise ValueError(
            "The current BETA asset has no revision field; bind its six file hashes instead."
        )
    if data.get("split") != "fixed_subject_partition":
        raise ValueError("Query-reliability primary requires its fixed subject partition.")
    primary = plan.get("data_contract", {}).get("primary_dataset", {})
    roles = {
        "training": sorted(str(value) for value in data.get("training_subject_ids", [])),
        "validation": sorted(str(value) for value in data.get("validation_subject_ids", [])),
        "test": sorted(str(value) for value in data.get("test_subject_ids", [])),
        "excluded": sorted(str(value) for value in data.get("excluded_subject_ids", [])),
    }
    expected_roles = {
        "training": sorted(primary.get("training_subject_ids") or []),
        "validation": sorted(primary.get("validation_subject_ids") or []),
        "test": sorted(primary.get("lockbox_subject_ids") or []),
        "excluded": sorted(primary.get("excluded_subject_ids") or []),
    }
    if roles != expected_roles:
        raise ValueError("Base config subject roles differ from the candidate plan.")
    if set(roles["test"]) != set(QUERY_RELIABILITY_BETA_LOCKBOX_SUBJECTS):
        raise ValueError("Base config no longer uses the globally reserved BETA lockbox.")
    derived_training, derived_lockbox = derive_query_reliability_beta_subject_partition()
    if roles["training"] != derived_training or roles["test"] != derived_lockbox:
        raise ValueError("The BETA subject partition does not match its SHA-256 rank rule.")
    if (
        primary.get("split_salt") != "query-reliability-spatial-v1|beta-long-window|subject"
        or primary.get("split_rank_serialization") != "sha256_utf8_of_split_salt_pipe_subject_id"
        or primary.get("split_rank_order") != "ascending_digest_then_subject_id"
        or primary.get("lockbox_rank_selection")
        != "last_20_largest_digest_after_removing_forced_training_subjects"
        or primary.get("exposed_subject_constraint")
        != "sub016_must_remain_training_or_split_is_invalid"
    ):
        raise ValueError("The BETA subject-partition serialization rule drifted.")
    if (
        primary.get("lockbox_n_samples") != 3200
        or primary.get("lockbox_n_subjects") != 20
        or primary.get("n_channels") != 64
        or primary.get("trials_per_participant_class") != 4
        or primary.get("lockbox_sample_identity_sha256")
        != QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256
    ):
        raise ValueError("The exact BETA lockbox sample-identity contract drifted.")
    if (
        primary.get("wrong_query_donor_mapping_schema")
        != "cfeg.query-reliability-wrong-query-donor-map.v2"
        or primary.get("wrong_query_donor_mapping_salt")
        != "query-reliability-spatial-v1|wrong-query-within-participant-class|v2"
        or primary.get("wrong_query_donor_mapping_rule")
        != "sha256_within_participant_class_nonself_sample_ring"
        or primary.get("wrong_query_donor_mapping_sha256")
        != QUERY_RELIABILITY_WRONG_QUERY_DONOR_MAPPING_SHA256
    ):
        raise ValueError("The exact wrong-query donor mapping contract drifted.")
    if _sha256_json(roles) != primary.get("split_assignment_sha256"):
        raise ValueError("Query-reliability subject partition digest is stale.")
    declared = [value for values in roles.values() for value in values]
    if len(declared) != 70 or len(set(declared)) != 70:
        raise ValueError(
            "Query-reliability BETA roles plus exclusions must contain 70 unique subjects."
        )
    if (
        len(roles["training"]) != 35
        or len(roles["validation"]) != 0
        or len(roles["test"]) != 20
        or len(roles["excluded"]) != 15
    ):
        raise ValueError("Query-reliability BETA role counts are not 35/0/20 plus 15 excluded.")
    if data.get("allow_empty_validation") is not True:
        raise ValueError("Query-reliability training must have no validation selection channel.")
    if base.get("train", {}).get("checkpoint_selection") != "fixed_epoch_blind":
        raise ValueError("Query-reliability checkpoint must be the fixed final epoch.")
    exposed = set(primary.get("previously_outcome_exposed_subjects") or [])
    if not exposed or not exposed.issubset(set(roles["training"])):
        raise ValueError("Previously outcome-exposed BETA subjects must be training-only.")
    if plan.get("data_contract", {}).get("wearable_v3_allowed") is not False:
        raise ValueError("The successor plan must keep wearable_v3 sealed.")
    if plan.get("data_contract", {}).get("synthetic_quality_v1_allowed") is not False:
        raise ValueError("The successor may not reuse the failed synthetic-quality outcome asset.")
    replication = plan.get("data_contract", {}).get("independent_replication", {})
    if (
        replication.get("status") != "planned_not_authorized"
        or replication.get("replication_mode")
        != "within_corpus_retraining_with_an_independent_preregistered_partition"
        or replication.get("beta_to_dong_zero_shot_transfer_claimed") is not False
    ):
        raise ValueError(
            "Dong must remain an unauthorized within-corpus retraining replication, "
            "not a BETA-to-Dong transfer claim."
        )
    if (
        plan.get("data_contract", {}).get("manifest_stimulus_phase_template_or_claim_allowed")
        is not False
        or plan.get("data_contract", {}).get("empirical_phase_in_current_waveform_allowed")
        is not True
    ):
        raise ValueError(
            "Manifest stimulus-phase templates must stay blocked while empirical "
            "waveform phase remains an allowed signal input."
        )


def _validate_asset_receipt_binding(base: dict, plan: Mapping) -> None:
    protocol = base.get("protocol", {})
    primary = plan.get("data_contract", {}).get("primary_dataset", {})
    if (
        protocol.get("asset_audit_receipt") != QUERY_RELIABILITY_ASSET_RECEIPT
        or protocol.get("asset_audit_receipt_sha256") != QUERY_RELIABILITY_ASSET_RECEIPT_SHA256
        or primary.get("audit_receipt") != QUERY_RELIABILITY_ASSET_RECEIPT
        or primary.get("audit_receipt_sha256") != QUERY_RELIABILITY_ASSET_RECEIPT_SHA256
    ):
        raise ValueError("Query-reliability BETA asset-audit binding is incomplete.")
    receipt_path = _canonical_project_path(
        QUERY_RELIABILITY_ASSET_RECEIPT, QUERY_RELIABILITY_ASSET_RECEIPT
    )
    if _sha256_file(receipt_path) != QUERY_RELIABILITY_ASSET_RECEIPT_SHA256:
        raise ValueError("Query-reliability BETA asset-audit receipt changed.")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    expected_observed_contract = {
        "n_rows": 11200,
        "n_subjects": 70,
        "n_classes": 40,
        "rows_per_subject": 160,
        "signals_shape": [11200, 64, 400],
        "signals_dtype": "float32",
        "channel_mask_shape": [11200, 64],
        "label_shape": [11200],
    }
    if (
        receipt.get("schema") != "cfeg.query-reliability-asset-audit.v1"
        or receipt.get("candidate_id") != "query-reliability-spatial-v1"
        or receipt.get("dataset_id") != "beta"
        or receipt.get("status") != "verified_frequency_only_phase_blocked"
        or receipt.get("files") != primary.get("asset_fingerprints")
        or receipt.get("observed_contract") != expected_observed_contract
        or receipt.get("phase_provenance", {}).get(
            "manifest_stimulus_phase_template_or_claim_allowed"
        )
        is not False
        or receipt.get("grants_human_outcome_execution_authority") is not False
    ):
        raise ValueError("Query-reliability BETA asset-audit receipt is invalid.")


def _validate_draft_statistical_contract(plan: Mapping) -> None:
    estimands = plan.get("estimands", {}) or {}
    gates = estimands.get("provisional_gates_not_yet_frozen", {}) or {}
    expected = {
        "primary_mean_delta_min": 0.02,
        "primary_one_sided_alpha": 0.05,
        "primary_positive_participant_fraction_min": 0.60,
        "clean_noninferiority_margin": -0.01,
        "clean_one_sided_alpha": 0.05,
        "deployment_clean_noninferiority_margin": -0.01,
        "deployment_clean_one_sided_alpha": 0.05,
    }
    observed = {key: gates.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            "Query-reliability draft statistical values drifted from their tested contract."
        )
    if estimands.get("statistical_unit") != "participant":
        raise ValueError("Query-reliability inference must remain participant-level.")
    if estimands.get("seed_reduction") != (
        "average_three_seed_probabilities_before_participant_metric"
    ):
        raise ValueError("Query-reliability seed reduction order changed.")
    if estimands.get("corruption_generalization_claim_allowed") is not False:
        raise ValueError(
            "Seen-corruption evaluation cannot be labeled corruption-OOD generalization."
        )
    expected_descriptive = {
        "clean_query_effect_without_augmentation": "Q1_CLEAN_minus_Q0_CLEAN",
        "corrupted_query_effect_without_augmentation": "Q1_CLEAN_minus_Q0_CLEAN",
        "clean_query_by_augmentation_interaction": (
            "(Q1_AUG_minus_Q0_AUG)_clean_minus_(Q1_CLEAN_minus_Q0_CLEAN)_clean"
        ),
        "corrupted_query_by_augmentation_interaction": (
            "(Q1_AUG_minus_Q0_AUG)_corrupted_minus_(Q1_CLEAN_minus_Q0_CLEAN)_corrupted"
        ),
        "corrupted_deployment_comparison": "Q1_AUG_minus_Q0_CLEAN",
    }
    experiment = plan.get("experiment", {}) or {}
    if (
        experiment.get("required_descriptive_contrasts") != expected_descriptive
        or experiment.get("descriptive_contrasts_are_promotion_tests") is not False
        or experiment.get("corrupted_deployment_comparison_gate_status")
        != "utility_threshold_not_yet_frozen"
    ):
        raise ValueError("The required 2x2 descriptive contrasts drifted.")
    wrong_query = plan.get("experiment", {}).get("same_checkpoint_wrong_query_audit", {}) or {}
    expected_wrong_query = {
        "checkpoint_role": "Q1_AUG",
        "information_rights_status": "analysis_only_counterfactual_non_deployable",
        "target_scenarios": "clean_plus_all_six_corruption_cells_and_three_draws",
        "donor_scenario": "same_eval_cell_and_draw_as_target",
        "pairing": ("deterministic_bijective_within_participant_and_class_nonself_trial"),
        "label_use_boundary": ("offline_pair_construction_only_never_model_input_or_deployment"),
        "uses_other_trial_from_same_target_participant": True,
        "strict_k0_deployment_rights_satisfied": False,
        "target_waveform_route": "target_waveform_enters_backbone",
        "donor_query_route": (
            "donor_waveform_and_donor_channel_qc_construct_spatial_operator_only"
        ),
        "channel_layout_mask_and_sampling_rate_must_match": True,
        "training_use_allowed": False,
        "excluded_from_primary_clean_and_deployment_estimands": True,
        "generated_only_inside_same_hidden_atomic_transaction_after_standard_predictions": True,
        "presentation_as_deployment_comparator_allowed": False,
    }
    if {key: wrong_query.get(key) for key in expected_wrong_query} != expected_wrong_query:
        raise ValueError("The same-checkpoint wrong-query audit contract drifted.")


def _validate_draft_outcome_bundle_contract(plan: Mapping) -> None:
    execution = plan.get("execution_authority", {}) or {}
    bundle = plan.get("outcome_bundle_contract", {}) or {}
    expected = {
        "status": "draft_not_execution_authority",
        "canonical_lockbox_n_subjects": 20,
        "canonical_lockbox_n_samples": 3200,
        "canonical_lockbox_sample_identity_sha256": (
            QUERY_RELIABILITY_LOCKBOX_SAMPLE_IDENTITY_SHA256
        ),
        "canonical_asset_fingerprint_bundle_sha256": (
            "43ecb4936ec3abaf2f92e602142a94e93a7c3e8cb459cf3d11ad6358d379cb5c"
        ),
        "canonical_runtime_split_sha256": (
            "d0afab8d17be77f80b3666722dc71036569b93b9370af6bbfeab9e0e7f14570c"
        ),
        "plan_family_source_asset_and_sample_bindings_must_be_internal": True,
        "private_staging_and_atomic_publication_required": True,
        "source_freeze_tag_must_be_annotated": True,
        "same_checkpoint_Q1_identity_intervention_artifact_required": True,
        "same_checkpoint_Q1_wrong_query_intervention_artifact_required": True,
        "feature_probe_and_ablation_artifacts_required_for_reliability_mechanism_claim": True,
        "evaluation_input_hash_schema": ("cfeg.query-reliability-evaluation-input-sha256.v1"),
        "affected_mask_hash_schema": ("cfeg.query-reliability-affected-mask-sha256.v1"),
        "clean_affected_mask_must_be_all_false": True,
        "verified_jobs_hash_schema": "cfeg.query-reliability-verified-jobs.v1",
        "verified_predictions_hash_schema": ("cfeg.query-reliability-verified-predictions.v1"),
        "verified_table_hash_serialization": (
            "schema_and_sorted_columns_then_length_prefixed_canonical_scalar_records_v1"
        ),
        "future_runner_must_bind_staged_file_hashes_in_receipt": True,
        "grants_human_outcome_execution_authority": False,
    }
    observed = {key: bundle.get(key) for key in expected}
    if observed != expected or execution.get("source_freeze_tag") is not None:
        raise ValueError(
            "The draft outcome bundle must remain internally bound and source-unfrozen."
        )
    expected_input_domain = [
        "sample_id_utf8",
        "canonical_float32_corrupted_waveform_with_dtype_and_shape",
        "canonical_bool_channel_mask_with_dtype_and_shape",
        "canonical_int64_channel_ids_with_dtype_and_shape",
        "canonical_float32_channel_query_qc_with_dtype_and_shape",
        "canonical_bool_channel_query_qc_missing_with_dtype_and_shape",
        "canonical_float64_sampling_rate_with_dtype_and_shape",
    ]
    if (
        bundle.get("evaluation_input_hash_domain") != expected_input_domain
        or bundle.get("affected_mask_hash_domain")
        != "canonical_bool_channel_vector_with_dtype_and_shape"
    ):
        raise ValueError("The evaluation input and affected-mask hash domains drifted.")
    audits = set(plan.get("required_mechanism_and_shortcut_audits") or [])
    mechanism = set(
        (plan.get("mechanism_attribution_boundary") or {}).get(
            "reliability_mechanism_claim_requires"
        )
        or []
    )
    if (
        "same_checkpoint_Q1_identity_intervention" not in audits
        or "same_checkpoint_Q1_identity_intervention_pass" not in mechanism
        or "same_checkpoint_Q1_wrong_query_intervention" not in audits
        or "same_checkpoint_Q1_wrong_query_intervention_pass" not in mechanism
    ):
        raise ValueError("The Q1 same-checkpoint interventions are not fully bound.")


def _treatment(cfg: Mapping) -> tuple[str, bool]:
    return (
        str(cfg["model"]["conditioning"]["spatial_reliability"]["query_qc_mode"]),
        bool(cfg["augment"]["query_reliability_corruption"].get("enabled", False)),
    )


def _base_contract_sha256(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.pop("runtime_contract", None)
    return _sha256_json(normalized)


def _runtime_recipe_sha256(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.pop("runtime_contract", None)
    normalized.setdefault("augment", {}).pop("channel_subset_ids", None)
    normalized["run_name"] = "<orchestrated-role>"
    normalized["output_dir"] = "<orchestrated-output>"
    return _sha256_json(normalized)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_project_path(value: object, expected: str) -> Path:
    if value != expected:
        raise ValueError(f"Expected canonical project path {expected!r}, got {value!r}.")
    candidate = _repository_root() / expected
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.resolve() != candidate.absolute()
    ):
        raise ValueError(f"Required query-reliability file is missing or unsafe: {expected}.")
    return candidate


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

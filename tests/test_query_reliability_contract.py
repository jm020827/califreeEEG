from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from cfeg import eval_loop, train_loop
from cfeg.data.splits import make_fixed_subject_partition
from cfeg.governance import GovernanceError
from cfeg.query_reliability_contract import (
    QUERY_RELIABILITY_ROLES,
    derive_query_reliability_beta_subject_partition,
    derive_query_reliability_wrong_query_donor_map,
    query_reliability_wrong_query_donor_mapping_sha256,
    reject_query_reliability_beta_lockbox_access,
    validate_query_reliability_candidate_bindings,
    validate_query_reliability_family,
    validate_query_reliability_training_preflight,
)
from cfeg.train_loop import run_training
from cfeg.utils.config import load_config, merge_overrides

REPO = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    return (
        load_config(REPO / "configs/train/query_reliability_candidate.yaml"),
        load_config(REPO / "configs/train/query_reliability_suite.yaml"),
    )


def _runtime(role: str = "Q1_AUG", seed: int = 11) -> dict:
    base, suite = _configs()
    digest = validate_query_reliability_family(base, suite)
    cfg = merge_overrides(
        base,
        [f"{key}={value!r}" for key, value in suite["variants"][role].items()],
    )
    cfg["seed"] = seed
    cfg["protocol"].update(
        {
            "query_reliability_family_role": role,
            "query_reliability_family_sha256": digest,
        }
    )
    return cfg


def test_draft_family_is_exact_outcome_blocked_2x2() -> None:
    base, suite = _configs()
    binding = validate_query_reliability_candidate_bindings(base)
    digest = validate_query_reliability_family(base, suite)

    assert binding["status"] == "implementation_draft_outcome_blocked"
    assert binding["outcome_execution_authorized"] is False
    assert len(digest) == 64
    assert tuple(suite["variants"]) == QUERY_RELIABILITY_ROLES
    assert validate_query_reliability_training_preflight(_runtime(), dry_run=True) == digest


def test_family_rejects_metadata_or_recipe_drift() -> None:
    base, suite = _configs()
    metadata = copy.deepcopy(base)
    metadata["model"]["condition_encoder"]["external_metadata_mode"] = "observed"
    with pytest.raises(ValueError, match="model contract mismatch"):
        validate_query_reliability_candidate_bindings(metadata)

    extra_axis = copy.deepcopy(suite)
    extra_axis["variants"]["Q1_AUG"]["train.lr"] = 0.01
    with pytest.raises(ValueError, match="outside the two declared treatment axes"):
        validate_query_reliability_family(base, extra_axis)

    old_asset = copy.deepcopy(base)
    old_asset["data"]["processed_dirs"] = ["data/processed/synthetic_quality_v1"]
    with pytest.raises(ValueError, match="only beta_v1"):
        validate_query_reliability_candidate_bindings(old_asset)


def test_every_non_dry_human_training_route_fails_before_dataset_access() -> None:
    cfg = _runtime("Q0_CLEAN")
    with pytest.raises(GovernanceError, match="outcome training is blocked"):
        run_training(cfg, dry_run=False)
    with pytest.raises(GovernanceError, match="human outcomes remain blocked"):
        validate_query_reliability_training_preflight(cfg, dry_run=False)

    missing_role, _suite = _configs()
    with pytest.raises(ValueError, match="exact 2x2 family role"):
        validate_query_reliability_training_preflight(missing_role, dry_run=True)

    changed_architecture = _runtime("Q0_CLEAN")
    changed_architecture["model"]["conditioning"]["architecture"] = "physical_hybrid_v1"
    with pytest.raises(GovernanceError, match="outcome training is blocked"):
        run_training(changed_architecture, dry_run=False)
    with pytest.raises(GovernanceError, match="cannot be retained"):
        validate_query_reliability_training_preflight(changed_architecture, dry_run=True)

    erased_candidate = _runtime("Q0_CLEAN")
    erased_candidate["protocol"]["candidate_id"] = "erased"
    with pytest.raises(ValueError, match="candidate ID"):
        validate_query_reliability_training_preflight(erased_candidate, dry_run=True)


def test_runtime_rejects_role_seed_and_recipe_drift() -> None:
    wrong_role = _runtime("Q1_AUG")
    wrong_role["protocol"]["query_reliability_family_role"] = "Q0_AUG"
    with pytest.raises(ValueError, match="role and treatment"):
        validate_query_reliability_training_preflight(wrong_role, dry_run=True)

    wrong_seed = _runtime("Q1_AUG")
    wrong_seed["seed"] = 999
    with pytest.raises(ValueError, match="undeclared optimization seed"):
        validate_query_reliability_training_preflight(wrong_seed, dry_run=True)

    drift = _runtime("Q1_AUG")
    drift["train"]["epochs"] = 11
    with pytest.raises(ValueError, match="runtime recipe differs"):
        validate_query_reliability_training_preflight(drift, dry_run=True)


def test_beta_lockbox_overlay_blocks_generic_outcomes_but_not_training_subjects() -> None:
    safe = pd.DataFrame({"dataset_id": ["beta", "wang"], "subject_id": ["sub016", "sub021"]})
    reject_query_reliability_beta_lockbox_access(safe, action="unit_safe")

    protected = pd.DataFrame({"dataset_id": ["beta", "beta"], "subject_id": ["sub016", "sub021"]})
    with pytest.raises(GovernanceError, match="BETA lockbox is reserved"):
        reject_query_reliability_beta_lockbox_access(protected, action="generic_baseline")


def test_generic_whole_cohort_preload_is_blocked_before_dataset_construction(
    monkeypatch,
) -> None:
    cfg = load_config(REPO / "configs/train/debug.yaml")
    cfg["data"]["processed_dirs"] = ["/unused/beta_v1"]
    cfg["data"]["preload_hdf5_to_memory"] = True
    cfg["runtime"] = {"device": "cpu"}
    manifest = pd.DataFrame(
        {
            "sample_id": ["train", "locked"],
            "dataset_id": ["beta", "beta"],
            "subject_id": ["sub016", "sub021"],
        }
    )
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("lockbox signal dataset must not be constructed")

    monkeypatch.setattr(train_loop, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(train_loop, "EEGProcessedDataset", forbidden_dataset)
    with pytest.raises(GovernanceError, match="whole_cohort_preload_before_split"):
        train_loop.run_training(cfg, dry_run=True)
    assert constructed is False


def test_generic_training_split_gate_runs_before_any_sample_read(monkeypatch) -> None:
    cfg = load_config(REPO / "configs/train/debug.yaml")
    cfg["data"].update(
        {
            "processed_dirs": ["/unused/beta_v1"],
            "split": "fixed_subject_partition",
            "training_subject_ids": ["sub021"],
            "validation_subject_ids": [],
            "test_subject_ids": ["sub016"],
            "excluded_subject_ids": [],
            "allow_empty_validation": True,
            "preload_hdf5_to_memory": False,
        }
    )
    cfg["runtime"] = {"device": "cpu"}
    manifest = pd.DataFrame(
        {
            "sample_id": ["locked", "safe"],
            "dataset_id": ["beta", "beta"],
            "subject_id": ["sub021", "sub016"],
            "label": [0, 0],
        }
    )
    sample_read = False

    class ManifestOnlyDataset:
        def __init__(self, *args, **kwargs):
            self.entries = [
                (Path("/unused"), index, row)
                for index, row in enumerate(manifest.to_dict(orient="records"))
            ]

        def __getitem__(self, index):
            nonlocal sample_read
            sample_read = True
            raise AssertionError("signal sample must remain unread")

        def __len__(self):
            return len(self.entries)

    monkeypatch.setattr(train_loop, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(train_loop, "EEGProcessedDataset", ManifestOnlyDataset)
    with pytest.raises(GovernanceError, match="training_or_checkpoint_selection"):
        train_loop.run_training(cfg, dry_run=True)
    assert sample_read is False


def test_generic_evaluation_blocks_query_checkpoint_and_beta_lockbox_pre_dataset(
    monkeypatch,
) -> None:
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("evaluation dataset must remain sealed")

    monkeypatch.setattr(eval_loop, "EEGProcessedDataset", forbidden_dataset)
    monkeypatch.setattr(
        eval_loop, "load_checkpoint", lambda *args, **kwargs: {"config": _runtime("Q1_AUG")}
    )
    with pytest.raises(GovernanceError, match="checkpoint outcomes are blocked"):
        eval_loop.load_evaluation_context({}, "unused.pt")
    assert constructed is False

    generic = load_config(REPO / "configs/train/debug.yaml")
    generic["runtime"] = {"device": "cpu"}
    manifest = pd.DataFrame(
        {
            "sample_id": ["locked"],
            "dataset_id": ["beta"],
            "subject_id": ["sub021"],
        }
    )
    monkeypatch.setattr(eval_loop, "load_checkpoint", lambda *args, **kwargs: {"config": generic})
    monkeypatch.setattr(eval_loop, "load_manifest", lambda _root: manifest)
    with pytest.raises(GovernanceError, match="checkpoint_evaluation"):
        eval_loop.load_evaluation_context(
            {"data": {"processed_dirs": ["/unused/beta_v1"]}},
            "unused.pt",
        )
    assert constructed is False


def test_fixed_partition_supports_preregistered_exclusions_and_no_validation() -> None:
    manifest = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "dataset_id": ["d"] * 6,
        }
    )
    split = make_fixed_subject_partition(
        manifest,
        training_subject_ids=["s1"],
        validation_subject_ids=[],
        test_subject_ids=["s2"],
        excluded_subject_ids=["s3"],
        allow_empty_validation=True,
    )
    assert split.train.tolist() == [0, 1]
    assert split.val.tolist() == []
    assert split.test.tolist() == [2, 3]

    with pytest.raises(ValueError, match="exclusions overlap"):
        make_fixed_subject_partition(
            manifest,
            training_subject_ids=["s1"],
            validation_subject_ids=[],
            test_subject_ids=["s2"],
            excluded_subject_ids=["s1", "s3"],
            allow_empty_validation=True,
        )


def test_beta_subject_partition_is_recomputed_from_the_declared_hash_rank() -> None:
    base, _suite = _configs()
    training, lockbox = derive_query_reliability_beta_subject_partition()
    assert training == sorted(base["data"]["training_subject_ids"])
    assert lockbox == sorted(base["data"]["test_subject_ids"])
    assert len(training) == 35
    assert len(lockbox) == 20
    assert set(training).isdisjoint(lockbox)


def test_wrong_query_donor_map_is_deterministic_within_participant_and_class() -> None:
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{label}-{trial}",
                "subject_id": subject,
                "label": label,
            }
            for subject in ("s1", "s2", "s3")
            for label in (0, 1)
            for trial in (0, 1)
        ]
    )
    mapping = derive_query_reliability_wrong_query_donor_map(samples)
    shuffled = derive_query_reliability_wrong_query_donor_map(
        samples.sample(frac=1.0, random_state=17)
    )
    pd.testing.assert_frame_equal(mapping, shuffled)
    assert mapping["target_subject_id"].eq(mapping["donor_subject_id"]).all()
    assert mapping["target_sample_id"].ne(mapping["donor_sample_id"]).all()
    assert mapping["target_sample_id"].nunique() == len(samples)
    assert mapping["donor_sample_id"].nunique() == len(samples)

    labels = samples.set_index("sample_id")["label"]
    assert all(
        labels[target] == labels[donor]
        for target, donor in mapping[["target_sample_id", "donor_sample_id"]].itertuples(
            index=False, name=None
        )
    )
    assert query_reliability_wrong_query_donor_mapping_sha256(mapping) == (
        query_reliability_wrong_query_donor_mapping_sha256(shuffled)
    )

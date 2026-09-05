from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cfeg import train_loop
from cfeg.data.splits import make_cross_subject_train_val_split
from cfeg.governance import (
    GovernanceError,
    ResearchAccess,
    _validate_confirmatory_lockbox_authorization,
    _validate_confirmatory_training_config,
    _validate_target_free_simulation,
    bind_cohort,
    reject_retired_physical_candidate_action,
    resolve_research_access,
    validate_analysis_plan_contract,
    validate_checkpoint_evaluation_access,
    validate_cohort_roles_contract,
    validate_frozen_analysis_plan,
    validate_physical_candidate_retirement,
    validate_physical_development_training_authorization,
)
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def test_physical_candidate_retirement_is_machine_bound_and_fail_closed() -> None:
    receipt = validate_physical_candidate_retirement()
    assert receipt["status"] == "retired_no_go"
    assert receipt["outcome_reveal_budget"]["consumed_indices"] == [1, 2]
    with pytest.raises(GovernanceError, match="retired by DEC-20260903-005"):
        reject_retired_physical_candidate_action()


def test_frozen_plan_rejects_unimplemented_confirmatory_controls() -> None:
    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    plan["status"] = "frozen"
    plan["success_threshold"] = 0.5
    plan["sesoi_balanced_accuracy"] = 0.01
    plan["inference"] = {"alpha": 0.05, "alternative": "greater"}
    plan["primary_fairness_hash"] = "f" * 64
    plan["source_lock"] = {
        "freeze_tag": "protocol-0.4-freeze",
        "implementation_contract_sha256": "a" * 64,
    }
    control = plan["development_control_protocol"]
    for key in (
        "shortcut_equivalence_margin",
        "pairing_mechanism_margin",
        "counterfactual_mechanism_margin",
        "wrong_metadata_safety_harm_margin",
    ):
        control[key] = 0.01
    control["minimum_bundle_changed_fraction"] = 0.95
    control["minimum_condition_flip_fraction"] = 1.0
    control["confirmatory_control_policy"] = "confirmatory_once"

    with pytest.raises(GovernanceError, match="confirmatory_once is not implemented"):
        validate_frozen_analysis_plan(plan)


def _wearable_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                f"sub{subject:03d}_trial{trial:03d}"
                for subject in range(1, 103)
                for trial in range(240)
            ],
            "dataset_id": "wearable",
            "subject_id": [
                f"sub{subject:03d}" for subject in range(1, 103) for _ in range(240)
            ],
        }
    )


def test_frozen_cohort_roles_bind_dev_and_primary_without_overlap() -> None:
    manifest = _wearable_manifest()
    dev_cfg = load_config(REPO / "configs/train/wearable_development.yaml")
    roles = load_config(REPO / "configs/governance/wearable_cohort_roles.yaml")

    dev_access = resolve_research_access(dev_cfg)
    primary_access = replace(
        dev_access,
        execution_phase="confirmatory_training",
        cohort_role="confirmatory_primary",
        role_config=roles["roles"]["confirmatory_primary"],
    )
    dev = bind_cohort(manifest, dev_access)
    primary = bind_cohort(manifest, primary_access)

    assert dev.expected_n_subjects == 3
    assert dev.expected_n_samples == 720
    assert primary.expected_n_subjects == 99
    assert primary.expected_n_samples == 23_760
    assert set(dev.selected_subject_ids) == {"sub001", "sub002", "sub003"}
    assert set(dev.selected_subject_ids).isdisjoint(primary.selected_subject_ids)


def test_unfrozen_plan_blocks_confirmatory_training_before_data_access() -> None:
    cfg = load_config(REPO / "configs/train/wearable_loso.yaml")
    with pytest.raises(GovernanceError, match="sealed until plan.status=frozen"):
        resolve_research_access(cfg)


@pytest.mark.parametrize("dry_run", [False, True])
def test_confirmatory_run_never_constructs_dataset(monkeypatch, dry_run: bool) -> None:
    cfg = load_config(REPO / "configs/train/wearable_loso.yaml")
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("dataset must remain sealed")

    monkeypatch.setattr(train_loop, "EEGProcessedDataset", forbidden_dataset)
    with pytest.raises(GovernanceError, match="sealed until plan.status=frozen"):
        train_loop.run_training(cfg, dry_run=dry_run)
    assert constructed is False


@pytest.mark.parametrize("flag", [False, None])
def test_wearable_v3_governance_cannot_be_disabled(flag: bool | None) -> None:
    cfg = load_config(REPO / "configs/train/wearable_development.yaml")
    if flag is None:
        cfg["protocol"].pop("governance_required")
    else:
        cfg["protocol"]["governance_required"] = flag
    with pytest.raises(GovernanceError, match="cannot be omitted or overridden"):
        resolve_research_access(cfg)


def test_governed_run_rejects_nonempty_output_before_dataset(tmp_path, monkeypatch) -> None:
    cfg = load_config(REPO / "configs/train/wearable_development.yaml")
    cfg["output_dir"] = str(tmp_path / "existing")
    Path(cfg["output_dir"]).mkdir()
    (Path(cfg["output_dir"]) / "artifact.txt").write_text("stale", encoding="utf-8")
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("dataset must not be constructed")

    monkeypatch.setattr(train_loop, "EEGProcessedDataset", forbidden_dataset)
    monkeypatch.setattr(
        train_loop,
        "validate_physical_development_training_authorization",
        lambda _cfg: None,
    )
    with pytest.raises(GovernanceError, match="not empty"):
        train_loop.run_training(cfg)
    assert constructed is False


def test_counterfactual_training_is_blocked_before_dataset_construction(monkeypatch) -> None:
    cfg = load_config(REPO / "configs/train/wearable_development.yaml")
    cfg["protocol"]["development_control"] = "counterfactual_wet_dry"
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("dataset must not be constructed")

    monkeypatch.setattr(train_loop, "EEGProcessedDataset", forbidden_dataset)
    with pytest.raises(GovernanceError, match="evaluation-only intervention"):
        train_loop.run_training(cfg, dry_run=True)
    assert constructed is False


def test_governed_wearable_requires_canonical_plan_path(tmp_path) -> None:
    cfg = load_config(REPO / "configs/train/wearable_development.yaml")
    copied_plan = tmp_path / "copied-plan.yaml"
    copied_plan.write_bytes((REPO / "configs/analysis/wearable_primary.yaml").read_bytes())
    cfg["protocol"]["analysis_plan"] = str(copied_plan)

    with pytest.raises(GovernanceError, match="require.*wearable_primary.yaml"):
        resolve_research_access(cfg)


def test_plan_and_role_contracts_reject_cohort_mutation() -> None:
    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    plan["primary_excluded_subject_ids"] = ["sub002", "sub003", "sub004"]
    with pytest.raises(GovernanceError, match="violates the wearable protocol"):
        validate_analysis_plan_contract(plan)

    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    plan["execution_manifest_path"] = "outputs/alternate/manifest.json"
    with pytest.raises(GovernanceError, match="violates the wearable protocol"):
        validate_analysis_plan_contract(plan)

    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    plan["development_control_protocol"]["control_seed"] = 99
    with pytest.raises(GovernanceError, match="violates the wearable protocol"):
        validate_analysis_plan_contract(plan)

    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    plan["primary_model_contract"]["max_channel_gain_delta"] = 0.9
    with pytest.raises(GovernanceError, match="violates the wearable protocol"):
        validate_analysis_plan_contract(plan)

    roles = load_config(REPO / "configs/governance/wearable_cohort_roles.yaml")
    roles["roles"]["development"]["include_subject_ids"] = [
        "sub002",
        "sub003",
        "sub004",
    ]
    with pytest.raises(GovernanceError, match="violates the frozen decision"):
        validate_cohort_roles_contract(roles)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), 999),
        (("data", "split_seed"), 999),
        (("data", "n_folds"), 4),
        (("data", "fold_index"), 9),
        (("data", "lockbox_subject_ids"), ["sub005"]),
    ],
)
def test_confirmatory_config_must_match_frozen_grid(path: tuple[str, ...], value) -> None:
    cfg = load_config(REPO / "configs/train/wearable_loso.yaml")
    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    plan["primary_fairness_hash"] = "f" * 64
    cfg["protocol"]["primary_ablation_role"] = "A0_eeg_only"
    cfg["protocol"]["primary_fairness_hash"] = "f" * 64
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(GovernanceError, match="differs from the frozen plan"):
        _validate_confirmatory_training_config(cfg, plan)


def test_confirmatory_training_requires_manifest_binding() -> None:
    cfg = load_config(REPO / "configs/train/wearable_loso.yaml", strict_env=False)
    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml", strict_env=False)
    plan["primary_fairness_hash"] = "f" * 64
    cfg["protocol"].update(
        {
            "primary_ablation_role": "A0_eeg_only",
            "primary_fairness_hash": "f" * 64,
            "execution_job_id": "seed42-fold0-A0_eeg_only",
            "execution_contract_sha256": "e" * 64,
        }
    )

    with pytest.raises(GovernanceError, match="execution_manifest"):
        _validate_confirmatory_training_config(cfg, plan)


def test_confirmatory_prediction_requires_orchestrator_authorization_before_data(
    monkeypatch,
) -> None:
    from cfeg import eval_loop, governance

    source = {
        "source_commit_sha": "a" * 40,
        "source_dirty": False,
        "source_tree_sha256": "b" * 64,
    }
    access = ResearchAccess(
        governed=True,
        execution_phase="confirmatory_training",
        cohort_role="confirmatory_primary",
        plan_path=str(REPO / "configs/analysis/wearable_primary.yaml"),
        plan_status="frozen",
        plan_sha256="c" * 64,
        cohort_roles_path=str(REPO / "configs/governance/wearable_cohort_roles.yaml"),
        cohort_roles_sha256="d" * 64,
        cohort_decision_id="DEC-20260830-001",
        role_config={},
        allow_outer_test_during_training=False,
    )
    cfg = {
        "runtime_contract": {
            "analysis_plan_sha256": access.plan_sha256,
            "cohort_roles_sha256": access.cohort_roles_sha256,
            **source,
        }
    }
    monkeypatch.setattr(governance, "resolve_research_access", lambda _: access)
    monkeypatch.setattr(governance, "current_source_revision_contract", lambda: source)
    monkeypatch.setattr(eval_loop, "load_checkpoint", lambda *args, **kwargs: {"config": cfg})
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("lockbox dataset must remain sealed")

    monkeypatch.setattr(eval_loop, "EEGProcessedDataset", forbidden_dataset)
    with pytest.raises(GovernanceError, match="generic evaluation route"):
        eval_loop.load_evaluation_context(
            {
                "data": {"split": "test"},
                "research_action": "primary_prediction_lockbox_clean",
                "scenario": "clean",
            },
            REPO / "outputs/confirmatory-primary/missing.pt",
        )
    assert constructed is False


def test_confirmatory_evaluation_rejects_generic_route_and_extra_options_before_data(
    monkeypatch,
) -> None:
    from cfeg import governance

    source = {
        "source_commit_sha": "a" * 40,
        "source_dirty": False,
        "source_tree_sha256": "b" * 64,
    }
    access = ResearchAccess(
        governed=True,
        execution_phase="confirmatory_training",
        cohort_role="confirmatory_primary",
        plan_path=str(REPO / "configs/analysis/wearable_primary.yaml"),
        plan_status="frozen",
        plan_sha256="c" * 64,
        cohort_roles_path=str(REPO / "configs/governance/wearable_cohort_roles.yaml"),
        cohort_roles_sha256="d" * 64,
        cohort_decision_id="DEC-20260830-001",
        role_config={},
        allow_outer_test_during_training=False,
    )
    cfg = {
        "runtime_contract": {
            "analysis_plan_sha256": access.plan_sha256,
            "cohort_roles_sha256": access.cohort_roles_sha256,
            **source,
        }
    }
    evaluation = {
        "data": {"processed_dirs": ["/audited/wearable_v3"], "split": "test"},
        "research_action": "primary_prediction_lockbox_clean",
        "scenario": "clean",
        "prediction_output_path": "/hidden/staging/job.csv",
        "confirmatory_authorization": {
            "execution_manifest": "canonical.json",
            "lockbox_reveal_receipt": "receipt.json",
        },
    }
    monkeypatch.setattr(governance, "resolve_research_access", lambda _: access)
    monkeypatch.setattr(governance, "current_source_revision_contract", lambda: source)
    monkeypatch.setattr(
        governance,
        "_validate_confirmatory_lockbox_authorization",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(GovernanceError, match="generic evaluation route"):
        validate_checkpoint_evaluation_access(cfg, evaluation)

    evaluation["mode"] = "robustness"
    with pytest.raises(GovernanceError, match="unregistered options"):
        validate_checkpoint_evaluation_access(
            cfg,
            evaluation,
            access_route="prediction_bundle",
        )


def test_confirmatory_prediction_requires_exact_hidden_staging_path(
    tmp_path,
    monkeypatch,
) -> None:
    from cfeg import governance

    manifest_path = tmp_path / "execution_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    checkpoint_path = tmp_path / "runs" / "job" / "final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    final_path = tmp_path / "predictions" / "job.csv"
    job = {
        "job_id": "job",
        "checkpoint_path": str(checkpoint_path),
        "prediction_csv": str(final_path),
    }
    manifest = {
        "execution_contract_sha256": "e" * 64,
        "jobs": [job],
    }
    access = ResearchAccess(
        governed=True,
        execution_phase="confirmatory_training",
        cohort_role="confirmatory_primary",
        plan_path=str(REPO / "configs/analysis/wearable_primary.yaml"),
        plan_status="frozen",
        plan_sha256="c" * 64,
        cohort_roles_path=str(REPO / "configs/governance/wearable_cohort_roles.yaml"),
        cohort_roles_sha256="d" * 64,
        cohort_decision_id="DEC-20260830-001",
        role_config={},
        allow_outer_test_during_training=False,
    )
    train_cfg = {"protocol": {"execution_manifest": str(manifest_path)}}
    evaluation = {
        "prediction_output_path": str(tmp_path / "outside.csv"),
        "confirmatory_authorization": {
            "execution_manifest": str(manifest_path),
            "lockbox_reveal_receipt": str(tmp_path / "receipt.json"),
        },
    }
    monkeypatch.setattr(
        governance,
        "_load_bound_execution_manifest",
        lambda *args, **kwargs: (manifest_path.resolve(), manifest, "f" * 64),
    )
    monkeypatch.setattr(governance, "_manifest_job_for_config", lambda *args: job)

    with pytest.raises(GovernanceError, match="hidden staging path"):
        _validate_confirmatory_lockbox_authorization(
            train_cfg,
            evaluation,
            access=access,
            checkpoint_path=checkpoint_path,
        )


def test_development_split_has_train_val_and_no_test() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_id": ["wearable"] * 6,
            "subject_id": ["sub001", "sub001", "sub002", "sub002", "sub003", "sub003"],
        }
    )
    split = make_cross_subject_train_val_split(manifest, seed=42, val_ratio=0.34)
    assert len(split.train) + len(split.val) == len(manifest)
    assert len(split.test) == 0
    assert set(manifest.iloc[split.train]["subject_id"]).isdisjoint(
        set(manifest.iloc[split.val]["subject_id"])
    )


def test_target_free_simulation_receipt_rejects_tampered_threshold(
    tmp_path,
) -> None:
    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml")
    receipt_path = REPO / plan["target_free_simulation"]["accepted_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["core_results"]["independent_symmetric"]["type_i_wilson95_high"] = 0.2
    tampered = tmp_path / "tampered-receipt.json"
    tampered.write_text(json.dumps(receipt), encoding="utf-8")
    plan["target_free_simulation"]["accepted_receipt"] = str(tampered)
    plan["target_free_simulation"]["accepted_receipt_sha256"] = hashlib.sha256(
        tampered.read_bytes()
    ).hexdigest()

    with pytest.raises(GovernanceError, match="violates a freeze threshold"):
        _validate_target_free_simulation(plan)


def test_development_checkpoint_rejects_test_and_all_access() -> None:
    cfg = load_config(REPO / "configs/train/wearable_development.yaml")
    with pytest.raises(GovernanceError, match="Development checkpoints"):
        validate_checkpoint_evaluation_access(cfg, {"data": {"split": "test"}})
    with pytest.raises(GovernanceError, match="Development checkpoints"):
        validate_checkpoint_evaluation_access(cfg, {"data": {"split": "all"}})
    with pytest.raises(GovernanceError, match="canonical grid orchestrator"):
        validate_checkpoint_evaluation_access(cfg, {"data": {"split": "val"}})


def test_outcome_gated_physical_development_rejects_generic_evaluation() -> None:
    cfg = load_config(REPO / "configs/train/wearable_physical_development_loso.yaml")

    with pytest.raises(GovernanceError, match="canonical grid orchestrator"):
        validate_checkpoint_evaluation_access(cfg, {"data": {"split": "val"}})

    with pytest.raises(GovernanceError, match="grid authorization"):
        validate_checkpoint_evaluation_access(
            cfg,
            {"data": {"split": "val"}},
            checkpoint_path="missing-final.pt",
            access_route="prediction_bundle",
        )

    cfg["protocol"]["development_outcome_gate_required"] = False
    with pytest.raises(GovernanceError, match="canonical grid orchestrator"):
        validate_checkpoint_evaluation_access(cfg, {"data": {"split": "val"}})


def test_physical_development_training_requires_a_manifest_job_before_data_access() -> None:
    cfg = load_config(REPO / "configs/train/wearable_physical_development_loso.yaml")

    with pytest.raises(GovernanceError, match="development_grid_manifest"):
        validate_physical_development_training_authorization(cfg)


def test_governed_development_cannot_bypass_grid_by_overriding_architecture(
    monkeypatch,
) -> None:
    cfg = load_config(REPO / "configs/train/wearable_physical_development_loso.yaml")
    cfg["model"]["conditioning"]["architecture"] = "prompt_adapter_v1"
    constructed = False

    def forbidden_dataset(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("dataset must not be constructed")

    monkeypatch.setattr(train_loop, "EEGProcessedDataset", forbidden_dataset)
    with pytest.raises(GovernanceError, match="development_grid_manifest"):
        train_loop.run_training(cfg)
    assert constructed is False

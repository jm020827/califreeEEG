from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cfeg.analysis.physical_mechanism import evaluate_predeclared_physical_gates
from cfeg.data.metadata_controls import build_development_control_plan
from cfeg.execution_manifest import sha256_json
from cfeg.governance import WEARABLE_V3_PHYSICAL_FREEZE_TAG, GovernanceError
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_physical_mechanism_loso as physical_runner
from run_ablation import (
    _validate_development_grid_execution,
    _validate_physical_mechanism_family,
)
from run_physical_mechanism_loso import (
    _expected_intervention_control_contract,
    _finalize_reveal_publication,
    _fsync_bundle_tree,
    _preflight_global_reveal,
    _prepare_private_intervention,
    _prepare_private_prediction,
    _publish_prepared_bundle,
    _rebuild_donor_control_contract,
    _reject_symlink_components,
    _validate_bundle_manifest,
    _validate_design_freeze_authorization,
    _write_bundle_manifest,
    _write_staged_analysis,
    build_grid_manifest,
    register_global_reveal,
    validate_grid_manifest,
)


def _allow_unit_test_source_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        physical_runner,
        "_validate_clean_source_freeze",
        lambda _source, _decision: None,
    )


def test_physical_freeze_tag_has_one_cross_module_authority() -> None:
    decision = json.loads(
        (REPO / "configs/governance/wearable_physical_reveal2_decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert physical_runner.PHYSICAL_FREEZE_TAG == WEARABLE_V3_PHYSICAL_FREEZE_TAG
    assert (
        decision["source_freeze"]["annotated_tag"]
        == WEARABLE_V3_PHYSICAL_FREEZE_TAG
    )


def test_physical_grid_prepares_exact_18_jobs_as_second_reveal(monkeypatch) -> None:
    _allow_unit_test_source_freeze(monkeypatch)
    root = REPO / "outputs/development-loso/physical-mechanism-v2"
    manifest = build_grid_manifest(
        "configs/train/wearable_physical_development_loso.yaml",
        "configs/train/wearable_physical_mechanism.yaml",
        root,
    )

    assert manifest["design_status"] == "frozen"
    assert manifest["expected_job_count"] == 18
    assert manifest["expected_intervention_count"] == 3
    assert manifest["outcome_reveal_index"] == 2
    assert len({job["job_id"] for job in manifest["jobs"]}) == 18
    assert {item["fold_index"] for item in manifest["interventions"]} == {0, 1, 2}
    validate_grid_manifest(manifest, manifest_path=root / "grid_manifest.json")


def test_physical_grid_rejects_job_config_tampering(monkeypatch) -> None:
    _allow_unit_test_source_freeze(monkeypatch)
    root = REPO / "outputs/development-loso/physical-mechanism-v2"
    manifest = build_grid_manifest(
        "configs/train/wearable_physical_development_loso.yaml",
        "configs/train/wearable_physical_mechanism.yaml",
        root,
    )
    tampered = copy.deepcopy(manifest)
    tampered["jobs"][0]["config"]["model"]["latent"]["enabled"] = True
    tampered["grid_content_sha256"] = sha256_json(
        {key: value for key, value in tampered.items() if key != "grid_content_sha256"}
    )

    with pytest.raises(ValueError, match="canonical six-role design"):
        validate_grid_manifest(tampered, manifest_path=root / "grid_manifest.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("design_status", "draft_pre_freeze"),
        ("global_reveal_ledger", "/tmp/fresh-ledger.json"),
    ],
)
def test_physical_grid_rejects_rehashed_outer_contract_tampering(
    field: str, value, monkeypatch
) -> None:
    _allow_unit_test_source_freeze(monkeypatch)
    root = REPO / "outputs/development-loso/physical-mechanism-v2"
    manifest = build_grid_manifest(
        "configs/train/wearable_physical_development_loso.yaml",
        "configs/train/wearable_physical_mechanism.yaml",
        root,
    )
    tampered = copy.deepcopy(manifest)
    tampered[field] = value
    tampered["grid_content_sha256"] = sha256_json(
        {key: item for key, item in tampered.items() if key != "grid_content_sha256"}
    )

    with pytest.raises(ValueError, match="canonical source config"):
        validate_grid_manifest(tampered, manifest_path=root / "grid_manifest.json")


def test_frozen_physical_grid_requires_owner_identity_reveal_and_margins() -> None:
    base = load_config(
        REPO / "configs/train/wearable_physical_development_loso.yaml", strict_env=False
    )
    base["development_grid"]["freeze_authorization"]["approved_by"] = None

    with pytest.raises(ValueError, match="explicit owner decision"):
        _validate_design_freeze_authorization(base)


def test_physical_grid_requires_outcome_gate_even_while_draft() -> None:
    base = load_config(
        REPO / "configs/train/wearable_physical_development_loso.yaml", strict_env=False
    )
    base["protocol"]["development_outcome_gate_required"] = False

    with pytest.raises(ValueError, match="development_outcome_gate_required=true"):
        _validate_design_freeze_authorization(base)


def test_predeclared_physical_gates_use_least_favourable_fold() -> None:
    subjects = pd.DataFrame(
        {
            "subject_id": ["sub001", "sub002", "sub003"],
            "fold_index": [0, 1, 2],
            "a0_balanced_accuracy": [0.14, 0.14, 0.14],
            "full_a2_balanced_accuracy": [0.20, 0.21, 0.22],
            "shuffle_train_balanced_accuracy": [0.15, 0.16, 0.17],
            "metadata_only_balanced_accuracy": [0.09, 0.09, 0.09],
        }
    )
    intervention_rows = []
    for fold in range(3):
        for scenario, score, changed, flip in (
            ("clean", 0.20, 0.0, 0.0),
            ("block_coherent_metadata_shuffle", 0.15, 0.9, 0.0),
            ("joint_wet_dry_counterfactual", 0.12, 0.9, 1.0),
        ):
            intervention_rows.append(
                {
                    "fold_index": fold,
                    "scenario": scenario,
                    "balanced_accuracy": score,
                    "metadata_control_effective_changed_fraction": changed,
                    "metadata_control_condition_flip_fraction": flip,
                }
            )
    interventions = pd.DataFrame(intervention_rows)
    training_controls = pd.DataFrame(
        {
            "fold_index": [0, 1, 2],
            "role": ["M3_full_shuffle_train"] * 3,
            "effective_changed_fraction": [0.9, 0.9, 0.9],
        }
    )
    control = {
        "chance_balanced_accuracy": 1.0 / 12.0,
        "clean_a2_mean_minimum_delta": 0.0,
        "clean_a2_subject_harm_margin": 0.03,
        "shortcut_equivalence_margin": 0.02,
        "pairing_mechanism_margin": 0.03,
        "counterfactual_mechanism_margin": 0.05,
        "wrong_metadata_safety_harm_margin": 0.03,
        "minimum_bundle_changed_fraction": 0.8,
        "minimum_condition_flip_fraction": 0.9,
        "confirmatory_control_policy": "development_gate_only",
    }

    result = evaluate_predeclared_physical_gates(
        subjects, interventions, training_controls, control
    )
    assert result["all_predeclared_diagnostic_gates_pass"] is True
    assert result["population_inference_allowed"] is False
    assert result["gates"]["clean_a2_directional_mean"]["passed"] is True
    assert (
        result["gates"]["clean_a2_observed_subject_safety_audit"][
            "redundancy_note"
        ]
    )

    subjects.loc[2, "metadata_only_balanced_accuracy"] = 0.20
    failed = evaluate_predeclared_physical_gates(
        subjects, interventions, training_controls, control
    )
    assert failed["all_predeclared_diagnostic_gates_pass"] is False
    assert failed["gates"]["shortcut_metadata_only"]["passed"] is False


def _passing_frozen_gate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    subjects = pd.DataFrame(
        {
            "subject_id": ["sub001", "sub002", "sub003"],
            "fold_index": [0, 1, 2],
            "a0_balanced_accuracy": [0.14, 0.14, 0.14],
            "full_a2_balanced_accuracy": [0.20, 0.21, 0.22],
            "shuffle_train_balanced_accuracy": [0.17, 0.18, 0.19],
            "metadata_only_balanced_accuracy": [0.09, 0.09, 0.09],
        }
    )
    rows = []
    for fold, clean in enumerate((0.20, 0.21, 0.22)):
        for scenario, score, changed, flip in (
            ("clean", clean, 0.0, 0.0),
            ("block_coherent_metadata_shuffle", clean - 0.03, 1.0, 0.0),
            ("joint_wet_dry_counterfactual", clean - 0.04, 1.0, 1.0),
        ):
            rows.append(
                {
                    "fold_index": fold,
                    "scenario": scenario,
                    "balanced_accuracy": score,
                    "metadata_control_effective_changed_fraction": changed,
                    "metadata_control_condition_flip_fraction": flip,
                }
            )
    training = pd.DataFrame(
        {
            "fold_index": [0, 1, 2],
            "role": ["M3_full_shuffle_train"] * 3,
            "effective_changed_fraction": [1.0, 1.0, 1.0],
        }
    )
    receipt = json.loads(
        (REPO / "configs/governance/wearable_physical_reveal2_decision.json").read_text(
            encoding="utf-8"
        )
    )
    return subjects, pd.DataFrame(rows), training, receipt["control_values"]


def test_frozen_physical_gate_contract_has_exact_values_and_ten_names() -> None:
    subjects, interventions, training, control = _passing_frozen_gate_inputs()
    assert control == {
        "chance_balanced_accuracy": 1.0 / 12.0,
        "clean_a2_mean_minimum_delta": 0.0,
        "clean_a2_subject_harm_margin": 0.03,
        "shortcut_equivalence_margin": 0.03,
        "pairing_mechanism_margin": 0.0125,
        "counterfactual_mechanism_margin": 0.0125,
        "wrong_metadata_safety_harm_margin": 0.03,
        "minimum_bundle_changed_fraction": 0.95,
        "minimum_condition_flip_fraction": 1.0,
        "confirmatory_control_policy": "development_gate_only",
    }
    result = evaluate_predeclared_physical_gates(
        subjects, interventions, training, control
    )
    assert list(result["gates"]) == [
        "clean_a2_directional_mean",
        "clean_a2_observed_subject_safety_audit",
        "shortcut_metadata_only",
        "training_pairing_shuffle",
        "inference_pairing_shuffle",
        "counterfactual_reliance",
        "wrong_metadata_safety",
        "training_shuffle_potency",
        "inference_bundle_potency",
        "counterfactual_condition_flip_potency",
    ]
    assert result["status"] == "pass_all_predeclared_diagnostic_gates"
    assert result["assay_valid"] is True


def test_physical_potency_failure_is_invalid_assay_not_negative_evidence() -> None:
    subjects, interventions, training, control = _passing_frozen_gate_inputs()
    training.loc[1, "effective_changed_fraction"] = 0.94

    result = evaluate_predeclared_physical_gates(
        subjects, interventions, training, control
    )

    assert result["assay_valid"] is False
    assert result["status"] == "invalid_assay_control_potency_failure"
    assert result["failed_potency_gates"] == ["training_shuffle_potency"]
    assert result["substantive_interpretation_allowed"] is False
    assert result["all_predeclared_diagnostic_gates_pass"] is False


def test_valid_assay_substantive_failure_is_diagnostic_no_go() -> None:
    subjects, interventions, training, control = _passing_frozen_gate_inputs()
    subjects.loc[2, "metadata_only_balanced_accuracy"] = 0.20

    result = evaluate_predeclared_physical_gates(
        subjects, interventions, training, control
    )

    assert result["assay_valid"] is True
    assert result["status"] == "diagnostic_no_go_one_or_more_substantive_gates_failed"
    assert result["failed_substantive_gates"] == ["shortcut_metadata_only"]
    assert result["substantive_interpretation_allowed"] is True
    assert result["all_predeclared_diagnostic_gates_pass"] is False


def test_physical_gate_integer_boundaries_are_stable_for_all_baselines() -> None:
    from cfeg.analysis.physical_mechanism import _lower_gate

    for baseline_count in range(238):
        exact_three = (baseline_count + 3) / 240 - baseline_count / 240
        gate = _lower_gate([exact_three] * 3, 0.0125)
        assert gate["passed"] is True
        assert gate["comparison_tolerance"] == 1e-12
    for baseline_count in range(239):
        exact_two = (baseline_count + 2) / 240 - baseline_count / 240
        assert _lower_gate([exact_two] * 3, 0.0125)["passed"] is False


def test_physical_publication_bundle_tree_hash_detects_tampering(tmp_path: Path) -> None:
    bundle = tmp_path / "reveal-bundle"
    for name in ("predictions", "interventions", "analysis"):
        (bundle / name).mkdir(parents=True, exist_ok=True)
    (bundle / "predictions" / "p.csv").write_text("prediction\n1\n", encoding="utf-8")
    (bundle / "interventions" / "i.csv").write_text("score\n0.1\n", encoding="utf-8")
    (bundle / "analysis" / "revealed_summary.json").write_text("{}\n", encoding="utf-8")
    stale_temporary = bundle / ".bundle_manifest.json.tmp-999999"
    stale_temporary.write_text("partial", encoding="utf-8")
    manifest = {
        "grid_content_sha256": "a" * 64,
        "decision_receipt_sha256": "b" * 64,
        "outcome_reveal_index": 2,
    }

    digest = _write_bundle_manifest(bundle, manifest)
    assert not stale_temporary.exists()
    assert _validate_bundle_manifest(bundle, manifest) == digest
    assert _write_bundle_manifest(bundle, manifest) == digest
    (bundle / "predictions" / "p.csv").write_text("prediction\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tree hash"):
        _validate_bundle_manifest(bundle, manifest)


def test_physical_bundle_rejects_nested_manifest_name_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "reveal-bundle"
    for name in ("predictions", "interventions", "analysis"):
        (bundle / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "grid_content_sha256": "a" * 64,
        "decision_receipt_sha256": "b" * 64,
        "outcome_reveal_index": 2,
    }
    digest = _write_bundle_manifest(bundle, manifest)
    assert _validate_bundle_manifest(bundle, manifest) == digest
    nested = bundle / "predictions" / "nested"
    nested.mkdir()
    (nested / "bundle_manifest.json").write_text("mutable\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nested directories"):
        _validate_bundle_manifest(bundle, manifest)


def test_physical_bundle_rejects_symlink_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "reveal-bundle"
    for name in ("predictions", "interventions", "analysis"):
        (bundle / name).mkdir(parents=True, exist_ok=True)
    target = tmp_path / "mutable.csv"
    target.write_text("prediction\n1\n", encoding="utf-8")
    (bundle / "predictions" / "linked.csv").symlink_to(target)
    manifest = {
        "grid_content_sha256": "a" * 64,
        "decision_receipt_sha256": "b" * 64,
        "outcome_reveal_index": 2,
    }

    with pytest.raises(ValueError, match="symlinks"):
        _write_bundle_manifest(bundle, manifest)


def test_staged_analysis_exact_resume_reuses_only_identical_artifacts(tmp_path: Path) -> None:
    staging = tmp_path / ".reveal-staging"
    staging.mkdir()
    subjects = pd.DataFrame({"subject_id": ["sub001"], "score": [0.1]})
    interventions = pd.DataFrame({"scenario": ["clean"], "score": [0.1]})
    summary = {"status": "descriptive"}

    _write_staged_analysis(
        staging,
        subjects=subjects,
        interventions=interventions,
        summary=summary,
    )
    _write_staged_analysis(
        staging,
        subjects=subjects,
        interventions=interventions,
        summary=summary,
    )
    (staging / "analysis" / "revealed_summary.json").write_text(
        '{"status":"tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differs from recomputation"):
        _write_staged_analysis(
            staging,
            subjects=subjects,
            interventions=interventions,
            summary=summary,
        )


def test_partial_private_prediction_is_removed_and_deterministically_recomputed(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "predictions" / "seed42-fold0-A0.csv"
    output.parent.mkdir()
    output.write_text("partial\n", encoding="utf-8")
    stale_temporary = output.with_name(
        f".{output.stem}_provenance.json.999999.deadbeef.tmp"
    )
    stale_temporary.write_text("partial", encoding="utf-8")
    called = []

    def fake_prediction(*args, **kwargs):
        assert not output.exists()
        assert not stale_temporary.exists()
        called.append((args, kwargs))
        output.write_text("prediction\n1\n", encoding="utf-8")
        output.with_name(f"{output.stem}_provenance.json").write_text(
            "{}\n", encoding="utf-8"
        )

    monkeypatch.setattr(physical_runner, "run_prediction", fake_prediction)
    job = {
        "prediction_staging_csv": str(output),
        "prediction_csv": str(tmp_path / "public" / output.name),
        "output_dir": str(tmp_path / "run"),
    }
    _prepare_private_prediction(
        job,
        processed_dir=str(tmp_path / "processed"),
        authorization={"fixed": "yes"},
    )
    assert len(called) == 1
    assert output.is_file()
    assert output.with_name(f"{output.stem}_provenance.json").is_file()


def test_partial_private_intervention_is_removed_and_deterministically_recomputed(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "interventions" / "fold0-A2.csv"
    output.parent.mkdir()
    partial = output.with_name(f"{output.stem}_clean_predictions.csv")
    partial.write_text("partial\n", encoding="utf-8")
    stale_temporary = output.with_name(
        f".{output.stem}_provenance.json.999999.deadbeef.tmp"
    )
    stale_temporary.write_text("partial", encoding="utf-8")
    called = []
    intervention = {
        "staging_output_csv": str(output),
        "checkpoint_path": str(tmp_path / "run" / "final.pt"),
        "config": {
            "output_csv": str(output),
            "perturbations": [
                {"name": "metadata_shuffle", "type": "metadata_shuffle"},
                {
                    "name": "counterfactual",
                    "type": "metadata_counterfactual_wet_dry",
                },
            ],
        },
    }

    def fake_evaluation(config, checkpoint, *, access_route):
        assert not any(output.parent.glob(f"{output.stem}*"))
        assert not stale_temporary.exists()
        called.append((config, checkpoint, access_route))

    monkeypatch.setattr(physical_runner, "run_evaluation", fake_evaluation)
    _prepare_private_intervention(intervention, authorization={"fixed": "yes"})
    assert len(called) == 1
    assert called[0][0]["development_authorization"] == {"fixed": "yes"}


def test_physical_intervention_validator_uses_distinct_frozen_donor_scopes() -> None:
    assert _expected_intervention_control_contract(
        "block_coherent_metadata_shuffle"
    ) == (
        "within_class_shuffle",
        "within_dataset_block_derangement_label_aligned",
    )
    assert _expected_intervention_control_contract("joint_wet_dry_counterfactual") == (
        "counterfactual_wet_dry",
        "same_dataset_subject_label_block_window_opposite_electrode",
    )


def _semantic_donor_asset() -> pd.DataFrame:
    rows = []
    for subject_index, subject_id in enumerate(("sub001", "sub002"), start=1):
        for electrode_index, electrode in enumerate(("dry", "wet"), start=1):
            impedance = float(100 * electrode_index + subject_index)
            for label in (0, 1):
                rows.append(
                    {
                        "sample_id": f"{subject_id}-{electrode}-label{label}",
                        "dataset_id": "wearable",
                        "subject_id": subject_id,
                        "session_id": "session0",
                        "run_id": "run0",
                        "label": label,
                        "window_start_sec": 0.0,
                        "window_duration_sec": 2.0,
                        "electrode_type": electrode,
                        "reference": "forehead",
                        "cap_type": "wearable",
                        "impedance_mean_kohm": impedance,
                        "impedance_max_kohm": impedance + 1.0,
                        "impedance_kohm_by_channel": [impedance, impedance + 1.0],
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("control", ["within_class_shuffle", "counterfactual_wet_dry"])
def test_physical_donor_validator_rebuilds_exact_semantics(control: str) -> None:
    asset = _semantic_donor_asset()
    plan = build_development_control_plan(
        asset,
        {"val": pd.Series(range(len(asset))).to_numpy()},
        control=control,
        seed=42,
    )

    rebuilt = _rebuild_donor_control_contract(
        asset,
        plan.donor_mapping,
        control=control,
        seed=42,
        observed_mapping_sha256=plan.contract["donor_mapping_sha256"],
    )
    assert rebuilt["donor_mapping_sha256"] == plan.contract["donor_mapping_sha256"]
    assert rebuilt["effective_external_metadata_changed_fraction"] == plan.contract[
        "effective_external_metadata_changed_fraction"
    ]

    tampered = plan.donor_mapping.copy()
    tampered.loc[0, "external_bundle_changed"] = not bool(
        tampered.loc[0, "external_bundle_changed"]
    )
    with pytest.raises(ValueError, match="canonical reconstruction"):
        tampered_sha256 = hashlib.sha256(
            tampered.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        _rebuild_donor_control_contract(
            asset,
            tampered,
            control=control,
            seed=42,
            observed_mapping_sha256=tampered_sha256,
        )


def test_physical_bundle_tree_fsync_includes_staging_parent(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "grid" / ".reveal-staging"
    child = bundle / "analysis"
    child.mkdir(parents=True)
    (child / "artifact.json").write_text("{}\n", encoding="utf-8")
    synced = []
    monkeypatch.setattr(
        physical_runner,
        "_fsync_directory",
        lambda directory: synced.append(directory),
    )

    _fsync_bundle_tree(bundle)

    assert child in synced
    assert bundle in synced
    assert synced[-1] == bundle.parent


def test_physical_bundle_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "mutable-target"
    for name in ("predictions", "interventions", "analysis"):
        (target / name).mkdir(parents=True, exist_ok=True)
    bundle = tmp_path / "reveal-bundle"
    bundle.symlink_to(target, target_is_directory=True)
    manifest = {
        "grid_content_sha256": "a" * 64,
        "decision_receipt_sha256": "b" * 64,
        "outcome_reveal_index": 2,
    }

    with pytest.raises(ValueError, match="root cannot be a symlink"):
        _write_bundle_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        _fsync_bundle_tree(bundle)


def test_physical_canonical_path_rejects_symlink_before_resolve(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "canonical-alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        _reject_symlink_components(alias / "physical-mechanism-v2")


def _status_manifest(tmp_path: Path) -> dict:
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(
        json.dumps(
            {
                "schema": "cfeg.development-reveal-ledger.v1",
                "entries": [
                    {
                        "reveal_index": 1,
                        "grid_content_sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "jobs": [],
        "interventions": [],
        "historical_reveal_ledger": str(historical_path),
        "global_reveal_ledger": str(tmp_path / "global.json"),
        "grid_content_sha256": "b" * 64,
        "outcome_reveal_index": 2,
        "recommended_max_outcome_reveals": 2,
        "absolute_max_outcome_reveals": 3,
        "design_status": "frozen",
        "expected_job_count": 18,
        "expected_intervention_count": 3,
    }


def test_physical_status_rejects_orphan_precommit_receipt(tmp_path: Path) -> None:
    root = tmp_path / "grid"
    root.mkdir()
    (root / "reveal_precommit_receipt.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no matching reveal ledger event"):
        physical_runner.print_status(_status_manifest(tmp_path), root)


def test_physical_status_rejects_symlinked_final_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "grid"
    (root / "reveal-bundle").mkdir(parents=True)
    external = tmp_path / "external-receipt.json"
    external.write_text("{}\n", encoding="utf-8")
    (root / "reveal_receipt.json").symlink_to(external)
    monkeypatch.setattr(
        physical_runner,
        "_preflight_global_reveal",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="types are invalid"):
        physical_runner.print_status(_status_manifest(tmp_path), root)


def test_physical_readiness_rejects_reveal_index_consumed_by_other_grid(
    tmp_path: Path,
) -> None:
    manifest = _status_manifest(tmp_path)
    Path(manifest["global_reveal_ledger"]).write_text(
        json.dumps(
            {
                "schema": "cfeg.development-reveal-ledger.v1",
                "entries": [
                    {
                        "reveal_index": 2,
                        "grid_content_sha256": "c" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reveal index must be 2"):
        _preflight_global_reveal(
            manifest,
            root=tmp_path / "grid",
            allow_receipt_recovery=False,
        )


def test_physical_publication_fsyncs_tree_before_budget_precommit(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "grid"
    staging = root / ".reveal-staging"
    final = root / "reveal-bundle"
    calls = []

    monkeypatch.setattr(
        physical_runner,
        "_validate_reveal_bundle_layout",
        lambda _manifest, *, use_staging: calls.append(("layout", use_staging)),
    )
    monkeypatch.setattr(
        physical_runner,
        "_validate_bundle_manifest",
        lambda _bundle, _manifest, *, expected_digest=None: calls.append(
            ("tree_hash", expected_digest)
        )
        or "d" * 64,
    )
    monkeypatch.setattr(
        physical_runner,
        "_fsync_bundle_tree",
        lambda _bundle: calls.append(("fsync_tree", None)),
    )
    monkeypatch.setattr(
        physical_runner,
        "register_global_reveal",
        lambda *_args, **_kwargs: calls.append(("budget_precommit", None)),
    )
    monkeypatch.setattr(
        physical_runner.os,
        "replace",
        lambda _source, _target: calls.append(("rename", None)),
    )
    monkeypatch.setattr(
        physical_runner,
        "_fsync_directory",
        lambda _directory: calls.append(("fsync_parent", None)),
    )
    monkeypatch.setattr(
        physical_runner,
        "_finalize_reveal_publication",
        lambda *_args, **_kwargs: calls.append(("final_receipt", None)),
    )
    monkeypatch.setattr(
        physical_runner,
        "_validate_complete_publication",
        lambda *_args, **_kwargs: calls.append(("final_validate", None)),
    )

    _publish_prepared_bundle(
        {},
        root=root,
        staging_dir=staging,
        final_dir=final,
        bundle_digest="d" * 64,
    )

    assert [name for name, _ in calls] == [
        "layout",
        "tree_hash",
        "fsync_tree",
        "budget_precommit",
        "rename",
        "fsync_parent",
        "final_receipt",
        "final_validate",
    ]


def test_physical_reveal_two_phase_publication_is_idempotent_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    historical_path = tmp_path / "historical.json"
    global_path = tmp_path / "global.json"
    root = tmp_path / "grid"
    root.mkdir()
    staging = root / ".reveal-staging"
    for name in ("predictions", "interventions", "analysis"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    (staging / "analysis" / "revealed_subjects.csv").write_text(
        "subject_id\nsub001\n", encoding="utf-8"
    )
    (staging / "analysis" / "revealed_interventions.csv").write_text(
        "scenario\nclean\n", encoding="utf-8"
    )
    (staging / "analysis" / "revealed_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )
    historical = {
        "schema": "cfeg.development-reveal-ledger.v1",
        "entries": [
            {
                "reveal_index": 1,
                "grid_content_sha256": "a" * 64,
                "grid_id": "historical-v1",
                "registered_at_utc": "2026-08-30T00:00:00+00:00",
                "scope": "historical",
                "confirmatory_claim_allowed": False,
            }
        ],
    }
    historical_path.write_text(json.dumps(historical), encoding="utf-8")
    manifest = {
        "grid_content_sha256": "b" * 64,
        "grid_id": "physical-v2",
        "outcome_reveal_index": 2,
        "recommended_max_outcome_reveals": 2,
        "absolute_max_outcome_reveals": 3,
        "historical_reveal_ledger": str(historical_path),
        "historical_reveal_ledger_sha256": hashlib.sha256(
            historical_path.read_bytes()
        ).hexdigest(),
        "global_reveal_ledger": str(global_path),
        "decision_receipt_sha256": "e" * 64,
        "canonical_output_root": str(root),
    }
    bundle_digest = _write_bundle_manifest(staging, manifest)

    first = register_global_reveal(
        manifest,
        root,
        bundle_path=root / ".reveal-staging",
        bundle_content_sha256=bundle_digest,
    )
    second = register_global_reveal(
        manifest,
        root,
        bundle_path=root / ".reveal-staging",
        bundle_content_sha256=bundle_digest,
    )
    assert first == second
    assert first["reveal_index"] == 2
    assert first["event_type"] == "reveal_budget_precommit"
    assert not (root / "reveal-bundle").exists()
    assert not (root / "reveal_receipt.json").exists()
    assert _preflight_global_reveal(manifest, root=root) == bundle_digest
    precommit_path = root / "reveal_precommit_receipt.json"
    immutable_precommit = precommit_path.read_bytes()

    precommit_path.unlink()
    assert _preflight_global_reveal(manifest, root=root) == bundle_digest
    assert precommit_path.read_bytes() == immutable_precommit

    with pytest.raises(ValueError, match="canonical private bundle"):
        register_global_reveal(
            manifest,
            root,
            bundle_path=staging,
            bundle_content_sha256="f" * 64,
        )

    later = json.loads(global_path.read_text(encoding="utf-8"))
    later["entries"].append(
        {"reveal_index": 3, "grid_content_sha256": "c" * 64, "grid_id": "later-v3"}
    )
    global_path.write_text(json.dumps(later), encoding="utf-8")
    assert (
        register_global_reveal(
            manifest,
            root,
            bundle_path=root / ".reveal-staging",
            bundle_content_sha256=bundle_digest,
        )
        == first
    )
    assert precommit_path.read_bytes() == immutable_precommit

    (root / ".reveal-staging").replace(root / "reveal-bundle")
    assert not (root / "reveal_receipt.json").exists()
    _finalize_reveal_publication(
        manifest,
        root=root,
        bundle_digest=bundle_digest,
    )
    final_receipt = (root / "reveal_receipt.json").read_bytes()
    _finalize_reveal_publication(
        manifest,
        root=root,
        bundle_digest=bundle_digest,
    )
    assert (root / "reveal_receipt.json").read_bytes() == final_receipt

    forged = json.loads(global_path.read_text(encoding="utf-8"))
    forged["entries"][0]["scope"] = "partial-one-file-reveal"
    global_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or forged"):
        _preflight_global_reveal(manifest, root=root)


def test_draft_and_historical_grids_cannot_train_through_generic_ablation() -> None:
    draft = load_config(
        REPO / "configs/train/wearable_physical_development_loso.yaml", strict_env=False
    )
    historical = load_config(
        REPO / "configs/train/wearable_development_loso.yaml", strict_env=False
    )

    with pytest.raises(GovernanceError, match="canonical grid orchestrator"):
        _validate_development_grid_execution(draft, dry_run=False)
    with pytest.raises(GovernanceError, match="completed historical"):
        _validate_development_grid_execution(historical, dry_run=False)
    _validate_development_grid_execution(draft, dry_run=True)

    bypass = copy.deepcopy(draft)
    bypass["development_grid"]["status"] = "frozen"
    bypass["development_grid"]["grid_id"] = "renamed-to-evade-history"
    bypass["protocol"]["development_outcome_gate_required"] = False
    with pytest.raises(GovernanceError, match="CLI overrides cannot disable"):
        _validate_development_grid_execution(bypass, dry_run=False)


def test_mechanism_family_rejects_undeclared_role_drift() -> None:
    base = load_config(
        REPO / "configs/train/wearable_physical_development_loso.yaml", strict_env=False
    )
    suite = load_config(
        REPO / "configs/train/wearable_physical_mechanism.yaml", strict_env=False
    )
    broken = copy.deepcopy(suite)
    broken["variants"]["M1_global_only"][
        "model.conditioning.channel_quality.enabled"
    ] = True

    with pytest.raises(ValueError, match="treatment mismatch"):
        _validate_physical_mechanism_family(base, broken)

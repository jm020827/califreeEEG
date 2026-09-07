from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

_REPOSITORY = Path(__file__).resolve().parents[1]
_MASTER = _REPOSITORY / "configs/analysis/metadata_calibration_efficiency_v3.yaml"
_SYNTHETIC = _REPOSITORY / "configs/analysis/metadata_calibration_v3_synthetic.yaml"
_ORIGINAL_AMENDMENT = (
    _REPOSITORY / "configs/governance/metadata_calibration_v3_preoutcome_amendment.json"
)
_FIRST_RECOVERY_AMENDMENT = (
    _REPOSITORY / "configs/governance/metadata_calibration_v3_1_recovery_amendment.json"
)
_EARLIER_RECOVERY_AMENDMENT = (
    _REPOSITORY / "configs/governance/metadata_calibration_v3_2_recovery_amendment.json"
)
_PRIOR_RECOVERY_AMENDMENT = (
    _REPOSITORY / "configs/governance/metadata_calibration_v3_3_recovery_amendment.json"
)
_RECOVERY_AMENDMENT = (
    _REPOSITORY / "configs/governance/metadata_calibration_v3_4_recovery_amendment.json"
)
_OWNER_KEY = _REPOSITORY / "configs/governance/metadata_calibration_v3_owner_authority.pub"
_V2_DENY = _REPOSITORY / "configs/governance/metadata_calibration_v2_terminal_deny_overlay.json"

_EXPECTED_V2_DENY_SHA256 = "dd4541765b6b021f4dc4c83fa8fc0b7c2eb76b997a093a0cff479f4fa9b3f7d0"
_EXPECTED_OWNER_KEY_SHA256 = "ce868830f9c47948a6abe2068863c8f39b12cba470e5493ff32a7c3c7178374b"
_EXPECTED_OWNER_FINGERPRINT = "SHA256:tm6CDH5eVtjTKNqBUwrBYwbq5RhZ48wo1QjP9c+mR+g"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_json_sha256(value: dict[str, str]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _line_hash(domain: str, fields: list[str]) -> str:
    encoded = ("\n".join([domain, *fields]) + "\n").encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_v3_contract_hash_links_and_frozen_trust_roots() -> None:
    master = _load_yaml(_MASTER)
    synthetic = _load_yaml(_SYNTHETIC)
    original = json.loads(_ORIGINAL_AMENDMENT.read_text(encoding="utf-8"))
    first_recovery = json.loads(_FIRST_RECOVERY_AMENDMENT.read_text(encoding="utf-8"))
    earlier_recovery = json.loads(_EARLIER_RECOVERY_AMENDMENT.read_text(encoding="utf-8"))
    prior_recovery = json.loads(_PRIOR_RECOVERY_AMENDMENT.read_text(encoding="utf-8"))
    recovery = json.loads(_RECOVERY_AMENDMENT.read_text(encoding="utf-8"))

    assert synthetic["master_plan"]["file_sha256"] == _file_sha256(_MASTER)
    recovery_sha256 = _file_sha256(_RECOVERY_AMENDMENT)
    assert master["preoutcome_amendment"]["file_sha256"] == recovery_sha256
    assert synthetic["preoutcome_amendment"]["file_sha256"] == recovery_sha256
    assert recovery["original_preoutcome_amendment"] == {
        "path": "configs/governance/metadata_calibration_v3_preoutcome_amendment.json",
        "file_sha256": _file_sha256(_ORIGINAL_AMENDMENT),
        "must_remain_byte_identical": True,
    }
    assert recovery["prior_recovery_amendment_v1"] == {
        "path": "configs/governance/metadata_calibration_v3_1_recovery_amendment.json",
        "file_sha256": _file_sha256(_FIRST_RECOVERY_AMENDMENT),
        "must_remain_byte_identical": True,
    }
    assert recovery["prior_recovery_amendment_v2"] == {
        "path": "configs/governance/metadata_calibration_v3_2_recovery_amendment.json",
        "file_sha256": _file_sha256(_EARLIER_RECOVERY_AMENDMENT),
        "must_remain_byte_identical": True,
    }
    assert recovery["prior_recovery_amendment_v3"] == {
        "path": "configs/governance/metadata_calibration_v3_3_recovery_amendment.json",
        "file_sha256": _file_sha256(_PRIOR_RECOVERY_AMENDMENT),
        "must_remain_byte_identical": True,
    }
    assert first_recovery["protocol_revision"] == "V3.1"
    assert earlier_recovery["protocol_revision"] == "V3.2"
    assert prior_recovery["protocol_revision"] == "V3.3"
    assert _file_sha256(_V2_DENY) == _EXPECTED_V2_DENY_SHA256
    assert original["retired_ancestor"]["deny_overlay_file_sha256"] == _EXPECTED_V2_DENY_SHA256

    assert _file_sha256(_OWNER_KEY) == _EXPECTED_OWNER_KEY_SHA256
    key_fields = _OWNER_KEY.read_text(encoding="ascii").strip().split()
    assert key_fields[0] == "ssh-rsa"
    wire_key = base64.b64decode(key_fields[1], validate=True)
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(wire_key).digest()).decode().rstrip(
        "="
    )
    assert fingerprint == _EXPECTED_OWNER_FINGERPRINT
    trust = master["governance"]["owner_execution_authorization_trust_root"]
    assert trust["public_key_file_sha256"] == _EXPECTED_OWNER_KEY_SHA256
    assert trust["SSH_public_key_fingerprint"] == _EXPECTED_OWNER_FINGERPRINT
    assert (
        original["owner_execution_authorization_trust_root"]["public_key_file_sha256"]
        == _EXPECTED_OWNER_KEY_SHA256
    )


def test_v3_4_recovery_receipt_seed_and_one_leaf_projection_are_exact() -> None:
    master = _load_yaml(_MASTER)
    synthetic = _load_yaml(_SYNTHETIC)
    recovery = json.loads(_RECOVERY_AMENDMENT.read_text(encoding="utf-8"))
    retired_v4 = recovery["retired_development_v4"]
    failure = retired_v4["observed_resume_failure"]
    durable = retired_v4["durable_execution_facts"]
    inference = retired_v4["control_flow_inference"]

    assert recovery["protocol_revision"] == "V3.4"
    assert recovery["scientific_objective_changed"] is False
    assert recovery["scientific_contract_delta"] == "synthetic.rng.development_root_seed_only"
    assert recovery["owner_approval"]["approved_by"] == "active_workspace_owner"
    assert "not a cryptographic signature" in recovery["owner_approval"]["attestation_note"]
    assert "given_before_development_v4_incident" in recovery["owner_approval"]["approval_basis"]
    assert retired_v4["source_commit"] == "198158663efd1c4481b23dd3d012b9d854ada8d1"
    assert retired_v4["source_tree"] == "ac7882494955a7abb5046a4dd950bd98d6fdd6b1"
    assert retired_v4["source_bundle_sha256"] == (
        "c57c81a2ffae4511e4de6159d11ea5ef482e13b8fdebd916eef98bdf7235250c"
    )

    emitted = base64.b64decode(failure["emitted_json_line_utf8_base64"], validate=True)
    assert emitted == (
        b'{"command":"resume","error":"TypeError","message":"primary_values_by_endpoint '
        b'must be an exact endpoint-ordered dictionary.","status":"FAIL_CLOSED"}\n'
    )
    assert hashlib.sha256(emitted).hexdigest() == failure["emitted_json_line_sha256"]
    assert durable["context_reference_artifact_count"] == 1
    assert durable["development_start_artifact_count"] == 1
    assert durable["development_result_artifact_count"] == 0
    assert durable["development_seed"] == 3_156_745_110
    assert durable["development_seed_consumed_by_terminal_start_receipt_policy"] is True
    assert durable["experimental_workflow_outcome_observation_count"] == 0
    assert inference["classification"] == (
        "strong_control_flow_inference_not_observed_outcome_or_durable_attestation"
    )
    assert inference["development_seedsequence_and_DGP_executed"] is True
    assert inference["must_not_be_reconstructed_or_used_as_evidence"] is True

    inventory_preimage = {
        "schema": retired_v4["artifact_inventory_schema"],
        "artifacts": [
            {"role": "development_bundle", **retired_v4["development_bundle"]},
            {"role": "context_reference", **retired_v4["context_reference"]},
            {
                "role": "development_start",
                **{
                    key: retired_v4["development_start"][key]
                    for key in ("path", "schema", "payload_sha256", "file_sha256")
                },
            },
        ],
    }
    inventory_digest = hashlib.sha256(
        json.dumps(
            inventory_preimage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert inventory_digest == retired_v4["artifact_inventory_sha256"]
    assert inventory_digest == "511de6688fa315830ddf98ed5a78eefde4e0e4ba6f15977c347281403e908dfa"

    replacement = recovery["replacement_development_seed"]
    preimage_bytes = json.dumps(
        replacement["preimage"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(preimage_bytes).digest()
    words = [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 32, 4)]
    forbidden = set(replacement["preimage"]["forbidden_seed_values"])
    eligible = [(index, word) for index, word in enumerate(words) if word and word not in forbidden]
    assert hashlib.sha256(preimage_bytes).hexdigest() == replacement["preimage_sha256"]
    assert replacement["preimage_sha256"] == (
        "11f503bdca78e8a7d0e853f9a7b484510dd749018c7735bce61ff7965b70b1cb"
    )
    assert len(preimage_bytes) == replacement["canonical_preimage_byte_length"] == 799
    assert words == [
        301269949,
        3396921511,
        3504886777,
        2813625425,
        232212737,
        2356622780,
        3860854678,
        1534112203,
    ]
    assert eligible[0] == (replacement["selected_word_index"], replacement["selected_seed"])
    assert eligible[0] == (0, 301_269_949)
    assert replacement["outcome_used_in_derivation"] is False

    projection = recovery["scientific_contract_projection"]

    def projection_bytes(master_value: dict[str, Any], synthetic_value: dict[str, Any]) -> bytes:
        value = {
            "schema": projection["schema"],
            "candidate_id": recovery["candidate_id"],
            "master": {key: master_value[key] for key in projection["master_keys"]},
            "synthetic": {key: synthetic_value[key] for key in projection["synthetic_keys"]},
        }
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    baseline_master = yaml.safe_load(
        subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                _REPOSITORY,
                "show",
                (
                    "198158663efd1c4481b23dd3d012b9d854ada8d1:"
                    "configs/analysis/metadata_calibration_efficiency_v3.yaml"
                ),
            ]
        )
    )
    baseline_synthetic = yaml.safe_load(
        subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                _REPOSITORY,
                "show",
                (
                    "198158663efd1c4481b23dd3d012b9d854ada8d1:"
                    "configs/analysis/metadata_calibration_v3_synthetic.yaml"
                ),
            ]
        )
    )
    current_projection = projection_bytes(master, synthetic)
    baseline_projection = projection_bytes(baseline_master, baseline_synthetic)
    assert len(current_projection) == projection["canonical_byte_length"] == 59_153
    assert hashlib.sha256(current_projection).hexdigest() == projection["sha256"]
    assert projection["sha256"] == (
        "670b61c767ff2a4b02c3a582aa8ac7e46b5632ada2806ffc109e2e79326d05ca"
    )
    assert hashlib.sha256(baseline_projection).hexdigest() == projection["retired_baseline_sha256"]
    assert projection["retired_baseline_sha256"] == (
        "5885f39908d8a33b130e0dc923abe560cc4587ebe7c58ed242647ce028f94712"
    )
    assert projection["original_pre_v3_3_baseline_sha256"] == (
        "0702d01be1e055d3203a3c1b78777db6456b8d527e5525b6d468fb52522f8a79"
    )
    normalized = copy.deepcopy(synthetic)
    normalized["rng"]["development_root_seed"] = 3_156_745_110
    assert projection_bytes(master, normalized) == baseline_projection
    assert synthetic["rng"]["development_root_seed"] == 301_269_949

    for plan in (master, synthetic):
        link = plan["preoutcome_amendment"]
        assert link["path"] == _RECOVERY_AMENDMENT.relative_to(_REPOSITORY).as_posix()
        assert link["file_sha256"] == _file_sha256(_RECOVERY_AMENDMENT)
        assert link["protocol_revision"] == "V3.4"
        assert link["development_attempt_id"] == "development-v5"
        assert link["original_file_sha256"] == _file_sha256(_ORIGINAL_AMENDMENT)
        assert link["prior_recovery_v1_file_sha256"] == _file_sha256(_FIRST_RECOVERY_AMENDMENT)
        assert link["prior_recovery_v2_file_sha256"] == _file_sha256(_EARLIER_RECOVERY_AMENDMENT)
        assert link["prior_recovery_v3_file_sha256"] == _file_sha256(_PRIOR_RECOVERY_AMENDMENT)
        assert (
            link["retired_scientific_contract_projection_sha256"]
            == (projection["retired_baseline_sha256"])
        )
        assert link["scientific_contract_projection_sha256"] == projection["sha256"]

    prerequisite = synthetic["development_bundle"]["recovery_prerequisite"]
    assert prerequisite["retired_v1_exact_file_count"] == 2
    assert prerequisite["retired_v2_exact_file_count"] == 1
    assert prerequisite["retired_v3_exact_file_count"] == 3
    assert prerequisite["retired_v3_development_seed_consumed"] is True
    assert prerequisite["retired_v3_result_artifact_count"] == 0
    assert prerequisite["adopt_retired_v1_context_exact_bytes"] is True
    assert prerequisite["retired_v3_context_must_be_byte_identical_to_retired_v1"] is True
    assert prerequisite["retired_v4_exact_file_count"] == 3
    assert prerequisite["retired_v4_development_seed_consumed"] is True
    assert prerequisite["retired_v4_result_artifact_count"] == 0
    assert prerequisite["retired_v4_context_must_be_byte_identical_to_retired_v1"] is True


def test_v3_exact_operator_grid_ids_tokens_and_digests() -> None:
    master = _load_yaml(_MASTER)
    residual = master["support_residual"]
    grid = residual["development_grid"]
    cells = grid["canonical_cells"]
    constants = grid["exact_digest_payload_constant_values"]

    assert [cell["index"] for cell in cells] == list(range(1, 10))
    assert len({cell["id"] for cell in cells}) == 9
    assert {(cell["prototype_prior_pseudocount"], cell["lambda_max"]) for cell in cells} == {
        (nu, lam) for nu in (1.0, 4.0, 16.0) for lam in (0.10, 0.20, 0.30)
    }
    for cell in cells:
        payload = {
            **constants,
            "grid_cell_id": cell["id"],
            "nu_token": str(cell["nu_token"]),
            "lambda_token": str(cell["lambda_token"]),
        }
        assert _canonical_json_sha256(payload) == cell["operator_instance_sha256"]
        scientific_id = master["scientific_candidate_id_template"].format(
            nu_token=cell["nu_token"],
            lambda_token=cell["lambda_token"],
        )
        assert scientific_id.endswith(f"nu_{cell['nu_token']}-lambda_{cell['lambda_token']}")


def test_v3_grid_row_arithmetic_is_exact() -> None:
    synthetic = _load_yaml(_SYNTHETIC)
    grid = synthetic["complete_grid"]
    families = grid["cells_per_parameter_pair"]

    b4 = families["B4_interface_calibrated_impedance_shift"]
    condition_cells = sum(
        int(spec["condition_cell_count"])
        for key, spec in families.items()
        if key != "B4_interface_calibrated_impedance_shift"
    ) + int(b4["condition_cell_count_total"])
    assert condition_cells == grid["condition_level_cells_per_parameter_pair"] == 172
    summaries = condition_cells + int(grid["B4_composite_cells_per_parameter_pair"])
    assert summaries == grid["participant_summary_cells_per_parameter_pair"] == 206
    rows_per_pair = summaries * int(synthetic["population"]["development_participants_per_family"])
    assert rows_per_pair == grid["development_participant_rows_per_parameter_pair"] == 9_888
    metric_rows = rows_per_pair * int(grid["development_parameter_pairs"])
    assert metric_rows == grid["development_participant_metric_rows"] == 88_992
    invariant_rows = (
        int(grid["N5_context_invariant_cases_per_parameter_pair"])
        + int(grid["N6_support_reliability_invariant_cases_per_parameter_pair"])
    ) * int(grid["development_parameter_pairs"])
    assert invariant_rows == grid["development_invariant_rows"] == 90
    assert metric_rows + invariant_rows == grid["total_development_rows"] == 89_082
    assert grid["future_selected_lockbox_total_rows"] == 19_786


def test_v3_canary_seed_and_probe_oracles_are_executable() -> None:
    synthetic = _load_yaml(_SYNTHETIC)
    canary = synthetic["governance_canary"]
    pulse = canary["expected_pulse"]

    seed_digest = _line_hash(
        canary["seed_derivation"]["domain_separator_ASCII"],
        [
            canary["public_fixture_sha256"],
            pulse["timestamp_utc"],
            str(pulse["chain_index"]),
            str(pulse["pulse_index"]),
            pulse["output_value"],
        ],
    )
    assert seed_digest == canary["seed_derivation"]["expected_seed_digest_sha256"]
    assert (
        int.from_bytes(bytes.fromhex(seed_digest), "big", signed=False)
        == canary["seed_derivation"]["expected_full_unsigned_256_bit_root_seed"]
    )
    probe = canary["probe_oracle"]
    assert (
        _line_hash(
            probe["domain_separator_ASCII"],
            [seed_digest, probe["exact_pre_validation_trace_ASCII"]],
        )
        == probe["expected_digest_sha256"]
    )


def test_v3_state_domains_authority_and_order_fail_closed() -> None:
    master = _load_yaml(_MASTER)
    synthetic = _load_yaml(_SYNTHETIC)
    governance = master["governance"]
    canary = governance["state_engine_domains"]["governance_canary"]

    expected_trace = synthetic["governance_canary"]["expected_trace"]
    assert canary["state_enum"] == expected_trace
    assert canary["bridge_to_scientific_lifecycle"] == {
        "prerequisite_scientific_state": "VERIFIED",
        "from_canary_state": "CANARY_FRESH_AUDITED",
        "to_scientific_state": "CANARY_PASSED",
        "accepted_capability_type": "FreshCanaryAuditCapability",
        "requires_fresh_audit_artifact_payload_and_file_sha256": True,
        "requires_exact_selected_freeze_commit_tree_test_fixture_trace_seed_and_probe_bindings": True,
        "direct_reverse_or_duplicate_bridge": "forbidden",
    }
    order = governance["required_order"]
    assert (
        order.index("future_beacon_attempt_manifest_with_exact_target_endpoint_and_statement")
        < order.index("target_bound_signed_execution_authorization_before_target")
        < order.index("wait_until_target")
        < order.index("global_O_EXCL_claim")
        < order.index("intrinsic_future_beacon_validation")
    )
    assert governance["future_beacon_selected"] is False
    assert governance["scientific_lockbox_authorized"] is False
    assert governance["human_EEG_outcome_authorized"] is False
    assert synthetic["scientific_lockbox"]["current_authority"] is False
    bundle_path = Path(synthetic["development_bundle"]["canonical_path"])
    assert bundle_path.is_absolute()
    assert bundle_path.is_relative_to(
        Path("/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v5")
    )
    retired_root = Path(master["governance"]["canonical_paths"]["retired_development_v1"])
    assert retired_root == Path(
        "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v1"
    )
    assert master["governance"]["canonical_paths"]["retired_development_v2"] == (
        "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v2"
    )
    assert master["governance"]["canonical_paths"]["retired_development_v3"] == (
        "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v3"
    )
    assert not bundle_path.is_relative_to(_REPOSITORY)


def test_v3_exact_paths_seed_statement_and_fresh_audit_parser_contract() -> None:
    master = _load_yaml(_MASTER)
    synthetic = _load_yaml(_SYNTHETIC)
    governance = master["governance"]

    statement = governance["future_attempt_manifest"]["exact_seed_statement_construction"]
    assert statement["manifest_field"] == "exact_seed_statement_base64"
    assert statement["manifest_field_encoding"] == "RFC4648_standard_base64_with_padding"
    assert statement["manifest_field"] in governance["future_attempt_manifest"]["must_bind"]
    assert "exact_seed_statement_base64_RFC4648_with_padding" not in _MASTER.read_text(
        encoding="utf-8"
    )
    sample_statement = (
        "domain\ncandidate\nscientific\nfreeze\ntarget\nendpoint\nprimitive\n"
    ).encode("ascii")
    encoded = base64.b64encode(sample_statement).decode("ascii")
    assert base64.b64decode(encoded, validate=True) == sample_statement

    manifest_fields = set(governance["future_attempt_manifest"]["must_bind"])
    assert "manifest_created_at_UTC" in manifest_fields
    assert governance["future_attempt_manifest"]["manifest_created_at_UTC_rule"].startswith(
        "strict_RFC3339_UTC"
    )
    for prefix in (
        "selected_method_freeze",
        "development_bundle",
        "test_evidence",
        "canary_result",
        "canary_fresh_audit",
    ):
        assert {
            f"{prefix}_schema",
            f"{prefix}_payload_sha256",
            f"{prefix}_file_sha256",
        } <= manifest_fields
    assert {
        "selected_method_freeze_sha256",
        "source_bundle_sha256",
        "test_evidence_sha256",
        "canary_result_sha256",
    }.isdisjoint(manifest_fields)

    identity = governance["scientific_attempt_identity"]
    assert identity["input"] == "exact_scientific_attempt_manifest_payload_sha256"
    assert identity["regex"] == "^sha256-[0-9a-f]{64}$"
    assert identity["caller_supplied_attempt_id"] == "forbidden"
    manifest_digest = hashlib.sha256(b"complete-manifest-payload").hexdigest()
    attempt_id = f"sha256-{manifest_digest}"
    assert len(attempt_id) == 71
    assert attempt_id.removeprefix("sha256-") == manifest_digest

    root = Path(governance["canonical_paths"]["canary"])
    expected_names = {
        "canary_authorization": "authorization.json",
        "canary_claim": "claim.json",
        "canary_beacon": "beacon.json",
        "canary_result": "result.json",
        "canary_fresh_audit": "fresh-audit.json",
    }
    for field, filename in expected_names.items():
        assert Path(governance["canonical_paths"][field]) == root / filename

    development_root = Path(governance["canonical_paths"]["development"])
    assert Path(governance["canonical_paths"]["development_bundle"]) == (
        development_root / "development-bundle.json"
    )
    assert Path(governance["canonical_paths"]["development_context_reference"]) == (
        development_root / "context-reference.json"
    )
    assert Path(governance["canonical_paths"]["development_result"]) == (
        development_root / "development-result.json"
    )
    scientific_root = Path(governance["canonical_paths"]["scientific"])
    global_claim_path = Path(governance["canonical_paths"]["scientific_global_claim"])
    assert global_claim_path == scientific_root / "global-claim.json"
    assert "{attempt_id}" not in str(global_claim_path)
    assert "scientific_claim_template" not in governance["canonical_paths"]
    global_claim = governance["scientific_global_claim"]
    assert Path(global_claim["canonical_path"]) == global_claim_path
    assert global_claim["path_is_candidate_wide_and_attempt_independent"] is True
    assert (
        global_claim["a_new_attempt_id_target_endpoint_or_manifest_cannot_create_another_claim"]
        is True
    )

    signed_fields = set(
        governance["owner_execution_authorization_trust_root"]["signed_payload_must_bind"]
    )
    assert {
        "attempt_id",
        "attempt_manifest_schema",
        "attempt_manifest_payload_sha256",
        "attempt_manifest_file_sha256",
    } <= signed_fields
    assert governance["owner_execution_authorization_trust_root"]["signed_at_UTC_rule"].startswith(
        "strict_RFC3339_UTC"
    )

    selected_fields = set(master["selected_method_transition"]["output_must_bind"])
    for prefix in ("development_bundle", "development_result"):
        assert {
            f"{prefix}_schema",
            f"{prefix}_payload_sha256",
            f"{prefix}_file_sha256",
        } <= selected_fields
    assert "development_bundle_sha256" not in selected_fields
    race_safe = governance["artifact_publication"]["race_safe_path_resolution"]
    assert race_safe["realpath_precheck_alone_is_authority"] is False
    assert "directory_fd" in race_safe["traversal"]
    assert "O_CREAT_O_EXCL_O_WRONLY_O_NOFOLLOW" in race_safe["final_creation"]
    assert race_safe["unavailable_directory_fd_O_NOFOLLOW_or_fstat_guarantee"] == "fail_closed"

    audit = synthetic["governance_canary"]["fresh_audit_artifact"]
    assert Path(audit["canonical_path"]) == root / "fresh-audit.json"
    assert audit["explicit_path_loader"] == {
        "reads_only_exact_declared_canonical_paths": True,
        "glob_directory_scan_or_implicit_artifact_discovery": "forbidden",
        "rejects_symlink_nonregular_or_realpath_mismatch": True,
        "output": "immutable_exact_path_to_bytes_mapping",
        "may_access_git_environment_network_RNG_or_global_store": False,
    }
    assert audit["pure_parser"] == {
        "input": "immutable_exact_path_to_bytes_mapping_only",
        "filesystem_git_environment_network_RNG_or_global_store_IO": "forbidden",
        "undeclared_missing_or_extra_path": "reject",
    }
    audit_fields = set(audit["must_bind"])
    for prefix in (
        "selected_method_freeze",
        "test_evidence",
        "canary_authorization",
        "canary_claim",
        "canary_beacon",
        "canary_result",
    ):
        assert {
            f"{prefix}_schema",
            f"{prefix}_payload_sha256",
            f"{prefix}_file_sha256",
        } <= audit_fields
    assert (
        "canary_authorization_claim_beacon_and_result_payload_and_file_sha256" not in audit_fields
    )
    publication = audit["publication"]
    assert publication["create_flags"].startswith("O_CREAT_O_EXCL_O_WRONLY")
    assert publication["published_regular_file_mode"] == "0400"
    assert publication["parent_directory_mode"] == "0700"

    assert Path(synthetic["scientific_lockbox"]["global_claim_canonical_path"]) == (
        scientific_root / "global-claim.json"
    )
    assert synthetic["scientific_lockbox"]["caller_supplied_attempt_id"] == "forbidden"

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

_REPOSITORY = Path(__file__).resolve().parents[1]
_MASTER = _REPOSITORY / "configs/analysis/metadata_calibration_efficiency_v3.yaml"
_SYNTHETIC = _REPOSITORY / "configs/analysis/metadata_calibration_v3_synthetic.yaml"
_AMENDMENT = _REPOSITORY / "configs/governance/metadata_calibration_v3_preoutcome_amendment.json"
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
    amendment = json.loads(_AMENDMENT.read_text(encoding="utf-8"))

    assert synthetic["master_plan"]["file_sha256"] == _file_sha256(_MASTER)
    amendment_sha256 = _file_sha256(_AMENDMENT)
    assert master["preoutcome_amendment"]["file_sha256"] == amendment_sha256
    assert synthetic["preoutcome_amendment"]["file_sha256"] == amendment_sha256
    assert _file_sha256(_V2_DENY) == _EXPECTED_V2_DENY_SHA256
    assert amendment["retired_ancestor"]["deny_overlay_file_sha256"] == _EXPECTED_V2_DENY_SHA256

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
        amendment["owner_execution_authorization_trust_root"]["public_key_file_sha256"]
        == _EXPECTED_OWNER_KEY_SHA256
    )


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
        Path("/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v1")
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
    assert global_claim["a_new_attempt_id_target_endpoint_or_manifest_cannot_create_another_claim"] is True

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
    assert "canary_authorization_claim_beacon_and_result_payload_and_file_sha256" not in audit_fields
    publication = audit["publication"]
    assert publication["create_flags"].startswith("O_CREAT_O_EXCL_O_WRONLY")
    assert publication["published_regular_file_mode"] == "0400"
    assert publication["parent_directory_mode"] == "0700"

    assert Path(synthetic["scientific_lockbox"]["global_claim_canonical_path"]) == (
        scientific_root / "global-claim.json"
    )
    assert synthetic["scientific_lockbox"]["caller_supplied_attempt_id"] == "forbidden"

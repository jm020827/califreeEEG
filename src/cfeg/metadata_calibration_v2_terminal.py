from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

V2_TERMINAL_DENY_SCHEMA = "cfeg.metadata-calibration-efficiency-v2.terminal-deny-overlay.v1"
V2_TERMINAL_DENY_SHA256 = "dd4541765b6b021f4dc4c83fa8fc0b7c2eb76b997a093a0cff479f4fa9b3f7d0"

_REPOSITORY = Path(__file__).resolve().parents[2]
V2_TERMINAL_DENY_PATH = (
    _REPOSITORY / "configs/governance/metadata_calibration_v2_terminal_deny_overlay.json"
)

_BLOCKED_ACTIONS = (
    "beta_dong_independent_gate",
    "beta_dong_outcome_reduction_or_selection",
    "beta_dong_query_label_join",
    "beta_dong_score_request_or_prediction",
    "choi2019_v2_replication_outcome",
    "synthetic_lockbox_reexecution",
    "wearable_held60_v2_access_or_outcome",
)
_ALLOWED_READ_ONLY_ACTIONS = (
    "allocation_replay",
    "artifact_integrity_audit",
    "external_inventory_schema_audit",
    "plan_validation",
    "unit_and_regression_tests",
)
_EXPECTED_OVERLAY: dict[str, Any] = {
    "schema": V2_TERMINAL_DENY_SCHEMA,
    "candidate_id": "metadata-calibration-efficiency-v2",
    "decision_id": "TERM-20260906-022",
    "status": "consumed_inconclusive_infrastructure_error",
    "scientific_result_created": False,
    "synthetic_plan_sha256": ("9a9683141ccd3d1fe9cf094aad955c6e317d112a73e35fbcfcdd2ef7d06e1f8c"),
    "terminal_file_sha256": ("6a2c2d21de7c796d5a96c4fef4a23f08ffdd81684ed12e9defce7fcffc796b14"),
    "terminal_receipt_sha256": ("e2382761a3171fd08451221bdcf59a64ed977dadc5d6f987c425b22fdbd8c857"),
    "automatic_retry": "forbidden",
    "blocked_operational_actions": list(_BLOCKED_ACTIONS),
    "allowed_read_only_actions": list(_ALLOWED_READ_ONLY_ACTIONS),
    "replacement_requirement": (
        "new_candidate_schema_seed_preoutcome_amendment_clean_freeze_"
        "authorization_and_future_beacon"
    ),
}


def validate_v2_terminal_deny_overlay() -> dict[str, Any]:
    """Validate the source-pinned post-terminal deny decision without EEG access."""

    path = V2_TERMINAL_DENY_PATH
    try:
        observed = path.lstat()
    except OSError as error:
        raise PermissionError("V2 terminal deny overlay is unavailable; fail closed.") from error
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise PermissionError("V2 terminal deny overlay must be a nonsymlink regular file.")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != V2_TERMINAL_DENY_SHA256:
        raise PermissionError("V2 terminal deny overlay byte hash drifted; fail closed.")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError("V2 terminal deny overlay is not valid JSON; fail closed.") from error
    if decoded != _EXPECTED_OVERLAY:
        raise PermissionError("V2 terminal deny overlay content drifted; fail closed.")
    return dict(decoded)


def deny_v2_terminal_operational_action(action: str) -> None:
    """Reject one known post-terminal execution action before any data is opened."""

    overlay = validate_v2_terminal_deny_overlay()
    if action not in overlay["blocked_operational_actions"]:
        raise PermissionError(f"Unknown V2 terminal-deny action {action!r}; fail closed.")
    raise PermissionError(
        "V2 V11 is consumed_inconclusive_infrastructure_error; operational action "
        f"{action!r} is terminally denied. A new candidate/schema/seed, pre-outcome "
        "amendment, clean freeze, authorization, and future beacon are required."
    )

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# These limits are established before NumPy/SciPy imports. Spawned FBCCA
# workers inherit them and therefore cannot multiply BLAS thread pools.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.metadata_calibration_v2_requests import (
    DEFAULT_V2_ALLOCATION_PATH,
    DEFAULT_V2_FILTERBANK_PATH,
    DEFAULT_V2_PLAN_PATH,
    build_external_score_request,
    build_fbcca_score_cache,
    build_inventory_receipt,
    build_subband_cache,
    build_template_score_cache,
    derive_prequential_gate,
    load_v2_external_contract,
    resolve_phase_candidates,
    validate_external_asset,
    validate_independent_selection_receipt,
    write_json_exclusive,
)
from cfeg.metadata_calibration_v2_terminal import deny_v2_terminal_operational_action


def _positive_workers(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 32:
        raise argparse.ArgumentTypeError("workers must be in [1,32]")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Governed BETA/Dong V2 inventory and query-outcome-free request producer. "
            "This executable has no source39, wearable, or held-data phase."
        )
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_V2_PLAN_PATH)
    parser.add_argument("--allocation", type=Path, default=DEFAULT_V2_ALLOCATION_PATH)
    parser.add_argument("--filterbank", type=Path, default=DEFAULT_V2_FILTERBANK_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="validate exact hashes/schema/allocation without query outcomes"
    )
    _add_asset_arguments(inventory)
    inventory.add_argument("--output", type=Path, required=True)

    request = subparsers.add_parser(
        "request", help="produce one immutable label-free final-query score request"
    )
    _add_asset_arguments(request)
    request.add_argument("--subject-id", required=True)
    request.add_argument("--candidate-key", required=True)
    request.add_argument("--budget", type=int, choices=(0, 1, 3), required=True)
    request.add_argument("--token-secret-file", type=Path, required=True)
    request.add_argument("--cache-dir", type=Path, required=True)
    request.add_argument("--output", type=Path, required=True)
    request.add_argument("--workers", type=_positive_workers, default=1)
    return parser


def _add_asset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("beta_v1", "dong2023_v1"), required=True)
    parser.add_argument("--phase", choices=("development", "independent_gate"), required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path)


def _read_secret(path: Path) -> bytes:
    resolved = path.expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("Query-token secret must be one nonsymlink regular file.")
    value = resolved.read_bytes()
    if len(value) < 32:
        raise ValueError("Query-token secret file must contain at least 32 bytes.")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "request":
        deny_v2_terminal_operational_action("beta_dong_score_request_or_prediction")
    contract = load_v2_external_contract(args.plan, args.allocation, args.filterbank)
    selection = args.selection_receipt
    # This check occurs before opening any independent-gate asset.
    authorization = None
    if args.phase == "independent_gate":
        if selection is None:
            raise PermissionError("Independent gate requires the immutable selection receipt.")
        authorization = validate_independent_selection_receipt(selection, contract=contract)
        candidates = (authorization.candidate,)
    else:
        candidates = resolve_phase_candidates(
            phase=args.phase,
            contract=contract,
            selection_receipt=selection,
        )
    asset = validate_external_asset(
        args.processed_dir,
        dataset_id=args.dataset,
        phase=args.phase,
        contract=contract,
        selection_authorization=authorization,
    )
    if args.command == "inventory":
        receipt = build_inventory_receipt(asset, contract=contract)
        written = write_json_exclusive(args.output, receipt)
        _print_summary(
            command=args.command,
            output=written,
            dataset=args.dataset,
            phase=args.phase,
            status=receipt["status"],
            sha256=receipt["inventory_receipt_sha256"],
        )
        return 0

    candidate_by_key = {value.candidate_key: value for value in candidates}
    candidate = candidate_by_key.get(args.candidate_key)
    if candidate is None:
        raise PermissionError(
            "Candidate is not authorized for this phase; independent_gate accepts only "
            "the immutable development selection."
        )
    participant = asset.participant(args.subject_id)
    secret = _read_secret(args.token_secret_file)
    scores = build_fbcca_score_cache(
        asset,
        participant=participant,
        contract=contract,
        token_secret=secret,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )
    subbands = None
    template = None
    if candidate.operator == "filterbank_target_template_residual" and args.budget > 0:
        subbands = build_subband_cache(
            asset,
            participant=participant,
            contract=contract,
            token_secret=secret,
            cache_dir=args.cache_dir,
        )
        template = build_template_score_cache(
            subbands,
            budget=args.budget,
            cache_dir=args.cache_dir,
        )
    if args.budget == 0:
        gate_enabled: bool | None = False
    elif args.budget == 1:
        # Development evaluates the frozen candidate; independent_gate can
        # arrive here only through a receipt with k1_global_enabled=true.
        gate_enabled = True
    else:
        gate_enabled = derive_prequential_gate(
            score_cache=scores,
            candidate=candidate,
            contract=contract,
            template_subbands=subbands,
        )
    request = build_external_score_request(
        score_cache=scores,
        candidate=candidate,
        budget=args.budget,
        contract=contract,
        gate_enabled=gate_enabled,
        template_cache=template if gate_enabled else None,
        selection_authorization=authorization,
    )
    written = write_json_exclusive(args.output, request)
    _print_summary(
        command=args.command,
        output=written,
        dataset=args.dataset,
        phase=args.phase,
        status="complete_query_outcome_free_request",
        sha256=_sha256_file(written),
    )
    return 0


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_summary(
    *,
    command: str,
    output: Path,
    dataset: str,
    phase: str,
    status: str,
    sha256: str,
) -> None:
    print(
        json.dumps(
            {
                "schema": "cfeg.metadata-calibration-v2-external-cli-result.v1",
                "command": command,
                "dataset_id": dataset,
                "phase": phase,
                "status": status,
                "output": str(output),
                "file_or_receipt_sha256": sha256,
                "query_outcomes_loaded": False,
                "v1_source39_accessed": False,
                "wearable_held60_accessed": False,
                "held_access_authorized": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

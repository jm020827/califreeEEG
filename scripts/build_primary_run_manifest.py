#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.execution_manifest import (
    build_primary_execution_manifest,
    sha256_json,
    validate_canonical_execution_paths,
    validate_primary_execution_manifest,
)
from cfeg.governance import (
    current_source_revision_contract,
    validate_analysis_plan_contract,
    validate_frozen_analysis_plan,
)
from cfeg.train_loop import _resolve_augmentation_channel_sets
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the sealed six-run 39-train/60-lockbox manifest."
    )
    parser.add_argument("command", choices=("generate", "validate", "status"))
    parser.add_argument("--base-config", default="configs/train/wearable_loso.yaml")
    parser.add_argument("--ablation-config", default="configs/train/ablation.yaml")
    parser.add_argument("--plan", default="configs/analysis/wearable_primary.yaml")
    parser.add_argument(
        "--manifest",
        default="outputs/confirmatory-primary/execution_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/confirmatory-primary/runs",
    )
    parser.add_argument(
        "--replace-unexecuted-draft",
        action="store_true",
        help=(
            "Atomically replace only a valid, unauthorized dev draft when no run, "
            "prediction, staging, or reveal artifact exists."
        ),
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan = load_config(plan_path, strict_env=False)
    validate_analysis_plan_contract(plan)
    if plan.get("status") == "frozen":
        validate_frozen_analysis_plan(plan)
    manifest_path = Path(args.manifest)
    canonical_manifest, _ = validate_canonical_execution_paths(
        plan,
        manifest_path=manifest_path,
        output_root=args.output_root,
    )
    manifest_path = canonical_manifest
    if args.command == "generate":
        base = load_config(args.base_config, strict_env=False)
        _resolve_augmentation_channel_sets(base)
        manifest = build_primary_execution_manifest(
            base=base,
            variants=load_config(args.ablation_config, strict_env=False)["variants"],
            plan=plan,
            output_root=args.output_root,
            manifest_path=args.manifest,
            plan_sha256=_sha256_file(plan_path),
            source_contract=current_source_revision_contract(),
        )
        _publish_manifest(
            manifest_path,
            manifest,
            plan=plan,
            replace_unexecuted_draft=args.replace_unexecuted_draft,
        )
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "jobs": len(manifest["jobs"]),
                    "execution_allowed": manifest["execution_allowed"],
                    "manifest_content_sha256": manifest["manifest_content_sha256"],
                },
                sort_keys=True,
            )
        )
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_digest = manifest.pop("manifest_content_sha256", None)
    if sha256_json(manifest) != recorded_digest:
        raise ValueError("Manifest content changed after generation.")
    validate_primary_execution_manifest(
        manifest,
        plan=plan,
        analysis_plan_sha256=_sha256_file(plan_path),
        manifest_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "jobs": len(manifest["jobs"]),
                "execution_allowed": manifest["execution_allowed"],
                "plan_status": manifest["plan_status"],
            },
            sort_keys=True,
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_manifest(
    path: Path,
    manifest: dict,
    *,
    plan: dict,
    replace_unexecuted_draft: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        replacing = path.exists()
        if replacing:
            if not replace_unexecuted_draft:
                raise FileExistsError(
                    f"Canonical execution manifest already exists: {path}. "
                    "Only --replace-unexecuted-draft can replace an unused dev draft."
                )
            _validate_replaceable_unexecuted_draft(path, plan=plan)
        elif replace_unexecuted_draft:
            raise FileNotFoundError(
                "--replace-unexecuted-draft was requested but no canonical draft exists."
            )

        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
        with pending.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if replacing:
                os.replace(pending, path)
            else:
                os.link(pending, path)
                pending.unlink()
            os.fsync(directory_fd)
        finally:
            if pending.exists():
                pending.unlink()
    finally:
        os.close(directory_fd)


def _validate_replaceable_unexecuted_draft(path: Path, *, plan: dict) -> None:
    try:
        draft = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Existing canonical manifest is not a valid dev draft.") from exc
    recorded = draft.pop("manifest_content_sha256", None)
    if (
        recorded != sha256_json(draft)
        or draft.get("schema") != "cfeg.primary-execution-manifest.v1"
        or draft.get("plan_status") != "dev_not_frozen"
        or draft.get("execution_allowed") is not False
        or draft.get("expected_job_count") != 6
        or len(draft.get("jobs") or []) != 6
    ):
        raise ValueError("Existing canonical manifest is not a valid unauthorized six-job draft.")

    output_root = Path(str(plan["execution_output_root"]))
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parents[1] / output_root
    prediction_root = path.parent / "predictions"
    reveal_receipt = path.parent / "lockbox_reveal_receipt.json"
    occupied = []
    for candidate in (output_root.resolve(), prediction_root.resolve()):
        if candidate.is_file() or (candidate.is_dir() and any(candidate.iterdir())):
            occupied.append(str(candidate))
    if reveal_receipt.exists():
        occupied.append(str(reveal_receipt))
    occupied.extend(str(candidate) for candidate in path.parent.glob(".predictions.*.staging"))
    if occupied:
        raise ValueError(
            "Cannot replace a dev draft after confirmatory artifacts appeared: "
            f"{sorted(occupied)}."
        )


if __name__ == "__main__":
    main()

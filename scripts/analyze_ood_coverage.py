#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.ood_coverage import compare_ood_coverage
from cfeg.data.schema import load_manifest
from cfeg.governance import current_source_revision_contract, validate_frozen_analysis_plan
from cfeg.prediction import load_verified_prediction_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired A0/treatment predictions by subject and OOD cell."
    )
    parser.add_argument("--baseline", required=True, help="A0 prediction CSV.")
    parser.add_argument("--candidate", required=True, help="A2/A3 prediction CSV.")
    parser.add_argument("--processed-dir", required=True, help="Directory containing manifest.*.")
    parser.add_argument(
        "--cell-columns",
        nargs="+",
        default=["subject_id", "dataset_id", "electrode_type", "window_duration_sec"],
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Constant cell dimension such as scenario=channel_drop_50 (repeatable).",
    )
    parser.add_argument("--success-threshold", type=float, required=True)
    parser.add_argument(
        "--expected-n-labels",
        type=int,
        default=None,
        help="Require every subject/cell to contain this many classes.",
    )
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--plan", default="configs/analysis/wearable_primary.yaml")
    parser.add_argument(
        "--allow-uncontracted",
        action="store_true",
        help="Allow exploratory non-A0/A2 artifacts without the primary provenance contract.",
    )
    args = parser.parse_args()

    tags = _parse_tags(args.tag)
    if args.allow_uncontracted:
        import pandas as pd

        baseline = pd.read_csv(args.baseline)
        candidate = pd.read_csv(args.candidate)
    else:
        baseline = load_verified_prediction_bundle(args.baseline)
        candidate = load_verified_prediction_bundle(args.candidate)
    cells, subjects, summary = compare_ood_coverage(
        baseline,
        candidate,
        load_manifest(args.processed_dir),
        cell_columns=args.cell_columns,
        success_threshold=args.success_threshold,
        expected_n_labels=args.expected_n_labels,
        tags=tags,
        require_primary_contract=not args.allow_uncontracted,
    )
    if not args.allow_uncontracted:
        _validate_current_contract(summary["primary_contract"], args.plan)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(prefix.with_name(f"{prefix.name}_cells.csv"), index=False)
    subjects.to_csv(prefix.with_name(f"{prefix.name}_subjects.csv"), index=False)
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)


def _parse_tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Tag must use KEY=VALUE syntax, got {value!r}.")
        key, item = value.split("=", 1)
        if not key or not item:
            raise ValueError(f"Tag must use non-empty KEY=VALUE syntax, got {value!r}.")
        if key in tags:
            raise ValueError(f"Duplicate tag key: {key}")
        tags[key] = item
    return tags


def _validate_current_contract(contract: dict, plan_value: str) -> None:
    plan_path = Path(plan_value).expanduser().resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    validate_frozen_analysis_plan(plan)
    roles_path = Path(str(plan["cohort_roles_config"]))
    if not roles_path.is_absolute():
        roles_path = (Path(__file__).resolve().parents[1] / roles_path).resolve()
    current_source = current_source_revision_contract()
    expected = {
        "analysis_plan_sha256": _file_sha256(plan_path),
        "cohort_roles_sha256": _file_sha256(roles_path),
        "source_commit_sha": current_source["source_commit_sha"],
        "source_dirty": current_source["source_dirty"],
        "source_tree_sha256": current_source["source_tree_sha256"],
    }
    mismatched = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Prediction contract differs from the current frozen plan: {mismatched}.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

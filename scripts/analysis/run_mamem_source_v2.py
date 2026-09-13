"""One bounded frozen MAMEM source candidate: generated, development, real.

No downloads, held60, replacement participants, retries, or tuning. Heavy arrays
and full model audit stay in the explicitly selected outside-Git cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/whwovy/data/mamem_i_v1_20260913")
CACHE = Path("/home/whwovy/data/mamem_recorded_event_source_v2")
RUN = ROOT / "docs/reports/mamem_recorded_event_source_v2_run"
DEADLINE = "2026-09-13T14:00:00+00:00"
INVENTORY_SHA = "67791a601d253d7b6d3edf080a1781329991e41e944a3bea5b759cfaed34fb41"
ARCHIVE_SHA = {
    "EEG-SSVEP-Part1.rar": "09fd628b903a23d4fab82dbb69664a22e3063f70c51fb34268718f5de66b4995",
    "EEG-SSVEP-Part2.rar": "2fde94585ae96e7b2076f6adab38b7e2885aff3c26963ffe9aa7d2f5c783edfa",
}
DEV_SHA = "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a"
CODE = [
    "scripts/analysis/run_mamem_source_v2.py",
    "scripts/analysis/prepare_mamem_i_probe_v1.py",
    "scripts/analysis/probe_mamem_i_din_v1.py",
    "src/cfeg/mamem_events_v1.py",
    "src/cfeg/mamem_events_v2.py",
    "src/cfeg/mamem_signal_v1.py",
    "src/cfeg/mamem_shrinkage_v1.py",
    "docs/mamem_recorded_event_source_v2_contract.md",
    "docs/mamem_recorded_event_source_v1_contract.md",
]
UNCHANGED = {
    "scripts/analysis/prepare_mamem_i_probe_v1.py": "b7502c14ac495ae377b3170d152a306e3ca714f8a8db6ca52408d71d7551cf6b",
    "scripts/analysis/probe_mamem_i_din_v1.py": "0ad67538c0f7a5eeb98bfd0543db61d65347f3485361a83a5edc1db052a31cbc",
    "src/cfeg/mamem_events_v1.py": "1c6762783fed206201a9c0539de5aa4c94d45564190bb08954ec4b0c242c096a",
    "src/cfeg/mamem_signal_v1.py": "164d21d638fc0b792b58067dc44d34ba41aaa585811ccade91101a6a0c69d2fc",
    "src/cfeg/mamem_shrinkage_v1.py": "85c1826369b6b182ed444fb5d0661724c33e6c4a32995048af82b76877dedb80",
    "docs/mamem_recorded_event_source_v1_contract.md": "81b9097ffeaf19c1c6dcb3bd500055c452c22ab0439b66bbd9a959458198acc2",
}
ARMS = ("Q", "Q2", "QM", "SHAM")
GENERATED_PERIODS_MS = (75.5, 66.5, 58.5, 52.5, 43.5)


def read_json(path, cap=2 * 1024**2):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "regular_json_file")
    require(path.stat().st_size <= cap, "json_read_cap")
    return json.loads(path.read_bytes(), parse_constant=lambda _: require(False, "json_nonfinite"))


def phase_subjects(phase):
    require(phase in ("generated", "development", "real"), "phase")
    return (
        [f"FAKE{n}" for n in range(4)]
        if phase == "generated"
        else ["S001"]
        if phase == "development"
        else [f"S{n:03}" for n in range(2, 12)]
    )


def validate_selected(selected):
    require(isinstance(selected, list), "selected_list")
    require(
        [(r["subject"], r["run"]) for r in selected]
        == [(f"S{n:03}", run) for n in range(1, 12) for run in "ab"],
        "exact_22_members",
    )
    for row in selected:
        require(
            set(row) == {"subject", "run", "archive", "member", "bytes", "crc32"}, "selected_keys"
        )
        archive = Path(row["archive"])
        require(archive.parent == SOURCE and archive.name in ARCHIVE_SHA, "archive_scope")
        require(
            row["member"]
            in (
                row["subject"] + row["run"] + ".mat",
                archive.stem + "/" + row["subject"] + row["run"] + ".mat",
            ),
            "member_scope",
        )
        require(
            type(row["bytes"]) is int
            and 0 < row["bytes"] <= 512 * 1024**2
            and re.fullmatch(r"[0-9a-f]{8}", row["crc32"]) is not None,
            "member_size_crc",
        )
    require(sum(r["bytes"] for r in selected[1:]) <= 4 * 1024**3, "planned_extraction_cap")


def validate_manifest(cfg):
    require(
        set(cfg)
        == {
            "schema",
            "created_utc",
            "deadline_utc",
            "selected",
            "archives",
            "inventory_sha256",
            "code_sha256",
            "development_subject",
            "max_real_fits",
            "max_generated_fits",
            "query_ready_elapsed",
        },
        "manifest_keyset",
    )
    require(
        cfg["schema"] == "cfeg.mamem-source-v2" and cfg["deadline_utc"] == DEADLINE,
        "frozen_schema_deadline",
    )
    require(
        cfg["archives"] == ARCHIVE_SHA
        and cfg["inventory_sha256"] == INVENTORY_SHA
        and cfg["development_subject"] == "S001"
        and cfg["max_real_fits"] == 80
        and cfg["max_generated_fits"] == 0
        and cfg["query_ready_elapsed"] == "UNKNOWN",
        "frozen_budget_source",
    )
    require(set(cfg["code_sha256"]) == set(CODE), "exact_code_pin_keyset")
    require(
        all(cfg["code_sha256"][name] == value for name, value in UNCHANGED.items()),
        "unchanged_v1_pins",
    )
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", v) for v in cfg["code_sha256"].values()),
        "code_pin_format",
    )
    validate_selected(cfg["selected"])


def require(value, reason):
    if not value:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining(cap):
    delta = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    require(delta > 0, "round_deadline")
    return min(cap, delta)


def sha(path, operation_deadline=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining(1)
            require(
                operation_deadline is None or time.monotonic() < operation_deadline,
                "hash_operation_deadline",
            )
            digest.update(chunk)
    return digest.hexdigest()


def serial(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError("unsupported_json")


def save(path, value, cap=2 * 1024**2):
    raw = json.dumps(value, default=serial, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    require(len(raw) <= cap, "json_cap")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def append(phase, kind, payload):
    remaining(1)
    path = RUN / (phase + "_ledger.jsonl")
    require(not path.exists() or path.stat().st_size < 256 * 1024, "ledger_cap")
    raw = json.dumps(dict(kind=kind, utc=now(), **payload), allow_nan=False).encode() + b"\n"
    require((path.stat().st_size if path.exists() else 0) + len(raw) <= 256 * 1024, "ledger_cap")
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def budget():
    require(shutil.disk_usage(CACHE.parent).free >= 14 * 1024**3, "free_reserve")
    if CACHE.exists():
        files = [p for p in CACHE.iterdir() if p.is_file()]
        require(
            sum(p.stat().st_size for p in files if p.suffix == ".mat") <= 4 * 1024**3,
            "extraction_cap",
        )
        require(sum(p.stat().st_size for p in files if p.suffix != ".mat") <= 1024**3, "cache_cap")


def freeze():
    remaining(1)
    budget()
    require(not RUN.exists() and not CACHE.exists(), "fresh_round_paths_required")
    require(sha(SOURCE / "inventory.json") == INVENTORY_SHA, "inventory_hash")
    inventory = json.loads((SOURCE / "inventory.json").read_bytes())
    selected = []
    for listing in inventory["listings"]:
        archive = listing["archive"]["Path"]
        require(
            str(Path(archive).parent) == str(SOURCE) and Path(archive).name in ARCHIVE_SHA,
            "archive_scope",
        )
        for member in listing["members"]:
            match = re.fullmatch(r"(?:EEG-SSVEP-Part[12]/)?(S\d{3})([ab])\.mat", member["Path"])
            if match:
                size = int(member["Size"])
                require(
                    0 < size <= 512 * 1024**2 and member["Folder"] == "-", "member_size_or_kind"
                )
                selected.append(
                    {
                        "subject": match[1],
                        "run": match[2],
                        "archive": archive,
                        "member": member["Path"],
                        "bytes": size,
                        "crc32": member["CRC"].lower(),
                    }
                )
    selected.sort(key=lambda r: (r["subject"], r["run"]))
    require(
        [(r["subject"], r["run"]) for r in selected]
        == [(f"S{n:03}", r) for n in range(1, 12) for r in "ab"],
        "exact_22_members",
    )
    require(sum(r["bytes"] for r in selected[1:]) <= 4 * 1024**3, "planned_extraction_cap")
    validate_selected(selected)
    code_pins = {name: sha(ROOT / name) for name in CODE}
    require(all(code_pins[name] == value for name, value in UNCHANGED.items()), "unchanged_v1_pins")
    RUN.mkdir()
    CACHE.mkdir()
    save(
        RUN / "manifest.json",
        {
            "schema": "cfeg.mamem-source-v2",
            "created_utc": now(),
            "deadline_utc": DEADLINE,
            "selected": selected,
            "archives": ARCHIVE_SHA,
            "inventory_sha256": INVENTORY_SHA,
            "code_sha256": code_pins,
            "development_subject": "S001",
            "max_real_fits": 80,
            "max_generated_fits": 0,
            "query_ready_elapsed": "UNKNOWN",
        },
    )


def manifest():
    remaining(1)
    require(not RUN.is_symlink() and not CACHE.is_symlink(), "round_directory_symlink")
    cfg = read_json(RUN / "manifest.json")
    validate_manifest(cfg)
    require(
        all(
            not (ROOT / name).is_symlink()
            and (ROOT / name).stat().st_size <= 2 * 1024**2
            and sha(ROOT / name) == expected
            for name, expected in cfg["code_sha256"].items()
        ),
        "frozen_code_hash",
    )
    require(sha(SOURCE / "inventory.json") == INVENTORY_SHA, "inventory_hash")
    inventory = read_json(SOURCE / "inventory.json")
    for row in cfg["selected"]:
        matches = [
            m
            for listing in inventory["listings"]
            if listing["archive"]["Path"] == row["archive"]
            for m in listing["members"]
            if m["Path"] == row["member"]
        ]
        require(
            len(matches) == 1
            and int(matches[0]["Size"]) == row["bytes"]
            and matches[0]["Folder"] == "-"
            and matches[0]["CRC"].lower() == row["crc32"],
            "inventory_selection_binding",
        )
    return cfg


def child_limits(kind):
    memory, cpu = (2, 90) if kind == "io" else (4, 600)
    resource.setrlimit(resource.RLIMIT_AS, (memory * 1024**3, memory * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))


def ledger_rows(phase):
    path = RUN / (phase + "_ledger.jsonl")
    if not path.exists():
        return []
    require(not path.is_symlink() and path.stat().st_size <= 256 * 1024, "ledger_read_cap")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    require(
        all(isinstance(row, dict) and isinstance(row.get("kind"), str) for row in rows),
        "ledger_record_shape",
    )
    return rows


def validate_fit_ledger(phase):
    rows = [r for r in ledger_rows(phase) if r["kind"].startswith("FIT_")]
    keys = (
        []
        if phase == "generated"
        else [(s, k, arm) for s in phase_subjects("real") for k in (1, 2) for arm in ARMS]
    )
    require(len(rows) == 2 * len(keys), "unique_fit_ledger_count")
    for n, (subject, k, arm) in enumerate(keys):
        for offset, kind in enumerate(("FIT_STARTED", "FIT_COMPLETE")):
            row = rows[2 * n + offset]
            require(
                set(row) == {"kind", "utc", "target", "k", "arm", "fit_index"}
                and row["kind"] == kind
                and row["target"] == subject
                and row["k"] == k
                and row["arm"] == arm
                and row["fit_index"] == n + 1,
                "unique_ordered_fit_ledger",
            )
    return len(keys)


def validate_io_ledger(phase):
    names = (
        []
        if phase == "generated"
        else [subject + run for subject in phase_subjects(phase) for run in "ab"]
    )
    rows = ledger_rows(phase)
    relevant = [row for row in rows if not row["kind"].startswith("FIT_")]
    kinds = ("EXTRACTION_STARTED", "EXTRACTION_COMPLETE", "IO_STARTED", "IO_COMPLETE")
    require(len(relevant) == 4 * len(names), "complete_io_ledger_count")
    for n, name in enumerate(names):
        for offset, kind in enumerate(kinds):
            row = relevant[n * 4 + offset]
            require(row["kind"] == kind and row["file"] == name, "complete_io_ledger_order")
            if kind == "IO_COMPLETE":
                require(
                    row["trial_receipt_sha256"] == sha(CACHE / (name + "_trials.json")),
                    "completed_io_receipt_hash",
                )
    if phase != "real":
        require(not any(row["kind"].startswith("FIT_") for row in rows), "nonreal_zero_fits")


def phase_complete(phase):
    terminal = read_json(RUN / (phase + "_terminal.json"))
    require(
        terminal.get("phase") == phase
        and terminal.get("status") == "COMPLETE"
        and terminal.get("attempts") == 1
        and terminal.get("validated") is True
        and terminal.get("fits") == (80 if phase == "real" else 0)
        and terminal.get("files")
        == (0 if phase == "generated" else 2 if phase == "development" else 20),
        "validated_phase_terminal",
    )
    return True


def validate_preflight(report, phase):
    subjects = phase_subjects(phase)
    require(phase in ("generated", "real"), "preflight_phase")
    require(
        set(report)
        == {"folds", "total_folds", "total_scalar_oracles", "metadata_extracted_from_query"},
        "preflight_keyset",
    )
    require(
        report["total_folds"] == 2 * len(subjects)
        and report["total_scalar_oracles"] == (240 if phase == "generated" else 1800)
        and report["metadata_extracted_from_query"] is False,
        "preflight_counts",
    )
    require(
        [(f["target"], f["k"]) for f in report["folds"]]
        == [(s, k) for s in subjects for k in (1, 2)],
        "preflight_fold_ids",
    )
    for fold in report["folds"]:
        source = sorted(set(subjects) - {fold["target"]})
        require(fold["source"] == source, "preflight_source_exclusion")
        require(
            fold["prior_audit"]
            == [
                {
                    "target": fold["target"],
                    "pseudo_target": s,
                    "contributors": sorted(set(source) - {s}),
                }
                for s in source
            ],
            "preflight_pseudo_exclusion",
        )
        sham = fold["sham"]
        require(
            sorted(sham["donor_indices"]) == list(range(len(source) * 5)), "sham_donor_permutation"
        )
        require(
            0 <= sham["changed_fraction"] <= sham["exact_changed_fraction"] <= 1
            and 0 <= sham["singleton_rows"] <= len(source) * 5
            and 0 <= fold["degenerate_oracles"] <= len(source) * 10,
            "preflight_numeric_counts",
        )
    require(
        any(f["sham"]["changed_fraction"] > 0 for f in report["folds"]),
        "preflight_sham_eligibility",
    )


def validate_trial_receipt(name, cfg=None):
    cfg = manifest() if cfg is None else cfg
    expected = [r for r in cfg["selected"] if r["subject"] + r["run"] == name]
    require(len(expected) == 1, "trial_selected_file")
    row = expected[0]
    inp = read_json(CACHE / (name + "_input.json"))
    require(
        set(inp) == set(row) | {"path", "sha256"}
        and all(inp[key] == value for key, value in row.items()),
        "input_receipt_binding",
    )
    expected_input = (
        SOURCE / "development_first.mat" if name == "S001a" else CACHE / (name + ".mat")
    )
    require(
        inp["path"] == str(expected_input) and re.fullmatch(r"[0-9a-f]{64}", inp["sha256"]),
        "input_receipt_path_hash",
    )
    receipt = read_json(CACHE / (name + "_trials.json"))
    require(
        set(receipt)
        == {
            "subject",
            "run",
            "records",
            "feature_path",
            "feature_sha256",
            "mat_sha256",
            "fully_decoded_variables",
            "numerical_eeg_scope",
            "physical_units",
            "labels",
            "metadata_extracted",
        },
        "trial_receipt_keys",
    )
    require(
        receipt["subject"] == row["subject"]
        and receipt["run"] == row["run"]
        and receipt["mat_sha256"] == inp["sha256"]
        and receipt["metadata_extracted"] is (row["run"] == "a")
        and receipt["fully_decoded_variables"] == ["eeg", "DIN_1", "samplingRate"],
        "trial_role_scope",
    )
    feature = CACHE / (name + "_features.npz")
    require(
        receipt["feature_path"] == str(feature)
        and not feature.is_symlink()
        and feature.is_file()
        and 0 < feature.stat().st_size <= 16 * 1024**2
        and sha(feature) == receipt["feature_sha256"],
        "exact_feature_path_hash",
    )
    records = receipt["records"]
    require(
        len(records) == 15 and [r["group_index"] for r in records] == list(range(8, 23)),
        "trial_group_coverage",
    )
    labels = [r["label"] for r in records]
    require(
        sorted(labels) == [j for j in range(5) for _ in range(3)]
        and all(len(set(labels[i : i + 3])) == 1 for i in range(0, 15, 3)),
        "trial_label_coverage",
    )
    previous = -1
    for record in records:
        require(
            set(record)
            == {"group_index", "label", "start0", "end0", "event_count", "metadata", "trial_end0"},
            "trial_record_keys",
        )
        require(
            all(
                type(record[key]) is int
                for key in ("group_index", "label", "start0", "end0", "event_count", "trial_end0")
            ),
            "trial_integer_fields",
        )
        require(
            previous < record["start0"]
            and record["end0"] == record["start0"] + 500
            and record["trial_end0"] == record["start0"] + 1000
            and 0 <= record["start0"]
            and record["trial_end0"] < 500000
            and record["event_count"] >= (4 if row["run"] == "a" else 0),
            "trial_window_order",
        )
        metadata = record["metadata"]
        require(
            (
                isinstance(metadata, list)
                and len(metadata) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in metadata)
            )
            if row["run"] == "a"
            else metadata is None,
            "trial_metadata_role",
        )
        previous = record["start0"]
    return receipt


def finite_shape(value, shape, lower=-math.inf, upper=math.inf):
    if not shape:
        return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper
    return (
        isinstance(value, list)
        and len(value) == shape[0]
        and all(finite_shape(item, shape[1:], lower, upper) for item in value)
    )


def validate_saved_fold_arrays(fold):
    require(finite_shape(fold["oracle"], (45, 2), 0, 1), "oracle_shape_range")
    require(
        set(fold["train_x"]) == set(ARMS) and set(fold["eval_x"]) == set(ARMS), "input_arm_keys"
    )
    for arm in ARMS:
        dim = 16 if arm == "Q" else 18
        model = fold["models"][arm]
        require(
            set(model) == {"mean", "scale", "coef", "intercept"}
            and finite_shape(model["mean"], (dim,))
            and finite_shape(model["scale"], (dim,))
            and all(v > 0 for v in model["scale"])
            and finite_shape(model["coef"], (dim, 2))
            and finite_shape(model["intercept"], (2,))
            and finite_shape(fold["lambdas"][arm], (5, 2), 0, 1)
            and finite_shape(fold["train_x"][arm], (45, dim))
            and finite_shape(fold["eval_x"][arm], (5, dim)),
            "saved_model_input_dimensions",
        )
    all_arms = {*ARMS, "TARGET_ONLY", "SOURCE_ONLY", "ZERO_SHOT"}
    require(
        set(fold["scores"]) == set(fold["predictions"]) == set(fold["accuracy"]) == all_arms,
        "output_arm_keys",
    )
    truth = fold["truth"]
    require(
        isinstance(truth, list)
        and all(type(v) is int for v in truth)
        and sorted(truth) == [j for j in range(5) for _ in range(3)],
        "query_truth_coverage",
    )
    for arm in all_arms:
        scores, predictions = fold["scores"][arm], fold["predictions"][arm]
        require(
            finite_shape(scores, (15, 5))
            and isinstance(predictions, list)
            and len(predictions) == 15
            and all(type(v) is int for v in predictions)
            and predictions == [max(range(5), key=row.__getitem__) for row in scores],
            "saved_score_prediction_shape",
        )
        require(
            fold["accuracy"][arm]
            == sum(a == b for a, b in zip(predictions, truth, strict=True)) / 15,
            "saved_accuracy_arithmetic",
        )
    require(
        type(fold["support_prefix_samples"]) is int
        and 0 < fold["support_prefix_samples"] < 500000
        and fold["support_prefix_seconds"] == fold["support_prefix_samples"] / 250
        and fold["query_ready_elapsed_seconds"] is None,
        "saved_cost_scope",
    )


def validate_fit_result(phase):
    preflight_path = RUN / (phase + "_preflight.json")
    preflight = read_json(preflight_path)
    validate_preflight(preflight, phase)
    summary = read_json(RUN / (phase + "_summary.json"))
    expected_fits = 0 if phase == "generated" else 80
    require(
        summary["phase"] == phase
        and summary["status"] == "COMPLETE"
        and summary["fits"] == expected_fits
        and summary["subjects"] == phase_subjects(phase)
        and summary["preflight_path"] == str(preflight_path)
        and summary["preflight_sha256"] == sha(preflight_path),
        "fit_summary_scope",
    )
    require(validate_fit_ledger(phase) == expected_fits, "fit_summary_ledger")
    if phase == "generated":
        require(
            set(summary)
            == {
                "phase",
                "status",
                "fits",
                "subjects",
                "preflight_path",
                "preflight_sha256",
                "completed_utc",
                "decision",
            }
            and summary["decision"] == "GENERATED_PREFLIGHT_ONLY_NOT_EFFICACY"
            and not (CACHE / "generated_full_result.json").exists(),
            "generated_zero_fit_only",
        )
        return
    artifact = CACHE / "real_full_result.json"
    require(
        set(summary)
        == {
            "phase",
            "status",
            "fits",
            "subjects",
            "preflight_path",
            "preflight_sha256",
            "completed_utc",
            "result_path",
            "result_sha256",
            "summary",
        }
        and summary["result_path"] == str(artifact)
        and sha(artifact) == summary["result_sha256"],
        "real_result_path_hash",
    )
    result = read_json(artifact, 16 * 1024**2)
    require(
        set(result) == {"fits", "folds", "psd_fallback_count"}
        and result["fits"] == 80
        and [(f["target"], f["k"]) for f in result["folds"]]
        == [(s, k) for s in phase_subjects("real") for k in (1, 2)],
        "real_result_ids_counts",
    )
    for fold, sealed in zip(result["folds"], preflight["folds"], strict=True):
        require(
            fold["fit_count"] == 4
            and set(fold["models"]) == set(ARMS)
            and set(fold["lambdas"]) == set(ARMS)
            and fold["prior_audit"] == sealed["prior_audit"]
            and fold["sham"] == sealed["sham"]
            and len(fold["oracle"]) == 45
            and len(fold["rows"]) == 45,
            "real_fold_model_role_count",
        )
        require(
            sorted((row["subject"], row["label"], row["k"]) for row in fold["rows"])
            == sorted((s, j, fold["k"]) for s in sealed["source"] for j in range(5)),
            "real_saved_source_rows",
        )
        validate_saved_fold_arrays(fold)
    require(summary["summary"]["subjects"] == phase_subjects("real"), "real_summary_subjects")


def validate_child_output(kind, args, status, pid):
    mode, argument = args
    phase = (
        argument
        if mode == "fit-child"
        else ("development" if argument.startswith("S001") else "real")
    )
    claim = read_json(RUN / ("_".join(args) + "_claim.json"))
    require(
        claim["pid"] == pid
        and claim["parent_pid"] == os.getpid()
        and claim["phase"] == phase
        and claim["args"] == args
        and claim["one_attempt"] is True,
        "child_claim_identity",
    )
    if kind == "io":
        require(
            status
            == {
                "status": "FEATURES_COMPLETE",
                "phase": phase,
                "file": argument,
                "windows": 15,
                "fits": 0,
            },
            "io_child_status",
        )
        validate_trial_receipt(argument)
    else:
        require(
            status
            == {"phase": phase, "status": "COMPLETE", "fits": 0 if phase == "generated" else 80},
            "fit_child_status",
        )
        validate_fit_result(phase)


def subprocess_child(kind, args):
    require(
        kind in ("io", "fit")
        and len(args) == 2
        and args[0] == ("io-child" if kind == "io" else "fit-child"),
        "child_launch_scope",
    )
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "src"),
    )
    # stdout is a small status object; stderr has a fixed parent file-size bound.
    tag = "_".join(args)
    phase = (
        args[1]
        if args[0] == "fit-child"
        else ("development" if args[1].startswith("S001") else "real")
    )
    save(
        RUN / (tag + "_authorization.json"),
        {
            "phase": phase,
            "kind": kind,
            "args": args,
            "parent_pid": os.getpid(),
            "script_sha256": sha(__file__),
            "manifest_sha256": sha(RUN / "manifest.json"),
        },
    )
    with (
        (RUN / (tag + "_stdout.txt")).open("xb") as out,
        (RUN / (tag + "_stderr.txt")).open("xb") as err,
    ):
        deadline = time.monotonic() + remaining(120 if kind == "io" else 600)
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *args], stdout=out, stderr=err, env=env
        )
        try:
            while proc.poll() is None:
                require(time.monotonic() < deadline, "child_wall_timeout")
                require(out.tell() <= 65536 and err.tell() <= 65536, "child_capture_cap")
                time.sleep(0.05)
            require(proc.returncode == 0, "child_nonzero_exit")
            require(out.tell() <= 65536 and err.tell() <= 65536, "child_capture_cap")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    status = read_json(RUN / (tag + "_stdout.txt"), 65536)
    validate_child_output(kind, args, status, proc.pid)


def extract(row, phase):
    operation_deadline = time.monotonic() + remaining(120)
    parent_cpu_started = time.process_time()
    name = row["subject"] + row["run"]
    append(phase, "EXTRACTION_STARTED", {"file": name, "member": row["member"]})
    if name == "S001a":
        path = SOURCE / "development_first.mat"
        require(
            path.stat().st_size == row["bytes"] and sha(path, operation_deadline) == DEV_SHA,
            "pinned_existing_development_mat",
        )
    else:
        from prepare_mamem_i_probe_v1 import bounded_run

        path = CACHE / (name + ".mat")
        budget()
        with path.open("xb") as stream:
            code, _, _ = bounded_run(
                [
                    "/usr/bin/unar",
                    "-q",
                    "-nr",
                    "-k",
                    "skip",
                    "-o",
                    "-",
                    row["archive"],
                    row["member"],
                ],
                row["bytes"],
                min(remaining(120), max(0.001, operation_deadline - time.monotonic())),
                stream,
                disk_guard=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        require(code == 0 and path.stat().st_size == row["bytes"], "extraction_failed_or_size")
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining(1)
            require(time.monotonic() < operation_deadline, "extraction_operation_deadline")
            crc = zlib.crc32(chunk, crc)
    require(f"{crc:08x}" == row["crc32"], "mat_crc")
    receipt = dict(row, path=str(path), sha256=sha(path, operation_deadline))
    require(
        time.monotonic() < operation_deadline and time.process_time() - parent_cpu_started < 90,
        "extraction_operation_limit",
    )
    save(CACHE / (name + "_input.json"), receipt)
    append(
        phase,
        "EXTRACTION_COMPLETE",
        {"file": name, "bytes": row["bytes"], "sha256": receipt["sha256"]},
    )


def io_child(name):
    cfg = manifest()
    import numpy as np
    from cfeg.mamem_events_v2 import parse_main_trials
    from scipy.io import loadmat, whosmat

    from cfeg.mamem_signal_v1 import analyze_window

    row = read_json(CACHE / (name + "_input.json"))
    expected = [r for r in cfg["selected"] if r["subject"] + r["run"] == name]
    require(
        len(expected) == 1 and all(row.get(k) == v for k, v in expected[0].items()),
        "input_selection_binding",
    )
    path = Path(row["path"])
    exact_path = SOURCE / "development_first.mat" if name == "S001a" else CACHE / (name + ".mat")
    require(
        path == exact_path and path.is_file() and path.stat().st_size == row["bytes"],
        "exact_input_path_and_size",
    )
    require(not path.is_symlink() and sha(path) == row["sha256"], "mat_changed")
    headers = whosmat(path)
    for variable in ("eeg", "DIN_1", "samplingRate"):
        require(sum(v[0] == variable for v in headers) == 1, "missing_or_duplicate_variable")
    shapes = {v[0]: v[1] for v in headers}
    classes = {v[0]: v[2] for v in headers}
    require(classes["eeg"] == "double" and classes["DIN_1"] == "cell", "variable_classes")
    require(
        shapes["samplingRate"] == (1, 1)
        and classes["samplingRate"]
        in (
            "double",
            "single",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "uint64",
            "int64",
        ),
        "sampling_scalar_header",
    )
    require(
        len(shapes["eeg"]) == 2 and shapes["eeg"][0] == 257 and shapes["eeg"][1] < 500000,
        "eeg_header",
    )
    require(
        len(shapes["DIN_1"]) == 2 and shapes["DIN_1"][0] == 4 and shapes["DIN_1"][1] <= 10000,
        "din_header",
    )
    loaded = loadmat(
        path,
        variable_names=["eeg", "DIN_1", "samplingRate"],
        squeeze_me=False,
        struct_as_record=True,
        verify_compressed_data_integrity=True,
    )
    require(
        set(loaded) <= {"__header__", "__version__", "__globals__", "eeg", "DIN_1", "samplingRate"}
        and all(k in loaded for k in ("eeg", "DIN_1", "samplingRate")),
        "loaded_key_whitelist",
    )
    rate = loaded["samplingRate"]
    require(
        rate.size == 1 and rate.dtype.kind in "iuf" and float(rate.item()) == 250.0, "sampling_rate"
    )
    records = parse_main_trials(
        loaded["DIN_1"], shapes["eeg"][1], include_metadata=row["run"] == "a"
    )
    arrays = {key: [] for key in ("covariance", "factors", "q", "q2", "zero_shot")}
    for record in records:
        result = analyze_window(loaded["eeg"], record["start0"], record["end0"])
        for key, values in arrays.items():
            values.append(result[key])
    feature_path = CACHE / (name + "_features.npz")
    with feature_path.open("xb") as stream:
        np.savez(stream, **{key: np.stack(value) for key, value in arrays.items()})
    require(feature_path.stat().st_size <= 16 * 1024**2, "feature_file_cap")
    save(
        CACHE / (name + "_trials.json"),
        {
            "subject": row["subject"],
            "run": row["run"],
            "records": records,
            "feature_path": str(feature_path),
            "feature_sha256": sha(feature_path),
            "mat_sha256": row["sha256"],
            "fully_decoded_variables": ["eeg", "DIN_1", "samplingRate"],
            "numerical_eeg_scope": "15 fixed windows, rows[0:256], 500 samples each",
            "physical_units": "UNVERIFIED",
            "labels": "MOABB double-integer compatibility key; inferred nominal label, not independent ground truth",
            "metadata_extracted": row["run"] == "a",
        },
    )
    print(
        json.dumps(
            {
                "status": "FEATURES_COMPLETE",
                "phase": "development" if name.startswith("S001") else "real",
                "file": name,
                "windows": len(records),
                "fits": 0,
            }
        )
    )


def load_data(subjects):
    import numpy as np

    require(subjects == phase_subjects("real"), "exact_real_load_subjects")
    cfg = manifest()
    data = {}
    receipts = ledger_rows("real")
    io_rows = [r for r in receipts if r["kind"] == "IO_COMPLETE"]
    require(
        [r["file"] for r in io_rows] == [s + run for s in subjects for run in "ab"],
        "all_source_io_sealed",
    )
    completed = {
        r["file"]: r["trial_receipt_sha256"] for r in receipts if r["kind"] == "IO_COMPLETE"
    }
    for subject in subjects:
        data[subject] = {}
        for run in "ab":
            name = subject + run
            trial_path = CACHE / (name + "_trials.json")
            require(sha(trial_path) == completed.get(name), "trial_receipt_binding")
            receipt = validate_trial_receipt(name, cfg)
            require(
                receipt["subject"] == subject
                and receipt["run"] == run
                and receipt["feature_path"] == str(CACHE / (name + "_features.npz")),
                "exact_feature_path_and_role",
            )
            require(sha(receipt["feature_path"]) == receipt["feature_sha256"], "feature_changed")
            records = receipt["records"]
            with np.load(receipt["feature_path"], allow_pickle=False) as arrays:
                require(
                    set(arrays.files) == {"covariance", "factors", "q", "q2", "zero_shot"},
                    "cache_keys",
                )
                shapes = {
                    "covariance": (15, 256, 256),
                    "factors": (15, 5, 2, 256, 2),
                    "q": (15, 5, 2, 4),
                    "q2": (15, 5, 2),
                    "zero_shot": (15, 5),
                }
                cached = {key: arrays[key] for key in shapes}
                require(
                    all(
                        value.shape == shapes[key]
                        and value.dtype == np.float64
                        and np.isfinite(value).all()
                        for key, value in cached.items()
                    ),
                    "cache_shape_dtype_finite",
                )
                for n, record in enumerate(records):
                    record["metadata"] = (
                        None if record["metadata"] is None else np.array(record["metadata"])
                    )
                    record.update({key: value[n].copy() for key, value in cached.items()})
            data[subject][run] = records
    return data


def generated_data():
    import numpy as np
    from cfeg.mamem_events_v2 import FREQUENCIES, parse_main_trials

    from cfeg.mamem_signal_v1 import analyze_window

    data = {}
    rng = np.random.default_rng(20260913)
    labels = [0, 1, 2, 3, 4, 0, 1, 2] + [j for j in (3, 0, 4, 1, 2) for _ in range(3)]
    total = 101 + 2500 * 22 + 1750
    for person in range(4):
        subject = f"FAKE{person}"
        data[subject] = {}
        for run in "ab":
            groups = []
            for g, label in enumerate(labels):
                period = GENERATED_PERIODS_MS[label]
                rel = np.arange(int(5000 / period) + 1) * period
                # Same samples/counts but variable tiny timestamp residuals: a
                # generated input/integration canary, not physical timing evidence.
                times = (
                    1000
                    + 10000 * g
                    + rel
                    + (0.04 + person * 0.03) * np.sin(np.arange(len(rel)) * 1.7)
                )
                samples = 101 + 2500 * g + np.rint(rel / 4).astype(int)
                group = np.empty((4, len(rel)), object)
                group[0] = None
                group[2] = None
                group[1] = times
                group[3] = samples
                groups.append(group)
            records = parse_main_trials(
                np.concatenate(groups, axis=1), total, include_metadata=run == "a"
            )
            eeg = np.zeros((257, total), dtype=np.float64)
            for row in records:
                t = np.arange(500) / 250
                waveform = np.sin(2 * np.pi * FREQUENCIES[row["label"]] * t)
                spatial = rng.normal(size=(256, 1))
                eeg[:256, row["start0"] : row["end0"]] = (
                    spatial * waveform + rng.normal(size=(256, 500)) * 0.5
                )
                row.update(analyze_window(eeg, row["start0"], row["end0"]))
            data[subject][run] = records
    return data


def fit_child(phase):
    require(phase in ("generated", "real"), "fit_phase")
    manifest()
    from cfeg.mamem_shrinkage_v1 import preflight, prepare, run_folds, summarize

    if phase == "real":
        validate_io_ledger("real")
    data = (
        generated_data() if phase == "generated" else load_data([f"S{n:03}" for n in range(2, 12)])
    )
    prepared = prepare(data)
    folds = preflight(prepared)
    expected_ids = (
        [f"FAKE{n}" for n in range(4)]
        if phase == "generated"
        else [f"S{n:03}" for n in range(2, 12)]
    )
    require(sorted(data) == expected_ids, "exact_subject_coverage")
    require(
        [(f["target"], f["k"]) for f in folds] == [(s, k) for s in expected_ids for k in (1, 2)],
        "exact_fold_coverage",
    )
    # Persist ALL fold eligibility before the first fit or efficacy output.
    eligibility = [
        {
            key: f[key]
            for key in ("target", "k", "source", "sham", "prior_audit", "degenerate_oracles")
        }
        for f in folds
    ]
    save(
        RUN / (phase + "_preflight.json"),
        {
            "folds": eligibility,
            "total_folds": len(folds),
            "total_scalar_oracles": sum(f["y"].size for f in folds),
            "metadata_extracted_from_query": False,
        },
    )
    summary = {
        "phase": phase,
        "fits": 0,
        "status": "COMPLETE",
        "subjects": expected_ids,
        "preflight_path": str(RUN / (phase + "_preflight.json")),
        "preflight_sha256": sha(RUN / (phase + "_preflight.json")),
        "completed_utc": now(),
    }
    validate_preflight(read_json(RUN / (phase + "_preflight.json")), phase)
    if phase == "generated":
        summary["decision"] = "GENERATED_PREFLIGHT_ONLY_NOT_EFFICACY"
        save(RUN / "generated_summary.json", summary)
        print(json.dumps({"phase": phase, "status": "COMPLETE", "fits": 0}))
        return
    result = run_folds(prepared, folds, lambda kind, value: append(phase, kind, value))
    require(result["fits"] == 80, "fit_count")
    artifact = CACHE / (phase + "_full_result.json")
    save(artifact, result, cap=16 * 1024**2)
    summary.update(
        {
            "fits": result["fits"],
            "status": "COMPLETE",
            "result_path": str(artifact),
            "result_sha256": sha(artifact),
            "summary": summarize(result),
            "completed_utc": now(),
        }
    )
    save(RUN / (phase + "_summary.json"), summary)
    print(json.dumps({"phase": phase, "status": "COMPLETE", "fits": result["fits"]}))


def execute(phase):
    cfg = manifest()
    require(phase in ("generated", "development", "real"), "phase")
    if phase != "generated":
        require(
            phase_complete("generated"),
            "generated_required",
        )
    if phase == "real":
        require(
            phase_complete("development"),
            "development_required",
        )
    save(
        RUN / (phase + "_attempt.json"),
        {
            "phase": phase,
            "started_utc": now(),
            "attempt": 1,
            "parent_pid": os.getpid(),
            "manifest_sha256": sha(RUN / "manifest.json"),
        },
    )
    terminal = {"phase": phase, "started_utc": now(), "attempts": 1}
    try:
        budget()
        if phase == "generated":
            subprocess_child("fit", ["fit-child", phase])
        else:
            if phase == "development":
                # Rehash both pinned archives before ANY newly selected extraction.
                for name, expected in ARCHIVE_SHA.items():
                    require(sha(SOURCE / name) == expected, "archive_hash")
                save(RUN / "archive_verification.json", {"archives": ARCHIVE_SHA, "utc": now()})
            for row in cfg["selected"]:
                if (row["subject"] == "S001") != (phase == "development"):
                    continue
                extract(row, phase)
                name = row["subject"] + row["run"]
                append(phase, "IO_STARTED", {"file": name})
                subprocess_child("io", ["io-child", name])
                append(
                    phase,
                    "IO_COMPLETE",
                    {"file": name, "trial_receipt_sha256": sha(CACHE / (name + "_trials.json"))},
                )
            if phase == "real":
                subprocess_child("fit", ["fit-child", phase])
        if phase in ("generated", "real"):
            validate_fit_result(phase)
        validate_io_ledger(phase)
        terminal.update(
            status="COMPLETE",
            fits=80 if phase == "real" else 0,
            files=0 if phase == "generated" else 2 if phase == "development" else 20,
            validated=True,
        )
    except Exception as exc:  # noqa: BLE001 -- terminal records must preserve every failure without retry
        terminal.update(
            status="STOPPED_NO_RETRY",
            error_type=type(exc).__name__,
            reason=str(exc)
            if isinstance(exc, ValueError)
            or (type(exc).__name__ == "Stop" and type(exc).__module__ == "probe_mamem_i_din_v1")
            else "details_in_bounded_child_logs",
        )
    terminal["ended_utc"] = now()
    try:
        entries = ledger_rows(phase)
    except Exception:  # noqa: BLE001 -- never lose the terminal on a malformed partial ledger
        entries = []
        terminal.update(status="STOPPED_NO_RETRY", validated=False, partial_ledger_unreadable=True)
    terminal["observed_counts"] = {
        kind: sum(row["kind"] == kind for row in entries)
        for kind in (
            "EXTRACTION_STARTED",
            "EXTRACTION_COMPLETE",
            "IO_STARTED",
            "IO_COMPLETE",
            "FIT_STARTED",
            "FIT_COMPLETE",
        )
    }
    save(RUN / (phase + "_terminal.json"), terminal)
    print(json.dumps(terminal))


def claim_child(mode, argument):
    """Exclusive worker claim; cannot invoke child paths outside the active parent."""
    require(mode in ("fit-child", "io-child"), "child_mode")
    if mode == "fit-child":
        require(argument in ("generated", "real"), "fit_phase")
        phase, kind = argument, "fit"
    else:
        require(re.fullmatch(r"S(?:00[1-9]|01[01])[ab]", argument or "") is not None, "io_name")
        phase, kind = ("development" if argument.startswith("S001") else "real"), "io"
    child_limits(kind)  # self-enforced before any NumPy/SciPy/raw/cache loading
    tag = mode + "_" + argument
    auth = read_json(RUN / (tag + "_authorization.json"))
    attempt = read_json(RUN / (phase + "_attempt.json"))
    require(
        auth["args"] == [mode, argument] and auth["phase"] == phase and auth["kind"] == kind,
        "child_authorization_scope",
    )
    require(auth["parent_pid"] == attempt["parent_pid"] == os.getppid(), "active_parent_required")
    require(
        attempt["phase"] == phase
        and attempt["attempt"] == 1
        and attempt["manifest_sha256"] == auth["manifest_sha256"] == sha(RUN / "manifest.json"),
        "attempt_manifest_binding",
    )
    require(
        auth["script_sha256"] == sha(__file__) and not (RUN / (phase + "_terminal.json")).exists(),
        "active_frozen_attempt",
    )
    if phase != "generated":
        require(
            phase_complete("generated"),
            "child_generated_gate",
        )
    if phase == "real":
        require(
            phase_complete("development"),
            "child_development_gate",
        )
    save(
        RUN / (tag + "_claim.json"),
        {
            "claimed_utc": now(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "phase": phase,
            "args": [mode, argument],
            "one_attempt": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "execute", "io-child", "fit-child"])
    parser.add_argument("argument", nargs="?")
    args = parser.parse_args()
    if args.mode == "freeze":
        require(args.argument is None, "freeze_no_argument")
        freeze()
    elif args.mode == "execute":
        execute(args.argument)
    elif args.mode == "io-child":
        claim_child(args.mode, args.argument)
        io_child(args.argument)
    else:
        claim_child(args.mode, args.argument)
        fit_child(args.argument)


if __name__ == "__main__":
    main()

"""One selective metadata pass on the fixed cohort; no waveform arrays requested."""

import argparse
import hashlib
import json
import os
import re
import resource
import signal
import time
from pathlib import Path

import numpy as np
from read_mobilebci_event_time_v2 import Projector, require
from scipy.io import loadmat

FIELDS = ["raw_fs", "raw_clab", "event", "t"]
EEG = ["Pz", "PO3", "POz", "PO4", "PO7", "PO8", "O1", "Oz", "O2"]
GYRO = ["HgyroX", "HgyroY", "HgyroZ"]


def plain(node):
    if node["kind"] == "struct":
        return {key: plain(value) for key, value in node["fields"].items()}
    if node["kind"] in ("cell", "sequence"):
        return [plain(value) for value in node["items"]]
    return node["value"] if node["kind"] == "scalar" else node["values"]


def normalize(spec, values):
    rate = float(values["raw_fs"])
    labels = values["raw_clab"]
    event = values["event"]
    classes = event["className"]
    codes = np.asarray(event["decs"])
    onehot = np.asarray(event["y"])
    times = np.asarray(event["time"], dtype=float)
    epoch_t = np.asarray(values["t"])
    expected_fs = 128.0 if spec["device"] == "IMU" else 500.0
    expected_channels = 27 if spec["device"] == "IMU" else 36
    needed = GYRO if spec["device"] == "IMU" else EEG
    issues = []

    def check(ok, name):
        if not ok:
            issues.append(name)

    check(rate == expected_fs, "raw_rate")
    require(np.isfinite(rate) and rate > 0, "invalid_rate_value")
    require(isinstance(labels, list) and all(isinstance(x, str) for x in labels), "label_structure")
    check(len(labels) == expected_channels and len(set(labels)) == len(labels), "raw_labels")
    check(all(name in labels for name in needed), "selected_channels_missing")
    check(classes == ["5.45", "8.57", "12"], "class_names")
    code_ok = codes.shape == (60,) and bool(np.isin(codes, [1, 2, 3]).all())
    check(code_ok, "event_codes")
    check(onehot.shape == (3, 60), "onehot_shape")
    if code_ok and onehot.shape == (3, 60):
        check(
            bool(
                np.isin(onehot, [0, 1]).all()
                and (onehot.sum(0) == 1).all()
                and (onehot.argmax(0) + 1 == codes).all()
            ),
            "onehot_agreement",
        )
        check([int((codes == c).sum()) for c in [1, 2, 3]] == [20, 20, 20], "class_counts")
    time_ok = times.shape == (60,) and bool(np.isfinite(times).all())
    check(time_ok and bool((np.diff(times) > 0).all()), "event_time_order")
    marker_samples = []
    grid_error = None
    if time_ok:
        candidate = (times - 1) * rate / 1000
        grid_error = float(np.max(np.abs(candidate - np.rint(candidate))))
        check(grid_error <= 1e-6, "marker_sample_grid")
        bounds = bool(
            (candidate - round(0.5 * rate) >= 0).all()
            and (candidate + round(1.5 * rate) <= spec["raw_shape"][0]).all()
        )
        check(bounds, "marker_window_bounds")
        if bounds:
            marker_samples = np.rint(candidate).astype(np.int64).tolist()
    check(
        epoch_t.shape == (400,) and bool(np.array_equal(epoch_t, np.arange(1, 401) * 10)),
        "epoch_axis",
    )
    check(spec["raw_shape"][1] == len(labels), "raw_shape_channel_count")
    return {
        key: spec[key]
        for key in ["id", "name", "path", "sha256", "subject", "speed", "device", "raw_shape"]
    } | {
        "raw_fs": rate,
        "raw_labels": labels,
        "class_names": classes,
        "event_codes": codes.tolist(),
        "event_times_ms_candidate": times.tolist(),
        "marker_samples_candidate": marker_samples,
        "sample_grid_max_error": grid_error,
        "event_onehot": onehot.tolist(),
        "onehot_agrees": code_ok and onehot.shape == (3, 60) and "onehot_agreement" not in issues,
        "epoch_t": epoch_t.tolist(),
        "issues": issues,
    }


def read_record(spec, limits, loader=loadmat, counters=None):
    path = Path(spec["path"])
    before = path.stat()
    require(path.resolve(strict=True) == path and before.st_size == spec["bytes"], "input_stat")
    digest = hashlib.sha256()
    if counters is not None:
        counters["checksum_passes"] += 1
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    require(digest.hexdigest() == spec["sha256"], "input_sha256")
    if counters is not None:
        counters["decode_calls"] += 1
    values = loader(path, variable_names=FIELDS, simplify_cells=True)
    require(
        set(values) - {"__header__", "__version__", "__globals__"} == set(FIELDS), "selected_fields"
    )
    projector = Projector(limits)
    try:
        sanitized = {field: plain(projector.project(values[field], field)) for field in FIELDS}
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"projection_failure:{projector.context}:{type(error).__name__}"
        ) from error
    record = normalize(spec, sanitized)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "input_changed",
    )
    record["projector_visited_units"] = projector.visited_units
    return record


def paired_roles(records):
    grouped = {}
    for row in records:
        key = (row["subject"], row["speed"])
        require(row["device"] not in grouped.get(key, {}), "duplicate_device_role")
        grouped.setdefault(key, {})[row["device"]] = row
    pairs = []
    for (subject, speed), devices in sorted(grouped.items()):
        issues = []
        pair = {"subject": subject, "speed": speed, "issues": issues, "budgets": {}}
        if set(devices) != {"IMU", "scalp"}:
            issues.append("pair_missing")
            pairs.append(pair)
            continue
        imu, eeg = devices["IMU"], devices["scalp"]
        pair["file_ids"] = [imu["id"], eeg["id"]]
        if imu["issues"] or eeg["issues"]:
            issues.append("file_metadata_issue")
        if imu["event_codes"] != eeg["event_codes"] or imu["class_names"] != eeg["class_names"]:
            issues.append("paired_event_classes")
        if issues:
            pairs.append(pair)
            continue
        codes = np.asarray(eeg["event_codes"])
        times = np.asarray(eeg["event_times_ms_candidate"])
        delta = np.asarray(imu["event_times_ms_candidate"]) - times
        pair["event_delta_ms_candidate"] = {
            "min": float(delta.min()),
            "max": float(delta.max()),
            "abs_max": float(np.abs(delta).max()),
        }
        pair["query_indices"] = list(range(40, 60))
        pair["query_class_counts"] = [int((codes[40:] == c).sum()) for c in [1, 2, 3]]
        if min(pair["query_class_counts"]) == 0:
            issues.append("query_class_missing")
        for k in [1, 2, 3, 5]:
            selected = sorted(
                np.concatenate([np.flatnonzero(codes == c)[:k] for c in [1, 2, 3]]).tolist()
            )
            end = selected[-1]
            if len(selected) != 3 * k or end >= 40:
                issues.append(f"support_query_overlap_k{k}")
            ready = (times[end] - 1) / 1000 + 1.5
            gap = float((times[40] - 1) / 1000 - ready)
            if gap < 0:
                issues.append(f"support_query_time_overlap_k{k}")
            pair["budgets"][str(k)] = {
                "support_indices": selected,
                "learner_trials": len(selected),
                "acquired_prefix_trials": end + 1,
                "marker_relative_ready_seconds": float(ready),
                "unused_gap_to_query_marker_seconds": gap,
            }
        pairs.append(pair)
    subjects = sorted({row["subject"] for row in pairs})
    eligible = [
        subject
        for subject in subjects
        if len(rows := [row for row in pairs if row["subject"] == subject]) == 3
        and all(not row["issues"] for row in rows)
    ]
    return pairs, eligible


def main(config):
    source = Path(config["acquisition_path"]).read_bytes()
    require(hashlib.sha256(source).hexdigest() == config["acquisition_sha256"], "acquisition_pin")
    require(
        len(config["files"]) == 96 and len({row["id"] for row in config["files"]}) == 96,
        "cohort_count",
    )
    acquired = json.loads(source)
    originals = {row["id"]: row for row in acquired["records"] + acquired["reused"]}
    require({row["id"] for row in config["files"]} == set(originals), "exact_cohort_ids")
    for key in ["name", "path"]:
        require(len({row[key] for row in config["files"]}) == 96, f"unique_{key}")
    original_pair_bytes = Path(config["original_pair_path"]).read_bytes()
    require(
        hashlib.sha256(original_pair_bytes).hexdigest() == config["original_pair_sha256"],
        "original_pair_pin",
    )
    original_pair = {row["id"]: row for row in json.loads(original_pair_bytes)["records"]}
    for spec in config["files"]:
        row = originals[spec["id"]]
        require(
            all(row[key] == spec[key] for key in ["name", "path", "bytes", "sha256"]), "file_pin"
        )
        match = re.fullmatch(r"(s\d{2})_(IMU|scalp)_SSVEP_(0\.0|0\.8|1\.6)\.mat", spec["name"])
        require(match is not None, "file_name")
        require(
            [spec["subject"], spec["device"], spec["speed"]] == list(match.groups()),
            "file_role_pin",
        )
        schema_record = row if "top_level_schema" in row else original_pair[spec["id"]]
        raw_shape = next(
            entry["shape"]
            for entry in schema_record["top_level_schema"]
            if entry["name"] == "raw_x"
        )
        require(spec["raw_shape"] == raw_shape, "raw_shape_pin")
    limits = config["limits"]
    output_path, journal_path = Path(config["output_path"]), Path(config["journal_path"])
    require(not output_path.exists() and not journal_path.exists(), "no_restart")
    result = {
        "schema": "cfeg.mobilebci-cohort-metadata-observation.v1",
        "records": [],
        "attempted_files": 0,
        "checksum_passes": 0,
        "decode_calls": 0,
        "waveform_numeric_decodes": 0,
        "human_fits": 0,
        "human_outcomes": 0,
        "journal_path": str(journal_path),
        "acquisition_sha256": config["acquisition_sha256"],
    }
    started = time.monotonic()
    try:
        with journal_path.open("x") as journal:
            for spec in config["files"]:
                remaining = limits["batch_seconds"] - (time.monotonic() - started)
                require(remaining > 0, "batch_deadline")
                signal.setitimer(signal.ITIMER_REAL, min(limits["file_seconds"], remaining))
                result["attempted_files"] += 1
                row = read_record(spec, limits, counters=result)
                line = json.dumps(row, allow_nan=False)
                journal.write(line + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                result["records"].append(row)
                signal.setitimer(signal.ITIMER_REAL, 0)
        remaining = limits["batch_seconds"] - (time.monotonic() - started)
        require(remaining > 0, "batch_deadline")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        pairs, eligible = paired_roles(result["records"])
        result.update(
            status="COMPLETE_METADATA_PASS",
            pairs=pairs,
            eligible_subjects=eligible,
            proposed_design_ready=len(eligible) >= 12 and len(pairs) == 48,
        )
        # Account for normal report serialization within the batch timer. Durable
        # failure serialization and fsync remain outside the compute deadline.
        bounded_report(result, limits["output_bytes"])
    except (
        ValueError,
        OSError,
        MemoryError,
        KeyError,
        TypeError,
        OverflowError,
        EOFError,
        IndexError,
        AttributeError,
        RuntimeError,
    ) as error:
        result.update(
            status="STOPPED_NO_RETRY", error_type=type(error).__name__, error=str(error)[:256]
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result["seconds"] = round(time.monotonic() - started, 6)
    rendered = bounded_report(result, limits["output_bytes"])
    with output_path.open("x") as output:
        output.write(rendered + "\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in [
                    "status",
                    "attempted_files",
                    "eligible_subjects",
                    "proposed_design_ready",
                    "seconds",
                    "error",
                ]
            }
        )
    )


def bounded_report(result, limit):
    try:
        rendered = json.dumps(result, indent=2, allow_nan=False)
        require(len(rendered.encode()) <= limit, "output_limit")
    except (ValueError, TypeError, MemoryError):
        result.update(status="STOPPED_OUTPUT_LIMIT", metadata_values_in_journal=True)
        result["records"] = [
            {"id": row["id"], "name": row["name"], "issues": row["issues"]}
            for row in result["records"]
        ]
        result.pop("pairs", None)
        result.pop("eligible_subjects", None)
        result["proposed_design_ready"] = False
        rendered = json.dumps(result, allow_nan=False)
        require(len(rendered.encode()) <= limit, "compact_output_limit")
    return rendered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    limit = config["limits"]["address_space_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def deadline(_signum, _frame):
        raise TimeoutError("metadata_deadline")

    signal.signal(signal.SIGALRM, deadline)
    main(config)

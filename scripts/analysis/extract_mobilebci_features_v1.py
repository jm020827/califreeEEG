"""One pinned raw_x pass, only fixed support/query features enter the cache."""

import argparse
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cfeg.analysis.mobilebci_features import (
    eeg_trials,
    gyro_trials,
    query_statistics,
    support_statistics,
)

EEG = ["Pz", "PO3", "POz", "PO4", "PO7", "PO8", "O1", "Oz", "O2"]
GYRO = ["HgyroX", "HgyroY", "HgyroZ"]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_raw(spec, record, state):
    path = Path(spec["path"])
    before = path.stat()
    require(path.resolve(strict=True) == path and before.st_size == spec["bytes"], "file_stat")
    digest = hashlib.sha256()
    state["checksum_passes"] += 1
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    require(digest.hexdigest() == spec["sha256"], "file_checksum")
    state["waveform_decode_calls"] += 1
    values = loadmat(path, variable_names=["raw_x"], simplify_cells=True)
    require(set(values) - {"__header__", "__version__", "__globals__"} == {"raw_x"}, "raw_only")
    raw = values["raw_x"]
    require(list(raw.shape) == record["raw_shape"] and raw.dtype.kind == "f", "raw_shape_dtype")
    state["decoded_numeric_elements"] += raw.size
    state["decoded_numeric_bytes"] += raw.nbytes
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "file_changed",
    )
    return raw


def run(config):
    metadata_bytes = Path(config["metadata_path"]).read_bytes()
    manifest_bytes = Path(config["manifest_path"]).read_bytes()
    require(hashlib.sha256(metadata_bytes).hexdigest() == config["metadata_sha256"], "metadata_pin")
    require(hashlib.sha256(manifest_bytes).hexdigest() == config["manifest_sha256"], "manifest_pin")
    metadata, manifest = json.loads(metadata_bytes), json.loads(manifest_bytes)
    require(
        metadata["proposed_design_ready"] and len(metadata["eligible_subjects"]) == 16,
        "cohort_ready",
    )
    records = {row["id"]: row for row in metadata["records"]}
    specs = {row["id"]: row for row in manifest["files"]}
    require(len(records) == 96 and set(records) == set(specs), "cohort_ids")
    paths = {key: Path(config[key]) for key in ["cache_path", "report_path", "journal_path"]}
    require(all(not path.exists() for path in paths.values()), "no_restart")
    state = {
        "schema": "cfeg.mobilebci-feature-extraction.v1",
        "status": "RUNNING",
        "checksum_passes": 0,
        "waveform_decode_calls": 0,
        "decoded_numeric_elements": 0,
        "decoded_numeric_bytes": 0,
        "human_fits": 0,
        "human_outcomes": 0,
        "query_imu_feature_trials": 0,
        "metadata_sha256": config["metadata_sha256"],
        "cache_path": str(paths["cache_path"]),
        "runs": [],
    }
    arrays = {}
    started = time.monotonic()
    limits = config["limits"]

    def timer(file=False):
        remaining = limits["batch_seconds"] - (time.monotonic() - started)
        require(remaining > 0, "batch_deadline")
        signal.setitimer(
            signal.ITIMER_REAL, min(remaining, limits["file_seconds"]) if file else remaining
        )

    try:
        with paths["journal_path"].open("x") as journal:
            for pair in metadata["pairs"]:
                require(not pair["issues"], "pair_metadata_issues")
                selected = pair["budgets"]["5"]["support_indices"]
                queries = pair["query_indices"]
                trial_maps = {}
                codes = None
                for file_id in pair["file_ids"]:
                    timer(file=True)
                    state.update(current_file_id=file_id, phase="raw_decode")
                    record = records[file_id]
                    raw = read_raw(specs[file_id], record, state)
                    state["phase"] = "selected_trial_extraction"
                    imu = record["device"] == "IMU"
                    channels = [record["raw_labels"].index(name) for name in (GYRO if imu else EEG)]
                    indices = selected if imu else selected + queries
                    function = gyro_trials if imu else eeg_trials
                    trials = function(raw, record["marker_samples_candidate"], indices, channels)
                    trial_maps[record["device"]] = dict(zip(indices, trials))
                    if not imu:
                        codes = np.asarray(record["event_codes"], dtype=np.int64) - 1
                    journal.write(
                        json.dumps(
                            {
                                "id": file_id,
                                "device": record["device"],
                                "sha256": record["sha256"],
                                "raw_numeric_elements": int(raw.size),
                                "raw_numeric_bytes": int(raw.nbytes),
                                "feature_trial_indices": indices,
                                "feature_channels": GYRO if imu else EEG,
                            }
                        )
                        + "\n"
                    )
                    journal.flush()
                    os.fsync(journal.fileno())
                    del raw
                timer()
                state["phase"] = "run_statistics"
                run_arrays = query_statistics(np.stack([trial_maps["scalp"][i] for i in queries]))
                run_arrays["query_labels"] = codes[queries]
                per_budget = {}
                floors = {}
                for k in [1, 2, 3, 5]:
                    budget = pair["budgets"][str(k)]
                    use = budget["support_indices"]
                    result, floors[str(k)] = support_statistics(
                        np.stack([trial_maps["scalp"][i] for i in use]),
                        codes[use],
                        np.stack([trial_maps["IMU"][i] for i in use]),
                    )
                    result["common"] = np.array(
                        [
                            float(pair["speed"]),
                            ["0.0", "0.8", "1.6"].index(pair["speed"]),
                            k,
                            budget["acquired_prefix_trials"],
                            budget["marker_relative_ready_seconds"],
                        ]
                    )
                    for key, value in result.items():
                        per_budget.setdefault(key, []).append(value)
                run_arrays.update({key: np.stack(value) for key, value in per_budget.items()})
                for key, value in run_arrays.items():
                    arrays.setdefault(key, []).append(value)
                state["runs"].append(
                    {"subject": pair["subject"], "speed": pair["speed"], "floors": floors}
                )
                del trial_maps
            timer()
            state["phase"] = "cache_serialization"
            arrays = {key: np.stack(value) for key, value in arrays.items()}
            require(
                sum(value.nbytes for value in arrays.values()) <= limits["cache_bytes"],
                "cache_array_budget",
            )
            require(all(np.isfinite(value).all() for value in arrays.values()), "cache_finite")
            with paths["cache_path"].open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            require(
                paths["cache_path"].stat().st_size <= limits["cache_bytes"], "cache_file_budget"
            )
            state.update(
                status="COMPLETE_FEATURE_EXTRACTION",
                cache_bytes=paths["cache_path"].stat().st_size,
                cache_sha256=hashlib.sha256(paths["cache_path"].read_bytes()).hexdigest(),
                cache_shapes={key: list(value.shape) for key, value in arrays.items()},
            )
    except (
        ValueError,
        OSError,
        MemoryError,
        KeyError,
        TypeError,
        EOFError,
        IndexError,
        AttributeError,
        RuntimeError,
        OverflowError,
    ) as error:
        state.update(
            status="STOPPED_NO_RETRY", error_type=type(error).__name__, error=str(error)[:256]
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    state["seconds"] = round(time.monotonic() - started, 6)
    with paths["report_path"].open("x") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: state.get(key)
                for key in ["status", "seconds", "waveform_decode_calls", "cache_bytes", "error"]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    resource.setrlimit(resource.RLIMIT_AS, (config["limits"]["address_space_bytes"],) * 2)

    def deadline(_sig, _frame):
        raise TimeoutError("extraction_deadline")

    signal.signal(signal.SIGALRM, deadline)
    run(config)

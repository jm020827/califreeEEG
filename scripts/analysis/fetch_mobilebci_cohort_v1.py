"""Finite cohort orchestration using the unchanged verified public-pair transport."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from fetch_mobilebci_public_pair_v1 import download, require

SUCCESS = "DOWNLOADED_VERIFIED_TOP_LEVEL_SCHEMA"


def progress(value):
    # Durable per-file journal is authoritative; a closed observer pipe must not
    # turn a completed file download into an unrecorded batch crash.
    try:
        print(json.dumps(value), flush=True)
    except (BrokenPipeError, OSError):
        pass


def selected_specs(inventory):
    ids = {
        file_id
        for bundle in inventory["subject_bundles"]
        if bundle["complete"]
        for file_id in bundle["file_ids"]
    }
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "bytes": row["size"],
            "md5": row["computed_md5"],
            "url": row["download_url"],
        }
        for row in inventory["ssvep_files"]
        if row["id"] in ids
    ]


def fetch_batch(files, directory, budget, reporter, downloader=download):
    started = time.monotonic()
    records = []
    for item in files:
        remaining = budget["batch_wall_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            return records, "STOPPED_BATCH_DEADLINE"
        per_file = {**budget, "file_wall_seconds": min(budget["file_wall_seconds"], remaining)}
        record = downloader(item, directory, per_file)
        records.append(record)
        reporter(record)
        if record["status"] != SUCCESS:
            return records, "STOPPED_FIRST_FAILURE_NO_RETRY"
    return records, "COMPLETE_ACQUISITION_SCHEMA_ONLY"


def run(config):
    source = Path(config["inventory_path"]).read_bytes()
    require(hashlib.sha256(source).hexdigest() == config["inventory_sha256"], "inventory_pin")
    inventory = json.loads(source)
    expected = selected_specs(inventory)
    require(len(expected) == 96 and len({item["id"] for item in expected}) == 96, "cohort_size")
    require(config["selected_files"] == expected, "selected_files_pin")
    require(
        sum(item["bytes"] for item in expected) <= config["budget"]["total_retained_bytes_max"],
        "retained_cap",
    )
    require(len(config["reuse"]) == 2, "reuse_count")
    reuse_ids = {item["id"] for item in config["reuse"]}
    pending = [item for item in expected if item["id"] not in reuse_ids]
    require(len(pending) == 94, "new_file_count")
    directory = Path(config["output_dir"])
    require(
        directory.resolve(strict=True) == directory and not list(directory.iterdir()),
        "fresh_output_dir",
    )
    report_path = Path(config["report_path"])
    require(not report_path.exists(), "report_exists_no_restart")
    with (directory / "attempt.once").open("x") as marker:
        marker.write("mobilebci-cohort-v1\n")
    reused = []
    for item in config["reuse"]:
        path = Path(item["path"])
        spec = next(row for row in expected if row["id"] == item["id"])
        before = path.stat()
        require(path.resolve(strict=True) == path and before.st_size == spec["bytes"], "reuse_stat")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        require(digest.hexdigest() == item["sha256"], "reuse_sha256")
        after = path.stat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "reuse_changed",
        )
        reused.append(
            {
                **spec,
                "path": str(path),
                "sha256": digest.hexdigest(),
                "status": "REUSED_CHECKSUM_VERIFIED",
            }
        )
    started = time.monotonic()
    with (directory / "acquisition.jsonl").open("x") as journal:

        def record_result(record):
            journal.write(json.dumps(record) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            progress(
                {
                    "index": len(records_progress) + 1,
                    "name": record["name"],
                    "status": record["status"],
                    "bytes": record.get("bytes", 0),
                    "seconds": record["elapsed_seconds"],
                }
            )
            records_progress.append(record["id"])

        records_progress = []
        records, status = fetch_batch(pending, directory, config["budget"], record_result)
    result = {
        "schema": "cfeg.mobilebci-cohort-acquisition-observation.v1",
        "status": status,
        "output_dir": str(directory),
        "reused": reused,
        "records": records,
        "new_initial_gets": len(records),
        "new_verified_files": sum(row["status"] == SUCCESS for row in records),
        "new_verified_bytes": sum(
            row.get("bytes", 0) for row in records if row["status"] == SUCCESS
        ),
        "seconds": round(time.monotonic() - started, 6),
        "waveform_numeric_decodes": 0,
        "event_value_decodes": 0,
        "fits": 0,
        "outcomes": 0,
        "held60": 0,
    }
    with report_path.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    progress(
        {
            "status": status,
            "report_path": str(report_path),
            "new_verified_files": result["new_verified_files"],
            "seconds": result["seconds"],
        }
    )


def selftest():
    from unittest.mock import patch

    files = [{"id": 1}, {"id": 2}]
    budget = {"batch_wall_seconds": 10, "file_wall_seconds": 3}
    seen = []

    def success(item, directory, limits):
        assert 0 < limits["file_wall_seconds"] <= 3
        return {"id": item["id"], "status": SUCCESS}

    records, status = fetch_batch(files, None, budget, seen.append, success)
    assert len(records) == len(seen) == 2 and status == "COMPLETE_ACQUISITION_SCHEMA_ONLY"
    records, status = fetch_batch(
        files, None, budget, lambda row: None, lambda *args: {"status": "FAILED"}
    )
    assert len(records) == 1 and status == "STOPPED_FIRST_FAILURE_NO_RETRY"
    records, status = fetch_batch(
        files,
        None,
        {**budget, "batch_wall_seconds": 0},
        lambda row: None,
        lambda *args: (_ for _ in ()).throw(AssertionError("should_not_start")),
    )
    assert records == [] and status == "STOPPED_BATCH_DEADLINE"
    inventory = {
        "subject_bundles": [
            {"complete": True, "file_ids": [1]},
            {"complete": False, "file_ids": [2]},
        ],
        "ssvep_files": [
            {"id": 1, "name": "generated", "size": 3, "computed_md5": "a", "download_url": "b"},
            {"id": 2},
        ],
    }
    assert selected_specs(inventory) == [
        {"id": 1, "name": "generated", "bytes": 3, "md5": "a", "url": "b"}
    ]
    with patch("builtins.print", side_effect=BrokenPipeError("generated observer closed")):
        progress({"generated": True})
    print(json.dumps({"status": "PASS", "checks": 5, "human_reads": 0, "network_requests": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        run(json.loads(args.config.read_text()))

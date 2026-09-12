"""Pinned archive-only acquisition. No archive extraction or EEG/DIN interpretation."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from fetch_mobilebci_public_pair_v1 import BoundedRedirect, Stop, allowed_url, copy_payload, require


GIB = 1024**3


def emit(row):
    try:
        print(json.dumps(row), flush=True)
    except (BrokenPipeError, OSError):
        pass


def reserve_check(free, remaining, reserve, extraction):
    require(free >= remaining + reserve + extraction, "disk_reserve_for_remaining_payload")


def remaining_seconds(manifest, now):
    frozen = datetime.fromisoformat(manifest["frozen_at_utc"]).timestamp()
    overall = datetime.fromisoformat(manifest["overall_deadline_utc"]).timestamp()
    remaining = min(frozen + 2700, overall) - now
    require(remaining > 0, "manifest_expired_no_network")
    require(now >= frozen, "manifest_from_future")
    return remaining


class GuardedWriter:
    def __init__(self, stream, directory, state, budget, clock=time.monotonic):
        self.stream, self.directory, self.state, self.budget = stream, directory, state, budget
        self.clock = clock
        self.last_report = clock()

    def write(self, chunk):
        require(self.clock() < self.state["deadline"], "batch_deadline")
        reserve_check(
            shutil.disk_usage(self.directory).free,
            self.state["remaining"],
            self.budget["reserve_bytes"],
            self.budget["extraction_bytes"],
        )
        count = self.stream.write(chunk)
        require(count == len(chunk), "short_disk_write")
        self.state["remaining"] -= count
        if self.clock() - self.last_report >= 20:
            emit({"status": "DOWNLOADING", "remaining_bytes": self.state["remaining"]})
            self.last_report = self.clock()
        return count


def exact_specs(catalogue):
    p = catalogue["projection"]
    require(catalogue["status"] == 200 and p["id"] == 2068677 and p["version"] == 1,
            "catalogue_identity")
    require(p["license"]["name"] == "CC BY 4.0", "catalogue_license")
    result = [
        {"id": f["id"], "name": f["name"], "bytes": f["size"],
         "md5": f["computed_md5"], "url": f["download_url"]}
        for f in p["files"] if f["name"].endswith(".rar")
    ]
    result.sort(key=lambda x: x["name"])
    require([x["id"] for x in result] == [3687681, 3687756], "archive_ids")
    require(sum(x["bytes"] for x in result) == 6585019498, "archive_total")
    for item in result:
        require(Path(item["name"]).name == item["name"], "unsafe_filename")
        require(allowed_url(item["url"]), "archive_host")
    return result


def download_one(spec, directory, state, budget):
    started = time.monotonic()
    partial, target = directory / (spec["name"] + ".part"), directory / spec["name"]
    require(not partial.exists() and not target.exists(), "no_overwrite")
    row = {"id": spec["id"], "name": spec["name"], "attempts": 1, "status": "STARTED"}
    redirect = BoundedRedirect(5)
    try:
        reserve_check(shutil.disk_usage(directory).free, state["remaining"],
                      budget["reserve_bytes"], budget["extraction_bytes"])
        require(time.monotonic() < state["deadline"], "batch_deadline")
        request = urllib.request.Request(spec["url"], headers={
            "Cache-Control": "no-cache", "Accept-Encoding": "identity"})
        with urllib.request.build_opener(redirect).open(request, timeout=30) as response:
            row["http_status"] = response.status
            require(response.status == 200, "expected_http_200")
            length = response.headers.get("Content-Length")
            require(length is None or int(length) == spec["bytes"], "content_length")
            with partial.open("xb") as stream:
                writer = GuardedWriter(stream, directory, state, budget)
                size, md5, sha = copy_payload(response, writer, spec["bytes"], 1024**2)
                stream.flush()
                os.fsync(stream.fileno())
            row.update(bytes=size, md5=md5, sha256=sha)
            require(md5 == spec["md5"], "md5_mismatch")
            # Hard-link publication refuses an existing destination atomically.
            os.link(partial, target)
            partial.unlink()  # Same verified payload remains at target, no content loss.
            row.update(status="ARCHIVE_DOWNLOADED_VERIFIED", path=str(target))
    except urllib.error.HTTPError as e:
        row.update(status="HTTP_ACCESS_FAILURE", http_status=e.code)
        e.close()
    except Exception as e:
        row.update(status="STOPPED_NO_RETRY", error_type=type(e).__name__,
                   reason=str(e) if isinstance(e, Stop) else "details_not_exported")
    row.update(seconds=round(time.monotonic() - started, 3), redirects=redirect.count,
               partial_bytes=partial.stat().st_size if partial.exists() else 0)
    return row


def acquisition(specs, directory, state, budget, result, cancel_timer, downloader=download_one):
    try:
        with (directory / "acquisition.jsonl").open("x") as journal:
            for spec in specs:
                start = {"status": "STARTED", "id": spec["id"], "name": spec["name"],
                         "bytes_expected": spec["bytes"], "attempts": 1}
                journal.write(json.dumps(start) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                emit(start)
                row = downloader(spec, directory, state, budget)
                result["records"].append(row)
                journal.write(json.dumps(row) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                emit(row)
                if row["status"] != "ARCHIVE_DOWNLOADED_VERIFIED":
                    break
        result["status"] = (
            "BOTH_ARCHIVES_VERIFIED" if len(result["records"]) == 2 and
            all(r["status"] == "ARCHIVE_DOWNLOADED_VERIFIED" for r in result["records"])
            else "STOPPED_NO_RETRY")
    except Exception as e:
        result.update(status="STOPPED_NO_RETRY", outer_error_type=type(e).__name__,
                      outer_reason=str(e) if isinstance(e, Stop) else "details_not_exported")
    finally:
        # Persist the terminal receipt without an already exhausted timer firing again.
        cancel_timer()
        result["partial_files"] = [
            {"name": s["name"] + ".part", "bytes": (directory / (s["name"] + ".part")).stat().st_size}
            for s in specs if (directory / (s["name"] + ".part")).exists()
        ]
        result["free_bytes_end"] = shutil.disk_usage(directory).free
        with (directory / "acquisition.json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        emit({"status": result["status"], "report": str(directory / "acquisition.json")})


def run(manifest_path):
    manifest = json.loads(Path(manifest_path).read_bytes())
    seconds = remaining_seconds(manifest, time.time())
    raw = Path(manifest["catalogue_path"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest["catalogue_sha256"], "catalogue_pin")
    specs = exact_specs(json.loads(raw))
    require(manifest["files"] == specs, "manifest_files")
    budget = manifest["budget"]
    require(budget == {"total_bytes_max": 7 * GIB, "file_bytes_max": 4 * GIB,
                       "reserve_bytes": 8 * GIB, "extraction_bytes": 512 * 1024**2,
                       "seconds": 2700}, "fixed_budget")
    require(all(s["bytes"] <= budget["file_bytes_max"] for s in specs), "file_cap")
    total = sum(s["bytes"] for s in specs)
    require(total <= budget["total_bytes_max"], "total_cap")
    directory = Path(manifest["output_dir"])
    require(str(directory) == "/home/whwovy/data/mamem_i_v1_20260913", "output_scope")
    require(directory.parent.resolve(strict=True) == directory.parent, "parent_not_symlink")
    require(not directory.exists(), "fresh_directory_no_restart")
    reserve_check(shutil.disk_usage(directory.parent).free, total,
                  budget["reserve_bytes"], budget["extraction_bytes"])
    directory.mkdir()
    seconds = remaining_seconds(manifest, time.time())
    state = {"remaining": total, "deadline": time.monotonic() + seconds}

    def deadline(_signum, _frame):
        raise Stop("batch_wall_deadline")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    result = {"schema": "cfeg.mamem-i-archive-acquisition.v1", "records": [],
              "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              "output_dir": str(directory), "EEG_interpretation": 0, "DIN_interpretation": 0,
              "fits": 0, "held60": 0}
    try:
        acquisition(specs, directory, state, budget, result,
                    cancel_timer=lambda: signal.setitimer(signal.ITIMER_REAL, 0))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    run(parser.parse_args().manifest)

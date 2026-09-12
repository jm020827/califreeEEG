"""List two verified archives, freeze one dev member, or extract that member only."""
import argparse
import hashlib
import io
import json
import os
import re
import resource
import selectors
import shutil
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from probe_mamem_i_din_v1 import CAP, Stop, hash_file, require

DATA = Path("/home/whwovy/data/mamem_i_v1_20260913")


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def limits(file_limit=CAP):
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))


def remaining_wall(requested):
    manifest = Path(__file__).resolve().parents[2] / "configs/mamem_i_archive_acquisition_v1.json"
    end = datetime.fromisoformat(json.loads(manifest.read_bytes())["overall_deadline_utc"]).timestamp()
    remaining = end - time.time()
    require(remaining > 0, "overall_deadline_no_execution")
    return min(requested, remaining)


def bounded_run(command, stdout_limit, timeout, destination=None, disk_guard=False):
    """Drain both pipes incrementally, with independent output/memory bounds."""
    output = destination if destination is not None else io.BytesIO()
    counts = {"stdout": 0, "stderr": 0}
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=lambda: limits(stdout_limit))
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                require(time.monotonic() < deadline, "subprocess_wall_timeout")
                for key, _ in selector.select(min(0.5, max(0, deadline - time.monotonic()))):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    channel = key.data
                    counts[channel] += len(data)
                    require(counts[channel] <= (stdout_limit if channel == "stdout" else 1024**2),
                            channel + "_capture_cap")
                    if channel == "stdout":
                        if disk_guard:
                            require(shutil.disk_usage(DATA).free >= 8 * 1024**3 + len(data),
                                    "extraction_reserve_during_write")
                        output.write(data)
        code = proc.wait(timeout=max(0.001, deadline - time.monotonic()))
        return code, output.getvalue() if destination is None else None, counts
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()


def parse_listing(text):
    require(len(text.encode()) <= 1024**2, "listing_cap")
    prefix, body = text.split("\n----------\n", 1)
    archive = dict(line.split(" = ", 1) for line in prefix.splitlines() if " = " in line)
    require(archive.get("Type") in ("Rar", "Rar5"), "archive_type")
    require(archive.get("Solid") == "-" and archive.get("Multivolume") == "-"
            and archive.get("Volumes") == "1", "solid_or_multivolume_not_supported")
    members = []
    for block in body.strip().split("\n\n"):
        item = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        name = item.get("Path", "")
        require(name and not PurePosixPath(name).is_absolute() and ".." not in name.split("/")
                and "\\" not in name and ":" not in name, "unsafe_member_path")
        require(all(item.get(k) == "-" for k in ("Encrypted", "Solid", "Split Before", "Split After")),
                "unsupported_member_encoding")
        require(item.get("Folder") in ("+", "-"), "unknown_member_kind")
        require(item.get("Host OS") == "Win32" and item.get("Attributes") in ("A", "D"),
                "unsupported_member_attributes")
        members.append(item)
    return archive, members


def select_member(listings):
    candidates, names = [], set()
    for listing in listings:
        for item in listing["members"]:
            name = item["Path"]
            require(name not in names, "duplicate_member_path")
            names.add(name)
            if item["Folder"] == "-" and name.endswith(".mat"):
                candidates.append({"archive": listing["archive"]["Path"], "member": name,
                                   "bytes": int(item["Size"]), "crc32": item["CRC"].lower()})
    require(candidates, "no_MAT_members")
    first = min(candidates, key=lambda x: x["member"])
    require(0 < first["bytes"] <= CAP, "first_MAT_cap_no_replacement")
    match = re.fullmatch(r"EEG-SSVEP-Part[12]/(S[0-9]{3})[a-z]\.mat", first["member"])
    require(match is not None, "unknown_subject_filename_no_inference")
    return dict(first, subject=match[1], subject_role="development_only_all_records",
                member_selection="lexicographic_first_MAT_across_both_archives",
                frozen_utc=datetime.now(timezone.utc).isoformat())


def inventory():
    receipt = json.loads((DATA / "acquisition.json").read_bytes())
    require(receipt["status"] == "BOTH_ARCHIVES_VERIFIED", "both_archives_required")
    require(len(receipt["records"]) == 2, "two_archives_required")
    listings = []
    for row in receipt["records"]:
        archive = DATA / row["name"]
        require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size == row["bytes"],
                "archive_stat_mismatch")
        remaining_wall(30)
        require(hash_file(archive) == row["sha256"], "archive_hash_mismatch")
        code, raw, _ = bounded_run(["/usr/bin/7z", "l", "-slt", str(archive)], 1024**2, remaining_wall(30))
        require(code == 0, "archive_listing_failed")
        metadata, members = parse_listing(raw.decode())
        require(metadata["Path"] == str(archive), "listing_archive_path")
        listings.append({"archive": metadata, "members": members,
                         "listing_sha256": hashlib.sha256(raw).hexdigest()})
    selected = select_member(listings)
    payload = {"schema": "cfeg.mamem-i-archive-inventory.v1", "listings": listings,
               "selected": selected, "acquisition_sha256": hash_file(DATA / "acquisition.json")}
    save(DATA / "inventory.json", payload)
    print(json.dumps({"selected": selected, "MAT_count": sum(
        m["Folder"] == "-" and m["Path"].endswith(".mat") for a in listings for m in a["members"])}))


def extract_one():
    selection = json.loads((DATA / "inventory.json").read_bytes())["selected"]
    expected = selection["bytes"]
    require(0 < expected <= CAP, "extraction_cap")
    require(shutil.disk_usage(DATA).free >= 8 * 1024**3 + expected, "extraction_reserve")
    output = DATA / "development_first.mat.part"
    final = DATA / "development_first.mat"
    require(not output.exists() and not final.exists(), "extraction_no_retry_or_overwrite")
    require(str(Path(selection["archive"]).parent) == str(DATA), "archive_scope")
    require(re.fullmatch(r"EEG-SSVEP-Part[12]/S[0-9]{3}[a-z]\.mat", selection["member"]),
            "literal_member_required")
    remaining_wall(120)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "selected": selection,
              "EEG_interpretation": 0, "DIN_interpretation": 0, "attempts": 1}
    try:
        # stdout mode creates no archive-provided paths. Parent streaming bounds
        # the single output; selected safe non-solid member only.
        with output.open("xb") as stream:
            code, _, counts = bounded_run(
                ["/usr/bin/unar", "-q", "-nr", "-k", "skip", "-o", "-",
                 selection["archive"], selection["member"]], expected, remaining_wall(120), stream, disk_guard=True)
            stream.flush()
            os.fsync(stream.fileno())
        report.update(returncode=code, capture_counts=counts)
        require(code == 0, "unar_failed")
        require(output.stat().st_size == expected, "extracted_size_mismatch")
        crc = 0
        with output.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                crc = zlib.crc32(chunk, crc)
        report.update(bytes=output.stat().st_size, crc32=f"{crc:08x}", sha256=hash_file(output))
        require(report["crc32"] == selection["crc32"], "extracted_crc_mismatch")
        os.link(output, final)
        output.unlink()  # Verified bytes remain at the new final name.
        role = dict(selection, mat_path=str(final), mat_sha256=report["sha256"])
        save(DATA / "development_role.json", role)
        report.update(status="ONE_MEMBER_EXTRACTED_VERIFIED", path=str(final))
    except Exception as e:
        report.update(status="EXTRACTION_STOPPED_NO_RETRY", error_type=type(e).__name__,
                      reason=str(e) if isinstance(e, Stop) else "details_not_exported")
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save(DATA / "extraction.json", report)
    print(json.dumps(report))


def probe(mode):
    destination = DATA / (mode + "_probe.json")
    parent_report = DATA / (mode + "_parent_receipt.json")
    require(not destination.exists() and not parent_report.exists(), "probe_no_retry")
    result = {"mode": mode, "started_utc": datetime.now(timezone.utc).isoformat(), "attempts": 1}
    try:
        code, _, counts = bounded_run(
            [sys.executable, str(Path(__file__).with_name("probe_mamem_i_din_v1.py")),
             mode, str(DATA / "development_role.json"), str(destination)], 1024**2, remaining_wall(120))
        result.update(returncode=code, capture_counts=counts, report_exists=destination.is_file())
        require(code == 0 and destination.is_file(), "worker_failure_or_missing_report")
        result.update(status="WORKER_REPORT_SAVED", report_sha256=hash_file(destination))
    except Exception as e:
        result.update(status="WORKER_STOPPED_NO_RETRY", error_type=type(e).__name__,
                      reason=str(e) if isinstance(e, Stop) else "details_not_exported")
    result["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save(parent_report, result)
    print(json.dumps(result))


if __name__ == "__main__":
    import sys
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["inventory", "extract", "schema", "din"])
    mode = p.parse_args().mode
    if mode == "inventory":
        inventory()
    elif mode == "extract":
        extract_one()
    else:
        probe(mode)

"""List two verified archives, freeze one dev member, or extract that member only."""
import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
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
        result = subprocess.run(["/usr/bin/7z", "l", "-slt", str(archive)], capture_output=True,
                                text=True, timeout=30, preexec_fn=limits)
        require(result.returncode == 0, "archive_listing_failed")
        metadata, members = parse_listing(result.stdout)
        require(metadata["Path"] == str(archive), "listing_archive_path")
        listings.append({"archive": metadata, "members": members,
                         "listing_sha256": hashlib.sha256(result.stdout.encode()).hexdigest()})
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
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "selected": selection,
              "EEG_interpretation": 0, "DIN_interpretation": 0, "attempts": 1}
    try:
        # stdout mode creates no archive-provided paths. RLIMIT_FSIZE hard bounds
        # the single output descriptor; selected safe non-solid member only.
        with output.open("xb") as stream:
            run = subprocess.run(["/usr/bin/unar", "-q", "-nr", "-k", "skip", "-o", "-",
                                  selection["archive"], selection["member"]],
                                 stdout=stream, stderr=subprocess.PIPE, timeout=120,
                                 preexec_fn=lambda: limits(expected))
            stream.flush()
            os.fsync(stream.fileno())
        report["returncode"] = run.returncode
        require(run.returncode == 0, "unar_failed")
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["inventory", "extract"])
    mode = p.parse_args().mode
    inventory() if mode == "inventory" else extract_one()

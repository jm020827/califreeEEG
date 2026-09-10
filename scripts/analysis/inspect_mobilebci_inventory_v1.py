"""One exact public metadata GET, never a raw EEG/IMU download."""

import argparse
import hashlib
import json
import re
import signal
import time
import urllib.request
from pathlib import Path

URL = "https://api.figshare.com/v2/articles/13604078/versions/1"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main(config):
    if config["url"] != URL or config["limits"]["requests"] != 1:
        raise ValueError("unexpected_scope")
    started = time.monotonic()
    result = {"schema": "cfeg.mobilebci-public-inventory-observation.v1", "url": URL, "requests": 1}
    try:
        request = urllib.request.Request(
            URL, headers={"User-Agent": "califreeEEG-public-metadata/1"}
        )
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
            if response.status != 200:
                raise ValueError("unexpected_status")
            payload = response.read(config["limits"]["response_bytes"] + 1)
        if len(payload) > config["limits"]["response_bytes"]:
            raise ValueError("response_size")
        body = json.loads(payload)
        if body["id"] != 13604078 or body["version"] != 1:
            raise ValueError("release_identity")
        records = []
        for item in body["files"]:
            if "SSVEP" in item["name"]:
                records.append(
                    {
                        key: item[key]
                        for key in ["id", "name", "size", "computed_md5", "download_url"]
                    }
                )
        records.sort(key=lambda row: row["name"])
        subjects = sorted(
            {
                int(match.group(1))
                for row in records
                if (match := re.match(r"^s(\d+)_", row["name"]))
            }
        )
        speeds = ["0.0", "0.8", "1.6"]
        devices = ["IMU", "scalp"]
        bundles = []
        for subject in subjects:
            missing, duplicates, selected = [], [], []
            for speed in speeds:
                for device in devices:
                    name = f"s{subject:02d}_{device}_SSVEP_{speed}.mat"
                    hits = [row for row in records if row["name"] == name]
                    if not hits:
                        missing.append(name)
                    elif len(hits) != 1:
                        duplicates.append(name)
                    else:
                        selected.append(hits[0])
            bundles.append(
                {
                    "subject": f"s{subject:02d}",
                    "complete": not missing and not duplicates,
                    "missing": missing,
                    "duplicates": duplicates,
                    "file_ids": [row["id"] for row in selected],
                    "bytes": sum(row["size"] for row in selected),
                }
            )
        result.update(
            status="COMPLETE_PUBLIC_INVENTORY_ONLY",
            response_bytes=len(payload),
            response_sha256=hashlib.sha256(payload).hexdigest(),
            article_id=body["id"],
            version=body["version"],
            license=body["license"],
            description=body["description"],
            total_files=len(body["files"]),
            total_bytes=sum(row["size"] for row in body["files"]),
            ssvep_files=records,
            subject_bundles=bundles,
            complete_subjects=[row["subject"] for row in bundles if row["complete"]],
            complete_cohort_bytes=sum(row["bytes"] for row in bundles if row["complete"]),
            raw_downloads=0,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="STOPPED", error_type=type(error).__name__, error=str(error)[:256])
    result["seconds"] = round(time.monotonic() - started, 6)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    def deadline(_signum, _frame):
        raise TimeoutError("inventory_deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config["limits"]["seconds"])
    try:
        main(config)
    finally:
        signal.alarm(0)

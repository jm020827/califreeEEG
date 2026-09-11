"""One exact anonymous header probe; no sample parsing or presigned URL logging."""

import argparse
import csv
import hashlib
import io
import json
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://repo-prod.prod.sagebase.org/repo/v1"
TARGETS = (
    ("Neuroscan", "syn64072665", "syn64072621"),
    ("Tobii", "syn64086409", "syn64086404"),
)


def allowed_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and not parsed.username
        and not parsed.password
        and (host == "repo-prod.prod.sagebase.org" or host.endswith(".amazonaws.com"))
    )


class Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.in_request = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.in_request += 1
        if self.in_request > 2 or not allowed_url(newurl):
            raise ValueError("redirect_boundary")
        if self.state["network_requests_attempted"] >= 8:
            raise ValueError("request_budget")
        self.state["redirects"] += 1
        self.state["network_requests_attempted"] += 1
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def parse_header(line, modality):
    if not line or len(line) > 65536 or not line.endswith(b"\n"):
        raise ValueError("header_line_boundary")
    decoded = line.decode("utf-8-sig").strip("\r\n")
    delimiter = "\t" if decoded.count("\t") > decoded.count(",") else ","
    fields = next(csv.reader(io.StringIO(decoded), delimiter=delimiter, strict=True))
    fields = [f.strip() for f in fields]
    expected = (
        {"ValidityLeft", "ValidityRight"}
        if modality == "Tobii"
        else {"FP1", "Fp1", "FP2", "Fp2", "Oz", "OZ", "O1", "O2"}
    )
    if not set(fields).intersection(expected):
        raise ValueError("expected_header_not_found_no_sample_values_retained")
    if modality == "Tobii" and not expected <= set(fields):
        raise ValueError("both_validity_headers_required")
    return fields


def probe():
    state = {
        "schema": "cfeg.eye-bci-header-probe.v1",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "logical_requests_attempted": 0,
        "network_requests_attempted": 0,
        "redirects": 0,
        "records": [],
        "numeric_samples_decoded": 0,
    }
    redirects = Redirects(state)
    opener = urllib.request.build_opener(redirects)
    started = time.monotonic()

    def get(url):
        if not allowed_url(url):
            raise ValueError("url_boundary")
        if state["logical_requests_attempted"] >= 4 or state["network_requests_attempted"] >= 8:
            raise ValueError("request_budget")
        state["logical_requests_attempted"] += 1
        state["network_requests_attempted"] += 1
        redirects.in_request = 0
        return opener.open(
            urllib.request.Request(
                url,
                headers={
                    "User-Agent": "califreeEEG-header-only/1.0",
                    "Accept-Encoding": "identity",
                },
            ),
            timeout=15,
        )

    try:
        for modality, entity, parent in TARGETS:
            record = {"modality": modality, "entity": entity, "version": 1, "parent": parent}
            state["records"].append(record)
            with get(f"{BASE}/entity/{entity}/version/1") as response:
                body = response.read(1048577)
                if len(body) > 1048576:
                    raise ValueError("metadata_size_boundary")
                metadata = json.loads(body)
                if not isinstance(metadata, dict):
                    raise TypeError("metadata_object_required")
                record["metadata_bytes"] = len(body)
                record["metadata_sha256"] = hashlib.sha256(body).hexdigest()
            if (
                metadata.get("id") != entity
                or metadata.get("parentId") != parent
                or metadata.get("name") != "SSVEP011.csv"
                or metadata.get("versionNumber") != 1
            ):
                raise ValueError("entity_identity_mismatch")
            record["entity_metadata"] = {
                k: metadata.get(k)
                for k in ("id", "name", "parentId", "versionNumber", "etag", "dataFileHandleId")
            }
            with get(f"{BASE}/entity/{entity}/version/1/file") as response:
                record["file_http_status"] = response.status
                line = response.readline(65537)
            record["first_line_bytes"] = len(line)
            record["first_line_sha256"] = hashlib.sha256(line).hexdigest()
            record["header_fields"] = parse_header(line, modality)
        state["status"] = "COMPLETE_HEADERS_ONLY"
    except (
        csv.Error,
        TypeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        UnicodeError,
    ) as error:
        state.update(
            status="STOPPED_NO_RETRY",
            error_type=type(error).__name__,
            http_status=getattr(error, "code", None),
            error_reason=str(error) if isinstance(error, ValueError) else type(error).__name__,
        )
    state["seconds"] = time.monotonic() - started
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("no_restart")
    # Reserve the attempt before network access. Never overwrite a prior attempt.
    attempt = args.output.with_suffix(".attempt.json")
    with attempt.open("x") as stream:
        json.dump({"attempt": 1, "targets": TARGETS}, stream)

    def deadline(_signal, _frame):
        raise TimeoutError("header_probe_deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    result = probe()
    signal.setitimer(signal.ITIMER_REAL, 0)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(rendered.encode()) > 262144:
        raise ValueError("report_size_budget")
    with args.output.open("x") as stream:
        stream.write(rendered)
    print(rendered, end="")

"""One-shot public download of the frozen pair; no waveform/marker value decoding."""

import argparse
import hashlib
import http.client
import io
import json
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

from scipy.io import savemat, whosmat


class Stop(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise Stop(reason)


def allowed_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in (None, 443)
        and (host == "ndownloader.figshare.com" or host.endswith(".amazonaws.com"))
    )


class BoundedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, limit):
        self.limit = limit
        self.count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(self.count < self.limit, "redirect_limit")
        require(allowed_url(newurl), "redirect_target_not_allowed")
        self.count += 1
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def copy_payload(source, target, expected, chunk_size):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    total = 0
    while True:
        chunk = source.read(min(chunk_size, expected - total + 1))
        if not chunk:
            break
        require(total + len(chunk) <= expected, "payload_exceeds_expected_size")
        target.write(chunk)
        md5.update(chunk)
        sha256.update(chunk)
        total += len(chunk)
    require(total == expected, "payload_size_mismatch")
    return total, md5.hexdigest(), sha256.hexdigest()


def schema(path, budget):
    variables = whosmat(path)
    require(len(variables) <= budget["top_level_variables_max"], "variable_limit")
    for name, _, _ in variables:
        require(len(name.encode()) <= budget["name_bytes_max"], "variable_name_limit")
    return [
        {"name": name, "shape": list(shape), "matlab_type": kind} for name, shape, kind in variables
    ]


def download(spec, directory, budget):
    record = {"id": spec["id"], "name": spec["name"], "attempts": 1, "status": "STARTED"}
    redirect = BoundedRedirect(budget["redirects_per_file_max"])
    partial = directory / (spec["name"] + ".part")
    target = directory / spec["name"]
    started = time.monotonic()

    def deadline(_signum, _frame):
        raise Stop("file_wall_deadline")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, budget["file_wall_seconds"])
    try:
        require(not partial.exists() and not target.exists(), "output_exists")
        require(allowed_url(spec["url"]), "input_url_not_allowed")
        request = urllib.request.Request(
            spec["url"],
            headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"},
        )
        opener = urllib.request.build_opener(redirect)
        with opener.open(request, timeout=budget["socket_timeout_seconds"]) as response:
            record["http_status"] = response.status
            require(response.status == 200, "expected_http_200")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                require(int(content_length) == spec["bytes"], "content_length_mismatch")
            with partial.open("xb") as out:
                size, md5, sha256 = copy_payload(
                    response, out, spec["bytes"], budget["io_chunk_bytes"]
                )
        record.update(bytes=size, md5=md5, sha256=sha256)
        require(md5 == spec["md5"], "md5_mismatch")
        partial.rename(target)
        record["path"] = str(target)
        record["status"] = "DOWNLOADED_VERIFIED"
        # whosmat reports top-level structure only; never call loadmat here.
        record["top_level_schema"] = schema(target, budget)
        record["status"] = "DOWNLOADED_VERIFIED_TOP_LEVEL_SCHEMA"
    except urllib.error.HTTPError as error:
        record.update(status="HTTP_ACCESS_FAILURE", http_status=error.code)
        try:
            body = error.read(budget["error_body_bytes_max"])
            record["error_body_bytes_read"] = len(body)
            record["error_body_class"] = (
                "expired_request_reported" if b"Request has expired" in body else "unclassified"
            )
        except (Stop, OSError, http.client.HTTPException):
            record["error_body_class"] = "unavailable"
        finally:
            error.close()
    except (Stop, OSError, ValueError, NotImplementedError, http.client.HTTPException) as error:
        record["status"] = (
            "VERIFIED_PAYLOAD_SCHEMA_UNRESOLVED"
            if record["status"] == "DOWNLOADED_VERIFIED"
            else "FAILED"
        )
        record["error_type"] = type(error).__name__
        record["reason"] = str(error) if isinstance(error, Stop) else "details_not_exported"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        record["redirects"] = redirect.count
        record["elapsed_seconds"] = round(time.monotonic() - started, 6)
        record["partial_bytes_retained"] = partial.stat().st_size if partial.exists() else 0
    return record


def run(config):
    projection_path = Path(config["input_projection"])
    projection_bytes = projection_path.read_bytes()
    require(
        hashlib.sha256(projection_bytes).hexdigest() == config["projection_sha256"], "source_pin"
    )
    projection = json.loads(projection_bytes)
    expected = [
        {
            "id": item["id"],
            "name": item["name"],
            "bytes": item["size"],
            "md5": item["computed_md5"],
            "url": item["download_url"],
        }
        for item in projection["selected_pair"]
    ]
    require(config["files"] == expected and len(expected) == 2, "exact_pair_pin")
    directory = Path(config["output_dir"])
    require(directory.resolve(strict=True) == directory and directory.is_dir(), "output_directory")
    require(not list(directory.iterdir()), "output_directory_not_empty_no_restart")
    require(
        sum(item["bytes"] for item in expected) < config["budget"]["total_retained_bytes_max"],
        "retained_budget",
    )
    # Exclusive marker survives failures and prevents accidental silent restarts.
    with (directory / "attempt.once").open("x") as marker:
        marker.write("mobilebci-public-pair-preflight-v1\n")
    result = {
        "schema": "cfeg.mobilebci-public-pair-observation.v1",
        "output_dir": str(directory),
        "records": [],
        "waveform_numeric_decodes": 0,
        "marker_value_decodes": 0,
        "fits": 0,
        "outcomes": 0,
    }
    for spec in expected:
        record = download(spec, directory, config["budget"])
        result["records"].append(record)
        if record["status"] != "DOWNLOADED_VERIFIED_TOP_LEVEL_SCHEMA":
            break
    complete = len(result["records"]) == 2 and all(
        item["status"] == "DOWNLOADED_VERIFIED_TOP_LEVEL_SCHEMA" for item in result["records"]
    )
    result["status"] = "PAIR_VERIFIED_SCHEMA_ONLY" if complete else "STOPPED_NO_RETRY"
    result["limitations"] = (
        "No exact sensor/time alignment or efficacy established by top-level schema. "
        "HTTP failure is specific to this single normal GET path, not proof the data are private. "
        "Signed redirect URLs and raw error body are not retained."
    )
    print(json.dumps(result, indent=2))


def selftest():
    out = io.BytesIO()
    size, md5, _ = copy_payload(io.BytesIO(b"abc"), out, 3, 2)
    assert size == 3 and md5 == hashlib.md5(b"abc").hexdigest()
    failures = 0
    for payload in (b"ab", b"abcd"):
        try:
            copy_payload(io.BytesIO(payload), io.BytesIO(), 3, 2)
        except Stop:
            failures += 1
    assert failures == 2
    assert allowed_url("https://ndownloader.figshare.com/files/26094953")
    assert not allowed_url("http://ndownloader.figshare.com/files/26094953")
    assert not allowed_url("https://attacker.example/object")
    fixture = io.BytesIO()
    savemat(fixture, {"generated": [1.0, 2.0]}, do_compression=True)
    fixture.seek(0)
    header = schema(fixture, {"top_level_variables_max": 4, "name_bytes_max": 128})
    assert header == [{"name": "generated", "shape": [1, 2], "matlab_type": "double"}]

    class BrokenBody(io.BytesIO):
        def read(self, *_args):
            raise OSError("generated-secret-not-for-export")

    spec = {"id": 1, "name": "failed.mat", "url": "https://ndownloader.figshare.com/files/1"}
    budget = {
        "redirects_per_file_max": 3,
        "file_wall_seconds": 3,
        "socket_timeout_seconds": 1,
        "error_body_bytes_max": 4096,
    }
    errors = [
        urllib.error.HTTPError(spec["url"], 403, "denied", {}, BrokenBody()),
        http.client.BadStatusLine("generated-secret-not-for-export"),
    ]
    for failure in errors:
        with patch("urllib.request.build_opener") as mocked:
            mocked.return_value.open.side_effect = failure
            record = download(spec, Path("generated-never-created-pair"), budget)
        assert record["status"] in ("HTTP_ACCESS_FAILURE", "FAILED")
        assert "generated-secret" not in json.dumps(record)
    print(json.dumps({"status": "PASS", "checks": 9, "fixture_bytes": len(fixture.getvalue())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        run(json.loads(args.config.read_text()))

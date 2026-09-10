"""Bounded publisher ZIP directory inspection; never read/decode any member."""

import argparse
import base64
import collections
import hashlib
import html
import http.client
import json
import re
import signal
import stat
import struct
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit


class StopInspection(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise StopInspection(reason)


def parse_footer(data, total, limits):
    require(len(data) == 22, "footer_length")
    sig, disk, start_disk, n_disk, n_all, size, offset, comment = struct.unpack("<4s4H2IH", data)
    require(sig == b"PK\x05\x06" and comment == 0, "unsupported_footer_or_comment")
    require(disk == start_disk == 0 and n_disk == n_all, "multidisk")
    require(n_all != 65535 and size != 0xFFFFFFFF and offset != 0xFFFFFFFF, "zip64")
    require(0 < n_all <= limits["entries_max"], "entry_count")
    require(46 * n_all <= size <= limits["directory_bytes_max"], "directory_size")
    require(offset >= 0 and offset + size == total - 22, "directory_layout")
    return {"entries": n_all, "offset": offset, "bytes": size}


def parse_directory(data, footer):
    require(len(data) == footer["bytes"], "directory_length")
    pos, names, offsets, entries = 0, set(), set(), []
    for _ in range(footer["entries"]):
        require(pos + 46 <= len(data), "truncated_directory_header")
        fields = struct.unpack_from("<4s6H3I5H2I", data, pos)
        (
            sig,
            made,
            version,
            flags,
            method,
            _,
            _,
            crc,
            csize,
            usize,
            nl,
            el,
            cl,
            disk,
            _,
            attrs,
            offset,
        ) = fields
        require(sig == b"PK\x01\x02", "directory_signature")
        require(disk == 0 and not flags & 0x2041, "split_or_encrypted_entry")
        require(version < 45 and 0xFFFFFFFF not in (csize, usize, offset), "zip64_entry")
        require(method in (0, 8), "unsupported_compression")
        require(nl > 0 and pos + 46 + nl + el + cl <= len(data), "entry_length")
        raw_name = data[pos + 46 : pos + 46 + nl]
        try:
            name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        except UnicodeError:
            raise StopInspection("filename_encoding") from None
        require(not any(ord(c) < 32 or ord(c) == 127 for c in name), "control_in_name")
        require(not name.startswith("/") and "\\" not in name and ":" not in name, "unsafe_name")
        parts = name.rstrip("/").split("/")
        require(all(p not in ("", ".", "..") for p in parts), "unsafe_path_component")
        canonical = unicodedata.normalize("NFC", name).casefold()
        require(canonical not in names and offset not in offsets, "duplicate_entry")
        names.add(canonical)
        offsets.add(offset)
        require(offset + 30 + csize <= footer["offset"], "member_extent_out_of_bounds")
        require(not ((made >> 8) == 3 and stat.S_ISLNK(attrs >> 16)), "symlink_entry")
        extra = data[pos + 46 + nl : pos + 46 + nl + el]
        ep = 0
        while ep < len(extra):
            require(ep + 4 <= len(extra), "extra_field_header")
            kind, length = struct.unpack_from("<HH", extra, ep)
            require(kind != 1 and ep + 4 + length <= len(extra), "zip64_or_bad_extra_field")
            ep += 4 + length
        basename = parts[-1]
        suffix = Path(basename).suffix.lower()
        is_dir = name.endswith("/")
        is_code = suffix in (".m", ".py", ".r", ".c", ".cpp", ".h", ".jl", ".ipynb")
        conservative_doc_name = bool(
            re.fullmatch(
                r"(?:readme|license|citation|authors|export|acquisition)(?:\.(?:txt|md|rst|pdf|doc|docx|htm|html|json|yaml|yml))?",
                basename.lower(),
            )
        )
        is_doc = conservative_doc_name or suffix in (
            ".txt",
            ".md",
            ".rst",
            ".pdf",
            ".doc",
            ".docx",
            ".htm",
            ".html",
            ".json",
            ".yaml",
            ".yml",
        )
        category = (
            "directory"
            if is_dir
            else "code_candidate"
            if is_code
            else "document_candidate"
            if is_doc
            else suffix
            if suffix
            in (
                ".xlsx",
                ".xls",
                ".csv",
                ".mat",
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".eps",
                ".svg",
                ".zip",
                ".gz",
                ".tar",
                ".bdf",
                ".edf",
            )
            else "unknown_extension"
            if suffix
            else "no_extension"
        )
        entry = {
            "name_sha256": hashlib.sha256(raw_name).hexdigest(),
            "category": category,
            "compressed_bytes": csize,
            "uncompressed_bytes_declared": usize,
            "compression_method": method,
            "local_header_offset_declared": offset,
            "crc32_declared": f"{crc:08x}",
        }
        if conservative_doc_name:
            entry["conservative_document_basename"] = basename
        entries.append(entry)
        pos += 46 + nl + el + cl
    require(pos == len(data), "unparsed_directory_trailer")
    return {
        "entry_count": len(entries),
        "categories": dict(collections.Counter(e["category"] for e in entries)),
        "entries": entries,
        "candidate_count": sum(
            e["category"] in ("code_candidate", "document_candidate") for e in entries
        ),
    }


def validate_response(status, headers, expected_status, limit, content_range=None, etag=None):
    require(status == expected_status, "http_status_mismatch")
    require(
        headers.get("content-encoding", "identity").lower() == "identity", "http_content_encoding"
    )
    require("transfer-encoding" not in headers, "http_transfer_encoding")
    try:
        length = int(headers["content-length"])
    except (KeyError, ValueError):
        raise StopInspection("missing_or_invalid_content_length") from None
    require(0 <= length <= limit, "response_size")
    if content_range is not None:
        require(headers.get("content-range") == content_range, "content_range_mismatch")
        start_end, _ = content_range.removeprefix("bytes ").split("/")
        start, end = map(int, start_end.split("-"))
        require(length == end - start + 1, "range_length_mismatch")
    if etag is not None:
        require(headers.get("etag") == etag and not etag.startswith("W/"), "etag_mismatch")
    return length


class Fetcher:
    def __init__(self, config, result):
        self.config, self.result = config, result
        self.started = time.monotonic()

    def request(self, url, method, limit, content_range=None):
        limits, source = self.config["budget"], self.config["sources"]
        require(len(self.result["requests"]) < limits["http_requests_max"], "request_budget")
        require(time.monotonic() - self.started < limits["network_wall_seconds_max"], "wall_budget")
        split = urlsplit(url)
        require(split.scheme == "https" and split.netloc == "oup.silverchair-cdn.com", "url_host")
        require(split.path in ("/article-minimal/5641733", source["zip_path"]), "url_path")
        record = {
            "method": method,
            "url_without_query": split.scheme + "://" + split.netloc + split.path,
            "requested_content_range": content_range,
            "body_bytes_read": 0,
        }
        self.result["requests"].append(record)
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "califreeEEG-public-metadata-review/1",
        }
        if content_range is not None:
            headers["Range"] = "bytes=" + content_range.split(" ")[1].split("/")[0]
            headers["If-Match"] = source["expected_etag"]
        connection = http.client.HTTPSConnection(
            split.netloc, timeout=limits["request_timeout_seconds"]
        )
        try:
            connection.request(
                method, split.path + ("?" + split.query if split.query else ""), headers=headers
            )
            response = connection.getresponse()
            pairs = response.getheaders()
            h = {k.lower(): v for k, v in pairs}
            critical_headers = {
                "content-length",
                "content-range",
                "etag",
                "content-encoding",
                "transfer-encoding",
            }
            counts = collections.Counter(k.lower() for k, _ in pairs)
            require(all(counts[k] <= 1 for k in critical_headers), "duplicate_control_header")
            record["status"] = response.status
            record["headers"] = {
                k: h[k]
                for k in (
                    "content-length",
                    "content-range",
                    "etag",
                    "content-type",
                    "content-encoding",
                    "accept-ranges",
                )
                if k in h
            }
            expected_etag = source["expected_etag"] if split.path == source["zip_path"] else None
            length = validate_response(
                response.status,
                h,
                206 if content_range else 200,
                limit,
                content_range,
                expected_etag,
            )
            if method == "HEAD":
                require(length == source["expected_archive_bytes"], "archive_size_changed")
                return b""
            require(
                sum(r["body_bytes_read"] for r in self.result["requests"]) + length
                <= limits["network_body_bytes_max"],
                "network_byte_budget",
            )
            chunks, received = [], 0
            while received < length:
                require(
                    time.monotonic() - self.started < limits["network_wall_seconds_max"],
                    "wall_budget",
                )
                chunk = response.read(min(16384, length - received))
                require(bool(chunk), "short_http_body")
                chunks.append(chunk)
                received += len(chunk)
                record["body_bytes_read"] = received
            body = b"".join(chunks)
            record["body_sha256"] = hashlib.sha256(body).hexdigest()
            return body
        finally:
            connection.close()


def inspect(config):
    result = {
        "schema": "cfeg.choi-supplement-directory-observation.v1",
        "requests": [],
        "status": "STARTED",
        "member_content_reads": 0,
        "human_fits": 0,
        "outcome_reveals": 0,
    }
    fetch = Fetcher(config, result)
    source, limits = config["sources"], config["budget"]

    def deadline(_signum, _frame):
        raise StopInspection("network_wall_deadline")

    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, limits["network_wall_seconds_max"])
    try:
        page = fetch.request(source["publisher_html"], "GET", limits["html_bytes_max"])
        matches = set()
        for raw in re.findall(r'href=["\x27]([^"\x27]+)["\x27]', page.decode("utf-8")):
            candidate = html.unescape(raw)
            if urlsplit(candidate).path == source["zip_path"]:
                matches.add(candidate)
        require(len(matches) == 1, "publisher_link_not_unique")
        url = matches.pop()
        fetch.request(url, "HEAD", source["expected_archive_bytes"])
        total = source["expected_archive_bytes"]
        footer_bytes = fetch.request(url, "GET", 22, f"bytes {total - 22}-{total - 1}/{total}")
        result["footer_base64"] = base64.b64encode(footer_bytes).decode("ascii")
        footer = parse_footer(footer_bytes, total, limits)
        result["footer"] = footer
        cd = fetch.request(
            url,
            "GET",
            limits["directory_bytes_max"],
            f"bytes {footer['offset']}-{footer['offset'] + footer['bytes'] - 1}/{total}",
        )
        result["inventory"] = parse_directory(cd, footer)
        result["status"] = "DIRECTORY_VERIFIED_CONTENT_UNOPENED"
    except (StopInspection, OSError, UnicodeError, http.client.HTTPException) as error:
        result["status"] = "STOPPED_WITHOUT_CONTENT_OPEN"
        result["failure_type"] = type(error).__name__
        result["failure_reason"] = (
            str(error) if isinstance(error, StopInspection) else "transport_or_decoding_error"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    result["elapsed_seconds"] = round(time.monotonic() - fetch.started, 3)
    result["archive_metadata_body_bytes_read"] = sum(
        r["body_bytes_read"] for r in result["requests"] if r["url_without_query"].endswith(".zip")
    )
    result["limitations"] = (
        "Only named requested ranges; no full ZIP hash, member content, decompression, CRC verification or sensor/timing qualification. Transport buffering may exceed application body reads on rejected responses."
    )
    return result


def selftest():
    import io
    import zipfile

    limits = {"entries_max": 256, "directory_bytes_max": 131072}
    fixture = io.BytesIO()
    with zipfile.ZipFile(fixture, "w") as zf:
        zf.writestr("README.txt", "generated fixture")
        zf.writestr("example.xlsx", "not a spreadsheet; generated fixture")
    blob = fixture.getvalue()
    footer = parse_footer(blob[-22:], len(blob), limits)
    directory = blob[footer["offset"] : -22]
    found = parse_directory(directory, footer)
    assert found["candidate_count"] == 1 and found["entry_count"] == 2
    cases = ["generated_directory_inventory"]

    def rejects(label, operation):
        try:
            operation()
        except StopInspection:
            cases.append(label)
        else:
            raise AssertionError(label)

    rejects("comment_or_bad_footer", lambda: parse_footer(b"X" * 22, len(blob), limits))
    rejects("entry_cap", lambda: parse_footer(blob[-22:], len(blob), {**limits, "entries_max": 1}))
    rejects("directory_extent", lambda: parse_footer(blob[-22:], len(blob) + 1, limits))
    unsafe = directory.replace(b"README.txt", b"../bad.txt")
    rejects("traversal", lambda: parse_directory(unsafe, footer))
    rejects(
        "ignored_range_before_read",
        lambda: validate_response(200, {"content-length": "22"}, 206, 22),
    )
    rejects(
        "wrong_range_length",
        lambda: validate_response(
            206,
            {"content-length": "21", "content-range": "bytes 0-21/100"},
            206,
            22,
            "bytes 0-21/100",
        ),
    )
    rejects(
        "changed_etag",
        lambda: validate_response(
            206, {"content-length": "22", "etag": '"changed"'}, 206, 22, etag='"expected"'
        ),
    )
    assert len(blob) < 131072
    return {
        "status": "PASS",
        "cases": cases,
        "case_count": len(cases),
        "fixture_bytes": len(blob),
        "network_requests": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.self_test:
        output = selftest()
    else:
        require(args.config is not None, "config_required")
        output = inspect(json.loads(args.config.read_text()))
    print(json.dumps(output, ensure_ascii=True, indent=2))

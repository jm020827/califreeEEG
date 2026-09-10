"""Extract only two frozen public-document candidates; never spreadsheet members."""

import argparse
import hashlib
import html
import http.client
import json
import re
import signal
import struct
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit

from inspect_choi_supplement_directory_v1 import Fetcher, StopInspection, require

DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md", ".rst", ".html", ".htm"}


def local_header(data, candidate, name_extra_limit):
    require(len(data) == 30, "local_header_length")
    sig, version, flags, method, _, _, crc, csize, usize, nl, el = struct.unpack("<4s5H3I2H", data)
    require(sig == b"PK\x03\x04" and version < 45, "local_header_signature_or_version")
    require(
        not flags & 0x2041 and method == candidate["compression_method"] == 8, "flags_or_method"
    )
    expected = (
        int(candidate["crc32_declared"], 16),
        candidate["compressed_bytes"],
        candidate["uncompressed_bytes_declared"],
    )
    require(
        (crc, csize, usize) == expected or (flags & 8 and (crc, csize, usize) == (0, 0, 0)),
        "local_central_mismatch",
    )
    require(0 < nl <= 512 and 0 <= el and nl + el <= name_extra_limit, "name_extra_size")
    data_start = candidate["local_header_offset_declared"] + 30 + nl + el
    require(
        data_start + candidate["compressed_bytes"] <= candidate["next_structure_offset"],
        "member_overlap",
    )
    return {"flags": flags, "name_bytes": nl, "extra_bytes": el, "data_start": data_start}


def identify(name_extra, header, candidate):
    nl = header["name_bytes"]
    require(len(name_extra) == nl + header["extra_bytes"], "name_extra_length")
    raw = name_extra[:nl]
    require(hashlib.sha256(raw).hexdigest() == candidate["name_sha256"], "name_hash_mismatch")
    name = raw.decode("utf-8" if header["flags"] & 0x800 else "cp437")
    require(not any(ord(c) < 32 or ord(c) == 127 for c in name), "name_control")
    require(not name.startswith("/") and "\\" not in name and ":" not in name, "name_path")
    require(all(p not in ("", ".", "..") for p in name.split("/")), "name_component")
    extra, pos = name_extra[nl:], 0
    while pos < len(extra):
        require(pos + 4 <= len(extra), "extra_header")
        kind, size = struct.unpack_from("<HH", extra, pos)
        require(kind != 1 and pos + 4 + size <= len(extra), "zip64_or_bad_extra")
        pos += 4 + size
    suffix = Path(name).suffix.lower()
    return {
        **candidate,
        **header,
        "member_name": name,
        "suffix": suffix,
        "document_format_allowed": suffix in DOCUMENT_EXTENSIONS,
    }


def decode_member(compressed, candidate, limit):
    require(len(compressed) == candidate["compressed_bytes"], "compressed_length")
    require(candidate["uncompressed_bytes_declared"] <= limit, "decoded_declaration_cap")
    decoder = zlib.decompressobj(-15)
    decoded = decoder.decompress(compressed, limit + 1)
    require(
        len(decoded) <= limit
        and decoder.eof
        and not decoder.unused_data
        and not decoder.unconsumed_tail,
        "deflate_boundary",
    )
    require(len(decoded) == candidate["uncompressed_bytes_declared"], "decoded_length")
    require(f"{zlib.crc32(decoded):08x}" == candidate["crc32_declared"], "crc_mismatch")
    return decoded


def validate_identity(candidate, frozen, limits):
    require(all(candidate[k] == v for k, v in frozen.items()), "identity_config_mismatch")
    flags = candidate["flags"]
    require(type(flags) is int and 0 <= flags <= 65535 and not flags & 0x2041, "identity_flags")
    name = candidate["member_name"]
    raw = name.encode("utf-8" if flags & 0x800 else "cp437")
    require(hashlib.sha256(raw).hexdigest() == frozen["name_sha256"], "identity_name_hash")
    require(len(raw) == candidate["name_bytes"] and 0 < len(raw) <= 512, "identity_name_length")
    extra = candidate["extra_bytes"]
    require(
        type(extra) is int and 0 <= extra and len(raw) + extra <= limits["name_extra_bytes_max"],
        "identity_extra_length",
    )
    require(
        candidate["data_start"] == frozen["local_header_offset_declared"] + 30 + len(raw) + extra,
        "identity_data_start",
    )
    require(
        candidate["suffix"] == Path(name).suffix.lower()
        and candidate["suffix"] in DOCUMENT_EXTENSIONS,
        "identity_suffix",
    )
    require(candidate["document_format_allowed"] is True, "identity_format")


def run(config, mode, identity=None, index=None, public_url=None, output_dir=None):
    result = {
        "schema": "cfeg.choi-supplement-document-observation.v1",
        "mode": mode,
        "status": "STARTED",
        "requests": [],
    }
    fetch = Fetcher(config, result)
    limits, source = config["budget"], config["sources"]

    def deadline(_signal, _frame):
        raise StopInspection("network_wall_deadline")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, limits["network_wall_seconds_max"])
    try:
        total = source["expected_archive_bytes"]
        if mode == "identity":
            page = fetch.request(source["publisher_html"], "GET", limits["html_bytes_max"])
            links = {
                html.unescape(x)
                for x in re.findall(r'href=["\x27]([^"\x27]+)["\x27]', page.decode("utf-8"))
                if urlsplit(html.unescape(x)).path == source["zip_path"]
            }
            require(len(links) == 1, "publisher_link_not_unique")
            url = links.pop()
            fetch.request(url, "HEAD", total)
            result["candidates"] = []
            for candidate in config["candidates"]:
                start = candidate["local_header_offset_declared"]
                raw = fetch.request(url, "GET", 30, f"bytes {start}-{start + 29}/{total}")
                head = local_header(raw, candidate, limits["name_extra_bytes_max"])
                start += 30
                length = head["name_bytes"] + head["extra_bytes"]
                name_extra = fetch.request(
                    url,
                    "GET",
                    limits["name_extra_bytes_max"],
                    f"bytes {start}-{start + length - 1}/{total}",
                )
                result["candidates"].append(identify(name_extra, head, candidate))
            result["transient_public_url"] = url
            result["status"] = "IDENTITIES_VERIFIED_CONTENT_UNOPENED"
        else:
            require(
                index in (0, 1) and identity["status"] == "IDENTITIES_VERIFIED_CONTENT_UNOPENED",
                "identity_required",
            )
            candidate = identity["candidates"][index]
            validate_identity(candidate, config["candidates"][index], limits)
            require(urlsplit(public_url).path == source["zip_path"], "document_url_not_zip")
            require(
                candidate["suffix"] in DOCUMENT_EXTENSIONS and candidate["document_format_allowed"],
                "not_document_format",
            )
            require(
                candidate["compressed_bytes"] <= limits["member_compressed_bytes_max"],
                "compressed_cap",
            )
            start, length = candidate["data_start"], candidate["compressed_bytes"]
            require(start + length <= candidate["next_structure_offset"], "payload_extent")
            compressed = fetch.request(
                public_url,
                "GET",
                limits["member_compressed_bytes_max"],
                f"bytes {start}-{start + length - 1}/{total}",
            )
            decoded = decode_member(compressed, candidate, limits["member_decoded_bytes_max"])
            if candidate["suffix"] == ".pdf":
                require(decoded.startswith(b"%PDF-"), "pdf_magic")
            if candidate["suffix"] == ".docx":
                require(decoded.startswith(b"PK\x03\x04"), "docx_container_magic")
            require(output_dir is not None and output_dir.is_dir(), "output_directory")
            destination = output_dir / f"candidate-{index + 1}{candidate['suffix']}"
            with destination.open("xb") as stream:
                stream.write(decoded)
            result["document"] = {
                "member_name": candidate["member_name"],
                "local_path": str(destination),
                "bytes": len(decoded),
                "sha256": hashlib.sha256(decoded).hexdigest(),
                "crc32_verified": candidate["crc32_declared"],
                "format_requires_parser_verification": True,
            }
            result["status"] = "EXACT_MEMBER_EXTRACTED_READING_PENDING"
    except (StopInspection, OSError, UnicodeError, zlib.error, http.client.HTTPException) as error:
        result["status"] = "STOPPED"
        result["failure_type"] = type(error).__name__
        result["failure_reason"] = (
            str(error)
            if isinstance(error, StopInspection)
            else "transport_or_decode_or_output_error"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result["elapsed_seconds"] = round(time.monotonic() - fetch.started, 3)
    return result


def selftest():
    decoded = b"%PDF-generated-test-only\n"
    encoder = zlib.compressobj(wbits=-15)
    compressed = encoder.compress(decoded) + encoder.flush()
    c = {
        "name_sha256": hashlib.sha256(b"manual.pdf").hexdigest(),
        "compression_method": 8,
        "crc32_declared": f"{zlib.crc32(decoded):08x}",
        "compressed_bytes": len(compressed),
        "uncompressed_bytes_declared": len(decoded),
        "local_header_offset_declared": 0,
        "next_structure_offset": 1000,
    }
    raw = struct.pack(
        "<4s5H3I2H",
        b"PK\x03\x04",
        20,
        0,
        8,
        0,
        0,
        zlib.crc32(decoded),
        len(compressed),
        len(decoded),
        10,
        0,
    )
    header = local_header(raw, c, 4096)
    assert identify(b"manual.pdf", header, c)["suffix"] == ".pdf"
    identity = identify(b"manual.pdf", header, c)
    assert decode_member(compressed, c, 1024) == decoded
    cases = ["matched_identity", "bounded_deflate_crc"]
    for name, operation in [
        ("wrong_name", lambda: identify(b"secret.pdf", header, c)),
        ("wrong_crc", lambda: decode_member(compressed, {**c, "crc32_declared": "00000000"}, 1024)),
        ("decoded_cap", lambda: decode_member(compressed, c, 1)),
        ("overlap", lambda: local_header(raw, {**c, "next_structure_offset": 1}, 4096)),
        (
            "forged_identity_start",
            lambda: validate_identity(
                {**identity, "data_start": 50}, c, {"name_extra_bytes_max": 4096}
            ),
        ),
        (
            "forged_identity_suffix",
            lambda: validate_identity(
                {**identity, "suffix": ".xlsx"}, c, {"name_extra_bytes_max": 4096}
            ),
        ),
    ]:
        try:
            operation()
        except StopInspection:
            cases.append(name)
        else:
            raise AssertionError(name)
    return {
        "status": "PASS",
        "cases": cases,
        "largest_fixture_bytes": max(len(raw), len(decoded), len(compressed)),
        "network_requests": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("identity", "document"))
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--public-url")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = (
        selftest()
        if args.self_test
        else run(
            json.loads(args.config.read_text()),
            args.mode,
            json.loads(args.identity.read_text()) if args.identity else None,
            args.index,
            args.public_url,
            args.output_dir,
        )
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))

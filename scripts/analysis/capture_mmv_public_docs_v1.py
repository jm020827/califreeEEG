"""One bounded anonymous document GET per invocation; never executes page code."""

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/mmv_public_capture_v1_contract.md"
RUN = ROOT / "docs/reports/mmv_public_capture_v1_run"
SEED = "https://www.scidb.cn/detail?dataSetId=270bcdeaab0c48adb8eee700479daafd"
CAP = 2 * 1024**2
DEADLINE = datetime.fromisoformat("2026-09-13T16:20:00+00:00").timestamp()
MIMES = {"text/html", "text/plain", "application/json", "text/javascript",
         "application/javascript", "application/x-javascript"}


class Stop(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Stop(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def allowed(url):
    try:
        p = urllib.parse.urlsplit(url)
        require(p.scheme == "https" and p.port in (None, 443), "https443_only")
        require(p.hostname in {"scidb.cn", "www.scidb.cn", "sciencedb.cn",
                               "www.sciencedb.cn"}, "official_host_only")
        require(not p.username and not p.password and not p.fragment, "url_credentials_fragment")
        require(not re.search(r"[\s\\]", url), "ambiguous_url")
        path = urllib.parse.unquote(p.path).lower()
        require(not re.search(r"\.(pdf|mat|cnt|edf|zip|gz|tar|h5|npy|npz|csv|tsv)$", path),
                "raw_or_pdf_file_forbidden")
        require(not re.search(r"/(login|signin|authorize|oauth|captcha)(/|$)", path),
                "auth_path_forbidden")
        require(not re.search(r"(?i)(token|secret|signature|password|api_key|credential)=", p.query),
                "credential_query_forbidden")
    except (ValueError, Stop):
        return False
    return True


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag in {"a", "link", "script"}:
            self.values.extend(v for k, v in attrs if k in {"href", "src"} and v)


def linked_urls(body, base, location=None):
    text = body.decode("utf8", errors="replace")
    parser = Links()
    parser.feed(text)
    # Only complete quoted literals, never synthesize dynamic JS/API expressions.
    literal_pattern = r'''(["'])((?:https://|//|/)[^\s"'<>`{}]+)\1'''
    literals = [m.group(2) for m in re.finditer(literal_pattern, text)
                if not text[:m.start()].rstrip().endswith("+")
                and not text[m.end():].lstrip().startswith("+")]
    values = parser.values + literals + ([location] if location else [])
    resolved = {urllib.parse.urldefrag(urllib.parse.urljoin(base, v))[0] for v in values}
    return sorted(u for u in resolved if allowed(u))


def challenge_or_auth_html(body):
    text = body.decode("utf8", errors="replace").lower()
    return bool(re.search(r'''id\s*=\s*["']challenge-form["']''', text)
                or re.search(r"<title[^>]*>\s*(just a moment|attention required|"
                             r"verify you|security check|sign in|log in|login)\b", text))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def receive(url, path, seconds, opener=None):
    """All retained body bytes including HTTP error/partial bytes get a hash."""
    record = {"url": url, "started_utc": utc(), "status": "STOPPED_NO_RETRY",
              "http_status": None, "automatic_redirects": 0}
    digest = hashlib.sha256()
    inspection = bytearray()
    count = 0
    start = time.monotonic()

    def timeout(_signum, _frame):
        raise Stop("wall_deadline")

    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(0, seconds))
        with path.open("xb") as out:
            try:
                require(seconds > 0, "wall_deadline_before_get")
                req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
                client = opener or urllib.request.build_opener(NoRedirect())
                try:
                    response = client.open(req, timeout=min(seconds, 30))
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    record.update(http_status=response.status, final_url=response.geturl(),
                                  content_type=response.headers.get("Content-Type"),
                                  content_length=response.headers.get("Content-Length"))
                    location = response.headers.get("Location")
                    if location:
                        resolved = urllib.parse.urldefrag(urllib.parse.urljoin(url, location))[0]
                        record["location"] = resolved if allowed(resolved) else "OUT_OF_SCOPE"
                    mime = (record["content_type"] or "").split(";", 1)[0].strip().lower()
                    require(response.headers.get("Content-Encoding", "identity").lower()
                            == "identity", "encoded_body_forbidden")
                    require(mime in MIMES, "non_document_mime")
                    while count < CAP:
                        try:
                            chunk = response.read(min(65536, CAP - count))
                        except http.client.IncompleteRead as error:
                            partial = error.partial[:CAP - count]
                            out.write(partial)
                            digest.update(partial)
                            count += len(partial)
                            raise Stop("incomplete_body") from error
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        inspection.extend(chunk)
                        count += len(chunk)
                    require(count < CAP, "body_cap_reached")
                    if record["content_length"] is not None:
                        require(int(record["content_length"]) == count, "content_length_mismatch")
                    if response.status == 200:
                        require(mime != "text/html" or not challenge_or_auth_html(inspection),
                                "challenge_or_auth_response")
                        record["status"] = "CAPTURED"
                    elif response.status in {301, 302, 303, 307, 308}:
                        require(record.get("location") not in {None, "OUT_OF_SCOPE"},
                                "redirect_out_of_scope")
                        record["status"] = "REDIRECT_AVAILABLE"
                    else:
                        raise Stop("http_non_success")
            finally:
                out.flush()
                os.fsync(out.fileno())
    except (Stop, OSError, ValueError, http.client.HTTPException) as error:
        record.update(status="STOPPED_NO_RETRY", error_type=type(error).__name__,
                      reason=str(error) if isinstance(error, Stop) else "details_not_exported")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        # An alarm can land between write and digest.update. Hash the retained
        # artifact after disabling the alarm; this is local, not a second GET.
        if path.exists():
            require(path.stat().st_size <= CAP, "retained_body_cap")
            retained = path.read_bytes()
            count, digest = len(retained), hashlib.sha256(retained)
        record.update(completed_utc=utc(), elapsed_seconds=time.monotonic() - start,
                      body_bytes=count, body_sha256=digest.hexdigest(), body_name=path.name)
    return record


def capture(directory, url, parent=None, opener=None, now=None):
    entered = time.monotonic()
    now = time.time() if now is None else now
    require(now < DEADLINE, "contract_expired")
    require(allowed(url), "url_not_allowed")
    pins = {"contract": sha(CONTRACT.read_bytes()), "code": sha(Path(__file__).read_bytes())}
    if not directory.exists():
        require(url == SEED and parent is None, "first_seed_only")
        directory.mkdir()
        save(directory / "manifest.json", {"schema": "cfeg.mmv-public-capture-v1",
             "created_utc": utc(), "external_start_epoch": now, "pins": pins,
             "max_attempts": 4, "body_cap": CAP, "total_body_cap": 4 * CAP})
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["pins"] == pins, "pins_changed")
    claims = sorted(directory.glob("claim_*.json"))
    n = len(claims)
    require(n < 4, "attempt_budget")
    receipts = []
    for index in range(1, n + 1):
        claim = json.loads((directory / f"claim_{index}.json").read_text())
        receipt = json.loads((directory / f"receipt_{index}.json").read_text())
        require(claim["url"] == receipt["url"] and claim["step"] == index, "claim_mismatch")
        require(receipt["status"] in {"CAPTURED", "REDIRECT_AVAILABLE"}, "prior_terminal_failure")
        body = (directory / receipt["body_name"]).read_bytes()
        require(len(body) == receipt["body_bytes"] and sha(body) == receipt["body_sha256"],
                "body_changed")
        require(url != claim["url"], "url_already_attempted")
        receipts.append(receipt)
    require(sum(r["body_bytes"] for r in receipts) < 4 * CAP, "total_body_budget")
    if n:
        require(isinstance(parent, int) and 1 <= parent <= n, "explicit_parent_required")
        r = receipts[parent - 1]
        links = linked_urls((directory / r["body_name"]).read_bytes(), r["url"],
                            r.get("location") if r.get("location") != "OUT_OF_SCOPE" else None)
        require(url in links, "not_literal_parent_link")
    else:
        require(url == SEED and parent is None, "first_seed_only")
    current = now + time.monotonic() - entered
    require(current < DEADLINE, "contract_expired_before_claim")
    remaining = 180 - (current - manifest["external_start_epoch"])
    require(remaining > 0, "external_window_closed")
    step = n + 1
    save(directory / f"claim_{step}.json", {"step": step, "url": url, "parent": parent,
                                          "started_utc": utc(), "pins": pins})
    current = now + time.monotonic() - entered
    remaining = min(30, DEADLINE - current, 180 - (current - manifest["external_start_epoch"]))
    record = receive(url, directory / f"body_{step}.bin", remaining, opener)
    record.update(step=step, parent=parent, pins=pins)
    save(directory / f"receipt_{step}.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SEED)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    record = capture(RUN, args.url, args.parent)
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record["status"] in {"CAPTURED", "REDIRECT_AVAILABLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

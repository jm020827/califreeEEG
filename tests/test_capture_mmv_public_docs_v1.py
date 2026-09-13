import http.client
import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "capture_mmv", Path(__file__).resolve().parents[1]
    / "scripts/analysis/capture_mmv_public_docs_v1.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Type": "text/html", "Content-Length": str(len(body))}

    def geturl(self):
        return m.SEED


class Client:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def open(self, *args, **kwargs):
        self.calls += 1
        return self.response


@pytest.mark.parametrize("url", ["http://www.scidb.cn/x", "https://scidb.cn.evil/x",
    "https://name:pass@www.scidb.cn/x", "https://www.scidb.cn:444/x",
    "https://www.scidb.cn/x.pdf", "https://www.scidb.cn/x.edf",
    "https://www.scidb.cn/x?token=secret", "https://www.scidb.cn/login",
    "https://www.scidb.cn/x#fragment", "https://www.scidb.cn/a b"])
def test_bad_urls(url):
    assert not m.allowed(url)


def test_literal_links():
    body = b'<script src="/app.js"></script><a href="/docs#p">doc</a>"/api/"+id'
    links = m.linked_urls(body, m.SEED)
    assert "https://www.scidb.cn/app.js" in links
    assert "https://www.scidb.cn/docs" in links
    assert not any("api/id" in link for link in links)
    assert "https://www.scidb.cn/api/" not in links


def test_capture_and_repeat_refused(tmp_path):
    client = Client(Response(b'<script src="/app.js"></script>'))
    run = tmp_path / "run"
    r = m.capture(run, m.SEED, opener=client, now=m.DEADLINE - 500)
    assert r["status"] == "CAPTURED" and client.calls == 1
    with pytest.raises(m.Stop, match="already_attempted"):
        m.capture(run, m.SEED, opener=client, now=m.DEADLINE - 499)
    r2 = m.capture(run, "https://www.scidb.cn/app.js", parent=1,
                   opener=Client(Response(b"public documentation")), now=m.DEADLINE - 498)
    assert r2["status"] == "CAPTURED"


@pytest.mark.parametrize("kind", ["mime", "encoding", "status", "length", "cap"])
def test_failures_preserved(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(m, "CAP", 32)
    headers = {"Content-Type": "text/html", "Content-Length": "4"}
    body, status = b"test", 200
    if kind == "mime":
        headers["Content-Type"] = "application/pdf"
    elif kind == "encoding":
        headers["Content-Encoding"] = "gzip"
    elif kind == "status":
        status = 403
    elif kind == "length":
        headers["Content-Length"] = "10"
    else:
        body, headers["Content-Length"] = b"x" * 50, "50"
    r = m.receive(m.SEED, tmp_path / "body", 5, Client(Response(body, status, headers)))
    assert r["status"] == "STOPPED_NO_RETRY"
    assert r["body_bytes"] <= 32
    assert m.sha((tmp_path / "body").read_bytes()) == r["body_sha256"]


def test_redirect_not_followed(tmp_path):
    response = urllib.error.HTTPError(m.SEED, 302, "redirect",
        {"Content-Type": "text/html", "Content-Length": "0", "Location": "/docs#section"},
        io.BytesIO(b""))
    class ErrorClient:
        def open(self, *args, **kwargs):
            raise response
    r = m.receive(m.SEED, tmp_path / "body", 5, ErrorClient())
    assert r["status"] == "REDIRECT_AVAILABLE"
    assert r["location"] == "https://www.scidb.cn/docs"
    assert m.NoRedirect().redirect_request(None, None, 302, None, None, "/docs") is None


def test_partial_exception_and_terminal(tmp_path):
    class Broken(Response):
        def read(self, size):
            if self.tell():
                raise OSError("transport fault")
            return super().read(2)
    run = tmp_path / "run"
    r = m.capture(run, m.SEED, opener=Client(Broken(b"text")), now=m.DEADLINE - 500)
    assert r["body_bytes"] == 2 and r["status"] == "STOPPED_NO_RETRY"
    with pytest.raises(m.Stop, match="prior_terminal_failure"):
        m.capture(run, "https://www.scidb.cn/docs", parent=1, now=m.DEADLINE - 499)


def test_expiry_and_literal_requirement(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(m.Stop, match="contract_expired"):
        m.capture(run, m.SEED, now=m.DEADLINE)
    m.capture(run, m.SEED, opener=Client(Response(b'<a href="/docs">a</a>')),
              now=m.DEADLINE - 500)
    with pytest.raises(m.Stop, match="not_literal"):
        m.capture(run, "https://www.scidb.cn/guessed", parent=1, now=m.DEADLINE - 499)
    with pytest.raises(m.Stop, match="window_closed"):
        m.capture(run, "https://www.scidb.cn/docs", parent=1, now=m.DEADLINE - 300)


def test_tampered_body_and_incomplete_claim(tmp_path):
    run = tmp_path / "run"
    m.capture(run, m.SEED, opener=Client(Response(b"docs")), now=m.DEADLINE - 500)
    (run / "body_1.bin").write_bytes(b"tampered")
    with pytest.raises(m.Stop, match="body_changed"):
        m.capture(run, "https://www.scidb.cn/docs", parent=1, now=m.DEADLINE - 499)
    (run / "receipt_1.json").unlink()
    with pytest.raises(FileNotFoundError):
        m.capture(run, "https://www.scidb.cn/docs", parent=1, now=m.DEADLINE - 499)


def test_four_attempts_and_pin(tmp_path, monkeypatch):
    run = tmp_path / "run"
    html = b'<a href="/a">a</a><a href="/b">b</a><a href="/c">c</a><a href="/d">d</a>'
    m.capture(run, m.SEED, opener=Client(Response(html)), now=m.DEADLINE - 500)
    for index, name in enumerate(("a", "b", "c"), 1):
        m.capture(run, "https://www.scidb.cn/" + name, parent=1,
                  opener=Client(Response(b"docs")), now=m.DEADLINE - 500 + index)
    with pytest.raises(m.Stop, match="attempt_budget"):
        m.capture(run, "https://www.scidb.cn/d", parent=1, now=m.DEADLINE - 495)
    monkeypatch.setattr(m, "CONTRACT", Path(__file__))
    with pytest.raises(m.Stop, match="pins_changed"):
        m.capture(run, "https://www.scidb.cn/d", parent=1, now=m.DEADLINE - 495)


def test_incomplete_read_and_wall_failure(tmp_path):
    class Partial(Response):
        def read(self, size):
            raise http.client.IncompleteRead(b"part", 3)
    r = m.receive(m.SEED, tmp_path / "partial", 5, Client(Partial(b"")))
    assert r["body_bytes"] == 4 and r["reason"] == "incomplete_body"
    assert r["body_sha256"] == m.sha(b"part")

    class DeadlineClient:
        def open(self, *args, **kwargs):
            raise m.Stop("wall_deadline")
    r = m.receive(m.SEED, tmp_path / "timeout", 5, DeadlineClient())
    assert r["reason"] == "wall_deadline" and r["body_bytes"] == 0


@pytest.mark.parametrize("body,stopped", [
    (b'<title>Just a moment...</title>', True),
    (b'<form id="challenge-form">verify</form>', True),
    (b'<title>Login</title><input type="password">', True),
    (b'<title>Public dataset</title><a href="/login">Login</a>', False),
])
def test_challenge_not_generic_login_link(tmp_path, body, stopped):
    r = m.receive(m.SEED, tmp_path / "body", 5, Client(Response(body)))
    assert (r["status"] == "STOPPED_NO_RETRY") == stopped
    assert r["body_bytes"] == len(body)


def test_preflight_elapsed_counts(tmp_path, monkeypatch):
    ticks = iter((0.0, 181.0))
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks))
    client = Client(Response(b"docs"))
    with pytest.raises(m.Stop, match="external_window_closed"):
        m.capture(tmp_path / "run", m.SEED, now=m.DEADLINE - 500, opener=client)
    assert client.calls == 0


def test_no_budget_before_get(tmp_path):
    client = Client(Response(b"docs"))
    r = m.receive(m.SEED, tmp_path / "body", 0, client)
    assert r["status"] == "STOPPED_NO_RETRY" and client.calls == 0

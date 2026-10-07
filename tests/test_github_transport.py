"""HTTPS传输失败可走固定GET，正常HTTP拒绝不更换通道。"""
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
import pytest
from src.chat.github_sync import _request_json


def test_transport_failure_uses_only_github_get(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.chat.github_sync.urlopen", lambda *a, **k: (_ for _ in ()).throw(URLError("TLS EOF")))
    monkeypatch.setattr("src.chat.github_sync.shutil.which", lambda _: "/usr/local/bin/gh")
    monkeypatch.setattr("src.chat.github_sync.getproxies", lambda: {"https": "http://127.0.0.1:1234", "no": "localhost"})
    def run(args, **kwargs):
        seen.update(args=args, **kwargs)
        return SimpleNamespace(returncode=0, stdout='{"private":false,"default_branch":"main"}')
    monkeypatch.setattr("src.chat.github_sync.subprocess.run", run)
    url = "https://api.github.com/repos/owner/repo"
    assert _request_json(url)["default_branch"] == "main"
    assert seen["args"] == ["/usr/local/bin/gh", "api", url, "--method", "GET"]
    assert seen["timeout"] == 30 and seen["env"]["https_proxy"] == "http://127.0.0.1:1234"
    assert seen["env"]["NO_PROXY"] == "localhost" and "shell" not in seen
    seen.clear()
    with pytest.raises(URLError): _request_json("https://outside.example/private")
    assert seen == {}


@pytest.mark.parametrize("status", [403, 404, 429])
def test_http_errors_do_not_fallback(monkeypatch, status):
    monkeypatch.setattr("src.chat.github_sync.urlopen", lambda *a, **k: (_ for _ in ()).throw(HTTPError("url", status, "rejected", {}, None)))
    monkeypatch.setattr("src.chat.github_sync.subprocess.run", lambda *a, **k: pytest.fail("HTTP拒绝不应改走备用通道"))
    with pytest.raises(ValueError): _request_json("https://api.github.com/repos/owner/repo")


@pytest.mark.parametrize("result", [SimpleNamespace(returncode=1, stdout="", stderr="secret token"), SimpleNamespace(returncode=0, stdout="invalid json")])
def test_fallback_failure_does_not_leak_stderr(monkeypatch, result):
    monkeypatch.setattr("src.chat.github_sync.urlopen", lambda *a, **k: (_ for _ in ()).throw(URLError("TLS EOF")))
    monkeypatch.setattr("src.chat.github_sync.shutil.which", lambda _: "/usr/local/bin/gh")
    monkeypatch.setattr("src.chat.github_sync.subprocess.run", lambda *a, **k: result)
    with pytest.raises(ValueError) as error: _request_json("https://api.github.com/repos/owner/repo")
    assert "secret token" not in str(error.value)

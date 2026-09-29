"""公开仓库按需同步的边界测试；不触网、不加载真实模型。"""

import pytest

from src.chat import github_sync


@pytest.mark.parametrize("value,expected", [
    ("HKUDS/OpenHarness", "HKUDS/OpenHarness"),
    ("https://github.com/HKUDS/OpenHarness.git", "HKUDS/OpenHarness"),
    ("  owner/repo/  ", "owner/repo"),
])
def test_normalize_repository(value, expected):
    assert github_sync.normalize_repository(value) == expected


@pytest.mark.parametrize("value", [
    "https://evil.example.com/owner/repo", "owner/repo/issues/1", "../repo", "owner/../repo",
    "owner/repo?x=1", "owner/repo#fragment", "owner", "",
])
def test_reject_invalid_repository(value):
    with pytest.raises(ValueError):
        github_sync.normalize_repository(value)


def test_fetch_issues_excludes_pull_requests_and_limits_snapshot(monkeypatch):
    calls = []

    def fake_request(url):
        calls.append(url)
        if "?" not in url:
            return {"full_name": "Owner/Repo"}
        return [
            {"number": 1, "title": "issue", "body": "problem"},
            {"number": 2, "title": "PR", "pull_request": {}},
        ]

    monkeypatch.setattr(github_sync, "_request_json", fake_request)
    name, records = github_sync.fetch_issues("owner/repo")
    assert name == "Owner/Repo"
    assert [item["number"] for item in records] == [1]
    assert len(calls) == 2


def test_private_repository_is_not_imported(monkeypatch):
    monkeypatch.setattr(github_sync, "_request_json", lambda url: {"full_name": "owner/private", "private": True})
    with pytest.raises(ValueError, match="公开"):
        github_sync.fetch_issues("owner/private")


def test_repository_id_is_case_insensitive_and_keeps_issue_scope():
    project = github_sync.repository_id("Owner/Repo")
    assert project == github_sync.repository_id("owner/repo")
    assert github_sync._normalize_issue({"number": 12, "title": "Crash"}, project)["id"] == f"{project}:12"


def test_manager_reuses_complete_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(github_sync, "GITHUB_INDEX_ROOT", tmp_path)
    project = github_sync.repository_id("Owner/Repo")
    directory = tmp_path / project
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    (directory / "bm25.pkl").touch()
    (directory / "docstore.pkl").touch()
    (directory / "chroma").mkdir()
    (directory / "chroma" / "chroma.sqlite3").touch()
    manager = github_sync.RepositorySyncManager()
    job = manager.start("owner/repo")
    assert job.status == "completed"
    assert job.repository_id == project

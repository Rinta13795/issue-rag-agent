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


def test_issue_url_import_reads_source_and_bounded_comments(monkeypatch):
    assert github_sync.parse_issue_url("https://github.com/Owner/Repo/issues/12#top") == ("Owner/Repo", 12)
    calls = []

    def fake_request(url):
        calls.append(url)
        if "/comments?" in url:
            return [{"user": {"login": "maintainer"}, "body": "Fixed in next release", "html_url": "https://github.com/Owner/Repo/issues/12#issuecomment-1"}]
        return {"number": 12, "title": "Crash on start", "body": "Steps to reproduce", "state": "open", "comments": 1,
                "html_url": "https://github.com/Owner/Repo/issues/12"}

    monkeypatch.setattr(github_sync, "_request_json", fake_request)
    issue = github_sync.fetch_issue("https://github.com/Owner/Repo/issues/12")
    assert issue.title == "Crash on start"
    assert issue.comments[0].author == "maintainer"
    assert len(calls) == 2


@pytest.mark.parametrize("value", [
    "http://github.com/a/b/issues/1", "https://evil.example/a/b/issues/1",
    "https://github.com/a/b/pull/1", "https://github.com/a/b/issues/0",
])
def test_issue_url_rejects_non_issue(value):
    with pytest.raises(ValueError):
        github_sync.parse_issue_url(value)


def test_issue_import_rejects_pull_request(monkeypatch):
    monkeypatch.setattr(github_sync, "_request_json", lambda url: {"number": 12, "pull_request": {}})
    with pytest.raises(ValueError, match="Pull Request"):
        github_sync.fetch_issue("https://github.com/Owner/Repo/issues/12")


def test_live_search_is_repository_scoped_and_excludes_pr(monkeypatch):
    calls = []

    def fake_request(url):
        calls.append(url)
        return {"items": [
            {"number": 12, "title": "Crash", "body": "Fails on start", "html_url": "https://github.com/Owner/Repo/issues/12"},
            {"number": 13, "title": "Fix", "pull_request": {}, "html_url": "https://github.com/Owner/Repo/pull/13"},
        ]}

    monkeypatch.setattr(github_sync, "_request_json", fake_request)
    results = github_sync.search_live_issues("Owner/Repo", "app crash")
    assert len(results) == 1
    assert results[0]["source"] == "github"
    assert results[0]["id"] == f"{github_sync.repository_id('Owner/Repo')}:12"
    from urllib.parse import parse_qs, urlsplit
    assert "repo:Owner/Repo is:issue app crash" == parse_qs(urlsplit(calls[0]).query)["q"][0]


def test_publish_without_server_token_is_known_failure(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(github_sync.GitHubPublishError) as exc:
        github_sync.create_github_issue("Owner/Repo", "Title", "Body")
    assert exc.value.definite is True

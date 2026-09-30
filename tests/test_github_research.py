from src.chat.models import SourceIssue
from src.chat.github_research import read_issue, read_pr, search_prs


def test_issue_links_are_same_repo_and_not_asserted_as_fixed(monkeypatch):
    monkeypatch.setattr("src.chat.github_research.fetch_issue", lambda url: SourceIssue(repository="a/b",number=1,title="Bug",body="x",state="open",url=url))
    monkeypatch.setattr("src.chat.github_research._request_json", lambda url: [
        {"event":"cross-referenced", "source":{"issue":{"number":2,"title":"fix","html_url":"https://github.com/a/b/pull/2","pull_request":{ "url":"api"}}}},
        {"event":"cross-referenced", "source":{"issue":{"number":3,"html_url":"https://github.com/other/b/pull/3","pull_request":{"url":"api"}}}},
    ])
    issue = read_issue("a/b",1)
    assert len(issue["linked_prs"]) == 1
    assert issue["linked_prs"][0]["relationship"] == "cross_referenced"


def test_timeline_failure_keeps_issue_and_reports_partial_result(monkeypatch):
    monkeypatch.setattr("src.chat.github_research.fetch_issue", lambda url: SourceIssue(repository="a/b",number=1,title="Bug",body="x",state="open",url=url))
    def fail(url): raise ValueError("rate limited")
    monkeypatch.setattr("src.chat.github_research._request_json", fail)
    result = read_issue("a/b",1)
    assert result["body"] == "x" and result["links_error"] == "rate limited"


def test_pr_diff_and_discussion_are_bounded(monkeypatch):
    def response(url):
        if "/files?" in url: return [{"filename":"x.py", "patch":"x"*3000} for _ in range(30)]
        if "/comments?" in url: return [{"body":"z"*1500, "user":{"login":"dev"}, "author_association":"MEMBER"} for _ in range(30)]
        return {"number":2,"body":"fix #1","title":"fix","merged":True,"state":"closed","changed_files":35,"html_url":"https://github.com/a/b/pull/2"}
    monkeypatch.setattr("src.chat.github_research._request_json", response)
    result = read_pr("a/b",2)
    assert result["merged"] and result["files_truncated"]
    assert sum(len(f["patch"]) for f in result["files"]) == 6000
    assert len(result["discussion"]) == 5 and result["discussion_truncated"]

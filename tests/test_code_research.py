"""外部参考的定点读取与来源归属，全部使用离线响应。"""
import base64

import pytest

from src.chat.code_research import read_repository_file, read_repository_tree, search_repositories

SHA = "a" * 40


def test_discovery_and_fixed_commit_tree(monkeypatch):
    seen = []
    def response(url):
        seen.append(url)
        if "/search/repositories" in url:
            return {"items": [{"full_name": "owner/reference", "private": False}, {"full_name": "owner/private", "private": True}]}
        if "/commits/" in url:
            return {"sha": SHA, "commit": {"tree": {"sha": "b" * 40}}}
        if "/git/trees/" in url:
            return {"tree": [{"type": "blob", "path": "src/main.py"}, {"type": "blob", "path": ".env"},
                             {"type": "blob", "path": "link.py", "mode": "120000"}]}
        return {"default_branch": "main", "private": False}
    monkeypatch.setattr("src.chat.code_research._request_json", response)
    assert len(search_repositories("persistent jobs python")["items"]) == 1
    tree = read_repository_tree("owner/reference")
    assert tree["ref"] == SHA and tree["files"] == ["src/main.py"]
    assert "/commits/main" in seen[2]


def test_file_read_has_repo_ref_and_lines(monkeypatch):
    def response(url):
        if "/contents/" in url:
            assert f"ref={SHA}" in url
            return {"type": "file", "encoding": "base64", "size": 16, "content": base64.b64encode(b"first\nsecond\n").decode()}
        return {"private": False}
    monkeypatch.setattr("src.chat.code_research._request_json", response)
    result = read_repository_file("owner/reference", "src/main.py", SHA, 2, 2)
    assert result["content"] == "2: second"
    assert result["repository"] == "owner/reference" and result["ref"] == SHA
    assert result["url"].endswith("#L2")
    with pytest.raises(ValueError): read_repository_file("owner/reference", ".env", SHA)
    with pytest.raises(ValueError): read_repository_file("owner/reference", "src/main.py", "main")


def test_private_and_non_files_rejected(monkeypatch):
    monkeypatch.setattr("src.chat.code_research._request_json", lambda url: {"private": True})
    with pytest.raises(ValueError): read_repository_tree("owner/private")
    monkeypatch.setattr("src.chat.code_research._request_json", lambda url: {"private": False, "type": "symlink"})
    with pytest.raises(ValueError): read_repository_file("owner/reference", "link.py", SHA)

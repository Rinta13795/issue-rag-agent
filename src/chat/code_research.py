"""外部参考仓库的发现与定点源码读取，不建全量代码索引。"""

import base64
from pathlib import PurePosixPath
from urllib.parse import quote, urlencode

from src.chat.code_workspace import MAX_FILE_BYTES, TEXT_SUFFIXES, _protected
from src.chat.github_sync import _request_json, normalize_repository


def search_repositories(query: str) -> dict:
    """按具体问题发现最多八个公开仓库，返回筛选所需元信息。"""
    params = urlencode({"q": query[:300], "per_page": 8})
    data = _request_json(f"https://api.github.com/search/repositories?{params}")
    return {"items": [{"repository": item["full_name"], "description": item.get("description"),
                       "language": item.get("language"), "stars": item.get("stargazers_count"),
                       "updated_at": item.get("pushed_at"), "url": item.get("html_url"),
                       "default_branch": item.get("default_branch"), "archived": item.get("archived", False)}
                      for item in data.get("items", []) if not item.get("private")],
            "incomplete_results": data.get("incomplete_results", False)}


def _repository(repository: str) -> tuple[str, dict]:
    """外部研究只读取公开仓库。"""
    repository = normalize_repository(repository)
    metadata = _request_json(f"https://api.github.com/repos/{repository}")
    if metadata.get("private"):
        raise ValueError("外部参考只支持公开仓库")
    return repository, metadata


def read_repository_tree(repository: str, ref: str | None = None) -> dict:
    """返回受限源码路径和固定 commit；后续读文件使用同一 commit。"""
    repository, metadata = _repository(repository)
    revision = quote(ref or metadata["default_branch"], safe="")
    commit = _request_json(f"https://api.github.com/repos/{repository}/commits/{revision}")
    sha = commit["sha"]
    tree_sha = commit["commit"]["tree"]["sha"]
    data = _request_json(f"https://api.github.com/repos/{repository}/git/trees/{tree_sha}?recursive=1")
    paths = [item["path"] for item in data.get("tree", []) if item.get("type") == "blob"
             and item.get("mode") != "120000" and not _protected(PurePosixPath(item["path"]))
             and (PurePosixPath(item["path"]).suffix.lower() in TEXT_SUFFIXES
                  or PurePosixPath(item["path"]).name in {"Dockerfile", "Makefile", ".gitignore"})]
    return {"repository": repository, "ref": sha, "files": paths[:700],
            "truncated": bool(data.get("truncated")) or len(paths) > 700,
            "url": f"https://github.com/{repository}/tree/{sha}"}


def read_repository_file(repository: str, path: str, ref: str, start_line: int = 1, end_line: int = 200) -> dict:
    """读取一个公开仓库文件的行号片段；拒绝凭据、软链接和大文件。"""
    repository, _ = _repository(repository)
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or _protected(relative):
        raise ValueError("仅能读取仓库内的普通源码相对路径")
    if relative.suffix.lower() not in TEXT_SUFFIXES and relative.name not in {"Dockerfile", "Makefile", ".gitignore"}:
        raise ValueError("仅支持文本源码、配置和文档")
    if start_line < 1 or end_line < start_line or end_line - start_line >= 300:
        raise ValueError("每次读取 1 到 300 行")
    if len(ref) != 40 or any(char not in "0123456789abcdef" for char in ref.lower()):
        raise ValueError("请先读取文件树，再使用返回的固定 commit ref")
    data = _request_json(f"https://api.github.com/repos/{repository}/contents/{quote(path, safe='/')}?{urlencode({'ref': ref})}")
    if not isinstance(data, dict) or data.get("type") != "file" or data.get("target") or data.get("submodule_git_url"):
        raise ValueError("目标不是普通源码文件")
    if data.get("encoding") != "base64" or int(data.get("size", 0)) > MAX_FILE_BYTES:
        raise ValueError("文件过大或没有可读正文")
    raw = base64.b64decode(data.get("content", ""))
    if len(raw) > MAX_FILE_BYTES or b"\0" in raw:
        raise ValueError("文件过大或不是文本源码")
    lines = raw.decode("utf-8").splitlines()
    content = "\n".join(f"{number}: {line}" for number, line in enumerate(lines[start_line - 1:end_line], start_line))
    return {"repository": repository, "path": path, "ref": ref, "total_lines": len(lines),
            "content": content[:18000], "truncated": end_line < len(lines) or len(content) > 18000,
            "url": f"https://github.com/{repository}/blob/{ref}/{quote(path, safe='/')}#L{start_line}"}

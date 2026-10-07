"""只读本地源码工具：通过路径读取，不需要连接项目。"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".local", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache", "chroma_db_en", ".ssh", ".aws", ".codex", ".agents"}
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".css", ".html", ".sh", ".sql", ".xml", ".rst", ".gitignore"}
MAX_FILE_BYTES = 250_000

def _protected(path: Path) -> bool:
    """跳过凭据、内部状态、依赖和二进制产物。"""
    name = path.name.lower()
    return (any(part in SKIP_DIRS for part in path.parts)
            or name.startswith(".env") or name in {".npmrc", ".netrc", ".pypirc", "credentials", "credentials.json", "secrets.json"}
            or name.endswith((".pem", ".key", ".p12", ".pfx", ".sqlite3", ".db", ".pkl", ".jsonl")))


def _resolve(root: Path, value: str, *, directory: bool = False) -> Path:
    """相对路径必须留在项目内，不允许通过软链接进入其他位置。"""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value.startswith("-") or _protected(relative):
        raise ValueError("只能访问项目内的普通源码相对路径")
    candidate = root / relative
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("源码工具不访问软链接")
        cursor = cursor.parent
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("路径超出当前项目")
    if not directory and candidate.suffix.lower() not in TEXT_SUFFIXES and candidate.name not in {"Dockerfile", "Makefile", ".gitignore"}:
        raise ValueError("仅支持文本源码、测试、配置和文档")
    return candidate


def _text(path: Path) -> str:
    """读取有界 UTF-8 文本，拒绝二进制内容。"""
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or b"\0" in raw:
        raise ValueError("文件过大或不是文本源码")
    return raw.decode("utf-8")


def _files(root: Path, directory: str = "."):
    """有界遍历；依赖目录和软链接不进入检索。"""
    base = _resolve(root, directory, directory=True)
    if not base.is_dir():
        raise ValueError("目录不存在")
    visited = 0
    for current, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not name.startswith(".") and not (Path(current) / name).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            visited += 1
            if visited > 10_000:
                return
            relative = path.relative_to(root)
            if path.is_symlink() or _protected(relative):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or name in {"Dockerfile", "Makefile", ".gitignore"}:
                yield path


class CodeWorkspace:
    def __init__(self, project_path: str | None = None):
        """直接读取给定绝对项目路径；省略时读取服务所在项目。"""
        path = Path(project_path).expanduser() if project_path else PROJECT_ROOT
        if not path.is_absolute() or not path.is_dir() or _protected(path):
            raise ValueError("请提供存在的本地项目绝对路径，受保护目录不可读取")
        self.root = path.resolve()
        if _protected(self.root):
            raise ValueError("受保护目录不可读取")
        self.info = {"name": self.root.name, "path": str(self.root)}

    def list_files(self, directory: str = ".") -> dict:
        """列出最多 500 个源码路径；达到上限时要求缩小目录。"""
        paths = []
        for path in _files(self.root, directory):
            paths.append(str(path.relative_to(self.root)))
            if len(paths) > 500:
                break
        return {"workspace": self.info, "files": paths[:500], "truncated": len(paths) > 500,
                "note": "仅列普通文本源码；大型项目请指定子目录"}

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> dict:
        """返回有界源码片段与行号。"""
        if start_line < 1 or end_line < start_line or end_line - start_line >= 300:
            raise ValueError("每次读取 1 到 300 行")
        text = _text(_resolve(self.root, path))
        lines = text.splitlines()
        selected = "\n".join(f"{number}: {line}" for number, line in enumerate(lines[start_line - 1:end_line], start_line))
        return {"path": path, "total_lines": len(lines), "content": selected[:18000],
                "truncated": end_line < len(lines) or len(selected) > 18000}

    def search_code(self, query: str, directory: str = ".") -> dict:
        """按字面关键词搜索源码，避免任意正则或 shell 执行。"""
        if not query or len(query) > 200:
            raise ValueError("搜索词长度应为 1 到 200 字符")
        matches, scanned, truncated = [], 0, False
        for path in _files(self.root, directory):
            scanned += 1
            if scanned > 1000:
                truncated = True
                break
            try:
                lines = _text(path).splitlines()
            except (ValueError, UnicodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                if query.casefold() in line.casefold():
                    matches.append({"path": str(path.relative_to(self.root)), "line": number, "text": line[:350]})
                    if len(matches) >= 60:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": truncated, "scanned_files": scanned}

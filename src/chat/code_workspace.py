"""本地源码工具：项目目录白名单、版本校验修改和有界检查。"""

import difflib
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".local", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache", "chroma_db_en", ".ssh", ".aws", ".codex", ".agents"}
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".css", ".html", ".sh", ".sql", ".xml", ".rst", ".gitignore"}
MAX_FILE_BYTES = 250_000
_EDIT_LOCK = threading.RLock()


def workspace_catalog() -> list[dict]:
    """服务器配置可信项目根；模型和浏览器不能自行扩大访问范围。"""
    configured = os.environ.get("ISSUE_AGENT_WORKSPACES")
    roots = json.loads(configured) if configured else [str(PROJECT_ROOT)]
    if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
        raise ValueError("ISSUE_AGENT_WORKSPACES 必须是绝对项目路径的 JSON 数组")
    result, seen = [], set()
    for value in roots:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("项目路径必须是绝对路径")
        root = path.resolve()
        if not root.is_dir() or root in seen:
            continue
        seen.add(root)
        result.append({"id": hashlib.sha256(str(root).encode()).hexdigest()[:16], "name": root.name, "path": str(root)})
    return result


def workspace_info(workspace_id: str | None) -> dict | None:
    """空 ID 表示未连接，必须显式选择服务端配置的项目。"""
    if workspace_id is None:
        return None
    item = next((item for item in workspace_catalog() if item["id"] == workspace_id), None)
    if item is None:
        raise ValueError("本地项目未配置或已移除，请重新选择")
    return item


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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    def __init__(self, workspace_id: str | None, allow_edits: bool = False):
        """只连接服务端白名单中的项目；修改与执行须由页面开启。"""
        info = workspace_info(workspace_id)
        if info is None:
            raise ValueError("请在对话的本地项目入口选择项目，随后可直接读取源码，无需粘贴代码")
        self.info, self.root, self.allow_edits = info, Path(info["path"]), allow_edits

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
        """返回行号和整文件版本，修改时使用该版本防止覆盖新变更。"""
        if start_line < 1 or end_line < start_line or end_line - start_line >= 300:
            raise ValueError("每次读取 1 到 300 行")
        text = _text(_resolve(self.root, path))
        lines = text.splitlines()
        selected = "\n".join(f"{number}: {line}" for number, line in enumerate(lines[start_line - 1:end_line], start_line))
        return {"path": path, "sha256": _sha(text), "total_lines": len(lines), "content": selected[:18000],
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

    def edit_file(self, path: str, old_text: str, new_text: str, expected_sha256: str) -> dict:
        """替换唯一代码片段或创建文件，原内容和版本均匹配才原子写入。"""
        if not self.allow_edits:
            raise ValueError("本对话为只读模式，请开启“允许修改并运行检查”后再修改")
        if len(new_text.encode("utf-8")) > MAX_FILE_BYTES or "\0" in new_text:
            raise ValueError("修改内容过大或含二进制内容")
        target = _resolve(self.root, path)
        with _EDIT_LOCK:
            original = _text(target) if target.exists() else ""
            version = _sha(original) if target.exists() else "missing"
            if expected_sha256 != version:
                raise ValueError("文件版本已变化；先重新读取再修改，不能覆盖其他改动")
            if target.exists():
                if not old_text or original.count(old_text) != 1:
                    raise ValueError("old_text 必须在文件中唯一匹配，请提供更多上下文")
                changed = original.replace(old_text, new_text, 1)
            else:
                if old_text:
                    raise ValueError("创建文件时 old_text 应为空，expected_sha256 应为 missing")
                changed = new_text
            if len(changed.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError("修改后文件超过大小限制")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".issue-agent-", dir=target.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                    stream.write(changed)
                os.chmod(temporary, target.stat().st_mode & 0o777 if target.exists() else 0o644)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            diff = "".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile=path, tofile=path))
            return {"path": path, "sha256": _sha(changed), "changed": original != changed,
                    "diff": diff[:16000], "truncated": len(diff) > 16000}

    def run_checks(self, check: str, paths: list[str] | None = None) -> dict:
        """运行固定项目检查，不开放任意 shell；项目测试仍是本机代码执行。"""
        if not self.allow_edits:
            raise ValueError("请开启“允许修改并运行检查”后再执行项目检查")
        paths = paths or []
        if check == "python-tests":
            for path in paths:
                target = _resolve(self.root, path)
                if target.suffix != ".py" or not target.is_file():
                    raise ValueError("测试路径必须是项目内存在的 Python 文件")
            python = self.root / ".venv" / "bin" / "python"
            command = [str(python) if python.exists() else "python3", "-m", "pytest", "-p", "no:cacheprovider", "-q", *(paths or ["tests"])]
            cwd = self.root
        elif check in {"frontend-build", "frontend-lint", "frontend-test"}:
            if paths:
                raise ValueError("前端检查不接受额外路径")
            command = ["npm", "run", check.removeprefix("frontend-")]
            cwd = self.root / "frontend"
        else:
            raise ValueError("检查仅支持 python-tests、frontend-build、frontend-lint、frontend-test")
        if not cwd.is_dir() or cwd.is_symlink():
            raise ValueError("项目检查目录不存在或是软链接")
        # 检查过程不继承服务端模型/GitHub 凭据；保留运行时必要路径。
        env = {key: value for key, value in os.environ.items()
               if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", key, re.I)}
        env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DEEPSEEK_API_KEY="", GITHUB_TOKEN="")
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            output.seek(0)
            raw = output.read(20001)
        text = raw[:20000].decode("utf-8", errors="replace")
        # 即使测试从项目配置读取凭据，也不把已知服务端秘密放入模型证据。
        for key, value in os.environ.items():
            if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", key, re.I) and len(value) >= 8:
                text = text.replace(value, "[redacted]")
        return {"check": check, "command": command, "exit_code": process.returncode, "timed_out": timed_out,
                "output": text, "truncated": len(raw) > 20000}

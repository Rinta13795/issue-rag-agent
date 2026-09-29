"""按需把公开 GitHub 仓库的近期 Issue 建成独立、可复用的本地快照。"""

import hashlib
import json
import os
import pickle
import re
import shutil
import ssl
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import certifi
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from config import INDEX_BATCH_SIZE
from src.chat.models import SyncRepositoryStatus
from src.docstore import build_docstore
from src.indexer import chunk_issue, load_embeddings, tokenize

GITHUB_INDEX_ROOT = Path(__file__).resolve().parents[2] / ".local" / "github_repos"
MAX_ISSUES = 300
MAX_PAGES = 20
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,39}/[A-Za-z0-9_.-]{1,100}$")


def normalize_repository(value: str) -> str:
    """只接受 owner/repo 或公开 GitHub 仓库 URL；请求始终发往固定 GitHub API 域。"""
    value = value.strip().rstrip("/")
    value = re.sub(r"^https://github\.com/", "", value, flags=re.IGNORECASE)
    if value.endswith(".git"):
        value = value[:-4]
    if not _REPOSITORY_PATTERN.fullmatch(value) or ".." in value:
        raise ValueError("请输入公开 GitHub 仓库的 owner/repo，例如 HKUDS/OpenHarness")
    return value


def repository_id(full_name: str) -> str:
    return "gh-" + hashlib.sha256(full_name.lower().encode()).hexdigest()[:16]


def snapshot_complete(project: str) -> bool:
    directory = GITHUB_INDEX_ROOT / project
    return all((directory / name).exists() for name in
               ("manifest.json", "bm25.pkl", "docstore.pkl", "chroma/chroma.sqlite3"))


def _request_json(url: str):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "issue-rag-agent"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=30,
                     context=ssl.create_default_context(cafile=certifi.where())) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError("仓库不存在、不是公开仓库，或当前 Token 无权访问") from exc
        if exc.code in (403, 429):
            raise ValueError("GitHub API 额度不足或访问受限；稍后重试，或在服务端配置 GITHUB_TOKEN") from exc
        raise ValueError(f"GitHub API 返回 HTTP {exc.code}") from exc


def fetch_issues(full_name: str) -> tuple[str, list[dict]]:
    owner, repo = full_name.split("/", 1)
    base = f"https://api.github.com/repos/{owner}/{repo}"
    metadata = _request_json(base)
    if metadata.get("private"):
        raise ValueError("当前只支持公开 GitHub 仓库，不会同步私有仓库")
    canonical = normalize_repository(metadata["full_name"])
    issues = []
    # GitHub Issues API 也返回 PR，故按实际 Issue 数截断，而非只取三页。
    for page in range(1, MAX_PAGES + 1):
        records = _request_json(f"{base}/issues?state=all&sort=updated&direction=desc&per_page=100&page={page}")
        for item in records:
            if "pull_request" not in item:
                issues.append(item)
                if len(issues) >= MAX_ISSUES:
                    return canonical, issues
        if len(records) < 100:
            break
    return canonical, issues


def _normalize_issue(item: dict, project: str) -> dict:
    return {
        "id": f"{project}:{item['number']}", "title": item.get("title") or "",
        "body": (item.get("body") or "")[:20000], "labels": [label["name"] for label in item.get("labels", [])],
        "component": "", "status": item.get("state", "open"), "project": project,
    }


def build_repository_snapshot(full_name: str, on_stage=lambda stage: None) -> str:
    """仅首次建库；临时目录完成后原子发布，失败不留下可见的半成品。"""
    requested = normalize_repository(full_name)
    on_stage("fetching")
    canonical, raw_issues = fetch_issues(requested)
    project = repository_id(canonical)
    final_dir = GITHUB_INDEX_ROOT / project
    if snapshot_complete(project):
        return project
    if not raw_issues:
        raise ValueError("该仓库没有可索引的公开 Issue")
    on_stage("indexing")
    issues = [_normalize_issue(item, project) for item in raw_issues]
    GITHUB_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{project}-", dir=GITHUB_INDEX_ROOT))
    try:
        bm25 = BM25Okapi([tokenize(f"{issue['title']} {issue['body']}") for issue in issues])
        with (temp_dir / "bm25.pkl").open("wb") as file:
            pickle.dump({"bm25": bm25, "ids": [issue["id"] for issue in issues]}, file)
        build_docstore(issues, path=str(temp_dir / "docstore.pkl"))
        vectorstore = Chroma(collection_name="issues", embedding_function=load_embeddings(),
                             persist_directory=str(temp_dir / "chroma"),
                             collection_metadata={"hnsw:space": "cosine"})
        docs, ids = [], []
        for issue in issues:
            for index, document in enumerate(chunk_issue(issue)):
                docs.append(document)
                ids.append(f"{issue['id']}:{index}")
                if len(docs) >= INDEX_BATCH_SIZE:
                    vectorstore.add_documents(docs, ids=ids)
                    docs, ids = [], []
        if docs:
            vectorstore.add_documents(docs, ids=ids)
        manifest = {
            "repository": canonical, "repository_id": project, "issue_count": len(issues),
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "coverage": f"最近更新的最多 {MAX_ISSUES} 条公开 Issue（最多扫描 {MAX_PAGES} 页，排除 PR）",
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if final_dir.exists():
            raise RuntimeError("目标快照目录已存在但不完整，请检查本地文件后重试")
        temp_dir.rename(final_dir)
        return project
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


class RepositorySyncManager:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="github-issue-sync")
        self._lock = threading.Lock()
        self._jobs: dict[str, SyncRepositoryStatus] = {}
        self._running: dict[str, str] = {}

    def start(self, value: str) -> SyncRepositoryStatus:
        full_name = normalize_repository(value)
        project = repository_id(full_name)
        if snapshot_complete(project):
            return SyncRepositoryStatus(job_id="cached", repository=full_name, status="completed",
                                        message="已复用本地 Issue 快照", repository_id=project)
        with self._lock:
            if full_name.lower() in self._running:
                return self._jobs[self._running[full_name.lower()]].model_copy()
            job = SyncRepositoryStatus(job_id=f"sync_{uuid.uuid4().hex[:12]}", repository=full_name,
                                       status="queued", message="等待同步")
            self._jobs[job.job_id] = job
            self._running[full_name.lower()] = job.job_id
        self._executor.submit(self._run, job.job_id, full_name)
        return job.model_copy()

    def get(self, job_id: str) -> SyncRepositoryStatus | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def _run(self, job_id: str, full_name: str) -> None:
        try:
            def update(stage: str):
                with self._lock:
                    self._jobs[job_id].status = stage
                    self._jobs[job_id].message = "正在读取公开 Issue" if stage == "fetching" else "正在建立本地检索快照"

            project = build_repository_snapshot(full_name, update)
            with self._lock:
                self._jobs[job_id].status = "completed"
                self._jobs[job_id].message = "快照可用于新对话"
                self._jobs[job_id].repository_id = project
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].message = str(exc) if isinstance(exc, ValueError) else "同步失败，请检查网络或本地模型后重试"
        finally:
            with self._lock:
                self._running.pop(full_name.lower(), None)


_manager: RepositorySyncManager | None = None


def get_sync_manager() -> RepositorySyncManager:
    global _manager
    if _manager is None:
        _manager = RepositorySyncManager()
    return _manager

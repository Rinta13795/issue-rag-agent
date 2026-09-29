"""聊天可选仓库；只展示本机确实存在的索引。"""

import json
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

from config import BM25_INDEX_PATH
from src.chat.github_sync import GITHUB_INDEX_ROOT
from src.chat.models import ChatRepository

OVERLAY_DIR = Path(__file__).resolve().parents[2] / ".local" / "openharness_index"


@lru_cache(maxsize=1)
def _gitbugs_repositories() -> tuple[ChatRepository, ...]:
    repositories: list[ChatRepository] = []
    base_path = Path(BM25_INDEX_PATH)
    if base_path.exists():
        with base_path.open("rb") as file:
            counts = Counter(issue_id.split(":", 1)[0] for issue_id in pickle.load(file)["ids"])
        repositories.extend(ChatRepository(id=project, label=project, issue_count=count, source="GitBugs 历史快照")
                            for project, count in sorted(counts.items()))
    return tuple(repositories)


def list_repositories() -> list[ChatRepository]:
    repositories = list(_gitbugs_repositories())
    manifest = OVERLAY_DIR / "manifest.json"
    if manifest.exists() and (OVERLAY_DIR / "bm25.pkl").exists() and (OVERLAY_DIR / "chroma" / "chroma.sqlite3").exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        repositories.append(ChatRepository(id="openharness", label="HKUDS / OpenHarness（旧快照）",
                                           issue_count=int(data["issue_count"]), source=f"GitHub 快照 · {data['synced_at'][:10]}",
                                           github_url="https://github.com/HKUDS/OpenHarness"))
    if GITHUB_INDEX_ROOT.exists():
        for directory in sorted(GITHUB_INDEX_ROOT.iterdir()):
            manifest = directory / "manifest.json"
            if not manifest.exists() or not (directory / "bm25.pkl").exists() or not (directory / "docstore.pkl").exists() or not (directory / "chroma" / "chroma.sqlite3").exists():
                continue
            data = json.loads(manifest.read_text(encoding="utf-8"))
            repositories.append(ChatRepository(
                id=data["repository_id"], label=data["repository"], issue_count=int(data["issue_count"]),
                source=f"GitHub 近期 Issue 快照 · {data['synced_at'][:10]} · {data['coverage']}",
                github_url=f"https://github.com/{data['repository']}",
            ))
    return repositories


def repository_exists(repository_id: str) -> bool:
    return any(repo.id == repository_id for repo in list_repositories())


def get_chat_retrieval_dependencies(repository_id: str):
    """GitHub 仓库使用独立快照；GitBugs 历史库共享原索引并硬过滤。"""
    from src.agent.graph import get_cached_reranker, get_retrieval_dependencies

    if repository_id == "openharness":
        return _get_overlay_retriever(str(OVERLAY_DIR)), get_cached_reranker()
    if repository_id.startswith("gh-"):
        if not repository_exists(repository_id):
            raise ValueError("该仓库快照不存在或不完整")
        return _get_overlay_retriever(str(GITHUB_INDEX_ROOT / repository_id)), get_cached_reranker()
    if not repository_exists(repository_id):
        raise ValueError("未找到该仓库的本地检索库")
    return get_retrieval_dependencies()


@lru_cache(maxsize=8)
def _get_overlay_retriever(directory: str):
    from langchain_chroma import Chroma

    from src.docstore import load_docstore
    from src.indexer import load_embeddings
    from src.retrievers.bm25_retriever import BM25Retriever
    from src.retrievers.hybrid_retriever import HybridRetriever
    from src.retrievers.vector_retriever import VectorRetriever

    vectorstore = Chroma(collection_name="issues", embedding_function=load_embeddings(),
                         persist_directory=str(Path(directory) / "chroma"))
    return HybridRetriever(VectorRetriever(vectorstore),
                           BM25Retriever(str(Path(directory) / "bm25.pkl")),
                           docstore=load_docstore(str(Path(directory) / "docstore.pkl")))

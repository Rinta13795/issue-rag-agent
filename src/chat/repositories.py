"""聊天可选仓库；只展示本机确实存在的索引。"""

import json
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

from config import BM25_INDEX_PATH
from src.chat.models import ChatRepository

OVERLAY_DIR = Path(__file__).resolve().parents[2] / ".local" / "openharness_index"


@lru_cache(maxsize=1)
def list_repositories() -> list[ChatRepository]:
    repositories: list[ChatRepository] = []
    base_path = Path(BM25_INDEX_PATH)
    if base_path.exists():
        with base_path.open("rb") as file:
            counts = Counter(issue_id.split(":", 1)[0] for issue_id in pickle.load(file)["ids"])
        repositories.extend(ChatRepository(id=project, label=project, issue_count=count, source="GitBugs 历史快照")
                            for project, count in sorted(counts.items()))
    manifest = OVERLAY_DIR / "manifest.json"
    if manifest.exists() and (OVERLAY_DIR / "bm25.pkl").exists() and (OVERLAY_DIR / "chroma" / "chroma.sqlite3").exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        repositories.append(ChatRepository(id="openharness", label="HKUDS / OpenHarness",
                                           issue_count=int(data["issue_count"]), source=f"GitHub 快照 · {data['synced_at'][:10]}"))
    return repositories


def repository_exists(repository_id: str) -> bool:
    return any(repo.id == repository_id for repo in list_repositories())


def get_chat_retrieval_dependencies(repository_id: str):
    """OpenHarness 独立索引；历史 8 库共享原索引，但检索时必须硬过滤。"""
    from src.agent.graph import get_cached_reranker, get_retrieval_dependencies

    if repository_id == "openharness":
        return _get_openharness_retriever(), get_cached_reranker()
    return get_retrieval_dependencies()


@lru_cache(maxsize=1)
def _get_openharness_retriever():
    from langchain_chroma import Chroma

    from src.docstore import load_docstore
    from src.indexer import load_embeddings
    from src.retrievers.bm25_retriever import BM25Retriever
    from src.retrievers.hybrid_retriever import HybridRetriever
    from src.retrievers.vector_retriever import VectorRetriever

    vectorstore = Chroma(collection_name="issues", embedding_function=load_embeddings(),
                         persist_directory=str(OVERLAY_DIR / "chroma"))
    return HybridRetriever(VectorRetriever(vectorstore),
                           BM25Retriever(str(OVERLAY_DIR / "bm25.pkl")),
                           docstore=load_docstore(str(OVERLAY_DIR / "docstore.pkl")))

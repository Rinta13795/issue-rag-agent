"""一次性脚本：ChromaDB 已有 N 个 chunk，从断点续传剩余部分，然后重建 BM25 索引。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from langchain_chroma import Chroma

from config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    INDEX_BATCH_SIZE,
)
from src.data_loader import clean, load_data, normalize
from src.indexer import chunk_issue, load_embeddings, build_bm25_index


def resume_chroma(all_issues: list[dict]) -> None:
    embeddings = load_embeddings()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    already_indexed = vectorstore._collection.count()
    logger.info("ChromaDB 当前已有 {} 个 chunk", already_indexed)

    all_chunks = []
    for issue in all_issues:
        all_chunks.extend(chunk_issue(issue))
    logger.info("全量 chunk 数量：{}", len(all_chunks))

    remaining = all_chunks[already_indexed:]
    if not remaining:
        logger.info("ChromaDB 已完整，无需补写")
        return

    logger.info("需要补写 {} 个 chunk（从第 {} 个开始）", len(remaining), already_indexed)

    for start in range(0, len(remaining), INDEX_BATCH_SIZE):
        batch = remaining[start:start + INDEX_BATCH_SIZE]
        vectorstore.add_documents(batch)
        done = already_indexed + start + len(batch)
        logger.info("ChromaDB 写入进度：{}/{}", done, len(all_chunks))

    logger.info("ChromaDB 补写完成，总 chunk：{}", vectorstore._collection.count())


if __name__ == "__main__":
    raw = load_data()
    normalized = [normalize(i) for i in raw]
    issues = clean(normalized)
    logger.info("issue 加载完成：{} 条", len(issues))

    resume_chroma(issues)

    logger.info("开始重建 BM25 索引（issue 粒度）")
    build_bm25_index(issues)
    logger.info("全部完成")

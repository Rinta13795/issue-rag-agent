"""评估运行器：对 4 种检索配置跑评估，记录指标和 P50/P95 延迟，结果保存为 JSON。

4 种配置：
- vector_only  ：纯向量检索基线
- bm25_only    ：纯 BM25 检索基线
- hybrid       ：BM25 + 向量 + RRF 混合检索
- hybrid_rerank：混合检索 + Cross-Encoder 重排序

相对 v1 修复的三个公平性问题：
1. 排除自命中：评测 query 本身就在索引里，v1 会把“检索到自己”当作占位结果，
   现在检索 top_k+1 条并剔除 query 自身 id 后再截断；
2. 公平截断点：Rerank 后最多只有 10 条候选，只计算 @5/@10 指标，
   不再和返回 30 条的配置比较 recall@30；
3. 可控抽样：--limit N 从 held-out test set 按固定 seed 抽样，
   结果 JSON 中记录样本量与抽样方式，避免把抽样结果当成全量结果。

运行：
    python -m eval.run_eval                 # 全量 held-out test set
    python -m eval.run_eval --limit 200     # 固定 seed 抽样 200 条
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EVAL_RESULTS_DIR,
    EVAL_SAMPLE_SIZE,
    EVAL_SEED,
    HYBRID_TOP_K,
    RERANK_TOP_K,
)
from eval.golden_set_builder import load_golden_set
from eval.metrics import compute_all
from src.docstore import load_docstore
from src.reranker import Reranker
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.vector_retriever import VectorRetriever

# Rerank 输出最多 top_k + 动态扩展（默认 10 条），只在这些截断点上评估。
RERANK_CUTOFFS = (5, 10)
RETRIEVAL_CUTOFFS = (5, 10, 30)


def _load_vector_retriever() -> VectorRetriever:
    """输入无，输出加载了已有 ChromaDB 索引的 VectorRetriever 实例。"""
    from langchain_chroma import Chroma
    from src.indexer import load_embeddings

    embeddings = load_embeddings()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return VectorRetriever(vectorstore)


def _exclude_self(retrieved: list[dict], query_id: str, top_k: int) -> list[str]:
    """输入检索结果、query 自身 id 和目标截断数，输出剔除自命中后的 id 列表。

    评测 query 本身也在索引中，检索到自己不算能力，必须剔除后再截断，
    否则自命中会挤占一个候选位置并抬高其余目标的排名难度。
    """
    ids = [str(doc["id"]) for doc in retrieved if str(doc["id"]) != str(query_id)]
    return ids[:top_k]


def _aggregate_metrics(all_metrics: list[dict], latencies: list[float]) -> dict:
    """输入逐条指标列表和延迟列表，输出各指标均值和 P50/P95 延迟。"""
    result = {
        k: round(float(np.mean([m[k] for m in all_metrics])), 4)
        for k in all_metrics[0]
    }
    result["p50_latency"] = round(float(np.percentile(latencies, 50)), 3)
    result["p95_latency"] = round(float(np.percentile(latencies, 95)), 3)
    return result


def evaluate_retriever(retriever, queries: list[dict], top_k: int = HYBRID_TOP_K) -> dict:
    """输入 retriever 实例和 query 列表，输出平均指标和延迟统计。"""
    all_metrics, latencies = [], []

    for q in tqdm(queries):
        t0 = time.time()
        # 多取 1 条：剔除自命中后仍能保证 top_k 个有效候选。
        results = retriever.search(q["text"], top_k=top_k + 1)
        latencies.append(time.time() - t0)

        retrieved_ids = _exclude_self(results, q["id"], top_k)
        all_metrics.append(compute_all(retrieved_ids, q["duplicate_of"], RETRIEVAL_CUTOFFS))

    return _aggregate_metrics(all_metrics, latencies)


def evaluate_hybrid_rerank(hybrid_retriever, reranker, queries: list[dict]) -> dict:
    """输入混合检索器、重排序器和 query 列表，输出 Hybrid+Rerank 配置的指标和延迟。

    Rerank 取 Top-5（分差接近时动态扩展），指标只在 @5/@10 截断点上计算。
    """
    all_metrics, latencies = [], []

    for q in tqdm(queries):
        t0 = time.time()
        candidates = hybrid_retriever.search(q["text"], top_k=HYBRID_TOP_K + 1)
        # 精排前剔除自命中，避免自己挤占 Rerank Top-5 的位置。
        candidates = [c for c in candidates if str(c["id"]) != str(q["id"])][:HYBRID_TOP_K]
        reranked = reranker.rerank(q["text"], candidates, top_k=RERANK_TOP_K)
        latencies.append(time.time() - t0)

        retrieved_ids = [str(r["id"]) for r in reranked]
        all_metrics.append(compute_all(retrieved_ids, q["duplicate_of"], RERANK_CUTOFFS))

    return _aggregate_metrics(all_metrics, latencies)


def sample_queries(queries: list[dict], limit: int | None, seed: int = EVAL_SEED) -> list[dict]:
    """输入全量 query 列表和抽样上限，输出固定 seed 的随机抽样结果。"""
    if limit is None or limit >= len(queries):
        return queries
    return random.Random(seed).sample(queries, limit)


def save_results(data: dict, path: str) -> None:
    """输入结果字典和路径，写入 JSON 文件。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存：{}", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"按固定 seed 从 test set 抽样的条数；不传则跑全量（建议先用 {EVAL_SAMPLE_SIZE} 快速对比）",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
        help="要评估的配置名列表",
    )
    args = parser.parse_args()

    _, test_set = load_golden_set()
    queries = sample_queries(test_set, args.limit)
    logger.info("held-out test set：全量 {} 条，本次评估 {} 条", len(test_set), len(queries))

    logger.info("初始化检索器...")
    vector_retriever = _load_vector_retriever()
    bm25_retriever = BM25Retriever(BM25_INDEX_PATH)
    # docstore 与在线链路一致：hybrid_rerank 配置下 BM25-only 候选补齐正文后才能被公平精排。
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, docstore=load_docstore())
    reranker = Reranker() if "hybrid_rerank" in args.configs else None

    configs = {
        "vector_only": lambda q: evaluate_retriever(vector_retriever, q),
        "bm25_only": lambda q: evaluate_retriever(bm25_retriever, q),
        "hybrid": lambda q: evaluate_retriever(hybrid_retriever, q),
        "hybrid_rerank": lambda q: evaluate_hybrid_rerank(hybrid_retriever, reranker, q),
    }

    run_meta = {
        "test_set_total": len(test_set),
        "evaluated_queries": len(queries),
        "sampling": f"seed={EVAL_SEED} random sample" if args.limit else "full test set",
        "self_hit_excluded": True,
        "rerank_cutoffs": list(RERANK_CUTOFFS),
    }

    all_results = {"_meta": run_meta}
    for name in args.configs:
        if name not in configs:
            logger.warning("跳过未知配置：{}", name)
            continue
        logger.info("开始评估：{}", name)
        metrics = configs[name](queries)
        all_results[name] = metrics
        save_results({"_meta": run_meta, **metrics}, f"{EVAL_RESULTS_DIR}/{name}.json")
        logger.info("完成 {}：{}", name, metrics)

    save_results(all_results, f"{EVAL_RESULTS_DIR}/summary.json")
    print(f"\n全部完成，结果保存在 {EVAL_RESULTS_DIR}/")


if __name__ == "__main__":
    main()

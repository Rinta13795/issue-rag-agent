"""检索评估指标：Recall@K、Precision@K、MRR、nDCG@K。

输入为检索返回的 issue_id 列表和 ground truth id 列表，输出各项指标浮点值。
所有函数均为无状态纯函数，可单独调用或通过 compute_all 批量计算。
"""

import numpy as np


def recall_at_k(retrieved_ids: list[str], golden_ids: list[str], k: int) -> float:
    """输入检索结果和 ground truth，输出 Top-K 召回率。"""
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(golden_ids)) / max(len(golden_ids), 1)


def precision_at_k(retrieved_ids: list[str], golden_ids: list[str], k: int) -> float:
    """输入检索结果和 ground truth，输出 Top-K 精确率。"""
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(golden_ids)) / k


def mrr(retrieved_ids: list[str], golden_ids: list[str]) -> float:
    """输入检索结果和 ground truth，输出首个命中位置的倒数排名。未命中返回 0。"""
    golden_set = set(golden_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in golden_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], golden_ids: list[str], k: int) -> float:
    """输入检索结果和 ground truth，输出 nDCG@K。

    nDCG 是 BEIR/MTEB/TREC 的标准主指标，比 MRR 更精细：考虑所有命中位置的折扣累积增益。

    IDCG 必须基于“全部相关目标数”计算（min(len(golden), k) 个理想命中），
    而不是基于“已检索到的 gains”重排：v1 的实现里只要命中 1 条，无论还有多少
    相关目标没被召回，nDCG 都会被高估（漏检的目标不计入理想分母）。
    """
    golden_set = set(golden_ids)
    gains = [1.0 if rid in golden_set else 0.0 for rid in retrieved_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    ideal_hit_count = min(len(golden_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hit_count))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all(
    retrieved_ids: list[str],
    golden_ids: list[str],
    cutoffs: tuple[int, ...] | None = None,
) -> dict:
    """输入检索结果和 ground truth，一次输出全部评估指标。

    返回 dict 包含 recall@5、recall@10、recall@30、precision@5、mrr、ndcg@10。
    cutoffs 可定制：Rerank 后最多只有 10 条结果，此时计算 recall@30 没有意义，
    调用方应传入 (5, 10) 保证不同方法只在公平的截断点上比较。
    """
    cutoffs = cutoffs or (5, 10, 30)
    result = {}
    for k in cutoffs:
        result[f"recall@{k}"] = recall_at_k(retrieved_ids, golden_ids, k)
    result["precision@5"] = precision_at_k(retrieved_ids, golden_ids, 5)
    result["mrr"] = mrr(retrieved_ids, golden_ids)
    result["ndcg@10"] = ndcg_at_k(retrieved_ids, golden_ids, 10)
    return result

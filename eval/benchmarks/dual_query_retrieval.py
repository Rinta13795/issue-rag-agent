"""比较“只检索改写 Query”与“原文 + 改写 Query 双路检索”的 Benchmark。

这个 Benchmark 固定 raw_issue、rewritten_query 和人工确认的 duplicate ID，
不会调用 Query Analysis LLM，因此测到的差异只来自检索策略，而不是模型输出波动。

运行示例：
    python -m eval.benchmarks.dual_query_retrieval run
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CASES_PATH = Path("eval/benchmarks/data/dual_query_cases.json")
DEFAULT_OUTPUT_PATH = Path("eval_results/dual_query_retrieval.json")


def _load_verified_cases(path: Path) -> list[dict[str, Any]]:
    """读取人工确认的案例，草稿不会进入正式结果。"""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    cases = data.get("cases", [])
    verified = [case for case in cases if case.get("status") == "verified"]
    if not verified:
        raise ValueError(
            f"{path} 中没有 status=verified 的案例；"
            "请先人工确认 rewritten_query 和 expected_duplicate_ids。"
        )

    required_fields = (
        "id",
        "project",
        "raw_issue",
        "rewritten_query",
        "expected_duplicate_ids",
        "query_analysis_version",
        "case_type",
    )
    seen_case_ids = set()
    for case in verified:
        missing = [field for field in required_fields if field not in case]
        if missing:
            raise ValueError(f"案例缺少必填字段：{case.get('id', '<unknown>')} -> {missing}")

        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("每条 verified 案例都必须有非空字符串 id")
        if case_id in seen_case_ids:
            raise ValueError(f"案例 id 重复：{case_id}")
        seen_case_ids.add(case_id)

        for field in (
            "project",
            "raw_issue",
            "rewritten_query",
            "query_analysis_version",
            "case_type",
        ):
            value = case[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"案例 {case_id} 的 {field} 必须是非空字符串")

        expected_ids = case["expected_duplicate_ids"]
        if not isinstance(expected_ids, list) or not expected_ids:
            raise ValueError(
                f"案例 {case_id} 的 expected_duplicate_ids 必须是非空列表"
            )
        invalid_expected_id = any(
            not isinstance(issue_id, str) or not issue_id.strip()
            for issue_id in expected_ids
        )
        if invalid_expected_id:
            raise ValueError(
                f"案例 {case_id} 的 expected_duplicate_ids 只能包含非空字符串"
            )

        component = case.get("component")
        if component is not None and (
            not isinstance(component, str) or not component.strip()
        ):
            raise ValueError(f"案例 {case_id} 的 component 只能是非空字符串或 null")
    return verified


def _validate_expected_ids_exist(cases: list[dict[str, Any]], retriever: Any) -> None:
    """确认标注目标存在于当前 BM25 索引，避免把不可达目标算成检索失败。"""
    indexed_ids = {
        str(issue_id)
        for issue_id in retriever.bm25_retriever.ids
    }
    unreachable = {
        case["id"]: sorted(set(case["expected_duplicate_ids"]) - indexed_ids)
        for case in cases
        if set(case["expected_duplicate_ids"]) - indexed_ids
    }
    if unreachable:
        raise ValueError(
            "以下 expected_duplicate_ids 不存在于当前 BM25 索引："
            f"{json.dumps(unreachable, ensure_ascii=False)}"
        )


def _ranking_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str],
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    """计算 Hit Rate、Recall、MRR 和使用完整相关集 IDCG 的 nDCG。"""
    expected = set(expected_ids)
    metrics: dict[str, float] = {}

    for cutoff in cutoffs:
        top_ids = retrieved_ids[:cutoff]
        hits = len(set(top_ids) & expected)
        metrics[f"hit_rate@{cutoff}"] = float(hits > 0)
        metrics[f"recall@{cutoff}"] = hits / max(len(expected), 1)

    first_hit_rank = next(
        (
            rank
            for rank, issue_id in enumerate(retrieved_ids, start=1)
            if issue_id in expected
        ),
        None,
    )
    metrics["mrr"] = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    ndcg_cutoff = 10
    gains = [
        1.0 if issue_id in expected else 0.0
        for issue_id in retrieved_ids[:ndcg_cutoff]
    ]
    dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )
    ideal_hit_count = min(len(expected), ndcg_cutoff)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hit_count + 1)
    )
    metrics["ndcg@10"] = dcg / idcg if idcg else 0.0
    return metrics


def _run_variant(
    retriever: Any,
    reranker: Any,
    case: dict[str, Any],
    variant: str,
    top_k: int,
) -> dict[str, Any]:
    """运行一个检索变体，并分别记录召回和重排结果。"""
    rewritten_query = case["rewritten_query"]
    component = case.get("component")
    filter_dict = {"component": component} if component else None

    retrieval_started = time.perf_counter()
    if variant == "single_rewritten":
        candidates = retriever.search(
            query=rewritten_query,
            top_k=top_k,
            filter_dict=filter_dict,
        )
    elif variant == "dual_query":
        candidates = retriever.search_queries(
            queries=[rewritten_query, case["raw_issue"]],
            top_k=top_k,
            filter_dict=filter_dict,
        )
    else:
        raise ValueError(f"未知 Benchmark 变体：{variant}")
    retrieval_latency = time.perf_counter() - retrieval_started

    # Reranker 会原地写入 rerank_score，所以复制候选，避免变体之间共享可变数据。
    rerank_input = [doc.copy() for doc in candidates]
    rerank_started = time.perf_counter()
    reranked = reranker.rerank(
        query=rewritten_query,
        docs=rerank_input,
    )
    rerank_latency = time.perf_counter() - rerank_started

    candidate_ids = [str(doc["id"]) for doc in candidates]
    reranked_ids = [str(doc["id"]) for doc in reranked]
    expected_ids = [str(issue_id) for issue_id in case["expected_duplicate_ids"]]

    retrieval_cutoffs = tuple(dict.fromkeys((5, 10, top_k)))
    rerank_cutoffs = (5, 10)
    return {
        "retrieved_ids": candidate_ids,
        "reranked_ids": reranked_ids,
        "retrieval_metrics": _ranking_metrics(
            candidate_ids,
            expected_ids,
            retrieval_cutoffs,
        ),
        "rerank_metrics": _ranking_metrics(
            reranked_ids,
            expected_ids,
            rerank_cutoffs,
        ),
        "retrieval_latency_seconds": retrieval_latency,
        "rerank_latency_seconds": rerank_latency,
        "total_latency_seconds": retrieval_latency + rerank_latency,
    }


def _mean_metrics(
    records: list[dict[str, Any]],
    variant: str,
    stage: str,
) -> dict[str, float]:
    """聚合某个变体在 Retrieval 或 Rerank 阶段的排名指标。"""
    key = f"{stage}_metrics"
    metric_names = records[0][variant][key].keys()
    return {
        name: round(
            statistics.mean(record[variant][key][name] for record in records),
            4,
        )
        for name in metric_names
    }


def _latency_summary(
    records: list[dict[str, Any]],
    variant: str,
) -> dict[str, float]:
    """聚合平均、P50 和 P95 延迟。"""
    totals = sorted(record[variant]["total_latency_seconds"] for record in records)

    def percentile(percent: float) -> float:
        if len(totals) == 1:
            return totals[0]
        position = (len(totals) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return totals[lower]
        weight = position - lower
        return totals[lower] * (1 - weight) + totals[upper] * weight

    return {
        "mean_total_seconds": round(statistics.mean(totals), 4),
        "p50_total_seconds": round(percentile(0.50), 4),
        "p95_total_seconds": round(percentile(0.95), 4),
        "mean_retrieval_seconds": round(
            statistics.mean(
                record[variant]["retrieval_latency_seconds"]
                for record in records
            ),
            4,
        ),
        "mean_rerank_seconds": round(
            statistics.mean(
                record[variant]["rerank_latency_seconds"]
                for record in records
            ),
            4,
        ),
    }


def summarize(
    records: list[dict[str, Any]],
    top_k: int,
    include_case_types: bool = True,
) -> dict[str, Any]:
    """汇总两种策略的指标差异、兜底收益、退化比例和延迟。"""
    if not records:
        raise ValueError("没有可聚合的 Benchmark 结果")

    variants = ("single_rewritten", "dual_query")
    summary: dict[str, Any] = {
        "run_count": len(records),
        "case_count": len({record["case_id"] for record in records}),
    }
    for variant in variants:
        summary[variant] = {
            "retrieval": _mean_metrics(records, variant, "retrieval"),
            "rerank": _mean_metrics(records, variant, "rerank"),
            "latency": _latency_summary(records, variant),
        }

    delta: dict[str, Any] = {"retrieval": {}, "rerank": {}, "latency": {}}
    for stage in ("retrieval", "rerank"):
        for metric, dual_value in summary["dual_query"][stage].items():
            single_value = summary["single_rewritten"][stage][metric]
            delta[stage][metric] = round(dual_value - single_value, 4)
    for metric, dual_value in summary["dual_query"]["latency"].items():
        single_value = summary["single_rewritten"]["latency"][metric]
        delta["latency"][metric] = round(dual_value - single_value, 4)
    summary["delta_dual_minus_single"] = delta

    retrieval_metric = f"hit_rate@{top_k}"
    retrieval_rescues = sum(
        not record["single_rewritten"]["retrieval_metrics"][retrieval_metric]
        and record["dual_query"]["retrieval_metrics"][retrieval_metric]
        for record in records
    )
    retrieval_regressions = sum(
        record["single_rewritten"]["retrieval_metrics"][retrieval_metric]
        and not record["dual_query"]["retrieval_metrics"][retrieval_metric]
        for record in records
    )
    rerank_rescues = sum(
        not record["single_rewritten"]["rerank_metrics"]["hit_rate@10"]
        and record["dual_query"]["rerank_metrics"]["hit_rate@10"]
        for record in records
    )
    rerank_regressions = sum(
        record["single_rewritten"]["rerank_metrics"]["hit_rate@10"]
        and not record["dual_query"]["rerank_metrics"]["hit_rate@10"]
        for record in records
    )
    count = len(records)
    summary["safety_checks"] = {
        f"retrieval_rescue_count@{top_k}": retrieval_rescues,
        f"retrieval_regression_count@{top_k}": retrieval_regressions,
        f"retrieval_rescue_rate@{top_k}": round(retrieval_rescues / count, 4),
        f"retrieval_regression_rate@{top_k}": round(
            retrieval_regressions / count,
            4,
        ),
        "rerank_rescue_count@10": rerank_rescues,
        "rerank_regression_count@10": rerank_regressions,
        "rerank_rescue_rate@10": round(rerank_rescues / count, 4),
        "rerank_regression_rate@10": round(rerank_regressions / count, 4),
    }

    # rewrite_drift 和 rewrite_good 的平均值可能方向相反，所以同时按案例类型报告，
    # 防止总体 Recall 掩盖某一类输入上的明显收益或退化。
    if include_case_types:
        case_types = sorted({record["case_type"] for record in records})
        summary["by_case_type"] = {
            case_type: summarize(
                [
                    record
                    for record in records
                    if record["case_type"] == case_type
                ],
                top_k,
                include_case_types=False,
            )
            for case_type in case_types
        }
    return summary


def run_benchmark(
    cases_path: Path,
    output_path: Path,
    repeats: int,
    limit: int | None,
    top_k: int,
) -> dict[str, Any]:
    """在相同索引和固定 Query 对上运行单路与双路检索。"""
    cases = _load_verified_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]

    # 延迟导入：查看帮助、读取结果或运行指标单测时不会加载 Chroma 和模型。
    from config import (
        BM25_INDEX_PATH,
        CHROMA_COLLECTION,
        CHROMA_PERSIST_DIR,
        EMBED_MODEL,
        RERANKER_MODEL,
    )
    from src.agent.graph import get_hybrid_retriever, get_reranker

    retriever = get_hybrid_retriever()
    reranker = get_reranker()
    _validate_expected_ids_exist(cases, retriever)
    records = []
    for repeat in range(1, repeats + 1):
        for case_index, case in enumerate(cases):
            # 交替运行顺序，减少缓存预热总是偏向同一个变体造成的延迟误差。
            variants = ["single_rewritten", "dual_query"]
            if (repeat + case_index) % 2:
                variants.reverse()

            outputs = {
                variant: _run_variant(
                    retriever=retriever,
                    reranker=reranker,
                    case=case,
                    variant=variant,
                    top_k=top_k,
                )
                for variant in variants
            }
            records.append(
                {
                    "case_id": case["id"],
                    "case_type": case["case_type"],
                    "repeat": repeat,
                    "expected_duplicate_ids": case["expected_duplicate_ids"],
                    "single_rewritten": outputs["single_rewritten"],
                    "dual_query": outputs["dual_query"],
                }
            )

    result = {
        "benchmark_id": "dual-query-retrieval-v1",
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "case_count": len(cases),
        "run_count": len(records),
        "repeats": repeats,
        "top_k": top_k,
        "fixed_variables": {
            "llm_query_analysis_called": False,
            "rerank_query": "rewritten_query for both variants",
            "component_filter": "same fixed case value for both variants",
            "embedding_model": EMBED_MODEL,
            "reranker_model": RERANKER_MODEL,
            "chroma_directory": CHROMA_PERSIST_DIR,
            "chroma_collection": CHROMA_COLLECTION,
            "bm25_index": BM25_INDEX_PATH,
        },
        "summary": summarize(records, top_k),
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行单路与双路检索对照")
    run_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--top-k", type=int, default=30)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats 必须大于等于 1")
    if args.top_k < 10:
        raise ValueError("--top-k 必须大于等于 10，才能计算固定的 Top-5/10 指标")

    result = run_benchmark(
        cases_path=args.cases,
        output_path=args.output,
        repeats=args.repeats,
        limit=args.limit,
        top_k=args.top_k,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"结果已保存：{args.output}")


if __name__ == "__main__":
    main()

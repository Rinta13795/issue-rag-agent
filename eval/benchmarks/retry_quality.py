"""评估“诊断信息是否改善低置信度重试”的 Benchmark。

本脚本不会在导入时加载模型。只有执行 run 子命令时，才会初始化完整 Agent，
产生真实 LLM 调用和本地检索开销。

运行示例：
    python -m eval.benchmarks.retry_quality run --variant legacy --label before
    python -m eval.benchmarks.retry_quality run --variant diagnostic --label after
    python -m eval.benchmarks.retry_quality compare \
        eval_results/retry_quality_before.json \
        eval_results/retry_quality_after.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CASES_PATH = Path("eval/benchmarks/data/retry_cases.json")
DEFAULT_RESULTS_DIR = Path("eval_results")

# 旧版只向下一轮展示三个字段。使用同一份当前代码切换 prompt，可以隔离
# “是否提供诊断上下文”这一变量，不必在两个 Git 提交之间来回切换。
LEGACY_RETRY_PROMPT = """## 上次重试反思

上次改写 query：{last_query}
上次置信度：{last_confidence}（低于阈值 0.7 触发重试）
上次决策：{last_decision}

上次置信度低说明召回不准。请换角度改写 query。"""

SUMMARY_KEYS = [
    "final_retrieval_hit_rate",
    "final_related_issue_hit_rate",
    "duplicate_decision_accuracy",
    "required_term_retention_rate",
    "forbidden_term_case_rate",
    "retry_trigger_rate",
    "average_retry_count",
    "average_latency_seconds",
]


def _load_verified_cases(path: Path) -> list[dict[str, Any]]:
    """只加载经过人工确认的案例，草稿案例不会进入正式结果。"""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    cases = data.get("cases", [])
    verified = [case for case in cases if case.get("status") == "verified"]
    if not verified:
        raise ValueError(
            f"{path} 中没有 status=verified 的案例；"
            "请先人工确认 expected_duplicate_ids 和实体标注。"
        )
    return verified


def _contains_term(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def _evaluate_record(case: dict[str, Any], final: dict, latency: float, repeat: int) -> dict:
    """把一次 Agent 输出转换为可聚合的单条结果。"""
    expected_ids = set(case["expected_duplicate_ids"])
    reranked_ids = [str(doc.get("id", "")) for doc in final.get("reranked_docs", [])]
    related_ids = [str(issue_id) for issue_id in final.get("related_issues", [])]
    rewritten_query = str(final.get("rewritten_query", ""))
    required_terms = case.get("required_terms", [])
    forbidden_terms = case.get("forbidden_terms", [])

    retained_terms = [
        term for term in required_terms
        if _contains_term(rewritten_query, term)
    ]
    forbidden_hits = [
        term for term in forbidden_terms
        if _contains_term(rewritten_query, term)
    ]

    return {
        "case_id": case["id"],
        "repeat": repeat,
        "expected_duplicate_ids": sorted(expected_ids),
        "final_decision": final.get("decision"),
        "final_confidence": final.get("confidence"),
        "final_related_issues": related_ids,
        "final_reranked_ids": reranked_ids,
        "final_rewritten_query": rewritten_query,
        "retry_count": final.get("retry_count", 0),
        "latency_seconds": round(latency, 4),
        "retrieval_hit": bool(expected_ids & set(reranked_ids)),
        "related_issue_hit": bool(expected_ids & set(related_ids)),
        "duplicate_decision_correct": (
            final.get("decision") == "duplicate"
            and bool(expected_ids & set(related_ids))
        ),
        "required_terms": required_terms,
        "retained_terms": retained_terms,
        "forbidden_terms": forbidden_terms,
        "forbidden_hits": forbidden_hits,
        "retry_history": final.get("previous_decisions", []),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """聚合最终检索、决策、改写忠实度、重试次数和延迟指标。"""
    if not records:
        raise ValueError("没有可聚合的 Benchmark 结果")

    total_required = sum(len(record["required_terms"]) for record in records)
    retained_required = sum(len(record["retained_terms"]) for record in records)
    count = len(records)

    return {
        "run_count": count,
        "case_count": len({record["case_id"] for record in records}),
        "final_retrieval_hit_rate": round(
            sum(record["retrieval_hit"] for record in records) / count, 4
        ),
        "final_related_issue_hit_rate": round(
            sum(record["related_issue_hit"] for record in records) / count, 4
        ),
        "duplicate_decision_accuracy": round(
            sum(record["duplicate_decision_correct"] for record in records) / count, 4
        ),
        "required_term_retention_rate": round(
            retained_required / total_required if total_required else 0.0, 4
        ),
        "forbidden_term_case_rate": round(
            sum(bool(record["forbidden_hits"]) for record in records) / count, 4
        ),
        "retry_trigger_rate": round(
            sum(record["retry_count"] > 1 for record in records) / count, 4
        ),
        "average_retry_count": round(
            statistics.mean(record["retry_count"] for record in records), 4
        ),
        "average_latency_seconds": round(
            statistics.mean(record["latency_seconds"] for record in records), 4
        ),
    }


def run_benchmark(
    cases_path: Path,
    variant: str,
    label: str,
    repeats: int,
    limit: int | None,
    output_path: Path,
) -> dict[str, Any]:
    """运行指定 retry prompt 变体并保存逐条结果和汇总指标。"""
    cases = _load_verified_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]

    # 延迟导入保证 compare 和纯指标测试不需要安装 LangChain 或加载模型。
    from src.agent import nodes
    from src.agent.graph import build_graph

    if variant == "legacy":
        nodes._PROMPTS["query_analysis_retry"] = LEGACY_RETRY_PROMPT

    graph = build_graph()
    records = []
    for repeat in range(1, repeats + 1):
        for case in cases:
            initial_state = {
                "raw_issue": case["raw_issue"],
                "retry_count": 0,
                "previous_decisions": [],
            }
            started = time.perf_counter()
            final = graph.invoke(initial_state)
            latency = time.perf_counter() - started
            records.append(_evaluate_record(case, final, latency, repeat))

    result = {
        "benchmark_id": "retry-context-v1",
        "status": "completed",
        "variant": variant,
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "repeats": repeats,
        "summary": summarize(records),
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def compare_results(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """比较两次结果；正负仅表示数值变化，是否更好需结合指标含义判断。"""
    with baseline_path.open(encoding="utf-8") as file:
        baseline = json.load(file)
    with candidate_path.open(encoding="utf-8") as file:
        candidate = json.load(file)

    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    delta = {
        key: round(candidate_summary[key] - baseline_summary[key], 4)
        for key in SUMMARY_KEYS
    }
    return {
        "benchmark_id": "retry-context-v1",
        "baseline": baseline.get("label", baseline_path.stem),
        "candidate": candidate.get("label", candidate_path.stem),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "delta": delta,
        "notes": {
            "higher_is_better": [
                "final_retrieval_hit_rate",
                "final_related_issue_hit_rate",
                "duplicate_decision_accuracy",
                "required_term_retention_rate",
            ],
            "lower_is_better": [
                "forbidden_term_case_rate",
                "average_latency_seconds",
            ],
            "diagnostic_only": [
                "retry_trigger_rate",
                "average_retry_count",
            ],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="运行一次 Benchmark")
    run_parser.add_argument("--variant", choices=["legacy", "diagnostic"], required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser("compare", help="比较两次 Benchmark 结果")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        output = args.output or DEFAULT_RESULTS_DIR / f"retry_quality_{args.label}.json"
        result = run_benchmark(
            cases_path=args.cases,
            variant=args.variant,
            label=args.label,
            repeats=args.repeats,
            limit=args.limit,
            output_path=output,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"结果已保存：{output}")
        return

    comparison = compare_results(args.baseline, args.candidate)
    rendered = json.dumps(comparison, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

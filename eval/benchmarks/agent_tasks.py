"""对话调查 Agent 的端到端任务评测。

和检索层评测（eval/run_eval.py）互补：检索层回答"搜不搜得到"，本脚本回答
"Agent 能不能顺着 Issue 找到真正的修复，并且不夸大结论"。

三个子命令：
    build    从公开仓库的 Issue 时间线自动标注案例（已合并 PR 修复 / 无修复），
             写成待核验草稿；人工抽查后把 status 改成 verified 才进入正式运行。
    run      用真实模型跑 InvestigationRuntime，记录工具轨迹、回答、引用与成本；
             --variant 切换工具集做消融（agent / no_pr_tools / search_only）。
    compare  比较两次运行的汇总指标。

评分全部基于可核对的事实（时间线、PR 合并状态、实际工具调用、引用 ID）；
LLM 裁判只作可选补充（--judge），不替代规则指标。

运行示例：
    python -m eval.benchmarks.agent_tasks build --repository owner/repo --limit 40
    python -m eval.benchmarks.agent_tasks run --variant agent --label agent_v1 --repeats 3
    python -m eval.benchmarks.agent_tasks run --variant search_only --label rag_baseline --repeats 3
    python -m eval.benchmarks.agent_tasks compare eval_results/agent_tasks_rag_baseline.json \\
        eval_results/agent_tasks_agent_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CASES_PATH = Path("eval/benchmarks/data/agent_cases.json")
DEFAULT_RESULTS_DIR = Path("eval_results")
DEFAULT_QUESTION = "这个问题现在解决了吗？如果解决了，是怎么解决的？请说明依据。"

# 消融变体：在同一份代码里收窄工具集，隔离"工具能力"这一个变量。
# search_only 近似"检索后直接回答"的传统 RAG 基线。
VARIANT_TOOLS: dict[str, set[str] | None] = {
    "agent": None,
    "no_pr_tools": {"search_issues", "read_issue", "draft_issue", "ask_user"},
    "search_only": {"search_issues", "ask_user"},
}

# PR 正文里关闭 Issue 的关键词（GitHub closing keywords 的常见写法）。
_CLOSING_TEMPLATE = r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:[\w.-]+/[\w.-]+)?#{number}\b"

# 回答中"已修复"类断言；用于无修复案例的夸大检测（规则代理，非真值）。
_FIX_CLAIM = re.compile(r"已修复|已经修复|已解决|已经解决|修复已合并|已合并修复|has been fixed|was fixed|fixed in", re.I)

SUMMARY_KEYS = [
    "completion_rate",
    "asked_user_rate",
    "key_evidence_recall",
    "citation_hit_rate",
    "overclaim_rate",
    "judge_correct_rate",
    "budget_hit_rate",
    "average_tool_calls",
    "average_model_calls",
    "average_total_tokens",
    "average_latency_seconds",
]


def closes_issue(pr_body: str, issue_number: int) -> bool:
    """输入 PR 正文和 Issue 编号，判断 PR 是否用关闭关键词指向该 Issue。"""
    pattern = re.compile(_CLOSING_TEMPLATE.format(number=issue_number), re.I)
    return bool(pattern.search(pr_body or ""))


def label_case(issue: dict[str, Any], prs: list[dict[str, Any]], number: int) -> dict[str, Any] | None:
    """输入 read_issue 的结果、关联 PR 的 read_pr 结果和 Issue 编号，输出预期标注或 None。

    只保留证据明确的两类，其余交给人工：
    - fixed_merged：Issue 已关闭，且至少一个已合并的关联 PR 用关闭关键词指向它；
    - no_fix：Issue 仍开着，时间线读取成功且没有任何关联 PR。
    """
    state = str(issue.get("state", ""))
    fix_prs = sorted(
        pr["number"] for pr in prs
        if pr.get("merged") and closes_issue(str(pr.get("body", "")), number)
    )
    if state == "closed" and fix_prs:
        return {"resolution": "fixed_merged", "fix_prs": fix_prs}
    if state == "open" and not issue.get("linked_prs") and not issue.get("links_error"):
        return {"resolution": "no_fix", "fix_prs": []}
    return None


def score_record(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """输入一个案例和一次运行记录，输出可聚合的单条指标。"""
    expected = case["expected"]
    fix_prs = set(expected.get("fix_prs", []))
    read_prs = {
        int(step["arguments"].get("number", 0))
        for step in record.get("steps", [])
        if step.get("tool") == "read_pr" and step.get("status") == "completed"
    }
    cited_prs = {
        int(match.group(1))
        for citation in record.get("citations", [])
        for match in [re.match(r"pr:.+#(\d+)$", str(citation))]
        if match
    }
    answer = str(record.get("answer") or "")
    is_fix_case = expected["resolution"] == "fixed_merged"
    return {
        "case_id": case["id"],
        "resolution": expected["resolution"],
        "repeat": record.get("repeat", 1),
        "completed": record.get("status") == "completed",
        "asked_user": record.get("status") == "waiting_for_user",
        # 关键证据召回：修复案例里，Agent 是否真的读到了那个修复 PR。
        "key_evidence_hit": bool(fix_prs & read_prs) if is_fix_case else None,
        # 引用命中：最终回答是否把修复 PR 作为依据引用出来。
        "citation_hit": bool(fix_prs & cited_prs) if is_fix_case else None,
        # 夸大：无修复案例里，回答却声称已修复。
        "overclaim": bool(_FIX_CLAIM.search(answer)) if not is_fix_case else None,
        "judge_verdict": record.get("judge_verdict"),
        "budget_hit": bool(record.get("budget_hit")),
        "tool_calls": len(record.get("steps", [])),
        "model_calls": int(record.get("model_calls") or 0),
        "total_tokens": int(record.get("prompt_tokens") or 0) + int(record.get("completion_tokens") or 0),
        "latency_seconds": float(record.get("latency_seconds") or 0.0),
    }


def _rate(values: list[bool | None]) -> float | None:
    """输入布尔值列表（None 表示不适用），输出比例；全部不适用时返回 None。"""
    applicable = [value for value in values if value is not None]
    return round(sum(applicable) / len(applicable), 4) if applicable else None


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """输入逐条指标，输出总体与按案例类型的汇总。"""
    if not scores:
        raise ValueError("没有可聚合的评测结果")

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        verdicts = [item["judge_verdict"] for item in items if item["judge_verdict"]]
        return {
            "run_count": len(items),
            "case_count": len({item["case_id"] for item in items}),
            "completion_rate": _rate([item["completed"] for item in items]),
            "asked_user_rate": _rate([item["asked_user"] for item in items]),
            "key_evidence_recall": _rate([item["key_evidence_hit"] for item in items]),
            "citation_hit_rate": _rate([item["citation_hit"] for item in items]),
            "overclaim_rate": _rate([item["overclaim"] for item in items]),
            "judge_correct_rate": _rate([verdict == "correct" for verdict in verdicts]) if verdicts else None,
            "budget_hit_rate": _rate([item["budget_hit"] for item in items]),
            "average_tool_calls": round(statistics.mean(item["tool_calls"] for item in items), 2),
            "average_model_calls": round(statistics.mean(item["model_calls"] for item in items), 2),
            "average_total_tokens": round(statistics.mean(item["total_tokens"] for item in items), 1),
            "average_latency_seconds": round(statistics.mean(item["latency_seconds"] for item in items), 2),
        }

    summary = aggregate(scores)
    summary["by_resolution"] = {
        resolution: aggregate([item for item in scores if item["resolution"] == resolution])
        for resolution in sorted({item["resolution"] for item in scores})
    }
    # 同一案例重复运行时结论是否一致：衡量模型随机性带来的不稳定。
    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in scores:
        by_case.setdefault(item["case_id"], []).append(item)
    repeated = [items for items in by_case.values() if len(items) > 1]
    if repeated:
        def outcome(item: dict[str, Any]) -> Any:
            return item["key_evidence_hit"] if item["resolution"] == "fixed_merged" else item["overclaim"]
        summary["consistency_rate"] = round(
            sum(len({outcome(item) for item in items}) == 1 for items in repeated) / len(repeated), 4
        )
    return summary


def _load_verified_cases(path: Path) -> list[dict[str, Any]]:
    """只加载人工核验过的案例；自动标注的草稿不进入正式结果。"""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    cases = [case for case in data.get("cases", []) if case.get("status") == "verified"]
    if not cases:
        raise ValueError(f"{path} 中没有 status=verified 的案例；请先 build 并人工核验。")
    return cases


def build_cases(repository: str, limit: int, output_path: Path, scan: int = 100) -> dict[str, Any]:
    """输入公开仓库名，扫描最近的 Issue，自动标注证据明确的案例并写成草稿。"""
    # 延迟导入：只有 build 才访问 GitHub。
    from src.chat.github_research import read_issue, read_pr
    from src.chat.github_sync import _request_json, normalize_repository

    repository = normalize_repository(repository)
    items = _request_json(
        f"https://api.github.com/repos/{repository}/issues?state=all&sort=updated&direction=desc&per_page={min(scan, 100)}"
    )
    cases, skipped = [], 0
    for item in items:
        if "pull_request" in item or len(cases) >= limit:
            continue
        number = int(item["number"])
        try:
            issue = read_issue(repository, number)
            prs = [read_pr(repository, link["number"]) for link in issue.get("linked_prs", [])[:5]]
        except Exception:
            skipped += 1
            continue
        expected = label_case(issue, prs, number)
        if expected is None:
            skipped += 1
            continue
        cases.append({
            "id": f"{repository}#{number}",
            "status": "auto_labeled",
            "repository": repository,
            "issue_number": number,
            "title": issue.get("title", ""),
            "question": DEFAULT_QUESTION,
            "expected": expected,
            "label_source": "github_timeline_auto",
        })
    data = {
        "benchmark_id": "agent-task-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "note": "自动标注草稿：人工抽查无误后把 status 改成 verified 才会进入正式运行。",
        "skipped_ambiguous_or_failed": skipped,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _session_record(session: Any, latency: float, repeat: int) -> dict[str, Any]:
    """把一次运行后的 ChatSession 转成与评分无关的原始记录，便于事后复盘。"""
    final = session.messages[-1] if session.messages and session.messages[-1].role == "assistant" else None
    steps = [
        {"tool": step.tool, "arguments": step.arguments, "status": step.status}
        for step in session.runtime_steps
    ]
    budget_hit = session.model_calls >= 6 or any(
        "预算" in str(step.result.get("error", "")) for step in session.runtime_steps
    )
    return {
        "repeat": repeat,
        "status": session.status,
        "answer": final.content if final else None,
        "citations": final.citations if final else [],
        "steps": steps,
        "model_calls": session.model_calls,
        "prompt_tokens": session.prompt_tokens,
        "completion_tokens": session.completion_tokens,
        "budget_hit": budget_hit,
        "final_response_mode": session.final_response_mode,
        "last_error": session.last_error,
        "latency_seconds": round(latency, 3),
    }


def judge_answer(case: dict[str, Any], answer: str) -> str | None:
    """可选的 LLM 裁判：对照标注判断结论是否正确，返回 correct / incorrect / hedged。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    from src.chat.context import parse_json_object

    judge = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
                       temperature=0, max_tokens=200)
    expected = case["expected"]
    truth = (f"该 Issue 已由已合并的 PR {expected['fix_prs']} 修复。" if expected["resolution"] == "fixed_merged"
             else "该 Issue 仍未关闭，没有任何关联 PR，不能说已修复。")
    response = judge.invoke([
        SystemMessage(content=("你是评测裁判。只比较回答的结论与事实是否一致，不评价文风。"
                               '只返回 JSON：{"verdict":"correct|incorrect|hedged","reason":"一句话"}。'
                               "hedged 表示回答没有给出明确结论。")),
        HumanMessage(content=f"事实：{truth}\n\n待评回答：{answer}"),
    ])
    verdict = parse_json_object(response.content).get("verdict")
    return verdict if verdict in ("correct", "incorrect", "hedged") else None


def run_benchmark(cases_path: Path, variant: str, label: str, repeats: int,
                  limit: int | None, output_path: Path, use_judge: bool) -> dict[str, Any]:
    """用真实模型逐案例运行 Agent，保存逐条记录、评分和汇总。"""
    cases = _load_verified_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]

    # 延迟导入：compare 与纯评分单测不需要模型和网络依赖。
    from src.chat import runtime
    from src.chat.github_sync import fetch_issue, repository_id
    from src.chat.memory import MemoryStore
    from src.chat.service import ChatService
    from src.chat.store import ChatStore

    allowed = VARIANT_TOOLS[variant]
    original = (runtime.DEFINITIONS, runtime.TOOLS)
    if allowed is not None:
        runtime.DEFINITIONS = {name: value for name, value in original[0].items() if name in allowed}
        runtime.TOOLS = [tool for tool in original[1] if tool["function"]["name"] in allowed]

    records, scores = [], []
    try:
        for repeat in range(1, repeats + 1):
            for case in cases:
                # 每次运行使用全新的内存会话与内存记忆，避免案例之间互相污染。
                store = ChatStore()
                service = ChatService(store=store, memory_store=MemoryStore())
                source = fetch_issue(f"https://github.com/{case['repository']}/issues/{case['issue_number']}")
                session = store.create(repository_id(source.repository), source_issue=source)
                message_id, _ = store.add_user_message(session.session_id, case.get("question", DEFAULT_QUESTION),
                                                       f"bench-{repeat}")
                started = time.perf_counter()
                service.process_turn(session.session_id, message_id)
                record = _session_record(store.get(session.session_id), time.perf_counter() - started, repeat)
                if use_judge and record["answer"]:
                    record["judge_verdict"] = judge_answer(case, record["answer"])
                record["case_id"] = case["id"]
                records.append(record)
                scores.append(score_record(case, record))
    finally:
        runtime.DEFINITIONS, runtime.TOOLS = original

    result = {
        "benchmark_id": "agent-task-v1",
        "variant": variant,
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "repeats": repeats,
        "judge": use_judge,
        "summary": summarize(scores),
        "scores": scores,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def compare_results(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """比较两次运行的汇总指标；正负只表示数值变化，好坏要结合指标含义判断。"""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    delta = {}
    for key in SUMMARY_KEYS:
        before, after = baseline["summary"].get(key), candidate["summary"].get(key)
        delta[key] = round(after - before, 4) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
    return {
        "baseline": baseline.get("label"), "candidate": candidate.get("label"),
        "baseline_summary": baseline["summary"], "candidate_summary": candidate["summary"], "delta": delta,
        "notes": {
            "higher_is_better": ["completion_rate", "key_evidence_recall", "citation_hit_rate", "judge_correct_rate"],
            "lower_is_better": ["overclaim_rate", "budget_hit_rate", "average_total_tokens", "average_latency_seconds"],
            "diagnostic_only": ["asked_user_rate", "average_tool_calls", "average_model_calls"],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="从 Issue 时间线自动标注案例草稿")
    build.add_argument("--repository", required=True)
    build.add_argument("--limit", type=int, default=40)
    build.add_argument("--scan", type=int, default=100)
    build.add_argument("--output", type=Path, default=DEFAULT_CASES_PATH)

    run = sub.add_parser("run", help="用真实模型运行一次评测")
    run.add_argument("--variant", choices=sorted(VARIANT_TOOLS), default="agent")
    run.add_argument("--label", required=True)
    run.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--limit", type=int)
    run.add_argument("--judge", action="store_true", help="额外调用 LLM 裁判评结论")
    run.add_argument("--output", type=Path)

    compare = sub.add_parser("compare", help="比较两次运行结果")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "build":
        data = build_cases(args.repository, args.limit, args.output, args.scan)
        print(f"已写入 {len(data['cases'])} 条草稿到 {args.output}；跳过 {data['skipped_ambiguous_or_failed']} 条")
        return
    if args.command == "run":
        if args.repeats < 1:
            raise ValueError("--repeats 必须大于等于 1")
        output = args.output or DEFAULT_RESULTS_DIR / f"agent_tasks_{args.label}.json"
        result = run_benchmark(args.cases, args.variant, args.label, args.repeats, args.limit, output, args.judge)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"结果已保存：{output}")
        return
    print(json.dumps(compare_results(args.baseline, args.candidate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

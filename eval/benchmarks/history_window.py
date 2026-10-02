"""历史窗口策略对比：同一组固定多轮对话，在 DeepSeek 上实测缓存命中、成本与质量。

方案（src/chat/history.py 的策略）：
    A sliding       滑动窗口，最近 10 条（当前默认）
    B stepped       阶梯截断：攒到 20 条，一次性砍回最近 10 条
    C compact       只追加；未压缩历史超过 16 条时，用一次模型调用把旧历史压成摘要，保留最近 6 条
    D case_summary  只追加；超过阈值时用当时的案例摘要替换旧历史，不额外调用模型
    E full          只追加、不截断（参照组）

用法：
    # 离线：用替身模型跑通全部方案，并输出提示词前缀的结构分析（不是 DeepSeek 实测）
    python -m eval.benchmarks.history_window dry-run
    # 实测：需要环境变量 DEEPSEEK_API_KEY，且能访问 api.deepseek.com；先估算费用，超过上限就停
    python -m eval.benchmarks.history_window run --schemes A B C D E --runs 2 --max-cny 5

隔离：每个方案、每次重复在独立子进程中运行，系统提示词最前面加唯一标记 [scheme-X run-N]，
压缩与记忆整理的提示词也加同样标记，避免不同方案、不同重复之间互相蹭缓存。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMES = {"A": "sliding", "B": "stepped", "C": "compact", "D": "case_summary", "E": "full"}
RESULTS_DIR = Path("eval_results/history_window")
ANSWER_MAX_TOKENS = 300
COMPACT_MAX_TOKENS = 300
ORGANIZER_MAX_TOKENS = 900

# 价格（美元 / 百万 token）。来源见 评测/10-历史窗口方案对比.md，运行前请按官网核对后用参数覆盖。
DEFAULT_PRICE_HIT = 0.0028
DEFAULT_PRICE_MISS = 0.14
DEFAULT_PRICE_OUT = 0.28
DEFAULT_USD_CNY = 7.1

# 固定的 14 轮对话：前几轮埋下细节，第 12-14 轮考查。
CONVERSATION = [
    "我在 Windows 11 上用 acme-cli 3.4.1，自动更新一直失败，报错码 UPD-7741，日志里有 signature mismatch on delta package。以后回答请用要点列出，命令只给 PowerShell 版本。",
    "我已经试过清空 %LOCALAPPDATA%\\acme\\cache 然后重启，没有用。",
    "维护者那边有没有关联的 PR？帮我查一下这个 Issue 的进展。",
    "如果那个 PR 合并了，是不是已经发布到正式版了？",
    "我们公司网络走代理，会不会影响增量包下载？",
    "如果改成下载完整安装包而不是增量包，具体怎么操作？",
    "我又试了用管理员身份运行更新，还是同样的错误。",
    "日志里还出现了 cert chain incomplete，这个和签名错误有关系吗？",
    "有没有办法临时关闭自动更新，等官方修复？",
    "团队里其他人用 macOS 没有这个问题，这说明什么？",
    "帮我总结一下目前还能尝试的方案。",
    "回到最开始，我第一次报的错误码具体是什么？只回答错误码。",
    "我前面说过自己已经试过哪两个方法？",
    "按照我一开始提的要求，给我临时关闭自动更新的命令。",
]

# 质量检查：只在前几轮出现过的细节。
QUALITY_CHECKS = {
    12: {"name": "错误码", "all": [r"UPD-7741"], "none": []},
    13: {"name": "试过的两个方法", "all": [r"缓存|cache|LOCALAPPDATA", r"管理员"], "none": []},
    14: {"name": "只给 PowerShell 命令", "all": [r"Set-ItemProperty|New-ItemProperty|Set-Service|Stop-Service|Disable-ScheduledTask|PowerShell|\$env:"],
         "none": [r"\bsudo\b", r"\bsystemctl\b", r"\blaunchctl\b", r"#!/bin/(ba)?sh"]},
}


def check_quality(turn: int, answer: str | None) -> dict | None:
    """输入轮次与回答，输出该轮质量检查结果；不是检查轮返回 None。"""
    import re

    spec = QUALITY_CHECKS.get(turn)
    if spec is None:
        return None
    text = answer or ""
    passed = all(re.search(p, text, re.I) for p in spec["all"]) and not any(re.search(p, text, re.I) for p in spec["none"])
    return {"turn": turn, "name": spec["name"], "passed": passed, "answer": text[:600]}


# ---------- 假工具数据（所有方案相同） ----------

SOURCE_ISSUE = {
    "repository": "acme/updater", "number": 352, "state": "open",
    "title": "Auto-update fails with UPD-7741 on Windows",
    "url": "https://github.com/acme/updater/issues/352",
    "body": "Since 3.4.0 the updater fails on Windows with UPD-7741: signature mismatch on delta package. macOS is not affected.",
}


def _fake_read_issue(repository, number):
    return {**SOURCE_ISSUE, "number": number, "comments": [
        {"author": "maintainer", "body": "Root cause: the delta signer drops an intermediate cert on Windows. Fix in #359.", "url": SOURCE_ISSUE["url"] + "#c1"}],
        "linked_prs": [{"number": 359, "url": "https://github.com/acme/updater/pull/359", "title": "Verify delta signature with full cert chain",
                        "relationship": "cross_referenced", "note": "时间线关联；是否修复该问题需读取 PR 核对"}]}


def _fake_read_pr(repository, number):
    return {"number": number, "title": "Verify delta signature with full cert chain", "url": f"https://github.com/acme/updater/pull/{number}",
            "state": "closed", "merged": True, "merged_at": "2026-09-20T08:00:00Z",
            "body": "Fixes #352. Include the intermediate certificate when verifying delta packages. Will ship in 3.4.3.",
            "discussion": [{"author": "maintainer", "body": "Merged; release 3.4.3 is planned but not published yet."}], "files": []}


def _fake_search_prs(repository, query):
    return {"relationship": "search_candidate", "note": "相似 PR 候选，不能当成当前 Issue 的关联修复",
            "items": [{"number": 359, "title": "Verify delta signature with full cert chain", "state": "closed",
                       "url": "https://github.com/acme/updater/pull/359", "body": "Fixes #352."}], "incomplete_results": False}


class _FakeRetriever:
    def search(self, **kwargs):
        project = kwargs.get("project", "")
        return [{"id": f"{project}:301", "title": "Delta update signature mismatch behind corporate proxy",
                 "body": "Proxy rewriting TLS can break delta downloads; full installer works."},
                {"id": f"{project}:288", "title": "Disable auto update on Windows",
                 "body": "Set HKCU\\Software\\Acme\\Updater AutoUpdate=0 via Set-ItemProperty."}]


class _FakeReranker:
    def rerank(self, query, docs):
        return [dict(doc, rerank_score=0.8 - 0.1 * i) for i, doc in enumerate(docs)]


# ---------- 调用记录 ----------

class Recorder:
    """包一层模型客户端：只暴露 invoke（运行时因此走非流式），记录每次调用的真实用量与提示词。"""

    def __init__(self, model, kind: str, log: list, state: dict):
        self.model, self.kind, self.log, self.state = model, kind, log, state

    def bind_tools(self, tools):
        return Recorder(self.model.bind_tools(tools), self.kind, self.log, self.state)

    def invoke(self, messages):
        turn = self.state["turn"]
        index = sum(1 for item in self.log if item["turn"] == turn and item["kind"] == self.kind)
        started = time.perf_counter()
        response = self.model.invoke(messages)
        entry = {"turn": turn, "kind": self.kind, "call_index": index,
                 "latency_seconds": round(time.perf_counter() - started, 3),
                 "prompt_text": _serialize(messages), **_usage(response)}
        self.log.append(entry)
        return response


def _serialize(messages) -> str:
    parts = []
    for message in messages:
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        calls = json.dumps(getattr(message, "tool_calls", None) or [], ensure_ascii=False, sort_keys=True)
        parts.append(f"<{message.type}>{content}{calls}")
    return "".join(parts)


def _usage(response) -> dict:
    """从 DeepSeek 响应读取用量；命中字段优先用 prompt_cache_hit_tokens。拿不到就记 None，不猜。"""
    meta = getattr(response, "response_metadata", None) or {}
    raw = meta.get("token_usage") or meta.get("usage") or {}
    usage = getattr(response, "usage_metadata", None) or {}
    prompt = raw.get("prompt_tokens", usage.get("input_tokens"))
    hit = raw.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
    if hit is None:
        hit = (usage.get("input_token_details") or {}).get("cache_read")
    miss = raw.get("prompt_cache_miss_tokens")
    if miss is None and isinstance(prompt, int) and isinstance(hit, int):
        miss = prompt - hit
    return {"prompt_tokens": prompt, "hit_tokens": hit, "miss_tokens": miss,
            "completion_tokens": raw.get("completion_tokens", usage.get("output_tokens"))}


# ---------- 替身模型（dry-run） ----------

class _StubModel:
    def __init__(self, kind: str, state: dict):
        self.kind, self.state = kind, state

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        turn = self.state["turn"]
        if self.kind == "organizer":
            content = json.dumps({"summary": f"（替身摘要）截至第 {turn} 轮：Windows 上 UPD-7741，已试清缓存与管理员运行。", "entries": [], "preferences": []}, ensure_ascii=False)
        elif self.kind == "compact":
            content = f"（替身压缩摘要）第 {turn} 轮之前的对话要点。"
        else:
            content = json.dumps({"answer": f"（替身回答）第 {turn} 轮。", "citations": []}, ensure_ascii=False)
        return AIMessage(content=content)


# ---------- 单个方案的一次运行（在子进程里执行） ----------

def run_one(scheme: str, run: int, output: Path, dry_run: bool, model_name: str | None) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.chat import runtime
    from src.chat.context import parse_json_object
    from src.chat.memory import MemoryStore
    from src.chat.memory_prompt import memory_prompt_payload
    from src.chat.models import SourceIssue
    from src.chat.prompts import MEMORY_ORGANIZER_SYSTEM
    from src.chat.service import ChatService
    from src.chat.store import ChatStore

    marker = f"[scheme-{scheme} run-{run}]"
    runtime.HISTORY_STRATEGY = SCHEMES[scheme]
    runtime.SYSTEM = f"{marker}\n{runtime.SYSTEM}"
    runtime.HISTORY_COMPACT_SYSTEM = f"{marker}\n{runtime.HISTORY_COMPACT_SYSTEM}"
    # 固定工具集：去掉 ask_user（会进入暂停恢复路径，绕过历史窗口）和 draft_issue（与窗口无关的额外调用）。
    keep = {"search_issues", "read_issue", "read_pr", "search_prs"}
    runtime.DEFINITIONS = {k: v for k, v in runtime.DEFINITIONS.items() if k in keep}
    runtime.TOOLS = [t for t in runtime.TOOLS if t["function"]["name"] in keep]
    runtime.read_issue, runtime.read_pr, runtime.search_prs = _fake_read_issue, _fake_read_pr, _fake_search_prs
    runtime.search_live_issues = lambda *a, **k: []
    runtime.list_repositories = lambda: []

    state, log = {"turn": 0}, []
    if dry_run:
        answer, compact, organizer = (_StubModel(k, state) for k in ("answer", "compact", "organizer"))
    else:
        from langchain_openai import ChatOpenAI

        from config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_TEMPERATURE

        def client(max_tokens):
            return ChatOpenAI(model=model_name or DEEPSEEK_MODEL, api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL,
                              temperature=LLM_TEMPERATURE, max_tokens=max_tokens, timeout=60, max_retries=2)
        answer, compact, organizer = client(ANSWER_MAX_TOKENS), client(COMPACT_MAX_TOKENS), client(ORGANIZER_MAX_TOKENS)

    with tempfile.TemporaryDirectory() as memory_dir:
        store = ChatStore()
        service = ChatService(store=store, answer_llm=Recorder(answer, "answer", log, state),
                              planner_llm=Recorder(compact, "compact", log, state), memory_store=MemoryStore(Path(memory_dir)))
        service.retrieval_provider = lambda repository_id: (_FakeRetriever(), _FakeReranker())
        organizer_client = Recorder(organizer, "organizer", log, state)

        def extract(payload):
            response = organizer_client.invoke([SystemMessage(content=f"{marker}\n{MEMORY_ORGANIZER_SYSTEM}"),
                                                HumanMessage(content=json.dumps(memory_prompt_payload(payload), ensure_ascii=False, separators=(",", ":")))])
            return parse_json_object(response.content)
        service.memory_service.extractor = extract

        source = SourceIssue(repository=SOURCE_ISSUE["repository"], number=SOURCE_ISSUE["number"], title=SOURCE_ISSUE["title"],
                             body=SOURCE_ISSUE["body"], state=SOURCE_ISSUE["state"], url=SOURCE_ISSUE["url"])
        session_id = store.create("gh-benchmark", source_issue=source).session_id
        turns, started_all = [], time.perf_counter()
        for turn, text in enumerate(CONVERSATION, start=1):
            state["turn"] = turn
            message_id, _ = store.add_user_message(session_id, text, f"turn-{turn}")
            started = time.perf_counter()
            service.process_turn(session_id, message_id)
            service.memory_service.wait_for_idle(timeout=180)  # 让案例摘要按生产节奏更新到下一轮
            session = store.get(session_id)
            last = session.messages[-1]
            reply = last.content if last.role == "assistant" else None
            turns.append({"turn": turn, "status": session.status, "error": session.last_error,
                          "latency_seconds": round(time.perf_counter() - started, 3), "answer": reply,
                          "quality": check_quality(turn, reply)})
        final = store.get(session_id)

    result = {"scheme": scheme, "strategy": SCHEMES[scheme], "run": run, "dry_run": dry_run, "marker": marker,
              "elapsed_seconds": round(time.perf_counter() - started_all, 2), "turns": turns, "calls": log,
              "history_summary": final.history_summary, "history_window_start": final.history_window_start}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------- 汇总 ----------

def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def summarize_run(result: dict, prices: dict) -> dict:
    """输入一次运行的原始记录，输出命中率、用量、花费、耗时与质量。"""
    calls = result["calls"]
    answer = [c for c in calls if c["kind"] == "answer"]
    first = [c for c in answer if c["call_index"] == 0]
    later = [c for c in answer if c["call_index"] > 0]

    def rate(items):
        items = [c for c in items if isinstance(c.get("hit_tokens"), int) and isinstance(c.get("prompt_tokens"), int)]
        total = sum(c["prompt_tokens"] for c in items)
        return round(sum(c["hit_tokens"] for c in items) / total, 4) if total else None

    def cost(items):
        usd = 0.0
        for c in items:
            hit, miss, out = c.get("hit_tokens"), c.get("miss_tokens"), c.get("completion_tokens") or 0
            if not isinstance(hit, int) or not isinstance(miss, int):
                return None
            usd += (hit * prices["hit"] + miss * prices["miss"] + out * prices["out"]) / 1_000_000
        return round(usd * prices["fx"], 4)

    # 离线结构分析：每轮首次调用与上一轮首次调用的字符级公共前缀占比（不是缓存实测）。
    by_turn = {c["turn"]: c["prompt_text"] for c in first}
    shares = [(_common_prefix(by_turn[t - 1], by_turn[t]) / len(by_turn[t])) for t in sorted(by_turn) if t - 1 in by_turn and by_turn[t]]
    quality = [t["quality"] for t in result["turns"] if t["quality"]]
    return {
        "scheme": result["scheme"], "strategy": result["strategy"], "run": result["run"],
        "cross_turn_hit_rate": rate([c for c in first if c["turn"] >= 2]),
        "intra_turn_hit_rate": rate(later),
        "overall_hit_rate": rate(answer),
        "answer_input_tokens": sum(c["prompt_tokens"] or 0 for c in answer if isinstance(c.get("prompt_tokens"), int)) or None,
        "total_input_tokens": sum(c["prompt_tokens"] for c in calls if isinstance(c.get("prompt_tokens"), int)) or None,
        "cost_cny_answer": cost(answer),
        "cost_cny_total": cost(calls),
        "compact_calls": sum(1 for c in calls if c["kind"] == "compact"),
        "elapsed_seconds": result["elapsed_seconds"],
        "failed_turns": sum(1 for t in result["turns"] if t["status"] != "completed"),
        "quality_passed": sum(q["passed"] for q in quality), "quality_total": len(quality),
        "quality_detail": {q["name"]: q["passed"] for q in quality},
        "structural_prefix_share": round(sum(shares) / len(shares), 4) if shares else None,
    }


def _subprocess_run(scheme: str, run: int, dry_run: bool, model: str | None) -> Path:
    output = RESULTS_DIR / f"{'dry_' if dry_run else ''}{scheme}_run{run}.json"
    command = [sys.executable, "-m", "eval.benchmarks.history_window", "_one", "--scheme", scheme, "--run", str(run), "--output", str(output)]
    if dry_run:
        command.append("--dry-run")
    if model:
        command += ["--model", model]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{scheme} run {run} 失败：{completed.stderr[-2000:]}")
    return output


def estimate_cost(dry_results: list[dict], runs: int, prices: dict) -> dict:
    """用离线运行的提示词长度粗估费用（字符数 / 1.5 近似 token，按全部未命中算上限）。"""
    total_chars = sum(len(c["prompt_text"]) for r in dry_results for c in r["calls"])
    calls = sum(len(r["calls"]) for r in dry_results)
    tokens = total_chars / 1.5 * runs
    worst = (tokens * prices["miss"] + calls * runs * ANSWER_MAX_TOKENS * prices["out"]) / 1_000_000 * prices["fx"]
    return {"estimated_input_tokens": int(tokens), "estimated_calls": calls * runs, "worst_case_cny": round(worst, 2),
            "note": "离线替身不会调用工具、记忆整理输出也更短，真实调用次数和长度通常更多；以此为下限参考，并按全部未命中给出上限。"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "run"):
        p = sub.add_parser(name)
        p.add_argument("--schemes", nargs="*", default=list(SCHEMES))
        p.add_argument("--runs", type=int, default=1 if name == "dry-run" else 2)
        p.add_argument("--model")
        p.add_argument("--price-hit", type=float, default=DEFAULT_PRICE_HIT)
        p.add_argument("--price-miss", type=float, default=DEFAULT_PRICE_MISS)
        p.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
        p.add_argument("--usd-cny", type=float, default=DEFAULT_USD_CNY)
        p.add_argument("--max-cny", type=float, default=5.0)
    one = sub.add_parser("_one")
    one.add_argument("--scheme", required=True)
    one.add_argument("--run", type=int, required=True)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--dry-run", action="store_true")
    one.add_argument("--model")
    args = parser.parse_args()

    if args.command == "_one":
        run_one(args.scheme, args.run, args.output, args.dry_run, args.model)
        return

    prices = {"hit": args.price_hit, "miss": args.price_miss, "out": args.price_out, "fx": args.usd_cny}
    dry = args.command == "dry-run"
    if not dry:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise SystemExit("缺少环境变量 DEEPSEEK_API_KEY")
        probe = [json.loads(_subprocess_run(s, 0, True, None).read_text(encoding="utf-8")) for s in args.schemes]
        estimate = estimate_cost(probe, args.runs, prices)
        print(json.dumps({"preflight_estimate": estimate}, ensure_ascii=False, indent=2))
        if estimate["worst_case_cny"] > args.max_cny:
            raise SystemExit(f"预估最坏费用 ¥{estimate['worst_case_cny']} 超过上限 ¥{args.max_cny}，已停止。确认后用 --max-cny 调高。")

    jobs = [(s, r) for s in args.schemes for r in range(1, args.runs + 1)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        outputs = list(pool.map(lambda job: _subprocess_run(job[0], job[1], dry, args.model), jobs))
    rows = [summarize_run(json.loads(p.read_text(encoding="utf-8")), prices) for p in outputs]
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "dry_run": dry, "prices_usd_per_m": prices, "rows": rows}
    out = RESULTS_DIR / ("dry_summary.json" if dry else "summary.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"结果已保存：{out}")


if __name__ == "__main__":
    main()

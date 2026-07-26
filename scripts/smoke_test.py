"""端到端冒烟测试：验证完整 LangGraph 链路可以跑通。

两种模式：
- offline（默认）：真实加载 ChromaDB / BM25 / docstore / Reranker，只把 LLM
  换成固定回复的替身，不产生任何 API 调用与费用。验证图流转、双路检索、
  正文补齐、精排和输出校验的集成正确性。
- --live：使用 .env 中的 DeepSeek key 真实调用 LLM（会产生费用）。

运行：
    python scripts/smoke_test.py
    python scripts/smoke_test.py --live
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

# 允许直接执行 `python scripts/smoke_test.py` 时从项目根目录导入 src 和 config。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_ISSUE = (
    "Firefox crashes with MOZ_CRASH when opening a PDF file in a background tab. "
    "Stack shows mozilla::dom::PDFViewer::Init. Started after updating to version 103."
)


class FakeSmokeLLM:
    """按调用次序返回 Query Analysis 与 Decision 回复的 LLM 替身。"""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list) -> types.SimpleNamespace:
        self.calls += 1
        if self.calls == 1:
            # 第一次调用是 Query Analysis：返回改写结果与关键词。
            content = json.dumps(
                {
                    "rewritten_query": "Firefox crash MOZ_CRASH PDF background tab PDFViewer",
                    "keywords": ["MOZ_CRASH", "PDFViewer", "crash", "PDF"],
                    "component": None,
                }
            )
        else:
            # 第二次调用是 Decision：高置信度 new，保证单轮结束、不触发重试。
            content = json.dumps(
                {
                    "decision": "new",
                    "confidence": 0.9,
                    "related_issues": [],
                    "reasoning": "冒烟测试固定回复：验证链路，不代表真实判断",
                }
            )
        return types.SimpleNamespace(content=content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="真实调用 DeepSeek（会产生费用）")
    args = parser.parse_args()

    from src.agent import nodes
    from src.agent.graph import get_graph

    started = time.perf_counter()
    graph = get_graph()
    load_seconds = time.perf_counter() - started
    print(f"[1/3] 依赖与图加载完成：{load_seconds:.1f}s")

    if not args.live:
        # 离线模式：依赖初始化后再替换 LLM，检索/重排仍走真实索引与模型。
        nodes._llm = FakeSmokeLLM()
        print("[2/3] 离线模式：LLM 已替换为固定回复替身（无 API 调用）")
    else:
        print("[2/3] live 模式：将真实调用 DeepSeek")

    started = time.perf_counter()
    final = graph.invoke(
        {"raw_issue": SMOKE_ISSUE, "retry_count": 0, "previous_decisions": []}
    )
    run_seconds = time.perf_counter() - started

    # 逐项断言链路产物，缺一个都说明集成有问题。
    assert final.get("rewritten_query"), "Query Analysis 未产出 rewritten_query"
    assert final.get("retrieved_docs"), "Retrieval 未召回任何候选"
    assert final.get("reranked_docs"), "Rerank 未输出候选"
    assert final.get("decision") in ("duplicate", "similar", "new"), "Decision 类别非法"
    assert 0.0 <= final.get("confidence", -1) <= 1.0, "confidence 超出 [0,1]"

    hydrated = sum(
        1
        for doc in final["retrieved_docs"]
        if str(doc.get("title", "")).strip() or str(doc.get("body", "")).strip()
    )
    print(
        f"[3/3] 链路跑通：{run_seconds:.1f}s | 召回 {len(final['retrieved_docs'])} 条"
        f"（有正文证据 {hydrated} 条）| 精排 {len(final['reranked_docs'])} 条"
        f" | decision={final['decision']} confidence={final['confidence']}"
    )
    print("SMOKE OK")


if __name__ == "__main__":
    main()

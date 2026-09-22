"""Issue RAG Agent 命令行入口。

用法：
    python main.py "Application crashes with NullPointerException when clicking login"
    python main.py --json "..."     # 输出机器可读 JSON
    echo "issue text" | python main.py    # 从 stdin 读取

首次调用会加载 Embedding、Reranker 模型和 ChromaDB/BM25 索引，耗时较长属正常现象。
"""

import argparse
import json
import sys


def main() -> None:
    """输入命令行参数中的 issue 文本，输出 duplicate/similar/new 判断结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue_text", nargs="?", help="新 issue 的原始文本；缺省时从 stdin 读取")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出完整结果")
    args = parser.parse_args()

    issue_text = args.issue_text or sys.stdin.read()
    if not issue_text.strip():
        parser.error("issue 文本为空：请通过参数或 stdin 提供内容")

    # 延迟导入：--help 和参数错误时不加载模型与索引。
    from src.agent.graph import run_agent

    result = run_agent(issue_text.strip())

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"decision   : {result['decision']}")
    print(f"confidence : {result['confidence']:.2f}")
    print(f"related    : {', '.join(result['related_issues']) or '（无）'}")
    print(f"reasoning  : {result['reasoning']}")
    print(f"rounds     : {result['retry_count']}")


if __name__ == "__main__":
    main()

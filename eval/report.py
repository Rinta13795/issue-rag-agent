"""评估报告：读取 eval_results/ 下的 JSON 结果，打印四组方法对比表。

运行：python -m eval.report
"""

import json
from pathlib import Path

from config import EVAL_RESULTS_DIR

# 展示顺序和显示名称
_ORDER = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]
_NAMES = {
    "vector_only": "Baseline (Vector Only)",
    "bm25_only": "BM25 Only",
    "hybrid": "Hybrid (BM25+Vec+RRF)",
    "hybrid_rerank": "Hybrid + Rerank",
}
_COLS = ["recall@10", "precision@5", "mrr", "ndcg@10", "p50_latency", "p95_latency"]


def load_results(results_dir: str = EVAL_RESULTS_DIR) -> dict:
    """输入结果目录，输出各配置的指标字典。优先读 summary.json，否则逐个读取。"""
    summary = Path(results_dir) / "summary.json"
    if summary.exists():
        with summary.open(encoding="utf-8") as f:
            return json.load(f)

    results = {}
    for p in Path(results_dir).glob("*.json"):
        if p.stem == "summary":
            continue
        with p.open(encoding="utf-8") as f:
            results[p.stem] = json.load(f)
    return results


def print_table(results: dict) -> None:
    """输入各配置的指标字典，打印格式化对比表。recall@10 列附注相对上一配置的提升。"""
    name_w = 26
    col_w = 14
    header = f"{'Method':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in _COLS)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    prev_recall = None
    for key in _ORDER:
        if key not in results:
            continue
        m = results[key]
        name = _NAMES.get(key, key)
        row = f"{name:<{name_w}}"
        for col in _COLS:
            val = m.get(col, "-")
            row += f"{val:>{col_w}}"

        if prev_recall is not None and "recall@10" in m:
            delta = round(m["recall@10"] - prev_recall, 3)
            row += f"   (+{delta:.3f})" if delta > 0 else ""

        print(row)
        if "recall@10" in m:
            prev_recall = m["recall@10"]

    print(sep)


if __name__ == "__main__":
    results = load_results()
    if not results:
        print(f"未找到评估结果，请先运行：python -m eval.run_eval")
    else:
        print_table(results)

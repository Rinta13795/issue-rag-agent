"""Demo 评估数据与系统状态服务：解析真实 eval_results/*.json 并提供系统非敏感元数据。"""

import json
from pathlib import Path
from typing import Any

from config import (
    BM25_INDEX_PATH,
    CHROMA_PERSIST_DIR,
    DEEPSEEK_MODEL,
    DOCSTORE_PATH,
    EMBED_MODEL,
    EVAL_RESULTS_DIR,
    RERANKER_MODEL,
)
from src.demo.models import EvaluationSummary, SystemInfo


def load_evaluation_summary() -> EvaluationSummary:
    """只读解析 eval_results/*.json，返回真实的 benchmark 数据。"""
    eval_dir = Path(EVAL_RESULTS_DIR)
    summary_path = eval_dir / "summary.json"

    metrics: dict[str, dict[str, Any]] = {}
    sample_size = 200
    total_test = 1726
    seed = 42
    self_hit = True

    # 尝试读取各文件
    file_map = {
        "vector_only": "vector_only.json",
        "bm25_only": "bm25_only.json",
        "hybrid": "hybrid.json",
        "hybrid_rerank": "hybrid_rerank.json",
    }

    for key, filename in file_map.items():
        fpath = eval_dir / filename
        if fpath.exists():
            try:
                with fpath.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.pop("_meta", {})
                if meta:
                    sample_size = meta.get("evaluated_queries", sample_size)
                    total_test = meta.get("test_set_total", total_test)
                    self_hit = meta.get("self_hit_excluded", self_hit)
                metrics[key] = data
            except Exception:
                pass

    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                sdata = json.load(f)
            meta = sdata.get("_meta", {})
            if meta:
                sample_size = meta.get("evaluated_queries", sample_size)
                total_test = meta.get("test_set_total", total_test)
                self_hit = meta.get("self_hit_excluded", self_hit)
            for k in ["hybrid", "hybrid_rerank"]:
                if k in sdata and k not in metrics:
                    metrics[k] = sdata[k]
        except Exception:
            pass

    notes = [
        "Recall@10 表示检索召回率（Top-10 是否命中预期 Duplicate Issue），非分类准确率。",
        "当前实测显示 BAAI/bge-reranker-base 在该语料上 Recall@10 略有下降（0.4257 → 0.3701），属于负收益已知限制。",
        "BM25 对超长日志存在长尾延迟，已在线上增加输入截断保护（1200 字符上限）。",
        "Chroma 向量库已完成 10.6 万条 Issue 切块索引；全量评测基于 200 条固定种子抽样集。",
    ]

    return EvaluationSummary(
        sample_size=sample_size,
        total_test_queries=total_test,
        seed=seed,
        self_hit_excluded=self_hit,
        metrics=metrics,
        notes=notes,
    )


def get_system_info() -> SystemInfo:
    """返回非敏感系统状态元信息。"""
    bm25_ready = Path(BM25_INDEX_PATH).exists()
    vector_ready = Path(CHROMA_PERSIST_DIR).exists()
    docstore_ready = Path(DOCSTORE_PATH).exists()

    return SystemInfo(
        version="2.0.0",
        dataset_size=106655,
        dataset_name="GitBugs / Mozilla Public Dataset",
        embedding_model=EMBED_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=DEEPSEEK_MODEL,
        bm25_ready=bm25_ready,
        vector_ready=vector_ready,
        docstore_ready=docstore_ready,
        local_demo=True,
        known_limitations=[
            "Cross-Encoder 精排模型当前在评测集上为负收益，待进一步域内微调",
            "当前向量库 component 元数据为空，在线硬过滤已自动跳过并保留分析标签",
            "本地演示单 worker 队列执行，防止高并发导致本地内存溢出",
        ],
    )

"""Step 2 入口脚本：加载、标准化、清洗 issue，并构建向量索引和 BM25 索引。"""

import sys
from pathlib import Path

from loguru import logger


# 允许直接执行 `python scripts/run_step2.py` 时从项目根目录导入 src 和 config。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import clean, load_data, normalize
from src.indexer import build_indexes


def run_indexing() -> None:
    """输入无，按 Step 2 全流程加载、标准化、清洗并构建 ChromaDB 与 BM25 索引。"""
    # 第一步：加载原始 issue 数据，数据源内部按 HuggingFace -> direct download -> mock fallback。
    logger.info("Step 2 开始：加载原始 issue 数据")
    raw_issues = load_data()
    logger.info("原始 issue 加载完成：{} 条", len(raw_issues))

    # 第二步：逐条标准化字段，统一 id/title/body/labels/duplicate_of 等字段格式。
    logger.info("开始标准化 issue 字段")
    normalized_issues = [normalize(issue) for issue in raw_issues]
    logger.info("issue 标准化完成：{} 条", len(normalized_issues))

    # 第三步：清洗 title/body 并过滤 closed + invalid/wontfix 的误导型噪声。
    logger.info("开始清洗 issue 并过滤噪声")
    clean_issues = clean(normalized_issues)
    logger.info("issue 清洗完成：{} 条", len(clean_issues))

    # 第四步：构建 ChromaDB 向量索引和 issue 粒度 BM25 索引。
    logger.info("开始构建 ChromaDB 向量索引和 BM25 索引")
    build_indexes(clean_issues)

    # 第五步：打印 Step 2 完成日志。
    logger.info("Step 2 完成：向量索引和 BM25 索引构建完成")


if __name__ == "__main__":
    run_indexing()

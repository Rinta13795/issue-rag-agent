"""一次性构建评估资产：docstore（候选正文补齐）+ Golden Set（可信评测集）。

docstore 和 golden set 都依赖同一条 load -> normalize -> clean 预处理链路，
分开执行会把最耗时的清洗（BeautifulSoup 处理 10 万条 body）重复跑两遍，
因此合并成一个入口，清洗一次、产出两份资产。

运行：python scripts/build_eval_assets.py
"""

import json
import sys
from pathlib import Path

from loguru import logger

# 允许直接执行 `python scripts/build_eval_assets.py` 时从项目根目录导入 src 和 config。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.golden_set_builder import build_golden_set, load_indexed_ids, save_golden_set
from src.data_loader import clean, load_data, normalize
from src.docstore import build_docstore


def main() -> None:
    """输入无，按共享预处理链路依次构建 docstore 和 golden set。"""
    logger.info("加载并预处理 issue 数据（与建索引同一条链路）")
    issues = clean([normalize(item) for item in load_data()])
    logger.info("预处理完成：{} 条 issue", len(issues))

    logger.info("构建 docstore")
    build_docstore(issues)

    logger.info("构建 golden set")
    indexed_ids = load_indexed_ids()
    eval_set, test_set, stats = build_golden_set(issues, indexed_ids)
    save_golden_set(eval_set, test_set, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

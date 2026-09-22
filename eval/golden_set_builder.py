"""Golden Set 构建：从 GitBugs 中提取 duplicate query 和 ground truth，构造可信评测集。

相对 v1 修复的四个有效性问题：
1. 数据先 normalize 再 clean（v1 直接 clean(load_data())，字段格式不统一）；
2. 过滤不可达 ground truth：duplicate_of 指向的 issue 必须存在于当前索引，
   否则检索永远不可能命中，会把数据问题算成检索失败；
3. 过滤自引用：duplicate_of 包含 query 自身 id 的脏数据；
4. duplicate family 隔离划分：通过 duplicate_of 边把互相关联的 issue 合并成
   family（并查集），同一 family 只会整体进入 eval set 或 test set，避免
   “调参时见过 A→B，测试时又来 B→A”这类隐性泄漏；划分同时按 project 分层。

运行：python -m eval.golden_set_builder
"""

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

from loguru import logger

from config import BM25_INDEX_PATH, EVAL_GOLDEN_PATH, EVAL_SEED
from src.data_loader import clean, load_data, normalize


def load_indexed_ids(bm25_path: str = BM25_INDEX_PATH) -> set[str]:
    """输入 bm25.pkl 路径，输出当前索引中全部 issue id 集合。

    BM25 和 ChromaDB 由同一批 clean issue 构建，用 BM25 的 id 列表代表
    “检索系统可达的 issue 全集”，加载成本远低于扫描 Chroma。
    """
    with Path(bm25_path).open("rb") as file:
        data = pickle.load(file)
    return {str(issue_id) for issue_id in data["ids"]}


def _find_root(parents: dict[str, str], node: str) -> str:
    """并查集查找：输入节点 id，输出所在 family 的根节点（带路径压缩）。"""
    root = node
    while parents[root] != root:
        root = parents[root]
    # 路径压缩：把沿途节点直接挂到根上，摊平后续查找成本。
    while parents[node] != root:
        parents[node], node = root, parents[node]
    return root


def _union(parents: dict[str, str], a: str, b: str) -> None:
    """并查集合并：把 a、b 所在的两个 family 合并为一个。"""
    root_a, root_b = _find_root(parents, a), _find_root(parents, b)
    if root_a != root_b:
        parents[root_b] = root_a


def build_golden_set(
    issues: list[dict],
    indexed_ids: set[str],
    seed: int = EVAL_SEED,
) -> tuple[list, list, dict]:
    """输入清洗后的 issue 列表和索引 id 集合，输出 eval set、test set 和构建统计。

    筛选条件：resolution=duplicate 且 duplicate_of 非空；
    每条 query 的 ground truth 只保留“非自身且存在于索引”的目标。
    """
    stats = {
        "input_issues": len(issues),
        "raw_duplicate_queries": 0,
        "dropped_no_reachable_target": 0,
        "dropped_self_reference_targets": 0,
        "dropped_unreachable_targets": 0,
    }

    # 第一步：筛选 duplicate query，并逐条清理 ground truth。
    golden: list[dict] = []
    seen_query_ids: set[str] = set()
    for issue in issues:
        if not issue.get("duplicate_of"):
            continue
        if str(issue.get("resolution", "")).lower() != "duplicate":
            continue

        query_id = str(issue["id"])
        # 同一 id 重复出现时只保留第一条，避免评测集中出现重复 query。
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        stats["raw_duplicate_queries"] += 1

        targets = []
        for target in issue["duplicate_of"]:
            target_id = str(target)
            if target_id == query_id:
                # 自引用是数据噪声：自己不可能是自己的 duplicate 目标。
                stats["dropped_self_reference_targets"] += 1
                continue
            if target_id not in indexed_ids:
                # 目标不在索引里，检索永远不可能命中，保留只会低估检索指标。
                stats["dropped_unreachable_targets"] += 1
                continue
            targets.append(target_id)

        if not targets:
            stats["dropped_no_reachable_target"] += 1
            continue

        golden.append(
            {
                "id": query_id,
                "text": f"{issue['title']}\n{issue['body']}",
                "duplicate_of": targets,
                "project": issue.get("project", ""),
            }
        )

    stats["usable_queries"] = len(golden)
    logger.info(
        "golden query 构建：{} 条 duplicate 记录 -> {} 条可评测 query（无可达目标 {} 条）",
        stats["raw_duplicate_queries"],
        len(golden),
        stats["dropped_no_reachable_target"],
    )

    # 第二步：用并查集把 query 与其 ground truth 连成 duplicate family。
    # 同一 family 的 query 只能整体进入一侧，防止 eval/test 之间互相泄漏。
    parents: dict[str, str] = {}
    for query in golden:
        for node in [query["id"], *query["duplicate_of"]]:
            parents.setdefault(node, node)
    for query in golden:
        for target in query["duplicate_of"]:
            _union(parents, query["id"], target)

    family_queries: dict[str, list[dict]] = defaultdict(list)
    for query in golden:
        family_queries[_find_root(parents, query["id"])].append(query)

    stats["family_count"] = len(family_queries)

    # 第三步：按 project 分层、以 family 为最小单位做 80/20 划分。
    # family 的 project 取其中多数 query 的 project；随机顺序由固定 seed 决定。
    rng = random.Random(seed)
    families_by_project: dict[str, list[list[dict]]] = defaultdict(list)
    for members in family_queries.values():
        project_votes = defaultdict(int)
        for query in members:
            project_votes[query["project"]] += 1
        majority_project = max(project_votes, key=project_votes.get)
        families_by_project[majority_project].append(members)

    eval_set: list[dict] = []
    test_set: list[dict] = []
    for project in sorted(families_by_project):
        families = families_by_project[project]
        rng.shuffle(families)
        total_queries = sum(len(members) for members in families)
        target_eval = total_queries * 0.8

        # 贪心装箱：按打乱后的顺序把 family 放入 eval set，直到达到 80% 配额。
        placed_eval = 0
        for members in families:
            if placed_eval + len(members) <= target_eval or placed_eval == 0:
                eval_set.extend(members)
                placed_eval += len(members)
            else:
                test_set.extend(members)

    rng.shuffle(eval_set)
    rng.shuffle(test_set)

    stats["eval_queries"] = len(eval_set)
    stats["test_queries"] = len(test_set)
    logger.info(
        "family 隔离划分完成：{} 个 family -> eval {} 条 / test {} 条",
        stats["family_count"],
        len(eval_set),
        len(test_set),
    )
    return eval_set, test_set, stats


def save_golden_set(
    eval_set: list,
    test_set: list,
    stats: dict,
    path: str = EVAL_GOLDEN_PATH,
) -> None:
    """输入 eval set、test set 和构建统计，保存为 JSON 文件。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "build_stats": {**stats, "seed": EVAL_SEED, "split_unit": "duplicate_family"},
        "eval_set": eval_set,
        "test_set": test_set,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Golden set 已保存：{}", out_path)


def load_golden_set(path: str = EVAL_GOLDEN_PATH) -> tuple[list, list]:
    """输入 JSON 文件路径，输出 eval set 和 held-out test set。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Golden set 加载完成：eval={} 条，test={} 条",
                len(data["eval_set"]), len(data["test_set"]))
    return data["eval_set"], data["test_set"]


if __name__ == "__main__":
    # 与建索引完全一致的预处理链路：load -> normalize -> clean（v1 缺 normalize）。
    issues = clean([normalize(item) for item in load_data()])
    indexed_ids = load_indexed_ids()
    eval_set, test_set, stats = build_golden_set(issues, indexed_ids)
    save_golden_set(eval_set, test_set, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

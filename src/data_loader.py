"""加载、标准化并清洗 Issue RAG Agent 的原始 issue 数据。

输入来自 HuggingFace、direct download 或 mock 文件的原始 issue；输出标准字段格式的 issue 列表。
本文件不负责切 chunk、不负责向量化、不负责写入 ChromaDB 或构建 BM25 索引。
"""

import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from bs4 import BeautifulSoup
from loguru import logger

#
def _get_first(raw_issue: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    """输入原始 issue 和候选字段名，输出第一个存在且不为 None 的字段值。"""
    # 按候选字段名顺序查找，兼容不同数据源的字段命名差异。
    for key in keys:
        if key in raw_issue and raw_issue[key] is not None:
            return raw_issue[key]
    # 如果候选字段都不存在，就返回调用方给定的默认值。
    return default


def _to_str_list(value: Any) -> list[str]:
    """输入任意标签或重复 issue 字段值，输出统一的字符串列表。——————解决字段不统一问题"""
    # None 表示没有值，统一转成空列表。
    if value is None:
        return []
    # list/tuple/set 已经是集合类型，只需要过滤空值并转成字符串。
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    # 字符串可能是 "bug,ui" 这种逗号分隔格式，统一拆开。
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    # 其他类型通常是单个 id 或标签，转成单元素字符串列表。
    return [str(value).strip()] if str(value).strip() else []



def normalize(raw_issue: dict[str, Any]) -> dict[str, Any]:
    """输入单条原始 issue，输出项目统一使用的标准 issue 字段格式。"""
    # 兼容 issue_id/id 等不同命名，把 issue 唯一标识统一成 id。
    issue_id = _get_first(raw_issue, ["id", "issue_id", "issueId"], "")

    # 兼容 title/name/summary 等标题字段，缺失时给空字符串。
    title = _get_first(raw_issue, ["title", "name", "summary"], "")

    # 兼容 body/description/content/text 等正文字段，缺失时给空字符串。
    body = _get_first(raw_issue, ["body", "description", "content", "text"], "")

    # labels 可能是字符串或列表，统一转成 list[str]。——————需解决2个问题，还要解决数值形态不同，其他都字符串不用改
    labels = _to_str_list(_get_first(raw_issue, ["labels", "label"], []))

    # component 缺失时保留 None，后续检索时再决定是否作为 metadata filter。
    component = _get_first(raw_issue, ["component", "module", "area"], None)

    # status 缺失时默认 open，避免后续 clean 阶段误判为空状态。
    status = _get_first(raw_issue, ["status", "state"], "open")

    # resolution 缺失时保留 None，clean 阶段只处理明确的 invalid/wontfix。
    resolution = _get_first(raw_issue, ["resolution", "resolve_status"], None)

    # duplicate_of 可能是 None、单个字符串或列表，统一转成 list[str]。
    duplicate_of = _to_str_list(
        _get_first(raw_issue, ["duplicate_of", "duplicate_of_id", "dup_of_id"], [])
    )

    # project 缺失时给空字符串，保持标准字段存在。
    project = _get_first(raw_issue, ["project", "repo", "repository"], "")

    # created_at 缺失时给空字符串，后续模块不需要猜测时间。
    created_at = _get_first(raw_issue, ["created_at", "createdAt", "created"], "")
    
    #前面逻辑先把所有数值取出来，后面统一issue格式
    # 返回 CLAUDE.md 约定的标准 issue 格式，不在 normalize 阶段做清洗或过滤。
    return {
        "id": str(issue_id),
        "title": str(title),
        "body": str(body),
        "labels": labels,
        "component": str(component) if component is not None else None,
        "status": str(status),
        "resolution": str(resolution) if resolution is not None else None,
        "duplicate_of": duplicate_of,
        "project": str(project),
        "created_at": str(created_at),
    }


def clean(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """输入 normalize 后的 issue 列表，输出去 HTML、解码实体、压缩空白并过滤无效 issue 后的列表。"""
    cleaned_issues: list[dict[str, Any]] = []
    #记录过滤条数
    skipped_count = 0

    for issue in issues:
        #先过滤再清洗
        # 先做 issue 级过滤：closed 且 resolution 是 invalid/wontfix 的历史 issue 会误导 Decision。
        status = str(issue.get("status", "open")).lower()#————字段缺失先使用默认，不直接过滤，还没人处理可能。
        resolution = str(issue.get("resolution", "")).lower()
        if status == "closed" and resolution in ("invalid", "wontfix"):
            skipped_count += 1
            continue

        # 复制一份 issue，避免调用方传入的原始列表被原地修改。
        cleaned_issue = issue.copy()

        for field in ("title", "body"):
            # title/body 缺失时按空字符串处理，保持标准字段存在。
            raw_text = str(cleaned_issue.get(field, ""))#保证不报错，不存在返回“”

            # 第一步：用 BeautifulSoup 去掉 HTML 标签，只保留可读文本。
            text_without_html = BeautifulSoup(raw_text, "html.parser").get_text(" ")

            # 第二步：解码 HTML 实体，例如 &amp; -> &，&lt; -> <。
            unescaped_text = html.unescape(text_without_html)

            # 第三步：把连续空格、换行、制表符压缩成一个空格，并去掉首尾空白。
            cleaned_issue[field] = re.sub(r"\s+", " ", unescaped_text).strip()
            r"""
              ┌───────────────────────────────┬────────────────┐
  │            用什么             │     做什么     │
  ├───────────────────────────────┼────────────────┤
  │ BeautifulSoup(...).get_text() │ 去 HTML 标签   │
  ├───────────────────────────────┼────────────────┤
  │ html.unescape()               │ 解码 HTML 实体 │
  ├───────────────────────────────┼────────────────┤
  │ re.sub(r"\s+", " ", ...)      │ 压缩空白       │
  └───────────────────────────────┴────────────────┘
"""

        cleaned_issues.append(cleaned_issue)

    logger.info("issue 清洗完成：输入 {} 条，过滤 {} 条，输出 {} 条", len(issues), skipped_count, len(cleaned_issues))
    return cleaned_issues

#返回issue结构化字段
def load_data(
    dataset_name: str = "gitbugs/dedup-2025",
    split: str = "train",
    direct_url: str | None = None,
    mock_path: str = "data/mock/mock_issues.json",
     #list[dict[str, Any]]：函数返回值是"一个列表，列表里每项是字典"
) -> list[dict[str, Any]]:
    """输入数据源配置，按 HuggingFace datasets -> direct download -> mock 文件顺序加载原始 issue 列表。"""
    # 第一级 fallback：优先使用 datasets.load_dataset 读取 GitBugs 数据集。
    try:
        from datasets import load_dataset
        logger.info("尝试从 HuggingFace datasets 加载数据集：{} ({})", dataset_name, split)
        #拿数据
        dataset = load_dataset(dataset_name, split=split)
        issues = [dict(item) for item in dataset]
        logger.info("HuggingFace datasets 加载成功，共 {} 条 issue", len(issues))
        return issues
    except Exception as exc:
        logger.warning("HuggingFace datasets 加载失败，准备尝试 direct download：{}", exc)

    # 第二级 fallback：如果调用方提供了 direct_url，则尝试直接下载 JSON 或 JSONL 数据。
    if direct_url:
        try:
            #解析数据，转化数据
            logger.info("尝试从 direct download 加载数据：{}", direct_url)
            with urlopen(direct_url) as response:
                raw_text = response.read().decode("utf-8")

            # direct download 可能返回 JSON 数组：[{},{}]格式，也可能返回 JSONL：全{}{}格式；先按 JSON 数组解析。
            try:
                #直接load进来
                parsed = json.loads(raw_text)

                issues = parsed if isinstance(parsed, list) else list(parsed.values())
            except json.JSONDecodeError:
                #数据重新整合进[]，拆成一行一行处理比如['','']
                issues = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

            logger.info("direct download 加载成功，共 {} 条 issue", len(issues))
            return [dict(item) for item in issues]
        except Exception as exc:
            logger.warning("direct download 加载失败，准备使用 mock 数据：{}", exc)
    else:
        logger.warning("未提供 direct_url，跳过 direct download，准备使用 mock 数据")

    # 第三级 fallback：最后读取本地 mock_issues.json，保证离线也能继续开发。(准备好的虚假数据，但涵盖所有可能数据)
    mock_file = Path(mock_path)
    logger.info("尝试从 mock 文件加载数据：{}", mock_file)
    #with ... as file: 是上下文管理器
    with mock_file.open("r", encoding="utf-8") as file:
        #文件内容读取出来，类型[dict],就是正常的返回格式
        issues = json.load(file)

    logger.info("mock 数据加载成功，共 {} 条 issue", len(issues))
    return [dict(item) for item in issues]

'''
  ┌─────────────────────────────────┬───────────────────────────────┐
  │              情况               │           用哪一级            │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ 在公司有网，能连 HuggingFace    │ 第一级，拿真实 15 万条        │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ 网不稳定 / 国内环境             │ 第二级，直接下载              │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ 完全离线 / 想快速测试 / CI 环境 │ 第三级，用你手写的 50 条 mock │
  └─────────────────────────────────┴───────────────────────────────┘

'''

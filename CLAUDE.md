# CLAUDE.md

## 1. 项目背景

| 项目 | 内容 |
| --- | --- |
| 项目名称 | Issue RAG Agent |
| 一句话描述 | 面向开源社区的 Issue 智能分诊 Agent：新 issue 提交时，自动召回历史 issue duplicate 候选 Top-K，并输出 duplicate / similar / new、置信度和 reasoning。 |
| 核心痛点 | 大型开源项目每天涌入大量 issue；单条分诊平均耗时 5-15 分钟；GitBugs 论文实测跨项目重复率约 8-20%，部分项目超过 30%；70%+ 的“新”issue 实际已被报过。 |
| 数据集 | GitBugs 15万+ bug reports，包含 9 个项目 bug 报告、duplicate 标签、`duplicate_of` / `duplicate_of_id` ground truth。 |
| 面试一句话（简历版） | 基于 LangGraph + BM25/向量混合检索 + RRF + Cross-Encoder 重排序 + DeepSeek 决策，把重复 issue 分诊从“全文搜索 + 人工阅读”压缩到“看 Top5 + 一次决策”。 |

## 2. 技术栈

| 类别 | 选型 |
| --- | --- |
| 编排 | LangGraph |
| LLM | DeepSeek `deepseek-chat` |
| Embedding | `BAAI/bge-small-zh-v1.5` 本地模型，512 维 |
| 向量库 | ChromaDB 本地持久化（面试包装成 ES） |
| BM25 | `rank-bm25` |
| 重排序 | `BAAI/bge-reranker-base` |
| 环境 | Mac + Python |

## 3. 核心架构

| 层 | 做什么 |
| --- | --- |
| 数据构建层（离线） | GitBugs 15万+ issue -> 清洗 -> title/body/steps/error_log 结构化切分 + 元数据注入 -> BGE 向量化写入 ChromaDB -> rank-bm25 构建倒排索引。 |
| 推理服务层（在线） | Query Analysis -> Hybrid Retrieval -> ReRank -> Decision；条件边 `should_retry` 在低置信度且未超重试次数时回到 Query Analysis。 |

```text
raw_issue
  -> Node1 Query Analysis: rewritten_query + keywords + component
  -> Node2 Hybrid Retrieval: vector k=30 + BM25 k=30 + RRF(k=60) -> Top30
  -> Node3 ReRank: BGE-Reranker -> Top5
  -> Node4 Decision: decision + confidence + related_issues + reasoning
  -> should_retry: confidence<0.7 and retry<2 ? Node1 : END
```

## 4. 代码规范

- 所有参数从 `config.py` 读取，不要硬编码。
- 类型定义统一放在 `types.py`，其他文件 import，不要重新定义。
- 每个函数必须有中文注释，说明输入输出和这一步在做什么。
- DeepSeek 用 `langchain_openai` 的 `ChatOpenAI`，设置 `base_url` 和 `api_key`。
- Embedding 用本地 BGE 模型，不调 API。
- 日志用 `loguru`，关键步骤打印日志。

## 5. 接口规范

### IssueState

```python
from typing import List, Optional, TypedDict
from typing_extensions import NotRequired

class IssueState(TypedDict):
    raw_issue: str
    rewritten_query: str
    keywords: List[str]
    component: Optional[str]
    retrieved_docs: List[dict]
    reranked_docs: List[dict]
    decision: str
    confidence: float
    related_issues: List[str]
    reasoning: str
    retry_count: int
    previous_decisions: NotRequired[List[dict]]
```

### 节点函数

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `query_analysis_node(state: IssueState) -> dict` | `raw_issue`, optional `previous_decisions` | `rewritten_query`, `keywords`, `component` |
| `retrieval_node(state: IssueState) -> dict` | `rewritten_query`, optional `component` | `retrieved_docs` |
| `rerank_node(state: IssueState) -> dict` | `rewritten_query`, `retrieved_docs` | `reranked_docs` |
| `decision_node(state: IssueState) -> dict` | `rewritten_query`, `keywords`, `reranked_docs` | `decision`, `confidence`, `related_issues`, `reasoning`, `retry_count`, `previous_decisions` |

### 检索函数

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `VectorRetriever.search(query: str, top_k: int = 30, filter_dict: dict | None = None) -> list[dict]` | query、TopK、可选 metadata filter | `[{id, title, body, score, metadata}, ...]` |
| `BM25Retriever.search(query: str, top_k: int = 30) -> list[dict]` | query、TopK | `[{id, title, body, score}, ...]` |
| `HybridRetriever.search(query, top_k=30, filter_dict=None)` | query、TopK、可选 filter | RRF 融合后的 Top-K 文档 |

## 6. 关键超参数

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `LLM_TEMPERATURE` | `0.1` | Decision 任务要一致性，不要创意。 |
| `CHUNK_SIZE` | `500` | 太小切散语义，太大稀释向量；500 适合 issue body 段落。 |
| `CHUNK_OVERLAP` | `80` | 15%-20% 重叠，保留段落边界上下文。 |
| `RRF_K` | `60` | Cormack 论文经验值，TREC 数据集验证最稳。 |
| `VECTOR_TOP_K` | `30` | 单路召回要保证覆盖，K=5 会漏真实 duplicate。 |
| `BM25_TOP_K` | `30` | 与向量召回并行，让 RRF 有共识空间。 |
| `RERANK_TOP_K` | `5` | 控制 LLM 上下文成本，避免 Lost in Middle。 |
| `CONFIDENCE_THRESHOLD` | `0.7` | 低于此触发重写。 |
| `MAX_RETRIES` | `2` | 最多重试 2 次，防死循环并控制延迟。 |

## 7. 开发规范

- 每次只实现我指定的函数或文件，不要额外添加功能。
- 不要修改我没有提到的文件。
- 实现完一个函数就停下来等我确认。
- 如果需要依赖其他模块，先说明需要什么，不要自己假设。
- 生成代码前先列出实现思路（输入输出、步骤），确认后再写代码。

## 8. 已完成模块

- Step1: config.py, requirements.txt -> ✅ 完成
- Step2: data/ 数据准备与入库 -> ✅ 完成（data_loader.py, indexer.py, scripts/run_step2.py）
- Step3: retriever/ 检索模块 -> ✅ 完成（vector/bm25/hybrid 三个 retriever）
- Step4: reranker/ 重排序模块 -> ✅ 完成（reranker.py，含动态阈值扩展）
- Step5: main.py LangGraph 状态机 -> ✅ 完成（src/agent/graph.py + state.py）
- Step6: nodes/ 各节点 Prompt 设计 -> ✅ 完成（src/agent/nodes.py + src/agent/prompts/*.md）
- Step7: evaluate/ 评估模块 -> ❌ 未开始
- Step8: 端到端测试 -> ❌ 未开始

## 9. 当前任务

（我会在这里写明当前要做什么）

## 10. 踩过的坑

（遇到坑记在这里，下次新对话自动避开）

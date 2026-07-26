# CLAUDE.md

## 1. 项目背景

| 项目 | 内容 |
| --- | --- |
| 项目名称 | Issue RAG Agent v2 |
| 一句话描述 | Issue 重复检测 RAG 系统：新 issue 提交时召回历史 duplicate 候选 Top-K，输出 duplicate / similar / new、置信度和 reasoning；LangGraph 负责固定四节点流程和低置信度回环。 |
| 与 v1 的关系 | 从 `../RAG项目` 复制而来，原项目保持不动；全部改动见 `IMPROVEMENTS.md`。 |
| 数据集 | GitBugs bug reports 本地快照 106,714 条（8 个项目），含 `duplicate_of` ground truth；数据与索引不随 Git 分发。 |

## 2. 技术栈（以代码事实为准）

| 类别 | 选型 |
| --- | --- |
| 编排 | LangGraph `StateGraph` |
| LLM | DeepSeek `deepseek-chat`，经 `langchain_openai.ChatOpenAI`（timeout=60s，max_retries=2） |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` 本地模型，384 维，CPU |
| 向量库 | ChromaDB 本地持久化 `./chroma_db_en`，collection `issues`，chunk 粒度 |
| BM25 | `rank-bm25` BM25Okapi，issue 粒度，`bm25.pkl` |
| 正文补齐 | `docstore.pkl`：issue_id -> {title, body}，为 BM25-only 候选补证据 |
| 重排序 | `BAAI/bge-reranker-base` Cross-Encoder，CPU |
| 服务入口 | `main.py`（CLI）、`api.py`（FastAPI） |
| 环境 | Mac + Python 3 |

## 3. 核心架构

```text
raw_issue
  -> Node1 Query Analysis: rewritten_query + keywords + component
  -> Node2 Retrieval: [rewritten_query, raw_issue] 双路
       每路 vector k=30 + BM25(query+keywords) k=30 + RRF(k=60)
       两路再做外层 RRF -> Top30 -> docstore 补齐空正文候选
  -> Node3 ReRank: BGE-Reranker -> Top5（分差近时最多扩到 Top10）
  -> Node4 Decision: decision + confidence + related_issues + reasoning
       程序级校验：枚举 / confidence 截断 [0,1] / related_issues 候选白名单
  -> should_retry: confidence<0.7 and retry_count<=2 ? Node1 : END
       重试 prompt 携带上轮决策、检索诊断和 Top-3 候选标题
```

## 4. 代码规范

- 所有参数从 `config.py` 读取，不要硬编码。
- LangGraph 共享状态类型定义在 `src/agent/state.py`（`IssueState`），其他类型就近定义。
- 每个函数必须有中文注释，说明输入输出和这一步在做什么。
- DeepSeek 用 `langchain_openai` 的 `ChatOpenAI`，设置 `base_url` 和 `api_key`。
- Embedding 用本地模型，不调 API。
- 日志用 `loguru`，关键步骤打印日志。
- 改动检索/评估逻辑时必须先跑 `python3 -m pytest tests/ -q`（当前 32 个测试）。

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
| `query_analysis_node(state) -> dict` | `raw_issue`, optional `previous_decisions` | `rewritten_query`, `keywords`, `component` |
| `retrieval_node(state) -> dict` | `rewritten_query`, `raw_issue`, `keywords`, optional `component` | `retrieved_docs` |
| `rerank_node(state) -> dict` | `rewritten_query`, `retrieved_docs` | `reranked_docs` |
| `decision_node(state) -> dict` | `rewritten_query`, `keywords`, `reranked_docs` | `decision`, `confidence`, `related_issues`, `reasoning`, `retry_count`, `previous_decisions` |

### 检索函数

| 函数 | 说明 |
| --- | --- |
| `VectorRetriever.search(query, top_k=30, filter_dict=None)` | chunk 召回聚合到 issue 粒度，返回 `[{id, title, body, score, metadata}]` |
| `BM25Retriever.search(query, top_k=30, extra_terms=None)` | keywords 经 extra_terms 追加进查询 token；过滤 0 分候选 |
| `HybridRetriever.search(query, top_k=30, filter_dict=None, bm25_extra_terms=None)` | 单 query 两路 RRF 融合 + docstore 补齐 |
| `HybridRetriever.search_queries(queries, ...)` | 多 query 外层 RRF 融合（在线链路用 `[rewritten, raw]`） |

## 6. 关键超参数

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `LLM_TEMPERATURE` | `0.1` | Decision 任务要一致性，不要创意。 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | 适合 issue body 段落，15% 重叠保留边界上下文。 |
| `RRF_K` | `60` | Cormack 论文经验值。 |
| `VECTOR_TOP_K` / `BM25_TOP_K` | `30` | 单路召回保证覆盖，让 RRF 有共识空间。 |
| `RERANK_TOP_K` | `5` | 控制 LLM 上下文成本；分差 < 0.05 时最多扩到 10。 |
| `DECISION_BODY_PREVIEW_CHARS` | `400` | docstore 补齐后可以给 Decision 更多证据。 |
| `CONFIDENCE_THRESHOLD` / `MAX_RETRIES` | `0.7` / `2` | 低置信度重写；防死循环并控制延迟。 |
| `DOCSTORE_BODY_MAX_CHARS` | `1200` | 覆盖 Reranker 800 字符与 Decision 400 字符的需求。 |

## 7. 评估约定

- Golden Set 由 `scripts/build_eval_assets.py` 生成：ground truth 必须可达
  （存在于索引）、无自引用；duplicate family 级 80/20 划分（并查集 + project 分层）。
- `eval/run_eval.py` 评估时剔除 query 自命中；Rerank 配置只算 @5/@10。
- 调参只允许用 eval set；test set 只出最终指标。
- 抽样评估必须固定 seed 并在结果 JSON 的 `_meta` 中注明样本量。

## 8. 模块状态

- 数据构建：`data_loader.py` + `indexer.py` + `scripts/run_step2.py` ✅
- 检索：vector / bm25 / hybrid + docstore 补齐 ✅
- 重排序：`reranker.py`（动态阈值扩展，无副作用） ✅
- Agent：`graph.py`（图缓存）+ `nodes.py`（输出校验） ✅
- 评估：`eval/`（golden set 修复 + 自命中剔除 + nDCG 修复） ✅
- 服务入口：`main.py` CLI + `api.py` FastAPI ✅
- 测试：`tests/` 32 个用例 ✅
- Benchmark：`eval/benchmarks/` 框架可用，等待人工 verified 案例 ⏳

## 9. 踩过的坑

- `duplicate_of` 大量指向索引外 issue：评估前必须做可达性过滤，否则指标被低估约 1/3。
- 评测 query 自身在索引里：不剔除自命中会污染全部排名指标。
- v1 nDCG 的 IDCG 用已检索 gains 计算：命中 1 条就满分，属于实现错误。
- Reranker 原地写 `rerank_score` 会污染 State：任何给候选打分的函数都先 copy。
- `.env.example` 一旦提交过真实 key，历史里就永远有：示例文件只放占位符。

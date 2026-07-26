# Issue RAG Agent v2

面向开源项目历史 Issue 的重复问题候选检索与辅助分诊系统。输入一段新 Issue 文本，
系统召回并重排历史候选，由 LLM 输出 `duplicate`、`similar` 或 `new` 判断、置信度、
相关 Issue ID 和理由。

> 准确定位：**Issue Duplicate Detection RAG System + LangGraph 条件工作流**。
> LangGraph 负责固定四节点流程和低置信度回环；没有工具选择或自主计划能力，
> 因此本文不称其为高度自主 Agent。

本项目由 v1（`../RAG项目`）复制改进而来，原项目未做修改。
**全部改进的动机与实测影响见 [IMPROVEMENTS.md](IMPROVEMENTS.md)**，
代码阅读顺序见 [docs/code-reading-guide.md](docs/code-reading-guide.md)。

## v2 相对 v1 的核心变化

| 维度 | v1 | v2 |
| --- | --- | --- |
| 评估有效性 | Golden Set 含 34% 不可达 ground truth；查询自命中不剔除；nDCG IDCG 实现错误；query 级随机划分有泄漏 | 可达性过滤 + 自命中剔除 + IDCG 修复 + duplicate family 级隔离划分 |
| BM25-only 候选 | 无 title/body，Reranker/Decision 拿不到证据 | `docstore.pkl` 融合后自动补齐正文 |
| keywords | 只在 Decision prompt 展示 | 作为补充 token 参与 BM25 检索 |
| LLM 输出 | 只做类型兜底 | 枚举校验 + confidence 截断 + related_issues 候选白名单 + 无证据降级 |
| 重试上下文 | 只有上轮 query/决策/计数 | 增加上轮 Top-3 候选 id/title/分数 |
| 服务入口 | 仅 `run_agent` 函数 | `main.py` CLI + `api.py` FastAPI |
| 测试 | 2 个文件 | 6 个文件、32 个用例 |
| 安全 | `.env.example` 含真实 API key | 占位符；真实 key 移入 gitignored `.env` |

## 整体数据流

```text
离线（scripts/run_step2.py + scripts/build_eval_assets.py）
  load_data -> normalize -> clean
    -> title/error_log/body chunk -> Embedding -> ChromaDB（chunk 粒度）
    -> title+body tokenize -> BM25Okapi -> bm25.pkl（issue 粒度）
    -> docstore.pkl（issue_id -> title/body，补证据用）
    -> golden_set.json（可达性过滤 + family 隔离划分）

在线（run_agent(issue_text)）
  Query Analysis（DeepSeek）: rewritten_query + keywords + component
  Retrieval: [rewritten_query, raw_issue] 双路
      每路 Vector Top30 + BM25(query+keywords) Top30 + RRF
      两路外层 RRF -> Top30 -> docstore 补齐空正文候选
  Rerank（bge-reranker-base）: Top5，分差 < 0.05 时最多扩到 Top10
  Decision（DeepSeek）: decision/confidence/related_issues/reasoning
      程序级校验：枚举、confidence∈[0,1]、related_issues 候选白名单
  should_retry: confidence < 0.7 且 retry_count <= 2 时回到 Query Analysis
      重试 prompt 携带上轮决策、检索诊断和 Top-3 候选
```

## 技术栈

| 层 | 实现 |
| --- | --- |
| 工作流 | LangGraph `StateGraph` + `IssueState`（编译结果进程内缓存） |
| LLM | DeepSeek `deepseek-chat`，`langchain_openai.ChatOpenAI`（timeout=60s, max_retries=2） |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`，本地 CPU，384 维 |
| 向量存储 | ChromaDB `./chroma_db_en`，collection `issues`，chunk 粒度 |
| 关键词检索 | `rank-bm25` BM25Okapi + jieba 分词，`bm25.pkl`，0 分候选过滤 |
| 正文补齐 | `docstore.pkl`：106,655 条 issue_id -> {title, body≤1200 字符} |
| 排序融合 | 双层 Reciprocal Rank Fusion，`RRF_K=60` |
| 重排序 | `BAAI/bge-reranker-base` Cross-Encoder，CPU |
| 服务 | FastAPI（`POST /triage`）+ CLI（`main.py`） |
| 日志 | Loguru |

## 快速开始

### 1. 环境

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # 填入自己的 DEEPSEEK_API_KEY
```

### 2. 构建离线资产（首次）

```bash
python -m scripts.run_step2            # ChromaDB + bm25.pkl（全量向量化，耗时长）
python scripts/build_eval_assets.py    # docstore.pkl + golden_set.json（约 3 分钟）
```

数据与索引不随 Git 分发；本地数据快照为 106,714 条 Issue、8 个项目。

### 3. 调用

```bash
# CLI
python main.py "Application crashes with NullPointerException when clicking login"
python main.py --json "..."

# HTTP 服务
uvicorn api:app --port 8000
curl -X POST localhost:8000/triage -H 'Content-Type: application/json' \
     -d '{"issue_text": "Firefox crashes when opening a PDF in a background tab"}'

# Python
python -c "from src.agent.graph import run_agent; print(run_agent('...'))"
```

返回字段：

```python
{
    "decision": "duplicate | similar | new",
    "confidence": float,          # LLM 自报值，已截断到 [0,1]，未做统计校准
    "related_issues": list[str],  # 已经过候选白名单过滤，不含幻觉 ID
    "reasoning": str,
    "retry_count": int,           # Decision 节点累计执行轮数：正常 1，最多 3
}
```

### 4. 评估

```bash
python scripts/build_eval_assets.py       # 重建 golden set（含构建统计）
python -m eval.run_eval --limit 200       # 固定 seed 抽样快速对比
python -m eval.run_eval                   # 全量 held-out test set
python -m eval.report                     # 打印四配置对比表
python -m pytest tests/ -q                # 32 个单元测试
```

## 评估方法

指标可信的前提是评测集干净，v2 在这里做了四件事（详见 IMPROVEMENTS.md 第一节）：

1. **可达性过滤**：ground truth 必须存在于索引；原始 13,104 条 duplicate 记录中
   4,505 条（34%）没有任何可达目标，v1 把它们全部计为检索失败。
2. **自命中剔除**：评测 query 本身在索引里，检索 top_k+1 后剔除自身再截断。
3. **family 隔离划分**：并查集把 query 与 ground truth 连成 duplicate family，
   同一 family 整体进入 eval 或 test（3,333 family -> eval 6,873 / test 1,726），
   并按 project 分层。
4. **公平截断点**：Rerank 输出最多 10 条，只算 @5/@10，不与 30 条配置比 recall@30。

### 检索指标（held-out test set 抽样 200 条，seed=42，自命中已剔除）

<!-- EVAL_RESULTS_TABLE -->

- Vector/BM25/Hybrid 在 @5/@10/@30 截断点计算；Hybrid+Rerank 只在 @5/@10。
- 延迟为单机 CPU（Mac）实测，含 Embedding 编码与 BM25 全量打分，不含 LLM 调用。
- 全量 test set（1,726 条）与端到端分类指标的运行命令已提供，可自行复现。

## 目录结构

```text
.
├── config.py                  # 全部超参数与路径
├── main.py                    # CLI 入口
├── api.py                     # FastAPI 服务
├── scripts/
│   ├── run_step2.py           # 离线建库（ChromaDB + BM25）
│   └── build_eval_assets.py   # docstore + golden set（一次预处理，两份产出）
├── src/
│   ├── data_loader.py         # load -> normalize -> clean
│   ├── indexer.py             # chunk / embedding / 建库（新建 collection 用 cosine）
│   ├── docstore.py            # issue_id -> {title, body} 正文补齐
│   ├── reranker.py            # Cross-Encoder 精排（无副作用）
│   ├── retrievers/            # vector / bm25 / hybrid（双层 RRF + 补齐）
│   └── agent/                 # state / graph / nodes / prompts
├── eval/
│   ├── golden_set_builder.py  # 可达性过滤 + family 隔离划分
│   ├── metrics.py             # Recall/Precision/MRR/nDCG（IDCG 已修复）
│   ├── run_eval.py            # 四配置评估（自命中剔除、可控抽样）
│   ├── report.py              # 对比表
│   └── benchmarks/            # retry / 双路召回 benchmark（待人工 verified 案例）
├── tests/                     # 32 个单元测试
└── IMPROVEMENTS.md            # v1 -> v2 全部改进及动机
```

## 已知限制（诚实清单）

- 在线接口只接收 Issue 文本，不能按 repository / 创建时间限定候选；
  `created_at` 不在 Chroma metadata 中，支持时间过滤需要重建索引。
- confidence 是 LLM 自报值，只做区间截断，未做统计校准。
- duplicate/similar/new 三分类没有人工标注测试集，仅检索层有定量指标。
- 现有 `chroma_db_en` 建于 Chroma 默认 l2 空间；因只做排序不影响结果，
  重建索引后将切换为显式 cosine。
- 向量路候选的 body 是最相关 chunk 而非完整正文（这是有意设计：对 Reranker
  是更好的证据），BM25-only 候选的 body 是正文前 1200 字符。
- 两个 benchmark（retry_quality / dual_query_retrieval）框架可运行，
  但需要人工确认的 verified 案例，本仓库不编造案例数据。
- 数据、索引与 docstore 不随 Git 分发；干净 clone 需先完成离线建库。

## License

仓库当前没有提交 LICENSE 文件，因此不声明具体开源许可证。

# v1 → v2 改进清单

本文档记录从原项目（`../RAG项目`）复制出本版本后所做的全部改进：每一项先说
v1 的问题，再说 v2 的修法。原项目未做任何修改。

## 一、评估有效性（最高优先级：v1 的指标不可信）

### 1. Golden Set 含大量“不可能答对”的 query

- **v1 问题**：`duplicate_of` 指向的目标 issue 有相当一部分不在索引里
  （被清洗过滤、或数据集本身缺失）。检索永远不可能命中它们，却被计为失败。
- **v2 修复**：构建 Golden Set 时用 BM25 索引的 id 全集校验每个 ground truth，
  剔除不可达目标与自引用；全部目标不可达的 query 整条丢弃。
- **实测影响**：13,104 条 duplicate 记录中 4,505 条（34%）没有任何可达目标，
  另有 6,114 个目标 id 被剔除。v1 的全部检索指标因此被系统性低估约三分之一。

### 2. 评测 query 自身就在索引里（自命中污染）

- **v1 问题**：query issue 本身也被索引，检索结果第一名往往是它自己，
  挤占候选位置、扭曲排名指标。
- **v2 修复**：`run_eval.py` 检索 `top_k + 1` 条，剔除 query 自身 id 后再截断。

### 3. eval/test 划分存在泄漏

- **v1 问题**：随机 query 级 80/20 划分。同一 duplicate family 的 A→B 进 eval、
  B→A 进 test，调参阶段实际见过测试样本的近亲。
- **v2 修复**：用并查集把 query 与 ground truth 连成 duplicate family，
  以 family 为最小划分单位、按 project 分层做 80/20。
  新划分：3,333 个 family → eval 6,873 条 / test 1,726 条。

### 4. nDCG 的 IDCG 算错

- **v1 问题**：`metrics.py` 用“已检索到的 gains 重排”当理想序，只要命中 1 条
  nDCG 就是满分，漏检目标不计入分母。
- **v2 修复**：IDCG 按 `min(len(golden), k)` 个理想命中计算，并补了针对该
  bug 的回归测试（`tests/test_metrics.py`）。

### 5. 不同方法在不公平的截断点上比较

- **v1 问题**：Hybrid+Rerank 最多返回 10 条，却计算 Recall@30，和返回 30 条的
  方法直接比较。
- **v2 修复**：`compute_all` 支持自定义 cutoffs；Rerank 配置只算 @5/@10。

### 6. Golden Set 构建缺 normalize

- **v1 问题**：`clean(load_data())` 跳过了 `normalize`，字段格式和建索引链路不一致。
- **v2 修复**：统一为 `clean([normalize(x) for x in load_data()])`，
  与 `scripts/run_step2.py` 建索引链路完全一致。

## 二、检索质量

### 7. BM25-only 候选没有正文证据（v1 README 承认的最大限制）

- **v1 问题**：bm25.pkl 只存 id；只被 BM25 召回的候选 title/body 为空，
  Reranker 对它打分、Decision 对它判断时都拿不到任何文本。
- **v2 修复**：新增 `src/docstore.py` + `docstore.pkl`（106,655 条
  issue_id → {title, body}，body 截断 1200 字符，72MB）。`HybridRetriever`
  融合后自动补齐空正文候选；docstore 缺失时自动退回 v1 行为。

### 8. keywords 从不参与检索

- **v1 问题**：Query Analysis 提取的 keywords 只在 Decision prompt 里展示。
- **v2 修复**：keywords 作为补充 token 传给 BM25（等效提升错误码、API 名的
  词频权重），经 `HybridRetriever → BM25Retriever` 全链路透传。

### 9. BM25 返回 0 分候选

- **v1 问题**：query 词一个都没命中的 issue 也进入 Top-30，浪费 RRF 配额。
- **v2 修复**：BM25 结果过滤 `score <= 0`。

### 10. Chroma 距离空间未显式配置

- **v1 问题**：collection 用默认 l2，`VectorRetriever` 却按 `1 - distance`
  转分数（只在 cosine 下语义成立）；因为只做排序所以结果没错，但分数含义是错的。
- **v2 修复**：`indexer.py` 新建 collection 时显式 `hnsw:space: cosine`。
  注意：只对重建后的索引生效，现有 `chroma_db_en` 保持不变（排序不受影响）。

## 三、Agent 决策鲁棒性

### 11. LLM 输出没有程序级校验

- **v1 问题**：decision 枚举、confidence 范围、related_issues 是否真实存在
  都不验证，幻觉 ID 直接返回给调用方。
- **v2 修复**（`decision_node`，配套 `tests/test_decision_validation.py`）：
  - decision 不在 `{duplicate, similar, new}` → 按解析失败降级为低置信度 new；
  - confidence 截断到 [0, 1]；
  - related_issues 按本轮候选 id 白名单过滤，幻觉 ID 记日志剔除；
  - duplicate/similar 却没有任何合法关联 ID 时降级为 new（confidence ≤ 0.4，
    在次数允许时自动触发重试）。

### 12. 重试时看不到上一轮检索到了什么

- **v1 问题**：重试 prompt 只有上轮 query/决策/计数统计，模型无法判断
  该“换方向”还是“加细节”。
- **v2 修复**：`previous_decisions` 记录上轮 Top-3 候选的 id/title/分数，
  重试 prompt 新增“上一轮 Top 候选”区块（`query_analysis_retry.md` v3）。

### 13. Decision 证据窗口过小

- **v1 问题**：每个候选 body 只展示 200 字符。
- **v2 修复**：提升到 400 字符并收入 config（`DECISION_BODY_PREVIEW_CHARS`）；
  配合 docstore 补齐，Decision 面对的候选证据明显更完整。

### 14. LLM 客户端无超时与重试

- **v2 修复**：`ChatOpenAI(timeout=60, max_retries=2)`，网络异常不再无限阻塞。

## 四、工程质量

### 15. `.env.example` 泄漏真实 API Key（安全问题）

- **v1 问题**：`.env.example` 中是真实格式的 DeepSeek key，且已随 git 提交。
- **v2 修复**：示例文件改为占位符；真实 key 移入 gitignored 的 `.env`。
  **提醒：旧 key 已进 git 历史，请到 DeepSeek 平台作废并更换。**

### 16. Reranker 原地修改调用方数据

- **v1 问题**：`rerank()` 直接往传入的 dict 写 `rerank_score`，污染 State 中的
  `retrieved_docs`，benchmark 必须自行 copy 防御。
- **v2 修复**：先浅拷贝再打分。

### 17. 每次 run_agent 重新构图

- **v2 修复**：`get_graph()` 缓存编译后的 LangGraph，重依赖与图对象都只初始化一次。

### 18. 缺少服务化入口

- **v2 新增**：`main.py`（CLI，支持参数/stdin、`--json`）和 `api.py`
  （FastAPI，`GET /health` + `POST /triage`，请求响应模型校验）。

### 19. 测试覆盖

- **v1**：2 个测试文件（混合检索融合、benchmark 指标）。
- **v2**：新增 metrics 回归、decision 校验、docstore 补齐、golden set
  划分共 4 个测试文件；全套 32 个测试通过。

### 20. gitignore 与仓库卫生

- 补充忽略 `chroma_db_en/`、`docstore.pkl`、`eval/data/`、`eval_results/`；
  仓库以干净历史重新初始化（原项目保持原样）。

## 尚未解决的已知限制（诚实清单）

- 在线接口仍不支持按 repository / 创建时间限定候选（`created_at` 不在
  Chroma metadata 中，需要重建索引才能支持时间过滤）。
- confidence 仍是 LLM 自报值，未做统计校准。
- duplicate/similar/new 三分类没有人工标注测试集，分类质量只能间接评估。
- 现有 `chroma_db_en` 仍是 l2 空间（不影响排序，重建后才切 cosine）。
- 两个 benchmark（retry_quality、dual_query_retrieval）仍需人工 verified
  案例才能运行，本次未编造案例。

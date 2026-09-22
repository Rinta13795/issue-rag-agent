# Issue RAG Agent 代码阅读地图

这份文档用于指导你亲自沿真实调用关系阅读项目。它不是审计报告，也不替你给出所有结论。建议每完成一条路径，就把“建议产出”保存到自己的学习笔记，再进入下一条。

> **v2 阅读提示**：本文写于 v1，阅读路径依然成立，但以下 v1 结论在 v2 中已经改变，
> 对照 [IMPROVEMENTS.md](../IMPROVEMENTS.md) 阅读：
>
> 1. “已提交基线 vs 工作区实验”的区分不再存在——v2 中 `eval/` 等全部已提交。
> 2. “BM25-only 候选没有 title/body” → 已由 `src/docstore.py` 融合后补齐。
> 3. “keywords 不参与检索” → 已作为 extra_terms 传给 BM25。
> 4. “Decision 输出无程序级校验” → 已有枚举/confidence 截断/related_issues 白名单。
> 5. “重试只读 query/decision/confidence” → 重试 prompt 已携带上轮 Top-3 候选。
> 6. “Golden Set 未过滤不可达目标、随机 query 级划分、nDCG IDCG 有误” → 均已修复。
> 7. 新增在线入口：`main.py`（CLI）与 `api.py`（FastAPI）；`run_agent` 的图已缓存。

## 使用说明

当前仓库有两类事实，阅读时必须分开：

**已提交基线**：`HEAD` 中的 `config.py`、`src/`、`scripts/run_step2.py` 等，是 README 描述正式能力的依据。

- **工作区实验**：当前未提交的 `config.py` / `src/indexer.py` 修改，以及 `eval/`、`chroma_db_en/`、`eval_results/` 等文件。可以阅读，但不能当作已提交能力。

推荐顺序不是目录顺序，而是：

```text
事实边界
→ 在线入口与 LangGraph 骨架
→ State 和节点契约
→ Query Analysis / DeepSeek
→ Hybrid Retrieval 分叉
   ├─ Vector → 追到 Chroma chunk 来源
   └─ BM25 → 追到 BM25 语料来源
→ RRF 候选对象
→ Reranker / Decision
→ retry 与失败路径
→ 离线数据和完整建库流程
→ 未提交评估实验
→ 配置、依赖、路径和复现边界
```

阅读过程中先回答三个固定问题：

1. 当前对象是什么形状（字符串、State、Document、候选 dict、索引对象）？
2. 当前函数从哪里取得它，又把它交给谁？
3. 这一步丢失、覆盖或新增了哪些信息？

---



## 路径 0：先建立仓库事实边界



### 阅读目标

分清“项目说明”“已提交代码”“本地运行产物”和“未提交实验”，避免用设计文档代替实现。

### 文件位置

- `[README.md](../README.md)`
- `[config.py](../config.py)`
- `[requirements.txt](../requirements.txt)`
- `[.gitignore](../.gitignore)`
- `[CLAUDE.md](../CLAUDE.md)`
- `[step_notes/](../step_notes/)`



### 函数或类入口

这一条没有业务函数入口。先从 `config.py` 的常量和 `.gitignore` 的产物边界开始。

### 关键变量与状态字段

- `EMBED_MODEL`、`CHROMA_PERSIST_DIR`、`BM25_INDEX_PATH`
- `VECTOR_TOP_K`、`BM25_TOP_K`、`HYBRID_TOP_K`、`RERANK_TOP_K`
- `CONFIDENCE_THRESHOLD`、`MAX_RETRIES`
- 被忽略的 `data/`、`chroma_db/`、`bm25.pkl`



### 调用链或数据链

```text
README 的能力描述
→ config.py 的实际参数
→ requirements.txt 的声明依赖
→ .gitignore 判断哪些运行条件没有随仓库分发
→ CLAUDE.md / step_notes 只作为历史设计记录
```



### 阅读时需要回答的问题

1. README 中哪些内容能在代码或配置里找到唯一对应物？
2. 当前 `config.py` 相对 `HEAD` 有哪些未提交变化？
3. 哪些目录存在于本机，但干净 clone 不会得到？
4. `requirements.txt` 锁定了精确版本和 Python 版本吗？
5. `CLAUDE.md` 和 `step_notes` 中哪些句子是约定或计划，而不是执行逻辑？



### 必要提示

重点比较 Embedding 模型、Chroma 路径和评估常量。不要把注释中的性能数字当作实测结果。

### 建议产出

写一张两列表：左列“已提交基线”，右列“本地未提交实验”，每列只列能从 Git 状态和文件确认的事实。

---



## 路径 1：从 `run_agent` 进入在线主链



### 阅读目标

先获得在线流程的最小骨架：输入在哪里进入、图在哪里构建、最终哪些字段离开系统。

### 文件位置

- `[src/agent/graph.py](../src/agent/graph.py)`
- `[src/agent/state.py](../src/agent/state.py)`
- `[src/agent/__init__.py](../src/agent/__init__.py)`



### 函数或类入口

- `run_agent`
- `build_graph`
- `should_retry`
- `IssueState`
- `src.agent.__getattr__`



### 关键变量与状态字段

- 参数 `issue_text`
- `initial_state`
- `raw_issue`、`retry_count`、`previous_decisions`
- `final`
- 对外返回的 `decision`、`confidence`、`related_issues`、`reasoning`、`retry_count`



### 调用链或数据链

```text
run_agent(issue_text)
→ build_graph()
→ graph.invoke(initial_state)
→ query_analysis
→ retrieval
→ rerank
→ decision
→ should_retry
→ retry 或 END
→ result dict
```

追到 `graph.invoke` 后先停止，不要立刻跳进 Retriever；先把节点顺序和条件边画出来。

### 阅读时需要回答的问题

1. `run_agent` 能否知道输入属于哪个 repository？
2. `initial_state` 实际初始化了哪些字段？
3. TypedDict 声明与运行时渐进写入之间有什么差别？
4. 哪些 State 字段存在于图内，却没有通过 `run_agent` 返回？
5. `MAX_RETRIES=2` 时，最坏情况下四个节点各执行几次？
6. `src.agent.__getattr__` 为什么延迟导入 `run_agent`？



### 必要提示

计数器是在 `decision_node` 中增加，条件判断发生在它之后。画轮次时从 `retry_count=0` 开始逐轮手算。

### 建议产出

自己画一张不超过 8 个节点的控制流图，并写出 `run_agent` 的输入/输出契约。

---



## 路径 2：建立 IssueState 的字段生产者与消费者表



### 阅读目标

理解 LangGraph 在这个项目中如何通过共享 State 连接节点，以及重试时哪些值被覆盖、保留或追加。

### 文件位置

- `[src/agent/state.py](../src/agent/state.py)`
- `[src/agent/graph.py](../src/agent/graph.py)`
- `[src/agent/nodes.py](../src/agent/nodes.py)`



### 函数或类入口

- `IssueState`
- `query_analysis_node`
- `retrieval_node`
- `rerank_node`
- `decision_node`
- `should_retry`



### 关键变量与状态字段

- `raw_issue`
- `rewritten_query`、`keywords`、`component`
- `retrieved_docs`、`reranked_docs`
- `decision`、`confidence`、`related_issues`、`reasoning`
- `retry_count`、`previous_decisions`



### 调用链或数据链

```text
initial_state
→ query_analysis_node 的局部 dict
→ retrieval_node 的局部 dict
→ rerank_node 的局部 dict
→ decision_node 的局部 dict
→ LangGraph 合并到共享 State
```

每读一个节点，只记录 `state[...]` / `state.get(...)` 的读取字段，以及 `return {...}` 的写入字段。

### 阅读时需要回答的问题

1. 每个字段的唯一生产者是谁？哪些字段会被多轮覆盖？
2. 哪个字段只追加，不覆盖原列表？
3. `previous_decisions` 中每一项实际保存了哪些键？
4. `retrieved_docs` 与 `reranked_docs` 会不会进入历史？
5. 哪些访问使用 `state[...]`，哪些使用 `state.get(...)`？这对失败路径意味着什么？
6. `IssueState` 中 `NotRequired` 只标在了哪个字段上？



### 必要提示

不要根据 `state.py` 末尾的大段说明直接填答案；以四个节点真实返回的 dict 为准。

### 建议产出

完成一张四列表：`字段 | 初始化/生产位置 | 读取位置 | retry 行为`。

---



## 路径 3：阅读依赖初始化与 Query Analysis



### 阅读目标

理解重模型与外部客户端何时加载、Retriever/Reranker 如何注入节点，以及第一处 DeepSeek 调用如何生成检索输入。

### 文件位置

- `[src/agent/graph.py](../src/agent/graph.py)`
- `[src/agent/nodes.py](../src/agent/nodes.py)`
- `[src/agent/prompts/query_analysis_system.md](../src/agent/prompts/query_analysis_system.md)`
- `[src/agent/prompts/query_analysis_retry.md](../src/agent/prompts/query_analysis_retry.md)`
- `[config.py](../config.py)`



### 函数或类入口

- `_ensure_dependencies`
- `get_hybrid_retriever`
- `get_reranker`
- `configure_dependencies`
- `_load_prompt`
- `query_analysis_node`
- `_parse_json`



### 关键变量与状态字段

- `_HYBRID_RETRIEVER`、`_RERANKER`、`_DEPENDENCIES_READY`
- `_retriever`、`_reranker`、`_llm`
- `_PROMPTS`、`retry_block`
- `response.content`、`fallback`、`parsed`
- `rewritten_query`、`keywords`、`component`



### 调用链或数据链

```text
build_graph
→ _ensure_dependencies
→ get_hybrid_retriever / get_reranker
→ configure_dependencies
→ 初始化 ChatOpenAI

query_analysis_node
→ 读取 raw_issue 与最近一次 previous_decisions
→ 拼 SystemMessage / HumanMessage
→ _llm.invoke
→ _parse_json
→ 写入 rewritten_query / keywords / component
```



### 阅读时需要回答的问题

1. 单纯 import `run_agent` 是否立刻加载 Chroma 和 Cross-Encoder？

1. `_PROMPTS` 在什么时候读取 Markdown 文件？
2. YAML frontmatter 是否会发送给模型？
3. retry prompt 实际获得上一轮哪些字段？
4. Query Analysis 解析失败时如何降级？
5. `keywords` 后续是否真的作为 BM25 query 使用？
6. LLM 输出中哪些字段做了类型检查，哪些没有？



### 必要提示

沿 `rewritten_query` 和 `keywords` 分别向后搜索。两个字段在名字上都像检索输入，但真实消费者不同。

### 建议产出

写出 Query Analysis 的“输入消息模板 → 解析 → fallback → State 更新”四段式流程，并列出重试 prompt 实际缺少的上下文。

---



## 路径 4：从 Retrieval 节点进入两路检索与 RRF



### 阅读目标

先理解 Hybrid Retrieval 的总控逻辑，再分别深入 Vector 和 BM25，避免一开始陷入索引细节。

### 文件位置

- `[src/agent/nodes.py](../src/agent/nodes.py)`
- `[src/retrievers/hybrid_retriever.py](../src/retrievers/hybrid_retriever.py)`
- `[src/retrievers/vector_retriever.py](../src/retrievers/vector_retriever.py)`
- `[src/retrievers/bm25_retriever.py](../src/retrievers/bm25_retriever.py)`
- `[config.py](../config.py)`



### 函数或类入口

- `retrieval_node`
- `HybridRetriever.__init__`
- `HybridRetriever.search`
- `VectorRetriever.search`
- `BM25Retriever.search`



### 关键变量与状态字段

- `filter_dict`
- `query`、`top_k`
- `vector_results`、`bm25_results`
- `rrf_scores`、`issue_docs`、`ranked_ids`
- 候选 dict 中的 `id`、`title`、`body`、`metadata`、`score`



### 调用链或数据链

```text
retrieval_node
→ HybridRetriever.search(rewritten_query, top_k, filter_dict)
→ VectorRetriever.search(..., filter_dict)
→ BM25Retriever.search(...)
→ 按 id 累加两路 RRF 分
→ 统一候选 dict
→ retrieved_docs
```



### 阅读时需要回答的问题

1. `filter_dict` 由哪个 State 字段生成？空值时是什么？
2. 过滤条件传给了哪一路，为什么另一条调用没有该参数？
3. 两个 Retriever 是并行调用还是顺序调用？
4. RRF 公式中的 rank 从 0 还是 1 开始？
5. 同一 ID 在两路出现时分数如何累计？
6. 只在 BM25 出现的候选如何补齐统一结构？
7. RRF 使用原始向量/BM25 分数吗？



### 必要提示

把 `issue_docs` 看成“候选内容表”，把 `rrf_scores` 看成“候选排名表”。两者直到最终循环才合并。

### 建议产出

手算一个两路各 3 条结果的小型 RRF 例子，并画出候选 dict 在融合前后的字段变化。

---



## 路径 5：追踪 Vector Retrieval 与 Issue 粒度聚合



### 阅读目标

理解 Chroma 按 chunk 召回、代码如何聚合回 Issue，以及被选中的 chunk 如何影响下游证据。

### 文件位置

- `[src/retrievers/vector_retriever.py](../src/retrievers/vector_retriever.py)`
- `[src/agent/graph.py](../src/agent/graph.py)`
- `[src/indexer.py](../src/indexer.py)`
- `[config.py](../config.py)`



### 函数或类入口

- `get_hybrid_retriever`
- `VectorRetriever.__init__`
- `VectorRetriever.search`
- `load_embeddings`
- `chunk_issue`



### 关键变量与状态字段

- `vectorstore` / `self.vs`
- `filter_dict`
- `VECTOR_CHUNK_FETCH_MULTIPLIER`
- `chunk_results`
- `doc.page_content`、`doc.metadata`、`distance`
- `issue_best`、`issue_id`、`score`
- metadata 的 `title`、`component`、`project`、`chunk_type`



### 调用链或数据链

```text
get_hybrid_retriever
→ Chroma(collection, embedding_function, persist_directory)
→ VectorRetriever(vectorstore)
→ similarity_search_with_score
→ chunk_results
→ 以 issue_id 分组
→ 每组保留最高分 chunk
→ issue 级 Top-K

离线反向追踪：
chunk_issue
→ Document(page_content, metadata)
→ build_chroma_index
→ ChromaDB
```



### 阅读时需要回答的问题

1. 在线加载的 Embedding 与离线建库是否来自同一配置？
2. Top-30 Issue 前实际请求多少个 chunk？
3. 同一 Issue 的多个 chunk 用 max、sum 还是平均方式聚合？
4. 返回候选的 `body` 是完整正文还是代表 chunk？
5. 如果最佳 chunk 是 title chunk，Decision 最终会看到什么？
6. `project` 和 `created_at` 哪个进入了 Chroma metadata？
7. Chroma collection 在哪里显式指定距离空间？`1 - distance` 的解释依据是什么？



### 必要提示

把 `chunk_issue` 的三种 `chunk_type` 与 VectorRetriever 写入候选 `body` 的那一行连起来看。

### 建议产出

画一张“一条 Issue → 多个 chunk → 向量 Top-N → 单个代表 chunk”的数据形状图，并记录聚合中丢失的信息。

---



## 路径 6：追踪 BM25 建库与在线查询的一致性



### 阅读目标

把离线 BM25 corpus、tokenize 规则、pickle 内容和在线 Top-K 映射完整连起来。

### 文件位置

- `[src/indexer.py](../src/indexer.py)`
- `[src/retrievers/bm25_retriever.py](../src/retrievers/bm25_retriever.py)`
- `[src/retrievers/hybrid_retriever.py](../src/retrievers/hybrid_retriever.py)`
- `[config.py](../config.py)`



### 函数或类入口

- `tokenize`
- `build_bm25_index`
- `BM25Retriever.__init__`
- `BM25Retriever.search`
- `HybridRetriever.search`



### 关键变量与状态字段

- `bm25_corpus`
- `BM25Okapi`
- `bm25_ids`
- pickle 中的 `bm25` 与 `ids`
- `query_tokens`、`scores`、`top_indices`
- BM25 结果的 `id`、`score`



### 调用链或数据链

```text
clean issue 的 title + body
→ tokenize
→ BM25Okapi(corpus)
→ pickle {bm25, ids}
→ BM25Retriever.__init__
→ tokenize(query)
→ get_scores
→ 下标排序
→ id / score
→ HybridRetriever
```



### 阅读时需要回答的问题

1. 建库和查询是否复用同一个 tokenizer？
2. tokenizer 如何处理中文、英文大小写和空格？
3. `ids` 的顺序为什么必须与 corpus 保持一致？
4. 在线检索是倒排取候选，还是对 `get_scores` 的全量结果排序？
5. 是否过滤分数等于 0 的候选？
6. pickle 中有没有保存 title、body 或项目字段？
7. BM25-only 候选进入 Reranker 后，`title + body` 会是什么？



### 必要提示

不要只看 `BM25Retriever.search` 的返回注释；继续追到 HybridRetriever 为 BM25-only ID 补字段的位置。

### 建议产出

写出 BM25 索引文件的最小逻辑结构，并画出“分数数组下标 → Issue ID”的映射关系。

---



## 路径 7：从 Reranker 追到 Decision 的候选证据



### 阅读目标

理解混合候选如何变成 Cross-Encoder 输入、如何动态截断，以及 LLM 最终实际看到了多少候选信息。

### 文件位置

- `[src/agent/nodes.py](../src/agent/nodes.py)`
- `[src/reranker.py](../src/reranker.py)`
- `[src/agent/prompts/decision_system.md](../src/agent/prompts/decision_system.md)`
- `[config.py](../config.py)`



### 函数或类入口

- `rerank_node`
- `Reranker.__init__`
- `Reranker.rerank`
- `_format_candidates`
- `decision_node`



### 关键变量与状态字段

- `pairs`
- `RERANK_DOC_MAX_CHARS`
- `scores`、`rerank_score`
- `cut`、`max_cut`、`score_gap`
- `reranked_docs`
- `candidates_block`、`user_msg`



### 调用链或数据链

```text
retrieved_docs
→ (rewritten_query, title + body[:800]) pairs
→ CrossEncoder.predict(pairs)
→ 写入 rerank_score
→ Top-5 / 动态扩展至最多 Top-10
→ _format_candidates
→ 每条 body 再截到前 200 字符
→ Decision 的 HumanMessage
```



### 阅读时需要回答的问题

1. Reranker 是逐条还是批量调用 `predict`？
2. 它会不会原地修改 `retrieved_docs` 内的 dict？
3. 动态扩展比较的是哪两个位置的分数？
4. 哪个配置控制最多额外保留多少条？
5. 800 字符截断和 200 字符截断分别发生在哪一层？
6. BM25-only 空上下文候选在两次截断后还能提供什么证据？
7. Decision prompt 能看到 RRF 分数吗？能看到 rerank 分数吗？



### 必要提示

追踪同一个候选 dict 的 `score` 与 `rerank_score`；再检查 `_format_candidates` 实际格式化了哪一个。

### 建议产出

选择一个候选对象，写出它从 Hybrid 输出到 Decision prompt 的逐步字段和文本变化。

---



## 路径 8：专门阅读 Decision、JSON 防御与 retry 失败路径



### 阅读目标

区分“Prompt 中要求模型遵守的规则”和“Python 代码真正验证的规则”，并理解解析失败如何影响重试。

### 文件位置

- `[src/agent/nodes.py](../src/agent/nodes.py)`
- `[src/agent/prompts/decision_system.md](../src/agent/prompts/decision_system.md)`
- `[src/agent/prompts/query_analysis_retry.md](../src/agent/prompts/query_analysis_retry.md)`
- `[src/agent/graph.py](../src/agent/graph.py)`



### 函数或类入口

- `_parse_json`
- `_format_candidates`
- `decision_node`
- `query_analysis_node`
- `should_retry`



### 关键变量与状态字段

- `cleaned`、`match`、`fallback`
- `decision`、`confidence`、`related_issues`、`reasoning`
- `history`
- `previous_decisions`
- `CONFIDENCE_THRESHOLD`、`MAX_RETRIES`



### 调用链或数据链

```text
LLM response.content
→ 去代码块
→ 正则提取 JSON 对象
→ json.loads 或 fallback
→ 轻量字段处理
→ history.append
→ retry_count + 1
→ should_retry

解析失败：
fallback(new, 0.0, [], ...)
→ 低 confidence
→ 在次数允许时回到 Query Analysis
```



### 阅读时需要回答的问题

1. 正则提取 JSON 是严格 schema 解析吗？
2. `decision` 是否验证为三个允许值之一？
3. `confidence` 是否限制在 `[0, 1]`？
4. `related_issues` 是否与当前候选 ID 做程序级白名单比对？
5. reasoning 是否强制引用某个候选证据？
6. Prompt Injection 防护位于 prompt、解析器还是候选数据隔离层？
7. 首轮、第二轮和第三轮解析都失败时，最终结果是什么形状？
8. retry 是否读取上一轮 Top-K 候选？它如何知道“上轮失败原因”？



### 必要提示

把 `decision_system.md` 的“硬性约束”逐条在 `decision_node` 中寻找对应校验代码。找不到的约束只属于提示词约束。

### 建议产出

做一张两列表：`Prompt 要求 | Python 是否验证`；再手算一次连续三轮低置信度的 State 变化。

---



## 路径 9：从离线入口读取数据加载、标准化和清洗



### 阅读目标

理解索引数据从哪里来、何时标准化、清洗具体做了什么，以及 fallback 在干净环境中的依赖。

### 文件位置

- `[scripts/run_step2.py](../scripts/run_step2.py)`
- `[src/data_loader.py](../src/data_loader.py)`
- `[.gitignore](../.gitignore)`



### 函数或类入口

- `run_indexing`
- `load_data`
- `_get_first`
- `_to_str_list`
- `normalize`
- `clean`



### 关键变量与状态字段

- `dataset_name`、`split`、`direct_url`、`mock_path`
- `raw_issues`、`normalized_issues`、`clean_issues`
- 标准字段 `id/title/body/.../created_at`
- `status`、`resolution`
- `skipped_count`



### 调用链或数据链

```text
run_indexing
→ load_data
   → Hugging Face datasets
   → 可选 direct download
   → 本地 mock JSON
→ [normalize(issue) for issue in raw_issues]
→ clean(normalized_issues)
→ build_indexes(clean_issues)
```



### 阅读时需要回答的问题

1. 哪些异常会让加载器进入下一层 fallback？
2. `run_indexing` 有没有向 `load_data` 传 direct URL？
3. 三层数据源的结果是否都会经过 `normalize`？
4. `normalize` 对 `duplicate_of`、labels 和 component 分别如何处理？
5. `clean` 清理哪些字段？是否清理 labels 或 stack trace？
6. 哪些 status/resolution 组合会被过滤？
7. 是否存在“短正文过滤”或“空正文过滤”？
8. fallback 本地文件是否被 Git 跟踪？



### 必要提示

把 `scripts/run_step2.py` 看成真实编排者；`data_loader.py` 内部注释给出的 mock 数量不等于仓库实际随附数据。

### 建议产出

画一张三层 fallback 决策图，并用一条虚构原始记录手写 `normalize → clean` 前后字段变化。

---



## 路径 10：完整追踪 Chunk、ChromaDB 与 BM25 的持久化



### 阅读目标

把离线流程从 clean Issue 一直追到两个磁盘产物，并识别它们各自保存的数据粒度。

### 文件位置

- `[scripts/run_step2.py](../scripts/run_step2.py)`
- `[src/indexer.py](../src/indexer.py)`
- `[config.py](../config.py)`
- `[.gitignore](../.gitignore)`



### 函数或类入口

- `extract_error_log`
- `remove_error_log`
- `chunk_issue`
- `load_embeddings`
- `build_chroma_index`
- `tokenize`
- `build_bm25_index`
- `build_indexes`



### 关键变量与状态字段

- `ERROR_LOG_PATTERN`
- `base_meta`
- `chunks`、`all_chunks`
- `CHUNK_SIZE`、`CHUNK_OVERLAP`、`CHUNK_SEPARATORS`
- `INDEX_BATCH_SIZE`
- `vectorstore`
- `bm25_corpus`、`bm25_ids`
- `CHROMA_PERSIST_DIR`、`BM25_INDEX_PATH`



### 调用链或数据链

```text
build_indexes(clean_issues)
├─ build_chroma_index
│  ├─ chunk_issue
│  │  ├─ title chunk
│  │  ├─ optional error_log chunk
│  │  └─ recursive body chunks
│  ├─ load_embeddings
│  └─ Chroma.add_documents(batch)
└─ build_bm25_index
   ├─ title + body
   ├─ tokenize
   └─ pickle.dump({bm25, ids})
```



### 阅读时需要回答的问题

1. 错误正则找到第一个匹配后，error chunk 覆盖到哪里？
2. body chunk 如何避免与 error chunk 重复？
3. title 是否无条件单独成块？
4. 小于 30 字符的限制作用于整条 Issue 还是 body 子块？
5. Chroma metadata 中有哪些字段，缺少哪些未来过滤可能需要的字段？
6. Chroma 写入前是否清空已有 collection？是否显式指定 document ID？
7. BM25 与 Chroma 使用的是同一批 clean Issue 吗？
8. 自召回断言只验证了什么，没有验证什么？



### 必要提示

`build_indexes` 的调用顺序很重要：若 Chroma 已完成而 BM25 失败，磁盘可能处于部分完成状态。

### 建议产出

自己画出两个持久化产物的 schema：Chroma 画 `Document + metadata`，BM25 画 `BM25Okapi + ids`。

---



## ‘















## 路径 11：阅读未提交的 Golden Set 与评估链



### 阅读目标

理解当前本地评估实验打算测什么，并亲自判断数据切分、检索边界和指标实现是否支持可信结论。

> 这一路所有 `eval/` 文件当前均未提交。阅读它们是为了理解本地实验，不代表正式仓库已经具备评估能力。



### 文件位置

- `eval/golden_set_builder.py`（未提交）
- `eval/metrics.py`（未提交）
- `eval/run_eval.py`（未提交）
- `eval/report.py`（未提交）
- `eval/data/golden_set.json`（未提交且被忽略）
- `eval_results/`（未提交）
- `[src/data_loader.py](../src/data_loader.py)`
- `[src/retrievers/hybrid_retriever.py](../src/retrievers/hybrid_retriever.py)`



### 函数或类入口

- `build_golden_set`
- `save_golden_set`、`load_golden_set`
- `recall_at_k`、`precision_at_k`、`mrr`、`ndcg_at_k`、`compute_all`
- `evaluate_retriever`
- `evaluate_hybrid_rerank`
- `_aggregate_metrics`
- `load_results`、`print_table`



### 关键变量与状态字段

- `golden`、`eval_set`、`test_set`
- `id`、`text`、`duplicate_of`、`project`
- `retrieved_ids`、`golden_ids`
- `gains`、`dcg`、`ideal_gains`、`idcg`
- `all_metrics`、`latencies`
- `configs`



### 调用链或数据链

```text
load_data
→ clean
→ 筛选 duplicate query
→ 随机 80/20 split
→ golden_set.json
→ run_eval 加载 test_set
→ 四类 Retriever 配置
→ compute_all
→ 聚合均值与 P50/P95
→ eval_results/*.json
→ report.py
```



### 阅读时需要回答的问题

1. Golden Set 构建前是否调用 `normalize`？
2. 划分单位是项目、时间、duplicate family 还是单条 query？
3. query 本身是否仍存在于被检索的全量索引？
4. Retriever 是否收到 `project` filter 或时间边界？
5. 构建器是否检查每个 `duplicate_of` 一定存在于索引？
6. `ndcg_at_k` 的 IDCG 来自全部相关目标，还是当前检索到的 gains？
7. Hybrid+Rerank 最多返回多少条？它的 Recall@30 与其他路是否同口径？
8. 四路结果文件是否都已产生？
9. 这里测的是检索，还是最终 `duplicate/similar/new` 分类？
10. 80/20 中的 test 为什么叫 held-out？代码中是否真的存在训练或调参隔离？



### 必要提示

先读一条 Golden Set 记录的字段，再追 `project` 从 JSON 到 `evaluate_retriever` 是否仍被使用。检查 nDCG 时，用“有两个相关目标、只检回一个”的例子手算。

### 建议产出

完成三件事：一张评估数据流图、一张四路评估调用表、一份“当前结果为什么不能直接引用”的代码证据清单。

---



## 路径 12：检查配置、依赖、数据路径与可复现边界



### 阅读目标

在不运行高成本任务的前提下，判断一个新环境要成功执行离线和在线流程，必须额外具备哪些仓库外条件。

### 文件位置

- `[config.py](../config.py)`
- `[requirements.txt](../requirements.txt)`
- `[.env.example](../.env.example)`
- `[.gitignore](../.gitignore)`
- `[scripts/run_step2.py](../scripts/run_step2.py)`
- `[src/agent/graph.py](../src/agent/graph.py)`
- `[src/indexer.py](../src/indexer.py)`



### 函数或类入口

- `load_data`
- `load_embeddings`
- `build_indexes`
- `get_hybrid_retriever`
- `get_reranker`
- `configure_dependencies`
- `run_agent`



### 关键变量与状态字段

- `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`
- `EMBED_MODEL`、`RERANKER_MODEL`、`EMBED_DEVICE`
- `CHROMA_PERSIST_DIR`、`CHROMA_COLLECTION`、`BM25_INDEX_PATH`
- `HF_HUB_OFFLINE`
- requirements 中所有 `>=` / `<` 约束



### 调用链或数据链

```text
.env / 环境变量
→ config.py
→ ChatOpenAI / HuggingFaceEmbeddings / CrossEncoder

远端或本地数据
→ run_step2
→ ChromaDB + bm25.pkl
→ build_graph 运行时加载
→ run_agent
```



### 阅读时需要回答的问题

1. 仓库声明了 Python 版本吗？依赖是精确锁定还是范围约束？
2. 哪些模型可能在首次运行时联网下载？
3. 在线入口启动前必须已经存在什么磁盘产物？
4. 这些数据和索引是否由 Git 分发？
5. `.env.example` 只应承担什么作用？真实 Key 应放在哪里？
6. 缺少 Chroma、BM25 文件、API Key或模型缓存时，分别在哪个初始化点失败？
7. 仓库是否存在测试、CI、Docker、API 服务或前端入口？
8. `python -m scripts.run_step2` 与 `run_agent` 哪一步会产生外部网络或高成本计算？



### 必要提示

这一步不要实际跑全量索引或 LLM。只需沿文件打开、模型初始化和路径读取的位置建立“前置条件清单”。阅读 `.env.example` 时不要复制、展示或提交任何可能的凭据值。

### 建议产出

写一份“干净环境启动前置条件”清单，分成数据、索引、模型、凭据、Python 依赖五组；同时标记哪些条件在 Git 仓库内、哪些必须从外部获得。

---



## 建议的个人阅读记录模板

每完成一条路径，用下面的格式记录，不需要一开始写得很长：

```markdown
## 路径 N：标题

### 我确认的输入与输出

### 我画出的调用链 / 数据链

### 关键对象的字段变化

### 正常路径

### 失败路径

### README 是否与代码一致

### 我仍然不理解的问题
```



## 推荐起点

第一次阅读从 `[src/agent/graph.py](../src/agent/graph.py)` 的 `run_agent` 开始。先向下看到 `graph.invoke(initial_state)` 和结果裁剪，再回到同文件的 `build_graph`、`should_retry`；完成路径 1 的控制流图后，再打开 `nodes.py`。不要先从 `indexer.py` 或 Step Notes 开始。
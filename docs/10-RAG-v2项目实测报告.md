# RAG v2 项目实测报告

> 测试日期：2026-07-31  
> 代码版本：`main` / `49927e00a0c8`  
> 报告用途：后续代码阅读、项目复盘、面试陈述与继续评测  
> 安全说明：报告不记录 API 密钥或其他凭据。

## 1. 结论先行

RAG v2 已经是一个**能够真实运行、适合学习完整 RAG/Agent 主链的工程原型**：本地索引、混合检索、正文补齐、Cross-Encoder 精排、DeepSeek Query Analysis、Decision 和低置信度重试都能接通，32 项模块级测试全部通过。

但它目前还不能被描述为“检索效果已经优化完成”或“具备生产可用性”。本轮实测发现两个直接影响线上行为的问题：

1. **组件过滤与索引数据不兼容。** Chroma 中 392,142 个 chunk 的 `component` 全部为空；DeepSeek 在 5 条典型 Issue 中却有 4 条生成非空组件。只要触发该过滤，向量召回就会变成 0，系统只能依赖 BM25 兜底。
2. **BM25 存在严重长尾延迟。** 200 条结果中 Hybrid P95 为 528.387 秒；本轮小样本复测也在一条超长 crash 日志上超过 60 秒，调用栈停在 `rank_bm25.get_scores()`。

此外，当前 Reranker 并未带来收益：在同一批 200 条样本上，它让 Recall、MRR、nDCG 全部下降。因此最准确的项目定位是：

> **主链可运行、修复方向基本正确、评估体系比 v1 可靠，但数据过滤、长文本检索和精排策略仍需继续迭代。**

## 2. 测试对象与数据资产

### 2.1 运行环境

| 项目 | 实测值 |
| --- | --- |
| Python | 3.13.7 |
| LangChain | 1.2.12 |
| ChromaDB | 1.5.9 |
| PyTorch | 2.8.0 |
| Transformers | 4.56.2 |
| LLM | DeepSeek `deepseek-chat` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`，CPU |
| Reranker | `BAAI/bge-reranker-base`，CPU |

### 2.2 本地资产

| 资产 | 规模/用途 | SHA-256 |
| --- | --- | --- |
| `data/` | 约 161 MB，原始 Issue 数据 | — |
| `chroma_db_en/` | 约 1.8 GB，chunk 级向量索引；当前 collection metadata 为 `None` | — |
| `bm25.pkl` | 约 101 MB，Issue 级 BM25 | `13a09c9f5105800e...` |
| `docstore.pkl` | 约 69 MB，补齐候选标题和正文 | `6ada1136bb279a8e...` |
| `golden_set.json` | 约 13 MB，检索评估集 | `789fa58e306cb525...` |

Golden Set 从 106,656 条输入 Issue 中得到 8,599 条可用 duplicate query，按 duplicate family 隔离后划分为 6,873 条 eval 和 1,726 条 test。该设计避免同一家族同时进入训练侧和测试侧，也排除了不可达目标与自引用目标。

## 3. 本轮执行的测试

本报告区分四类证据：源码直接事实、自动测试、单次或少量运行记录、本地评估产物。运行记录只说明受测样本的行为；由这些事实推导的原因会明确保留为待验证判断。

| 层级 | 方法 | 结果 | 能证明什么 |
| --- | --- | --- | --- |
| 模块测试 | `pytest tests/ -q` | **32 passed / 3.97s** | 核心纯逻辑与部分模块边界正确 |
| 离线全链冒烟 | 真实索引、真实 Reranker、假 LLM | **通过 / 6.3s** | 不调用 API 时，LangGraph 与本地 RAG 资产可以接通 |
| DeepSeek 在线冒烟 | 真实索引、真实 Reranker、真实 LLM | **通过 / 25.5s** | DeepSeek 接口、JSON 解析、Decision 和重试链可运行 |
| 检索质量 | 固定 seed 的 test 抽样 200 条 | 已有完整指标 | 比较 Vector、BM25、Hybrid、Rerank |
| 长文本小样本复测 | 临时抽样 10 条 | 第 5 条超过 60s 后中止 | Hybrid/BM25 长尾延迟可重现 |
| Component 体检 | Chroma 全量字段统计 + 5 条 LLM 抽样 | **发现系统性不匹配** | 当前在线向量过滤存在结构性失效风险 |

### 3.1 离线冒烟明细

- 依赖与图加载：6.9 秒。
- 单次 Agent 执行：6.3 秒。
- 混合召回：30 条。
- 有标题或正文证据：30 条。
- 精排输出：5 条。
- 最终固定结果：`new / 0.9`。

这里的 LLM 是固定回复替身，因此它只证明工程接线和数据形状正确，**不能证明模型分类准确率**。

### 3.2 DeepSeek 在线冒烟明细

测试 Issue 为 Firefox 打开 PDF 后崩溃的描述。DeepSeek 将其改写为更集中的技术 Query，并返回 `component=pdf`。随后发生：

1. 带 `component=pdf` 的向量检索返回 0 条；
2. BM25 返回 30 条，docstore 为候选补齐正文；
3. Reranker 输出 5 条；
4. Decision 返回 `new / 0.3`，触发两次重试；
5. 三轮结束后仍为 `new / 0.3`。

完整在线执行耗时 25.5 秒。该结果证明重试回环可以跑通，也说明重试没有解决错误过滤造成的证据不足。

## 4. 检索指标

评估条件：held-out test set 共 1,726 条，固定 `seed=42` 抽取 200 条；排除 query 自命中。Rerank 最多返回 10 条，因此只比较其 @5/@10 指标。

样本中 Firefox 93 条、Thunderbird 59 条、SeaMonkey 2 条，Mozilla 系合计占 77%。划分按 duplicate family 隔离，但没有时间切分；因此这些结果适合做当前配置的内部对比，不代表其他代码库或未来数据的总体表现。

| 方法 | Recall@5 | Recall@10 | Recall@30 | Precision@5 | MRR | nDCG@10 | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector Only | 0.3338 | 0.3965 | 0.4835 | 0.074 | 0.2709 | 0.2886 | 0.022s | 0.035s |
| BM25 Only | 0.2730 | 0.3337 | 0.4305 | 0.062 | 0.2492 | 0.2589 | 2.901s | 49.016s |
| Hybrid | **0.3484** | **0.4257** | **0.5394** | **0.080** | **0.2775** | **0.2974** | 3.054s | 528.387s |
| Hybrid + Rerank | 0.3130 | 0.3701 | — | 0.071 | 0.2292 | 0.2542 | 6.336s | 53.087s |

### 4.1 可以得出的结论

- Hybrid 相比 Vector Only，Recall@10 提高 0.0292，Recall@30 提高 0.0559，说明 BM25 与向量召回确实存在互补。
- Hybrid 的质量提升幅度不大，但延迟代价极高。P50 从 0.022 秒升至 3.054 秒，主要成本来自全量 BM25 评分。
- 当前 Reranker 使 Recall@10 从 0.4257 降至 0.3701，MRR 从 0.2775 降至 0.2292，不能宣称“精排提升了检索质量”。
- Hybrid 的 P95 远高于 BM25 Only，不能简单解释成 RRF 本身慢。本轮至少有一个超长 Query 慢例停在 BM25 全量打分阶段，但全部长尾仍需逐阶段 tracing 才能完成归因。

### 4.2 不能得出的结论

- 不能把 Recall@10 0.4257 表述为“系统准确率 42.57%”。它只表示 duplicate 目标在 Top-10 中的平均召回情况。
- 不能用 `1 / MRR` 推导“平均命中排名”。当前 MRR 还包含未命中时的 0。
- 不能从 Golden Set 推导 `duplicate/similar/new` 三分类准确率。现有标签只有 duplicate 关系，没有人工标注的 similar 与 new 对照集。
- 不能根据单次在线冒烟判断 DeepSeek Decision 的总体效果。

## 5. 关键问题与优先级

### P0：Component 过滤与索引字段不一致

**证据：** Chroma 全量 392,142 个 chunk 的 `component` 值全部为空。5 条典型 Issue 的 DeepSeek 分析中，4 条分别产生 `login`、`terminal`、`pdf`、`imap`，1 条为 `null`。

**影响：** 对当前索引而言，任何非空 component 都会让向量召回直接归零，Hybrid 实际退化为 BM25。5 条样本中的 4 条只是风险信号，不能外推为真实流量比例。当前离线评估不经过 LLM component 过滤，因此 200 条 Hybrid 指标不能直接代表完整在线 Agent；在非空 component 请求上，它会高估向量支路的实际贡献。

**建议：** 在修复前，在线检索不要直接用自由文本 component 做硬过滤；可先改为软加权，或只有当 component 属于索引中真实存在的白名单时才过滤。若要保留硬过滤，需要重新构建有可靠组件字段的索引并单独评估过滤收益。

### P0：BM25 对超长 Query 的长尾延迟

**证据：** 已有 Hybrid P95 为 528.387 秒；本轮 10 条临时复测在第 5 条超长 crash 日志上超过 60 秒，停止时位于 `rank_bm25.get_scores()`。

**影响：** API 请求可能长时间占用工作线程，重试最多执行三轮，会进一步放大最坏延迟。

**建议：** 对进入 BM25 的 Query 做长度和 token 上限；保留错误码、堆栈签名、标题和关键词，丢弃重复日志。随后增加逐 query 延迟记录，重新测 P95/P99。

### P1：Reranker 当前产生负收益

**证据：** 同一 200 条样本上，加入 Reranker 后 Recall@5、Recall@10、MRR、nDCG@10 全部下降。

**可能原因：** 模型与 Issue duplicate 判定任务不匹配；只看 800 字符导致证据截断；训练目标更接近一般相关性而非严格 duplicate；动态 Top-K 不能挽回被错误降序的目标。

**建议：** 保存逐 query 排名明细，比较 rerank 前后 ground truth 的名次变化；按项目、文本长度、错误日志类型做 bad case 分桶；在完成分析前不要默认启用 Reranker。

### P1：当前向量索引的距离空间未被资产元数据证明

代码为新建 collection 配置了 cosine，但当前本地 collection 的 metadata 为 `None`，不能据此认定现有 1.8 GB 索引已经使用 cosine。`VectorRetriever` 又把返回 distance 按 `1 - distance` 解释为相似度，因此当前分值语义需要在重建或迁移验证后再确认。这个问题不一定改变所有排序，但会影响分数解释、阈值设计和实验可复现性。

### P1：缺少端到端分类测试集

**影响：** 当前只能评价“能否召回已知 duplicate”，不能评价 LLM 对 duplicate、similar、new 的分类边界，也无法证明低置信度重试带来净收益。

**建议：** 建立三分类人工标注集，至少记录 query、候选证据、正确类别、可接受关联 ID、判定理由；同时比较首轮与最终轮的净提升、反向恶化率和成本。

### P2：测试覆盖仍以模块级为主

现有 32 项测试有价值，但没有系统覆盖 API 首请求并发、索引重复构建、断点恢复、真实 LLM 格式漂移、三轮重试质量和全链分类准确率。报告中应把它表述为“32 项模块级测试通过”，不能扩写为“生产链路已充分验证”。

## 6. 项目成熟度判断

| 维度 | 判断 |
| --- | --- |
| 学习 RAG 核心架构 | **适合**。数据处理、双路召回、RRF、精排、Decision、Retry 都具备 |
| 本地演示 | **适合，但需说明限制**。离线和 DeepSeek 在线链都能运行 |
| 检索实验 | **基本可用**。Golden Set 和自命中/family 修复较可靠，但缺逐条记录 |
| 精排效果 | **未达标**。当前实测为负收益 |
| 分类效果 | **尚未建立证据**。缺少三分类标注集 |
| 延迟与稳定性 | **未达标**。BM25 长尾和三轮重试会造成严重延迟 |
| 生产部署 | **不建议**。需先解决 P0 问题并补充端到端评估 |

## 7. 后续阅读时如何使用本报告

阅读代码时，建议带着以下问题逐层核对：

1. 读 `src/agent/graph.py`：为什么低于 0.7 会重试，最大为什么是三轮？
2. 读 `src/agent/nodes.py`：`component` 从哪里产生，又怎样变成 Chroma 硬过滤？
3. 读 `src/retrievers/hybrid_retriever.py`：单 Query 的 Vector/BM25 RRF 与双 Query 的外层 RRF如何叠加？
4. 读 `src/retrievers/bm25_retriever.py`：为什么 Query 越长，`get_scores()` 的成本可能越高？
5. 读 `src/docstore.py`：为什么 BM25-only 候选必须补齐正文，才能公平参与精排？
6. 读 `src/reranker.py`：当前输入截断、Top-K 和动态扩展为什么可能让 duplicate 目标掉出前列？
7. 读 `eval/golden_set_builder.py`、`eval/metrics.py`、`eval/run_eval.py`：为什么要排除自命中、按 family 划分，以及为什么 Rerank 不能比较 Recall@30？

如果主要目标仍是理解 v1，可以用 v1 学习主干结构；但以下部分应直接以 v2 为准：Decision 输出校验、BM25 零分过滤与正文补齐、评估集构造、nDCG、自命中剔除和 family 隔离。即使读 v2，也必须结合本报告认识到 component、延迟和 Reranker 的现存问题。

## 8. 下一轮最小评测计划

1. 暂时关闭不受控的 component 硬过滤，复跑 200 条在线等价检索。
2. 给 BM25 Query 设置字符/token 上限，并保存逐 query 延迟，复测 P50/P95/P99。
3. 落盘每条 query 的检索与 rerank 排名，完成至少 30 条 Reranker bad case 归因。
4. 建立包含 duplicate、similar、new 的人工标注集，每类至少 50 条。
5. 在固定温度、固定 Prompt、固定模型版本下，对比 0 次重试与最多 2 次重试的准确率、恶化率、延迟和调用成本。

完成以上五步后，才适合形成可对外引用的“项目效果报告”。当前这份报告应作为**内部实测基线与阅读校准材料**使用。

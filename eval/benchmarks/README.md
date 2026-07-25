# 单项改动 Benchmark：Retry 诊断上下文

## 这次测什么

当前改动让低置信度重试从只看到：

```text
rewritten_query + decision + confidence
```

变成还能看到候选数量、证据缺失数、最高分和前两名分差。

本 Benchmark 要回答：

> 同一批低置信度 Issue 中，加入诊断上下文后，最终检索和 duplicate 判断是否更准，
> 同时有没有增加语义漂移、重试次数和延迟？

它不用于证明整个项目已经准确，也不替代完整检索和分类评估。

## 对照方式

脚本提供两个变体：

| 变体 | 含义 |
| --- | --- |
| `legacy` | 下一轮只看到 query、decision、confidence |
| `diagnostic` | 下一轮还能看到本次新增的检索诊断 |

两个变体使用相同代码、相同索引和相同案例，只替换 retry prompt。这样比直接比较两个
Git 提交更容易隔离“诊断上下文”这一项改动。

## 指标说明

| 指标 | 说明 | 期望方向 |
| --- | --- | --- |
| `final_retrieval_hit_rate` | 最终 reranked_docs 是否包含人工标注的 duplicate | 越高越好 |
| `final_related_issue_hit_rate` | LLM 返回的 related_issues 是否命中 duplicate | 越高越好 |
| `duplicate_decision_accuracy` | 是否判断 duplicate 且相关 ID 正确 | 越高越好 |
| `required_term_retention_rate` | 原文关键实体在最终 rewritten_query 中的保留比例 | 越高越好 |
| `forbidden_term_case_rate` | 最终 query 是否出现人工标注的无依据内容 | 越低越好 |
| `retry_trigger_rate` | 实际触发重试的案例比例 | 只用于解释结果 |
| `average_retry_count` | Decision 平均执行次数 | 结合效果和成本判断 |
| `average_latency_seconds` | 单条端到端平均耗时 | 越低越好 |

当前 history 没有保存每一轮的完整候选 ID，因此本版不能严格计算“首轮失败、第二轮命中”
的 Retry Recovery Rate。它衡量的是最终结果，不会假装已经具备逐轮召回指标。

## 案例要求

案例保存在 `eval/benchmarks/data/retry_cases.json`。当前只有不会执行的示例，状态是
`draft_not_run`。

正式案例需要满足：

1. 使用不会在当前索引中自召回的新 Issue 文本；
2. 人工确认 `expected_duplicate_ids` 确实是同一问题；
3. duplicate ID 必须存在于当前索引；
4. `required_terms` 只能来自原文；
5. 只有人工核验后才能将案例状态改为 `verified`。

建议先准备 20–50 条，覆盖：

- 原文描述含糊；
- component 容易误判；
- 候选分数接近；
- BM25-only 候选缺少正文；
- 首轮低置信度但换一种检索表达可能恢复的情况。

## 后续运行

运行会调用真实 LLM 和本地检索，当前不要直接执行。准备好案例后：

```bash
python -m eval.benchmarks.retry_quality run \
  --variant legacy \
  --label before \
  --repeats 3

python -m eval.benchmarks.retry_quality run \
  --variant diagnostic \
  --label after \
  --repeats 3

python -m eval.benchmarks.retry_quality compare \
  eval_results/retry_quality_before.json \
  eval_results/retry_quality_after.json \
  --output eval_results/retry_quality_comparison.json
```

重复三次是为了降低 LLM 输出波动对结论的影响。

## 当前状态

```text
Benchmark ID：retry-context-v1
目标改动：低置信度 retry 增加检索诊断上下文
修改前行为：legacy
修改后行为：diagnostic
案例状态：尚未人工标注
运行状态：NOT RUN
当前不能声明：召回率、分类准确率或 retry recovery 已提升
```

---

# 单项改动 Benchmark：原文与改写双路召回

## 这次测什么

生产检索现在有两个变体：

| 变体 | 检索输入 |
| --- | --- |
| `single_rewritten` | 只使用固定的 `rewritten_query` |
| `dual_query` | 分别使用 `rewritten_query` 和 `raw_issue`，再用外层 RRF 融合 |

Benchmark 要回答三个问题：

1. 原文兜底是否提高 Retrieval Top-30 的 Recall 和 Hit Rate；
2. 新召回的正确候选经过现有 Reranker 后，是否仍能留在 Top-10；
3. 双倍检索带来的延迟是否可以接受，以及是否出现原先能命中、双路反而漏掉的退化案例。

这不是 Prompt 对比。案例直接保存固定的 `rewritten_query`，运行时不调用 Query Analysis
LLM，从而把唯一实验变量限制为“是否增加 raw_issue 检索”。

## 为什么同时测 Retrieval 和 Rerank

只看 Retrieval Top-30 可能得到虚假的乐观结论：原文虽然把正确 Issue 捞进候选池，
但后面的 Reranker 仍使用 `rewritten_query` 打分，正确 Issue 可能再次被删掉。因此报告同时包含：

- Retrieval：Hit Rate、Recall、MRR、nDCG@10；
- Rerank：Top-5/10 Hit Rate、Recall、MRR、nDCG@10；
- Safety：`rescue` 和 `regression` 的案例数与比例；
- Cost：平均、P50、P95 总延迟，以及 Retrieval/Rerank 分段平均延迟。

其中：

- `rescue`：单路没有命中，但双路命中；
- `regression`：单路能够命中，但双路没有命中。

不能只报告平均 Recall 上升，也必须检查 `regression_count` 和延迟。

## 案例要求

案例保存在 `eval/benchmarks/data/dual_query_cases.json`。当前只有不会执行的草稿示例。
建议准备至少 30–50 条人工确认案例，并按下列类型分组：

- `rewrite_drift`：改写增加了原文没有的根因或删掉关键实体；
- `rewrite_good`：改写准确，用于检查原文噪声是否导致退化；
- `rewrite_fallback`：改写为空或格式失败，最终等于原文；
- `long_noisy_raw`：原始 Issue 很长、日志噪声很多；
- `exact_entity`：包含错误码、类名、函数名或版本号。

每条案例必须固定保存：

```text
raw_issue
rewritten_query
component
expected_duplicate_ids
project
query_analysis_version
case_type
```

`component` 在两个变体中保持相同，无法确认时填 `null`。`expected_duplicate_ids` 必须存在于
本次使用的索引中；只有人工核验后才能把状态改成 `verified`。同一个真实 Query Analysis
版本应同时收集成功和失败样本，不能只挑双路有利的案例。报告会按 `case_type` 分组，
用来区分双路召回对改写漂移和正常改写分别产生的影响。Runner 启动后还会检查标注 ID
是否存在于当前 BM25 索引；存在不可达目标时直接停止，不会把它错误计为检索失败。

## 运行方式

准备好 verified 案例后运行：

```bash
python -m eval.benchmarks.dual_query_retrieval run \
  --repeats 3 \
  --output eval_results/dual_query_retrieval.json
```

脚本会加载真实 Vector、BM25 和 Reranker，但不会调用 LLM。`--repeats 3` 主要用于让延迟
统计更稳定；排名结果本身通常是确定的。

## 建议的通过条件

在正式案例数量足够前，不应填写具体阈值。第一版可以先采用方向性门槛：

```text
Retrieval Recall@30：双路不得低于单路，且 rewrite_drift 子集应有明确提升
Rerank Hit Rate@10：提升必须能保留到精排之后
Regression Rate：逐条审查，不能只被平均值掩盖
P95 延迟：记录增幅，再结合服务目标决定是否接受
```

如果 Retrieval Recall 上升但 Rerank Top-10 没有改善，下一步应检查 Reranker 使用
`rewritten_query` 是否把原文召回的正确候选再次压低，而不是直接宣布双路方案有效。

## 当前状态

```text
Benchmark ID：dual-query-retrieval-v1
目标改动：增加 raw_issue 兜底召回，并与 rewritten_query 结果做外层 RRF
修改前行为：single_rewritten
修改后行为：dual_query
案例状态：尚未人工标注
运行状态：NOT RUN
当前不能声明：Recall、Rerank 命中率或端到端准确率已经提升
```

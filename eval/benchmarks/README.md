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

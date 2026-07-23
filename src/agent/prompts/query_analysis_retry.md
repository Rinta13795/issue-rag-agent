---
name: query_analysis_retry
description: 重试时拼接到 user prompt 末尾的反思块，含上轮决策与检索诊断
version: v2
---

## 上次重试反思

上次改写 query：{last_query}
上次关键词：{last_keywords}
上次 component：{last_component}
上次置信度：{last_confidence}（低于阈值 0.7 触发重试）
上次决策：{last_decision}
上次 related_issues：{last_related_issues}
上次判断理由：{last_reasoning}

检索诊断：
- RRF 融合后候选数：{retrieved_count}
- Reranker 输出候选数：{candidate_count}
- 缺少 title/body 证据的候选数：{missing_evidence_count}
- 最高 rerank 分数：{top_score}
- 前两名分差：{score_gap}

不要把低置信度直接等同于“召回不准”，也不要把上次 reasoning 当作事实。结合原始
issue 和以上诊断重新改写：

- 没有候选时，适当放宽 query；若 component 缺少明确依据，返回 null，避免过滤过严。
- 候选较多且前两名分差很小时，补充能区分候选的错误码、API、触发动作或对象。
- 缺失证据的候选较多时，不要编造候选内容；优先保留原始 issue 中可验证的技术信号。
- 仍无法判断时，忠实保留原始描述，不要补写原文不存在的错误码、组件或根因。

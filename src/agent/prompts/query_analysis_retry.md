---
name: query_analysis_retry
description: 重试时拼接到 user prompt 末尾的反思块，含上次 query/置信度/决策
version: v1
---

## 上次重试反思

上次改写 query：{last_query}
上次置信度：{last_confidence}（低于阈值 0.7 触发重试）
上次决策：{last_decision}

上次置信度低说明召回不准。可能原因：
- query 太宽泛，没抓住核心技术信号
- 关键词太常见，被噪声淹没
- component 判断错误，metadata filter 把正确答案排除

请换角度改写：尝试不同技术词组合、调整 component 判断、聚焦更精确症状。

# Step 6: 各节点 Prompt 设计

## 这一步做什么

为 Query Analysis 和 Decision 两个 LLM 节点设计结构化 Prompt，并实现候选 issue 格式化。

## 涉及的文件

- `src/prompts.py`
- `src/agent/nodes.py`

## 输入输出

- 输入：原始 issue、上一轮反馈、改写 query、keywords、Top5 candidates。
- 输出：LLM 可解析 JSON，以及 Decision 节点需要的候选 issue 文本。

## 关键设计细节

Query Analysis Prompt：

```text
角色:你是开源项目 issue 分析专家,擅长把口语化 bug 报告改写成结构化技术查询。

任务:给定一条 issue,输出 JSON,包含三个字段:
1. rewritten_query: 重写后的技术查询(英文为主,覆盖核心症状 + 组件 + 错误关键词)
2. keywords: BM25 检索用的关键词列表,重点提取:
   - 错误码(NullPointerException、E_INVAL 等)
   - API/函数名(useState、saveDocument 等)
   - 组件名(SaveManager、ReactRouter 等)
   - 报错位置(文件名:行号)
3. component: 推断的核心组件名(首字母大写),无法推断填 null

原则:
- 不照抄原文,重写要更技术、更精炼
- 关键词 3-8 个,按重要性降序
- 同义改写优先列出(white screen / blank page / UI rendering failure 都列)
- 错误码、堆栈位置必须放 keywords
{feedback}

Issue:
{raw_issue}

只输出 JSON,不要解释:
```

Query Analysis JSON 目标格式：

```json
{
  "rewritten_query": "Application UI renders blank white screen after clicking the Save button. macOS Sonoma environment.",
  "keywords": ["save", "white screen", "blank page", "click", "macOS", "UI render"],
  "component": "SaveManager"
}
```

Decision Prompt 核心约束：

- 角色：开源项目维护者，判断新 issue 是否为历史 issue 的重复。
- 输入：`query`、`keywords`、按检索相关性排序的 Top5 `candidates`。
- 输出 JSON：`decision`、`confidence`、`related_issues`、`reasoning`。
- `decision` 三选一：`duplicate`、`similar`、`new`。
- `duplicate`：症状、根因、复现路径都对应。
- `similar`：同组件或类似现象，但不是同一个 bug。
- `new`：候选中无匹配。
- 置信度：0.9+ 表示错误码、堆栈、复现步骤都对应；0.7-0.9 表示描述匹配但缺细节；0.5-0.7 表示不太确定；<0.5 表示极不确定，建议触发重写。
- 判断要点：同模块但根因不同是 similar；描述像但错误码不同是 similar；用户痛点和根因都一致才是 duplicate。

`format_candidates` 输出格式：

```text
[1] issue_id=ProjA-12000 | score=0.91
    Title: White screen after clicking save (macOS)
    Body excerpt: Application crashes with TypeError after save click...
    Labels: bug, ui
    Component: SaveManager

[2] issue_id=ProjA-9800 | score=0.87
    Title: Save button unresponsive after Vite upgrade
    ...
```

Prompt 约束：

- Decision 只基于 Top5 candidates 判断，不引入候选外 issue。
- `reasoning` 控制在 80 字以内。
- `new` 时 `related_issues` 返回空列表。
- retry 时 `{feedback}` 注入上一轮 Top3 线索。

## 关键参数

- `LLM_TEMPERATURE = 0.1`
- `LLM_MAX_TOKENS = 1024`
- `CONFIDENCE_THRESHOLD = 0.7`
- `RERANK_TOP_K = 5`

## 依赖哪些已完成的模块

- Step 1：LLM 配置。
- Step 5：Prompt 被节点函数调用。

## 完成标志

- [ ] `QUERY_ANALYSIS_PROMPT` 实现。
- [ ] `DECISION_PROMPT` 实现。
- [ ] `format_candidates(docs)` 输出原文指定格式。
- [ ] Query Analysis 输出 JSON 字段完整。
- [ ] Decision 输出 JSON 字段完整。

## 踩过的坑

# 计划：用 Harness 回答“为什么需要 AI、为什么不直接用通用 Agent”

状态：仅计划，尚未实施。本文不新增功能，只整理现有能力并补文档与小修复。

## 要解决的问题

面试官的质疑会以不同说法反复出现：

- 智能分诊用传统程序不就能解决吗？
- 为什么不直接让 Claude Code / Codex 来做，还要自己设计 Agent？
- 模型不确定，出错怎么办？它和“ChatGPT 加搜索”有什么区别？

这些都是同一个问题：**哪里交给模型、哪里用程序，有什么手段保证出错可控，有什么证据。** 答案要说明项目与通用 Agent 的差别，而不是声称能力更强。

## 核心框架：Agent = 模型 + Harness

Harness 指模型之外的全部程序：工具定义、调用循环、预算、权限范围、上下文管理、输出校验、状态保存与恢复、测试与评测。

- Claude Code、Codex 本身就是通用 Harness：权限宽、适用面广，约束靠使用者事后管理。
- 本项目是任务专用 Harness：只做“读当前仓库的 Issue/PR 并调查”，把能力收窄，换取可控、可验证、可复现。
- 差别不在模型更聪明，而在 Harness 为这个任务设计得更窄、更可检查。

“Harness”还有一个相关含义：测试夹具（test harness）。本项目用替身模型离线回归，就是在测 Harness 本身。

注意：该术语在社区用法并不完全统一。面试前找一个可靠来源核对自己的定义，不要把没核对的说法当作权威表述。

## 现有能力与证据（均已核对源码与测试）

| Harness 要素 | 代码位置 | 对应测试 |
| --- | --- | --- |
| 工具收窄（6 个工具，参数校验） | `src/chat/runtime.py` 的 `DEFINITIONS` | `test_chat_runtime.py::test_followup_reads_linked_pr_and_rejects_fake_citation` |
| 范围限制（仅当前仓库，剔除 `repo:` 等限定符） | `src/chat/github_research.py`、`src/chat/github_sync.py`、检索路径的仓库前缀过滤 | `test_search_keeps_rerank_and_repository_filter`、`test_hybrid_retriever.py` 仓库隔离用例 |
| 预算（模型调用 6 次、工具调用 10 次，重复调用缓存） | `InvestigationRuntime.run` | `test_tool_and_model_budgets`、`test_repeated_tools_cached_errors_return_to_model` |
| 引用校验（只保留实际读过的证据 ID） | `src/chat/final_response.py` | `test_final_response.py`、上面的假引用用例 |
| 状态语义（PR 说明只能是 supported，用户原话才可 verified；“解决了”不认定具体方案） | `src/chat/memory_service.py` | `test_memory_service.py::test_unspecified_resolution_cannot_validate_guessed_pr` |
| 暂停与恢复、重启后不重复发布 | `src/chat/store.py`、`runtime.py` | `test_question_restart_idempotence_and_resume`、`test_interrupted_issue_publish_is_not_repeated_after_restart` |
| 固定流程用程序，不交给模型 | `src/agent/graph.py`（单次分诊） | `test_decision_validation.py` |

## 已知缺口（面试中如实承认，不夸大）

1. 没有“Agent 优于简单方案/通用 Agent”的对比数据；对话调查没有任何评测。
2. `api.py` 允许任意来源跨域，发布 Issue 接口使用服务端 Token，削弱“边界由程序保证”的说法。
3. 文档仍称项目“不是自主 Agent”，与对话调查不符。
4. 旧 planner 架构留下无调用的提示词与函数，会让读代码的人看到两套设计。

## 任务清单

1. 面试手册新增专题：为什么需要 AI、为什么不用传统程序、为什么不直接用通用 Agent（口语稿 + 变体问法对照 + 追问弹药）。
2. 修正 `docs/04a` Q5：单次分诊是固定流程，对话调查才是模型选工具；并补充对话调查的架构说明。
3. 把上表整理为“边界证据表”，放进专题，便于面试时指给对方看。
4. 收窄 `api.py` 的跨域来源，并给发布接口加最小保护；补一个对应测试。
5. （可选）清理 planner 死代码与前端空区块。
6. （可选，本地做）取 5 个真实 Issue，对比通用 Agent 与本系统的耗时、token、动作、是否越界，写成定性记录。不编造统计数字。

## 不做的事

- 不为了显得“灵活”增加功能。
- 不编造对比数据或评测数字。
- 不把未经核对的术语说法写成定论。

## 验收

- 能在一分钟内说清：模型管什么、程序管什么、出错怎么兜底、有什么证据、哪里还不足。
- 手册中的每条“程序保证”都能指到代码位置和测试。
- `python3 -m pytest tests/ -q` 全部通过（改动代码时先跑）。

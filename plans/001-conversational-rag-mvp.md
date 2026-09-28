# Plan 001: 为 Issue RAG 增加有上下文、按需取证的对话入口

> **执行记录（2026-09-28）**：本计划已实施。下文“当前状态”记录的是实施前基线，非现在的代码状态。离线测试和前端构建已通过；真实模型调用因本地凭据鉴权失败尚未验证成功。保留原有单次分诊入口，未提交或推送 Git。
>
> **漂移检查（先执行）**：`git diff --stat 547a896..HEAD -- api.py config.py src/agent/graph.py src/retrievers/hybrid_retriever.py src/reranker.py frontend/src/App.tsx frontend/src/api.ts frontend/src/types.ts frontend/src/index.css tests`。若这些位置已变化，先与“当前状态”逐项核对；关键契约不一致时停止并报告。

## 状态与边界

- 优先级：P1；工作量：M（约数个工作日，含前后端及离线测试）；风险：中（共享检索依赖、异步会话状态）。
- 依赖：本地已有向量、BM25、docstore 索引；有效的模型服务凭据只从本地 `.env` / 环境变量读取。
- 计划基线：`547a896`，2026-09-28。
- 目标：用户能像聊天一样描述问题、补充线索、追问候选；Agent 只在需要历史证据时调用现有 RAG，再根据证据用自然语言回答。
- 非目标：通用编程 Agent、网页搜索、PR / 文档知识库、长期跨会话记忆、自动修复代码、生产级多租户服务、宣称三分类准确率。

## 为什么做

当前界面是一次输入对应一次分诊运行。用户说“第一条为什么像”、补充一个错误码，或贴出很长但有效线索很少的日志时，现有入口无法理解对话指代，也会把新提交当作独立任务。第一版重点不是把表单换成聊天气泡，而是实现“上下文过滤 → 按需 RAG → 证据化回答”，同时避免每轮重跑昂贵的检索和低置信度循环。

## 当前状态与代码证据

- `frontend/src/App.tsx:18-19,40-76,86-110`：只有 `workspace / knowledge / quality` 三个视图；`handleAnalyze` 每次调用 `createRun`，没有 conversation/session 状态。
- `api.py:103-131`：`POST /api/runs` 每次创建独立 `run_id`；`GET /api/runs/{run_id}` 只读单次快照。
- `src/agent/graph.py:56-79`：依赖进程内缓存；现有四节点低置信度时会自动回到 Query Analysis。聊天中的普通追问不能直接调用整条图。
- `src/agent/nodes.py:317-326`：检索把改写 Query 与完整 `raw_issue` 双路传入。多轮聊天不能把全部聊天记录作为 `raw_issue` 直接重查，否则长日志与闲聊会污染检索。
- `src/retrievers/hybrid_retriever.py:31-37` 暴露 `search(query, top_k, filter_dict, bm25_extra_terms)`；`src/reranker.py:25-31` 暴露 `rerank(query, docs, top_k, dynamic_threshold)`。复用它们，不另造一套检索算法。
- `src/demo/runner.py:118-125` 向前端只放候选 body 前 240 字；`src/docstore.py:16-30` 的 docstore 正文本身最多保存 1200 字。不能据此宣称读过完整 Issue 或确定修复方案。
- `src/demo/run_store.py:30-38` 是进程内状态（默认最多 20 条、TTL 一小时）；聊天 MVP 可沿用“本地、重启丢失”的明确边界，但须单独管理 session，不把 run 当 session。
- `config.py:9-32` 已从本地环境加载 DeepSeek 配置；不得把用户曾在聊天里提供的密钥写进源码、计划、测试、日志或示例。
- `tests/test_demo_api.py` 用 `unittest + FastAPI TestClient`；`tests/test_run_store.py` 用纯单元测试。新增测试沿用这两种模式，模拟 LLM 与 Retriever，不依赖真实付费调用。
- 前端现有设计变量在 `frontend/src/index.css`。实施任何视觉修改前必须完整阅读 `/Users/chenyuhang/Desktop/ai-style/设计风格手册.md`，逐条检查第五节反面清单。沿用纸面、墨色、细线、5px 圆角，禁止渐变、组件投影、彩色气泡及图标墙。

## MVP 产品契约

每个会话维护一张小型“问题卡片”，不是不断增长的 LLM 聊天摘要：

```text
session_id
messages：原话和回复，供 UI 展示；不全量回灌模型
issue_facts：用户实际说过的故障事实，每项记录 value、source_message_id、原文摘录
open_question：当前最关键的待澄清点（最多一项）
focus_candidate_id：上一轮正在讨论的候选 Issue ID（可空）
last_candidates：上次检索的少量候选 ID、标题、证据摘录、分数
last_search_fingerprint：上次使用的已验证检索事实指纹
metrics：每轮模型调用数、检索次数、耗时；有 usage 时记录 token，不猜测金额
```

用户每发一条消息，只判断相对当前卡片的变化：

| 新消息 | 行为 | 例子 |
| --- | --- | --- |
| 指向已有候选的追问 | 只用已有证据回答，不检索 | “第一条为什么像？” |
| 新增可检索故障事实 | 合并到卡片；事实指纹变化才检索一次 | “补充：v2.1，返回 401” |
| 信息不足但指代明确 | 回答已知部分，再只追问一个关键细节；允许用户坚持按现状搜索 | “登录不了” |
| 指代不明 | 先澄清指的是哪条候选，不擅自选 | “那这个呢？”但当前有多条候选 |
| 长篇叙述或日志 | 提取症状、动作、精确错误词及必要环境；只新增原文能定位的事实 | 几十行依赖列表，真正现象是启动崩溃 |

**不按字数决定是否检索**：短消息可能包含高价值错误码，长消息可能没有新事实。用户原话保存在 session 中，但模型输入只包含当前消息、问题卡片、最近两轮必要对话和最多 3 条候选证据。事实必须能在对应用户消息里找到原样片段；模型推测的根因、版本、组件不能写进 `issue_facts`。若事实提取失败，保持原状态并澄清。

## 对话执行路径与成本上限

1. 一个受限的 Context Planner 处理当前消息，输出结构化动作 `reply / clarify / retrieve`、新增事实及其原文摘录、候选指代、检索 Query；程序校验动作枚举、事实原文、候选 ID 白名单和字段长度。对 `reply / clarify`，Planner 同次调用可以给出答复，避免第二次 LLM 调用。
2. `retrieve` 时只用已验证用户事实组成有长度上限的检索输入；保留错误码、函数名、关键日志行原样。不要把整段聊天或未经验证的 LLM 扩写塞入原文路。通过共享依赖复用现有 HybridRetriever 和 Reranker；每轮最多一次检索，不调用现有四节点图的自动低置信度回环，也不默认调用三分类 Decision。
3. 检索后，用一次 LLM 调用形成面向用户的回答：说清找到什么、为什么相关、还缺什么。传入当前问题、问题卡片、最多 3 条候选的有界证据。输出结构化引用 ID，由程序检查必须属于本轮候选；没有证据时不能编造历史 Issue 或确定修复步骤。
4. 聊天和原有分诊是两个入口：聊天可提示“转到单次分诊”查看 `duplicate / similar / new`，但普通对话不得把相似候选直接包装成已验证的 duplicate。原有 `/triage`、`/api/runs` 和四节点链保持行为不变。
5. 默认每轮最多 2 次模型调用（Planner + 检索后回答）、1 次检索、0 次自动认知重试。Planner 的普通追问分支只需 1 次模型调用。分别限制用户消息长度、事实数、候选数和单候选摘录长度；具体数值集中配置并在测试中验证，不把字符上限误称为精确 token 上限。模型供应商 usage 字段不存在时只记录调用次数，不虚构费用。

候选文本与用户输入都视为**数据**，不是可执行指令。Prompt 只能要求基于证据，不能保证事实正确；候选 ID 校验只能拦截越界引用，不能证明引用内容支持结论。LLM 故障时返回可见的“当前无法完成分析”，不得复用既有 Decision 中按精排分数合成的高置信 `duplicate/new` 作为聊天事实。

## API 与运行形态

- 新增 `POST /api/chat/sessions`：创建空本地会话，返回 `session_id`。
- 新增 `GET /api/chat/sessions/{session_id}`：返回会话快照（消息、状态、候选摘要、问题卡片、计数），不存在或过期返回 404。
- 新增 `POST /api/chat/sessions/{session_id}/messages`：验证非空输入与长度，追加用户消息，返回待处理消息 ID；后台执行，避免长 BM25/模型调用占住浏览器请求。重复提交用客户端生成的 `client_message_id` 幂等处理。
- 前端轮询 `GET` 获取状态即可；第一版不实现 token 级流式输出。状态至少有 `thinking / retrieving / answering / completed / failed`，失败不能留下永久 loading；同一会话同一时间只处理一条消息。
- session 是有界进程内存储（TTL、容量、逐 session 锁）；浏览器只保存 `session_id` 用于刷新恢复。服务重启后友好提示会话失效并允许新建；页面明确标注“本地会话，非持久记忆”。

接口字段用 Pydantic 明确定义，前端 `frontend/src/types.ts` 同步。不要把完整服务密钥、环境变量、内部异常栈或模型原始 Prompt 放进会话快照。现有 API 的跨域设置只适合本地演示；本计划不将其宣称为可公网部署的权限模型。

## 实施步骤（每步完成后验证）

### 1. 定义会话数据契约和 Context Planner

新增 `src/chat/models.py`、`src/chat/context.py`、`src/chat/prompts.py` 与 `tests/test_chat_context.py`。上下文函数必须是可注入模型客户端的纯业务层：从当前消息提取可溯源事实、解析当前候选指代、组装有界输入；非法 JSON、无原文依据事实、越界候选 ID、空 Query 均安全降级。至少测试短消息、长噪声日志、指代明确/不明、重复事实、虚构事实和空消息。

**验证**：`python -m pytest tests/test_chat_context.py -q` → 全部通过，无真实网络调用。

### 2. 复用检索组件并实现对话服务

在 `src/agent/graph.py` 提供一个经 `_ensure_dependencies()` 获取已缓存检索器与精排器的公开 accessor（或抽出同等共享依赖函数）；不要从聊天代码读 `nodes.py` 的私有全局变量，也不要每条消息重建 Chroma/BM25/Cross-Encoder。新增 `src/chat/service.py`、`src/chat/store.py`，用注入的假检索器/假模型测试动作路由、检索指纹去重、候选证据裁剪、引用白名单、调用失败与成本计数。不要调用 `run_agent()` 来回答普通追问，因为它固定执行 Decision 和条件重试。

**验证**：`python -m pytest tests/test_chat_service.py -q` → “追问已有候选”检索调用 0 次，“补充新事实”最多 1 次；旧测试不受影响。

### 3. 接入 FastAPI，保留旧接口

在 `api.py` 增加上述三个路由、请求校验与后台任务；业务细节留在 `src/chat/`。参照 `tests/test_demo_api.py` 增加 `tests/test_chat_api.py`，用离线替身断言创建会话、续聊、幂等、过期、并发消息被拒绝或排队的明确行为、失败状态可恢复，以及 `/api/runs` 仍可用。测试中不触发真实模型下载或外部 API。

**验证**：`python -m pytest tests/test_chat_api.py tests/test_demo_api.py -q` → 全部通过。

### 4. 做最小聊天工作台

在 `frontend/src/App.tsx` 增加“对话排查”视图，保留现有“单次分诊”“知识与数据源”“质量评估”；在 `frontend/src/api.ts` 和 `frontend/src/types.ts` 增加会话请求与类型；聊天组件可新建于 `frontend/src/components/ChatWorkspace.tsx`，样式加在现有 `frontend/src/index.css`。界面至少展示消息流、输入框、当前问题卡片、候选证据及来源 ID、正在理解/检索/整理的状态、错误恢复和“新对话”。候选证据可以展开，但不得伪造 GitHub URL 或显示“已确认修复”。刷新后尝试恢复同一 session；404 时提示新建。桌面与手机窄屏均可用，键盘 Enter 提交、Shift+Enter 换行。

**视觉前置与验收**：完整阅读 `/Users/chenyuhang/Desktop/ai-style/设计风格手册.md`，严格复用 token、节标三件套、留白和表单样式，交付前逐条核对第五节 10 项反面清单；遇到手册未覆盖的视觉决策先向用户提问，不能自造新样式。

**验证**：`cd frontend && npm run build` → TypeScript 与 Vite 均成功；`cd frontend && npm run lint` → 无新增错误。用本地浏览器实际检查桌面和 ≤640px 窄屏，记录检查结果。

### 5. 全量回归与有限真实试用

用以下对话逐条手测：①“登录不了”→ 只追问必要细节或允许按原文搜索；②“补充：v2.1，401”→ 卡片新增精确事实并触发一次检索；③“第一条为什么像”→ 引用当前候选，不重检索；④“那这个呢”且焦点不明→ 澄清；⑤ 贴大量环境信息但只有一句崩溃现象→ 检索输入聚焦现象，原始日志仍可查看；⑥ 候选为空/模型 402、超时→ 不编造结果，能恢复继续聊。若凭据或余额不可用，完成离线测试后如实记录真实调用未验证，不要把 mock 演示说成在线成功。

**验证**：`python -m pytest tests/ -q` 和 `cd frontend && npm run build` 均通过；`git status --short` 仅显示本计划范围内的实现与测试文件（以及用户既有修改）。记录是否实际运行过在线模型、调用次数与可用的 usage 统计。

## 文件范围

**允许修改/新增**：`api.py`、`config.py`（只加聊天上限配置）、`src/agent/graph.py`（只暴露共享依赖）、`src/chat/`、`frontend/src/App.tsx`、`frontend/src/api.ts`、`frontend/src/types.ts`、`frontend/src/index.css`、`frontend/src/components/ChatWorkspace.tsx`、`tests/test_chat_*.py`、`plans/README.md` 的状态行。若需要额外前端小组件，可放在 `frontend/src/components/` 并列明用途。

**不要修改**：`src/retrievers/` 的召回算法、`src/reranker.py` 的打分算法、`src/agent/nodes.py` 的原有四节点语义与自动重试、索引/评测数据、历史结果、现有单次分诊接口返回格式；不要加入 PR/Docs/网页搜索、长期 Memory 框架或新 UI 组件库。不要提交凭据、改动 `.env`、推送 Git。

## 完成标准

- [x] 同一 session 中短追问能解析当前候选，不新增 run、不做检索（离线测试）。
- [x] 新增用户事实可溯源到原文，发生实质变化时至多检索一次；模型推测不得进入事实卡片（离线测试）。
- [x] 长日志不会整段进入每轮 Prompt；仅有界的事实和候选证据被送入模型（离线测试）。
- [x] 结构化答案引用的 Issue ID 必须属于本会话当前候选；无证据时明确不确定。自由文本仍主要受 Prompt 约束，不能声称事实准确性已被程序证明。
- [x] 模型/检索调用次数和耗时可查看；未展示未经价格配置计算的“实际费用”。
- [x] 会话刷新可恢复，服务重启失效可理解；错误/超时不会永久卡住。
- [x] 旧分诊入口和原有测试通过；新增聊天测试和前端构建通过。
- [x] 视觉按设计手册第五节反面清单检查过；桌面与窄屏浏览器已目视检查。

**尚未验证**：有效模型凭据下的完整在线对话、真实检索结果质量与实际 token 成本；此次模型服务返回鉴权失败。

## STOP 条件与维护备注

若当前代码和上述接口/行为已明显漂移，需扩大到改动原有检索或四节点算法，发现必须依赖尚未入库的 PR/文档证据，或视觉手册没有覆盖关键交互而无法按规则实现，停止并向用户说明，不要自行扩大范围。凭据失效或余额不足**不阻止离线实现**，但禁止声称真实在线对话已跑通。

本 MVP 的进程内会话只适合个人本地单进程使用；部署多 worker、跨设备同步、长期偏好记忆和安全隔离都需要后续单独设计。面试表述应为“实现了有界上下文与按需检索的多轮排查原型”，不能说“模型永不幻觉”或“已验证减少多少成本”，除非后续有实测记录。

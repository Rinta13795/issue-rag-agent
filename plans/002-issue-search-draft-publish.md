# Plan 002：让对话助手先查历史，再经用户确认提交 GitHub Issue

> 历史设计草案，记录实施前的判断与取舍。当前实现与排错入口见 [Issue 对话工作流](../docs/issue-case-workflow.md)；下文“当前代码事实”和验收门槛均以计划基线为时间点，不代表此刻仍未实现。

## 状态与目标

- 状态：已按本轮“已有 Issue 导入优先”的路径实施；在线 GitHub/模型/发布仍待人工验收。
- 优先级：P1；工作量：M—L；风险：中（新增对公开仓库的写操作）。
- 依赖：已有仓库绑定对话和 GitHub 仓库快照。
- 计划基线：`f4dd0e9`，2026-09-29。
- 漂移检查：`git diff --stat f4dd0e9..HEAD -- api.py src/chat frontend/src/components/ChatWorkspace.tsx frontend/src/api.ts frontend/src/types.ts tests`

## 为什么做

当前助手能讨论问题并查已同步的历史 Issue，但不能完成用户真正想要的动作：判断是否已有反馈，在没有合适记录时整理并提交新 Issue。截图中的“体验问题该提 Issue 还是 PR？”本应先得到决策建议，却被当成故障排查反复索要日志。新能力应形成“回答问题 → 按需查重 → 生成草稿 → 用户确认 → 发布”的闭环，而不是把所有对话自动导向提单。

## 当前代码事实

- `src/chat/prompts.py` 的规划器仅支持 `reply|clarify|retrieve`；它没有“准备提交”意图。
- `src/chat/service.py:213-225` 在连续 `clarify` 时会覆盖模型回复为固定的“不能读取本机终端或日志”文本，不检查本轮是否属于一般咨询。
- `src/chat/github_sync.py:34-90` 能解析公开仓库并从 GitHub 读取 Issue，但快照只保留最近更新的最多 300 条，且后续不会自动刷新。
- `src/chat/repositories.py` 的 `ChatRepository.github_url` 可以区分有 GitHub 目标的仓库与仅有 GitBugs 历史索引的项目；会话由 `ChatSession.repository_id` 固定仓库。
- `api.py:206-277` 有仓库同步、会话与发消息接口，没有 Issue 草稿或发布接口。
- `frontend/src/components/ChatWorkspace.tsx` 显示对话和候选，暂无可编辑草稿、确认发布和发布结果状态。
- `tests/test_chat_service.py` 用 FakeLLM/FakeRetriever 覆盖离线会话行为；新测试应沿用替身模式，不能依赖真实 GitHub 或真实模型。
- 项目约定见 `CLAUDE.md`：Python/FastAPI，聊天类型集中在 `src/chat/models.py`，关键步骤有中文注释，测试命令为 `python3 -m pytest tests/ -q`。前端命令见 `frontend/package.json`：`npm run lint`、`npm run build`。
- 前端视觉应遵守 `/Users/chenyuhang/Desktop/ai-style/设计风格手册.md`：纸面与墨色层级、1px hairline、5px 圆角、赭橘只用于微型标记/交互反馈；产出前逐项核对第五节反面清单。

## 产品行为：三种意图，不混为一谈

1. **普通咨询**：例如“体验问题该提 Issue 还是 PR？”直接解释判断标准，可以建议“要不要查当前仓库是否已有反馈”，但不自动检索、不索要错误码、更不提交。
2. **查历史**：例如“帮我看看有没有人反馈过这个问题”。先查会话绑定仓库的本地快照；发布前还必须实时查 GitHub，避免因 300 条快照和过期数据误判为“没人提过”。展示来源链接、检索范围与不确定性；候选相似不等于同一个问题。
3. **准备提交**：例如“如果没有，就帮我提一个 Issue”。先查并展示可能重复项，再生成草稿。即使未找到候选，也只能表述“在本次可查范围内未找到明显重复”，不能证明全仓库不存在。用户可修改标题/正文，明确确认后才执行写入。

推荐用户路径：用户描述体验问题 → 助手给出简短建议并询问是否检查已有反馈 → 用户要求检查 → 展示历史候选和证据 → 用户选择已有 Issue 或“起草新 Issue” → 预览/编辑 → 点击“确认发布” → 返回真实 GitHub URL。用户也可直接要求“起草”，但发布前必须完成实时查重或明确标出查重不可用，绝不静默跳过。

## 最小架构

```text
会话（固定 repository_id）
  ├─ 对话规划：reply / clarify / retrieve / draft_issue
  ├─ 读工具：本地快照检索 + GitHub 当前仓库实时 Issue 搜索
  ├─ 草稿：标题、正文、依据、来源消息、状态（draft/approved/publishing/published/failed）
  └─ 写工具：独立的确认发布接口；只接受草稿 ID，不让模型直接调用
```

技术取舍：第一版沿用仓库现有的 GitHub REST 访问方式，在 `src/chat/` 下封装可替换的 GitHub 客户端；暂不引入 MCP。MCP 是以后替换/增加工具来源的选择，不是“提交 Issue”这个需求的前提。LLM 只负责理解意图与整理草稿；目标仓库、是否可发布、是否获得确认，由服务端程序判断。

## 数据与权限边界

- 草稿属于会话，绑定创建时的 `repository_id` 和规范化的 `owner/repo`；不能在请求体里临时换仓库。GitBugs-only 仓库没有可写的 GitHub 目标，只提供“复制草稿”，不显示发布按钮。
- 发布必须有服务端 GitHub 凭据及对应仓库创建 Issue 的权限；前端不输入、不保存 Token。未来公开上线时改为用户授权的 GitHub App/OAuth，不把开发者个人 Token 当所有用户的身份。
- 新建 Issue 公开可见。确认界面展示仓库、标题、完整正文和公开提示；用户可以取消或编辑。历史候选必须链接原文，不能把模型相似度写成“确定重复”。
- 只有显式点击“确认发布”才能写入 GitHub；聊天消息“好”“继续”不触发写操作。服务端对草稿及发布请求做幂等保护，避免双击重复创建。遇到网络超时且无法确认是否成功时，不自动重试 POST，先提示用户核验仓库。
- 第一版只创建 Issue，不自动关旧 Issue、加标签、指派成员、评论、创建 PR 或修改代码。PR 需要实际代码改动与验证，不能仅凭一次聊天生成。

## 实施步骤与验收门槛

### 1. 先修正意图分流的产品契约

在 `src/chat/prompts.py`、`src/chat/context.py`、`src/chat/service.py` 区分“一般咨询/查历史/起草 Issue”；保留现有故障事实提取，但不要让连续澄清的固定兜底覆盖与日志无关的问题。一般咨询允许直接回答，同时提出一个可选的下一步；不把建议自动转为工具调用。

验证：在 `tests/test_chat_service.py` 加入“该提 Issue 还是 PR”的两轮回归测试，断言不检索、不要求日志、能直接回答；运行 `.venv/bin/python -m pytest tests/test_chat_service.py -q` 全部通过。

### 2. 增加实时只读查重工具

在 `src/chat/` 新增 GitHub Issue 搜索适配层，只使用会话仓库对应的规范化 `owner/repo`，排除 PR，返回编号、标题、摘要、状态、URL、更新时间。复用本地检索结果作为第一批线索，发布前执行实时查询；如果 GitHub 限流/不可达，要在界面标记“实时查重未完成”，不能说“没有重复”。不要对整个仓库重建索引。

验证：用假 GitHub 客户端测试“找到相关 Issue”“没有命中”“API 失败”“结果包含 PR 需排除”四种情况；运行 `.venv/bin/python -m pytest tests/test_chat_service.py tests/test_github_repository_sync.py -q` 全部通过。

### 3. 草稿生成、持久化与编辑

在 `src/chat/models.py`、`src/chat/store.py` 增加会话内 IssueDraft。草稿只引用用户明确说过的实际/预期行为、复现步骤、环境和已有候选；未知项写“待补充”，不得编造版本或复现步骤。增加生成与编辑草稿 API，服务端检查长度和仓库归属。重启后草稿仍可读取，用户修改后以修改版本为发布内容。

验证：在 `tests/test_chat_store_persistence.py` 和 `tests/test_chat_api.py` 覆盖新建、编辑、重启恢复、跨会话访问拒绝；运行对应测试全部通过。

### 4. 显式确认后发布

在 `src/chat/` 封装 GitHub 创建 Issue 的写适配层，在 `api.py` 增加发布接口。服务端校验草稿属于当前会话、绑定的是 GitHub 仓库、凭据可用、用户提交的是当前草稿版本，并执行幂等保护。成功后保存 Issue URL/编号，页面不再显示“可提交”；失败保留草稿。对 401/403、仓库禁用 Issues、限流、网络超时分别给可理解的错误，不向页面泄露凭据或底层响应。

验证：用假写客户端在 `tests/test_chat_api.py` 覆盖未授权、未确认、重复点击、成功一次、失败保留草稿和超时不盲目重发；运行 `.venv/bin/python -m pytest tests/test_chat_api.py -q` 全部通过。

### 5. 对话 UI 与端到端体验

在 `frontend/src/components/ChatWorkspace.tsx` 添加“查看相似反馈”“起草 Issue”“编辑草稿”“确认发布”入口。候选和草稿可并排/顺序查看，但不能把检索结果当作已证实的重复。发布按钮仅在可写仓库、草稿可用时出现；确认处必须展示仓库全名、完整正文和公开提示。更新 `frontend/src/api.ts`、`frontend/src/types.ts`，视觉沿用现有组件与手册，不新造一套界面。

验证：`cd frontend && npm run lint` 与 `cd frontend && npm run build` 均通过；人工走通“普通咨询不触发检索”“查到相似项选择查看”“无明显命中起草并编辑”“取消不发布”“确认后出现真实 URL”五条路径。人工验收记录不得写成自动化测试已覆盖。

## 完成标准

- 一般提问得到直接回答，不再机械索要日志；不会产生 GitHub 写入。
- 用户可在绑定仓库内查历史；结果注明“本地快照”还是“GitHub 实时”，以及失败/覆盖边界。
- 用户可检查和修改 Issue 草稿；没有明确点击确认时，GitHub 创建调用次数为 0。
- 同一草稿成功发布后返回 URL；重复确认不会再创建第二条。
- 全量 `.venv/bin/python -m pytest tests/ -q`、前端 lint/build 通过；至少一轮人工真仓库演示使用用户自己控制的测试仓库，正式项目不作为测试目标。
- 修改前按用户既定流程新建 `codex/` 分支，分步提交，PR 描述写清“为什么做、用户路径、权限边界、验证结果”，经审阅后再合并；本计划本身不授权创建 PR 或发布 Issue。

## STOP 条件与后续

- 会话绑定的仓库无法可靠映射到唯一 `owner/repo`，停止发布路径，不猜目标仓库。
- GitHub 凭据权限模型无法区分本机单用户与未来多用户，停止上线设计；先只交付本机草稿和复制功能。
- 为实现第一版必须读取私有仓库、自动修改代码、自动提 PR 或自动发布未经预览的 Issue，停止并重新评审范围。
- 后续可扩展：评论已有 Issue、将新证据补到旧 Issue、读取 PR/Release Note、用户级 GitHub 授权和上线审计；均不属于本计划的 MVP。

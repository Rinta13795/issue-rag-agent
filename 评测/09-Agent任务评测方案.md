# 09 Agent 任务评测方案：从"搜得到"到"查得对"

> 项目从单次分诊（固定四节点）演进成对话调查 Agent（模型自己选工具）之后，
> 原有评测只能证明检索组件的召回能力。本文在不推翻 00-08 的前提下，把评测
> 扩成三层，补上 Agent 层。
>
> 状态：方案与脚本已就绪（`eval/benchmarks/agent_tasks.py`，评分逻辑有单测），
> **尚未用真实模型跑出结果**。没有跑出的数字，不写进简历和 README。

## 1. 三层评测体系

| 层 | 回答什么问题 | 方法 | 状态 | 复用了什么 |
| --- | --- | --- | --- | --- |
| L1 检索层 | 搜得到吗？ | GitBugs duplicate 标注自建 golden set，Recall / MRR / nDCG，确定性跑一次 | ✅ 已有 | 01 测试集设计、`eval/run_eval.py`、三处评测失效修正 |
| L2 行为层 | 边界守得住吗？ | 替身模型按剧本输出（故意编造引用、超预算、重复调用），断言程序的反应 | ✅ 已有 | 02 文档的"Assertion 测试"思路；`tests/test_chat_runtime.py` 等 |
| L3 任务层 | 查得对吗？会不会夸大？ | 真实 Issue + 自动标注的客观答案，真实模型端到端跑，重复 n 次 | 🟡 脚本就绪，待跑 | 02 文档的"端到端重复 n 次"、benchmark 的"verified 才入正式结果"约定 |

一句话：**L1 测工具，L2 测护栏，L3 测任务。** 三层对应 02 文档的两类测试：
L1、L3 是指标测试，L2 是断言测试。

## 2. L3 的核心：用 GitHub 时间线做免费的客观标注

Agent 回答"这个问题修了没有、怎么修的"，难点是没有标准答案。GitHub 本身就
带客观事实：Issue 有没有关闭、关联 PR 合没合并、PR 正文有没有写 `Fixes #12`。
这些都可以程序判定，不需要人工从零标注。

自动标注规则（`label_case`）：

| 类别 | 条件 | 用途 |
| --- | --- | --- |
| `fixed_merged` | Issue 已关闭，且至少一个**已合并**的关联 PR 用 fix / close / resolve `#编号` 指向它 | 正样本：Agent 应该找到这个 PR 并引用它 |
| `no_fix` | Issue 仍开着，时间线读取成功，且没有任何关联 PR | 负样本：Agent 不应该说"已修复" |
| 其他 | 证据不明确（PR 没合并、只是提到、时间线读取失败） | 自动跳过，不硬标 |

两条纪律（沿用 01 文档的标注原则）：

1. 自动标注只是草稿，状态为 `auto_labeled`；人工抽查无误改成 `verified` 才进入正式运行。
2. 负样本是"没有关联 PR"，不等于"真的没修"（可能有人修了但没关联），所以只用来测"是否夸大"，不用来测"是否答对"。

## 3. 指标

| 指标 | 定义 | 适用 | 方向 |
| --- | --- | --- | --- |
| 关键证据召回 `key_evidence_recall` | 修复案例中，Agent 是否真的调用 `read_pr` 读到了那个修复 PR | fixed_merged | 越高越好 |
| 引用命中 `citation_hit_rate` | 最终回答是否把修复 PR 作为依据引用 | fixed_merged | 越高越好 |
| 夸大率 `overclaim_rate` | 无修复案例中，回答却声称"已修复"（规则代理） | no_fix | 越低越好 |
| 裁判正确率 `judge_correct_rate` | 可选：LLM 裁判对照事实判 correct / incorrect / hedged | 全部 | 越高越好 |
| 完成率 / 反问率 | 正常给出回答 / 停下来向用户提问的比例 | 全部 | 诊断用 |
| 预算触顶率 | 跑满 6 次模型调用或工具预算的比例 | 全部 | 越低越好 |
| 成本 | 平均工具调用数、模型调用数、token、耗时 | 全部 | 越低越好 |
| 一致性 `consistency_rate` | 同一案例重复运行时，结论是否一致 | 重复 ≥2 次 | 越高越好 |

设计取舍：

- **规则指标为主，LLM 裁判为辅。** 证据召回、引用命中都基于实际工具调用和引用 ID，可复核；裁判只补"结论对不对"这一项，并且可以关掉。
- **引用合法性不用测。** 程序已经保证引用只能来自读过的资料（L2 有断言），L3 只测"引用对不对"。
- **夸大检测是关键词代理。** 会把"不能说已修复"这类否定句误判为夸大，绝对值只作参考，用于版本间相对比较。

## 4. 消融：用同一份代码切换工具集

沿用 retry benchmark 的做法：不切换 Git 提交，只在运行时收窄工具集，隔离"工具能力"这一个变量。

| 变体 | 可用工具 | 回答什么问题 |
| --- | --- | --- |
| `agent` | 全部 6 个 | 完整系统 |
| `no_pr_tools` | 去掉 `read_pr`、`search_prs` | 顺着 PR 追查这一步值多少 |
| `search_only` | 只有 `search_issues`（和 `ask_user`） | 近似"检索后直接回答"的传统 RAG 基线 |

`search_only` 对 `agent` 的差距，就是"为什么要做 Agent 而不是 RAG"的实验证据；
`no_pr_tools` 对 `agent` 的差距，说明收益具体来自哪个工具。

## 5. 随机性与可复现（沿用 00、04 文档）

- 每次运行用全新的内存会话、关闭长期记忆，避免案例之间互相污染。
- 端到端含模型随机性：每个案例默认重复 3 次，报均值，并单列一致性。
- 案例文件随结果一起保存路径；变体、重复次数、是否用裁判写进结果 JSON。
- 逐条保存原始记录（工具轨迹、回答、引用、token），补上 08 文档指出的"只有聚合、没有逐条明细"的缺口，可以事后做 badcase 归因。

## 6. 运行方式

```bash
# 1) 自动标注：从一个公开仓库扫描最近的 Issue，写成草稿
python -m eval.benchmarks.agent_tasks build --repository owner/repo --limit 40

# 2) 人工抽查 eval/benchmarks/data/agent_cases.json，无误的改成 "status": "verified"

# 3) 跑完整 Agent 和 RAG 基线，各重复 3 次
python -m eval.benchmarks.agent_tasks run --variant agent --label agent_v1
python -m eval.benchmarks.agent_tasks run --variant search_only --label rag_baseline

# 4) 对比
python -m eval.benchmarks.agent_tasks compare \
  eval_results/agent_tasks_rag_baseline.json eval_results/agent_tasks_agent_v1.json
```

需要：有效的 DeepSeek 密钥（run）、可访问 GitHub API（build 和 run，建议配置 `GITHUB_TOKEN`）。

## 7. 和旧评测的衔接

| 旧资产 | 在新体系里的位置 |
| --- | --- |
| golden set、`run_eval.py`、三处评测失效修正 | L1，原样保留，继续衡量 `search_issues` 背后的检索组件 |
| 02 文档的断言测试思路 | L2，已落在 Agent 行为测试里 |
| 02 文档的"端到端重复 n 次、报均值" | L3 的随机性控制 |
| benchmark 的"verified 才入正式结果""变体隔离" | L3 的标注纪律与消融方式 |
| 08 文档指出的"缺逐条明细" | L3 逐条保存原始记录 |

## 8. 迭代路线

1. **v1（当前）**：两类自动标注 + 规则指标 + 三个变体消融。
2. **v2**：加"PR 已合并但未发布"类（读取 release 信息），测 Agent 能否区分"合并"和"发布"。
3. **v3**：加多轮案例（用户补充"试了还是不行"），测记忆中"PR 说修复了"和"用户验证有效"的区分。
4. **v4**：抽 30 条人工复核裁判结论，校准 LLM 裁判的可信度。

## 9. 已知局限（面试时主动说）

- 自动标注依赖 PR 正文写了关闭关键词，覆盖的是"规范的仓库"，对随意的仓库会大量跳过。
- 负样本只能测"夸大"，测不了"漏答"。
- `search_only` 只是传统 RAG 的近似：系统提示词仍然提到其他工具。
- 目前还没有真实运行结果。

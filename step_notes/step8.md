# Step 8: 端到端测试

## 这一步做什么

编写端到端测试，验证 duplicate、similar、new 三类输入能通过完整 Agent 流程得到预期决策。

## 涉及的文件

- `tests/test_e2e.py`

## 输入输出

- 输入：三条人工构造的 issue 文本。
- 输出：`run_agent(issue)` 返回的决策结果，并通过 pytest 断言。

## 关键设计细节

用例 1：duplicate，应该高置信度命中。

```python
def test_e2e_duplicate():
    issue = """
    Title: React app crashes on hot reload after Vite upgrade
    Body: After upgrading to Vite 5.0, every hot reload causes:
    TypeError: Cannot read property 'children' of null
        at ReactReconciler.js:1234
    Steps: 1) npm run dev  2) edit any component  3) save
    Expected: hot reload works
    Actual: blank page + console error
    """
    result = run_agent(issue)
    assert result["decision"] == "duplicate"
    assert result["confidence"] >= 0.8
    assert any("vite" in r.lower() or "react" in r.lower() for r in result["related_issues"])
```

用例 2：similar。

```python
def test_e2e_similar():
    issue = """
    Title: Memory leak when scrolling list on mobile Safari
    Body: Heap grows ~5MB per minute when scrolling a virtualized list
    on iOS 17.2 Safari. Desktop Chrome is fine.
    """
    result = run_agent(issue)
    assert result["decision"] in ("similar", "duplicate")
    assert 0.5 <= result["confidence"] <= 0.95
```

用例 3：new。

```python
def test_e2e_new():
    issue = """
    Title: Feature request: quantum computing backend
    Body: Would love to add support for IBM Quantum / IonQ as a compute backend.
    Not a bug, just a feature proposal.
    """
    result = run_agent(issue)
    assert result["decision"] == "new"
    assert result["confidence"] >= 0.6
```

验证标准：

- 用例 1：单元测试通过，并从 log 确认 `retry_count <= 1`。
- 用例 2：测试通过，`reasoning` 包含“同组件不同 bug”类表述。
- 用例 3：测试通过，`related_issues` 为空。

端到端流程覆盖：`run_agent(issue)` -> `build_graph()` -> `query_analysis_node` -> `retrieval_node` -> `rerank_node` -> `decision_node` -> `should_retry` -> final result。

结果格式应与 Step 5 的 `run_agent` 一致：

```python
{
    "decision": final["decision"],
    "confidence": final["confidence"],
    "related_issues": final["related_issues"],
    "reasoning": final["reasoning"],
    "retry_count": final["retry_count"]
}
```

测试关注点：duplicate 类必须高置信度命中；similar 类允许 similar 或 duplicate，但置信度限制在 0.5 到 0.95；new 类必须 `decision == "new"` 且关联 issue 为空；日志中应能看到关键节点执行情况。

## 关键参数

- `CONFIDENCE_THRESHOLD = 0.7`
- `MAX_RETRIES = 2`
- `RERANK_TOP_K = 5`

## 依赖哪些已完成的模块

- Step 1：配置和依赖。
- Step 2：数据和索引。
- Step 3：检索模块。
- Step 4：重排序模块。
- Step 5：LangGraph 状态机和 `run_agent`。
- Step 6：Prompt。

## 完成标志

- [ ] `test_e2e_duplicate` 通过。
- [ ] `test_e2e_similar` 通过。
- [ ] `test_e2e_new` 通过。
- [ ] log 能确认 retry 行为符合预期。
- [ ] `related_issues` 和 `reasoning` 字段格式正确。

## 踩过的坑

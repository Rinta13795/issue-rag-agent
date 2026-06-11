# Step 7: 评估模块

## 这一步做什么

用 GitBugs 的 `duplicate_of` 构建 Golden Set，评估 Recall@10、Precision@5、MRR 和 **nDCG@10** 四类指标，并对比四路检索方案。

## 涉及的文件

- `eval/golden_set_builder.py`
- `eval/metrics.py`
- `eval/run_eval.py`

## 输入输出

- 输入：清洗后的 issues、retriever 实例、duplicate ground truth。
- 输出：各方法的 `recall@10`、`precision@5`、`mrr`、`nDCG@10` 平均值和对比表。

## 关键设计细节

Golden Set 构建：

```python
def build_golden_set(issues: list[dict], seed=42) -> tuple[list, list]:
    """提取所有 duplicate_of 非空的 issue 作为 query,80/20 划分"""
    golden = [
        {"id": i["id"], "text": f"{i['title']}\n{i['body']}",
         "duplicate_of": i["duplicate_of"], "project": i["project"]}
        for i in issues if i.get("duplicate_of")
    ]
    random.Random(seed).shuffle(golden)
    cut = int(len(golden) * 0.8)
    return golden[:cut], golden[cut:]
```

指标函数：

```python
def recall_at_k(retrieved_ids, golden_ids, k):
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(golden_ids)) / max(len(golden_ids), 1)

def precision_at_k(retrieved_ids, golden_ids, k):
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(golden_ids)) / k

def mrr(retrieved_ids, golden_ids):
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in golden_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids, golden_ids, k):
    """业界主流主指标。考虑 ranking 质量，比 MRR 更精细。"""
    import numpy as np
    gains = [1.0 if rid in golden_ids else 0.0 for rid in retrieved_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0
```

> nDCG 是 BEIR / MTEB / TREC 等公开 benchmark 的标准主指标。可用 `pytrec_eval` 替换自实现版本以保证与公开榜单可比。

评估主函数：

```python
def evaluate(retriever, queries, k=10) -> dict:
    metrics = {"recall@10": [], "precision@5": [], "mrr": [], "ndcg@10": []}
    for q in tqdm(queries, desc="Eval"):
        retrieved = retriever.search(q["text"], top_k=k * 2)
        ids = [d["id"] for d in retrieved]
        metrics["recall@10"].append(recall_at_k(ids, q["duplicate_of"], 10))
        metrics["precision@5"].append(precision_at_k(ids, q["duplicate_of"], 5))
        metrics["mrr"].append(mrr(ids, q["duplicate_of"]))
        metrics["ndcg@10"].append(ndcg_at_k(ids, q["duplicate_of"], 10))
    return {k: round(np.mean(v), 4) for k, v in metrics.items()}
```

`run_eval.py` 对比方案：

```python
results = {
    "Vector Only": evaluate(VectorRetriever(...), eval_q),
    "BM25 Only": evaluate(BM25Retriever(...), eval_q),
    "Hybrid (BM25+Vec+RRF)": evaluate(HybridRetriever(...), eval_q),
    "Hybrid + Rerank": evaluate(HybridWithRerank(...), eval_q),
}
print_table(results)
```

预期输出：

```text
+-----------------------+----------+-------------+--------+---------+
| Method                | Recall@10| Precision@5 | MRR    | nDCG@10 |
+-----------------------+----------+-------------+--------+---------+
| Vector Only           | 0.612    | 0.234       | 0.452  | 0.523   |
| BM25 Only             | 0.553    | 0.198       | 0.398  | 0.471   |
| Hybrid (BM25+Vec+RRF) | 0.741    | 0.412       | 0.612  | 0.682   |
| Hybrid + Rerank       | 0.823    | 0.658       | 0.781  | 0.812   |
+-----------------------+----------+-------------+--------+---------+
```

评测数据集设计：

- 从全集筛选 `resolution=duplicate` 的 issue，约 6-8%。
- 每条 query 的 `duplicate_of` 作为 ground truth。
- 80/20 划分 eval set 和 held-out test。
- Test corpus 是全部 15万 issue。

> **未来工作**：计划扩展到 BEIR CQADupStack 公开 benchmark 做横评，验证方法在 StackExchange 类社区问答场景的泛化性。本期 MVP 不实现。

## 关键参数

- `seed = 42`
- `k = 10`
- `precision@5`、`nDCG@10`
- `top_k = k * 2`

## 依赖哪些已完成的模块

- Step 2：清洗后的 issues 和 ground truth。
- Step 3：Vector、BM25、Hybrid 检索器。
- Step 4：Hybrid + Rerank 对比方案。

## 完成标志

- [ ] `build_golden_set` 完成 80/20 划分。
- [ ] `recall_at_k`、`precision_at_k`、`mrr`、`ndcg_at_k` 实现。
- [ ] `evaluate` 能遍历 queries 并返回四项平均指标。
- [ ] `run_eval.py` 输出四组方法对比表（含 nDCG@10）。

## 踩过的坑


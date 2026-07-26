# Step 4: 重排序模块

## 这一步做什么

用 BGE Cross-Encoder 对混合检索 Top30 做精排，保留 Top5，并支持动态阈值扩展。

## 涉及的文件

- `src/reranker.py`

## 输入输出

- 输入：`query: str`，`docs: list[dict]`，通常来自 Step 3 的 Hybrid Retrieval Top30。
- 输出：按 `rerank_score` 降序排列的 `list[dict]`，默认 Top5，分差小于阈值时最多扩展到 Top10。

## 关键设计细节

重排序使用 `sentence_transformers.CrossEncoder`，模型为 `BAAI/bge-reranker-base`。文档中标注该模型约 278M，CPU 对 30 个 doc 约 600-900ms。

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL, device=EMBED_DEVICE)

    def rerank(self, query: str, docs: list[dict],
               top_k: int = 5, dynamic_threshold: float = 0.05) -> list[dict]:
        if not docs:
            return []
        pairs = [(query, f"{d['title']}\n{d['body']}"[:800]) for d in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        sorted_docs = sorted(docs, key=lambda x: -x["rerank_score"])

        cut = top_k
        while cut < len(sorted_docs) and cut < top_k + 5:
            if sorted_docs[cut-1]["rerank_score"] - sorted_docs[cut]["rerank_score"] < dynamic_threshold:
                cut += 1
            else:
                break
        return sorted_docs[:cut]
```

关键逻辑：

- 空文档直接返回空列表。
- pair 格式是 `(query, title + body)`。
- 文档内容截断到 800 字符，避免 BERT input 过长。
- 每个 doc 写入 `rerank_score: float`。
- 默认保留 Top5。
- 如果第 `top_k` 和第 `top_k+1` 分差小于 `DYNAMIC_THRESHOLD_DIFF`，继续扩展。
- 动态扩展最多 `top_k + 5`，即默认最多 Top10。

为什么只精排 Top5：

- LLM context 成本：5 issue 约 3500 tok/请求。
- 超过 8 个文档时容易 Lost in Middle。
- 实务上 1-2 个 duplicate 已足够判断。

为什么不能召回 K=200 后精排：

- Cross-Encoder 是 O(K)，30 个 doc 已约 500ms，200 个会到 3s+。
- 低相关 doc 进入 reranker 会让 score 分布失真。

## 关键参数

- `RERANKER_MODEL = "BAAI/bge-reranker-base"`
- `EMBED_DEVICE = "cpu"`，也可为 `"cuda"` / `"mps"`
- `RERANK_TOP_K = 5`
- `DYNAMIC_THRESHOLD_DIFF = 0.05`

## 依赖哪些已完成的模块

- Step 1：读取 reranker 模型、设备和阈值。
- Step 3：输入来自混合检索 Top30。

## 完成标志

- [ ] `Reranker.__init__` 正确加载 CrossEncoder。
- [ ] `rerank` 支持空输入。
- [ ] `rerank` 给每个 doc 写入 `rerank_score`。
- [ ] `rerank` 按分数降序返回 Top5 或动态扩展结果。
- [ ] 文档截断到 800 字符。

## 踩过的坑


# Step 3: 检索模块

## 这一步做什么

实现向量检索、BM25 检索和 RRF 融合，输出混合召回 Top-K 文档。

## 涉及的文件

- `src/retrievers/vector_retriever.py`
- `src/retrievers/bm25_retriever.py`
- `src/retrievers/hybrid_retriever.py`
- `src/retrievers/__init__.py`

## 输入输出

- 输入：query 字符串、TopK、可选 metadata filter、Step 2 构建好的 ChromaDB 和 `bm25.pkl`。
- 输出：`list[dict]`，元素包含 `id`、`title`、`body`、`score`，融合后额外包含 `rrf_score`。

## 关键设计细节

向量检索使用 ChromaDB。Chroma 返回 cosine distance，越小越相关；转为 `1 - distance` 便于 debug。向量库按 chunk 存储，返回时聚合到 issue 粒度，同 issue 多 chunk 取最高分。

```python
class VectorRetriever:
    def __init__(self, vectorstore):
        self.vs = vectorstore

    def search(self, query: str, top_k: int = 30,
               filter_dict: dict | None = None) -> list[dict]:
        results = self.vs.similarity_search_with_score(
            query, k=top_k * 2, filter=filter_dict
        )
        issue_best = {}
        for doc, dist in results:
            iid = doc.metadata["issue_id"]
            sim = 1 - dist
            if iid not in issue_best or sim > issue_best[iid]["score"]:
                issue_best[iid] = {
                    "id": iid,
                    "title": doc.metadata.get("title", ""),
                    "body": doc.page_content,
                    "score": sim,
                    "metadata": dict(doc.metadata)
                }
        return sorted(issue_best.values(), key=lambda x: -x["score"])[:top_k]
```

BM25 检索使用 `rank-bm25`。query 先走 Step 2 的 `tokenize`，再用 `np.argsort(scores)[::-1]` 取 TopK，只保留 `score > 0`。

```python
class BM25Retriever:
    def __init__(self, bm25_pkl_path: str, issues_dict: dict):
        data = pickle.load(open(bm25_pkl_path, "rb"))
        self.bm25 = data["bm25"]
        self.ids = data["ids"]
        self.issues = issues_dict

    def search(self, query: str, top_k: int = 30) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id": self.ids[i],
                "title": self.issues[self.ids[i]]["title"],
                "body": self.issues[self.ids[i]]["body"][:500],
                "score": float(scores[i])
            }
            for i in top_idx if scores[i] > 0
        ]
```

RRF 用排名融合，不做线性加权，避免 BM25 分数和向量分数量纲不一致：

```python
def rrf_merge(rankings: list[list[dict]], k: int = 60, top_k: int = 30) -> list[dict]:
    rrf_scores = defaultdict(float)
    id_to_doc = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            iid = doc["id"]
            rrf_scores[iid] += 1.0 / (k + rank)
            if iid not in id_to_doc:
                id_to_doc[iid] = doc
    sorted_pairs = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    result = []
    for iid, rrf_s in sorted_pairs:
        doc = id_to_doc[iid].copy()
        doc["rrf_score"] = rrf_s
        result.append(doc)
    return result
```

混合检索：向量检索可以用 component metadata filter；BM25 始终全量搜索，防止跨组件 issue 被过滤掉。

`HybridRetriever.search(query, top_k=30, filter_dict=None)`：先调用 `VectorRetriever.search` 和 `BM25Retriever.search`，再 `rrf_merge([vec, bm25], k=RRF_K, top_k=top_k)`。

## 关键参数

- `VECTOR_TOP_K = 30`
- `BM25_TOP_K = 30`
- `RRF_K = 60`
- `HYBRID_TOP_K = 30`

## 依赖哪些已完成的模块

- Step 1：读取检索 TopK 和 RRF 参数。
- Step 2：依赖 ChromaDB、`bm25.pkl`、`issues_dict`、`tokenize`。

## 完成标志

- [ ] `VectorRetriever.search` 实现 chunk 到 issue 聚合。
- [ ] `BM25Retriever.search` 实现分词、打分、TopK 返回。
- [ ] `rrf_merge` 实现多路排名融合。
- [ ] `HybridRetriever.search` 返回融合结果。

## 踩过的坑

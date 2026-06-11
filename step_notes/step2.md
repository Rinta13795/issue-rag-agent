# Step 2: 数据准备与入库

## 这一步做什么

加载 GitBugs issue，完成清洗、结构化切分、Embedding 入 ChromaDB，并构建 BM25 索引。

## 涉及的文件

- `data/raw/gitbugs_dump.jsonl`
- `data/processed/issues.jsonl`
- `data/mock/mock_issues.json`
- `src/data_loader.py`
- `src/indexer.py`
- `chroma_db/`
- `bm25.pkl`

## 输入输出

- 输入：GitBugs 15万+ raw issues，或下载失败时的 mock issues。
- 输出：清洗后的 `issues.jsonl`、ChromaDB 向量库、`bm25.pkl` 关键词索引。

## 关键设计细节

数据获取顺序：

1. `datasets.load_dataset("gitbugs/dedup-2025", split="train")`
2. HuggingFace direct download
3. Fallback 到 `data/mock/mock_issues.json`

标准化后的 issue 格式：

```python
{
    "id": str,
    "title": str,
    "body": str,
    "labels": list[str],
    "component": str | None,
    "status": str,
    "resolution": str | None,
    "duplicate_of": list[str],
    "project": str,
    "created_at": str,
}
```

清洗规则：

清洗规则：BeautifulSoup 去 HTML；移除多余空白和 HTML 实体；过滤 `len(body) < 20` 的噪声；过滤 `status=closed` 但 `resolution=invalid/wontfix` 的 issue；正则提取 `Traceback`、`Error:`、`at xxx.{java,py,ts,tsx,js}:NN` 等 error_log。

mock 数据要求：至少 50 条，包含 10 组 duplicate，每组 2-3 条，覆盖 vscode、react、kubernetes，中英文混合；每组 duplicate 要“描述差异大但本质相同”。

切分主函数：

```python
def chunk_issue(issue: dict) -> list[Document]:
    base_meta = {
        "issue_id": issue["id"], "title": issue["title"],
        "labels": ",".join(issue["labels"]),
        "component": issue.get("component") or "",
        "status": issue["status"], "project": issue["project"]
    }
    chunks = [Document(
        page_content=f"Title: {issue['title']}",
        metadata={**base_meta, "chunk_type": "title"}
    )]
    if err := extract_error_log(issue["body"]):
        chunks.append(Document(
            page_content=f"Error: {err}",
            metadata={**base_meta, "chunk_type": "error_log"}
        ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS
    )
    for sub in splitter.split_text(remove_error_log(issue["body"])):
        if len(sub.strip()) < 30:
            continue
        chunks.append(Document(page_content=sub, metadata={**base_meta, "chunk_type": "body"}))
    return chunks
```

Embedding 和 ChromaDB：用 `langchain_huggingface.HuggingFaceEmbeddings`，`model_name=EMBED_MODEL`，`device=EMBED_DEVICE`，`encode_kwargs={"normalize_embeddings": True}`；用 `langchain_chroma.Chroma` 写入 `CHROMA_COLLECTION` 和 `CHROMA_PERSIST_DIR`，按 500 条分批 `add_documents` 避免 OOM。

BM25 用 issue 粒度，不用 chunk 粒度；BM25 对长文本更稳：

```python
def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in jieba.cut(text):
        tokens.extend([t.lower() for t in tok.split() if t.strip()])
    return tokens

bm25_corpus = [tokenize(f"{i['title']} {i['body']}") for i in all_issues]
bm25 = BM25Okapi(bm25_corpus)  # k1=1.5, b=0.75 默认值
pickle.dump({"bm25": bm25, "ids": [i["id"] for i in all_issues]}, open("bm25.pkl", "wb"))
```
入库后抽样自召回：用 sample title 查 ChromaDB，检查 sample id 是否在 Top5。

## 关键参数

- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_SEPARATORS`
- `EMBED_MODEL`, `EMBED_DEVICE`
- `CHROMA_COLLECTION`, `CHROMA_PERSIST_DIR`

## 依赖哪些已完成的模块

- Step 1：依赖 `config.py` 参数和 `requirements.txt` 依赖。

## 完成标志

- [ ] `data_loader.py` 能加载、清洗、标准化 issue。
- [ ] `mock_issues.json` 满足数量、项目、duplicate 组要求。
- [ ] `chunk_issue(issue)` 输出带完整 metadata 的 `Document` 列表。
- [ ] ChromaDB 分批写入成功。
- [ ] `bm25.pkl` 持久化成功。
- [ ] 抽样自召回验证通过。

## 踩过的坑

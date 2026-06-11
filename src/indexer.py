"""构建 Issue RAG Agent 离线索引的模块。

本文件依赖 data_loader.py 输出的标准 issue 列表。
职责包括切 chunk、向量化、写入 ChromaDB、构建 BM25；当前先实现 error_log 拆分工具函数。

————在还基础债务了哎呀
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

from config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL,
    INDEX_BATCH_SIZE,
)


# Python Traceback 正则：识别以 "Traceback (most recent call last):" 开头的错误段落。————写正则表达式
TRACEBACK_PATTERN = r"Traceback \(most recent call last\):"

# Error 信息正则：识别 "Error:"、"TypeError:"、"ValueError:" 等以 Error 结尾的错误信息。
ERROR_MESSAGE_PATTERN = r"\b(?:[A-Za-z_][A-Za-z0-9_]*Error|Error):"

# 堆栈行正则：识别 "at xxx.java:NN"、"at xxx.py:NN"、"at xxx.ts:NN" 等堆栈位置。
STACK_LINE_PATTERN = r"^\s*at\s+\S+\.(?:java|py|tsx|ts|js):\d+"

# 共享 error 正则（上面的正则为这个服务的）：extract_error_log 和 remove_error_log 必须使用同一套规则，避免拆分逻辑不一致。
'''
解析成正则对象使用
'''
ERROR_LOG_PATTERN = re.compile(
    f"{TRACEBACK_PATTERN}|{ERROR_MESSAGE_PATTERN}|{STACK_LINE_PATTERN}",
    # re.MULTILINE 让堆栈行正则里的 ^ 可以匹配每一行的开头。
    re.MULTILINE,
)


def extract_error_log(body: str) -> str:
    """输入清洗后的 body 字符串，输出从第一个 error 匹配位置到文本结束的错误文本。"""
    # 使用共享正则查找第一个错误位置，保证和 remove_error_log 的判断完全一致。————找第一个错误位置！！！

    """扫描是否以这个符合这一格式的正则，返回记录位置，如果没有这个位置，就返回None"""
    match = ERROR_LOG_PATTERN.search(body)

    # 没有匹配到错误信息时，按约定返回空字符串。
    if not match:
        return ""

    # 从第一个 error （匹配位置截到文本结束！！），保留完整错误上下文用于 error_log chunk。
    return body[match.start():].strip()


def remove_error_log(body: str) -> str:
    """输入清洗后的 body 字符串，输出删除 error 部分后的剩余 body。"""
    # 使用共享正则查找第一个错误位置，保证和 extract_error_log 的判断完全一致。
    match = ERROR_LOG_PATTERN.search(body)

    # 没有匹配到错误信息时，按约定返回原始 body，不做任何修改。
    if not match:
        return body

    # 删除从第一个 error 匹配位置到文本结束的内容，剩余部分用于 body chunk。————error一般都是在最后的
    return body[:match.start()].strip()

"""我理解了，这部分将error提取出来的逻辑是，使用正则匹配对应所有的错误字段为首的位置，将error和body分离开，接着使用2个辅助函数分别将2部分文本提取出来

————用正则找到 body 里第一个 error 的位置，把 body 切成两半，error 单独存一个 
  ▎ chunk，剩余描述存 body chunk，目的是让 error 信号不被稀释。
"""

def chunk_issue(issue: dict) -> list[Document]:
    """输入 clean 后的标准 issue，输出 title/error_log/body 三类 Document chunk 列表。"""
    # Document 和 TextSplitter 只在切 chunk 时需要，放到函数内导入，避免 BM25 仅用 tokenize 时被向量库依赖卡住。
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 所有 chunk 共享同一份基础元数据，便于后续按 issue_id、component、status 等字段过滤或聚合。
    base_meta = {
        "issue_id": issue["id"],
        "title": issue["title"],
        "labels": ",".join(issue["labels"]),
        "component": issue.get("component") or "",
        "status": issue["status"],
        "project": issue["project"],
    }

    chunks: list[Document] = []

    # 1. 每条 issue 必须生成 title chunk，保证短标题也能被单独检索到。
    """title要当元数据也当chunk，附加每一个内容的元数据(元数据不向量化)"""
    chunks.append(
        Document(
            page_content=f"Title: {issue['title']}",
            metadata={**base_meta, "chunk_type": "title"},
        )
    )

    # 2. 如果 body 中存在错误日志，则单独生成 error_log chunk，保留错误码和堆栈等精确信号。
    err = extract_error_log(issue["body"])
    if err:
        chunks.append(
            Document(
                page_content=f"Error: {err}",
                metadata={**base_meta, "chunk_type": "error_log"},
            )
        )

    # 3. body chunk 先去掉 error 部分，避免错误日志重复进入 body 语义切分。
    body_without_error = remove_error_log(issue["body"])

    # 使用 config.py 中的切分参数，避免在函数里硬编码 chunk 大小和分隔符。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,#递归切分这里
    )

    # 对剩余 body 做递归切分，过短碎片不建 chunk，减少向量库噪声。
    for sub in splitter.split_text(body_without_error):
        if len(sub.strip()) < 30:
            continue
        chunks.append(
            Document(
                page_content=sub,
                metadata={**base_meta, "chunk_type": "body"},
            )
        )

    return chunks


def load_embeddings() -> HuggingFaceEmbeddings:
    """输入无，输出本地 BGE Embedding 模型实例，用于 ChromaDB 向量化。"""
    # HuggingFaceEmbeddings 只在真正构建向量索引时需要，避免导入 indexer 时提前加载重依赖。
    from langchain_huggingface import HuggingFaceEmbeddings

    # 从 config.py 读取模型名和运行设备，避免在索引逻辑中硬编码。
    logger.info("加载本地 Embedding 模型：{} ({})", EMBED_MODEL, EMBED_DEVICE)

    # normalize_embeddings=True 后可以稳定使用 cosine 相似度做向量检索。
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_chroma_index(all_issues: list[dict]) -> tuple[Chroma, list[Document]]:
    """输入清洗后的标准 issue 列表，输出写入完成的 ChromaDB 实例和全部 chunk。"""
    # Chroma 只在真正写入向量库时需要，BM25 检索导入 tokenize 时不应依赖它。
    from langchain_chroma import Chroma

    # 逐条 issue 调用 chunk_issue，ChromaDB 使用 chunk 粒度保存语义向量。
    all_chunks: list[Document] = []
    """每次处理一条issue，慢慢全部处理完,变成一整个all_chunks"""
    for issue in all_issues:
        all_chunks.extend(chunk_issue(issue))

    logger.info("chunk 构建完成：{} issues -> {} chunks", len(all_issues), len(all_chunks))

    # 初始化本地 BGE Embedding 和 ChromaDB 持久化 collection。包含表名+文件夹位置
    embeddings = load_embeddings()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # 按 config.py 中的批大小分批写入，避免一次性 add_documents 导致内存压力过大。————每次500个500个读
    for start in range(0, len(all_chunks), INDEX_BATCH_SIZE):
        batch = all_chunks[start:start + INDEX_BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info("ChromaDB 写入进度：{}/{} chunks", min(start + len(batch), len(all_chunks)), len(all_chunks))

    # 入库后自召回验证：用第一条 issue 的 title 查 Top-5，必须能召回自身 issue_id。
    if all_issues:
        sample = all_issues[0]
        hits = vectorstore.similarity_search(sample["title"], k=5)
        hit_ids = [doc.metadata.get("issue_id") for doc in hits]
        assert sample["id"] in hit_ids, "自召回失败"
        print(f"✅ 入库验证通过: {len(all_chunks)} chunks, {len(all_issues)} issues")

    return vectorstore, all_chunks


def tokenize(text: str) -> list[str]:
    """输入中英混合文本，输出 jieba + 空格细分 + 小写后的 token 列表。"""
    tokens: list[str] = []

    # 先用 jieba 处理中文分词，再对每个 token 按空格细分，兼容英文短语和代码片段。
    for token in jieba.cut(text):
        # 每个子 token 统一转小写，减少大小写差异对 BM25 的影响。
        tokens.extend(part.lower() for part in token.split() if part.strip())

    return tokens


def build_bm25_index(all_issues: list[dict]) -> BM25Okapi:
    """输入清洗后的标准 issue 列表，输出 issue 粒度 BM25 索引并持久化到磁盘。"""
    # BM25 使用 issue 粒度：title + body 合并后分词，避免 chunk 切散导致关键词共现丢失。
    bm25_corpus = [
        #使用列表推导式不断添加各个issue词语“列表”进这个列表里
        tokenize(f"{issue['title']} {issue['body']}")
        for issue in all_issues
    ]
    #输出各种词对应的数据比如vscode_issue1

    # 用 rank-bm25 构建倒排检索索引，k1=1.5、b=0.75 使用库默认值。
    bm25 = BM25Okapi(bm25_corpus)
    bm25_ids = [issue["id"] for issue in all_issues]

    # 将 BM25 索引和 issue id 顺序一起保存，检索时才能把分数映射回 issue。
    output_path = Path(BM25_INDEX_PATH)
    with output_path.open("wb") as file:
        pickle.dump({"bm25": bm25, "ids": bm25_ids}, file)

    logger.info("BM25 索引构建完成：{} issues -> {}", len(all_issues), output_path)
    return bm25

'''关键词索引和向量化结合到一个方法里面'''
def build_indexes(all_issues: list[dict]) -> tuple[Chroma, BM25Okapi]:
    """输入清洗后的标准 issue 列表，输出 ChromaDB 向量库和 BM25 索引。"""
    # 第一块：加载本地 Embedding，并把 chunk 粒度 Document 写入 ChromaDB。
    vectorstore, _ = build_chroma_index(all_issues)

    # 第二块：按 issue 粒度构建 BM25 索引并保存到 bm25.pkl。
    bm25 = build_bm25_index(all_issues)

    return vectorstore, bm25

"""项目全局配置：集中管理模型、路径、检索、重排和 Agent 超参数。"""

import os

from dotenv import load_dotenv


load_dotenv()


# DeepSeek API Key：从 .env 文件读取，用于调用 DeepSeek LLM。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# DeepSeek API Base URL：给 langchain_openai.ChatOpenAI 的 base_url 使用。
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek 模型名称：用于 Query Analysis 和 Decision 节点。
DEEPSEEK_MODEL = "deepseek-chat"

# LLM 温度：决策类任务使用低温度，提高输出一致性。
LLM_TEMPERATURE = 0.1

# LLM 最大输出 token 数：限制单次模型输出长度。
LLM_MAX_TOKENS = 1024


# Embedding 模型名称：本地 HuggingFace BGE 模型，不调用 API。
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# Embedding 运行设备：Mac 可用 cpu 或 mps，GPU 环境可用 cuda。
EMBED_DEVICE = "cpu"

# Embedding 向量维度：bge-small-zh-v1.5 输出 512 维向量。
EMBED_DIM = 512


# ChromaDB 持久化目录：向量库本地保存路径。
CHROMA_PERSIST_DIR = "./chroma_db"

# ChromaDB collection 名称：存放 issue chunk 的集合名。
CHROMA_COLLECTION = "issues"

# ChromaDB 批量写入大小：每批 500 条 Document，避免一次性写入导致 OOM。
INDEX_BATCH_SIZE = 500

# BM25 索引保存路径：rank-bm25 构建完成后用 pickle 持久化到该文件。
BM25_INDEX_PATH = "bm25.pkl"


# 文本切分长度：适合 issue body 段落，避免语义过散或向量稀释。
CHUNK_SIZE = 500

# 文本切分重叠长度：保留段落边界上下文。
CHUNK_OVERLAP = 80

# 文本切分分隔符：用于 RecursiveCharacterTextSplitter 递归切分。
CHUNK_SEPARATORS = ["\n\n", "\n", "。", ".", "!", "?", ";", ";"]


# 向量检索 TopK：单路召回数量，保证召回覆盖。
VECTOR_TOP_K = 30

# 向量检索 chunk 召回倍率：先多召回 top_k 的 2 倍 chunk，再聚合到 issue 粒度。————(只要 issue #101 有任意一个 chunk 被召回，issue #101 就算召回了。)
VECTOR_CHUNK_FETCH_MULTIPLIER = 2

# BM25 检索 TopK：关键词检索召回数量，和向量检索并行。
BM25_TOP_K = 30

# RRF 平滑常数：Cormack 2009 经验值，用于排名融合。
RRF_K = 60

# 混合检索 TopK：RRF 融合后保留的候选数量。
HYBRID_TOP_K = 30


# 重排序模型名称：Cross-Encoder 精排模型。
RERANKER_MODEL = "BAAI/bge-reranker-base"

# 重排序保留 TopK：控制进入 LLM Decision 的候选 issue 数量。
RERANK_TOP_K = 5

# 重排序输入文档最大字符数：CrossEncoder 底层 BERT 有长度限制，截断 title+body 避免超长。
RERANK_DOC_MAX_CHARS = 800

# 动态扩展阈值：第 top_k 和 top_k+1 分差小于该值时扩展候选。
DYNAMIC_THRESHOLD_DIFF = 0.05

# 动态扩展最多额外保留数量：Top5 和后续分数很接近时最多额外扩展 5 条。
RERANK_MAX_EXTRA_DOCS = 5


# 置信度阈值：低于该值触发 Query Analysis 重写循环。
CONFIDENCE_THRESHOLD = 0.7

# 最大重试次数：最多重写 2 次，防止死循环并控制延迟。
MAX_RETRIES = 2

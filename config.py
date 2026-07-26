"""项目全局配置：集中管理模型、路径、检索、重排和 Agent 超参数。"""

import os

from dotenv import load_dotenv


# override=True：项目 .env 优先于 shell 继承的同名环境变量。
# 实际踩坑：shell 里残留失效的 DEEPSEEK_API_KEY 时，默认行为会静默用旧 key 导致 401。
load_dotenv(override=True)


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

# LLM 单次请求超时秒数：避免网络异常时节点无限阻塞。
LLM_TIMEOUT = 60

# LLM 客户端自动重试次数：针对超时、限流等瞬时错误。
LLM_MAX_RETRIES = 2


# Embedding 模型名称：本地 HuggingFace 英文模型，不调用 API。数据集全为英文 issue，改用英文 all-MiniLM-L6-v2。
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding 运行设备：Mac 可用 cpu 或 mps，GPU 环境可用 cuda。
EMBED_DEVICE = "cpu"

# Embedding 向量维度：all-MiniLM-L6-v2 输出 384 维向量。
EMBED_DIM = 384


# ChromaDB 持久化目录：向量库本地保存路径。英文索引重建到新目录，旧 zh 索引 ./chroma_db 保留可回滚。
CHROMA_PERSIST_DIR = "./chroma_db_en"

# ChromaDB collection 名称：存放 issue chunk 的集合名。
CHROMA_COLLECTION = "issues"

# ChromaDB 批量写入大小：每批 500 条 Document，避免一次性写入导致 OOM。
INDEX_BATCH_SIZE = 500

# BM25 索引保存路径：rank-bm25 构建完成后用 pickle 持久化到该文件。
BM25_INDEX_PATH = "bm25.pkl"

# Issue 正文文档库路径：issue_id -> {title, body} 映射，用于给只被 BM25 召回的候选补齐正文证据。
DOCSTORE_PATH = "docstore.pkl"

# 文档库正文截断长度：Reranker 只看前 800 字符、Decision 只看前几百字符，1200 字符已覆盖两者需求。
DOCSTORE_BODY_MAX_CHARS = 1200


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

# BM25 最低分数：0 分表示 query 词一个都没命中，属于无效候选，不进入 RRF 融合。
BM25_MIN_SCORE = 0.0

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

# Decision 允许的类别枚举：LLM 输出不在其中时按解析失败处理。
VALID_DECISIONS = ("duplicate", "similar", "new")

# Decision prompt 中每个候选 body 的展示字符数：候选正文已由 docstore 补齐，可以给模型更多证据。
DECISION_BODY_PREVIEW_CHARS = 400

# 重试 prompt 中展示的上一轮 Top 候选数量：让 Query Analysis 知道上轮检索到了什么。
RETRY_TOP_CANDIDATES = 3


# 评估模块随机种子：固定 80/20 划分结果，保证实验可复现。
EVAL_SEED = 42

# Golden set 保存路径：build_golden_set 输出的 JSON 文件位置。
EVAL_GOLDEN_PATH = "eval/data/golden_set.json"

# 评估结果保存目录：run_eval.py 输出的各配置指标 JSON 文件存放位置。
EVAL_RESULTS_DIR = "eval_results"

# 评估抽样数量：held-out test set 全量约 2600 条，全跑耗时过长；默认抽样后仍可给出稳定对比。
EVAL_SAMPLE_SIZE = 200

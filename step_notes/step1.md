# Step 1: 环境与配置

## 这一步做什么

建立项目依赖、环境变量模板和全局配置文件，集中管理模型名、路径和超参数。

## 涉及的文件

- `requirements.txt`
- `config.py`
- `.env.example`

## 输入输出

- 输入：设计文档 5.2 节的依赖列表、模型配置、检索参数、Agent 参数。
- 输出：可复用的项目配置入口，后续所有模块只从 `config.py` 读取参数。

## 关键设计细节

依赖来自原文 `requirements.txt`：

```text
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
langchain-chroma>=0.1.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
rank-bm25>=0.2.2
jieba>=0.42.1
openai>=1.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0
datasets>=2.20.0
huggingface-hub>=0.24.0
tqdm>=4.66.0
loguru>=0.7.0
beautifulsoup4>=4.12.0
numpy<2.0
pandas>=2.0.0
pytest>=8.0.0
```

`config.py` 基础结构：

```python
import os
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DEVICE = "cpu"  # 或 "cuda" / "mps"
EMBED_DIM = 512

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "issues"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNK_SEPARATORS = ["\n\n", "\n", "。", ".", "!", "?", ";", ";"]

VECTOR_TOP_K = 30
BM25_TOP_K = 30
RRF_K = 60
HYBRID_TOP_K = 30

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_K = 5
DYNAMIC_THRESHOLD_DIFF = 0.05

CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES = 2
```

`.env.example`：

```text
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxx
HF_HUB_OFFLINE=0
LANGCHAIN_TRACING_V2=false
```

LLM 请求重点是 `messages`，API Key 在请求头；DeepSeek 调用后续用 `ChatOpenAI(base_url=..., api_key=...)`。

## 关键参数

- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`
- `EMBED_MODEL`, `EMBED_DEVICE`, `EMBED_DIM`
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_SEPARATORS`
- `VECTOR_TOP_K`, `BM25_TOP_K`, `RRF_K`, `HYBRID_TOP_K`
- `RERANKER_MODEL`, `RERANK_TOP_K`, `DYNAMIC_THRESHOLD_DIFF`
- `CONFIDENCE_THRESHOLD`, `MAX_RETRIES`

## 依赖哪些已完成的模块

无。Step 1 是后续所有模块的基础。

## 完成标志

- [ ] `requirements.txt` 写入完整依赖。
- [ ] `config.py` 写入所有模型、路径、检索、重排、Agent 参数。
- [ ] `.env.example` 写入 DeepSeek 和 LangChain 环境变量模板。
- [ ] 后续模块没有硬编码超参数。

## 踩过的坑


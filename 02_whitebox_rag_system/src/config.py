# -*- coding: utf-8 -*-
"""
配置加载与环境变量管理
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 尝试加载 .env 文件
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        # 手动简易解析 .env
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

class Config:
    BASE_DIR = BASE_DIR
    CORPUS_DIR = BASE_DIR / "data" / "corpus"
    BENCHMARKS_DIR = BASE_DIR / "benchmarks"
    CACHE_DIR = BASE_DIR / "data" / "cache"

    # DeepSeek LLM
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))

    # Embedding API
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()

    # Reranker API
    RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "").strip()
    RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()

    # 超参数
    RETRIEVAL_TOP_K_SPARSE = int(os.getenv("RETRIEVAL_TOP_K_SPARSE", "10"))
    RETRIEVAL_TOP_K_DENSE = int(os.getenv("RETRIEVAL_TOP_K_DENSE", "10"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

cfg = Config()

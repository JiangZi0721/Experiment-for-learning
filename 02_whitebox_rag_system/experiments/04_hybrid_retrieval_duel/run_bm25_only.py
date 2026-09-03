# -*- coding: utf-8 -*-
"""
实验四：双路对抗实验 - 纯稀疏检索 (BM25 Only)
运行本脚本：仅使用 Okapi BM25 进行词频与逆文档频率匹配，观察其在精准词与词汇鸿沟场景下的表现。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import cfg
from src.chunker import StructuralMarkdownChunker
from src.retrieval.bm25_retriever import BM25Retriever

def run_bm25(query: str, top_k: int = 5):
    print("\n" + "=" * 70)
    print(f"  [实验四·单路分支] 纯稀疏检索 (BM25 Only)")
    print(f"  查询词: {query}")
    print("=" * 70)

    chunks = StructuralMarkdownChunker().chunk_corpus(cfg.CORPUS_DIR)
    bm25 = BM25Retriever(); bm25.fit(chunks)

    res = bm25.retrieve(query, top_k=top_k)
    print(f"\n[BM25 检索结果 Top-{top_k}]:")
    for r in res:
        print(f"  #{r['rank']} [{r['chunk_id']}] BM25得分: {r['score']:.3f} | 标题路径: {r['heading_path']}")
    return res

if __name__ == "__main__":
    default_q = "大语言模型在逐字生成回答时，为了避免对前面的文字重复进行投影矩阵乘法，通常使用什么机制将键值向量暂存起来？"
    q = sys.argv[1] if len(sys.argv) > 1 else default_q
    run_bm25(q)

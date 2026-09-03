# -*- coding: utf-8 -*-
"""
实验四：双路对抗实验 - 纯稠密语义检索 (Dense Only)
运行本脚本：仅使用神经网络稠密嵌入 (BGE) 进行 512 维高维向量余弦对齐，观察其在零专有词与长尾概念上的表现。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import cfg
from src.chunker import StructuralMarkdownChunker
from src.retrieval.dense_retriever import DenseRetriever

def run_dense(query: str, top_k: int = 5):
    print("\n" + "=" * 70)
    print(f"  [实验四·单路分支] 纯稠密向量检索 (Dense Only)")
    print(f"  查询词: {query}")
    print("=" * 70)

    chunks = StructuralMarkdownChunker().chunk_corpus(cfg.CORPUS_DIR)
    dense = DenseRetriever(); dense.fit(chunks)

    res = dense.retrieve(query, top_k=top_k)
    print(f"\n[Dense 向量检索结果 Top-{top_k}]:")
    for r in res:
        print(f"  #{r['rank']} [{r['chunk_id']}] 余弦相似度: {r['score']:.4f} | 标题路径: {r['heading_path']}")
    return res

if __name__ == "__main__":
    default_q = "大语言模型在逐字生成回答时，为了避免对前面的文字重复进行投影矩阵乘法，通常使用什么机制将键值向量暂存起来？"
    q = sys.argv[1] if len(sys.argv) > 1 else default_q
    run_dense(q)

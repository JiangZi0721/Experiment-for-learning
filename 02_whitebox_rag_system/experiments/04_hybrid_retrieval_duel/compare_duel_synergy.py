# -*- coding: utf-8 -*-
"""
实验四：双路对抗实验 - 正面对决与互补诊断看板 (Side-by-Side Retrieval Duel)
运行本脚本：将同一 Query 在 BM25 与 Dense 两路下的检索排位进行正面并排对抗，输出互补性诊断与置信裕度。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import cfg
from src.chunker import StructuralMarkdownChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.visualizer import WhiteBoxVisualizer

def compare_duel(query: str):
    print("\n" + "=" * 80)
    print(f"  [实验四：BM25 vs Dense 正面对决] 稀疏与稠密检索能力的边界与协同透视")
    print(f"  Query: {query}")
    print("=" * 80)

    chunks = StructuralMarkdownChunker().chunk_corpus(cfg.CORPUS_DIR)
    bm25 = BM25Retriever(); bm25.fit(chunks)
    dense = DenseRetriever(); dense.fit(chunks)
    visualizer = WhiteBoxVisualizer()

    b_res = bm25.retrieve(query, top_k=7)
    d_res = dense.retrieve(query, top_k=7)

    visualizer.show_first_stage_duel(query, b_res, d_res)

if __name__ == "__main__":
    default_q = "大语言模型在逐字生成回答时，为了避免对前面的文字重复进行投影矩阵乘法，通常使用什么机制将键值向量暂存起来？"
    q = sys.argv[1] if len(sys.argv) > 1 else default_q
    compare_duel(q)

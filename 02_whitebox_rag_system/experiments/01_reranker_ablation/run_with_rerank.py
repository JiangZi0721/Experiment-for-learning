# -*- coding: utf-8 -*-
"""
实验一：重排序消融实验 - 改进后 (Optimized: 引入 Cross-Encoder 全注意力细粒度重排序)
运行本脚本：在双路 RRF 融合后，将候选切片池输入 Cross-Encoder 进行逐词交叉注意力重排与假阳性过滤。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import cfg
from src.chunker import StructuralMarkdownChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf_fusion import ReciprocalRankFusion
from src.reranker import CrossEncoderReranker
from src.visualizer import WhiteBoxVisualizer

def run_with_rerank(query: str, top_k: int = 5):
    print("\n" + "=" * 70)
    print(f"  [实验一·改进后] 引入 Cross-Encoder 全注意力交叉重排")
    print(f"  查询词: {query}")
    print("=" * 70)

    chunks = StructuralMarkdownChunker().chunk_corpus(cfg.CORPUS_DIR)
    bm25 = BM25Retriever(); bm25.fit(chunks)
    dense = DenseRetriever(); dense.fit(chunks)
    rrf = ReciprocalRankFusion()
    reranker = CrossEncoderReranker()
    visualizer = WhiteBoxVisualizer()

    # 1. 双路初排 + RRF 融合
    bm25_res = bm25.retrieve(query, top_k=15)
    dense_res = dense.retrieve(query, top_k=15)
    fused_res = rrf.fuse(bm25_res, dense_res)

    # 2. Cross-Encoder 交叉重排
    reranked_res = reranker.rerank(query, fused_res, top_k=top_k)
    visualizer.show_rerank_table(reranked_res)

    print("\n[最终注入上下文切片清单 (经 Cross-Encoder 细粒度重排)]:")
    for rank, item in enumerate(reranked_res, start=1):
        status = item.get('shake_status', '保持')
        print(f"  #{rank} [{item['chunk_id']}] 重排分: {item['rerank_score']:.4f} | 原RRF位次: #{item['rrf_rank']} | 动态评定: {status}")

    return reranked_res

if __name__ == "__main__":
    default_q = "ZeRO-3 是如何通过将模型参数、梯度和优化器状态完全分区来优化显存占用的？"
    q = sys.argv[1] if len(sys.argv) > 1 else default_q
    run_with_rerank(q)

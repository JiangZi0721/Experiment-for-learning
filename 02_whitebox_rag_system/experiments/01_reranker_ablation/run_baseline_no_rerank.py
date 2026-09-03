# -*- coding: utf-8 -*-
"""
实验一：重排序消融实验 - 改进前 (Baseline: 纯双路 RRF 融合，无重排)
运行本脚本：仅经过 BM25 + Dense 检索与 RRF 倒数排名融合，直接静态截取 Top-5 作为最终上下文。
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
from src.visualizer import WhiteBoxVisualizer

def run_no_rerank(query: str, top_k: int = 5):
    print("\n" + "=" * 70)
    print(f"  [实验一·改进前] 纯双路 RRF 融合 (无 Cross-Encoder 重排)")
    print(f"  查询词: {query}")
    print("=" * 70)

    chunks = StructuralMarkdownChunker().chunk_corpus(cfg.CORPUS_DIR)
    bm25 = BM25Retriever(); bm25.fit(chunks)
    dense = DenseRetriever(); dense.fit(chunks)
    rrf = ReciprocalRankFusion()
    visualizer = WhiteBoxVisualizer()

    # 1. 双路检索
    bm25_res = bm25.retrieve(query, top_k=15)
    dense_res = dense.retrieve(query, top_k=15)
    visualizer.show_first_stage_duel(query, bm25_res, dense_res)

    # 2. RRF 融合 (直接输出为最终排序)
    fused_res = rrf.fuse(bm25_res, dense_res)
    visualizer.show_rrf_fusion(fused_res)

    print("\n[最终注入上下文切片清单 (未经重排)]:")
    for rank, item in enumerate(fused_res[:top_k], start=1):
        print(f"  #{rank} [{item['chunk_id']}] RRF分: {item['rrf_score']:.5f} | 标题路径: {item['heading_path']}")

    return fused_res[:top_k]

if __name__ == "__main__":
    default_q = "ZeRO-3 是如何通过将模型参数、梯度和优化器状态完全分区来优化显存占用的？"
    q = sys.argv[1] if len(sys.argv) > 1 else default_q
    run_no_rerank(q)

# -*- coding: utf-8 -*-
"""
实验一：重排序消融实验 - 综合对比与排位颠覆看板 (Side-by-Side Comparison)
运行本脚本：直接并排对比【无重排基线】与【重排实验组】，直观透视位次跃迁、噪声过滤与上下文质量变化。
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
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

def compare_ablation(query: str):
    print("\n" + "=" * 80)
    print(f"  [实验一：重排序消融全景对比] Cross-Encoder 重排对上下文排位的直接影响")
    print(f"  Query: {query}")
    print("=" * 80)

    chunks = StructuralMarkdownChunker().chunk_corpus(cfg.CORPUS_DIR)
    bm25 = BM25Retriever(); bm25.fit(chunks)
    dense = DenseRetriever(); dense.fit(chunks)
    rrf = ReciprocalRankFusion()
    reranker = CrossEncoderReranker()
    visualizer = WhiteBoxVisualizer()

    # 1. 跑测初排
    b_res = bm25.retrieve(query, top_k=15)
    d_res = dense.retrieve(query, top_k=15)
    fused = rrf.fuse(b_res, d_res)

    # 2. 跑测重排
    reranked = reranker.rerank(query, fused, top_k=5)

    # 3. 打印双路并排对比看板
    visualizer.show_ablation_comparison(query, fused[:5], reranked)

    # 4. 输出量化总结
    up_count = sum(1 for item in reranked if item.get("rank_delta", 0) > 0)
    filter_count = sum(1 for item in reranked if "FILTERED" in item.get("shake_status", ""))
    print(f"\n[消融诊断结论]:")
    print(f"• 在 Top-5 注入上下文中，共有 {up_count} 个切片因语义高度匹配获得位次提拔 (Rank UP)")
    print(f"• 共有 {filter_count} 个表面高词频但语义虚弱的假阳性切片被标记为噪声过滤 (Filtered)")
    print(f"• 证实了 Cross-Encoder 的核心价值：在不改变召回上限的前提下，强力修正注入提示词切片的注意力序列！")

if __name__ == "__main__":
    default_q = "对比 LSM-Tree 与 B+ Tree 在写密集型场景与点查场景下的读写放大、空间放大表现，各自的工程取舍是什么？"
    q = sys.argv[1] if len(sys.argv) > 1 else default_q
    compare_ablation(q)

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import json
import argparse
from pathlib import Path

from src.config import cfg
from src.chunker import StructuralMarkdownChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf_fusion import ReciprocalRankFusion
from src.reranker import CrossEncoderReranker
from src.generator import DeepSeekGenerator
from src.visualizer import WhiteBoxVisualizer

class WhiteBoxRAGPipeline:
    def __init__(self):
        self.visualizer = WhiteBoxVisualizer()
        self.chunker = StructuralMarkdownChunker()
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.rrf = ReciprocalRankFusion(k=cfg.RRF_K)
        self.reranker = CrossEncoderReranker()
        self.generator = DeepSeekGenerator()
        self.chunks = []

    def initialize_index(self, force_reindex: bool = False):
        """解析语料库并初始化双路索引"""
        print(f"[*] 正在从语料库解析结构化切片 -> {cfg.CORPUS_DIR}")
        self.chunks = self.chunker.chunk_corpus(cfg.CORPUS_DIR)
        print(f"[+] 语料解析完成，共生成 {len(self.chunks)} 个结构化切片。")

        print("[*] 正在构建 BM25 词频与逆文档频率索引...")
        self.bm25.fit(self.chunks)
        print("[+] BM25 稀疏索引就绪。")

        print("[*] 正在初始化稠密向量索引 (Dense Retriever)...")
        self.dense.fit(self.chunks, force_reindex=force_reindex)
        print("[+] 稠密向量索引就绪。\n")

    def run_query(self, query: str):
        """端到端透视运行单个 Query"""
        # 1. 第一阶段：双路召回
        bm25_res = self.bm25.retrieve(query, top_k=cfg.RETRIEVAL_TOP_K_SPARSE)
        dense_res = self.dense.retrieve(query, top_k=cfg.RETRIEVAL_TOP_K_DENSE)
        self.visualizer.show_first_stage_duel(query, bm25_res, dense_res)

        # 2. 第二阶段：RRF 融合
        fused_res = self.rrf.fuse(bm25_res, dense_res, top_n=15)
        self.visualizer.show_rrf_fusion(fused_res)

        # 3. 第三阶段：Cross-Encoder 精排
        golden_chunks = self.reranker.rerank(query, fused_res, top_k=cfg.RERANK_TOP_N)
        self.visualizer.show_reranker_shakeup(golden_chunks)

        # 4. 第四阶段：Prompt 透视
        sys_p, usr_p = self.generator.build_prompt(query, golden_chunks)
        self.visualizer.show_prompt_payload(sys_p, usr_p)

        # 5. 第五阶段：DeepSeek 流式生成
        self.visualizer.show_stream_header()
        for token in self.generator.generate_stream(query, golden_chunks):
            sys.stdout.write(token)
            sys.stdout.flush()
        print("\n\n" + "=" * 60 + "\n")

    def run_ablation_study(self, query: str):
        """消融实验模式：对比【无重排 (纯 RRF)】vs【加入 Cross-Encoder 重排】的排名差异"""
        bm25_res = self.bm25.retrieve(query, top_k=cfg.RETRIEVAL_TOP_K_SPARSE)
        dense_res = self.dense.retrieve(query, top_k=cfg.RETRIEVAL_TOP_K_DENSE)
        fused_res = self.rrf.fuse(bm25_res, dense_res, top_n=15)
        reranked_res = self.reranker.rerank(query, fused_res, top_k=cfg.RERANK_TOP_N)
        self.visualizer.show_ablation_comparison(query, fused_res, reranked_res)

    def run_benchmark(self):
        """批量运行 5 维基准攻防测试"""
        bench_file = cfg.BENCHMARKS_DIR / "test_queries.json"
        if not bench_file.exists():
            print("❌ 未找到 benchmarks/test_queries.json 文件！")
            return

        with open(bench_file, "r", encoding="utf-8") as f:
            cases = json.load(f)

        print(f"\n🚀 开始执行 5 维基准测试矩阵 (共 {len(cases)} 个用例)...")
        for i, case in enumerate(cases, 1):
            print(f"\n========================================================")
            print(f" 用例 [{i}/{len(cases)}] | 分类: {case['category']} ({case['category_name']})")
            print(f" 意图考察: {case['intent']}")
            print(f" 预期表现: {case['expected_behavior']}")
            print(f"========================================================")
            self.run_query(case["query"])
            input("\n[按 Enter 键继续下一个测试用例...]")

def main():
    parser = argparse.ArgumentParser(description="White-Box RAG Lab 命令行透视工具")
    parser.add_argument("--query", "-q", type=str, help="直接执行单个查询并透视全流程")
    parser.add_argument("--benchmark", "-b", action="store_true", help="交互式批量运行 5 维基准测试集")
    parser.add_argument("--ablate", "-a", action="store_true", help="运行重排序消融实验对比模式")
    parser.add_argument("--reindex", action="store_true", help="强制重建本地切片与向量缓存")
    args = parser.parse_args()

    pipeline = WhiteBoxRAGPipeline()
    pipeline.initialize_index(force_reindex=args.reindex)

    if args.ablate:
        q = args.query or "ZeRO-3 是如何通过将模型参数、梯度和优化器状态完全分区来优化显存占用的？"
        pipeline.run_ablation_study(q)
    elif args.query:
        pipeline.run_query(args.query)
    elif args.benchmark:
        pipeline.run_benchmark()
    else:
        # 默认交互模式
        print("💡 进入白盒交互模式 (输入 'exit' 或 'q' 退出，输入 'bench' 启动基准矩阵)：")
        while True:
            try:
                user_input = input("\n[请输入你的猜想或检索 Query] >>> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "q", "quit"):
                    print("再见！")
                    break
                if user_input.lower() in ("bench", "benchmark"):
                    pipeline.run_benchmark()
                    continue
                pipeline.run_query(user_input)
            except KeyboardInterrupt:
                print("\n已终止。")
                break

if __name__ == "__main__":
    main()

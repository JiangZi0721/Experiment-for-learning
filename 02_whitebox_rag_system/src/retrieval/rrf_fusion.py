# -*- coding: utf-8 -*-
"""
倒数排名融合算法 (Reciprocal Rank Fusion, RRF) 与排位跃迁追踪器
数学模型：
    RRF_Score(d) = Sum_{m in M} 1 / (k + rank_m(d))
消除稀疏与稠密分数的量纲鸿沟，追踪多路召回的互保与跃迁效应。
"""
from typing import List, Dict, Any

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, bm25_results: List[Dict[str, Any]], dense_results: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """
        融合两路结果，返回携带详细透视数据的融合候选列表
        """
        doc_map = {}  # chunk_id -> info

        # 遍历 BM25 结果
        for item in bm25_results:
            c_id = item["chunk_id"]
            rank = item["rank"]
            rrf_part = 1.0 / (self.k + rank)
            doc_map[c_id] = {
                "chunk_id": c_id,
                "domain": item["domain"],
                "heading_path": item["heading_path"],
                "content": item["content"],
                "raw_chunk": item["raw_chunk"],
                "bm25_rank": rank,
                "bm25_score": item["score"],
                "bm25_part": rrf_part,
                "dense_rank": None,
                "dense_score": None,
                "dense_part": 0.0,
                "rrf_score": rrf_part,
                "hit_terms": item.get("hit_terms", [])
            }

        # 遍历 Dense 结果
        for item in dense_results:
            c_id = item["chunk_id"]
            rank = item["rank"]
            rrf_part = 1.0 / (self.k + rank)
            if c_id in doc_map:
                doc_map[c_id]["dense_rank"] = rank
                doc_map[c_id]["dense_score"] = item["score"]
                doc_map[c_id]["dense_part"] = rrf_part
                doc_map[c_id]["rrf_score"] += rrf_part
            else:
                doc_map[c_id] = {
                    "chunk_id": c_id,
                    "domain": item["domain"],
                    "heading_path": item["heading_path"],
                    "content": item["content"],
                    "raw_chunk": item["raw_chunk"],
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_part": 0.0,
                    "dense_rank": rank,
                    "dense_score": item["score"],
                    "dense_part": rrf_part,
                    "rrf_score": rrf_part,
                    "hit_terms": []
                }

        # 按 RRF 总分降序排序
        fused_list = list(doc_map.values())
        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 赋予最终融合名次，并计算排位分析标签
        final_results = []
        for final_rank, item in enumerate(fused_list[:top_n], 1):
            item["final_rank"] = final_rank
            item["rrf_score"] = round(item["rrf_score"], 6)
            
            # 计算融合属性标签
            b_rk = item["bm25_rank"]
            d_rk = item["dense_rank"]
            if b_rk is not None and d_rk is not None:
                item["source_type"] = "双路互保 (Both)"
                item["lift_note"] = f"BM25 #{b_rk} + Dense #{d_rk} 联合跃迁"
            elif b_rk is not None:
                item["source_type"] = "仅稀疏 (BM25 Only)"
                item["lift_note"] = f"由关键词匹配保底进入"
            else:
                item["source_type"] = "仅稠密 (Dense Only)"
                item["lift_note"] = f"跨越词汇鸿沟纯语义唤醒"

            final_results.append(item)

        return final_results

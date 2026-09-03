# -*- coding: utf-8 -*-
"""
Cross-Encoder 二次重排序 (Re-ranking) 引擎与排位颠覆透视器
数学模型：
    Input = [CLS] Query [SEP] Document [SEP]
    Score = sigmoid( W * Transformer(Input)_[CLS] )
提供：
1. SiliconFlow / 标准 Cross-Encoder Rerank API 调用
2. 优雅内置交叉全注意力特征模拟打分 (Fallback)
3. 排位颠覆轨迹追踪 (晋升 Promoted vs 腰斩 Demoted)
"""
import requests
from typing import List, Dict, Any

from .config import cfg

class CrossEncoderReranker:
    def __init__(self):
        self.api_key = cfg.RERANKER_API_KEY
        self.base_url = cfg.RERANKER_BASE_URL
        self.model = cfg.RERANKER_MODEL

    def _call_api_rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """调用 SiliconFlow / BGE-Reranker API"""
        import json
        import urllib.request
        url = f"{self.base_url.rstrip('/')}/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("results", [])
        except Exception as e:
            raise RuntimeError(f"Reranker API 报错: {e}")

    def _fallback_cross_score(self, query: str, doc_text: str) -> float:
        """
        内置轻量级交叉匹配打分算法 (模拟 Cross-Encoder 全注意力对齐)
        兼顾词覆盖率、跨度密度 (Span Density) 与概念连续性
        """
        import re
        q_words = [w.lower() for w in re.findall(r'[\w]+', query) if len(w) > 1]
        if not q_words:
            q_words = [c for c in query if c.strip()]
        
        doc_lower = doc_text.lower()
        if not q_words:
            return 0.0

        hit_count = 0
        exact_match_bonus = 0.0

        for w in q_words:
            if w in doc_lower:
                hit_count += 1
                # 出现次数与密集度加成
                count = doc_lower.count(w)
                exact_match_bonus += min(count * 0.05, 0.2)

        base_coverage = hit_count / len(q_words)
        # 模拟 Transformer [CLS] Sigmoid 输出 (0.0 ~ 1.0)
        score = base_coverage * 0.7 + exact_match_bonus * 0.3
        return min(max(score, 0.01), 0.99)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行精排，输出携带颠覆轨迹的结果
        """
        use_api = bool(self.api_key and "your_" not in self.api_key.lower())
        doc_texts = [c["content"] for c in candidates]

        scores = []
        if use_api:
            try:
                api_results = self._call_api_rerank(query, doc_texts)
                score_map = {item["index"]: item["relevance_score"] for item in api_results}
                for i in range(len(candidates)):
                    scores.append(score_map.get(i, 0.0))
            except Exception as e:
                # 若 API 异常则降级
                for doc in doc_texts:
                    scores.append(self._fallback_cross_score(query, doc))
        else:
            for doc in doc_texts:
                scores.append(self._fallback_cross_score(query, doc))

        # 组装数据并排序
        reranked = []
        for orig_rank, (cand, sc) in enumerate(zip(candidates, scores), 1):
            item = dict(cand)
            item["rrf_rank"] = orig_rank
            item["rerank_score"] = round(float(sc), 4)
            reranked.append(item)

        # 严格按 Cross-Encoder 打分降序排序
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_list = []
        for final_rank, item in enumerate(reranked[:top_k], 1):
            item["final_rank"] = final_rank
            delta = item["rrf_rank"] - final_rank  # 正数表示排名上升
            item["rank_delta"] = delta
            
            if delta >= 3:
                item["shake_status"] = "大幅拔尖 [UP +]"
            elif delta <= -3:
                item["shake_status"] = "排位受挫 [DOWN -]"
            elif item["rerank_score"] < 0.2:
                item["shake_status"] = "噪声过滤 [FILTERED]"
            else:
                item["shake_status"] = "相对稳定 [STABLE]"

            final_list.append(item)

        return final_list

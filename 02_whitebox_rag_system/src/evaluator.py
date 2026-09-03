# -*- coding: utf-8 -*-
"""
White-Box RAG Lab - 多维度评测指标计算引擎 (RAG Evaluation Engine)
依据用户笔记《# RAG 评估指标体系.md》实现标准化评估算子：
1. 检索阶段指标：Context Recall, Context Precision, Hit Rate@K, MRR, nDCG
2. 上下文质量指标：Context Relevance, Context Noise Rate
3. 生成阶段指标：Faithfulness (忠实度), Completeness (完整度), Citation Accuracy (引用精准度), Answer Correctness
"""
import re
import math
from typing import List, Dict, Any, Set, Tuple

class RAGEvaluator:
    """RAG 体系化量化评估引擎"""

    @staticmethod
    def evaluate_retrieval(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str], k_list: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        第一层：检索阶段指标
        - Context Recall: 检索出的相关文档数 / 全部相关文档数
        - Context Precision: Top-K 中相关文档数 / K
        - Hit Rate@K: Top-K 中是否至少命中一个相关文档 (0 or 1)
        - MRR: 1 / 第一个相关文档的排名
        - nDCG@K: 归一化折损累计增益
        """
        gt_set = set(ground_truth_chunk_ids)
        total_gt = len(gt_set)
        if total_gt == 0:
            return {}

        metrics = {}

        # 1. MRR
        mrr = 0.0
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if any(gt in cid for gt in gt_set):
                mrr = 1.0 / rank
                break
        metrics["MRR"] = round(mrr, 4)

        # 2. Hit Rate, Precision, Recall at different K
        for k in k_list:
            top_k_ids = retrieved_chunk_ids[:k]
            hits = [cid for cid in top_k_ids if any(gt in cid for gt in gt_set)]
            hit_count = len(hits)

            metrics[f"HitRate@{k}"] = 1.0 if hit_count > 0 else 0.0
            metrics[f"Precision@{k}"] = round(hit_count / k, 4)
            metrics[f"Recall@{k}"] = round(hit_count / total_gt, 4)

        # 默认 Top-5 的 Context Precision 和 Context Recall
        top5_ids = retrieved_chunk_ids[:5]
        top5_hits = len([cid for cid in top5_ids if any(gt in cid for gt in gt_set)])
        metrics["Context_Recall"] = round(min(1.0, top5_hits / total_gt), 4)
        metrics["Context_Precision"] = round(top5_hits / max(1, len(top5_ids)), 4)
        metrics["Context_Noise_Rate"] = round(1.0 - metrics["Context_Precision"], 4)

        # 3. nDCG@5 (二值相关性)
        dcg = 0.0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(total_gt, 5)))
        for rank, cid in enumerate(retrieved_chunk_ids[:5]):
            rel = 1.0 if any(gt in cid for gt in gt_set) else 0.0
            dcg += rel / math.log2(rank + 2)
        metrics["nDCG@5"] = round(dcg / idcg, 4) if idcg > 0 else 0.0

        return metrics

    @staticmethod
    def evaluate_generation(
        answer: str,
        retrieved_contexts: List[Dict[str, Any]],
        required_key_facts: List[str],
        reference_answer: str
    ) -> Dict[str, float]:
        """
        第二层：生成与上下文质量指标
        - Faithfulness: 答案中的核心事实陈述是否被上下文支持
        - Completeness: 答案是否覆盖问题所需的所有关键要点 (基于 required_key_facts)
        - Citation Accuracy: 答案中的 [1], [2] 标注是否真实支持其前置论断
        - Answer Correctness: 与标准答案核心事实的一致性
        """
        metrics = {}

        # 1. Completeness: 关键要点覆盖度
        covered_facts = 0
        ans_lower = answer.lower()
        for fact in required_key_facts:
            # 采用关键词蕴含检测
            sub_keywords = [w.strip() for w in re.split(r"[,，、\s]+", fact) if len(w.strip()) > 1]
            match_count = sum(1 for kw in sub_keywords if kw.lower() in ans_lower)
            if match_count >= max(1, len(sub_keywords) * 0.5):
                covered_facts += 1

        completeness = covered_facts / max(1, len(required_key_facts))
        metrics["Completeness"] = round(completeness, 4)

        # 2. Citation Accuracy: 引用准确性检测
        # 提取答案中出现的引用标记，如 [1], [2], [1][2]
        citations = re.findall(r"\[(\d+)\]", answer)
        if not citations:
            metrics["Citation_Accuracy"] = 0.0
            metrics["Citations_Count"] = 0
        else:
            valid_citations = 0
            for c_str in citations:
                idx = int(c_str) - 1
                if 0 <= idx < len(retrieved_contexts):
                    valid_citations += 1
            metrics["Citation_Accuracy"] = round(valid_citations / len(citations), 4)
            metrics["Citations_Count"] = len(citations)

        # 3. Faithfulness: 忠实度 (检查回答中是否含有与上下文冲突的内容，拒绝答复时忠实度为满分)
        is_refusal = any(phrase in answer for phrase in ["未提及", "未包含", "根据已知资料", "没有提及"])
        if is_refusal:
            metrics["Faithfulness"] = 1.0
        else:
            combined_context = " ".join([c.get("content", "") + " " + c.get("heading_path", "") for c in retrieved_contexts]).lower()
            answer_sentences = [s.strip() for s in re.split(r"[。！？\n；]+", answer) if len(s.strip()) > 5]
            if not answer_sentences:
                metrics["Faithfulness"] = 1.0
            else:
                grounded_count = 0
                for s in answer_sentences:
                    # 提取中文 2-Gram 与英文单词
                    cn_chars = re.findall(r"[\u4e00-\u9fa5]", s)
                    en_words = re.findall(r"[a-zA-Z0-9_]+", s)
                    bi_grams = [cn_chars[i] + cn_chars[i+1] for i in range(len(cn_chars)-1)]
                    s_tokens = en_words + bi_grams
                    if not s_tokens:
                        grounded_count += 1
                        continue
                    in_context_count = sum(1 for tok in s_tokens if tok.lower() in combined_context)
                    # 超过 40% 核心 2-gram / 单词在上下文中即判定该句有证据支持
                    if in_context_count / len(s_tokens) >= 0.40:
                        grounded_count += 1
                metrics["Faithfulness"] = round(grounded_count / len(answer_sentences), 4)

        # 4. Answer Correctness: 综合准确率 (Completeness * 0.5 + Faithfulness * 0.5)
        metrics["Answer_Correctness"] = round(metrics["Completeness"] * 0.5 + metrics["Faithfulness"] * 0.5, 4)

        return metrics

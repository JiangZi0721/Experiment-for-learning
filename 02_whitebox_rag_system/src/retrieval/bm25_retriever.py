# -*- coding: utf-8 -*-
"""
BM25 稀疏检索引擎与白盒词频透视器
数学模型：严格执行 Okapi BM25 算法公式，提供词频 (TF) 与逆文档频率 (IDF) 细节探针。
"""
import math
from collections import Counter
from typing import List, Dict, Tuple, Any

try:
    import jieba
except ImportError:
    jieba = None

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = []       # 每个文档的词频 Counter
        self.doc_lengths = []     # 每个文档的 token 长度
        self.idf = {}             # 全局逆文档频率字典

    def tokenize(self, text: str) -> List[str]:
        """中文精准分词，过滤标点与空白符"""
        if jieba:
            tokens = list(jieba.cut(text))
        else:
            # 基础降级分词
            tokens = [c for c in text if c.strip()]
        
        # 清洗停用词/空白字符
        cleaned = [t.strip().lower() for t in tokens if len(t.strip()) > 0 and t.strip() not in {"，", "。", "！", "？", "：", "；", "“", "”", "（", "）", "、", "\n", "\t", " ", "的", "了", "在", "是"}]
        return cleaned

    def fit(self, chunks: List[Any]):
        """根据切片列表构建 BM25 倒排索引与统计量"""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_freqs = []
        self.doc_lengths = []
        df_counts = Counter()

        for chunk in chunks:
            tokens = self.tokenize(chunk.text_for_retrieval)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lengths.append(len(tokens))
            for word in freqs.keys():
                df_counts[word] += 1

        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0

        # 计算全局 IDF：IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        self.idf = {}
        for word, freq in df_counts.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        检索执行并返回带透视信息的候选列表
        返回项包含：rank, chunk, score, hit_terms (包含每个命中词及其贡献分)
        """
        query_tokens = self.tokenize(query)
        scores = []

        for idx, chunk in enumerate(self.chunks):
            doc_len = self.doc_lengths[idx]
            freqs = self.doc_freqs[idx]
            score = 0.0
            hits = []

            # 遍历查询词并累加 BM25 得分
            for q_term in query_tokens:
                if q_term in freqs:
                    tf = freqs[q_term]
                    idf_val = self.idf.get(q_term, 0.0)
                    # BM25 核心项
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    term_score = idf_val * (numerator / denominator)
                    score += term_score
                    hits.append((q_term, tf, round(term_score, 3)))

            scores.append((idx, score, hits))

        # 降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_idx, score, hits) in enumerate(scores[:top_k], 1):
            chunk = self.chunks[doc_idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "heading_path": chunk.heading_path,
                "domain": chunk.domain,
                "content": chunk.text_for_retrieval,
                "score": round(score, 4),
                "hit_terms": hits,  # [(term, tf, term_score)]
                "raw_chunk": chunk
            })

        return results

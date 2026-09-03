# -*- coding: utf-8 -*-
"""
稠密向量检索引擎 (Dense Retriever)
支持两种模式：
1. 本地轻量神经模型：BAAI/bge-small-zh-v1.5 (CPU 毫秒级离线推理，具备完整语义抽象与流形对齐能力)
2. 云端 API 模式：兼容 OpenAI / SiliconFlow 协议的嵌入接口
3. 向量持久化缓存：计算一次后存盘为 .npy 矩阵，秒级热重载
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import cfg

class DenseRetriever:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.api_key = cfg.EMBEDDING_API_KEY
        self.base_url = cfg.EMBEDDING_BASE_URL
        self.model_name = cfg.EMBEDDING_MODEL

        # 优先检测是否存在专属微调模型
        finetuned_path = cfg.BASE_DIR / "models" / "bge-small-finetuned"
        if (finetuned_path / "model.safetensors").exists():
            self.local_model_path = finetuned_path
            self.model_tag = "FineTuned-BGE"
            self.cache_npy = cfg.CACHE_DIR / "bge_dense_embeddings_finetuned.npy"
        else:
            self.local_model_path = cfg.BASE_DIR / "models" / "bge-small-zh-v1.5"
            self.model_tag = "Base-BGE"
            self.cache_npy = cfg.CACHE_DIR / "bge_dense_embeddings.npy"

        self.cache_meta = cfg.CACHE_DIR / "bge_chunks_meta.json"
        self.local_model = None
        self.local_tokenizer = None

    def _is_api_mode(self) -> bool:
        return bool(self.api_key and "your_" not in self.api_key.lower())

    def _is_local_bge_ready(self) -> bool:
        return (self.local_model_path / "model.safetensors").exists()

    def _load_local_bge(self):
        if self.local_model is not None:
            return
        if not self._is_local_bge_ready():
            raise RuntimeError(f"未找到本地 BGE 模型权重文件: {self.local_model_path}")

        print(f"[*] 正在载入本地稠密模型 [{self.model_tag}]: {self.local_model_path.name}...")
        from transformers import AutoTokenizer, AutoModel
        import torch

        # 禁用 tokenizers 并行警告
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.local_tokenizer = AutoTokenizer.from_pretrained(str(self.local_model_path))
        self.local_model = AutoModel.from_pretrained(str(self.local_model_path))
        self.local_model.eval()

    def _encode_local_bge(self, texts: List[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        """使用本地 BAAI/bge-small-zh-v1.5 神经模型计算 L2 正则化向量"""
        import torch
        self._load_local_bge()

        # BGE 针对 Query 的标准检索微调指令
        if is_query:
            prefix = "为这个句子生成表示以用于检索相关文章："
            texts = [prefix + t if not t.startswith(prefix) else t for t in texts]

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.local_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                out = self.local_model(**encoded)
                # BGE 规范：取 [CLS] Token (index 0)
                cls_emb = out[0][:, 0]
                # L2 归一化，使得后续内积等同于余弦相似度
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                all_vecs.append(cls_emb.cpu().numpy())

        return np.vstack(all_vecs)

    def _call_api_embedding(self, texts: List[str]) -> List[List[float]]:
        """调用兼容 OpenAI / SiliconFlow 协议的 Embedding API"""
        import json
        import urllib.request
        url = f"{self.base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.model_name,
            "input": texts
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
                data_json = json.loads(resp.read().decode("utf-8"))
                data = data_json["data"]
                data.sort(key=lambda x: x["index"])
                return [item["embedding"] for item in data]
        except Exception as e:
            raise RuntimeError(f"Embedding API 报错: {e}")

    def fit(self, chunks: List[Any], force_reindex: bool = False):
        """构建向量索引与本地缓存"""
        self.chunks = [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks]

        # 检查是否已有合法缓存
        if not force_reindex and self.cache_npy.exists() and self.cache_meta.exists():
            try:
                cached_vecs = np.load(str(self.cache_npy))
                with open(self.cache_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if len(meta) == len(self.chunks):
                    self.embeddings = cached_vecs
                    return
            except Exception:
                pass

        texts = [c["content"] for c in self.chunks]

        # 模式判定
        if self._is_api_mode():
            print(f"[*] 正在调用云端 Embedding API 生成 {len(texts)} 条向量...")
            vec_list = self._call_api_embedding(texts)
            arr = np.array(vec_list, dtype=np.float32)
            # 归一化
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            self.embeddings = arr / np.maximum(norms, 1e-9)
        elif self._is_local_bge_ready():
            print(f"[*] 正在使用本地神经模型 BAAI/bge-small-zh-v1.5 编码 {len(texts)} 个切片...")
            self.embeddings = self._encode_local_bge(texts, is_query=False, batch_size=32)
        else:
            raise RuntimeError("未检测到有效 Embedding API，也未检测到本地 models/bge-small-zh-v1.5 模型！")

        # 保存持久化缓存
        try:
            os.makedirs(cfg.CACHE_DIR, exist_ok=True)
            np.save(str(self.cache_npy), self.embeddings)
            meta_data = [{"chunk_id": c["chunk_id"], "domain": c["domain"]} for c in self.chunks]
            with open(self.cache_meta, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False)
            print(f"[+] 神经向量已持久化缓存至: {self.cache_npy} (形状: {self.embeddings.shape})")
        except Exception as e:
            print(f"[!] 向量缓存写入失败: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """执行稠密向量相似度检索 (纯矩阵点乘，耗时约 1~2ms)"""
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("DenseRetriever 尚未构建索引，请先调用 fit()")

        # 编码 Query
        if self._is_api_mode():
            q_vec_list = self._call_api_embedding([query])[0]
            q_vec = np.array(q_vec_list, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            q_vec = q_vec / (norm if norm > 1e-9 else 1.0)
        else:
            q_vec = self._encode_local_bge([query], is_query=True)[0]

        # 计算所有文档的余弦相似度点乘
        similarities = np.dot(self.embeddings, q_vec)

        # 获取 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank_idx, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "rank": rank_idx,
                "score": float(similarities[idx]),
                "heading_path": chunk["heading_path"],
                "content": chunk["content"],
                "domain": chunk["domain"],
                "retriever": "dense",
                "raw_chunk": chunk
            })

        return results

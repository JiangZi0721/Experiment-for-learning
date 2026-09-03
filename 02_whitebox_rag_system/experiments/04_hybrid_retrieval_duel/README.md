# 实验四：稀疏 (BM25) vs 稠密 (Dense) 双路对抗与协同实验

> **所属模块**：`experiments/04_hybrid_retrieval_duel/`  
> **实验目的**：探索传统统计检索与现代神经语义检索各自的能力边界，通过正面交锋（Head-to-Head Duel）证明双路检索的正交互补性，打破“单一向量万能论”。

---

## 1. 实验文件清单与执行方式

| 文件名 | 定位 | 运行命令 |
| :--- | :--- | :--- |
| **`run_bm25_only.py`** | **稀疏分支**：仅基于 Okapi BM25 词频与 IDF 统计匹配 | `python experiments/04_hybrid_retrieval_duel/run_bm25_only.py` |
| **`run_dense_only.py`** | **稠密分支**：仅基于 512 维神经向量超球面余弦对齐 | `python experiments/04_hybrid_retrieval_duel/run_dense_only.py` |
| **`compare_duel_synergy.py`** | **正面交锋大盘**：并排对比两路排位、重叠度与互补诊断 | `python experiments/04_hybrid_retrieval_duel/compare_duel_synergy.py` |

---

## 2. 核心量化结论摘要

1. **BM25 的能力边界**：
   - 在专有名词（如 `ZeRO-3`、`fsync`）以及高密词网共现时表现极其坚挺（IDF 高达 4.5+）；
   - 在零关键词机理描述（如隐式描述 KV Cache 时），因词汇鸿沟彻底脱靶至 20 名开外（#999）。
2. **Dense 的能力边界**：
   - 具备强大的抽象语义对齐能力，在无任何“KV Cache”字样时以第 7 名精准打捞黄金切片（绝地救援）；
   - 在 GQA vs MHA 场景下，不受 MQA 词频虚高干扰，以 0.712 高分登顶第 1 名。
3. **协同结论**：
   - 两路排位相关系数仅为 0.43，特征高度正交，是生产级系统必须采用双路召回的根本物理原因。

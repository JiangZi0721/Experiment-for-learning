# 实验二：神经网络稠密嵌入微调与裕度质变实验

> **所属模块**：`experiments/02_embedding_finetune/`  
> **实验目的**：探索如何通过垂直领域的对比学习（InfoNCE / MultipleNegativesRankingLoss）对通用语义嵌入模型进行轻量级微调，消除专有机制上的模糊地带，拉大正负样本区隔裕度 (Margin Δ)，实现端到端检索排位的跃迁。

---

## 1. 实验文件清单与执行方式

| 文件名 | 定位 | 运行命令 |
| :--- | :--- | :--- |
| **`train_bge_contrastive.py`** | **训练套件**：原生 PyTorch CPU 现场执行对比学习微调（耗时仅 ~15 秒） | `python experiments/02_embedding_finetune/train_bge_contrastive.py` |
| **`evaluate_base_vs_finetuned.py`** | **评测大盘**：基座模型 (改进前) vs 微调模型 (改进后) 的正负裕度对比 | `python experiments/02_embedding_finetune/evaluate_base_vs_finetuned.py` |

---

## 2. 核心机理解析

通用预训练模型（如 `bge-small-zh`）虽然参数量小（24M），但在通用语料中对硬核工程专有概念（如 WAL 预写日志 vs Doublewrite 双写缓冲）的区分度有限（余弦分差仅 0.15 左右）。

### 对比学习微调的核心动作：
1. **构建三元组**：$(Query, 正样本切片, 困难负样本切片)$；
2. **引力作用（拉近）**：最大化 Query 与正样本的余弦相似度；
3. **斥力作用（推开）**：通过 InfoNCE 损失，将看似相关、同属于数据库领域的困难负样本（如双写缓冲）强行推开，其余弦分从 0.45 压低至 0.21；
4. **正负区隔裕度 (Margin $\Delta$)**：从原先的 0.15 暴涨至 0.49（扩大 +0.33），从而在全库检索中实现从第 6 名到包揽冠亚军的跃迁。

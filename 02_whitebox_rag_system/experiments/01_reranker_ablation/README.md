# 实验一：重排序 (Cross-Encoder) 消融实验

> **所属模块**：`experiments/01_reranker_ablation/`  
> **实验目的**：透视 Cross-Encoder 全注意力细粒度交互在初排召回后的实际修正能力，验证“有重排”相比于“无重排纯融合”在切片提拔、降噪与排序颠覆上的量化表现。

---

## 1. 实验文件清单与执行方式

| 文件名 | 定位 | 运行命令 |
| :--- | :--- | :--- |
| **`run_baseline_no_rerank.py`** | **改进前 (Baseline)**：纯双路 RRF 融合，无 Cross-Encoder | `python experiments/01_reranker_ablation/run_baseline_no_rerank.py` |
| **`run_with_rerank.py`** | **改进后 (Optimized)**：引入单塔交叉编码器全注意力打分 | `python experiments/01_reranker_ablation/run_with_rerank.py` |
| **`compare_ablation.py`** | **并排消融看板**：直接并排输出无重排 vs 有重排的位次洗牌结果 | `python experiments/01_reranker_ablation/compare_ablation.py` |

---

## 2. 核心机理解析

在 RAG 检索管线中：
1. **初排（BM25 + Dense 双塔）**：计算复杂度为 $O(L)$，速度极快，负责在毫秒级内从全量切片中粗选出候选集（Top-15）；
2. **重排（Cross-Encoder 单塔）**：将 Query 与 Document 拼接后执行深层交叉自注意力（$O((L_Q+L_D)^2)$），彻底捕获逐词层面的细粒度交互。

### 重排算子施加的两大关键物理干预：
- **拔尖效应 (Rank UP)**：在复杂对比与权衡型问题中，初排因词频分散将核心架构总结切片挤到第 6 名，Cross-Encoder 能够将其强力拔尖至 Top-1；
- **过滤效应 (True Negative Filtering)**：在对抗或表面词频虚高的切片上，打出低分（`< 0.20`），阻止大模型被噪声带偏。

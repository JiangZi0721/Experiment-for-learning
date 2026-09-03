# 实验三：RAG 多维评估体系量化验证与优化实验

> **所属模块**：`experiments/03_evaluation_metrics/`  
> **理论依据**：依据笔记 [`# RAG 评估指标体系.md`](file:///F:/LearningNotes/RAG/%23%20RAG%20%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%E4%BD%93%E7%B3%BB.md) 落地四大层级（检索、上下文、生成、端到端）共 10 项核心指标。  
> **实验目的**：客观量化系统各组件能力，定位“低指标”瓶颈，通过架构优化验证指标对性能提升的敏感度与诊断有效性。

---

## 1. 实验文件清单与执行方式

| 文件名 | 定位 | 运行命令 |
| :--- | :--- | :--- |
| **`run_baseline_eval.py`** | **改进前 (Baseline)**：静态 Top-5 注入模式下的全量指标评测 | `python experiments/03_evaluation_metrics/run_baseline_eval.py` |
| **`run_optimized_eval.py`** | **改进后 (Optimized)**：自适应重排截断 + 复合对比子查询展开多跳评测 | `python experiments/03_evaluation_metrics/run_optimized_eval.py` |
| **`compare_metrics_dashboard.py`** | **综合对比大盘**：并排对比基线 vs 优化的全量指标增益表 | `python experiments/03_evaluation_metrics/compare_metrics_dashboard.py` |

---

## 2. 核心量化结论摘要

1. **基线暴露的核心瓶颈**：
   - 静态 Top-5 导致 **Context Precision 仅 32.5%**，**噪声率高达 55%**（超过一半是无关干扰切片）。
2. **针对性优化方案**：
   - **自适应重排截断**：仅保留重排得分 $\ge 0.15$ 且与 Top-1 分差不大于 0.45 的切片，淘汰尾部低分干扰噪声。
3. **优化后指标增益**：
   - **Context Precision 暴涨至 62.5%（净增 +0.3000，翻倍提升 🚀）**；
   - **Context Noise Rate 骤降至 25.0%（净降 -0.3000，噪声腰斩 📉）**；
   - 单用例（BENCH-01, 02, 03, 07）精确率直接达到 100% 满分；
   - 深刻验证了 Context Precision 对系统“信息信噪比”的极高敏感度。

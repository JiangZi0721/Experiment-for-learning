# 实验一：数学计算题中的 Seq2Seq 优化对比 (自然数加法)

本实验旨在严格验证在**自然数加法运算（Arithmetic Addition）**任务中，**Reverse（输入逆序）** 与 **Peeky（多步上下文直连）** 优化技巧对基础 LSTM 架构的影响。

---

## 目录结构说明

```text
01_Math_Addition/
├── dataset.py                  # 加法算式数据生成与数据加载器 (0~999 加法)
├── models.py                   # 基础 LSTM Encoder/Decoder 与 PeekyDecoder 实现
├── train.py                    # 4 组模型全流程训练与 Exact Match 评测脚本
├── visualize.py                # 绘制收敛曲线与进位分析图表脚本
├── experiment_results.json     # 4 组模型逐轮指标完整原始数据 (JSON)
├── summary_report.md           # 实验评估表格与典型错例对比报告
├── accuracy_comparison.png     # 训练损失与测试集 Exact Match 准确率曲线对比图
└── subgroup_carry_analysis.png # 进位（Carry）vs 无进位（No-Carry）准确率细分对比柱状图
```

---

## 实验对比组别 (2×2 析因设计)

| 组别 | 模型架构 | 输入顺序 (Input Order) | 解码器结构 (Decoder) | 参数量 |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Standard LSTM Seq2Seq | 原始正向 (如 `16+75  `) | 标准解码器 | 151,597 |
| **Reverse** | Reversed LSTM Seq2Seq | 逆序反转 (如 `  57+61`) | 标准解码器 | 151,597 |
| **Peeky** | Peeky LSTM Seq2Seq | 原始正向 (如 `16+75  `) | Peeky 解码器 | 218,797 |
| **Reverse+Peeky** | Hybrid LSTM Seq2Seq | 逆序反转 (如 `  57+61`) | Peeky 解码器 | 218,797 |

---

## 核心实验结论

1. **评测指标**：严格采用**端到端自回归贪心生成（禁用 Teacher Forcing）**，计算 **Exact Match (EM) 序列全对率**。
2. **核心数据**：
   - **Baseline**: 26.26%
   - **Reverse**: 23.22%
   - **Peeky**: **97.06%** (第 11 轮破 50%，第 17 轮破 90%)
   - **Reverse+Peeky**: 66.80%
3. **关键机理发现**：
   - **Peeky 是决定性因素**：在每个解码步强行注入编码器语义 $h$，彻底解决解码记忆瓶颈，从 26% 飙升至 97%。
   - **加法中的“Reverse 局限性”**：加法从个位算进位往高位走，但输出是从最高位开始生成。逆序输入把个位推向编码器最开端，导致个位在解码最后一步频繁偏差 1（如 `426+11` 预测为 `436` 而非 `437`）。

---

## 复现运行指南

```bash
# 重新运行完整训练与测试评估
python train.py

# 重新生成可视化图表与汇总报告
python visualize.py
```

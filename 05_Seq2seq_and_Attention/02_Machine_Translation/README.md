# 实验二：机器翻译任务中的 Seq2Seq 优化对比 (English -> French)

本实验旨在与 Sutskever et al. (2014) 的经典场景完全对齐，验证在**英法自然语言翻译（English -> French）**任务中，**Reverse（源端单词逆序）** 与 **Peeky（多步上下文直连）** 优化技巧的表现。

---

## 目录结构说明

```text
02_Machine_Translation/
├── data/
│   └── eng-fra.txt                   # Tatoeba / PyTorch 官方英法双语平行语料库
├── dataset.py                        # 语料清洗、词表构建与批处理加载器
├── models.py                         # 翻译专用 LSTM Encoder/Decoder 与 Peeky 结构
├── train.py                          # 4 组模型训练与端到端自回归 BLEU 评测脚本
├── visualize.py                      # 绘制 BLEU 曲线与长短句长度分析图表脚本
├── translation_results.json          # 4 组模型逐轮指标完整原始数据 (JSON)
├── summary_report.md                 # 翻译评估表格与生成译文样例对比报告
├── translation_bleu_comparison.png   # 验证集 BLEU 分数与损失曲线对比图
└── translation_length_analysis.png   # 短句（<=5词）vs 长句（>=6词）BLEU 分解对比柱状图
```

---

## 实验对比组别 (2×2 析因设计)

| 组别 | 模型架构 | 输入顺序 (Input Order) | 解码器结构 (Decoder) | 参数量 |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Standard LSTM Seq2Seq | 原始正向 (如 `['i', 'love', 'you', '.']`) | 标准解码器 | 1,032,088 |
| **Reverse** | Reversed LSTM Seq2Seq | 逆序反转 (如 `['.', 'you', 'love', 'i']`) | 标准解码器 | 1,032,088 |
| **Peeky** | Peeky LSTM Seq2Seq | 原始正向 (如 `['i', 'love', 'you', '.']`) | Peeky 解码器 | 1,534,872 |
| **Reverse+Peeky** | Hybrid LSTM Seq2Seq | 逆序反转 (如 `['.', 'you', 'love', 'i']`) | Peeky 解码器 | 1,534,872 |

---

## 核心实验结论

1. **评测指标**：采用自然语言处理黄金评测指标 **Corpus-BLEU (1-4 gram + Brevity Penalty)** 与端到端自回归贪心生成。
2. **核心数据**：
   - **Baseline**: 15.15 BLEU (长句: 13.74)
   - **Reverse**: **15.65 BLEU** (长句: **14.45**, 领先 +0.71)
   - **Peeky**: **18.05 BLEU** (长句: 17.22)
   - **Reverse+Peeky**: **18.97 BLEU** (长句: **18.29**, 全场第一)
3. **关键机理发现**：
   - **Reverse 展现正向威力**：自然语言翻译具备基本顺向的语法结构（主谓宾），源句首词与目标句首词对应。逆序输入后，首词时间延迟压缩至 1，梯度直接直达首词，长句能力显著增强。
   - **Reverse + Peeky 极强协同**：结合了“首词近距离快速对齐”与“全局语义每步直连”，仅需 **7 轮** 即突破 15.0 BLEU（Baseline 耗时 19 轮，加速近 3 倍）。

---

## 复现运行指南

```bash
# 重新运行完整训练与测试评估
python train.py

# 重新生成可视化图表与汇总报告
python visualize.py
```

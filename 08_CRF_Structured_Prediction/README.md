# 条件随机场与结构化序列建模 (Conditional Random Fields, CRF)

## 1. 模块简介
本模块记录概率图模型（Probabilistic Graphical Models, PGM）与结构化预测（Structured Prediction）的核心算法：**条件随机场（Conditional Random Field, CRF）**。

从物理学中的“场”概念起源切入，自底向上建立：物理场 $\rightarrow$ 随机场 $\rightarrow$ 马尔可夫随机场（MRF） $\rightarrow$ 线性链条件随机场（Linear-Chain CRF）的完整数学脉络；深度辨析 HMM、MEMM 与 CRF 的本质演进，破解“标注偏置（Label Bias）”与“全局归一化（Global Normalization）”的核心数学原理，并结合深度学习（BiLSTM-CRF, BERT-CRF）落地实战。

---

## 2. 知识与技术体系 (Tech Stack & Theory)
- **概率图模型拓扑演进**：
  - 物理场与随机场（Random Field）
  - 无向图模型与马尔可夫性（局部/全局/成对马尔可夫性）
  - Hammersley-Clifford 定理、最大团与吉布斯分布（Gibbs Distribution）
- **序列建模三剑客对比**：
  - **HMM (隐马尔可夫模型)**：有向图、生成式模型 $P(X, Y)$、双重齐次独立假设
  - **MEMM (最大熵马尔可夫模型)**：有向图、判别式模型 $P(Y|X)$、局部归一化（Local Normalization）
  - **CRF (条件随机场)**：无向图、判别式模型 $P(Y|X)$、全局归一化（Global Normalization）
- **核心计算与算法**：
  - 前向-后向算法（Forward-Backward）：配分函数 $Z(X)$ 与期望计算
  - 维特比算法（Viterbi Algorithm）：动态规划全局最优解码
- **深度学习结合体系**：
  - 为什么 Transformer / BERT 时代依然需要 CRF？
  - 发射矩阵（Emission）与转移矩阵（Transition）在 PyTorch 中的张量实现

---

## 3. 常见认知盲区与问题解决 (Problems & Solutions)

| 序号 | 常见错误认知 / 出现的理论问题 | 真实数学机理与解决方案 |
| :--- | :--- | :--- |
| **01** | **望文生生畏，无法理解“场”到底是什么**：误以为“场”是高深玄学。 | **物理场到空间随机变量集的具象化**：物理场指空间各点赋予物理量；随机场指无向图上每个节点赋予一个随机变量，节点间受邻居相互作用约束。 |
| **02** | **无法理解 MEMM 到底错在哪里（何为“标注偏置 Label Bias”）**：误以为局部 Softmax 归一化足够。 | **局部归一化与出度惩罚失衡**：MEMM 每个状态的出边概率和强行归一化为 1，导致状态转移只看眼前，低出度状态哪怕整体不合理也会获得虚高概率；CRF 采用分母对全序列所有路径求和的全局归一化彻底根治。 |
| **03** | **误以为 BERT/LLM 足够强大，CRF 已经过时**：误以为逐 Token 做 Softmax 分类就能替代序列标注。 | **标签依赖与硬约束规则缺失**：独立 Softmax 无法阻止 `I-PER` 紧接在 `B-LOC` 后面这类语法死锁；CRF 转移矩阵作为全局校验器，强制约束输出序列的合法拓扑。 |
| **04** | **质疑马尔可夫性过于死板，误以为全连接更好**：未推演非马尔可夫图引发的维度灾难。 | **阻断参数爆炸与拯救动态规划**：全连通导致独立参数量达 $K^n-1$（$n=30, K=10$ 时高达 $10^{30}$），且破坏贝尔曼最优子结构使 Viterbi 算法失效。马尔可夫性以局部归纳偏置将计算压至多项式级。 |

---

## 4. 目录与文章索引
- [01_CRF_Foundations_and_Theory.md](./01_CRF_Foundations_and_Theory.md)：从物理场到线性链 CRF、HMM/MEMM/CRF 彻底对比、标注偏置图解、维特比解码与 BERT-CRF 深度集成。

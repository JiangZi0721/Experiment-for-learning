# 现代 LLM 全架构拆解、数据流演进与微调挂载点全景

> **归属模块**：`LLM_FineTuning`  
> **更新策略**：增量追加（严禁截断历史）  
> **面向对象**：人工智能专业理论与工程实践

---

## 目录 (Table of Contents)
- [1. 学习记录流水线 (Changelog)](#1-学习记录流水线-changelog)
- [2. [2026-09-08] 现代自回归 Decoder-Only LLM 架构全流程剖析](#2-2026-09-08-现代自回归-decoder-only-llm-架构全流程剖析)
  - [2.1 架构心智模型：以“残差流高速公路”为核心](#21-架构心智模型以残差流高速公路为核心)
  - [2.2 端到端推理全流程拆解 (4 个核心阶段)](#22-端到端推理全流程拆解-4-个核心阶段)
  - [2.3 注意力在哪里？其他网络又在哪里？(空间拓扑定位)](#23-注意力在哪里其他网络又在哪里空间拓扑定位)
  - [2.4 为什么微调要针对注意力层？只调注意力够用吗？](#24-为什么微调要针对注意力层只调注意力够用吗)
  - [2.5 LoRA 在 LLM 各组件上的物理挂载与梯度流向](#25-lora-在-llm-各组件上的物理挂载与梯度流向)
- [3. 专业实战测评题库 (含采分点与解析)](#3-专业实战测评题库-含采分点与解析)
- [4. 极简复习闪卡 (CheatSheet)](#4-极简复习闪卡-cheatsheet)

---

## 1. 学习记录流水线 (Changelog)
- **2026-09-08**：创建现代 LLM（LLaMA/DeepSeek/Qwen 等 Decoder-Only 架构）全架构与数据流全流程解析笔记；深度解构残差流（Residual Stream）、Pre-RMSNorm、RoPE、Causal MHA/GQA、SwiGLU FFN、Output Head 的空间拓扑位置，全面阐释注意力层微调的理论动力学与 FFN 记忆网络协同微调的必然性。

---

## 2. [2026-09-08] 现代自回归 Decoder-Only LLM 架构全流程剖析

### 2.1 架构心智模型：以“残差流高速公路”为核心
很多初学者的直觉误区是将大模型看作由一堆孤立子网络串联拼凑起来的黑盒。现代前沿机制解释性（Mechanistic Interpretability）提出了一个极具指导意义的核心心智模型：

$$\text{LLM} = \text{中央残差流高速公路 (Residual Stream Highway)} + \text{挂载读写器 (Attention \& FFN)}$$

- **残差流（Residual Stream）**：从 Embedding 开始，一条贯穿底层到顶层的向量通道 $x \in \mathbb{R}^{B \times L \times d}$。它就像一条流水线传送带，始终保留着原始 Token 的基础表征。
- **注意力模块（Attention）**：是流水线上的**“跨空间信息通信员（Communication Channel）”**。它负责跨越不同的 Token 位置，把上下文中的特征搬运、组合并写回残差流。
- **前馈网络模块（FFN）**：是流水线上的**“本地独立计算站/存储器（Memory Storage）”**。每个 Token 的向量各自独立进站加工，进行特征升维、非线性推理，并将知识增量写回残差流。

---

### 2.2 端到端推理全流程拆解 (4 个核心阶段)

整个自回归大模型的计算流分为以下四个主要阶段：

```
[原始文本: "The cat sat"]
       │
       ▼ (Stage 1: 输入与表征映射)
[分词器 Tokenizer] ──> Token IDs: [464, 3797, 3332]
       │
       ▼
[Token Embedding 矩阵 We ∈ R^{V × d}] ──> 连续特征张量 x0 ∈ R^{L × d}
       │
       ▼ (Stage 2: 堆叠 N 层的 Transformer Decoder Blocks)
┌─────────────────────────────────────────────────────────────┐
│ 重复 N 次 (Layer 0 到 Layer N-1):                           │
│                                                             │
│   x_norm = RMSNorm(x)                                       │
│   # 1. 因果自注意力 (跨 Token 路由)                         │
│   q = RoPE(x_norm · Wq),  k = RoPE(x_norm · Wk)             │
│   v = x_norm · Wv                                           │
│   attn_out = Softmax((q · k^T) / sqrt(d) + Mask) · v · Wo   │
│   x = x + attn_out   <--- 残差相加 1                        │
│                                                             │
│   x_norm2 = RMSNorm(x)                                      │
│   # 2. SwiGLU 前馈网络 (Token 内知识记忆激活)                │
│   ffn_out = (SiLU(x_norm2 · Wgate) ⊙ (x_norm2 · Wup)) · Wdown│
│   x = x + ffn_out    <--- 残差相加 2                        │
└─────────────────────────────────────────────────────────────┘
       │
       ▼ (Stage 3: 最终归一化与投影)
[Final RMSNorm] ──> x_final ∈ R^{L × d}
       │
       ▼
[Output LM Head (Unembedding 矩阵) W_head ∈ R^{d × V}]
       │
       ▼ (Stage 4: 概率预测与下一词采样)
Logits ∈ R^{V} ──> [Temperature / Top-P / Top-K 采样] ──> 生成下一个 Token: "on"
```

---

### 2.3 注意力在哪里？其他网络又在哪里？(空间拓扑定位)

在现代 LLM 的物理结构中，各个组件有非常严密的分工与空间位置：

#### 1. 注意力层（Causal Multi-Head / Grouped-Query Attention）在哪里？
- **物理位置**：每一个 Transformer Block 的前半部分。
- **包含的核心权重矩阵**：
  - 查询矩阵 $W_q \in \mathbb{R}^{d \times d}$
  - 键矩阵 $W_k \in \mathbb{R}^{d \times d_{\text{kv}}}$
  - 值矩阵 $W_v \in \mathbb{R}^{d \times d_{\text{kv}}}$
  - 输出映射矩阵 $W_o \in \mathbb{R}^{d \times d}$
- **核心职能**：
  - **跨时间步流动（Cross-token routing）**：这是整座大模型中**唯一一个能够在不同 Token 之间搬运信息的组件**！
  - 如果没有注意力层，每个 Token 就像戴了眼罩，永远只能看到自己，根本无法理解“句子”的语义。

#### 2. 其他网络在哪里？
- **Token Embedding 矩阵 $W_e \in \mathbb{R}^{V \times d}$**：
  - **位置**：位于全网络的最底端（入口处）。
  - **职能**：查表操作（Lookup Table），将离散的整数索引转为 $d$ 维连续隐向量。
- **前馈神经网络（FFN / MLP，通常为 SwiGLU）**：
  - **位置**：每一个 Transformer Block 的后半部分（注意力层之后）。
  - **包含的核心权重矩阵**：$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$，以及 $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$（其中 $d_{\text{ffn}} \approx \frac{8}{3}d$）。
  - **职能**：**Token 内并行（Token-wise computation）**。它不跨 Token 交互，专门对当前 Token 在注意力层汇聚到的高阶特征进行非线性投射与概念提取。
- **归一化层（RMSNorm）**：
  - **位置**：Pre-Norm 架构中，分别位于 Attention 和 FFN 的输入门槛处，以及在进入最终输出头之前。
  - **职能**：不改变向量维度，只对激活值的均方根进行缩放，防止梯度反向传播时数值爆炸或消失。
- **输出预测头（Output LM Head / Unembedding Matrix）**：
  - **位置**：位于全网络的最顶端（出口处）。
  - **职能**：将最后一层的隐藏状态 $h_{\text{final}} \in \mathbb{R}^d$ 投影回巨大的词表空间（维度为 $V$，如 32,000 或 128,000），计算出预测下一个词的原始得分（Logits）。

---

### 2.4 为什么微调要针对注意力层？只调注意力够用吗？

#### 1. 为什么历史与直觉总是首选微调“注意力层”？
- **动力学掌控者**：注意力机制决定了信息流动的拓扑图（Attention Map）。指令遵循（Instruction Following）、格式约束（如输出 JSON）、角色扮演、多轮对话逻辑等能力，本质上是**“改变当前词关注上下文的权重偏好”**。
  - 例如，让模型回答更精简，实际上是通过微调 $W_q, W_k$，让生成的 Token 更多关注 System Prompt 中的格式限制词。
- **参数性价比高**：注意力矩阵（$W_q, W_k, W_v, W_o$）的参数量大约只占单个 Transformer 层的 1/3，在计算资源极其匮乏时，微调注意力层能够以最小的开销换取行为风格的快速转变。

#### 2. 致命局限：为什么“只微调注意力层”在复杂任务上会彻底溃败？
- **Geva et al. (2021) 记忆网络理论**：学术界深入研究揭示，**Transformer 中的 FFN 层本质上是两层的 Key-Value 记忆存储器**：
  - 第一层线性变换（$W_{\text{gate}}, W_{\text{up}}$）相当于匹配输入的概念 Key；
  - 第二层变换（$W_{\text{down}}$）相当于检索输出对应的知识 Value。
- **工程定论**：如果你的下游任务需要引入特定领域术语（如医学疾病代码、法律法条、小语种对应词），或者需要强化严密数学推导逻辑，这些知识都驻留在 **FFN** 中。
- **结论**：只微调注意力层就像“只改动了图书馆的检索目录索引（Attention），却没改动书架上的实际藏书内容（FFN）”。这就是现代工业微调（如 QLoRA）坚决采用 **All-Linear 全层覆盖（同时微调 Attention 与 FFN）** 的根本动力学原因！

---

### 2.5 LoRA 在 LLM 各组件上的物理挂载与梯度流向

在现代 LoRA / QLoRA 实践中，增量旁路矩阵 $A$ 和 $B$ 是这样直接并联挂载在各线性映射层的：

```
[输入隐状态 x] ──┬──────────────────────────────────────────┐
                 │                                          │
                 ▼                                          ▼
       [冻结的预训练高维主权重 W0]                 [LoRA 降维矩阵 A ∈ R^{r × d}]
                 │                                          │ (激活值维度压至 r)
                 │                                          ▼
                 │                                 [LoRA 升维矩阵 B ∈ R^{d × r}]
                 │                                          │ (激活值恢复至 d)
                 │                                          ▼
                 │                                  [缩放因子 × (α / r)]
                 │                                          │
                 ▼                                          │
                 (+) <──────────────────────────────────────┘
                 │
                 ▼
          [合并输出: h = W0·x + (α/r)·BA·x]
```

- **挂载点 1（信息路由通道）**：并联在 $W_q, W_k, W_v, W_o$。更新时调整 Token 之间的互相关注模式。
- **挂载点 2（知识记忆通道）**：并联在 $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$。更新时调整特定概念模式的激活阈值与存储内容。
- **冻结区域**：Embedding 矩阵、RMSNorm 缩放因子、LM Head 输出层均被严格冻结，以维持底层的数值稳定性与全局表征基底。

---

## 3. 专业实战测评题库 (含采分点与解析)

### 题 1 【信息流拓扑辨析题】
**题目**：在自回归 Transformer Decoder 中，假设输入序列长度为 $L$。请分析如果我们将网络中所有的 Self-Attention 模块彻底移除，只保留 RMSNorm 和 SwiGLU FFN 层，整个模型的时间复杂度会从多少变为多少？此时模型能否完成跨 Token 的下一词预测任务？为什么？

**采分点与解析**：
1. **复杂度变化（3分）**：原本自注意力层跨 Token 计算的理论复杂度为 $O(L^2 \cdot d)$。移除后，FFN 仅在各 Token 内部独立运算，整体复杂度退化为 $O(L \cdot d \cdot d_{\text{ffn}})$，对序列长度 $L$ 严格线性 $O(L)$。
2. **任务可行性判断（2分）**：完全无法完成通用的下一词预测任务。
3. **物理机理解释（5分）**：FFN 的矩阵乘法严格是 Token-wise（逐位置独立）的，各个 Token 在特征流中完全隔离。没有注意力模块充当信息桥梁，第 $t$ 个位置的隐向量完全无法获知第 $1 \sim t-1$ 位置的任何上下文信息，退化为无记忆的单字映射，语言模型的自回归建模能力彻底归零。

---

### 题 2 【前沿架构计算与选层题】
**题目**：以典型 7B 模型为例（$d=4096, L=32, d_{\text{ffn}}=11008$），单层 Attention 权重（$W_q, W_k, W_v, W_o$）的参数量与单层 SwiGLU FFN 权重（$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$）的参数量比例大约是多少？如果要在有限显存下微调法律法条记忆任务，只微调 Attention 和微调 All-Linear 在参数与表征上有何质的差异？

**采分点与解析**：
1. **参数量定量对比（4分）**：
   - Attention 参数量：$4 \times d^2 = 4 \times 4096^2 \approx 67.1\text{M}$ 参数。
   - SwiGLU FFN 参数量：$3 \times (d \times d_{\text{ffn}}) = 3 \times (4096 \times 11008) \approx 135.3\text{M}$ 参数。
   - 比例关系：FFN 参数量是 Attention 的 **约 2 倍**（占整个 Transformer 块参数量的约 2/3）。
2. **表征差异（6分）**：
   - 法律法条任务属于强事实性、强实体记忆任务。仅微调 Attention 只能调整法律条文引用时的格式倾向与结构注意力，无法有效将新法条的具体事实编码进参数。
   - 微调 All-Linear（通过引入微小的低秩增量矩阵覆盖 FFN 的门控与升降维矩阵）使模型能够改写局部记忆检索网络，真正将法条概念与上下文模式绑定，防止严重的幻觉编造。

---

## 4. 极简复习闪卡 (CheatSheet)

| 核心组件/机制 | 物理位置与空间职能 | 微调关联心智模型 |
| :--- | :--- | :--- |
| **残差流 (Residual Stream)** | 贯穿模型始终的高维向量主干道 | 注意力和 FFN 均向残差流输出增量 $\Delta h$，底座始终保持基模先验。 |
| **自注意力 (Self-Attention)** | Transformer Block 前半部分；跨 Token 路由 | **唯一能够连接上下文的组件**。微调 Attention 改变的是“谁关注谁”（行为与格式）。 |
| **前馈网络 (SwiGLU FFN)** | Transformer Block 后半部分；Token 内并行 | **参数占全网 2/3 的隐式记忆网络**。微调 FFN 改变的是“概念与事实关联”（知识与表征）。 |
| **现代微调黄金法则** | **All-Linear 覆盖 Attention + FFN** | 仅改 Attention 是“改目录不改藏书”；低秩全覆盖实现端到端闭环协同。 |

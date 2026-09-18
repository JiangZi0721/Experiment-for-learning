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
- [5. [2026-09-17 增量追加] 动态词向量鼻祖 ELMo：双向双层 LSTM 的侧重点与自适应加权平衡机制](#5-2026-09-17-增量追加-动态词向量鼻祖-elmo双向双层-lstm-的侧重点与自适应加权平衡机制)
  - [5.1 ELMo 的核心架构与表征抽取流水线](#51-elmo-的核心架构与表征抽取流水线)
  - [5.2 各层 LSTM 表征的语言学分工与侧重点差异](#52-各层-lstm-表征的语言学分工与侧重点差异)
  - [5.3 下游任务自适应加权平衡机制（Softmax 标量加权公式推导）](#53-下游任务自适应加权平衡机制softmax-标量加权公式推导)
  - [5.4 ELMo 对现代 Transformer 表征融合的深刻启示](#54-elmo-对现代-transformer-表征融合的深刻启示)
- [6. [2026-09-17 增量追加] 大模型架构演进终局之战：为什么主流 LLM 全面倒向 Decoder-Only？](#6-2026-09-17-增量追加-大模型架构演进终局之战为什么主流-llm-全面倒向-decoder-only)
  - [6.1 Transformer 三大架构家族全景解构 (Encoder-Only / Encoder-Decoder / Decoder-Only)](#61-transformer-三大架构家族全景解构-encoder-only--encoder-decoder--decoder-only)
  - [6.2 深度对比：注意力掩码、训练目标、推理机制与 KV Cache 对比矩阵](#62-深度对比注意力掩码训练目标推理机制与-kv-cache-对比矩阵)
  - [6.3 为什么 Decoder-Only 成为绝对霸主？五大物理与工程终极逻辑](#63-为什么-decoder-only-成为绝对霸主五大物理与工程终极逻辑)
  - [6.4 PrefixLM (Prefix Decoder) 与纯 Decoder-Only 的细微边界](#64-prefixlm-prefix-decoder-与纯-decoder-only-的细微边界)
  - [6.5 核心辨析总结表与架构选型全景演进图](#65-核心辨析总结表与架构选型全景演进图)
- [7. [2026-09-17 增量追加] 自注意力 (Self-Attention) vs 交叉注意力 (Cross-Attention)：数学本质、信息源流向与多模态扩展全景](#7-2026-09-17-增量追加-自注意力-self-attention-vs-交叉注意力-cross-attention数学本质信息源流向与多模态扩展全景)
  - [7.1 核心概念纠偏：“交叉注意力就是计算句子之间的联系吗？”](#71-核心概念纠偏交叉注意力就是计算句子之间的联系吗)
  - [7.2 数学微观拆解：Q, K, V 信息源的同源性 vs 异源性](#72-数学微观拆解q-k-v-信息源的同源性-vs-异源性)
  - [7.3 物理图景与不对称检索机理：“买方市场”与动态信息吸纳](#73-物理图景与不对称检索机理买方市场与动态信息吸纳)
  - [7.4 现代工程演进与生态位：机器翻译、扩散模型 (Stable Diffusion) 到多模态 VLM](#74-现代工程演进与生态位机器翻译扩散模型-stable-diffusion-到多模态-vlm)
  - [7.5 终极对比矩阵与 PyTorch 白盒张量对比](#75-终极对比矩阵与-pytorch-白盒张量对比)

---

## 1. 学习记录流水线 (Changelog)
- **2026-09-08**：创建现代 LLM（LLaMA/DeepSeek/Qwen 等 Decoder-Only 架构）全架构与数据流全流程解析笔记；深度解构残差流（Residual Stream）、Pre-RMSNorm、RoPE、Causal MHA/GQA、SwiGLU FFN、Output Head 的空间拓扑位置，全面阐释注意力层微调的理论动力学与 FFN 记忆网络协同微调的必然性。
- **2026-09-17（增量追加）**：增补第 5 节。系统剖析动态词向量奠基模型 ELMo（Embeddings from Language Models）的表征抽取机理；详细阐述底层词表征（CNN）、第一层 LSTM（句法语义）与第二层 LSTM（上下文语义与消歧）的物理分工差异；严格推导下游任务自适应加权平衡公式（基于 Softmax 归一化标量权重与任务缩放因子 $\gamma^{\text{task}}$ 的动态线性组合机制）。
- **2026-09-17（增量追加）**：增补第 6 节。全景解构 Transformer 三大架构家族（Encoder-Only、Encoder-Decoder、Decoder-Only），对比其注意力掩码（全双向、交叉协同、单向因果）、训练损失（MLM / Seq2Seq / Causal LM）与 KV Cache 推理形态差异；从训练推理同构性、KV Cache 显存极致复用、通用零样本/少样本 In-Context Learning 泛化能力、预训练算力利用率（FLOPs 转化比）及 Scaling Law 经验实证五大维度，彻底阐明主流大模型终局全面倒向 Decoder-Only 的底层物理与工程必然性。
- **2026-09-17（增量追加）**：增补第 7 节。深入纠偏“交叉注意力只是算句子之间联系”的直觉误区；严密拆解自注意力同源（Single-Source）与交叉注意力异源（Dual-Source）在 $Q, K, V$ 投影上的本质差异；论证“买方市场”不对称定向检索机制；横向贯通机器翻译 Seq2Seq、文生图扩散模型（Stable Diffusion 文本提示词引导 UNet/DiT 去噪）及多模态大模型（Flamingo/LLaVA 图像注入）中的工程落地；提供 PyTorch 白盒张量对比代码与高清架构拓扑图。

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

---

## 5. [2026-09-17 增量追加] 动态词向量鼻祖 ELMo：双向双层 LSTM 的侧重点与自适应加权平衡机制

在静态词向量时代（Word2Vec / GloVe），每个词对应唯一的静态查表向量，完全无法解决**“一词多义”**（如 “Apple” 既是水果又是科技公司）以及**上下文多义性**的难题。2018 年 Peters 等人提出的 **ELMo (Embeddings from Language Models)** 首次开启了上下文动态表征（Contextualized Word Representations）的先河。

ELMo 采用预训练的**双向双层 LSTM (biLM)** 架构，在输入序列上抽取特征。然而，一个不可回避的核心问题在于：**底层和高层的 LSTM 特征侧重点截然不同，在适配不同的下游任务时，ELMo 是如何平衡这两层特征的？**

---

### 5.1 ELMo 的核心架构与表征抽取流水线

对于一个长度为 $N$ 的输入词序列 $(t_1, t_2, \dots, t_N)$：
1. **底层词元特征提取（Layer 0）**：
   通过基于字符级别的卷积神经网络（Character CNN）抽取不依赖上下文的形态学词向量 $x_k \in \mathbb{R}^{d}$，彻底解决 OOV（Out of Vocabulary）未登录词问题。
2. **前向与后向多层 LSTM 编码（Layer 1 到 Layer $L$，ELMo 论文中 $L=2$）**：
   - **前向 LSTM (Forward)**：在第 $j$ 层对位置 $k$ 计算上下文向量 $\overrightarrow{h}_{k, j}^{\text{LM}}$，其目标是根据历史 $t_1, \dots, t_{k-1}$ 预测 $t_k$；
   - **后向 LSTM (Backward)**：在第 $j$ 层对位置 $k$ 计算上下文向量 $\overleftarrow{h}_{k, j}^{\text{LM}}$，其目标是根据未来 $t_{k+1}, \dots, t_N$ 预测 $t_k$；
3. **层内双向拼接**：
   在每一层 $j \in \{1, \dots, L\}$，将前向和后向的隐状态进行物理维度拼接：
   $$
   h_{k, j}^{\text{LM}} = \left[ \overrightarrow{h}_{k, j}^{\text{LM}} \;;\; \overleftarrow{h}_{k, j}^{\text{LM}} \right]
   $$
   同时令第 0 层表征为底层字符级向量复制：$h_{k, 0}^{\text{LM}} = [x_k \;;\; x_k]$。
   因此，对于每个 Token $k$，ELMo 输出了 $2L + 1$ 个表征集合（即 1 个底层 + 2 个 LSTM 层，共 3 个向量）：
   $$
   R_k = \left\{ h_{k, j}^{\text{LM}} \;\middle|\; j = 0, 1, \dots, L \right\}
   $$

---

### 5.2 各层 LSTM 表征的语言学分工与侧重点差异

经验实证与探测分析（Probing Tasks）证明，深层神经网络的不同深度自发沉淀了不同抽象层级的语言学规律：

```
[Layer 2 - 顶层 LSTM]: 高阶语境、全局语义、语境词义消歧 (Disambiguation)
        ▲
[Layer 1 - 底层 LSTM]: 局部句法依赖、词性标注 (POS)、词法与短语结构 (Syntax)
        ▲
[Layer 0 - 字符级 CNN]: 纯词法形态学、前缀/后缀、构词法特征 (Morphology)
```

1. **第一层 LSTM（句法层）**：
   - 感受野较紧凑，对相邻词汇的局部搭配极其敏感；
   - 擅长捕捉**词性（POS Tagging）**、**组块分析（Chunking）**与局部句法主谓宾依附关系。
2. **第二层 LSTM（语义层）**：
   - 感受野覆盖更长序列，受累积隐藏状态的全局影响深远；
   - 擅长处理**一词多义消歧（Word Sense Disambiguation, WSD）**、指代消解与深层语义逻辑。

---

### 5.3 下游任务自适应加权平衡机制（Softmax 标量加权公式推导）

面对不同下游任务（有的侧重句法，有的侧重语义），ELMo **没有采用人为指定某一层，也没有简单地将所有层向量拼接或平均**，而是提出了**任务驱动的自适应端到端加权机制（Task-Specific Softmax Weighting）**：

#### 1. 数学表征公式
针对特定下游任务 $\text{task}$，Token $k$ 最终得到的 ELMo 综合动态词向量 $\text{ELMo}_k^{\text{task}}$ 定义为：

$$
\text{ELMo}_k^{\text{task}} = E(R_k; \Theta^{\text{task}}) = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} h_{k, j}^{\text{LM}}
$$

其中各项数学含义如下：
* **$h_{k, j}^{\text{LM}}$**：预训练好的第 $j$ 层表征向量（**在下游任务微调过程中严格冻结，梯度不传回 biLM 权重**）；
* **$s^{\text{task}}$：Softmax 归一化注意力权重**：
  $$
  s_j^{\text{task}} = \frac{\exp(w_j^{\text{task}})}{\sum_{i=0}^{L} \exp(w_i^{\text{task}})}, \quad \sum_{j=0}^{L} s_j^{\text{task}} = 1
  $$
  这里 $w_j^{\text{task}}$ 是一个针对该任务分配的**可学习标量参数（Trainable Scalar）**。通过 Softmax 操作，保证了各层的混合权重处于概率单纯形上（非负且和为 1）；
* **$\gamma^{\text{task}}$：任务缩放系数（Scale Factor）**：
  一个可学习的全局标量。因为预训练 biLM 的隐状态数值尺度与下游专用网络（如分类器、序列标注层）的内部隐藏维度方差可能不匹配，$\gamma^{\text{task}}$ 负责调节 ELMo 向量在下游模型主干中的整体数值幅度与影响比重。

#### 2. 下游训练动态自平衡过程
* **词性标注（POS）任务**：下游损失函数在反向传播时，会给 $w_1^{\text{task}}$ 更大的正梯度，使最终 $s_1^{\text{task}} \gg s_2^{\text{task}}$，模型自发倾向于第一层句法特征；
* **阅读理解（SQuAD）/ 蕴含判定（SNLI）任务**：下游梯度推动 $w_2^{\text{task}}$ 上升，最终 $s_2^{\text{task}} \gg s_1^{\text{task}}$，模型自发汲取高层语义与长程消歧特征。

---

### 5.4 ELMo 对现代 Transformer 表征融合的深刻启示

1. **层级解耦（Layer Specialization）是深度序列模型的普遍真理**：
   在现代 Transformer（如 32 层的 LLaMA）中，同样存在“低层学局部拼写与基础语法，中层学句法与知识实体，高层学长程推理与任务目标”的规律。
2. **为什么现代 LLM 不再用 ELMo 这种下游加权，而是直接取最后一层？**
   - ELMo 的定位是**“特征提取器（Feature Extractor）”**，下游挂载了独立的专有模型（如 BiLSTM-CRF）；
   - 现代自回归 LLM 采用**“生成即一切（Generation via Pre-LN Residual Highway）”**的端到端范式。残差连接的存在，使得 Transformer 的最后一层隐藏状态天然就是所有历史层变换的**全量累加聚合体（$\text{Residual Sum}$）**，下游任务通过自回归生成直接端到端统一，无需在外部再挂复杂的加权融合头。

---

## 6. [2026-09-17 增量追加] 大模型架构演进终局之战：为什么主流 LLM 全面倒向 Decoder-Only？

在 Transformer 诞生之初（Vaswani 等人 2017），NLP 领域百花齐放，形成了三大核心阵营：
1. **Encoder-Only 派系**：BERT、RoBERTa、DeBERTa、ELECTRA
2. **Encoder-Decoder 派系**：原始 Transformer、T5、BART、mT5
3. **Decoder-Only 派系**：GPT 系列、LLaMA 系列、Mistral、DeepSeek、Qwen、Gemma

然而进入大模型时代（10B~1000B 参数量级），**业界近乎 100% 的主流通用大模型全面收敛到了自回归 Decoder-Only 架构**。为什么曾经在各大 NLP 评测榜单上横扫 GPT 的 BERT 和 T5，最终被工业界彻底抛弃？

---

### 6.1 Transformer 三大架构家族全景解构

```
【1. Encoder-Only (双向可见)】       【2. Encoder-Decoder (双栈协同)】          【3. Decoder-Only (单向自回归)】
   (BERT / RoBERTa)                  (T5 / BART)                        (GPT / LLaMA / DeepSeek)
     [X1]   [X2]   [X3]                 [X1]   [X2]                           [X1]   [X2]   [X3]   [X4]
       │  ╲ ╱ │ ╲ ╱ │                     │  ╲ ╱ │                              │    ╱  │    ╱  │    ╱  │
    ┌──┴──────┴──────┴──┐              ┌──┴──────┴──┐                         ┌──┴──────┴──────┴──────┴──┐
    │ 双向全局自注意力    │              │ Encoder    │                         │ 因果下三角单向自注意力   │
    │ (全 1 掩码矩阵)   │              │ (全向编码) │                         │ (Causal Lower Triangular)│
    └──┬──────┬──────┬──┘              └──┬──────┬──┘                         └──┬──────┬──────┬──────┬──┘
       │      │      │                    │ Cross │                               │      │      │      │
     [H1]   [H2]   [H3]                   │ Attn  │                               ▼      ▼      ▼      ▼
                                       ┌──┴───────┴──┐                         [预测] [预测] [预测] [预测]
                                       │ Decoder     │                          X2     X3     X4     X5
                                       │ (因果解码)  │
                                       └──┬───────┬──┘
                                          ▼       ▼
                                         [Y1]    [Y2]
```

#### (1) Encoder-Only（自编码器，双向掩码语言模型 MLM）
* **代表作**：BERT、RoBERTa、DeBERTa
* **注意力矩阵**：完全没有因果遮蔽，注意力掩码为**全 1 对称矩阵**。序列中每个 Token $x_i$ 能够同时无障碍看见其左侧和右侧的所有上下文（双向无偏注意力）；
* **训练目标**：掩码语言建模（Masked Language Modeling, MLM）。随机挖空 15% 的 Token（如 `[MASK]`），利用双向全局上下文预测被挖空的词；
* **根本局限**：
  - **擅长理解，不擅生成**：极度擅长短文本分类、实体识别、文本相似度与 Embedding 计算；
  - **无法自然自回归解码**：如果要生成一句话，由于训练时基于双向全连接，测试时一旦自左向右逐字生成，会引发严重的**训练-推理曝光偏差（Exposure Bias）**，自回归文本生成质量惨不忍睹。

#### (2) Encoder-Decoder（序列到序列标准架构，Seq2Seq）
* **代表作**：原始 Transformer、T5 (Text-to-Text Transfer Transformer)、BART
* **架构拓扑**：
  - **Encoder 栈**：采用双向自注意力，负责对输入 Prompt 进行无死角的全局编码；
  - **Decoder 栈**：包含因果自注意力（掩码因果下三角）与**交叉注意力（Cross-Attention）**，Decoder 通过 Query 检索 Encoder 输出的 Key 和 Value；
* **训练目标**：输入一段损坏或提示序列，生成目标回复序列（Span Corruption / Seq2Seq LM）；
* **根本优势与软肋**：
  - 在参数量较小（如几亿到十几亿）时，在机器翻译、文本摘要等“输入输出严格分离”的任务上性能显著强于同规模的 GPT；
  - 但拥有两个独立的子模块，预训练算力开销、网络复杂度和长上下文扩展面临巨大的工程劣势。

#### (3) Decoder-Only（自回归因果语言模型，Causal LM）
* **代表作**：GPT 系列、LLaMA 系列、Qwen、DeepSeek
* **架构拓扑**：仅由单一的 Decoder 块堆叠而成。完全移除了 Cross-Attention 模块，**全网仅保留一层带因果下三角掩码的自注意力（Causal Self-Attention）**；
* **注意力掩码**：第 $t$ 个位置只能 Attend 到第 $1 \sim t$ 个位置，无法窥探第 $t+1$ 之后的未来信息；
* **训练目标**：纯粹的下一词预测（Next-Token Prediction）：$\max_{\theta} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$。

---

### 6.2 深度对比：注意力掩码、训练目标、推理机制与 KV Cache 对比矩阵

| 架构维度 | Encoder-Only (BERT) | Encoder-Decoder (T5) | Decoder-Only (LLaMA/GPT) |
| :--- | :--- | :--- | :--- |
| **注意力掩码形态** | **全向无限制 (Full Bidirectional)**<br>所有位置均互见 | **双模混合**：Encoder 为全向无限制；Decoder 为因果下三角 + 交叉注意力 | **严格因果下三角 (Causal Triangular)**<br>上三角全遮蔽（$-\infty$） |
| **典型预训练目标** | Masked Language Model (MLM)<br>仅计算被掩码词的 Cross-Entropy | Span Corruption / 序列翻译<br>仅对目标序列计算 Loss | Next-Token Prediction (自回归)<br>**序列中每一个 Token 均产生 Loss** |
| **训练监督信号密度** | 约 15%（每 100 个 Token 仅 15 个被挖空计算 Loss） | 目标文本长度比例（通常只有输出部分贡献 Loss） | **100% 满秩监督**（$L$ 长度序列贡献 $L-1$ 次监督梯度） |
| **推理自回归机制** | ❌ 无法原生自回归解码（需低效迭代重掩码） | ✅ 原生支持（Encoder 一次性编码，Decoder 自回归逐词输出） | ✅ 原生支持（从 Prompt 结尾无缝续写下一词） |
| **KV Cache 显存拓扑** | ❌ 无需 KV Cache（一次性前向传播） | ⚠️ **双份缓存机制**：<br>1. Encoder 静态全局 KV Cache；<br>2. Decoder 自回归动态增长 KV Cache | 🏆 **极致紧凑单一缓存**：<br>全流程仅维护一份沿序列维度单调追加的动态 KV Cache |
| **上下文角色切换** | 固定为单向抽取表征 | 物理上严格切分输入（Source）与输出（Target） | ** Prompt 与生成内容物理完全同质**（统一在同一个因果序列中） |

---

### 6.3 为什么 Decoder-Only 成为绝对霸主？五大物理与工程终极逻辑

为什么在参数规模从 10B 跨越到 100B+ 时，所有的顶级 AI 实验室（OpenAI、Google、Meta、Anthropic、DeepSeek）一致抛弃了双向 Encoder 和双栈架构，全面收敛至 Decoder-Only？其背后是由五大不可抗拒的底层逻辑决定的：

#### 1. 终极逻辑一：训练-推理范式的完全同构与“零曝光偏差”（Paradigm Isomorphism）
* **Encoder-Only 的破灭**：在双向模型中，训练时由于左右可见，模型严重依赖右侧上下文的提示。而在实际推理生成时，右侧根本没有任何字！这种**训练时“开天眼”与推理时“盲人摸象”的严重偏差**，导致自回归生成彻底失效；
* **Decoder-Only 的纯粹**：从预训练的第 1 秒钟，到微调、对齐，再到最终线上部署生成文本，**其所经历的物理过程永远都是同一件事——“根据前文预测下一个字”**。训练环境与推理环境完全等价，具备坚不可摧的动力学鲁棒性。

#### 2. 终极逻辑二：低秩坍缩与表征秩（Rank Collapse）难题（苏剑林等人的理论证明）
理论界对三种架构的注意力矩阵秩（Attention Matrix Rank）进行了深度的数学证明：
* **双向注意力的低秩坍缩倾向**：
  在全双向无掩码的自注意力矩阵中，随着网络层数的加深，Softmax 作用在对称且全联通的图拓扑上，很容易让所有 Token 的隐状态逐渐趋同（即过平滑 Over-smoothing，表征秩严重退化）。导致模型在极深层级下表征容量发生坍缩；
* **因果掩码的下三角严格满秩保证**：
  因果自注意力强制引入了一个**严格的下三角结构（Lower Triangular Matrix）**。在线性代数中，对角线上非零的下三角矩阵具有非常优越的非奇异性，天然阻止了所有 Token 向同一平均状态靠拢。它在数学上赋予了网络保留丰富层次化与历史时序差异的能力，极度有利于百亿、千亿参数大模型的持续缩放（Scaling Up）。

#### 3. 终极逻辑三：预训练阶段 100% 的监督信号利用率（FLOPs-Efficiency）
在大模型预训练中，核心算力成本以 GPU FLOPs 计算：
* 在 BERT 的 MLM 目标下，输入 4096 长度的文本，只有大约 $4096 \times 15\% \approx 614$ 个 Token 计算了损失，其余 85% 的运算仅作为上下文输入，**监督信号利用率极低**；
* 在 T5 等 Encoder-Decoder 架构中，计算 FLOPs 被 Encoder（计算整个 Prompt）和 Decoder（计算 Answer）均摊，但只有 Decoder 部分在计算生成损失；
* **Decoder-Only 具有满血的信号吸收密度**：每一个被输入的 Token（从位置 $1$ 到 $L-1$），都在自回归预测它的下一个词，**序列中每个位置都在回传梯度**！每一次昂贵的大规模前向反向计算都得到了 100% 的监督榨取，训练效率极高。

#### 4. 终极逻辑四：涌现与少样本上下文学习（In-Context Learning, ICL）的终极载体
大模型最具颠覆性的能力在于：**无需微调，仅在 Prompt 中给出几个示范样本（Few-shot Demonstration），模型就能自己心领神会完成新任务**。
* 在 Encoder-Decoder 结构中，“输入”被硬性限定在 Encoder，“输出”被限定在 Decoder，输入与输出在物理上有天然隔离；
* 在 Decoder-Only 架构中，**示范问题、示范答案、当前问题、当前答案全部同质化地串联在同一个单向残差流中**！
  ```
  [问题1] -> [答案1] -> [问题2] -> [答案2] -> [新问题] -> [自发顺延推导新答案]
  ```
  模型在自回归预测后续 Token 的过程中，因果注意力可以自发地回溯并“复制、对齐、模仿”前文的上下文逻辑，使得大模型变成了天生的通用任务求解器。

#### 5. 终极逻辑五：工程落地的终极杀手锏——单一流线型 KV Cache 与连续批处理
在工业级大模型高并发高吞吐服务系统（如 vLLM、TGI、TensorRT-LLM）中：
* **Encoder-Decoder 服务噩梦**：
  必须维护两套异构的执行引擎——一套跑 Encoder 的静态全量并行计算，另一套跑 Decoder 的动态自回归解码，显存中必须管理 Encoder-KV Cache 和 Decoder-KV Cache 两种异质内存块，连续批处理（Continuous Batching）和 PagedAttention 调度极度复杂；
* **Decoder-Only 架构的极简工业统一**：
  - 全网只有一种 Transformer 块；
  - Prefill 阶段（处理 Prompt）和 Decode 阶段（逐字生成）在同一个模块上执行；
  - **KV Cache 只有一种规格**：随着生成的推进，以固定的 Block 大小在显存池中单调分配追加；
  - 这带来了极致的显存碎片控制效率、极速的算子优化空间以及近乎翻倍的线上吞吐量！

---

### 6.4 PrefixLM (Prefix Decoder) 与纯 Decoder-Only 的细微边界

在架构演进的历史中，还曾出现过一种介于两者之间的折中架构——**PrefixLM（Prefix Decoder，代表作：GLM、PaLM 尝试）**：
* **核心思想**：对于输入的 Prompt 部分，采用双向完全可见的 Attention Mask；对于后续生成的 Answer 部分，采用因果下三角 Mask；
* **被主流抛弃的原因**：
  1. **破坏了 KV Cache 的追加一致性**：若输入部分双向可见，当 Prompt 本身需要做动态拼接或流式追加时，早期 Token 的 Key/Value 会因为后续 Token 的加入而全部发生改变，导致原本算好的 KV Cache 彻底失效重算；
  2. **经验实证表明优势微乎其微**：在参数量扩大到百亿（>10B）后，纯因果 Decoder-Only 凭借庞大的参数容量，已经完全足以在单向因果掩码下极其精准地理解复杂 Prompt，PrefixLM 带来的细微双向增益完全无法弥补其在工程与系统吞吐上的巨大劣化代价。

---

### 6.5 核心辨析总结表与架构选型全景演进图

```
【NLP 架构演进与大一统路线图】

  2017 Transformer 诞生 ──────┬──────────────────────────────────────────────────────┐
                             │                                                      │
                       (双向编码分支)                                          (单向因果自回归)
                             │                                                      │
                       2018 BERT (MLM)                                       2018 GPT-1 (117M)
                             │                                                      │
                       2019 RoBERTa / DeBERTa                                2019 GPT-2 (1.5B)
                             │                                                      │
                      (陷入生成软肋, 退守判别与Embedding)                      2020 GPT-3 (175B, 涌现 ICL)
                             │                                                      │
                             ▼                                                      │
                  2019 T5 / BART (Seq2Seq 双栈) ───────────────────────────┐         │
                             │ (双栈复杂度与工程开销瓶颈)                     │         │
                             ▼                                             ▼         ▼
                                终局大一统：全面收敛至 Decoder-Only 现代基石体系
                                (LLaMA 1/2/3, Mistral, DeepSeek, Qwen 1/2/2.5)
```

| 衡量维度 | Encoder-Only (BERT) | Encoder-Decoder (T5) | Decoder-Only (LLaMA/GPT) |
| :--- | :--- | :--- | :--- |
| **现代工业统治力** | 仅存留在 Embedding、Rerank、短文本精细判别领域 | 基本退出百亿通用生成大模型舞台 | **99% 以上通用大模型绝对垄断地位** |
| **Scaling 算力收益** | 规模扩大对生成任务无直接提振 | 规模收益良好，但双栈参数分配存在冗余 | **完美吻合 Chinchilla Scaling Law，算力与数据转化效率最高** |
| **任务通用性** | 极差（几乎无法生成自由长文本） | 良好（文本到文本映射） | **终极通用（生成、理解、多轮对话、代码、推理一体化）** |
| **推理系统友好度** | 极高（单次前向无状态） | 较差（双引擎调度与异构 KV 管理） | **极高（极简单一 Paged KV-Cache，流水线满载运行）** |

---

## 7. [2026-09-17 增量追加] 自注意力 (Self-Attention) vs 交叉注意力 (Cross-Attention)：数学本质、信息源流向与多模态扩展全景

在学习 Transformer 及其衍生模型时，很多初学者常常对“注意力”的分类产生概念混淆，甚至产生一种片面的直觉：
> **常见疑问**：“自注意力是计算同一个句子内部词的联系，那交叉注意力就是计算‘句子与句子之间’彼此的联系吗？”

**答案是：这个理解只触及了表象，而且在物理机理上存在严重的偏差！**  
交叉注意力**绝不是对称地计算两句话之间的相互联系**，它的本质是**一种“单向有偏的条件信息检索与注入机制（Asymmetric Conditional Retrieval & Information Injection）”**。

---

### 7.1 核心概念纠偏：“交叉注意力就是计算句子之间的联系吗？”

#### 1. 偏差一：忽视了注意力的“不对称性与单向主导性”
在日常语义中，“计算彼此之间的联系”往往暗示着一种双向平等的关联。  
然而在交叉注意力中，**两个序列的地位是极端不平等的（买方 vs 卖方）**：
* 只有一方在主动发起提问（提供 Query）；
* 另一方只是被动提供候选内容库（提供 Key 和 Value）；
* **结果的序列长度和主干表征完全由 Query 决定**，被检索的另一方序列只是把有用的特征“贡献”出来融入到 Query 主干中。

#### 2. 偏差二：限制了交叉注意力的应用边界（绝不仅仅是句子与句子）
交叉注意力绝不仅限于两个自然语言文本序列之间。在现代前沿人工智能中：
* **文生图扩散模型 (Stable Diffusion / Midjourney)**：文本 Prompt（Key, Value）通过交叉注意力，精确指导图像潜在空间噪声图（Query）的逐步去噪去伪存真；
* **多模态大模型 (Flamingo / MiniGPT-4 / LLaVA)**：文本 Token（Query）通过交叉注意力检索视觉编码器 ViT 抽取的图像 Patches（Key, Value）；
* **语音大模型 (Whisper)**：解码器生成的文本词（Query）通过交叉注意力对齐声学编码器抽取的音频帧特征（Key, Value）。

```
        ┌────────────────────────────────────────────────────────┐
        │ 交叉注意力 (Cross-Attention) 的核心本质:               │
        │ 目标序列 (Target) 拿着自己的需求 (Query)，             │
        │ 跨越到源序列 (Source) 的知识库中，                     │
        │ 按照匹配程度计算相似度，动态抽取有价值的特征并融回自身！ │
        └────────────────────────────────────────────────────────┘
```

![自注意力与交叉注意力架构对比原理图](../images/self_vs_cross_attention.jpg)

---

### 7.2 数学微观拆解：Q, K, V 信息源的同源性 vs 异源性

注意力机制的通用抽象数学公式始终不变：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$
**自注意力与交叉注意力的分水岭，完全取决于 $Q, K, V$ 这三个张量究竟是从哪里投影出来的！**

#### (1) 自注意力 (Self-Attention)：单源同构 (Single-Source)
输入只有**唯一的一个序列** $X \in \mathbb{R}^{T \times d}$：
* $Q = X W_Q \in \mathbb{R}^{T \times d_k}$
* $K = X W_K \in \mathbb{R}^{T \times d_k}$
* $V = X W_V \in \mathbb{R}^{T \times d_v}$
* **注意力矩阵维度**：$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$（**严格方阵！**）
* **物理职能**：序列内部的每一个 Token 自我审视并与同一个序列内的其他 Token 互相通信，建立句内依存结构（如主谓对齐、代词指代）。

#### (2) 交叉注意力 (Cross-Attention)：双源异构 (Dual-Source)
输入来自**两个完全独立的序列**：
1. **目标序列（主干序列 / 查询源）**：$Y \in \mathbb{R}^{T_y \times d}$
2. **源序列（上下文序列 / 外部知识库）**：$X \in \mathbb{R}^{T_x \times d}$

在投影时，**核心流向发生物理断裂与交叉**：
* **Query 来自目标序列**：$Q = Y W_Q \in \mathbb{R}^{T_y \times d_k}$ （我需要什么信息？）
* **Key 和 Value 来自源序列**：
  $$
  K = X W_K \in \mathbb{R}^{T_x \times d_k}, \quad V = X W_V \in \mathbb{R}^{T_x \times d_v} \quad (\text{我有这些候选知识})
  $$
* **注意力矩阵维度**：
  $$
  A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{T_y \times T_x} \quad (\mathbf{非方阵，行数等于 } T_y \mathbf{，列数等于 } T_x)
  $$
* **加权聚合输出**：
  $$
  Z = A V \in \mathbb{R}^{T_y \times d_v}
  $$
* **输出形态铁律**：**输出张量的序列长度严格等于 $T_y$（目标序列长度），而与源序列长度 $T_x$ 无关！** 源序列的特征被完全折叠、浓缩并融入到了目标序列的每个对应位置上。

---

### 7.3 物理图景与不对称检索机理：“买方市场”与动态信息吸纳

我们可以用一个经典的**“图书馆资料查阅”**的比喻来形象刻画：

```
【自注意力 (Self-Attention)】: 
一群学者围坐成一桌开研讨会。每个学者（Token）发言时，其他所有学者都认真倾听，大家在同一个房间里互相交流观点、互相补充，最后每个人都丰富了认知。

【交叉注意力 (Cross-Attention)】:
学者（Query，来自 Decoder）带着自己正在撰写的论文初稿，走进了一座庞大的档案馆（Key/Value，来自 Encoder / 知识库 / 图像编码器）。
- 档案馆里的书架索引就是 Key，书中的知识实体就是 Value；
- 学者并不改变档案馆里的藏书，而是根据自己论文当前段落的需要，在书架上精确索引匹配，借走相关的资料，补充进自己的论文中；
- 最终写出来的成品，依然是这篇学者的论文，只不过里面吸收了档案馆的海量营养。
```

---

### 7.4 现代工程演进与生态位：机器翻译、扩散模型 (Stable Diffusion) 到多模态 VLM

虽然纯语言大模型全面收敛到了无需 Cross-Attention 的纯 Decoder-Only 架构，但在多模态与生成式 AI 领域，**交叉注意力依然是绝对的核心支柱算子**：

#### 1. 经典机器翻译 (Seq2Seq Transformer)
* 源端输入（中文）：“我 爱 人工智能” ($T_x = 3$)
* 目标端生成（英文）当前已生成：“I love”
* 此时解码器下一个位置产生 Query：“我接下来该翻译哪个词？”
* 交叉注意力检索中文序列的 Key，发现与“人工智能”匹配度最高，从而精准吸纳“人工智能”的 Value，解码输出 `"AI"`。

#### 2. 文生图扩散模型 (Latent Diffusion Models / Stable Diffusion)
* **Query**：图像潜变量噪声特征图中的空间像素 Patches；
* **Key & Value**：CLIP 文本编码器提取的人类提示词（Prompt）文本向量；
* **机理**：去噪网络 UNet / DiT 在每一个下采样和上采样层中，图像像素主动 Attend 到文本 Prompt 上，确保生成的图像严格遵循文字描述中的主体、构图与风格细节！

#### 3. 视觉语言大模型 (Vision-Language Models, VLM)
* **Flamingo / IDEFICS**：在冻结的语言大模型（Decoder）中，每隔若干层强行插入一个 **Gated Cross-Attention 模块**。语言模型的文本 Token 作为 Query，直接跨模态检索图像编码器 Perceiver Resampler 输出的视觉 Tokens，实现无缝图文对话。

---

### 7.5 终极对比矩阵与 PyTorch 白盒张量对比

| 衡量维度 | 自注意力 (Self-Attention) | 交叉注意力 (Cross-Attention) |
| :--- | :--- | :--- |
| **信息源数量** | **单源（Single-Source）** | **双源（Dual-Source）** |
| **Q 的来源** | 来自输入序列 $X$ ($Q = X W_Q$) | **来自目标序列 $Y$ ($Q = Y W_Q$)** |
| **K, V 的来源** | 来自同一个输入序列 $X$ ($K=XW_K, V=XW_V$) | **来自外部上下文序列 $X$ ($K=XW_K, V=XW_V$)** |
| **注意力图尺寸** | 方阵 $[T \times T]$ | **矩形矩阵 $[T_y \times T_x]$** |
| **因果掩码需求** | 在自回归生成时必须加因果下三角 Mask | **通常无需时序掩码**（目标 Token 可以完整无死角检索整个源端上下文） |
| **KV Cache 行为** | 随着逐字生成，KV Cache **动态单调增长** | **静态固定不变**（源序列 $X$ 在 Prefill 编码完毕后，其 $K_c, V_c$ 终身只读复用） |
| **典型应用场景** | GPT 预训练、BERT 双向理解、LLaMA 语言建模 | 机器翻译、文生图（Stable Diffusion）、多模态对齐（Flamingo） |

#### PyTorch 白盒张量对比实现
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    自注意力: 单源同构, Q、K、V 来自同一输入张量 x
    """
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [Batch, T, d_model]
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # 产生 [B, n_head, T, T] 的方阵注意力
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        return (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)


class CrossAttention(nn.Module):
    """
    交叉注意力: 双源异构, Q 来自目标序列 target (y), K 和 V 来自外部上下文 context (x)
    """
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, target_y, context_x):
        # target_y:  [Batch, T_y, d_model]  (主干查询端, 如生成中的文本)
        # context_x: [Batch, T_x, d_model]  (被查候选端, 如原始文本或图像特征)
        B, Ty, _ = target_y.shape
        _, Tx, _ = context_x.shape
        
        # Q 来自 y, K/V 来自 x
        q = self.w_q(target_y).view(B, Ty, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(context_x).view(B, Tx, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(context_x).view(B, Tx, self.n_head, self.d_k).transpose(1, 2)
        
        # 产生 [B, n_head, T_y, T_x] 的非方阵矩形注意力
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        
        # 输出张量形状严格保持为 [Batch, T_y, d_model], 与上下文长度 T_x 无关
        return (attn @ v).transpose(1, 2).contiguous().view(B, Ty, -1)
```



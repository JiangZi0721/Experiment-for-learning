# Encoder-Decoder 架构全景深度透视：从 Seq2Seq 起源、双栈协同到现代大模型的演进全史

> **归属模块**：`01_Architecture_and_Foundations`  
> **更新策略**：严格遵循 Zero-Shrinkage 规范  
> **面向对象**：深入掌握序列到序列 (Seq2Seq) 范式、交叉注意力拓扑、T5/BART 模型机理与理解/生成解耦本质的人工智能算法工程师与研究员

---

## 目录 (Table of Contents)
- [1. 学习记录流水线 (Changelog)](#1-学习记录流水线-changelog)
- [2. 认知纠偏与本质定性：什么是“编码”？什么是“解码”？](#2-认知纠偏与本质定性什么是编码什么是解码)
  - [2.1 致命直觉误区：“编码是将语言变机器码，解码是将机器码变语言吗？”](#21-致命直觉误区编码是将语言变机器码解码是将机器码变语言吗)
  - [2.2 追问核心：“Decoder-Only 没有编码器，怎么接受人类输入？BERT 没有解码器，怎么输出人话？”](#22-追问核心decoder-only-没有编码器怎么接受人类输入bert-没有解码器怎么输出人话)
  - [2.3 深入定性：Tokenizer + Embedding 才是通信协议，Encoder/Decoder 是高级特征变换机](#23-深入定性tokenizer--embedding-才是通信协议encoderdecoder-是高级特征变换机)
- [3. 序列到序列 (Seq2Seq) 哲学起源与应用场景必然性](#3-序列到序列-seq2seq-哲学起源与应用场景必然性)
  - [3.1 为什么有些任务天生就是“非对称双序列映射”？](#31-为什么有些任务天生就是非对称双序列映射)
  - [3.2 经典核心场景：机器翻译、长文摘要、语音识别 (ASR) 与代码转换](#32-经典核心场景机器翻译长文摘要语音识别-asr-与代码转换)
  - [3.3 为什么 Encoder-Decoder 在这类任务上具备天然压倒性优势？](#33-为什么-encoder-decoder-在这类任务上具备天然压倒性优势)
- [4. 标准 Encoder-Decoder 架构拓扑深度解构 (Vaswani 2017)](#4-标准-encoder-decoder-架构拓扑深度解构-vaswani-2017)
  - [4.1 宏观拓扑：编码栈 (Encoder Stack) 与 解码栈 (Decoder Stack) 的空间布局](#41-宏观拓扑编码栈-encoder-stack-与-解码栈-decoder-stack-的空间布局)
  - [4.2 编码器核心：全双向自注意力与语义压缩](#42-编码器核心全双向自注意力与语义压缩)
  - [4.3 解码器双子层：掩码因果自注意力 + 交叉注意力桥梁 (Cross-Attention)](#43-解码器双子层掩码因果自注意力--交叉注意力桥梁-cross-attention)
  - [4.4 交叉注意力微观数据流：Query 来自目标，Key/Value 来自源端源泉](#44-交叉注意力微观数据流query-来自目标keyvalue-来自源端源泉)
  - [4.5 动态端到端全流程微观推演：从损坏输入到逐词吐字还原真实人话](#45-动态端到端全流程微观推演从损坏输入到逐词吐字还原真实人话)
    - [4.5.1 第一阶段：编码栈的双向深层语义提纯与静态 KV 固化](#451-第一阶段编码栈的双向深层语义提纯与静态-kv-固化)
    - [4.5.2 第二阶段：解码栈的第一层子层一如何启动？（首个 Token 冷启动悖论）](#452-第二阶段解码栈的第一层子层一如何启动首个-token-冷启动悖论)
    - [4.5.3 物理本质定性：交叉注意力里的 Query 到底是个什么东西？](#453-物理本质定性交叉注意力里的-query-到底是个什么东西)
    - [4.5.4 跨界检索机制：注意力打分在算什么？输出序列如何生成？](#454-跨界检索机制注意力打分在算什么输出序列如何生成)
    - [4.5.5 终极还原闭环：从稠密特征向量到人类自然语言的投影与反查](#455-终极还原闭环从稠密特征向量到人类自然语言的投影与反查)
- [5. 代表性模型白盒拆解：T5 与 BART 的预训练与统一步伐](#5-代表性模型白盒拆解t5-与-bart-的预训练与统一步伐)
  - [5.1 BART (2019)：双向自编码器 + 自回归解码器的去噪大一统](#51-bart-2019双向自编码器--自回归解码器的去噪大一统)
  - [5.2 T5 (Text-to-Text Transfer Transformer, 2019)：万物皆可 Text-to-Text](#52-t5-text-to-text-transfer-transformer-2019万物皆可-text-to-text)
    - [5.2.1 任务统一设计：“Translate English to German: ...”](#521-任务统一设计translate-english-to-german-)
    - [5.2.2 跨距损坏掩码 (Span Corruption) 预训练目标推导](#522-跨距损坏掩码-span-corruption-预训练目标推导)
    - [5.2.3 相对位置偏置 (Relative Position Bias) 创新](#523-相对位置偏置-relative-position-bias-创新)
- [6. 推理动力学与系统服务复杂性：为什么工业界生成大模型倒向了 Decoder-Only？](#6-推理动力学与系统服务复杂性为什么工业界生成大模型倒向了-decoder-only)
  - [6.1 预填充 (Prefill) 与 解码 (Decode) 的双栈流水线分离困境](#61-预填充-prefill-与-解码-decode-的双栈流水线分离困境)
  - [6.2 异构 KV Cache 内存灾难：Encoder 静态 Cache 与 Decoder 动态追加 Cache](#62-异构-kv-cache-内存灾难encoder-静态-cache-与-decoder-动态追加-cache)
  - [6.3 算力与参数分配的冗余性：50% 算力沉睡在静态编码栈](#63-算力与参数分配的冗余性50-算力沉睡在静态编码栈)
- [7. PyTorch 白盒工程实现：从零手写标准 Transformer Encoder-Decoder](#7-pytorch-白盒工程实现从零手写标准-transformer-encoder-decoder)
  - [7.1 编码器栈与解码器栈代码实现](#71-编码器栈与解码器栈代码实现)
  - [7.2 自回归前向传播与端到端 Seq2Seq 推理循环](#72-自回归前向传播与端到端-seq2seq-推理循环)
- [8. 三大架构家族终极横向对比矩阵 (BERT vs T5 vs GPT)](#8-三大架构家族终极横向对比矩阵-bert-vs-t5-vs-gpt)
- [9. 专业实战测评题库 (含采分点与解析)](#9-专业实战测评题库-含采分点与解析)
- [10. 极简复习闪卡 (CheatSheet)](#10-极简复习闪卡-cheatsheet)

---

## 1. 学习记录流水线 (Changelog)
- **2026-09-17**：创建 Encoder-Decoder (Seq2Seq) 架构全景知识库文档；彻底纠偏“编码解码等于机器码与自然语言转换”的直觉误解；从数学和信息论角度重新定义 Tokenizer/Embedding 的协议层与 Encoder/Decoder 的深度表征变换机；深度拆解跨语言翻译、文本摘要等非对称任务场景的必然性；解构原始 Transformer 双栈协同机制与 Cross-Attention 桥梁；剖析 T5（Text-to-Text 统一与 Span Corruption）与 BART（去噪序列重构）；推导双引擎调度与异构 KV Cache 对线上并发吞吐的掣肘原因；提供纯 PyTorch 端到端 Seq2Seq 白盒代码实现与大厂面试考核题库。
- **2026-09-18（增量追加）**：增补第 4.5 节。系统拆解端到端微观计算数据流闭环；白盒解答 7 大核心疑难：编码器双向全局提纯输出静态 KV、解码器第一层面对纯空白时依赖 `<BOS>` 特殊引导符冷启动破局机理、交叉注意力中 Query 的物理本质（“当前需求提货单”而非人类词汇）、Cross-Attention 软对齐检索在算哪部分上下文更重要、输出张量是“混合上下文向量”而非概率、自回归逐词步进吐字循环，以及经过 LM Head 线性分类投影与 Softmax 采样最终映射回人类语言的完整数学闭环。

---

## 2. 认知纠偏与本质定性：什么是“编码”？什么是“解码”？

很多刚接触 NLP 和深度学习的同学，会产生一种非常朴素的**“计算机底层式误解”**：
> **经典直觉误区**：  
> 1. “编码（Encoding）不就是把人类看的自然语言，编译转化成计算机能懂的 01 二进制或者机器向量吗？”  
> 2. “解码（Decoding）不就是把计算机算出的内部机器向量，翻译还原成人类能看懂的人话吗？”  
> 3. “如果按这个逻辑，**BERT 叫 Encoder-Only，难道它只能输入、不会说人话？**”  
> 4. “**GPT 叫 Decoder-Only，难道它没有编码器，是怎么听懂人类输入的内容的？它不需要配一个对应的编码器吗？**”

**答案是：这个直觉把“协议转换（通信层）”与“深度特征变换（计算层）”完全搞混了！**

---

### 2.1 致命直觉误区：“编码是将语言变机器码，解码是将机器码变语言吗？”

在计算机网络和通信中，“编码/解码”确实指明文与密文、模拟信号与数字信号的转换。  
但是在**深度学习与 Transformer 的学术语境中**，概念完全不是这样的：

```
                    【真实的数据流架构层级划分】

  人类文字: "人工智能"
      │
      ▼ ═════════════════════════════════════════════════════
  【通信协议层: Tokenizer (分词) + Embedding (查表映射)】
      │  作用: 真正负责将自然语言符号 ──> 转化为计算机高维稠密浮点向量
      ▼ ═════════════════════════════════════════════════════
  稠密数字张量: X ∈ R^{L × d} (计算机完全可以计算的矩阵)
      │
      ├───────────────────────┬───────────────────────┐
      ▼                       ▼                       ▼
【计算层: Encoder 变换机】  【计算层: Decoder 变换机】  【通信协议层: LM Head 输出】
  - 关注: 全双向全局理解    - 关注: 单向因果自回归生成  - 作用: 将高维浮点向量
  - 拓扑: 任意词互见        - 拓扑: 未来被因果掩码遮蔽    还原为词表概率 Softmax
  - 代表: BERT              - 代表: GPT / LLaMA         - 输出: 重新变成人类可读文字
```

* **真正把“人类语言变成计算机数字”的组件是谁？**  
  **是 Tokenizer（分词器）和 Embedding 查表层！** 它把字符串切成子词 ID，再查表变成 $[L, d_{\text{model}}]$ 的浮点数矩阵。在进入 Encoder 或 Decoder 之前，数据**早已经是计算机语言（稠密数学向量）**了！
* **真正把“计算机数字变回人类语言”的组件是谁？**  
  **是 LM Head（线性分类头）+ 词表反查（Detokenizer）！** 它把隐藏向量投影到词表维度 $V$，通过 Argmax/采样 拿到词 ID，再反查转回字符串。

---

### 2.2 追问核心：“Decoder-Only 没有编码器，怎么接受人类输入？BERT 没有解码器，怎么输出人话？”

明白了上述分工，你心中的困惑就会瞬间烟消云散：

#### 1. Decoder-Only（GPT / LLaMA）没有编码器，怎么听懂人类输入？
* 人类输入的 Prompt（比如“请写一首诗”），**根本不需要经过任何专门的 Encoder**！
* 它直接通过 Tokenizer 变成 Token IDs，再经过 Embedding 变成连续向量，直接喂进 Decoder 栈的前几层；
* Decoder 拥有自注意力机制，Prompt 内部的词与词之间同样可以互相计算 Attention（在下三角掩码下，第 1 个词看自己，第 2 个词看第 1、2 个词，Prompt 内部的信息已被充分交互理解）；
* **在它眼里，人类输入的 Prompt 和它接下来要生成的 Answer，地位完全一样，都是同一根因果时间线上的 Token 序列！**

#### 2. BERT 叫 Encoder-Only，为什么它也能输出文本标签？
* BERT 包含 Embedding 模块（接受人类输入），也包含输出层（把隐向量投影回词表预测被挖空的字）；
* 它之所以叫 **Encoder-Only**，不是因为它不能输入输出，而是因为它的 Transformer 块内部**只有无拘无束的“全向双向注意力”，完全没有防止偷看未来的“因果掩码”，也没有“交叉注意力”**！

---

### 2.3 深入定性：Tokenizer + Embedding 才是通信协议，Encoder/Decoder 是高级特征变换机

在现代模型设计中，我们应当树立最专业的心智模型：
* **Tokenizer + Embedding**：底层的**硬件调制解调器（Modem）**，负责符号与向量之间的相互翻译；
* **Encoder（编码机）**：**“无向图特征提纯器”**。对输入序列做 $360^\circ$ 全方位无死角的深层语义提纯与压缩（特征到特征的映射）；
* **Decoder（解码机）**：**“有向因果自回归生成器”**。利用过去的历史状态，以步进方式自回归地“凭空展开”出下一个时间步的特征。

---

## 3. 序列到序列 (Seq2Seq) 哲学起源与应用场景必然性

既然纯 Decoder-Only 这么强大，为什么历史乃至现代某些领域依然需要 **Encoder-Decoder 双栈架构**？

### 3.1 为什么有些任务天生就是“非对称双序列映射”？

在自然语言的深水区，存在一类非常特殊的任务，具有强烈的**“非对称性”**：
1. **源序列（Source）与目标序列（Target）的语言体系完全异构**（如中文 $\to$ 英文）；
2. **源序列是固定的、静态的已知世界**，需要全盘考虑其语法主谓宾与文化内涵；
3. **目标序列是动态的、流式生成的未来世界**，必须严格按照时序因果逐字吐出；
4. **两者的长度完全不成比例**（例如将一篇 10,000 字的研报，压缩提炼为一段 200 字的核心摘要）。

这种**“输入端需要极致全局理解，输出端需要因果生成”**的任务，在学术上被称为 **序列到序列 (Sequence-to-Sequence, 简称 Seq2Seq)**。

---

### 3.2 经典核心场景：机器翻译、长文摘要、语音识别 (ASR) 与代码转换

```
【场景 1: 机器翻译 (Machine Translation)】
源语言 (中文, 全向理解): "这个 方案 必须 在 周五 之前 落实"
                               │ (通过 Cross-Attention 跨语系桥梁对齐)
                               ▼
目标语言 (英文, 自回归解码): "This plan must be implemented before Friday."

【场景 2: 抽取/生成式文本摘要 (Text Summarization)】
超长源文档 (输入 Encoder): [长达 5000 字的技术架构白皮书，一次性全局吸收]
                               │ (提取最密集的抽象语义知识库)
                               ▼
短篇摘要 (Decoder 逐字输出): "本文提出了一种新型稀疏注意力机制..." (长度仅 100 字)

【场景 3: 语音识别与跨模态生成 (Whisper / Audio-to-Text)】
声学信号 (输入 Encoder): [80 维 Log-Mel 频谱图音频帧，全局时频特征无死角编码]
                               │ (跨模态 Cross-Attention 对齐)
                               ▼
文本转录 (Decoder 自回归): "Hello and welcome to our podcast..."
```

---

### 3.3 为什么 Encoder-Decoder 在这类任务上具备天然压倒性优势？

在同等参数量下（如 5 亿到 30 亿参数的小规模基准）：
1. **Encoder 的“全双向注意力”彻底解放了源文本的理解力**：
   如果用纯 Decoder 做翻译，源语言中文被排在前头，在因果下三角 Mask 限制下，第 1 个中文词无法看到整句话的末尾连词与语气词，理解被严重割裂；而 Encoder 允许所有词自由互看，句法树建构极其牢固；
2. **静态上下文与动态生成彻底解耦**：
   无论 Decoder 生成了多少个词，Encoder 对源语言的分析只需在最开始**全量执行一次**并永久缓存为 Key/Value，架构职责泾渭分明，不易发生 Prompt 语义漂移。

---

## 4. 标准 Encoder-Decoder 架构拓扑深度解构 (Vaswani 2017)

### 4.1 宏观拓扑：编码栈 (Encoder Stack) 与 解码栈 (Decoder Stack) 的空间布局

原始 Transformer 本尊就是一个严丝合缝的 Encoder-Decoder 架构：

```
                       [输出词表概率分布 Logits]
                                  ▲
                       ┌──────────┴──────────┐
                       │  Linear Head (LM)   │
                       └──────────▲──────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │  Decoder Stack (重复 N 次)                    │
          │                                               │
          │      ┌─────────────────────────────┐          │
          │      │    Feed-Forward Network     │          │
          │      └──────────────▲──────────────┘          │
          │                     │ (+) 残差加法            │
          │      ┌──────────────┴──────────────┐          │
          │      │   LayerNorm / RMSNorm       │          │
          │      └──────────────▲──────────────┘          │
          │                     │                         │
          │      ┌──────────────┴──────────────┐          │
          │      │   Cross-Attention (交叉)    │◄───┐     │
          │      └──────────────▲──────────────┘    │     │
          │                     │ (+) 残差加法      │     │
          │      ┌──────────────┴──────────────┐    │     │
          │      │   Masked Causal Attention   │    │     │
          │      └──────────────▲──────────────┘    │     │
          └─────────────────────┼───────────────────┼─────┘
                                │                   │
                 [目标端已生成 Token (Shifted Right)] │
                                                    │
                 (来自 Encoder 最终输出的上下文向量) ───┘
                                ▲
          ┌─────────────────────┴─────────────────────────┐
          │  Encoder Stack (重复 N 次)                    │
          │                                               │
          │      ┌─────────────────────────────┐          │
          │      │    Feed-Forward Network     │          │
          │      └──────────────▲──────────────┘          │
          │                     │ (+) 残差加法            │
          │      ┌──────────────┴──────────────┐          │
          │      │ Bidirectional Self-Attention│          │
          │      │ (全向无掩码自注意力)        │          │
          │      └──────────────▲──────────────┘          │
          └─────────────────────┼─────────────────────────┘
                                ▲
                 [源端输入序列 (Source Text, e.g., 中文)]
```

---

### 4.2 编码器核心：全双向自注意力与语义压缩
对于源端序列 $X = (x_1, x_2, \dots, x_{T_x}) \in \mathbb{R}^{T_x \times d}$：
* 内部由 $N$ 个标准的 Encoder 块堆叠而成；
* 每一层均为全双向自注意力，掩码矩阵除 Padding 外全为 1；
* 最终输出隐状态张量 $H_{\text{enc}} \in \mathbb{R}^{T_x \times d}$。这个张量浓缩了源端所有词在当前全篇语境下的最高阶向量表征。

---

### 4.3 解码器双子层：掩码因果自注意力 + 交叉注意力桥梁 (Cross-Attention)
Decoder 栈的结构比 Encoder 复杂得多，每一个 Decoder 块内部包含**两个连续的注意力子层**：
1. **子层一：掩码因果自注意力（Masked Causal Self-Attention）**：
   - 仅仅作用在**目标端自身已经生成的历史序列**上；
   - 施加严格的因果下三角掩码，确保当前生成位置 $t$ 不能偷看 $t+1$ 之后的未来答案；
   - 产生目标端的自回归表征 $H_{\text{causal}} \in \mathbb{R}^{T_y \times d}$；
2. **子层二：交叉注意力（Cross-Attention）**：
   - 这是沟通编码栈与解码栈的**唯一灵魂跨界桥梁**！
   - 它的作用是将刚刚提纯好的目标需求向量，与源端编码器沉淀的历史上下文库进行深度对齐。

---

### 4.4 交叉注意力微观数据流：Query 来自目标，Key/Value 来自源端源泉

在这一层中，张量计算发生精妙的异源分流：

$$
Q = H_{\text{causal}} \cdot W_Q \in \mathbb{R}^{T_y \times d_k} \quad (\mathbf{来自解码器自身})
$$
$$
K = H_{\text{enc}} \cdot W_K \in \mathbb{R}^{T_x \times d_k}, \quad V = H_{\text{enc}} \cdot W_V \in \mathbb{R}^{T_x \times d_v} \quad (\mathbf{来自编码器输出})
$$

计算注意力权重矩阵：
$$
\text{Attention Matrix } A = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{T_y \times T_x}
$$

* **矩阵物理含义**：
  行数代表**目标语言当前写出的每个单词**，列数代表**源语言原文的每个单词**。
  第 $i$ 行的权重，精准指示了“为了翻译出目标语言第 $i$ 个词，模型应当把目光投向源语言文章中的哪些词”；
* 加权求和输出：$Z = A V \in \mathbb{R}^{T_y \times d_v}$，其序列长度严格为 $T_y$。

---

### 4.5 动态端到端全流程微观推演：从损坏输入到逐词吐字还原真实人话

很多工程师在学习 Encoder-Decoder（如 BART/T5 修复加噪损坏文本、或标准机器翻译）时，心中都会冒出一连串**极度硬核、环环相扣的微观疑问**：
> 1. 编码器的作用是不是先进行双向编码理解序列，输出隐状态张量，用于后续的 $K$ 和 $V$？
> 2. 解码器产生的 **Query 到底是个什么东西**？它是一段文字吗？
> 3. 在解码器第一层中，面对尚未生成的空白，子层一是怎么预测第一个 Token 的？（冷启动从何而来？）
> 4. 交叉注意力中，$Q$ 和编码器的每个 $K, V$ 计算，是在算哪个 Token 需要更加注意吗？
> 5. 交叉注意力最后输出的，难道就是最终的预测词吗？
> 6. 那个被掩盖或要翻译的输出序列，是**逐 Token 生成**的吗？
> 7. 每一步生成是不是都是先拿到词向量稠密表示，最后才通过词表映射回人类语言？

我们以一个最经典的去噪修复任务为例，全程进行**单步微观白盒推导**：
* **任务目标**：输入损坏句子 `"巴黎 是 [MASK] 的 首都"`，要求模型输出完整纯净句子 `"巴黎 是 法国 的 首都"`。

---

#### 4.5.1 第一阶段：编码栈的双向深层语义提纯与静态 KV 固化

**答案：完全正确！编码器的使命就是彻底理解输入，并制造静态的“外部知识库”。**

1. 损坏句子经过分词（Tokenizer）和嵌入（Embedding）得到数字矩阵 $X \in \mathbb{R}^{5 \times d}$；
2. 输入到 $N$ 层编码器（Encoder Stack）。因为**全双向自注意力完全没有时序掩码**，`"巴黎"` 可以看到 `"[MASK]"`，`"首都"` 也可以看到 `"巴黎"`；
3. 经过 $N$ 层的多头自注意力与 FFN 非线性提炼，最终输出编码隐状态：
   $$
   H_{\text{enc}} \in \mathbb{R}^{5 \times d}
   $$
4. **关键定性**：
   这个 $H_{\text{enc}}$ 并不是人类能看懂的字，而是**一张立体的、高度抽象的语义全景图**。
   随后它通过权重矩阵线性映射出：
   $$
   K_{\text{enc}} = H_{\text{enc}} W_K \in \mathbb{R}^{5 \times d_k}, \quad V_{\text{enc}} = H_{\text{enc}} W_V \in \mathbb{R}^{5 \times d_v}
   $$
   **这两个矩阵在后续解码的整个过程中，永远保持只读、静态固化，再也不需要重新计算！**

---

#### 4.5.2 第二阶段：解码栈的第一层子层一如何启动？（首个 Token 冷启动悖论）

**疑问核心**：“在刚开始的时候，解码器明明什么字都还没生成，它怎么有东西输入到子层一去预测第一个词？”

**真相：依靠人造的特殊起始引导符（`<BOS>` 或 `<SOS>` 或 `<pad>`）实行冷启动！**

```
【第 1 步 (Step 1) 的微观数据流】

解码器输入 (纯粹的人造哨兵):
  Token: [<BOS>] (Begin Of Sentence, 句子起始符, ID 通常为 1 或 0)
     │
     ▼
  Embedding 查表 ──> 拿到词向量 y_0 ∈ R^{1 × d}
     │
     ▼
  【子层一: 因果自注意力】
     - 此时输入序列长度只有 1 个 Token（就是 [<BOS>] 自己）；
     - 自己跟自己做自注意力，注意力权重 100% 落在自己身上；
     - 输出经过残差与 LayerNorm，得到第一个中间隐状态 h_causal^{(0)} ∈ R^{1 × d}。
```

> [!IMPORTANT]
> **破除迷思**：解码器的子层一（因果自注意力）**在这一步根本不需要、也无法预测具体的词**！它的作用仅仅是**把当前已经存在的历史 Token（在第一步只有 `<BOS>`），加工成一个具备上下文感知的内部特征向量**！

---

#### 4.5.3 物理本质定性：交叉注意力里的 Query 到底是个什么东西？

**疑问核心**：“解码器产生的 Query 到底是个什么东西？它是一段文字吗？”

**真相：Query 绝对不是人类的自然语言文字，它是一个“高维语义需求探针（Semantic Need Probe）”或“提货单”！**

1. 在第 1 步中，子层一处理完 `[<BOS>]` 后，拿到了特征向量 $h_{\text{causal}}^{(0)} \in \mathbb{R}^{1 \times d}$；
2. 乘以交叉注意力的查询投影矩阵：
   $$
   Q = h_{\text{causal}}^{(0)} \cdot W_Q \in \mathbb{R}^{1 \times d_k}
   $$
3. **Query 的物理含义翻译成大白话**：
   此时经过训练的神经网络，这个 $Q$ 向量在高维几何空间中的方向，代表着一个强烈的意图信号：
   > **“我是整个句子的开头（刚从 `<BOS>` 起步），请问原文在讲什么主体？我需要抓取整句话开头的核心语义实体！”**

---

#### 4.5.4 跨界检索机制：注意力打分在算什么？输出序列如何生成？

**疑问核心**：“交叉注意力是在算哪个 Token 需要更加注意吗？最后输出的就是预测的词吗？”

1. **注意力打分究竟在算什么？**  
   **完全正确！它就是在计算“当前要生成的这个位置，到底该重点关注原文里的哪几个 Token！”**
   我们将这唯一的 $Q \in \mathbb{R}^{1 \times d_k}$，与编码器静态保存的 5 个 Token 的 $K_{\text{enc}} \in \mathbb{R}^{5 \times d_k}$ 做点积并 Softmax：
   $$
   \text{Attn\_Weights} = \text{softmax}\left( \frac{Q \cdot K_{\text{enc}}^T}{\sqrt{d_k}} \right) = [0.85, \; 0.05, \; 0.03, \; 0.02, \; 0.05]
   $$
   *看！权重第 0 位（0.85）极其巨大，这表明解码器自发发现：“要写开头的第一个词，必须把 85% 的精力聚焦在原文的‘巴黎’上！”*

2. **交叉注意力最后输出的是什么？**  
   **它输出的绝不是最终的单词，而是一个“加权融合后的上下文特征向量（Context Vector）”！**
   $$
   Z = \text{Attn\_Weights} \cdot V_{\text{enc}} = 0.85 \times \vec{v}_{\text{巴黎}} + 0.05 \times \vec{v}_{\text{是}} + \dots \in \mathbb{R}^{1 \times d}
   $$
   这个 $Z$ 向量，把原文中“巴黎”的核心语义深深地刻印了进来；
3. 随后这个 $Z$ 经过残差连接、LayerNorm、FFN，在解码栈中层层传递，直到穿透最后一层 Decoder，输出最终的顶层稠密向量 $h_{\text{out}} \in \mathbb{R}^{1 \times d}$。

---

#### 4.5.5 终极还原闭环：从稠密特征向量到人类自然语言的投影与反查

**疑问核心**：“每一步生成是不是都是先得到对应的词向量稠密表示，随后通过协议/词表得到输出的人类语言？是逐 Token 生成的吗？”

**完全正确！这是一个不折不扣的标准自回归逐步吐字循环（Auto-Regressive Loop）：**

```
                           【第 1 步的最终吐字】
顶层隐藏向量 h_out ∈ R^{1 × d} (包含强烈"巴黎"语义的纯数字)
       │
       ▼
【语言模型头 (LM Head)】: 线性变换 W_vocab ∈ R^{d × V} (V 为词表大小, 如 32,000)
       │
       ▼
全词表未归一化打分 (Logits): L ∈ R^{1 × V}
       │
       ▼
Softmax 归一化概率: P = softmax(L) ∈ R^{1 × V}
       │
       ▼
贪婪采样 (Argmax): 找到概率最大的单词 ID (假设 ID 888 对应的单词是 "巴黎")
       │
       ▼
词表反查 (Detokenizer): 吐出人类可读文本: "巴黎"！
```

---

#### 4.5.6 完整的后续时间步滚雪球自回归循环

此时模型只吐出了第 1 个字，后续是如何修复被掩盖的词的？

* **第 2 步 (Step 2)**：
  - 解码器输入变为两个 Token：`[<BOS>, "巴黎"]`；
  - 子层一（自注意力）让 `"巴黎"` 结合 `<BOS>`，形成新的 Query：*“我已经写了‘巴黎’，接下来该接什么谓语动词？”*；
  - 交叉注意力检索原文，高概率聚焦在 `"是"` 上；
  - 解码器吐出第 2 个词：`"是"`。
* **第 3 步 (Step 3 - 最精彩的修复填空时刻！)**：
  - 解码器输入变为三个 Token：`[<BOS>, "巴黎", "是"]`；
  - 子层一提取新的 Query：*“前文是‘巴黎是...’，且原句对应位置是 `[MASK]`，紧接着后面是‘首都’，那么‘巴黎是某某的首都’，这个空应该填什么国名？”*；
  - 交叉注意力检索编码器的全局全景图（编码器早已把巴黎和首都的逻辑暗线打通），模型在此处以绝对高概率吐出：`"法国"`！
* **持续循环**：
  - 依次吐出 `"的"`、`"首都"`；
  - 直到第 6 步，解码器根据完整语义输出特殊的终止标记 `<EOS>`（End Of Sentence）；
  - **自回归生成终止，完整的纯净自然语言呈现：“巴黎 是 法国 的 首都”**。

---

#### 4.5.7 核心疑问与微观认知对照清单

| 你的疑问问题 | 真实微观底层机理 | 经典心智模型 |
| :--- | :--- | :--- |
| **编码器在干嘛？** | 全双向注意力提炼输入，输出不可变的 $K, V$ 静态矩阵。 | 制造一个完整的“参考资料档案馆”。 |
| **Query 到底是什么？** | 当前解码位置由历史 Token 投射出的**高维语义需求向量**。 | 拿着写好的草稿去档案馆开具的“提货单/搜索关键词”。 |
| **第一步怎么预测？** | 依赖特殊起始符 `<BOS>` 消除空白，强行冷启动。 | 敲下第 1 个按键，给流水线送入第 1 个工件。 |
| **交叉注意力在算什么？** | 算当前生成的单词和输入原文哪部分语义最具相关度。 | 软对齐（Soft Alignment），决定目光聚焦在哪里。 |
| **输出是概率还是词？** | 交叉注意力只输出混合特征向量；最后由 **LM Head** 投影到全词表选出概率最大的 Token。 | 特征提纯完成后，最后交给印刷机打印成字。 |
| **是不是逐词生成？** | **必须是！** 严格按时间步自回归步进，新吐出的词立刻加入下一轮的输入，直到吐出 `<EOS>`。 | 滚雪球式因果逐字展开。 |


在工业大模型爆发的前夜，Encoder-Decoder 体系诞生了两大巅峰神作：**Facebook 的 BART** 与 **Google 的 T5**。

### 5.1 BART (2019)：双向自编码器 + 自回归解码器的去噪大一统

Meta 团队发布的 BART (Bidirectional and Auto-Regressive Transformers) 巧妙地将 **BERT 的双向去噪能力** 与 **GPT 的自回归生成能力** 熔于一炉：

```
[原始纯净文档]: "The cat sat on the mat"
       │
       ▼ (注入五种恶意噪声破坏)
[损坏后的序列]: "The [MASK] sat on [MASK] mat" / "the mat The cat sat on" (句子乱序)
       │
       ▼ (送入 Encoder)
[双向 Encoder 深度全向编码]
       │
       ▼ (Cross-Attention 传递上下文)
[自回归 Decoder 逐步还原全量纯净文档]: "The cat sat on the mat" (计算生成损失)
```

**BART 的五大去噪扰动技巧**：
1. **Token Masking**：随机将 Token 替换为 `[MASK]`（类似 BERT）；
2. **Token Deletion**：直接物理删除某些单词（模型必须学会自发检测漏词）；
3. **Text Infilling**：用一个单标签 `[MASK]` 替换一段不定长连续切片（甚至包含 0 个词）；
4. **Sentence Permutation**：将段落内的句子打乱顺序（强迫模型学习因果逻辑）；
5. **Document Rotation**：随机挑选一个词作为开头并将文本旋转（强迫模型识别篇章起点）。

---

### 5.2 T5 (Text-to-Text Transfer Transformer, 2019)：万物皆可 Text-to-Text

Google 团队的 T5（Colin Raffel 等人）是工业 NLP 思想的一次登峰造极的大一统。

#### 5.2.1 任务统一设计：“万物皆可文字输入，文字输出”
在 T5 之前，分类任务输出标量类别，回归任务输出实数，翻译任务输出文本。  
T5 首次提出：**所有 NLP 问题都可以被硬性重构为“输入一段纯文本，输出一段纯文本（Text-to-Text）”！**

```
【分类任务】
输入: "cola sentence: The course is jumping well."
输出目标文本: "unacceptable"

【情感分析】
输入: "sst2 sentence: It is a truly charming movie."
输出目标文本: "positive"

【机器翻译】
输入: "translate English to German: That is good."
输出目标文本: "Das ist gut."

【机器问答】
输入: "question: What is the capital of France? context: Paris is the capital of France."
输出目标文本: "Paris"
```

#### 5.2.2 跨距损坏掩码 (Span Corruption) 预训练目标推导
T5 摒弃了单个词掩码的传统，采用了极其高效的 **跨距破坏（Span Corruption）** 机制：
* 样本中约 15% 的连续词段被挖掉，并用专属哨兵标记（Sentinel Tokens，如 `<X>`, `<Y>`, `<Z>`）替代；
* **Encoder 仅负责读入被挖空后的损坏文本**；
* **Decoder 的学习目标是仅输出被挖掉的词段及对应的哨兵标签**！

```
原始文本: "Thank you for inviting me to your party last week"
损坏输入 (Encoder): "Thank you <X> me to your party <Y> week"
目标输出 (Decoder): "<X> for inviting <Y> last <Z>"
```
* **算力极其高效**：Decoder 不需要费时费力地把 Encoder 原文完完整整复读一遍，它只需要高密度地预测被挖走的精华 Token，大幅削减了 Decoder 的训练开销！

#### 5.2.3 相对位置偏置 (Relative Position Bias) 创新
T5 彻底放弃了绝对位置编码，提出了一种简单优雅的**可学习相对位置偏置标量**：
直接在注意力分数矩阵上加上一个与相对距离 $i - j$ 绑定的标量偏置 $b_{i-j}$：
$$
\text{Attention}(Q, K) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + B_{\text{rel}} \right)
$$
这个设计具备极佳的长距离泛化性能，直接启发了后来的 ALiBi 等相对位置偏置机制。

---

## 6. 推理动力学与系统服务复杂性：为什么工业界生成大模型倒向了 Decoder-Only？

回到最核心的商业与系统问题：**既然 Encoder-Decoder 在翻译和理解上这么强，为什么在今天的百亿千亿大模型（LLaMA-3、DeepSeek、GPT-4）时代，它几乎在通用语言模型领域全面绝迹了？**

答案不在于“智商高低”，而在于**“线上服务推理的底层硬件内存墙与调度灾难”**！

### 6.1 预填充 (Prefill) 与 解码 (Decode) 的双栈流水线分离困境

在基于 vLLM、TGI 等现代高并发大模型推理框架中：
* **Decoder-Only 架构**：全网只有一种统一的算子块。Prefill（处理输入）和 Decode（生成输出）在同一个网络结构上运行，批处理动态合并（Continuous Batching）调度非常纯粹；
* **Encoder-Decoder 架构**：
  - 输入必须先在 Encoder 上完整跑一遍静态矩阵乘法；
  - 然后再把数据调度交给 Decoder 跑自回归解码循环；
  - 两个子模块在硬件执行上的吞吐量、计算强度（FLOPs/Byte）严重不对称，异构流水线调度导致 GPU 显存带宽与 Tensor Core 发生严重的**气泡等待（Pipeline Bubbles）**。

---

### 6.2 异构 KV Cache 内存灾难：Encoder 静态 Cache 与 Decoder 动态追加 Cache

在 Decoder-Only 架构中，KV Cache 是高度均质、极易切块分页管理的（如 PagedAttention）：
* 每一个 Token 对应一份 $[K, V]$，随着时间单调向后追加。

而在 Encoder-Decoder 架构中，系统必须在 GPU 显存池中**维护两份完全不同性质的异构 KV Cache**：
1. **Cross-Attention 的 Encoder KV Cache**：
   - 形状由输入的 Source Prompt 长度决定，长度为 $T_x$；
   - 整个生成期间是**静态固定只读的**；
2. **Self-Attention 的 Decoder KV Cache**：
   - 形状随着自回归生成动态增长，长度从 $1$ 逐渐膨胀到 $T_y$；
3. **显存碎片化与调度复杂度翻倍**：
   管理两套完全不同生命周期、不同增长速率的显存内存块，导致显存碎片急剧增加，难以最大化 Batch Size，线上并发吞吐量相比纯 Decoder 遭受腰斩。

---

### 6.3 算力与参数分配的冗余性：50% 算力沉睡在静态编码栈

假设你有一个 70B 参数的 Encoder-Decoder 模型（如 35B Encoder + 35B Decoder）：
* 当用户发起一次生成 1000 个字的推理长任务时：
  - **35B 的 Encoder 仅在最开始的第 1 个 Token 计算中参与了一瞬间**；
  - 在后续漫长的 999 步自回归解码循环中，**35B 的 Encoder 参数完全静止沉睡，但依然死死霸占着宝贵的数百 GB 高速显存（HBM）**！
* **硬件投资回报率极低**：
  相比之下，70B 的 Decoder-Only 模型的全部 70B 参数在每一步生成时都被 100% 调动参与深度推理，算力利用效率和参数参数密度具有压倒性的经济学优势！

---

## 7. PyTorch 白盒工程实现：从零手写标准 Transformer Encoder-Decoder

以下代码给出一个纯 PyTorch 手写实现的极简 Seq2Seq Transformer 骨架，白盒展示 **Encoder 自注意力、Decoder 因果自注意力与 Cross-Attention 交叉检索的张量流转全过程**。

### 7.1 编码器栈与解码器栈代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    通用多头注意力模块: 既可充当自注意力 (Q=K=V), 也可充当交叉注意力 (Q来自解码端, K/V来自编码端)
    """
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q_in, k_in, v_in, mask=None):
        B, Tq, _ = q_in.shape
        _, Tk, _ = k_in.shape
        
        # 1. 投影切分为多头 [B, n_head, T, d_k]
        q = self.w_q(q_in).view(B, Tq, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k_in).view(B, Tk, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v_in).view(B, Tk, self.n_head, self.d_k).transpose(1, 2)
        
        # 2. 缩放点积
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_head, Tq, Tk]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.w_o(out)


class EncoderLayer(nn.Module):
    """Encoder 块: 双向自注意力 + 前馈网络 (全双向，无因果掩码)"""
    def __init__(self, d_model=512, n_head=8, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # 纯双向交互
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=src_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    """Decoder 块: 因果自注意力 + 跨序列交叉注意力 + 前馈网络"""
    def __init__(self, d_model=512, n_head=8, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, y, enc_out, tgt_causal_mask=None, src_pad_mask=None):
        # 1. 目标端自身因果自注意力 (只能看过去)
        y = y + self.self_attn(self.ln1(y), self.ln1(y), self.ln1(y), mask=tgt_causal_mask)
        # 2. 交叉注意力跨界桥梁 (Q来自自己, K/V来自Encoder输出)
        y = y + self.cross_attn(self.ln2(y), enc_out, enc_out, mask=src_pad_mask)
        # 3. 前馈非线性映射
        y = y + self.mlp(self.ln3(y))
        return y
```

---

### 7.2 自回归前向传播与端到端 Seq2Seq 推理循环

```python
class Seq2SeqTransformer(nn.Module):
    """标准端到端序列到序列 Transformer"""
    def __init__(self, src_vocab=32000, tgt_vocab=32000, d_model=512, n_layers=6, n_head=8):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.encoder_stack = nn.ModuleList([EncoderLayer(d_model, n_head) for _ in range(n_layers)])
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model, n_head) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, tgt_vocab, bias=False)

    def encode(self, src_tokens, src_mask=None):
        x = self.src_emb(src_tokens)
        for layer in self.encoder_stack:
            x = layer(x, src_mask)
        return x  # enc_out: [B, Tx, d_model]

    def decode(self, tgt_tokens, enc_out, tgt_mask=None, src_mask=None):
        y = self.tgt_emb(tgt_tokens)
        for layer in self.decoder_stack:
            y = layer(y, enc_out, tgt_causal_mask=tgt_mask, src_pad_mask=src_mask)
        return self.lm_head(y)  # logits: [B, Ty, tgt_vocab]

    @torch.no_grad()
    def generate(self, src_tokens, start_token_id, end_token_id, max_len=50):
        """端到端自回归翻译/生成解码"""
        self.eval()
        # 1. 编码栈仅执行单次 Prefill 编码
        enc_out = self.encode(src_tokens)
        
        # 2. 解码栈从 <BOS> 启动逐字自回归吐字
        B = src_tokens.size(0)
        generated_tokens = torch.tensor([[start_token_id]], device=src_tokens.device).repeat(B, 1)
        
        for _ in range(max_len):
            Ty = generated_tokens.size(1)
            # 生成下三角因果掩码
            causal_mask = torch.tril(torch.ones(Ty, Ty, device=src_tokens.device)).view(1, 1, Ty, Ty)
            
            # 前向解码
            logits = self.decode(generated_tokens, enc_out, tgt_mask=causal_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            if (next_token == end_token_id).all():
                break
                
        return generated_tokens
```

---

## 8. 三大架构家族终极横向对比矩阵 (BERT vs T5 vs GPT)

| 对比维度 | Encoder-Only (BERT) | Encoder-Decoder (T5 / BART) | Decoder-Only (GPT / LLaMA) |
| :--- | :--- | :--- | :--- |
| **注意力拓扑构成** | **100% 全双向无掩码** | **双模**：Encoder 全向 + Decoder 因果/交叉 | **100% 严格因果下三角掩码** |
| **跨序列交互方式** | 在输入层用 `[SEP]` 强行拼接序列 | **通过 Cross-Attention 算子硬件级交互** | 全部串联在同一个单向因果流中 |
| **典型黄金代表** | BERT, RoBERTa, DeBERTa | T5, mT5, BART, Whisper (语音识别) | GPT-1~4, LLaMA-1~3, DeepSeek, Qwen |
| **理论核心优势** | 对整句上下文具有最严密的深层双向理解力 | 源端全局理解 + 目标端自由生成的解耦闭环 | 训练推理同构、长上下文拓展容易、KV Cache 极度高效 |
| **理论致命短板** | 无法自回归生成连续长文本（曝光偏差） | 线上服务异构双 KV Cache，显存利用率低 | 对 Prompt 的理解必须受制于因果单向掩码 |
| **现代工业地位** | 专精于 **RAG 向量检索与重排** (Embedding/Rerank) | 专精于 **语音多模态 ASR** (Whisper)、代码转换与翻译 | **统治 99% 以上的百亿/千亿通用生成大模型** |

---

## 9. 专业实战测评题库 (含采分点与解析)

### 题 1 【架构本质与机制辨析题】
**题目**：很多初学者认为“BERT 是编码器，只能把自然语言编码成机器码；而 GPT 是解码器，需要搭配一个编码器才能把人类语言还原”。请从模型底层的数据流和架构算子角度，彻底批驳这一错误观点。

**采分点与解析**：
1. **协议层与计算层解耦（4分）**：
   - 自然语言符号到连续浮点向量的转换是由 **Tokenizer + Embedding 查表层** 完成的；
   - 机器内部向量到自然语言的还原是由 **LM Head 分类投影层 + Detokenizer** 完成的；
   - 编码和解码在底层处理的都是连续的数学高维张量，根本不存在“一个是机器码、一个是人话”的分别。
2. **GPT 与 BERT 的计算拓扑本质（6分）**：
   - GPT 叫 Decoder-Only，是指它的网络块从始至终只采用“因果下三角自注意力掩码”，Prompt 与输出直接在同一根自回归流中同质流动，它原生即可接受人类输入；
   - BERT 叫 Encoder-Only，是指它全网只采用“全双向无限制注意力”，它通过 MLM 分类头同样可以预测词表输出；
   - “Only” 指代的是 Transformer 内部注意力的**时序掩码约束模式**，而非指代外部接口的功能缺失。

---

### 题 2 【工业系统落地与推理瓶颈题】
**题目**：在现代高并发大模型推理服务（如使用 vLLM 或 PagedAttention）中，为什么同样是 70B 参数规模，Encoder-Decoder 架构在工程落地上的吞吐量与显存利用率显著劣于纯 Decoder-Only 架构？

**采分点与解析**：
1. **显存利用率与参数空转（5分）**：
   - 在 Encoder-Decoder 中，庞大的 Encoder 权重（约占总参数的 50%）仅在处理 Prompt 的 Prefill 阶段运行一次，而在长达数百步的自回归吐字 Decode 阶段完全空转，却持续占据数百 GB 宝贵的 HBM 显存；
   - Decoder-Only 的全部参数在每一步生成时都在 100% 全速运算，硬件投资回报率（ROI）显著更高。
2. **异构 KV Cache 内存调度困境（5分）**：
   - 系统必须在显存中同时维护 Cross-Attention 的**静态全量 Encoder KV Cache** 与 自注意力的**动态单调追加 Decoder KV Cache**；
   - 两种 Cache 的生命周期、内存块大小与增长行为完全不一致，极易导致严重的显存碎片化，难以进行高效的连续批处理（Continuous Batching）调度。

---

## 10. 极简复习闪卡 (CheatSheet)

| 核心概念/机制 | 物理原理与核心规律 | 考点与工程设计心智 |
| :--- | :--- | :--- |
| **通信协议 vs 计算机** | Tokenizer/Embedding 负责语言/向量转换 | Encoder 负责无向全局提纯，Decoder 负责有向因果逐字展开。 |
| **Seq2Seq 核心拓扑** | Encoder (全双向) + Decoder (因果自注意力 + 交叉注意力) | 完美契合机器翻译、长文摘要与语音识别等输入输出不对称场景。 |
| **Cross-Attention 数据源** | $Q$ 来自目标解码器已生成序列，$K, V$ 来自源端编码器输出 | 唯一跨界桥梁，加权注意力矩阵指示目标词聚焦源端哪个词。 |
| **T5 的万物大一统** | Text-to-Text 统一所有任务，跨距损坏 (Span Corruption) | 训练时 Decoder 仅输出被挖走的哨兵词段，大幅节省预训练 FLOPs。 |
| **双栈架构为何衰退** | 异构双 KV Cache 导致显存碎片化，Encoder 在生成期空转 | 线上推理并发吞吐被砍半，最终让位于统一流线型的 Decoder-Only。 |


# 实验一：Word2Vec 的标准高速化实现与底层原理透视

> 本实验隶属于 [Experiment-for-learning](https://github.com/JiangZi0721/Experiment-for-learning) 核心技术透视系列。

本项目是专为深度学习与自然语言处理初学者打造的 **Word2Vec 高速化完整工程实现**。项目完全基于 Python 和 **纯 NumPy** 开发，不依赖任何第三方深度学习框架 (如 PyTorch / TensorFlow)，旨在清晰、透明、严谨地揭示现代自然语言处理词向量技术底层的数学原理与算法细节。

---

## 目录
- [1. 为什么需要高速化？(核心痛点与瓶颈分析)](#1-为什么需要高速化核心痛点与瓶颈分析)
- [2. 三大核心高速化创新](#2-三大核心高速化创新)
  - [2.1 Embedding 层 (替代低效的 One-Hot 矩阵乘法)](#21-embedding-层-替代低效的-one-hot-矩阵乘法)
  - [2.2 多分类转二分类 (SigmoidWithLoss)](#22-多分类转二分类-sigmoidwithloss)
  - [2.3 负采样技术 (Negative Sampling) 与 0.75 次幂平滑](#23-负采样技术-negative-sampling-与-075-次幂平滑)
- [3. 深入剖析与模拟：CBOW 前向预测全流程 (以窗口为 6 为例)](#3-深入剖析与模拟cbow-前向预测全流程-以窗口为-6-为例)
  - [3.1 架构高清数据流图解](#31-架构高清数据流图解)
  - [3.2 答疑一：上下文 12 个向量是【相加】还是【并列拼接】？](#32-答疑一上下文-12-个向量是相加还是并列拼接)
  - [3.3 答疑二：与 W_out 运算后，最终得到的到底是什么？](#33-答疑二与-w_out-运算后最终得到的到底是什么)
  - [3.4 端到端完整数值与张量流转模拟 (Step-by-Step Simulation)](#34-端到端完整数值与张量流转模拟-step-by-step-simulation)
- [4. PTB 数据集与文本预处理流水线](#4-ptb-数据集与文本预处理流水线)
- [5. 项目整体代码结构](#5-项目整体代码结构)
- [6. 核心数学推导一览](#6-核心数学推导一览)
- [7. 快速开始与使用指南](#7-快速开始与使用指南)
  - [7.1 环境准备](#71-环境准备)
  - [7.2 运行核心层单元测试 (梯度检验)](#72-运行核心层单元测试-梯度检验)
  - [7.3 极速上手 Demo (5秒体验完整闭环)](#73-极速上手-demo-5秒体验完整闭环)
  - [7.4 完整语料库训练 (train.py 命令行参数说明)](#74-完整语料库训练-trainpy-命令行参数说明)
  - [7.5 CBOW 与 Skip-Gram 对比基准测试 (compare_models.py)](#75-cbow-与-skip-gram-对比基准测试-compare_modelspy)
  - [7.6 词向量语义评测与词类比推理 (eval.py)](#76-词向量语义评测与词类比推理-evalpy)
  - [7.7 交互式语义代数计算器 (word_math_cli.py)](#77-交互式语义代数计算器-word_math_clipy)
  - [7.8 词向量空间 2D 降维可视化 (visualize.py)](#78-词向量空间-2d-降维可视化-visualizepy)
- [8. 常见问题与避坑指南 (FAQ)](#8-常见问题与避坑指南-faq)

---

## 1. 为什么需要高速化？(核心痛点与瓶颈分析)

在简单的朴素 Word2Vec (如简单 CBOW) 中，主要存在 **两大不可承受的计算瓶颈**：

```
[朴素 Word2Vec 瓶颈]
输入层: One-hot 向量 (1, V)  @  权重矩阵 W_in (V, H)   ===> O(V × H) 的巨大开销！
中间隐层: h (1, H)
输出层: Softmax( h @ W_out ) 全词表归一化               ===> 分母需要累加 V 个 exp(z_i)，开销 O(V)！
```

假设词表大小 $V = 100,000$，隐藏层维度 $H = 100$：
1. **输入端矩阵乘法严重浪费算力**：One-hot 向量只有 1 位是 1，其余 99,999 位全为 0。做矩阵乘法实质上只是在提取 $W_{in}$ 的某一行，但计算开销却是 $100,000 \times 100$ 次乘加运算。
2. **输出端全词表 Softmax 归一化极慢**：Softmax 分母为 $\sum_{k=1}^V \exp(s_k)$，每处理一个词都要计算 100,000 次指数与求和，反向传播时还需遍历所有词，计算代价呈线性爆炸 $O(V)$。

因此，**Word2Vec 的标准高速化方案**应运而生。

---

## 2. 三大核心高速化创新

### 2.1 Embedding 层 (替代低效的 One-Hot 矩阵乘法)

- **前向传播原理**：
  直接通过索引切片提取权重行：`out = W[idx]`。
  计算复杂度由原本的 $O(V \times H)$ 瞬间降为 $O(1)$。
- **反向传播核心细节 (关键陷阱)**：
  在反向传播中，上游传来的梯度 `dout` 必须累加回 `dW` 的对应行。
  > **注意**：在一个批次 (Batch) 或上下文窗口中，**同一个单词可能会出现多次**！
  > 如果直接写 `dW[idx] = dout`，后出现的词梯度会直接把前面出现的词梯度**覆盖掉**！
  > 因此，本项目使用 NumPy 的原子级累加函数：
  ```python
  np.add.at(self.dW, self.idx, dout)
  ```

---

### 2.2 多分类转二分类 (SigmoidWithLoss)

将“在全词表 $V$ 个候选词中挑出哪一个是正确词”的 **$V$ 分类问题**，重构为 **二分类问题 (Binary Classification)**：
> **“给定上下文隐层向量 $h$，当前词 $w$ 是否是真实出现的上下文词？”**
- **正样本**：真实在文本中共同出现的词，监督标签 $t = 1$。
- **负样本**：随机采样的无关词，监督标签 $t = 0$。

此时输出层不再使用全量 Softmax，而是使用点乘层 (`EmbeddingDot`) 计算中间向量 $h$ 与目标词向量 $W_{out}[t]$ 的内积，再送入 `Sigmoid` 函数：
$$y = \sigma(x) = \frac{1}{1 + e^{-x}}$$
$$L = - \left[ t \log(y) + (1 - t) \log(1 - y) \right]$$

---

### 2.3 负采样技术 (Negative Sampling) 与 0.75 次幂平滑

如果只对正样本进行二分类，模型会迅速退化（直接把所有词的预测得分都置为正无穷大）。为了让模型具备鉴别能力，必须引入**负样本**。

1. **采多少个负样本？**
   对于每个正样本，随机抽取 $K$ 个无关词作为负样本（小数据集推荐 $K = 5 \sim 20$；大规模数据推荐 $K = 2 \sim 5$）。
   计算复杂度直接从 $O(V)$ 锐减至 $O(K + 1)$！当 $V=100,000, K=5$ 时，**计算量缩小了约 20,000 倍**！

2. **如何抽取负样本？(为什么是 0.75 次幂？)**
   若按均匀分布采样，无法反映自然语言中高频词更多的特点；
   若直接按原始词频采样，停用词 (如 "the", "a", "of") 会霸占几乎所有的负样本名额，低频关键语义词永远得不到更新机会。
   Mikolov 等人经实证提出 **0.75 次幂平滑 (3/4 次方)**：
   $$P'(w_i) = \frac{[P(w_i)]^{0.75}}{\sum_j [P(w_j)]^{0.75}}$$
   - **效果示例**：
     假设高频词频为 $10000$，低频词频为 $100$。
     原始词频比例：$10000 / 100 = 100$ 倍。
     经 0.75 次幂后：$10000^{0.75} = 1000$，$100^{0.75} \approx 31.6$。
     平滑后比例变为 $1000 / 31.6 \approx 31.6$ 倍。
     **低频词被采样的相对几率提升了 3 倍以上！**

---

## 3. 深入剖析与模拟：CBOW 前向预测全流程 (以窗口为 6 为例)

为了让学习者直观透彻地掌握 CBOW 模型的真实计算机制，我们针对读者最关心的两个核心疑问：
> **1. 上下各 6 个单词向量和权重矩阵相乘之后得到 1×H 的向量，这些向量是相加还是并列？**
> **2. 最后需要和 W_out 再进行运算，得到的应该是什么？**

在此展开端到端完整的数据流模拟与图解。

### 3.1 架构高清数据流图解

以下流程图完整展现了数据张量形状的逐层蜕变：

```mermaid
flowchart TD
    subgraph S1["第一阶段：输入与词嵌入 (Embedding Lookup)"]
        Context["12 个上下文词文本<br>左侧 6 词: federal, reserve, chairman, said, the, us<br>右侧 6 词: is, growing, very, steadily, in, recent"]
        IDs["12 个整数 ID 索引<br>idx = [102, 54, 890, 12, 3, 401, 7, 650, 110, 89, 9, 312]"]
        W_in[("输入词嵌入矩阵 W_in<br>形状: (V, H)")]
        Vectors["12 个独立的词向量 (每个形状均为 1 × H)<br>v_{-6}, v_{-5}, ..., v_{+6}"]
        Context --> IDs
        IDs --> |行切片检索 O(1)| Vectors
        W_in -.-> |检索权重| Vectors
    end

    subgraph S2["第二阶段：隐层聚合 (相加/求平均)"]
        Agg["【核心问答】：相加还是并列？<br>必须进行【逐元素相加 / 求平均】！<br>h = (1/12) * sum(v_1, ..., v_12)"]
        H["单一的上下文隐层表征向量 h<br>【形状保持定长: 1 × H】"]
        Vectors --> Agg
        Agg --> H
    end

    subgraph S3["第三阶段：与输出权重 W_out 的交互运算"]
        direction TB
        subgraph NS["路径 B: 高速化负采样 (二分类 - 本项目实现)"]
            PosTarget["正样本中心词: target = market (ID: 508)<br>从 W_out 提取对应行: w_pos (1 × H)"]
            NegWords["K 个随机负样本词: neg_ids = [25, 410, 98, ...]<br>从 W_out 提取对应行: w_neg_k (1 × H)"]
            
            DotPos["点积运算 (内积): score_pos = h · w_pos<br>【得到: 1 个标量数值 (Logit)】"]
            DotNeg["K 次点积运算: score_neg_k = h · w_neg_k<br>【得到: K 个标量数值 (Logits)】"]
            
            SigPos["Sigmoid(score_pos)<br>【得到: 正样本概率 y_pos ≈ 1】"]
            SigNeg["Sigmoid(score_neg_k)<br>【得到: 负样本概率 y_neg_k ≈ 0】"]
            
            LossNS["负采样总损失 Loss:<br>-log(y_pos) - sum(log(1 - y_neg_k))"]
            
            PosTarget --> DotPos --> SigPos --> LossNS
            NegWords --> DotNeg --> SigNeg --> LossNS
        end

        subgraph SM["路径 A: 传统朴素 Softmax (全词表多分类)"]
            W_out_all[("全量输出权重矩阵 W_out<br>形状: (H, V)")]
            Matmul["矩阵乘法: Score = h @ W_out<br>【得到: 1 × V 的全词表得分向量】"]
            Softmax["Softmax 归一化<br>【得到: 1 × V 的全词表概率分布】"]
            LossSM["多分类交叉熵损失 Loss:<br>-log(P(market | context))"]
            
            W_out_all --> Matmul --> Softmax --> LossSM
        end
    end

    H ==> DotPos
    H ==> DotNeg
    H -.-> Matmul
```

---

### 3.2 答疑一：上下文 12 个向量是【相加】还是【并列拼接】？

> **权威结论：在 Word2Vec (CBOW) 中，这些向量必须【逐元素相加】或【求平均】(Sum / Average)，绝对不是并列拼接 (Concatenation)！**

```python
# 代码中的真实操作 (详见 models.py)
h_all = self.embed.forward(contexts)   # 形状: (batch_size, 12, H)
h = np.mean(h_all, axis=1)             # 沿上下文轴求平均，形状变为: (batch_size, H)
```

为什么必须是“求和/平均”而不是“拼接”？背后的三大数学与工程哲学：

1. **词袋模型 (Bag-of-Words) 的本质**：
   - CBOW 的名称就是“连续**词袋**模型 (Continuous **Bag**-of-Words)”。在语言学中，“袋 (Bag)”意味着无序集合。
   - 向量相加在语义几何空间中代表“概念叠加”与“语义交融”。将 12 个词的向量叠加，就如同在调色盘中将 12 种颜色混合，形成当前语境的底色。这种操作对局部微小的语序变动（例如 *“said the US”* vs *“the US said”*）具有天然的容错与泛化鲁棒性。

2. **维度守恒与参数定长（最核心的架构原因）**：
   - 如果采用**相加/求平均**：合成后的隐层向量 $h$ 的形状永远是严格定长的 $(1, H)$。
     无论上下文窗口设置是 2、6 还是 10，后接输出层矩阵的尺寸永远恒定为 $(H, V)$。**模型参数量与窗口大小彻底解耦！**
   - 如果采用**并列拼接 (Concatenation)**：拼接后的向量形状会膨胀为 $(1, 12H)$。
     此时输出层权重矩阵必须膨胀为 $(12H, V)$，参数量**直接暴增 12 倍**！在词表 $V=100,000, H=100$ 时，单层权重参数量将从 $1000$ 万直接飙升至 **$1.2$ 亿**，计算与显存将直接被拖垮。

3. **应对句子边界的自适应能力**：
   - 当遇到句首或句尾时，由于词数不足，窗口无法凑满 12 个词（例如句首只有 3 个右侧词）。
   - 若采用求平均，只需除以实际存在的有效词数 $C_{\text{actual}}$ 即可，无需对缺失位置进行任何复杂的零填充 (Padding)；而若采用拼接，必须强行补零，不仅改变输入空间分布，还容易引入大量噪声。

> **📚 历史演进注记**：
> 早在 2003 年，深度学习鼻祖 Yoshua Bengio 提出的经典前馈神经网络语言模型 (NNLM) 中，为了严格保持序列语序，确实采用了“拼接”方式；然而，Mikolov 等人在 2013 年提出 Word2Vec 时，正是大胆抛弃了计算笨重的“拼接”与“非线性隐藏层激活”，改用**无序相加/求平均**，才换取了数百上千倍的极速训练能力，开创了分布式词向量的新纪元！

---

### 3.3 答疑二：与 W_out 运算后，最终得到的到底是什么？

要回答“得到的是什么”，必须严格区分**传统朴素 Softmax** 与 **现代高速化负采样** 两种完全不同的实现路径：

#### 路径 A：传统朴素 Softmax（全词表多分类模式）
1. **执行的运算**：**全量矩阵乘法 (Matrix Multiplication)**
   $$\text{Score} = h \cdot W_{\text{out}} \quad (\text{维度：} (1, H) \times (H, V) = (1, V))$$
2. **运算后得到的是**：
   - **全词表未归一化得分向量 (Logits，维度 $1 \times V$)**：其中的第 $i$ 个数值，代表字典中第 $i$ 个单词成为中心词的“匹配打分”。
3. **经 Softmax 归一化后得到的是**：
   - **全词表概率分布向量 ($P \in \mathbb{R}^{1 \times V}$)**：每个分量都在 $(0, 1)$ 之间，且全向量累加严格等于 $1.0$。它表示模型认为全字典中各个单词作为中心词的真实概率。
4. **最终目标**：最大化真实中心词的概率，计算全词表交叉熵损失 $L = -\log P(\text{target})$。

---

#### 路径 B：高速化负采样（二分类逻辑回归模式 —— 本项目标准实现）
负采样**根本不与完整的 $W_{\text{out}}$ 做矩阵乘法**！它将问题转化为“二分类是非题”：
1. **对正样本（真实的中心目标词，例如 `market`，监督标签 $t = 1$）**：
   - 从 $W_{\text{out}}$ 中切片提取对应行向量：$w_{\text{pos}} = W_{\text{out}}[\text{target}]$ (形状为 $1 \times H$)。
   - **执行的运算**：**向量点积 (内积 Dot Product)**
     $$\text{score}_{\text{pos}} = h \cdot w_{\text{pos}}^{\top} = \sum_{j=1}^H h_j \cdot w_{\text{pos}, j}$$
   - **得到的是**：**单个标量数值 (Scalar Score)**！
   - 经过 Sigmoid 激活函数后，得到的是一个**标量概率**：
     $$y_{\text{pos}} = \sigma(\text{score}_{\text{pos}}) = \frac{1}{1 + e^{-\text{score}_{\text{pos}}}} \in (0, 1)$$
     它代表模型认为“在当前上下文 $h$ 之下，该词确实是上下文真实中心词”的二分类置信度（期望逼近 1.0）。

2. **对 $K$ 个负样本（随机抽取的噪声假词，例如抽到 `banana`, `airplane`, ...，监督标签 $t = 0$）**：
   - 从 $W_{\text{out}}$ 中提取这 $K$ 个词对应的行向量：$w_{\text{neg}_k} = W_{\text{out}}[\text{neg}_k]$ (每个均为 $1 \times H$)。
   - **执行的运算**：**$K$ 次向量点积 (内积)**
     $$\text{score}_{\text{neg}_k} = h \cdot w_{\text{neg}_k}^{\top}$$
   - **得到的是**：**$K$ 个标量得分**！
   - 经过 Sigmoid 激活后，得到的是 $K$ 个**负样本标量概率** $y_{\text{neg}_k} \in (0, 1)$（期望逼近 0.0）。

3. **最终得到的产物 (二分类交叉熵总损失)**：
   $$L = -\log(y_{\text{pos}}) - \sum_{k=1}^K \log(1 - y_{\text{neg}_k})$$
   **结论**：在负采样中，与 $W_{\text{out}}$ 运算后得到的不是全词表概率，而是针对正词与少数负词的**点积分数**与**二分类损失标量**！运算量直接从 $V$ 次降到了 $K+1$ 次（从 100,000 次锐减至 6 次）！

---

### 3.4 端到端完整数值与张量流转模拟 (Step-by-Step Simulation)

我们以句子：
> *"... federal reserve chairman said the us **[ market ]** is growing very steadily in recent ..."*

进行一次单样本真实数值追踪模拟（设词表 $V=10,000$，向量维度 $H=4$，负采样数 $K=2$）：

| 步骤 | 操作名称 | 输入与数据对象 | 实际数学运算 | 输出结果与张量形状 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | **词元化与索引** | 12 个上下文词文本 | 查词典字典映射 `word_to_id` | 12 个整数 ID：`[102, 54, 890, 12, 3, 401, 7, 650, 110, 89, 9, 312]` |
| **Step 2** | **Embedding 查表** | 12 个整数 ID，权重 $W_{\text{in}} (10000, 4)$ | 索引切片 `v = W_in[idx]` (O(1)) | 12 个向量，每个均为 $(1, 4)$，整体矩阵为 $(12, 4)$ |
| **Step 3** | **隐层求平均聚合** | 12 个 $(1, 4)$ 词向量 | 沿列方向求平均：$h = \frac{1}{12}\sum_{c=1}^{12} v_c$ | **单个上下文综合向量 $h$，形状 $(1, 4)$** |
| **Step 4** | **提取正目标词向量** | 目标词 `market` (ID: 508) | 提取行：$w_{\text{pos}} = W_{\text{out}}[508]$ | 正样本权重向量 $w_{\text{pos}}$，形状 $(1, 4)$ |
| **Step 5** | **计算正样本得分** | $h (1, 4)$ 与 $w_{\text{pos}} (1, 4)$ | 点积：$\text{score}_{\text{pos}} = \sum_{j=1}^4 h_j \cdot w_{\text{pos}, j}$ | **标量得分**，例如 `+2.85` |
| **Step 6** | **正样本 Sigmoid** | 标量得分 `+2.85` | $y_{\text{pos}} = 1 / (1 + e^{-2.85})$ | **标量置信度概率**：`0.945` (高概率接近 1) |
| **Step 7** | **抽取负样本并点积** | 采样词 `[banana(92), sky(1402)]` | 提取对应行并做点积：$\text{score}_{\text{neg}} = h \cdot w_{\text{neg}}^{\top}$ | **2 个标量得分**，例如 `[-1.90, -3.10]` |
| **Step 8** | **负样本 Sigmoid** | 标量得分 `[-1.90, -3.10]` | $y_{\text{neg}} = 1 / (1 + e^{-\text{score}})$ | **2 个标量置信度**：`[0.130, 0.043]` (接近 0) |
| **Step 9** | **汇总二分类损失** | 正置信度 `0.945`，负置信度 `[0.130, 0.043]` | $L = -\log(0.945) - \log(1-0.130) - \log(1-0.043)$ | **总标量损失**：`0.057 + 0.139 + 0.044 = 0.240` |
| **Step 10** | **反向传播梯度流** | 上游损失标量 $L$ | 导数极简：$\frac{\partial L}{\partial s} = y - t$ (正样本 $y-1$, 负样本 $y-0$) | 优雅更新 $W_{\text{out}}$ 与 $W_{\text{in}}$ 中涉及的稀疏行 |

---

## 4. PTB 数据集与文本预处理流水线

Penn Treebank (PTB) 是自然语言处理领域的权威基准语料。
本项目内置完整的数据流水线 (`dataset.py`)：
1. **自动下载与容灾**：启动时自动连接官方源与镜像源下载 `ptb.train.txt`；若在无网络隔离环境中，自动无缝切换至内置预备语料。
2. **文本词元化与词典构建**：自动建立全局 `word_to_id` 与 `id_to_word` 双向映射。
3. **高频词二次下采样 (Subsampling)**：
   根据 Mikolov 经典公式，以一定概率丢弃高频词：
   $$P_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{f(w)}}$$
   缩减语料冗余信息，大幅提升训练速度与词向量语义质量。
4. **滑动窗口上下文提取**：高效生成 `(contexts, target)` 训练集。

---

## 5. 项目整体代码结构

```text
01_word2vec_acceleration/
│
├── README.md               # [本文档] 核心原理、数学推导、全套使用指南与演进剖析
├── dataset.py              # PTB 数据集自动下载、预处理、下采样、词表与窗口提取
├── layers.py               # 核心网络层 (Embedding, EmbeddingDot, SigmoidWithLoss)
├── negative_sampling.py    # 负采样模块 (UnigramSampler 0.75平滑采样, NegativeSamplingLoss)
├── models.py               # 完整模型封装 (CBOWModel, SkipGramModel)
├── optimizer.py            # 参数优化器与梯度裁剪 (Adam, SGD, clip_grads)
├── trainer.py              # 训练生命周期管理 (Mini-batch 洗牌、吞吐率统计、模型持久化)
├── eval.py                 # 词向量质量评估 (余弦相似度、最相似词检索、词类比推理)
├── compare_models.py       # CBOW vs Skip-Gram 全方位横向基准性能对比脚本
├── word_math_cli.py        # [新增] 交互式语义代数计算器 (king - man + woman)
├── visualize.py            # [新增] 词向量 2D 降维可视化工具 (PCA / t-SNE 静态图与交互网页)
├── load_pretrained.py      # [新增] 预训练 GloVe / Word2Vec 词向量格式转换工具
├── generate_pretrained_weights.py # [新增] 通用百科语义空间权重生成器
├── quick_demo.py           # 5 秒体验极速 Demo 脚本
├── train.py                # 完整功能命令行训练脚本
├── assets/                 # 架构图解与可视化资源目录
│   ├── generate_diagram.py            # 数据流高清图绘图脚本
│   └── word_embeddings_interactive.html # 交互式 HTML5 浏览器散点图
└── tests/
    └── test_layers.py      # 核心网络层单元测试与数值梯度检验 (Gradient Check)
```

---

## 6. 核心数学推导一览

### 损失函数 (二分类交叉熵目标)
$$L = - \log \sigma(h \cdot W_{\text{out}}[t]) - \sum_{k=1}^K \log \sigma(- h \cdot W_{\text{out}}[n_k])$$
其中：
- $h$：上下文在隐层的综合表征向量
- $t$：正样本中心词索引
- $n_k$：第 $k$ 个采样的负词索引
- $\sigma(x) = \frac{1}{1 + e^{-x}}$，且利用恒等式 $1 - \sigma(x) = \sigma(-x)$

### 梯度反向传播 (极简优雅的导数形式)
设输入给 Sigmoid 的得分标量为 $x$，监督标签为 $t$ ($t \in \{0, 1\}$)：
$$\frac{\partial L}{\partial x} = \frac{y - t}{N}$$
其中 $y = \sigma(x)$ 为预测概率，$N$ 为 Batch 大小。
- 当样本为正样本 ($t=1$) 时：$\frac{\partial L}{\partial x} = \frac{y - 1}{N}$
- 当样本为负样本 ($t=0$) 时：$\frac{\partial L}{\partial x} = \frac{y - 0}{N} = \frac{y}{N}$

隐藏层向量与输出权重的梯度反传：
$$\frac{\partial L}{\partial h} = \left( \frac{\partial L}{\partial x} \right) \cdot W_{\text{out}}[\text{idx}]$$
$$\frac{\partial L}{\partial W_{\text{out}}[\text{idx}]} = \left( \frac{\partial L}{\partial x} \right) \cdot h$$

---

## 7. 快速开始与使用指南

### 7.1 环境准备
本项目采用纯 Python + NumPy 实现，仅需标准 Python 3.7+ 环境：
```bash
python -c "import numpy; print(numpy.__version__)"
```

### 7.2 运行核心层单元测试 (梯度检验)
在开始训练前，可运行数值梯度检验脚本，验证反向传播解析梯度与中心差分数值梯度的误差（确保误差在 $10^{-7}$ 以下）：
```bash
python tests/test_layers.py
```
**预期输出**：
```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.107s

OK
 [PASS] EmbeddingDot 数值梯度检验通过！相对误差: 1.71e-12
 [PASS] Embedding 重复索引梯度累加测试通过！
 [PASS] NegativeSamplingLoss 前向与反向维度及数值稳定性测试通过！
 [PASS] SigmoidWithLoss 数值梯度检验通过！相对误差: 1.70e-07
```

---

### 7.3 极速上手 Demo (5秒体验完整闭环)
专为即时体验而设，自动加载轻量切片、初始化 CBOW 高速化模型、5轮迭代并在终端展示近义词分析：
```bash
python quick_demo.py
```

---

### 7.4 完整语料库训练 (train.py 命令行参数说明)

使用 `train.py` 可自由配置各项超参数并执行训练：

#### 常用训练命令示例：
```bash
# 1. 使用 CBOW 模型在 PTB 上训练 10 轮 (默认配置)
python train.py --model cbow --epochs 10 --hidden_size 100 --batch_size 128

# 2. 使用 Skip-Gram 模型训练
python train.py --model skipgram --epochs 5 --hidden_size 100

# 3. 自定义窗口大小与负采样数量
python train.py --model cbow --window_size 3 --negative_samples 10 --lr 0.002
```

#### 完整参数列表：
| 参数名 | 默认值 | 作用说明 |
| :--- | :--- | :--- |
| `--model` | `cbow` | 模型类型：`cbow` (训练更快) 或 `skipgram` (低频词学习更好) |
| `--hidden_size` | `100` | 词嵌入向量维度 $H$ |
| `--window_size` | `5` | 单侧上下文窗口大小 (总窗口词数为 $2 \times \text{window}$) |
| `--negative_samples` | `5` | 负采样词抽取个数 $K$ |
| `--power` | `0.75` | 负采样概率分布平滑指数 |
| `--epochs` | `10` | 训练轮数 |
| `--batch_size` | `128` | Mini-batch 批次样本数 |
| `--lr` | `0.001` | 学习率 |
| `--optimizer` | `adam` | 优化器：`adam` 或 `sgd` |
| `--max_grad_norm` | `5.0` | 梯度裁剪阈值 (防止梯度爆炸) |
| `--max_words` | `None` | 最大读取词数 (设为如 50000 可用于快速调试，默认读取全部) |
| `--no_subsampling` | `False` | 若加上此标志则关闭高频词二次下采样 |
| `--save_path` | `None` | 模型保存路径 (默认存入 `weights/` 目录) |
| `--skip_eval` | `False` | 是否跳过训练结束后的语义相似度与类比评测 |

---

### 7.5 CBOW 与 Skip-Gram 对比基准测试 (compare_models.py)

本项目提供了专用的对比基准脚本 `compare_models.py`，可在**完全相同的数据切片与参数条件**下，直接运行两者的横向对比实验：

```bash
python compare_models.py
```

#### 实测 Benchmark 对比结果表 (PTB 数据集实测)：
| 对比维度 | CBOW 模型 | Skip-Gram 模型 | 核心机理差异 |
| :--- | :--- | :--- | :--- |
| **训练吞吐量 (Samples/s)** | **~15,284 样本/秒** | **~10,045 样本/秒** | CBOW 将上下文求平均为单一向量，前向/反向仅需计算一次损失；Skip-Gram 需对 $2W$ 个上下文词分别做 $K+1$ 次点积，运算量高约 50% |
| **总训练耗时 (5 Epochs)** | **2.73 秒 (更快)** | **4.15 秒** | **CBOW 提速约 1.5 倍** |
| **高频词表现** | **极优且平滑** | 良好 | CBOW 的隐层求平均对常见高频词与句法特征有平滑去噪效果 |
| **低频稀有词表现** | 容易被高频词冲淡 | **极佳 (显著胜出)** | Skip-Gram 不对上下文做平均，每个词独立提供监督信号，稀有词向量能学到更纯粹的语义 |
| **类比推理 (Analogy)** | 语法类比较好 | **语义类比更强** | Mikolov 原论文指出，Skip-Gram 在大规模词类比（如国家-首都、性别变换）上显著优于 CBOW |

---

### 7.6 词向量语义评测与词类比推理

在训练完成后，系统会自动调用 `eval.py` 中的评测函数：

1. **最相近词检索 (余弦相似度)**：
   ```python
   from eval import most_similar
   most_similar("bank", word_to_id, id_to_word, model.word_vecs, top=5)
   ```
2. **经典的向量代数类比推理**：
   $$\vec{d} \approx \vec{b} - \vec{a} + \vec{c}$$
   ```python
   from eval import analogy
   # 类比: he 之于 his 犹如 she 之于 ?
   analogy("he", "his", "she", word_to_id, id_to_word, model.word_vecs, top=5)
   ```

---

### 7.7 交互式语义代数计算器 (word_math_cli.py)

Word2Vec 最具魅力的特性是高维稠密空间中的“线性平移代数特性”。本项目提供了一个功能完备的交互式终端计算器 `word_math_cli.py`：

#### 使用方法：
```bash
# 1. 启动交互式终端 (默认优先加载通用百科预训练模型)
python word_math_cli.py

# 2. 或在命令行直接求解算式
python word_math_cli.py "king - man + woman"
python word_math_cli.py "paris - france + japan"
```

#### 交互式终端运行效果展示：
```text
VectorCalc [通用百科] >>> king - man + woman
----------------------------------------------------------------------
  [*] 算式求解: king - man + woman
----------------------------------------------------------------------
  Rank  1 | 预测词: queen            | 相似度: +0.9328 [####################--]
  Rank  2 | 预测词: empress          | 相似度: +0.9245 [####################--]
  Rank  3 | 预测词: princess         | 相似度: +0.9244 [####################--]
----------------------------------------------------------------------
```
- 支持 `+`（概念融合）、`-`（特征剥离）以及单词同义词检索；
- 支持在交互终端输入 `switch` 自由在【通用百科模型】与【PTB财经模型】之间无缝热切换！

---

### 7.8 词向量空间 2D 降维可视化 (visualize.py)

为了直观验证神经网络是否真的学习到了语言的语义分布规律，本项目提供了双模可视化工具 `visualize.py`：

```bash
# 1. 使用 PCA 算法生成语义聚类散点图
python visualize.py --method pca --mode cluster

# 2. 使用 t-SNE 算法生成高频词全景分布
python visualize.py --method tsne --mode top --top_n 80
```

#### 生成成果包括：
1. **出版级高清静态图 (`assets/word_embeddings_2d.png`)**：
   - 自动按照**金融商业**、**机构实体**、**人物职位**、**时间周期**、**动作变化** 5 大语义特征分簇着色；
   - 包含药丸气泡标注与分类图例，零 Matplotlib 依赖（纯 Pillow 渲染，100% 跨平台免踩坑）。
2. **零依赖交互式 HTML5 网页 (`assets/word_embeddings_interactive.html`)**：
   - 可以在任意浏览器中双击直接打开；
   - 鼠标悬停可高亮当前词汇、坐标数值与语义分类，支持平移缩放。

---

## 8. 常见问题与避坑指南 (FAQ)

### Q1: 为什么同一个词索引多次出现时，直接赋值 `dW[idx] = dout` 是致命错误？
在一个 Batch 中，如果上下文出现了两次同一个词（比如 `idx = [2, 0, 2]`），直接切片赋值 `dW[idx] = dout` 会执行 Python 的连续赋值，后一个索引位置会**直接覆写并抹杀**前一个位置的梯度更新！正确的反向传播必须是数学上的全微分累加 $\frac{\partial L}{\partial W} = \sum \frac{\partial L}{\partial \text{out}} \frac{\partial \text{out}}{\partial W}$，NumPy 中必须使用 `np.add.at(dW, idx, dout)`。

### Q2: 为什么最终取 $W_{\text{in}}$ 作为词向量，而不是 $W_{\text{out}}$？
- $W_{\text{in}}$ 是中心词（或上下文词）输入映射矩阵，每一行代表一个词的分布式向量。
- $W_{\text{out}}$ 是输出二分类的表征权重。
- 在主流学术界与实践中，通常直接采用 $W_{\text{in}}$ 作为最终的词向量；部分研究者也会将 $W_{\text{in}} + W_{\text{out}}$ 或拼接作为词向量。本项目默认导出 $W_{\text{in}}$。

### Q3: 为什么 Adam 优化器在 Word2Vec 训练中大幅领先普通 SGD？
在稀疏文本数据中，不同单词的出现频次呈现出极端长尾分布（Zipf 定律）。高频词更新频繁，低频词更新极少。SGD 对所有参数采用统一的学习率，容易导致高频词震荡、低频词迟迟不收敛。Adam 维护了每个参数的一阶动量与二阶方差，能为低频参数自适应提供更大的有效步长，从而大幅加快收敛速度。

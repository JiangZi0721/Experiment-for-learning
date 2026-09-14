# LLM Foundations & Core Attention Kernels Lab (大语言模型架构基石与核心算子实验室)

> 这是一个专为系统化掌握大语言模型 (LLM) 底层计算架构、关键加速算子与显存优化技术而构建的**白盒可透视实战工程**。
> 涵盖 **RoPE 旋转位置编码相对距离不变性证明、KV Cache 压缩演进（MHA / GQA / MQA / DeepSeek MLA）显存动力学建模、FlashAttention 在线分块 Softmax (Online Softmax Tiling) 精确等价性推导、以及 RMSNorm / LoRA 手撕算子实战**。

---

## 📖 目录 (Table of Contents)

1. [👶 初学者零门槛极速通关：5分钟读懂大模型核心算子](#-初学者零门槛极速通关5分钟读懂大模型核心算子)
   - [1.1 为什么绝对位置编码不行？RoPE 的复数旋转魔法](#11-为什么绝对位置编码不行rope-的复数旋转魔法)
   - [1.2 KV Cache 显存危机：从 MHA 到 DeepSeek MLA 的演进史](#12-kv-cache-显存危机从-mha-到-deepseek-mla-的演进史)
   - [1.3 FlashAttention 的本质：为什么不用存 N×N 注意力矩阵？](#13-flashattention-的本质为什么不用存-nn-注意力矩阵)
2. [📐 核心算子微积分与底层推导](#-核心算子微积分与底层推导)
   - [2.1 RoPE 2D 分块复数内积与相对位置恒等式证明](#21-rope-2d-分块复数内积与相对位置恒等式证明)
   - [2.2 KV Cache 显存容量理论推导 (Bytes / Token)](#22-kv-cache-显存容量理论推导-bytes--token)
   - [2.3 FlashAttention Online Softmax 动态平移数学公式](#23-flashattention-online-softmax-动态平移数学公式)
3. [📁 工程目录组织规范](#-工程目录组织规范)
4. [🧪 三大基准验证实验](#-三大基准验证实验)
   - [实验一：RoPE 旋转位置编码相对距离不变性严格检验](#实验一rope-旋转位置编码相对距离不变性严格检验)
   - [实验二：KV Cache 演进全景建模与显存压缩大比拼](#实验二kv-cache-演进全景建模与显存压缩大比拼)
   - [实验三：FlashAttention Online Softmax 切块计算数学等价性实证](#实验三flashattention-online-softmax-切块计算数学等价性实证)
5. [🚀 快速开始与复现指南](#-快速开始与复现指南)

---

## 👶 初学者零门槛极速通关：5分钟读懂大模型核心算子

### 1.1 为什么绝对位置编码不行？RoPE 的复数旋转魔法
在早期 Transformer（如 Sinusoidal PE）中，位置信息作为加性偏置直接加在词向量上（$x + PE$）。
这种做法破坏了词向量本身的语义范数，且在长度外推时很容易失效。

**RoPE (Rotary Position Embedding)** 的优雅之处在于：
- 它不使用“加法”，而是将向量切分成 2D 子空间，在复平面上对向量施加旋转：$\mathbf{R}_{\Theta, m} \mathbf{q}$；
- 旋转矩阵满足正交性：$\mathbf{R}_m^T \mathbf{R}_n = \mathbf{R}_{n-m}$；
- **惊人推论**：Query 与 Key 计算点积时，绝对位置 $m$ 与 $n$ 自动抵消，结果**严格只依赖于相对距离 $(m - n)$**！
- 即使把整个句子平移 100 个位置（$m+100, n+100$），注意力打分严格分毫不差！

---

### 1.2 KV Cache 显存危机：从 MHA 到 DeepSeek MLA 的演进史

在自回归生成阶段（Decode 阶段），每次生成一个新字，都要读取上文所有词的 Key 与 Value 向量。
在 128K 超长上下文中，**KV Cache 显存会直接把多张 A100/H100 爆掉**！

1. **MHA (Multi-Head Attention)**：Q/K/V 具有相同头数（如 64 头），128K 单并发需占用 **256 GB** 显存！
2. **GQA (Grouped-Query Attention, LLaMA-3 标配)**：每 8 个 Q 共享 1 组 KV，显存直降 8 倍至 **32 GB**；
3. **MQA (Multi-Query Attention)**：全网仅留 1 组 KV，显存压至 **4 GB**，但多头表征能力有明显损失；
4. **DeepSeek MLA (Multi-Head Latent Attention, DeepSeek-V2/V3 核心王牌)**：
   - 彻底颠覆“单纯减少头数”的思路，将 KV 投影压缩为低秩潜在向量 $c_t^{KV}$（512维）；
   - 在线解码时只需缓存极小的隐向量，解耦 RoPE 独立计算；
   - **显存压低 96.5%（仅 9 GB），但完整保留了全量多头注意力的极致表征能力**！

---

### 1.3 FlashAttention 的本质：为什么不用存 N×N 注意力矩阵？

传统 Attention 算子最大的吞吐瓶颈不是 GPU 浮点算力，而是 **HBM（显存高带宽内存）的读写带宽（Memory-Bound）**！
计算 $S = Q K^T$（产生 $N \times N$ 矩阵写入显存），再读出来做 $\text{Softmax}$，再写回显存，再与 $V$ 相乘……
在长文本下，这个 $N \times N$ 矩阵不仅占满几十 GB 显存，读写往返更浪费了 80% 的时间。

**FlashAttention 的核心突破（Online Softmax）**：
- 采用 GPU SRAM 片上高速缓存分块（如 32×32 切块）；
- 在局部切块内动态维护最大值 $m$ 与归一化分母 $l$；
- 块与块交替时，通过缩放因子 $\alpha = \exp(m_{prev} - m_{new})$ 动态修正上一个块的累加和；
- **自始至终无需在 HBM 显存中保存 $N \times N$ 矩阵，显存占用压至 $O(1)$，速度提升数倍**！

---

## 📁 工程目录组织规范

```text
10_LLM_Foundations_and_Kernels/
├── README.md                           # 本全景教学与实验报告
├── requirements.txt                    # 依赖清单 (torch, numpy, matplotlib, rich)
├── main.py                             # 一站式全景 CLI 交互主控面板
│
├── src/
│   ├── kernels.py                      # 纯白盒核心算子 (RMSNorm, RoPE, LoRA, FlashAttention Online Softmax)
│   └── kv_cache_benchmark.py           # KV Cache 显存动力学建模与演进分析
│
├── experiments/
│   ├── exp1_rope_rotation.py           # 实验一：RoPE 相对位置不变性与衰减实证
│   ├── exp2_kv_cache_analysis.py       # 实验二：KV Cache 四大架构显存极限对比
│   └── exp3_flash_attention_correctness.py # 实验三：FlashAttention 在线分块等价性证明
│
├── docs/                               # 深度大模型理论专著与算法面试手撕题库
│   ├── 01_Architecture_and_Foundations/
│   ├── 02_Training_and_FineTuning/
│   ├── 03_Inference_and_Serving/
│   └── 04_Interviews_and_HandsOn/
│
└── images/                             # 实验自动生成的高清实证图表
    ├── llm_rope_relative_invariance.png # RoPE 相对距离衰减效应曲线
    └── llm_kv_cache_compression.png    # MHA vs GQA vs MQA vs MLA 显存对比
```

---

## 🚀 快速开始与复现指南

### 1. 一键运行全景实验
```bash
# 一键运行全部实验
python main.py --exp all

# 检验 RoPE 相对平移不变性
python main.py --exp 1

# 评估 KV Cache 架构显存消耗
python main.py --exp 2

# 验证 FlashAttention 在线分块等价性
python main.py --exp 3
```

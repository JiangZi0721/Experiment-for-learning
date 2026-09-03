# -*- coding: utf-8 -*-
"""
White-Box RAG Lab 完整语料库批量生成器
覆盖 4 大领域，共 100 篇高质量 Markdown 技术拆解文档。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")

# 引入已有的两组或定义完整的生成字典
# 3. 03_llm_infra (25 篇)
LLM_INFRA = [
    ("transformer_mha_foundations", "Transformer 标准多头注意力 (MHA) 架构推导", """# Transformer 标准多头注意力 (MHA) 架构推导

## 1. 缩放点积注意力 (Scaled Dot-Product Attention)
注意力核心公式：
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
- **除以 $\\sqrt{d_k}$ 的数学必要性**：当维度 $d_k$ 很大时，点积结果的方差为 $d_k$。如果不缩放，点积绝对值过大将导致 Softmax 函数进入极小梯度的饱和区（Saturation），造成反向传播梯度弥散。

## 2. Multi-Head Attention (MHA) 机制
- 将输入投影为 $h$ 组不同的 $Q_i, K_i, V_i$ 矩阵。
- 每个注意力头各自拥有独立的投影权重 $W_i^Q, W_i^K, W_i^V$。
- 在不同的表示子空间（Representation Subspaces）中并行捕捉长距离上下文依赖，最后拼接输出并乘以 $W^O$。"""),

    ("multi_query_attention_mqa", "多查询注意力 (MQA): 极致压缩 Key-Value 头的工程革新", """# 多查询注意力 (MQA): 极致压缩 Key-Value 头的工程革新

## 1. MHA 在大模型自回归解码中的瓶颈
在自回归推理（Inference / Decode）阶段，模型每次仅生成一个 Token。由于注意力操作需要读取历史全量 KV Cache，内存带宽（Memory Bandwidth Bound）成为主要吞吐瓶颈。

## 2. Multi-Query Attention (MQA) 设计
- **结构差异**：保持 Query 头数量（如 32 个头）不变，但将所有头共享**唯一的一个 Key 头和一个 Value 头**。
- **显存压制**：KV Cache 的显存占用直接骤降为原来的 $1/h$（例如减少 90% 以上）。
- **取舍代价**：大幅压缩了模型对复杂上下文的多重泛化与注意力表征能力，模型容量和评测分数有一定微弱损耗。"""),

    ("grouped_query_attention_gqa", "分组查询注意力 (GQA): 显存带宽与模型容量的黄金折中", """# 分组查询注意力 (GQA): 显存带宽与模型容量的黄金折中

## 1. GQA 的分组折中思想
Grouped-Query Attention (GQA) 是 Llama-2-70B、Llama-3 等当代主流开源大模型的标准标配：
- 将 $H_Q$ 个 Query 头均匀划分为 $G$ 个组（Group）。
- 每个组内的 Query 头共享同一个 Key 头和 Value 头（即共有 $G$ 对 KV 头）。
- 当 $G=H_Q$ 时，GQA 退化为标准 MHA；当 $G=1$ 时，GQA 退化为 MQA。

## 2. 性能收益实测
- 在维持接近 MHA 的高表达能力的同时，推理时 KV Cache 的内存占用和带宽开销降低数倍，服务并发 Batch Size 提升 2~3 倍，是工业界公认的最佳平衡点。"""),

    ("flash_attention_v1_v2", "FlashAttention 核心原理：Tiling 切块与重计算显存革新", """# FlashAttention 核心原理：Tiling 切块与重计算显存革新

## 1. 传统注意力的 GPU 显存墙问题
标准 Attention 需要在 GPU 高带宽显存（HBM）中完整物化 $N \times N$ 的注意力中间矩阵 $S = QK^T$ 和 $P = \\text{softmax}(S)$。长上下文时 $O(N^2)$ 的读写 I/O 导致严重的访存开销。

## 2. FlashAttention 关键技术
- **Tiling 分块计算**：利用 GPU 片上极速 SRAM（SRAM 速度比 HBM 快一个数量级），将 $Q, K, V$ 划分为小块依次加载进 SRAM。
- **在线 Softmax 归一化 (Online Softmax)**：在不保存全局 Softmax 矩阵的情况下，通过动态缩放因子逐步合并部分 Softmax 的分子分母。
- **反向传播重计算 (Recomputation)**：在前向传播中完全不保存 $N \times N$ 的中间激活值，反向传播时直接利用 SRAM 快速重新计算，将空间复杂度从 $O(N^2)$ 降低到 $O(N)$。"""),

    ("kv_cache_memory_calculation", "大模型推理 KV Cache 显存占用物理公式与推演", """# 大模型推理 KV Cache 显存占用物理公式与推演

## 1. 显存占用公式
对于一个 Transformer 模型，每个 Token 在单层中需要缓存一个 Key 向量和一个 Value 向量。全局显存大小计算公式为：
$$\\text{Memory}_{KV} = 2 \\times 2 \\times n_{layers} \\times d_{model} \\times \\text{seq\\_len} \\times \\text{batch\\_size} \\times \\text{bytes\\_per\\_elem}$$
- 第一个 $2$ 代表 Key 和 Value 两个张量。
- 第二个 $2$ 代表 FP16/BF16 精度占用 2 字节。
- $n_{layers}$ 为层数，$d_{model}$ 为隐藏层维度。

## 2. 实例测算
以 70B 模型（层数 80，隐藏层 8192，采用 FP16）为例：
- 单个 Token 的 KV 缓存占用约为 $2 \\times 2 \\times 80 \\times 8192 = 2.62\\text{ MB}$。
- 若并发并发数为 16，上下文长度达到 8K，仅 KV Cache 就需消耗近 $335\\text{ GB}$ 显存，远超参数自身显存。"""),

    ("paged_attention_vllm_paging", "PagedAttention 与 vLLM 显存虚拟分页管理", """# PagedAttention 与 vLLM 显存虚拟分页管理

## 1. 传统 KV Cache 的显存碎片痛点
由于自回归生成长度事先未知，传统系统必须为每个请求预先分配一段**物理连续**的最大长度显存空间。导致严重的内部碎片（Internal Fragmentation，预分配未使用）和外部碎片，显存有效利用率不足 40%。

## 2. PagedAttention 操作系统级分页思想
- 借鉴虚拟内存页表（Page Table）机制，将连续的虚拟 Token 映射到不连续的物理内存块（Physical Block，通常每个块存 16 个 Token）。
- **动态按需分配**：每生成 16 个 Token 才申请一个新的物理块。
- **写时复制 (Copy-On-Write)**：在并行采样（Parallel Sampling）和束搜索（Beam Search）中，多个序列共享同一段 Prompt 的物理 KV 块，分支时才按需复制，显存利用率提升至 96% 以上。"""),

    ("zero_1_2_3_deepspeed_partitioning", "DeepSpeed ZeRO 显存优化：从 ZeRO-1 到 ZeRO-3 全景拆解", """# DeepSpeed ZeRO 显存优化：从 ZeRO-1 到 ZeRO-3 全景拆解

## 1. 模型训练显存构成
显存主要被模型状态（Model States）占据：参数（Parameters $P$）、梯度（Gradients $G$）、优化器状态（Optimizer States $O$，Adam 中包含一阶动量和二阶动量，占用 $12\\times P$ 显存）。

## 2. ZeRO 三阶段切片
- **ZeRO-1 (Optimizer State Partitioning, $P_{os}$)**：将 Adam 优化器状态均匀切分到 $N$ 个 GPU 上。显存减少 4 倍，通信开销为 0。
- **ZeRO-2 (Gradient Partitioning, $P_{os+g}$)**：进一步将梯度也在 GPU 间切分。每个 GPU 仅保留自身负责参数的梯度。显存减少 8 倍，通信开销无额外增加。
- **ZeRO-3 (Parameter Partitioning, $P_{os+g+p}$)**：**将模型参数本身也彻底打碎分区**。在前向传播计算某一层时，通过 `All-Gather` 动态拉取参数，算完立即释放内存；反向传播时再次拉取，算完再释放。允许单张消费级显卡训练百亿参数模型。"""),

    ("tensor_parallelism_megatron", "张量并行 (Tensor Parallelism) Megatron-LM 矩阵切分", """# 张量并行 (Tensor Parallelism) Megatron-LM 矩阵切分

## 1. 为什么需要张量并行
当单层模型的参数量超过单张 GPU 显存上限时，必须将层内的权重矩阵在多个 GPU 间切分并发执行。

## 2. MLP 层的切分规范
- **第一层 $W_1$ (Column Parallel)**：将矩阵按列切分。输入 $X$ 广播到所有 GPU，并发执行 $Y_i = \\text{GeLU}(X W_{1,i})$。
- **第二层 $W_2$ (Row Parallel)**：将矩阵按行切分。每张卡计算 $Z_i = Y_i W_{2,i}$。
- **跨卡聚合**：最后仅需执行一次 `All-Reduce (Sum)` 操作即可得到最终的 $Z = \\sum Z_i$，通信与计算完美流水化。"""),

    ("pipeline_parallelism_gpipe_1f1b", "流水线并行 (Pipeline Parallelism): GPipe 气泡与 1F1B 调度", """# 流水线并行 (Pipeline Parallelism): GPipe 气泡与 1F1B 调度

## 1. 流水线气泡 (Bubble) 痛点
将深度为 $L$ 的网络按层划分为多个阶段（Stages）分配给不同的 GPU。初级朴素流水线中，后级 GPU 必须等待前级输出，导致大量计算单元空闲等待（Bubble）。

## 2. GPipe 与微批次 (Micro-batch)
GPipe 将全局 Batch 切分为 $M$ 个 Micro-batch，使各个阶段能够交叠计算，将气泡比例压缩至 $\\frac{K-1}{M+K-1}$。

## 3. 1F1B 稳态调度 (One Forward, One Backward)
在稳态运行期，每张 GPU 交替执行一次前向计算（Forward）和一次反向计算（Backward）。这使得反向传播尽早释放前向激活值内存，峰值显存占用大幅低于 GPipe。"""),

    ("quantization_awq_vs_gptq", "大模型量化前沿：AWQ 激活感知量化 vs GPTQ 二阶误差优化", """# 大模型量化前沿：AWQ 激活感知量化 vs GPTQ 二阶误差优化

## 1. 权重离散化的挑战
将 FP16 权重截断为 INT4 时，极易导致大模型在特定任务上困惑度（Perplexity）剧增甚至胡言乱语。

## 2. GPTQ (基于二阶海森矩阵优化)
- 借鉴 Optimal Brain Surgeon (OBS) 思想，利用逆海森矩阵（Hessian）计算量化某一行权重对全局输出造成的误差，并动态补偿调整尚未量化的相邻权重。

## 3. AWQ (Activation-aware Weight Quantization)
- 核心洞察：**权重并不等权重要，受大激活值（Activation Outliers）影响的仅占 1% 的显著权重才是维持模型精度的命脉**。
- 策略：保护这 1% 的显著权重不被过度量化，或者在量化前根据激活值分布对权重矩阵执行全局平滑缩放，推理开销极低。"""),

    ("rope_rotary_position_embedding", "旋转位置编码 (RoPE) 数学原理与外推性", """# 旋转位置编码 (RoPE) 数学原理与外推性

## 1. 绝对位置编码与相对位置编码的融合
传统 Absolute Embedding 简单将位置向量与 Token 嵌入相加，缺乏相对距离的几何不变性。

## 2. 旋转矩阵的复数推导
RoPE 将二维向量视为复数，通过正交旋转矩阵对 Query 和 Key 进行旋转：
$$R_{\\Theta, m}^d = \\text{diag}\\left(R_{\\theta_1, m}, R_{\\theta_2, m}, ..., R_{\\theta_{d/2}, m}\\right)$$
其中内积满足：
$$\\langle R_m q, R_n k \\rangle = g(q, k, m-n)$$
两个 Token 的注意力得分只取决于它们在序列中的**相对距离 $m-n$**，天然具备极强的长上下文外推（Length Extrapolation）能力。"""),

    ("moe_sparse_mixture_of_experts", "混合专家模型 (MoE): 门控路由 (Gating) 与负载均衡", """# 混合专家模型 (MoE): 门控路由 (Gating) 与负载均衡

## 1. 稀疏激活思想 (Conditional Computation)
保持参数总量暴增（如从 7B 扩展到 64B），但对于每个输入的 Token，通过门控网络（Router / Gate）仅动态激活其中的 Top-K 个专家前馈网络（FFN Experts），使实际浮点计算量（FLOPs）维持在极低水平。

## 2. 专家负载均衡辅助损失 (Auxiliary Load Balancing Loss)
- **路由崩溃风险**：门控网络倾向于反复选择少数几个表现较好的专家，导致其他专家参数未充分训练，且被选中的专家产生严重硬件吞吐瓶颈。
- **解决机制**：引入辅助损失函数，惩罚专家选择概率方差，强制 Token 均匀分流到不同专家。"""),

    ("continuous_batching_orca", "大模型连续批处理 (Continuous Batching / Dynamic Batching)", """# 大模型连续批处理 (Continuous Batching / Dynamic Batching)

## 1. 静态批处理 (Static Batching) 的短板
传统推理由客户端批量提交 $N$ 个请求。系统必须等待**最长的一个请求完全生成完毕**才能释放该批次，导致短序列请求白白等待（GPU 利用率出现严重尾部空洞）。

## 2. 迭代级调度 (Iteration-level Scheduling)
- Orca 与 vLLM 引入了连续批处理：调度器在每一个生成步骤（Step）结束后介入。
- 一旦某个请求生成结束符 `<|endoftext|>`，系统立即将其弹出并释放显存；同时将队列中新到达的请求动态插入当前 Step 的空位继续并发计算，系统吞吐量提升 2~4 倍。"""),

    ("speculative_decoding_acceleration", "推测解码 (Speculative Decoding): 小模型草稿与大模型并行验证", """# 推测解码 (Speculative Decoding): 小模型草稿与大模型并行验证

## 1. 访存瓶颈与算力富余
大模型自回归生成单个 Token 时，GPU 的算力核心（Tensor Cores）大部分时间在空闲等待从显存中加载权重。

## 2. 推测执行两步走
1. **小模型草稿 (Drafting)**：使用一个极快的小草稿模型（如 1B 模型）廉价地先行生成 $K$ 个后续 Token。
2. **大模型并行验证 (Verification)**：将这 $K$ 个 Token 作为整段序列一次性送入大模型，大模型利用其并行计算能力在一次前向传播中同时评估这 $K$ 个 Token 的条件概率，通过拒绝采样（Rejection Sampling）决定接受前 $M$ 个。
3. 严格保证最终输出概率分布与单独使用大模型完全等价，整体生成延迟降低 2~3 倍。"""),

    ("lora_qlora_peft_mechanics", "参数高效微调：LoRA 低秩适应与 QLoRA 双重量化原理", """# 参数高效微调：LoRA 低秩适应与 QLoRA 双重量化原理

## 1. LoRA (Low-Rank Adaptation)
- 冻结预训练大模型原本的权重矩阵 $W_0 \\in \\mathbb{R}^{d \\times k}$。
- 引入低秩分解矩阵旁路：$\\Delta W = B \\times A$，其中 $A \\in \\mathbb{R}^{r \\times k}$ 采用高斯初始化，$B \\in \\mathbb{R}^{d \\times r}$ 初始化为 0，且秩 $r \\ll \\min(d, k)$（通常取 8 或 16）。
- 仅训练 $A$ 和 $B$，可训练参数量骤降至原模型的 0.1% 以下。

## 2. QLoRA (高效 4-bit 量化微调)
- **NF4 (NormalFloat 4)**：针对正态分布权重优化的理论最优量化数据类型。
- **双重量化 (Double Quantization)**：对量化本身的常数因子进行二次量化，每个参数节省 0.37 bit。
- **分页优化器 (Paged Optimizers)**：利用 CUDA 统一内存解决微调显存峰值 OOM 崩溃。"""),

    ("fp8_mixed_precision_gemm", "FP8 混合精度训练与 GEMM 矩阵乘法硬件加速", """# FP8 混合精度训练与 GEMM 矩阵乘法硬件加速

## 1. 两种 FP8 格式规范 (E4M3 vs E5M2)
- **E4M3 (1位符号 + 4位指数 + 3位尾数)**：拥有更高精度（尾数多），主要用于前向传播中的激活值与权重计算。
- **E5M2 (1位符号 + 5位指数 + 2位尾数)**：拥有与 FP16 相同的动态范围（指数多），主要用于反向传播中梯度梯度的传递，防止下溢（Underflow）。

## 2. 硬件算力跃迁
NVIDIA Hopper 架构（H100/H800）集成了第四代 Tensor Core，原生支持 FP8 GEMM，理论算力吞吐量是 FP16 的整整 2 倍，且能直接减少一半的通信与显存带宽压力。"""),

    ("sequence_parallelism_ulysses", "长序列并行：DeepSpeed Ulysses 与 Ring-Attention", """# 长序列并行：DeepSpeed Ulysses 与 Ring-Attention

## 1. 显存瓶颈转移至序列长度
当上下文长度扩展到 128K 或 1M 时，即使使用张量并行，单层注意力的中间激活值也会彻底打爆显存。

## 2. DeepSpeed Ulysses
- 在注意力头维度（Heads）和序列维度（Sequence）之间执行优雅的 `All-to-All` 全局转置通信。
- 序列被平均切分到各卡；在计算 Self-Attention 之前，将序列拼全并将头切开，使得单机内的计算单元依然可以复用标准 FlashAttention。

## 3. Ring-Attention
将注意力分块在环形网络中流动传递，允许上下文长度随着集群 GPU 数量的线性增加而实现无上限水平扩展。"""),

    ("chunked_prefill_prompt_caching", "分块预填充 (Chunked Prefill) 与 Prompt Caching 系统", """# 分块预填充 (Chunked Prefill) 与 Prompt Caching 系统

## 1. Prefill 阶段与 Decode 阶段的算力冲突
- **Prefill (首字生成)**：Compute-bound，计算密度极高，耗时长，会瞬间卡死正在执行的轻量 Decode 请求，造成 TTFT（首字延迟）急剧劣化。
- **Decode (增量流式)**：Memory-bandwidth-bound。

## 2. Chunked Prefill 分块技术
- 将一个几千 Token 的超长 Prompt 切分成多个固定大小的 Chunk（如 512）。
- 将一个 Prefill Chunk 与若干处于 Decode 阶段的请求打包在同一个 Batch 内并发执行，彻底熨平算力波动。

## 3. Prompt Caching (上下文提示词缓存)
对于多轮对话中重复出现的高频 System Prompt 或前序知识库，显存中保留已计算好的 KV 块哈希索引，新请求直接复用缓存，跳过 Prefill 计算。"""),

    ("activation_checkpointing_gradient", "激活重算 (Activation Checkpointing / Gradient Checkpointing)", """# 激活重算 (Activation Checkpointing / Gradient Checkpointing)

## 1. 训练显存的罪魁祸首：中间激活值
反向传播计算梯度需要用到前向传播的所有中间激活值（Activations）。层数越深，激活值占用的显存呈线性爆炸。

## 2. 重算机制的以时间换空间
- 前向传播时，系统只保留部分“检查点”（Checkpoints，例如每个 Transformer Block 的输入），丢弃内部细粒度的注意力中间激活值。
- 反向传播到达该 Block 时，立即重新执行一次局部的微型前向传播算回中间值，随后计算梯度并释放。
- 仅仅增加了约 20%~30% 的计算时间，却能降低 70% 的激活值显存占用。"""),

    ("cuda_graph_launch_overhead", "CUDA Graph 静态图加速：消除小算子内核启动开销", """# CUDA Graph 静态图加速：消除小算子内核启动开销

## 1. CPU 提交瓶颈 (Kernel Launch Overhead)
大模型 Decode 阶段每生成一个 Token 需要串行调用数百个细粒度算子（LayerNorm、Add、RMSNorm 等）。每次 GPU 内核启动需要约 3~5 微秒的 CPU 调度开销。当批量较小时，CPU 提交速度甚至赶不上 GPU 执行速度。

## 2. CUDA Graph 预捕获机制
- 将这数百个内核调用及其内存依赖拓扑在预热阶段一次性捕获为一张静态的有向无环图（Graph）。
- 运行时，CPU 只需向 GPU 提交一个单一的工作单元执行该 Graph，内核调度开销归零，显著缩短单 Token 生成的端到端延迟。"""),

    ("kernel_fusion_triton_cuda", "算子融合 (Kernel Fusion) 与 OpenAI Triton 编译优化", """# 算子融合 (Kernel Fusion) 与 OpenAI Triton 编译优化

## 1. 访存密集型算子的浪费
在连续执行 `x = LayerNorm(x + Residual)` 时，传统框架会先启动一个加法内核写入全局显存，再启动一个归一化内核从全局显存读入。两次往返显存开销极大。

## 2. 算子融合 (Fusion)
- 将相加、归一化、激活函数全部揉合到同一个 GPU 线程块中执行，中间结果直接存放在片上寄存器中，彻底省去回写 HBM 的带宽开销。
- **Triton 的价值**：让算法工程师不必手写繁杂晦涩的 CUDA C/C++ 共享内存同步与指针偏移代码，直接用高级 Pythonic 语法编译出媲美手写 CUDA 的极限性能算子。"""),

    ("attention_masks_causal_prefix", "注意力掩码体系：因果因果掩码 (Causal Mask) 与前缀掩码 (Prefix Mask)", """# 注意力掩码体系：因果因果掩码 (Causal Mask) 与前缀掩码 (Prefix Mask)

## 1. 因果掩码 (Causal Mask / Lower Triangular)
在自回归语言模型中，为了防止模型在预测第 $t$ 个 Token 时“穿越”偷看未来的信息，将注意力矩阵右上三角的权重强行填充为 $-\\infty$，经 Softmax 后概率为 0。

## 2. 前缀掩码 (Prefix Masking)
在 PrefixLM（如 ChatGLM、Encoder-Decoder 混合架构）中：
- 前序 Prompt 部分的 Token 之间允许相互双向可见（提高对已知上下文的理解深度）。
- 生成的回答部分严格遵守单向因果掩码，实现生成式任务与理解式任务的最佳融合。"""),

    ("alibi_attention_linear_biases", "ALiBi 线性偏置注意力：无需位置编码的长文本泛化", """# ALiBi 线性偏置注意力：无需位置编码的长文本泛化

## 1. 抛弃位置嵌入向量
ALiBi (Attention with Linear Biases) 彻底移除了加在输入层的位置编码向量。

## 2. 在 Softmax 处直接施加线性惩罚
注意力分数计算直接注入关于键值相对距离的固定斜率惩罚项：
$$\\text{Softmax}\\left(q_i k_j^T - m \\cdot |i - j|\\right)$$
- 其中 $m$ 是为每个注意力头预先设定的几何递减常数斜率。
- 离得越远的 Token，受到的负偏置越大。
- **卓越的外推性**：在 2048 长度上训练的模型，推理时能平滑外推到 8000 长度而无需微调。"""),

    ("kv_cache_compression_h2o", "动态 KV Cache 剪枝策略：H2O (Heavy Hitter Oracle)", """# 动态 KV Cache 剪枝策略：H2O (Heavy Hitter Oracle)

## 1. 幂律注意力分布
在大模型长上下文注意力中，绝大多数 Token 分配到的注意力权重极低，只有少数关键的“重打击者”（Heavy Hitters, $H_2$）和最近的几个局部 Token 贡献了 95% 以上的注意力质量。

## 2. H2O 动态剔除机制
- 维护固定预算的 KV Cache 容量。
- 动态统计累积注意力分数。
- 每次新 Token 加入导致超出容量时，果断淘汰历史上注意力累计得分最低的非关键 Token，将超长上下文的显存占用恒定锁定在预设常数级别。"""),

    ("direct_preference_optimization_dpo", "对齐技术演进：PPO 强化学习 vs DPO 直接偏好优化", """# 对齐技术演进：PPO 强化学习 vs DPO 直接偏好优化

## 1. 传统 RLHF (PPO) 的四模型复杂链路
需要维护 Actor 模型、Critic 模型、Reward 模型和 Reference 模型，训练极易不稳定，显存开销巨大，超参数调试难度极高。

## 2. DPO (Direct Preference Optimization) 的数学奇迹
Rafailov 等人证明，可以通过数学推导将强化学习中的奖励函数隐式参数化为策略模型与参考模型的对数几率差（Log-Ratio）。
- **去中心化训练**：完全摒弃独立的 Reward 和 Critic 模型。
- 直接利用偏好数据对 $(x, y_w, y_l)$ 构建二元交叉熵损失，训练稳定性与收敛速度实现质的飞跃。""")
]

print(f"Loaded {len(LLM_INFRA)} LLM Infra topics.")

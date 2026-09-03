# 参数高效微调：LoRA 低秩适应与 QLoRA 双重量化原理

## 1. LoRA (Low-Rank Adaptation)
- 冻结预训练大模型原本的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$。
- 引入低秩分解矩阵旁路：$\Delta W = B \times A$，其中 $A \in \mathbb{R}^{r \times k}$ 采用高斯初始化，$B \in \mathbb{R}^{d \times r}$ 初始化为 0，且秩 $r \ll \min(d, k)$（通常取 8 或 16）。
- 仅训练 $A$ 和 $B$，可训练参数量骤降至原模型的 0.1% 以下。

## 2. QLoRA (高效 4-bit 量化微调)
- **NF4 (NormalFloat 4)**：针对正态分布权重优化的理论最优量化数据类型。
- **双重量化 (Double Quantization)**：对量化本身的常数因子进行二次量化，每个参数节省 0.37 bit。
- **分页优化器 (Paged Optimizers)**：利用 CUDA 统一内存解决微调显存峰值 OOM 崩溃。

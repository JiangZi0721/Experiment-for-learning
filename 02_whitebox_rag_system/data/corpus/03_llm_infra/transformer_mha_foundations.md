# Transformer 标准多头注意力 (MHA) 架构推导

## 1. 缩放点积注意力 (Scaled Dot-Product Attention)
注意力核心公式：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **除以 $\sqrt{d_k}$ 的数学必要性**：当维度 $d_k$ 很大时，点积结果的方差为 $d_k$。如果不缩放，点积绝对值过大将导致 Softmax 函数进入极小梯度的饱和区（Saturation），造成反向传播梯度弥散。

## 2. Multi-Head Attention (MHA) 机制
- 将输入投影为 $h$ 组不同的 $Q_i, K_i, V_i$ 矩阵。
- 每个注意力头各自拥有独立的投影权重 $W_i^Q, W_i^K, W_i^V$。
- 在不同的表示子空间（Representation Subspaces）中并行捕捉长距离上下文依赖，最后拼接输出并乘以 $W^O$。

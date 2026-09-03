# FlashAttention 核心原理：Tiling 切块与重计算显存革新

## 1. 传统注意力的 GPU 显存墙问题
标准 Attention 需要在 GPU 高带宽显存（HBM）中完整物化 $N \times N$ 的注意力中间矩阵 $S = QK^T$ 和 $P = \text{softmax}(S)$。长上下文时 $O(N^2)$ 的读写 I/O 导致严重的访存开销。

## 2. FlashAttention 关键技术
- **Tiling 分块计算**：利用 GPU 片上极速 SRAM（SRAM 速度比 HBM 快一个数量级），将 $Q, K, V$ 划分为小块依次加载进 SRAM。
- **在线 Softmax 归一化 (Online Softmax)**：在不保存全局 Softmax 矩阵的情况下，通过动态缩放因子逐步合并部分 Softmax 的分子分母。
- **反向传播重计算 (Recomputation)**：在前向传播中完全不保存 $N \times N$ 的中间激活值，反向传播时直接利用 SRAM 快速重新计算，将空间复杂度从 $O(N^2)$ 降低到 $O(N)$。

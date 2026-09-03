# 张量并行 (Tensor Parallelism) Megatron-LM 矩阵切分

## 1. 为什么需要张量并行
当单层模型的参数量超过单张 GPU 显存上限时，必须将层内的权重矩阵在多个 GPU 间切分并发执行。

## 2. MLP 层的切分规范
- **第一层 $W_1$ (Column Parallel)**：将矩阵按列切分。输入 $X$ 广播到所有 GPU，并发执行 $Y_i = \text{GeLU}(X W_{1,i})$。
- **第二层 $W_2$ (Row Parallel)**：将矩阵按行切分。每张卡计算 $Z_i = Y_i W_{2,i}$。
- **跨卡聚合**：最后仅需执行一次 `All-Reduce (Sum)` 操作即可得到最终的 $Z = \sum Z_i$，通信与计算完美流水化。

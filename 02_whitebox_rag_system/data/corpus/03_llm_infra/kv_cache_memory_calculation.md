# 大模型推理 KV Cache 显存占用物理公式与推演

## 1. 显存占用公式
对于一个 Transformer 模型，每个 Token 在单层中需要缓存一个 Key 向量和一个 Value 向量。全局显存大小计算公式为：
$$\text{Memory}_{KV} = 2 \times 2 \times n_{layers} \times d_{model} \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes\_per\_elem}$$
- 第一个 $2$ 代表 Key 和 Value 两个张量。
- 第二个 $2$ 代表 FP16/BF16 精度占用 2 字节。
- $n_{layers}$ 为层数，$d_{model}$ 为隐藏层维度。

## 2. 实例测算
以 70B 模型（层数 80，隐藏层 8192，采用 FP16）为例：
- 单个 Token 的 KV 缓存占用约为 $2 \times 2 \times 80 \times 8192 = 2.62\text{ MB}$。
- 若并发并发数为 16，上下文长度达到 8K，仅 KV Cache 就需消耗近 $335\text{ GB}$ 显存，远超参数自身显存。

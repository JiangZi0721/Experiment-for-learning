# 分组查询注意力 (GQA): 显存带宽与模型容量的黄金折中

## 1. GQA 的分组折中思想
Grouped-Query Attention (GQA) 是 Llama-2-70B、Llama-3 等当代主流开源大模型的标准标配：
- 将 $H_Q$ 个 Query 头均匀划分为 $G$ 个组（Group）。
- 每个组内的 Query 头共享同一个 Key 头和 Value 头（即共有 $G$ 对 KV 头）。
- 当 $G=H_Q$ 时，GQA 退化为标准 MHA；当 $G=1$ 时，GQA 退化为 MQA。

## 2. 性能收益实测
- 在维持接近 MHA 的高表达能力的同时，推理时 KV Cache 的内存占用和带宽开销降低数倍，服务并发 Batch Size 提升 2~3 倍，是工业界公认的最佳平衡点。

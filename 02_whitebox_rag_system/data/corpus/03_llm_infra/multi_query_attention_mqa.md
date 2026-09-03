# 多查询注意力 (MQA): 极致压缩 Key-Value 头的工程革新

## 1. MHA 在大模型自回归解码中的瓶颈
在自回归推理（Inference / Decode）阶段，模型每次仅生成一个 Token。由于注意力操作需要读取历史全量 KV Cache，内存带宽（Memory Bandwidth Bound）成为主要吞吐瓶颈。

## 2. Multi-Query Attention (MQA) 设计
- **结构差异**：保持 Query 头数量（如 32 个头）不变，但将所有头共享**唯一的一个 Key 头和一个 Value 头**。
- **显存压制**：KV Cache 的显存占用直接骤降为原来的 $1/h$（例如减少 90% 以上）。
- **取舍代价**：大幅压缩了模型对复杂上下文的多重泛化与注意力表征能力，模型容量和评测分数有一定微弱损耗。

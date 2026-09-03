# 长序列并行：DeepSpeed Ulysses 与 Ring-Attention

## 1. 显存瓶颈转移至序列长度
当上下文长度扩展到 128K 或 1M 时，即使使用张量并行，单层注意力的中间激活值也会彻底打爆显存。

## 2. DeepSpeed Ulysses
- 在注意力头维度（Heads）和序列维度（Sequence）之间执行优雅的 `All-to-All` 全局转置通信。
- 序列被平均切分到各卡；在计算 Self-Attention 之前，将序列拼全并将头切开，使得单机内的计算单元依然可以复用标准 FlashAttention。

## 3. Ring-Attention
将注意力分块在环形网络中流动传递，允许上下文长度随着集群 GPU 数量的线性增加而实现无上限水平扩展。

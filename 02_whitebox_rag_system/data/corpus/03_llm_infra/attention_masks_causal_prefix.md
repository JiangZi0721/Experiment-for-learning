# 注意力掩码体系：因果因果掩码 (Causal Mask) 与前缀掩码 (Prefix Mask)

## 1. 因果掩码 (Causal Mask / Lower Triangular)
在自回归语言模型中，为了防止模型在预测第 $t$ 个 Token 时“穿越”偷看未来的信息，将注意力矩阵右上三角的权重强行填充为 $-\infty$，经 Softmax 后概率为 0。

## 2. 前缀掩码 (Prefix Masking)
在 PrefixLM（如 ChatGLM、Encoder-Decoder 混合架构）中：
- 前序 Prompt 部分的 Token 之间允许相互双向可见（提高对已知上下文的理解深度）。
- 生成的回答部分严格遵守单向因果掩码，实现生成式任务与理解式任务的最佳融合。

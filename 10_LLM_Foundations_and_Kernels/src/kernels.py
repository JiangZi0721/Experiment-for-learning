# -*- coding: utf-8 -*-
"""
Core LLM Operators & Attention Kernels Implementation:
1. RoPE (Rotary Position Embedding) 旋转位置编码
2. RMSNorm (Root Mean Square Normalization) 均方根归一化
3. LoRA (Low-Rank Adaptation) 低秩自适应微调层
4. FlashAttention-Style Online Softmax Tiling 核心切块推演
5. KV Cache 压缩架构 (MHA vs MQA vs GQA vs MLA) 显存建模
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    去掉均值中心化，仅通过均方根缩放，节约 7%~10% 计算开销并保持训练稳定
    y = x / sqrt(mean(x^2) + eps) * gamma
    """
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed

class RoPEEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding, 旋转位置编码)
    通过复数旋转将绝对位置乘入 Q, K 向量，使得二者内积自发只依赖相对位置距离 (m - n)
    """
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super(RoPEEmbedding, self).__init__()
        self.dim = dim
        # theta_i = 10000^(-2(i-1)/dim)
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)

        # 预先生成旋转位置角度网格
        seq_idx = torch.arange(max_seq_len).float()
        angles = torch.outer(seq_idx, theta)  # [max_seq_len, dim / 2]
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def rotate_half(self, x):
        """将相邻偶奇分量配对旋转: [x0, x1, x2, x3...] -> [-x1, x0, -x3, x2...]"""
        return torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)

    def forward(self, x, seq_len=None):
        """
        :param x: [batch, num_heads, seq_len, head_dim]
        """
        if seq_len is None:
            seq_len = x.shape[2]

        cos = self.cos_cached[:seq_len, :].repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len, :].repeat_interleave(2, dim=-1)  # [seq_len, head_dim]

        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(1)

        # R(x) = x * cos + rotate_half(x) * sin
        return (x * cos) + (self.rotate_half(x) * sin)

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation, 经典低秩自适应微调)
    h = W0 * x + (alpha / r) * B * A * x
    其中 W0 冻结，A 采用高斯初始化，B 采用全 0 初始化确保微调伊始输出恒等
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=16.0, lora_dropout=0.0):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.pretrained = nn.Linear(in_features, out_features, bias=False)
        self.pretrained.weight.requires_grad = False  # 冻结基座

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x):
        base_out = self.pretrained(x)
        if self.r > 0:
            lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
            return base_out + lora_out * self.scaling
        return base_out

def flash_attention_online_softmax_reference(Q, K, V, block_size=32):
    """
    FlashAttention 算法的 Python 原生白盒参考实现 (Online Softmax Tiling)
    展示无需在显存中落盘完整的 (N x N) Attention 矩阵，
    仅凭局部局部累积统计量 m (最大值) 和 l (分母累加和)，在 O(1) SRAM 显存内完成精确计算。
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(N, 1, device=Q.device)
    m = torch.full((N, 1), -float('inf'), device=Q.device)

    scale = 1.0 / math.sqrt(d)

    # 外层循环：遍历 Keys/Values 的分块 (SRAM 逐块加载)
    for j in range(0, N, block_size):
        K_j = K[j:j+block_size]  # [block_size, d]
        V_j = V[j:j+block_size]  # [block_size, d]

        # 内层循环：遍历 Query 的分块
        for i in range(0, N, block_size):
            Q_i = Q[i:i+block_size]  # [block_size, d]

            # 1. 局部点积 S_ij = Q_i * K_j^T * scale
            S_ij = torch.matmul(Q_i, K_j.t()) * scale  # [b_size, b_size]

            # 2. 计算当前块的新局部最大值
            m_prev = m[i:i+block_size]
            m_curr = torch.max(S_ij, dim=-1, keepdim=True).values
            m_new = torch.maximum(m_prev, m_curr)

            # 3. 计算对齐缩放系数
            alpha = torch.exp(m_prev - m_new)
            P_ij = torch.exp(S_ij - m_new)

            # 4. 更新分母求和 l
            l_prev = l[i:i+block_size]
            l_new = alpha * l_prev + P_ij.sum(dim=-1, keepdim=True)

            # 5. 更新输出矩阵 O
            O_prev = O[i:i+block_size]
            O_new = alpha * O_prev + torch.matmul(P_ij, V_j)

            O[i:i+block_size] = O_new
            l[i:i+block_size] = l_new
            m[i:i+block_size] = m_new

    # 最终统一除以精确归一化分母 l
    O = O / l
    return O

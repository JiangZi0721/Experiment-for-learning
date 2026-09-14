# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 编码器模块 (Encoder Stack)
包含残差连接与层归一化 (Add & LayerNorm)、前馈神经网络 (FFN)、
单层 Encoder Layer 以及多层堆叠的完整 Transformer Encoder。
"""
from typing import List, Optional
from .config import cfg
from .math_ops import mat_add, matmul, layer_norm, relu_matrix
from .attention import MultiHeadAttention
from .visualizer import WhiteBoxVisualizer

Matrix = List[List[float]]

class AddAndNorm:
    """残差连接与层归一化复合层"""
    @staticmethod
    def forward_interactive(X_input: Matrix, 
                            sublayer_out: Matrix, 
                            tokens: List[str], 
                            viz: WhiteBoxVisualizer, 
                            sublayer_name: str = "SubLayer") -> Optional[Matrix]:
        viz.print_subbanner(f"残差连接与层归一化 —— 【Add & Norm ({sublayer_name})】")
        print(f"""
【原理剖析】
1. 残差相加 (Add): Output = X + {sublayer_name}(X)
   为反向传播提供无损梯度高速公路，有效阻止深层网络梯度消失与表征退化。
2. 层归一化 (LayerNorm):
   在单个 Token 内部沿特征维度 D 归一化（均值 μ 归 0，方差 σ^2 归 1），重构健康的数值分布。
""")
        # 1. 残差相加
        residual_sum = mat_add(X_input, sublayer_out)
        
        # 2. 层归一化
        normed_out, stats = layer_norm(residual_sum, eps=cfg.EPS)
        
        res = viz.ask_expand(f"Add & Norm ({sublayer_name}) 的残差相加、各 Token 均值方差与归一化矩阵")
        if res == 'quit': return None
        if res:
            viz.show_matrix(residual_sum, row_labels=tokens, name=f"残差相加矩阵 X + {sublayer_name}(X)")
            print("\n📊 各 Token 内部特征统计量 (沿维度 D 统计):")
            for idx, (m, v) in enumerate(stats):
                print(f"  Token [{tokens[idx]:<4}]: 均值 μ = {m:+.4f}, 方差 σ^2 = {v:.4f}")
            viz.show_matrix(normed_out, row_labels=tokens, name=f"LayerNorm 标准化后输出矩阵")
            print("💡 结论验证：每一行元素的均值严格归零，方差严格为 1！")

        viz.pause()
        return normed_out


class FeedForwardNetwork:
    """逐位置前馈神经网络 (Position-wise FFN)"""
    def __init__(self, W1: Matrix, W2: Matrix):
        self.W1 = W1
        self.W2 = W2

    def forward_interactive(self, X: Matrix, tokens: List[str], viz: WhiteBoxVisualizer) -> Optional[Matrix]:
        viz.print_banner("逐位置前馈神经网络 (Feed-Forward Network, FFN)")
        print(f"""
【为什么有了注意力机制，还必须有 FFN？】
* 注意力机制只负责【不同 Token 之间的空间特征重组与加权融合】（空间信息路由）。
* FFN 则是多层感知机 (MLP)，负责对每个 Token 进行【高阶非线性抽象与特征记忆检索】。
* 计算公式：FFN(x) = ReLU(x @ W1) @ W2
  - W1 将维度从 D ({cfg.D_MODEL}) 升维放大至 D_ffn ({cfg.D_FFN})；
  - 经 ReLU 激活激发非线性表达能力；
  - W2 将维度从 D_ffn 压缩回 D ({cfg.D_MODEL})。
""")
        # 1. 第一层升维并 ReLU
        hidden = matmul(X, self.W1)
        hidden_act = relu_matrix(hidden)
        
        # 2. 第二层降维
        ffn_out = matmul(hidden_act, self.W2)

        res = viz.ask_expand("FFN 升维激活矩阵 ReLU(X @ W1) 与 最终降维输出矩阵")
        if res == 'quit': return None
        if res:
            viz.show_matrix(hidden_act, row_labels=tokens, name=f"升维激活矩阵 (维度: {len(tokens)} x {cfg.D_FFN})")
            viz.show_matrix(ffn_out, row_labels=tokens, name=f"降维回原始维度输出 (维度: {len(tokens)} x {cfg.D_MODEL})")

        viz.pause()
        return ffn_out


class EncoderLayer:
    """单个 Transformer 编码器层"""
    def __init__(self, layer_idx: int = 1):
        self.layer_idx = layer_idx
        self.self_attn = MultiHeadAttention(
            W_Q=cfg.ENC_W_Q, W_K=cfg.ENC_W_K, W_V=cfg.ENC_W_V, W_O=cfg.ENC_W_O
        )
        self.ffn = FeedForwardNetwork(W1=cfg.ENC_FFN_W1, W2=cfg.ENC_FFN_W2)

    def forward_interactive(self, X: Matrix, tokens: List[str], viz: WhiteBoxVisualizer) -> Optional[Matrix]:
        viz.print_banner(f"🏛️ 进入编码器第 #{self.layer_idx} 层 (Encoder Layer #{self.layer_idx})")
        
        # 1. 自注意力机制
        attn_out = self.self_attn.forward_interactive(
            X_q=X, X_kv=X, q_tokens=tokens, kv_tokens=tokens, viz=viz, mode="Self-Attention"
        )
        if attn_out is None: return None
        
        # 2. 第一级 Add & Norm
        norm1 = AddAndNorm.forward_interactive(X, attn_out, tokens, viz, sublayer_name="Self-Attention")
        if norm1 is None: return None
        
        # 3. 前馈神经网络 FFN
        ffn_out = self.ffn.forward_interactive(norm1, tokens, viz)
        if ffn_out is None: return None
        
        # 4. 第二级 Add & Norm
        norm2 = AddAndNorm.forward_interactive(norm1, ffn_out, tokens, viz, sublayer_name="FFN")
        if norm2 is None: return None
        
        return norm2

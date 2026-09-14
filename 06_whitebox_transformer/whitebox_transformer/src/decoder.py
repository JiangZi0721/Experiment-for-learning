# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 解码器模块 (Decoder Stack & Output Generator)
包含掩码自注意力 (Masked Self-Attention)、编解码器交叉注意力 (Cross-Attention)、
解码器 FFN、三级 Add & Norm，以及最终线性输出层 (Linear Projection) 与词表 Softmax 预测。
"""
from typing import List, Optional, Tuple
from .config import cfg
from .math_ops import matmul, softmax_row, create_causal_mask
from .attention import MultiHeadAttention
from .encoder import AddAndNorm, FeedForwardNetwork
from .visualizer import WhiteBoxVisualizer

Matrix = List[List[float]]
Vector = List[float]

class DecoderLayer:
    """
    单个 Transformer 解码器层 (拥有三大子层结构)
    1. Masked Multi-Head Self-Attention + Add & Norm
    2. Multi-Head Cross-Attention (Q来自Decoder, K/V来自Encoder) + Add & Norm
    3. Feed-Forward Network (FFN) + Add & Norm
    """
    def __init__(self, layer_idx: int = 1):
        self.layer_idx = layer_idx
        # 1. 掩码自注意力
        self.masked_self_attn = MultiHeadAttention(
            W_Q=cfg.DEC_SELF_W_Q, W_K=cfg.DEC_SELF_W_K, W_V=cfg.DEC_SELF_W_V, W_O=cfg.DEC_SELF_W_O
        )
        # 2. 交叉注意力
        self.cross_attn = MultiHeadAttention(
            W_Q=cfg.DEC_CROSS_W_Q, W_K=cfg.DEC_CROSS_W_K, W_V=cfg.DEC_CROSS_W_V, W_O=cfg.DEC_CROSS_W_O
        )
        # 3. 前馈网络
        self.ffn = FeedForwardNetwork(W1=cfg.DEC_FFN_W1, W2=cfg.DEC_FFN_W2)

    def forward_interactive(self, 
                            X_tgt: Matrix, 
                            tgt_tokens: List[str], 
                            encoder_memory: Matrix, 
                            src_tokens: List[str], 
                            viz: WhiteBoxVisualizer) -> Optional[Matrix]:
        viz.print_banner(f"🏛️ 进入解码器第 #{self.layer_idx} 层 (Decoder Layer #{self.layer_idx})")
        
        # ======================================================================
        # 子层 1：因果掩码自注意力 (Masked Self-Attention) + Add & Norm
        # ======================================================================
        viz.print_subbanner("子层 1：因果掩码自注意力 (Masked Self-Attention)")
        # 生成下三角掩码
        causal_mask = create_causal_mask(len(tgt_tokens))
        
        masked_self_out = self.masked_self_attn.forward_interactive(
            X_q=X_tgt, 
            X_kv=X_tgt, 
            q_tokens=tgt_tokens, 
            kv_tokens=tgt_tokens, 
            viz=viz, 
            mode="Masked Self-Attention", 
            mask=causal_mask
        )
        if masked_self_out is None: return None
        
        norm1 = AddAndNorm.forward_interactive(
            X_tgt, masked_self_out, tgt_tokens, viz, sublayer_name="Masked Self-Attention"
        )
        if norm1 is None: return None

        # ======================================================================
        # 子层 2：交叉注意力机制 (Cross-Attention) + Add & Norm
        # ======================================================================
        viz.print_subbanner("子层 2：编解码交叉注意力 (Cross-Attention: Decoder Q <- Encoder K,V)")
        print(f"当前 Query (解码端): {tgt_tokens} (M = {len(tgt_tokens)})")
        print(f"当前 Key/Value (编码端记忆): {src_tokens} (N = {len(src_tokens)})")
        
        cross_out = self.cross_attn.forward_interactive(
            X_q=norm1, 
            X_kv=encoder_memory, 
            q_tokens=tgt_tokens, 
            kv_tokens=src_tokens, 
            viz=viz, 
            mode="Cross-Attention", 
            mask=None  # 交叉注意力中解码器可以自由查看源端全部上下文，无需 Causal Mask
        )
        if cross_out is None: return None
        
        norm2 = AddAndNorm.forward_interactive(
            norm1, cross_out, tgt_tokens, viz, sublayer_name="Cross-Attention"
        )
        if norm2 is None: return None

        # ======================================================================
        # 子层 3：前馈神经网络 (FFN) + Add & Norm
        # ======================================================================
        viz.print_subbanner("子层 3：逐位置前馈网络 (FFN) 与最终收敛")
        ffn_out = self.ffn.forward_interactive(norm2, tgt_tokens, viz)
        if ffn_out is None: return None
        
        norm3 = AddAndNorm.forward_interactive(
            norm2, ffn_out, tgt_tokens, viz, sublayer_name="Decoder FFN"
        )
        if norm3 is None: return None

        return norm3


class OutputLinearHead:
    """解码器终点输出层：线性投影 (Linear) + 词表 Softmax (Next-Token Generator)"""
    def __init__(self, W_vocab: Matrix, vocab: List[str]):
        self.W_vocab = W_vocab
        self.vocab = vocab

    def forward_interactive(self, 
                            decoder_out: Matrix, 
                            step_idx: int, 
                            viz: WhiteBoxVisualizer) -> Optional[Tuple[str, int, Vector]]:
        viz.print_banner(f"🎯 最终生成层：输出线性投影与词表 Softmax 预测 (Step #{step_idx})")
        print(f"""
【预测原理】
* 解码器输出矩阵维度为 (M x D)，其中最后一行代表【当前最新生成位置的综合表征向量】。
* 线性层 (Linear Head): Logits = x_last @ W_vocab
  - 将维度从隐藏维度 D ({cfg.D_MODEL}) 投影映射到目标词表大小 |V| ({len(self.vocab)})。
* 归一化 (Softmax): Probabilities = Softmax(Logits)
  - 将无界的 Logits 转化为和为 1.0 的全词表概率分布。
* 决策策略 (Greedy Decoding / 贪婪解码):
  - 选取概率最高的 Token 作为下一个预测输出！
""")
        # 取最新位置的表征向量
        last_hidden = decoder_out[-1]
        
        # 矩阵乘法计算 logits: (1 x D) @ (D x |V|) -> (1 x |V|)
        logits = [0.0 for _ in range(len(self.vocab))]
        for v_idx in range(len(self.vocab)):
            logits[v_idx] = sum(last_hidden[d] * self.W_vocab[d][v_idx] for d in range(cfg.D_MODEL))

        # Softmax 转化为概率分布
        probs = softmax_row(logits)

        # 选出最高概率 Token
        best_id = max(range(len(probs)), key=lambda i: probs[i])
        predicted_token = self.vocab[best_id]

        res = viz.ask_expand("线性层投影 Logits、全词表概率分布与贪婪决策过程")
        if res == 'quit': return None
        if res:
            viz.show_matrix([logits], row_labels=["Logits"], col_labels=self.vocab, name="未归一化对数概率 Logits (1 x |V|)")
            viz.show_prob_distribution(probs, self.vocab, top_k=len(self.vocab))
            print(f"\n🏆 贪婪决策命中：Token '{predicted_token}' (ID: {best_id}, 概率: {probs[best_id]*100:.2f}%)")

        viz.pause()
        return predicted_token, best_id, probs

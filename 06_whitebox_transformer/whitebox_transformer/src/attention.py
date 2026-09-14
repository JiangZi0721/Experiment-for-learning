# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 注意力引擎 (Attention Engine)
包含缩放点积注意力 (Scaled Dot-Product Attention)、多头注意力 (Multi-Head Attention)、
因果掩码注意力 (Masked Self-Attention) 与 编解码器交互交叉注意力 (Cross-Attention)。
"""
import math
from typing import List, Tuple, Optional, Any
from .config import cfg
from .math_ops import matmul, mat_transpose, softmax_matrix, mat_add
from .visualizer import WhiteBoxVisualizer

Matrix = List[List[float]]

class ScaledDotProductAttention:
    """缩放点积注意力核心算子"""
    @staticmethod
    def forward(Q: Matrix, K: Matrix, V: Matrix, 
                mask: Optional[List[List[bool]]] = None) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
        """
        计算: Attention(Q, K, V) = Softmax( (Q @ K^T) / sqrt(d_k) + Mask ) @ V
        返回: (输出上下文矩阵 B, 原始点积矩阵, 缩放后矩阵, 注意力权重矩阵)
        """
        d_k = len(Q[0])
        scale = math.sqrt(d_k)
        
        # 1. 计算点积 Q @ K^T
        K_T = mat_transpose(K)
        raw_scores = matmul(Q, K_T)
        
        # 2. 缩放除以 sqrt(d_k)
        scaled_scores = [[val / scale for val in row] for row in raw_scores]
        
        # 3. 应用 Softmax (内部支持 mask_row)
        attn_weights = softmax_matrix(scaled_scores, mask=mask)
        
        # 4. 与 Value 加权求和
        output_B = matmul(attn_weights, V)
        
        return output_B, raw_scores, scaled_scores, attn_weights


class MultiHeadAttention:
    """
    通用多头注意力模块
    可作为：
    1. Encoder Self-Attention (无掩码自注意力)
    2. Decoder Masked Self-Attention (下三角因果自注意力)
    3. Decoder Cross-Attention (Q来自Decoder，K,V来自Encoder Memory)
    """
    def __init__(self, W_Q: Matrix, W_K: Matrix, W_V: Matrix, W_O: Matrix, 
                 n_heads: int = cfg.N_HEADS, d_model: int = cfg.D_MODEL):
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.W_O = W_O
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

    def forward_interactive(self, 
                            X_q: Matrix, 
                            X_kv: Matrix, 
                            q_tokens: List[str], 
                            kv_tokens: List[str], 
                            viz: WhiteBoxVisualizer, 
                            mode: str = "Self-Attention", 
                            mask: Optional[List[List[bool]]] = None) -> Optional[Matrix]:
        """
        交互式多头注意力全流程推演
        mode: "Self-Attention", "Masked Self-Attention", 或 "Cross-Attention"
        """
        viz.print_banner(f"多头注意力机制推演 —— 【{mode}】")
        
        # 理论先导讲解
        if mode == "Self-Attention":
            print("""
【自注意力 (Self-Attention) 核心原理】
* Q, K, V 全部由同一个输入矩阵 X 通过不同的投影矩阵生成。
* 每个 Token 扮演三个角色：
  1. Query (q_i): 发出检索意图的提问者。
  2. Key (k_j): 供他人检索匹配的特征标签。
  3. Value (v_j): 准备贡献给全句的实质内容。
""")
        elif mode == "Masked Self-Attention":
            print("""
【因果掩码自注意力 (Masked Self-Attention) 核心原理】
* 在自回归（Autoregressive）解码过程中，模型生成当前词时【严禁偷看未来的词】。
* 在训练阶段，我们一次性将整句目标文本送入模型，为了杜绝“穿越作弊”，必须在注意力矩阵右上三角施加【因果掩码 (Causal Mask)】。
* 做法：将未来位置的注意力分数强行置为 -∞，经过 Softmax 之后，未来词的注意力权重【绝对为 0】！
""")
        elif mode == "Cross-Attention":
            print("""
【交叉注意力 (Cross-Attention / 编解码交互) 核心原理】
* 编解码器信息交汇的咽喉要道！
* 灵魂角色划分：
  - Query (Q): 来自【解码器 (Decoder) 前一层】（代表：“我当前翻译/生成到了这一步，我急需什么源端信息？”）。
  - Key (K) & Value (V): 来自【编码器 (Encoder) 最终输出记忆矩阵 (Memory)】（代表：“这是源端全文的真实事实依据”）。
* 维度奇迹：即使解码端只有 2 个词，编码端有 100 个词，Q(2xD) @ K^T(Dx100) 得到 (2x100) 注意力图，
  再乘以 V(100xD)，输出依然精确保持 (2xD)！完美解耦源语言与目标语言的句子长度！
""")

        # 1. 线性投影
        Q = matmul(X_q, self.W_Q)
        K = matmul(X_kv, self.W_K)
        V = matmul(X_kv, self.W_V)

        # 询问是否展开 Q, K, V 投影结果
        res_qkv = viz.ask_expand(f"{mode} 的 Q, K, V 线性投影矩阵与向量数值")
        if res_qkv == 'quit': return None
        if res_qkv:
            viz.show_matrix(Q, row_labels=[f"q({t})" for t in q_tokens], name=f"Query 矩阵 Q ({len(Q)}x{len(Q[0])})")
            viz.show_matrix(K, row_labels=[f"k({t})" for t in kv_tokens], name=f"Key 矩阵 K ({len(K)}x{len(K[0])})")
            viz.show_matrix(V, row_labels=[f"v({t})" for t in kv_tokens], name=f"Value 矩阵 V ({len(V)}x{len(V[0])})")

        # 2. 拆分多头并计算
        head_outputs = []
        head_attn_maps = []
        
        for h in range(self.n_heads):
            # 切片子空间
            start_idx = h * self.d_k
            end_idx = (h + 1) * self.d_k
            
            Q_h = [row[start_idx:end_idx] for row in Q]
            K_h = [row[start_idx:end_idx] for row in K]
            V_h = [row[start_idx:end_idx] for row in V]
            
            out_h, raw_h, scaled_h, attn_h = ScaledDotProductAttention.forward(Q_h, K_h, V_h, mask=mask)
            head_outputs.append(out_h)
            head_attn_maps.append(attn_h)

        # 3. 拼接所有头 (Concat)
        concat_out = []
        for i in range(len(X_q)):
            merged_row = []
            for h in range(self.n_heads):
                merged_row.extend(head_outputs[h][i])
            concat_out.append(merged_row)

        # 4. 输出线性层 W_O 融合
        final_mha = matmul(concat_out, self.W_O)

        # 询问是否展开多头注意力图与拼接矩阵
        res_heads = viz.ask_expand(f"{mode} 各子空间注意力热力图、多头 Concat 拼接与 W_O 融合输出")
        if res_heads == 'quit': return None
        if res_heads:
            for h in range(self.n_heads):
                viz.print_subbanner(f"Head #{h+1} 独立注意力子空间 (维度 d_k = {self.d_k})")
                viz.show_attention_map(head_attn_maps[h], q_tokens, kv_tokens, name=f"Head #{h+1} Attention Map")
            
            viz.show_matrix(concat_out, row_labels=q_tokens, name="多头横向拼接矩阵 Concat(Head_1, Head_2)")
            viz.show_matrix(final_mha, row_labels=q_tokens, name=f"{mode} 最终输出 = Concat @ W_O")
            print("💡 透视解读：不同的头捕获了不同的语境依赖模式，经 W_O 融合后重新回到统一语义表征。")

        viz.pause()
        return final_mha

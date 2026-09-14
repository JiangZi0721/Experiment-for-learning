# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 输入预处理模块 (Embedding & Positional Encoding)
包含词嵌入查表、正弦余弦位置编码、以及针对用户直觉（One-Hot 拼接投影 vs 逐元素相加）的数学对比透视。
"""
import math
from typing import List, Tuple, Optional
from .config import cfg
from .math_ops import mat_add, matmul
from .visualizer import WhiteBoxVisualizer

Matrix = List[List[float]]

class TokenEmbedding:
    """词嵌入层：将离散 Token 映射为 D 维稠密连续向量"""
    def __init__(self, vocab: List[str], embedding_table: Matrix):
        self.vocab = vocab
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.embedding_table = embedding_table
        self.d_model = len(embedding_table[0])

    def forward(self, tokens: List[str]) -> Matrix:
        vectors = []
        for t in tokens:
            tid = self.token2id.get(t, 0)
            # 深拷贝向量
            vectors.append(list(self.embedding_table[tid]))
        return vectors


class PositionalEncoding:
    """正弦/余弦绝对位置编码 (Vaswani et al., 2017)"""
    def __init__(self, d_model: int = cfg.D_MODEL):
        self.d_model = d_model

    def generate(self, seq_len: int) -> Matrix:
        pe = []
        for pos in range(seq_len):
            row = []
            for i in range(self.d_model // 2):
                denom = 10000.0 ** ((2 * i) / self.d_model)
                row.append(math.sin(pos / denom))
                row.append(math.cos(pos / denom))
            pe.append(row)
        return pe

    @staticmethod
    def demonstrate_one_hot_vs_addition(viz: WhiteBoxVisualizer):
        """
        对比透视：用户的直觉（One-Hot 拼接 + 线性映射）与标准 Transformer（直接逐元素相加）
        """
        viz.print_subbanner("深度专题：One-Hot 拼接投影 vs 逐元素相加的数学本质")
        print("""
【用户的天才猜想与代数等价性】
设词嵌入向量为 e_token ∈ R^(1 x D)，位置以独热向量 e_pos ∈ R^(1 x N) 表示。
将两者水平拼接为：[e_token, e_pos] ∈ R^(1 x (D + N))。
乘以一个投影权重矩阵 W = [W_token; W_pos] ∈ R^((D + N) x D)：
    [e_token, e_pos] @ [W_token; W_pos] = e_token @ W_token + e_pos @ W_pos

👉 极其精彩的结论：
   因为 e_pos 是 One-Hot 向量，e_pos @ W_pos 本质上就是【从 W_pos 中查出第 pos 行的位置嵌入向量】！
   只要 W_token 取单位矩阵 I，拼接投影在数学上就【严格等价于词嵌入向量与位置嵌入向量直接相加】！

【为什么标准 Transformer 坚持直接逐元素相加？】
1. 计算与内存经济性：
   若拼接，每层自注意力投影矩阵 W_Q, W_K 必须承受 (D + N) 维度，计算量随句长 N 增加而膨胀。
2. 高维几何的天然正交性：
   在 D=512 的高维空间中，特征空间极其广袤。随机两个向量近乎正交。
   词语义与位置编码相加后，各自在几何子空间中保持独立，后续的多头投影能轻松将其解耦解出。
""")


class InputEmbeddingPipeline:
    """完整的输入预处理流：Token -> Embedding + PE -> X"""
    def __init__(self, vocab: List[str], embedding_table: Matrix, is_target: bool = False):
        self.tokenizer = TokenEmbedding(vocab, embedding_table)
        self.pe_gen = PositionalEncoding(d_model=len(embedding_table[0]))
        self.is_target = is_target

    def run_interactive(self, tokens: List[str], viz: WhiteBoxVisualizer) -> Optional[Matrix]:
        name_prefix = "目标端 (Decoder)" if self.is_target else "源端 (Encoder)"
        viz.print_banner(f"{name_prefix} 输入预处理：Token Embedding 与位置编码 (Positional Encoding)")
        
        print(f"当前输入 Token 序列: {tokens} (序列长度 N = {len(tokens)})")
        print("预处理目标：将离散字符转换为携带位置坐标信息的连续高维输入矩阵 X ∈ R^(N x D)")
        
        # 1. 词嵌入查表
        emb_matrix = self.tokenizer.forward(tokens)
        
        # 2. 生成正弦余弦位置编码
        pe_matrix = self.pe_gen.generate(len(tokens))
        
        # 3. 逐元素相加
        X_input = mat_add(emb_matrix, pe_matrix)

        res = viz.ask_expand(f"{name_prefix} 词嵌入矩阵、正弦 PE 矩阵与相加融合矩阵 X")
        if res == 'quit':
            return None
        if res:
            viz.show_matrix(emb_matrix, row_labels=tokens, name=f"{name_prefix} 词嵌入矩阵 E (无位置信息)")
            viz.show_matrix(pe_matrix, row_labels=[f"Pos_{i}" for i in range(len(tokens))], name="正弦/余弦位置编码矩阵 PE")
            viz.show_matrix(X_input, row_labels=tokens, name=f"{name_prefix} 预处理最终输出矩阵 X = E + PE")
            print("💡 透视解读：每个词向量此时不仅拥有本身的词义特征，还被刻上了精确的时间/空间坐标印记！")

        viz.pause()
        return X_input

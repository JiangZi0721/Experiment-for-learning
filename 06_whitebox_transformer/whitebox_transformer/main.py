# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 交互式白盒透视学习系统 - 主程序入口
运行方式:
    python main.py
或在父目录下:
    python main.py
"""
import sys
import os

# 确保 Windows 终端 UTF-8 编码兼容
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 将当前目录与上级目录加入 sys.path，保证包导入万无一失
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
from course_index import COURSE_INDEX
from src.config import cfg
from src.visualizer import WhiteBoxVisualizer
from src.embedding import PositionalEncoding, InputEmbeddingPipeline
from src.attention import MultiHeadAttention
from src.encoder import EncoderLayer
from src.decoder import DecoderLayer, OutputLinearHead
from src.pipeline import WhiteBoxTransformerPipeline

def print_menu():
    print("\n" + "═" * 76)
    print("      🧭 WhiteBox Transformer 架构全景白盒透视系统 (Standard Edition)")
    print("═" * 76)
    print("  [1] 🌟 端到端 Seq2Seq 全流程推演 (中文输入 -> Encoder -> Decoder 自回归生成英文)")
    print("  [2] 🏛️ 深入透视【编码器 Encoder】完整层架构 (Embedding -> MHA -> Add&Norm -> FFN)")
    print("  [3] 🎭 深入透视【解码器 Decoder】因果掩码与交叉注意力 (Masked Self-Attn & Cross-Attn)")
    print("  [4] 🎯 深入透视【输出生成层】线性投影与词表 Softmax 概率分布")
    print("  [5] 📐 理论专题透视与数学证明 (相加vs拼接 / 除以sqrt(d) / LayerNorm / 多头本质)")
    print("  [0] 🚪 退出系统")
    print("═" * 76)

def run_theory_topics(viz: WhiteBoxVisualizer):
    while True:
        print("\n" + "─" * 60)
        print("                📐 理论专题与数学证明")
        print("─" * 60)
        print("  [1] 专题 1：One-Hot 拼接投影 vs 直接相加的代数证明 (回应您的直觉)")
        print("  [2] 专题 2：为什么除以 sqrt(d_k)？(Softmax 饱和与梯度弥散推导)")
        print("  [3] 专题 3：多头注意力为什么不增加计算量？(FLOPs 对比与子空间理论)")
        print("  [4] 专题 4：为什么坚决使用 LayerNorm 而非 BatchNorm？(NLP 序列动态特性)")
        print("  [0] 返回上级主菜单")
        print("─" * 60)
        ch = input("请选择专题编号 (0-4): ").strip()
        if ch == '1':
            PositionalEncoding.demonstrate_one_hot_vs_addition(viz)
            viz.pause()
        elif ch == '2':
            viz.print_banner("专题 2：除以 sqrt(d_k) 的数学方差推导")
            print("""
【假设条件】
设 q = [q_1, ..., q_{d_k}], k = [k_1, ..., k_{d_k}] 为 d_k 维向量。
假设分量 q_m, k_m 独立同分布，均值 E[q_m] = E[k_m] = 0，方差 Var(q_m) = Var(k_m) = 1。

【点积方差计算】
点积为：S = q · k = sum_{m=1}^{d_k} (q_m * k_m)
由于独立性：
  E[q_m * k_m] = E[q_m] * E[k_m] = 0
  Var(q_m * k_m) = E[(q_m * k_m)^2] - (E[q_m * k_m])^2
                 = E[q_m^2] * E[k_m^2] - 0
                 = 1 * 1 = 1
因此，d_k 个方差为 1 的独立变量相加：
  Var(S) = sum_{m=1}^{d_k} Var(q_m * k_m) = d_k
  标准差 Std(S) = sqrt(d_k)！

【灾难性后果】
当 d_k 较大时（如 64），点积的值域大幅扩散，进入几十量级。
当大数值送入 Softmax(z_i) = exp(z_i) / sum(exp(z_j)) 时：
  最大值处的概率趋近于 1，其余几乎全部为 0。
此时局部导数 ∂Softmax(z_i)/∂z_j = p_i * (δ_{ij} - p_j) ≈ 0！
反向传播梯度直接消失，模型参数彻底死锁停止学习！

【终极解法】
除以 sqrt(d_k) 使得 Var(S / sqrt(d_k)) = Var(S) / d_k = d_k / d_k = 1！
将方差恒定拉回到 1.0，让激活值始终处于 Softmax 导数最敏感、学习最迅速的黄金梯度区！
""")
            viz.pause()
        elif ch == '3':
            viz.print_banner("专题 3：多头注意力机制的计算复杂度与子空间智慧")
            print("""
【计算量 FLOPs 等价性证明】
设序列长度为 N，模型维度为 D，头数为 h，每个头维度为 d_k = D / h。
1. 单头点积复杂度：
   Q @ K^T: (N x D) @ (D x N) -> N * N * D 次乘加运算
   Softmax @ V: (N x N) @ (N x D) -> N * N * D 次乘加运算
   单头总复杂度 = 2 * N^2 * D

2. h 个头的多头点积复杂度：
   单个头 Q_i @ K_i^T: (N x d_k) @ (d_k x N) -> N * N * (D / h)
   单个头 Softmax @ V_i: (N x N) @ (N x (D / h)) -> N * N * (D / h)
   单个头总复杂度 = 2 * N^2 * (D / h)
   全部 h 个头的总复杂度 = h * [2 * N^2 * (D / h)] = 2 * N^2 * D！

👉 结论：多头机制【完全没有增加任何浮点计算负担】！

【物理本质：多表征子空间】
单头注意力就像只用一种滤镜看世界（容易被最显眼的词完全霸占注意力）。
多头机制通过将高维空间切分成多个独立子空间，允许不同头并行探索：
- Head 1: 关注紧邻修饰语
- Head 2: 关注跨句主谓宾搭配
- Head 3: 关注情感指向与指代关系
最终通过 W_O 线性融合，集百家之所长！
""")
            viz.pause()
        elif ch == '4':
            viz.print_banner("专题 4：为什么 NLP 坚决选择 LayerNorm 而非 BatchNorm？")
            print("""
【BatchNorm 的致命软肋】
1. 依赖跨 Batch 统计量：
   BatchNorm 是“横向”在整个 Batch 的所有句子同一维度求均值方差。
   在 NLP 中，每个句子的实际长度各不相同（必须大量填充 <PAD>）。
   不同 Batch 之间句子长度的剧烈差异导致计算出的统计量严重抖动。
2. 小 Batch 崩溃：
   若推理时只有 1 条句子，BatchNorm 无法计算有效 Batch 方差。

【LayerNorm 的完美适配】
1. 单样本独立计算：
   LayerNorm 是“纵向”在每个样本自身、单个 Token 的所有特征维度 D 上求均值方差。
   无论 Batch 是 1 还是 1024，无论句子长 4 个词还是 512 个词，
   每个词都独立计算自己的均值和方差，完全不受序列长度变化和 Batch 大小的干扰！
""")
            viz.pause()
        elif ch == '0':
            break

def main():
    parser = argparse.ArgumentParser(description="WhiteBox Transformer CLI")
    parser.add_argument("--mode", type=str, default="menu", choices=["menu", "full", "encoder", "decoder", "theory"])
    parser.add_argument("--list-lessons", action="store_true", help="列出 30 讲课程目录后退出")
    args = parser.parse_args()

    if args.list_lessons:
        print("\n".join(COURSE_INDEX))
        return

    pipeline = WhiteBoxTransformerPipeline()
    viz = pipeline.viz

    if args.mode == "full":
        pipeline.run_autoregressive_generation(cfg.SRC_VOCAB)
        return
    elif args.mode == "encoder":
        pipeline.run_encoder_phase(cfg.SRC_VOCAB)
        return

    while True:
        print_menu()
        choice = input("请选择功能编号 (0-5): ").strip()
        
        if choice == '1':
            pipeline.run_autoregressive_generation(cfg.SRC_VOCAB)
        elif choice == '2':
            pipeline.run_encoder_phase(cfg.SRC_VOCAB)
        elif choice == '3':
            viz.print_banner("独立透视：解码器 Decoder 内部机理推演")
            # 先快速获取 encoder memory
            X_src = pipeline.src_embedding_pipe.run_interactive(cfg.SRC_VOCAB, viz)
            if X_src:
                enc_mem = pipeline.encoder.forward_interactive(X_src, cfg.SRC_VOCAB, viz)
                if enc_mem:
                    sample_tgt = ["<BOS>", "I", "love"]
                    X_tgt = pipeline.tgt_embedding_pipe.run_interactive(sample_tgt, viz)
                    if X_tgt:
                        pipeline.decoder.forward_interactive(X_tgt, sample_tgt, enc_mem, cfg.SRC_VOCAB, viz)
        elif choice == '4':
            viz.print_banner("独立透视：输出线性头与词表 Softmax 预测")
            dummy_last = [[0.8, -0.2, 0.5, 0.7]]
            pipeline.output_head.forward_interactive(dummy_last, step_idx=1, viz=viz)
        elif choice == '5':
            run_theory_topics(viz)
        elif choice == '0':
            print("\n感谢使用 WhiteBox Transformer 全景透视系统，祝您深度学习之旅收获满满！")
            break
        else:
            print("输入无效，请输入 0 到 5 之间的数字。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断。")

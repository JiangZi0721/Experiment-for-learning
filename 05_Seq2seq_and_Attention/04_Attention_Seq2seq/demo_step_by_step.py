"""
================================================================================
Attention 机制全流程单步推演独立实验与原理解析脚本
(Standalone Step-by-Step Attention Inference & Mechanism Demonstration)

本脚本完全独立于特定任务，用于直观、具象地展示 Attention 机制在序列生成（预测）
全生命周期中每一步的数学计算、张量流向、注意力分布动态转移过程。
================================================================================
"""

import sys
import os
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子以确保数值确定性与可复现性
torch.manual_seed(42)

class GeneralAttentionDemo(nn.Module):
    """
    通用 Seq2Seq Attention 演示模型 (Task-Agnostic)
    实现基于点积的对齐打分、Softmax 归一化、动态上下文加权与特征融合。
    """
    def __init__(self, vocab_size=20, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 编码器：输出全部时间步隐藏状态 hs = [h_1, ..., h_{T_enc}]
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 解码器：单步循环单元
        self.decoder_cell = nn.LSTMCell(embed_dim, hidden_dim)
        
        # 融合层与词表分类投影层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward_encoder(self, x_seq):
        """
        阶段一：编码器生成全息特征矩阵
        输入: x_seq [1, T_enc]
        输出: hs [1, T_enc, H], (h_T, c_T)
        """
        x_embed = self.embedding(x_seq) # [1, T_enc, D]
        hs, (h_T, c_T) = self.encoder_lstm(x_embed) # hs: [1, T_enc, H]
        return hs, (h_T.squeeze(0), c_T.squeeze(0))

    def attention_step(self, decoder_state_s, hs):
        """
        阶段二核心：单步注意力打分与动态上下文生成
        decoder_state_s: 当前时间步解码隐状态 Query [1, H]
        hs: 编码器所有时间步隐状态 Keys/Values [1, T_enc, H]
        """
        # 1. 点积内积打分 (Score / Energy)
        # s: [1, 1, H], hs: [1, T_enc, H] -> scores: [1, T_enc]
        scores = torch.bmm(hs, decoder_state_s.unsqueeze(2)).squeeze(2)
        
        # 2. Softmax 归一化为概率对齐权重 a
        attn_weights = F.softmax(scores, dim=1) # [1, T_enc]
        
        # 3. 加权求和得到当前步定制的动态上下文向量 c
        # a: [1, 1, T_enc], hs: [1, T_enc, H] -> context: [1, H]
        context = torch.bmm(attn_weights.unsqueeze(1), hs).squeeze(1)
        
        return attn_weights, context, scores


def run_live_prediction_demonstration():
    print("=" * 80)
    print("      Attention 机制在整个自回归预测过程中的单步推演独立实验")
    print("=" * 80)
    
    # 模拟一个通用的抽象序列任务：输入 6 个 token，解码预测 4 个 token
    source_tokens = ["<BOS>", "Data_A", "Action_X", "Entity_Y", "Modifier_Z", "<EOS>"]
    T_enc = len(source_tokens)
    
    model = GeneralAttentionDemo(vocab_size=25, embed_dim=8, hidden_dim=16)
    model.eval()
    
    # 模拟输入序列 Token ID
    x_input = torch.tensor([[1, 5, 8, 12, 19, 2]], dtype=torch.long)
    
    print(f"\n【输入源序列 (Source Sequence)】 长度 T_enc = {T_enc}:")
    for idx, tok in enumerate(source_tokens):
        print(f"  位置 i={idx}: 字符 '{tok}' (ID: {x_input[0, idx].item()})")
    
    # ---------------------------------------------------------
    # 阶段一：编码器执行 (不压缩，全息保留)
    # ---------------------------------------------------------
    print("\n" + "-" * 80)
    print("【阶段一：编码器执行】打破固定维度单向量瓶颈")
    print("-" * 80)
    with torch.no_grad():
        hs, (h_last, c_last) = model.forward_encoder(x_input)
    
    print(f"  传统 Seq2Seq 做法: 丢弃中间状态，仅取最后向量 h_last: 形状 {h_last.shape}")
    print(f"  ★ Attention 机制做法: 保留全部中间特征矩阵 hs: 形状 {hs.shape} [Batch=1, T_enc={T_enc}, Hidden=16]")
    print("  矩阵 hs 中每一行向量分别代表对应源字符在上下文中的全息表征。")
    
    # ---------------------------------------------------------
    # 阶段二：解码器自回归逐步预测 (Step-by-Step Autoregressive Decoding)
    # ---------------------------------------------------------
    print("\n" + "-" * 80)
    print("【阶段二：解码器自回归预测单步详细透视】")
    print("-" * 80)
    
    # 初始化解码器
    h_dec = h_last.clone()
    c_dec = c_last.clone()
    current_token_id = torch.tensor([0], dtype=torch.long) # <BOS>
    
    steps_to_predict = 4
    history_weights = []
    
    for t in range(1, steps_to_predict + 1):
        print(f"\n>>> [时间步 t = {t}] 开始预测第 {t} 个输出字符:")
        
        with torch.no_grad():
            # 1. 词嵌入
            y_emb = model.embedding(current_token_id) # [1, D]
            
            # 2. 解码器循环步更新，生成当前步的 Query 隐状态 s_t
            h_dec, c_dec = model.decoder_cell(y_emb, (h_dec, c_dec))
            query_s_t = h_dec # [1, H]
            print(f"  1. 解码器生成当前步的查询向量 (Query) s_{t}: 形状 {query_s_t.shape}")
            print(f"     物理含义: 解码器在说 —— '我现在准备预测第 {t} 个词，根据前文，我需要从源端寻找相关信息'")
            
            # 3. Attention 模块打分与加权
            attn_weights, context_c_t, raw_scores = model.attention_step(query_s_t, hs)
            
            print(f"  2. 计算与源端各位置的匹配得分 (Score = s_{t} · h_i):")
            scores_list = [f"{s:.3f}" for s in raw_scores[0].tolist()]
            print(f"     内积得分: {scores_list}")
            
            print(f"  3. Softmax 归一化得到注意力对齐分布 (Attention Weights a_{t}):")
            weights_list = attn_weights[0].tolist()
            for i, (tok, w) in enumerate(zip(source_tokens, weights_list)):
                bar = "█" * int(w * 30)
                print(f"     位置 i={i} [{tok:<10}]: 权重 {w*100:5.1f}% | {bar}")
            
            focused_idx = torch.argmax(attn_weights).item()
            print(f"     ★ 本步注意力最高聚焦于位置 i={focused_idx} ('{source_tokens[focused_idx]}')")
            
            # 4. 上下文向量合成
            print(f"  4. 动态上下文向量合成 (Context Vector) c_{t} = \u2211 a_({t}, i) * h_i:")
            print(f"     张量形状: {list(context_c_t.shape)}, 模长 L2-Norm: {context_c_t.norm().item():.4f}")
            print(f"     物理含义: 经由概率加权，提取了当前步专属的定制化源端语义！")
            
            # 5. 特征融合与最终字符预测
            combined = torch.cat([query_s_t, context_c_t], dim=-1) # [1, 2H]
            fused = torch.tanh(model.fusion(combined))             # [1, H]
            logits = model.classifier(fused)                       # [1, Vocab]
            probs = F.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            top_prob = probs[0, pred_id].item()
            
            print(f"  5. 拼接融合 [s_{t}; c_{t}] -> 投影到词表分布:")
            print(f"     融合向量维度: {list(fused.shape)} | 发射预测 Token ID: {pred_id} (置信度: {top_prob*100:.2f}%)")
            
            # 记录历史权重矩阵供全景对齐分析
            history_weights.append(weights_list)
            
            # 迭代到下一步的输入
            current_token_id = torch.tensor([pred_id], dtype=torch.long)

    # ---------------------------------------------------------
    # 阶段三：输出完整的 二维注意力对齐矩阵 (2D Alignment Matrix)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("【阶段三：全生命周期 二维注意力对齐矩阵 (2D Attention Matrix) 全景图】")
    print("=" * 80)
    col_header = "预测时间步 / 源位置"
    print(f"{col_header:<18}", end="")
    for tok in source_tokens:
        print(f"{tok:>12}", end="")
    print()
    print("-" * (18 + 12 * len(source_tokens)))
    
    for t, w_row in enumerate(history_weights, 1):
        print(f"Step t={t} (y_{t})       ", end="")
        for w in w_row:
            print(f"{w*100:11.2f}%", end="")
        print()
    print("-" * (18 + 12 * len(source_tokens)))

    print("\n推演核心归纳：")
    print("1. 为什么它是动态的？因为每一步的 Query s_t 都不同，驱动对齐权重 a_t 产生焦点位移。")
    print("2. 为什么它没有信息瓶颈？因为无论源序列多长，hs 都完整存在内存中，随时供解码器按需检索。")
    print("3. 为什么可解释？二维权重矩阵可以直接看清模型在生成每个词时到底注视着输入的哪个字符。")
    print("=" * 80)

if __name__ == "__main__":
    run_live_prediction_demonstration()

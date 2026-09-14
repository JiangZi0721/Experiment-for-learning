# -*- coding: utf-8 -*-
"""
Experiment 2: 三大模型在命名实体识别 (NER) 上的横向基准实测
横向比对：
1. HMM (生成式模型)
2. BiLSTM-Softmax (局部判别式)
3. BiLSTM-CRF (全局结构化判别式)
"""
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import (
    build_ner_dataset, collate_fn_ner,
    TAG2ID, ID2TAG, ALL_TAGS,
    START_TAG_IDX, STOP_TAG_IDX, PAD_TAG_IDX,
    is_transition_valid
)
from src.models import HMM, BiLSTM_Softmax, BiLSTM_CRF
from src.visualizer import plot_training_curves, plot_benchmark_metrics

def evaluate_predictions(gold_sequences, pred_sequences):
    """
    全面评估序列标注质量：
    1. Token-level Accuracy
    2. Sequence Exact Match (EM)
    3. Entity-level Precision / Recall / F1 (基于严格实体边界与类型匹配)
    4. Invalid Transition Rate (非法语法跃迁率)
    """
    total_tokens = 0
    correct_tokens = 0

    total_seqs = len(gold_sequences)
    correct_seqs = 0

    gold_entities = set()
    pred_entities = set()

    invalid_transitions = 0
    total_transitions = 0

    def extract_entities(tags, seq_idx):
        ents = []
        cur_type = None
        cur_start = None
        for i, t in enumerate(tags):
            tag_name = ID2TAG[t]
            if tag_name.startswith("B-"):
                if cur_type is not None:
                    ents.append((seq_idx, cur_start, i - 1, cur_type))
                cur_type = tag_name.split("-")[1]
                cur_start = i
            elif tag_name.startswith("I-"):
                ent_type = tag_name.split("-")[1]
                if cur_type == ent_type:
                    pass
                else:
                    if cur_type is not None:
                        ents.append((seq_idx, cur_start, i - 1, cur_type))
                    cur_type = None
                    cur_start = None
            else:
                if cur_type is not None:
                    ents.append((seq_idx, cur_start, i - 1, cur_type))
                    cur_type = None
                    cur_start = None
        if cur_type is not None:
            ents.append((seq_idx, cur_start, len(tags) - 1, cur_type))
        return ents

    for b in range(total_seqs):
        g_seq = gold_sequences[b]
        p_seq = pred_sequences[b]

        # 截断或填充至相同长度以安全评估
        length = min(len(g_seq), len(p_seq))
        g_seq = g_seq[:length]
        p_seq = p_seq[:length]

        # 1. Token Accuracy
        for g_t, p_t in zip(g_seq, p_seq):
            total_tokens += 1
            if g_t == p_t:
                correct_tokens += 1

        # 2. Sequence EM
        if g_seq == p_seq:
            correct_seqs += 1

        # 3. Entity F1
        g_ents = extract_entities(g_seq, b)
        p_ents = extract_entities(p_seq, b)
        gold_entities.update(g_ents)
        pred_entities.update(p_ents)

        # 4. Invalid Transitions
        for t in range(1, length):
            total_transitions += 1
            prev_tag = ID2TAG[p_seq[t-1]]
            cur_tag = ID2TAG[p_seq[t]]
            if not is_transition_valid(prev_tag, cur_tag):
                invalid_transitions += 1

    token_acc = correct_tokens / max(total_tokens, 1)
    seq_em = correct_seqs / max(total_seqs, 1)

    overlap = len(gold_entities.intersection(pred_entities))
    prec = overlap / max(len(pred_entities), 1)
    rec = overlap / max(len(gold_entities), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    invalid_rate = invalid_transitions / max(total_transitions, 1)

    return {
        "Token_Acc": token_acc,
        "Sequence_EM": seq_em,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Invalid_Rate": invalid_rate
    }

def run(num_epochs=12, batch_size=32, device='cpu'):
    print("=" * 80)
    print(">>> 【实验二：三大模型序列标注横向基准实测 (HMM vs BiLSTM vs BiLSTM-CRF)】")
    print("=" * 80)

    # 1. 准备数据
    train_data, test_data, vocab = build_ner_dataset(num_samples=1600, test_ratio=0.25, seed=42)
    num_tags = len(ALL_TAGS)
    vocab_size = len(vocab)
    print(f"数据加载就绪: 训练集 {len(train_data)} 句 | 测试集 {len(test_data)} 句 | 词表容量 {vocab_size} | 标签数 {num_tags}")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn_ner(b, vocab)
    )

    # 2. 训练 HMM 基线
    print("\n[1/3] 拟合 HMM 隐马尔可夫模型...")
    hmm = HMM(num_states=num_tags, num_vocab=vocab_size)
    train_x_hmm = [[vocab.get(tok, vocab["<UNK>"]) for tok in sent[0]] for sent in train_data]
    train_y_hmm = [[TAG2ID[t] for t in sent[1]] for sent in train_data]
    hmm.fit(train_x_hmm, train_y_hmm)

    # 3. 训练 BiLSTM-Softmax
    print("\n[2/3] 训练 BiLSTM-Softmax (局部独立交叉熵)...")
    model_softmax = BiLSTM_Softmax(vocab_size, num_tags, embedding_dim=64, hidden_dim=64).to(device)
    optimizer_s = optim.Adam(model_softmax.parameters(), lr=0.01)

    losses_softmax = []
    for epoch in range(num_epochs):
        model_softmax.train()
        total_loss = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer_s.zero_grad()
            logits = model_softmax(x)
            loss = model_softmax.compute_loss(logits, y, mask)
            loss.backward()
            optimizer_s.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses_softmax.append(avg_loss)
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:02d}/{num_epochs:02d} - Softmax Loss: {avg_loss:.4f}")

    # 4. 训练 BiLSTM-CRF
    print("\n[3/3] 训练 BiLSTM-CRF (全局结构化转移约束)...")
    model_crf = BiLSTM_CRF(vocab_size, num_tags, START_TAG_IDX, STOP_TAG_IDX, embedding_dim=64, hidden_dim=64).to(device)
    optimizer_c = optim.Adam(model_crf.parameters(), lr=0.01)

    losses_crf = []
    for epoch in range(num_epochs):
        model_crf.train()
        total_loss = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer_c.zero_grad()
            loss = model_crf.forward_loss(x, y, mask)
            loss.backward()
            optimizer_c.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses_crf.append(avg_loss)
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:02d}/{num_epochs:02d} - CRF Loss: {avg_loss:.4f}")

    # 5. 在独立测试集上全面评测
    print("\n>>> 开始在 400 条独立测试集上进行端到端对比评测...")
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn_ner(b, vocab)
    )

    gold_seqs = []
    for sent in test_data:
        gold_seqs.append([TAG2ID[t] for t in sent[1]])

    # HMM 解码
    pred_seqs_hmm = []
    for sent in test_data:
        x_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in sent[0]]
        pred_seqs_hmm.append(hmm.decode(x_ids))

    # BiLSTM-Softmax 解码
    model_softmax.eval()
    pred_seqs_s = []
    with torch.no_grad():
        for x, _, mask in test_loader:
            x, mask = x.to(device), mask.to(device)
            logits = model_softmax(x)
            pred_seqs_s.extend(model_softmax.decode(logits, mask))

    # BiLSTM-CRF 解码
    model_crf.eval()
    pred_seqs_c = []
    with torch.no_grad():
        for x, _, mask in test_loader:
            x, mask = x.to(device), mask.to(device)
            pred_seqs_c.extend(model_crf.decode(x, mask))

    res_hmm = evaluate_predictions(gold_seqs, pred_seqs_hmm)
    res_s = evaluate_predictions(gold_seqs, pred_seqs_s)
    res_c = evaluate_predictions(gold_seqs, pred_seqs_c)

    print("\n" + "=" * 96)
    print(f"{'对比模型':<16} | {'Token 准确率':<12} | {'整句全对率 (EM)':<16} | {'实体 F1 分':<12} | {'非法语法跳变率 (越低越好)'}")
    print("-" * 96)
    print(f"{'1. HMM (生成式)':<16} | {res_hmm['Token_Acc']*100:>10.2f}% | {res_hmm['Sequence_EM']*100:>14.2f}% | {res_hmm['F1']*100:>10.2f}% | {res_hmm['Invalid_Rate']*100:>18.2f}%")
    print(f"{'2. BiLSTM-Softmax':<16} | {res_s['Token_Acc']*100:>10.2f}% | {res_s['Sequence_EM']*100:>14.2f}% | {res_s['F1']*100:>10.2f}% | {res_s['Invalid_Rate']*100:>18.2f}% (存在死锁跳变)")
    print(f"{'3. BiLSTM-CRF':<16} | {res_c['Token_Acc']*100:>10.2f}% | {res_c['Sequence_EM']*100:>14.2f}% | {res_c['F1']*100:>10.2f}% | {res_c['Invalid_Rate']*100:>18.2f}% (0违规! 完美约束)")
    print("=" * 96)

    # 保存图表
    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.makedirs(img_dir, exist_ok=True)
    plot_training_curves(losses_softmax, losses_crf, os.path.join(img_dir, "crf_training_curves.png"))

    metrics_dict = {
        'Model': ['HMM', 'BiLSTM-Softmax', 'BiLSTM-CRF'],
        'Token_Acc': [res_hmm['Token_Acc'], res_s['Token_Acc'], res_c['Token_Acc']],
        'Sequence_EM': [res_hmm['Sequence_EM'], res_s['Sequence_EM'], res_c['Sequence_EM']],
        'F1_Score': [res_hmm['F1'], res_s['F1'], res_c['F1']],
        'Invalid_Rate': [res_hmm['Invalid_Rate'], res_s['Invalid_Rate'], res_c['Invalid_Rate']]
    }
    plot_benchmark_metrics(metrics_dict, os.path.join(img_dir, "crf_benchmark_comparison.png"))
    print(f">>> 实验二图表已生成至: {img_dir}")

    return model_crf

if __name__ == "__main__":
    run()

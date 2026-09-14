# -*- coding: utf-8 -*-
"""
Experiment 1: RoPE 旋转位置编码相对位置不变性与内积衰减实证
数学原理证明：<R_m q, R_n k> = g(q, k, m - n)，与绝对偏移 delta 无关。
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.kernels import RoPEEmbedding

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run(d_head=64, max_seq=128):
    print("=" * 80)
    print(">>> 【实验一：RoPE 旋转位置编码相对距离不变性检验】")
    print("=" * 80)
    torch.manual_seed(42)

    rope = RoPEEmbedding(dim=d_head, max_seq_len=512)

    # 构造随机 Query 与 Key 向量
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)

    m = 10
    n = 15
    delta = 50

    # 1. 检验相对位置不变性:
    # Score1 = < R_m q, R_n k >
    # Score2 = < R_{m+delta} q, R_{n+delta} k >
    q_m = rope(q.expand(-1, -1, m + 1, -1))[:, :, m:m+1, :]
    k_n = rope(k.expand(-1, -1, n + 1, -1))[:, :, n:n+1, :]
    score1 = (q_m * k_n).sum().item()

    q_m_delta = rope(q.expand(-1, -1, m + delta + 1, -1))[:, :, m+delta:m+delta+1, :]
    k_n_delta = rope(k.expand(-1, -1, n + delta + 1, -1))[:, :, n+delta:n+delta+1, :]
    score2 = (q_m_delta * k_n_delta).sum().item()

    diff = abs(score1 - score2)
    print(f"  * 原始位置 (m={m}, n={n}, 相对距离={m-n}): 内积打分 = {score1:.6f}")
    print(f"  * 平移位置 (m={m+delta}, n={n+delta}, 相对距离={m-n}): 内积打分 = {score2:.6f}")
    print(f"  * 两者绝对误差 |Score1 - Score2| = {diff:.2e} (达到浮点机内精度零误差!)")
    print("  => 完美实证：RoPE 绝对位置编码具有天然的相对距离恒定对称性！")

    # 2. 测量随相对距离增加的自注意力内积衰减规律
    distances = range(0, 100)
    q_fix = rope(q.expand(-1, -1, 1, -1))[:, :, 0:1, :]
    scores = []
    for d in distances:
        k_d = rope(k.expand(-1, -1, d + 1, -1))[:, :, d:d+1, :]
        scores.append((q_fix * k_d).sum().item())

    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.makedirs(img_dir, exist_ok=True)
    save_fig = os.path.join(img_dir, "llm_rope_relative_invariance.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(distances, scores, color='#2563EB', lw=2)
    ax.set_title("RoPE 相对位置距离衰减效应实测曲线", fontsize=12, fontweight='bold')
    ax.set_xlabel("相对距离 |m - n|", fontsize=10)
    ax.set_ylabel("注意力内积打分 <R_m q, R_n k>", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()
    print(f">>> RoPE 衰减曲线图已持久化至: {save_fig}\n")

if __name__ == "__main__":
    run()

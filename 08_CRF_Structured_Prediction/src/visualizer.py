# -*- coding: utf-8 -*-
"""
Visualizer module for CRF Structured Prediction
1. 学习曲线 (Training Loss Curve)
2. 评测指标全景对比柱状图 (Accuracy, F1, Sequence EM, Invalid Rate)
3. CRF 状态转移矩阵热力图 (Transition Matrix Heatmap)
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体与负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(losses_softmax, losses_crf, save_path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    epochs = range(1, len(losses_softmax) + 1)
    ax.plot(epochs, losses_softmax, label="BiLSTM-Softmax (局部交叉熵)", color="#EF4444", lw=2, marker='o')
    ax.plot(epochs, losses_crf, label="BiLSTM-CRF (全局负对数似然)", color="#3B82F6", lw=2, marker='s')
    ax.set_title("训练损失收敛曲线对比 (Training Loss Convergence)", fontsize=13, fontweight='bold')
    ax.set_xlabel("训练轮次 (Epoch)", fontsize=11)
    ax.set_ylabel("损失值 (Loss)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_benchmark_metrics(metrics, save_path):
    """
    metrics: dict of {
        'Model': ['HMM', 'BiLSTM-Softmax', 'BiLSTM-CRF'],
        'Token_Acc': [...],
        'Sequence_EM': [...],
        'F1_Score': [...],
        'Invalid_Rate': [...]
    }
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    models = metrics['Model']
    colors = ['#9CA3AF', '#EF4444', '#10B981']

    # 1. Token Accuracy
    axes[0].bar(models, metrics['Token_Acc'], color=colors, width=0.5)
    axes[0].set_title("Token 级单点准确率", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(metrics['Token_Acc']):
        axes[0].text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

    # 2. Sequence EM (整句全对率)
    axes[1].bar(models, metrics['Sequence_EM'], color=colors, width=0.5)
    axes[1].set_title("整句全对率 (Sequence Exact Match)", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(metrics['Sequence_EM']):
        axes[1].text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

    # 3. F1 Score
    axes[2].bar(models, metrics['F1_Score'], color=colors, width=0.5)
    axes[2].set_title("实体级加权 F1 评测分", fontsize=11, fontweight='bold')
    axes[2].set_ylim(0, 1.05)
    for i, v in enumerate(metrics['F1_Score']):
        axes[2].text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

    # 4. Invalid Transition Rate (非法语法跳变率 - 越低越好)
    axes[3].bar(models, metrics['Invalid_Rate'], color=['#6B7280', '#DC2626', '#059669'], width=0.5)
    axes[3].set_title("非法语法跳变率 (越低越好)", fontsize=11, fontweight='bold')
    axes[3].set_ylim(0, max(max(metrics['Invalid_Rate']) * 1.3, 0.15))
    for i, v in enumerate(metrics['Invalid_Rate']):
        axes[3].text(i, v + 0.005, f"{v*100:.2f}%", ha='center', fontweight='bold', color='red' if v > 0.01 else 'green')

    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("三大模型在命名实体识别 (NER) 任务上的横向基准实测", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_transition_matrix(trans_matrix, tag_names, save_path):
    """
    绘制 CRF 学习到的转移矩阵热力图
    """
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    im = ax.imshow(trans_matrix, cmap="RdYlGn", interpolation="nearest")

    ax.set_xticks(range(len(tag_names)))
    ax.set_yticks(range(len(tag_names)))
    ax.set_xticklabels(tag_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tag_names, fontsize=10)

    # 在格子中绘制数值
    for i in range(len(tag_names)):
        for j in range(len(tag_names)):
            val = trans_matrix[i, j]
            color = "white" if abs(val) > np.max(np.abs(trans_matrix)) * 0.6 else "black"
            text_val = f"{val:.1f}" if abs(val) < 500 else "-INF"
            ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=8, fontweight='bold')

    ax.set_title("CRF 学习到的状态转移矩阵 A (Trans[i, j]: 从标签 j 转移到标签 i)", fontsize=12, fontweight='bold')
    ax.set_xlabel("前一步标签 (From Tag j)", fontsize=11)
    ax.set_ylabel("当前步标签 (To Tag i)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

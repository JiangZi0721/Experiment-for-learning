# -*- coding: utf-8 -*-
"""
Visualizer module for MoE Sparse Routing:
1. 路由塌缩对比柱状图 (Routing Collapse vs Balanced Distribution)
2. 架构收敛曲线 (Dense vs Standard MoE vs DeepSeekMoE)
3. 专家专业化分工热力图 (Domain vs Expert Activation Heatmap)
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_routing_collapse(dist_no_aux, dist_with_aux, save_path):
    """
    绘制有无辅助损失下的专家负载对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    num_exp = len(dist_no_aux)
    experts = [f"Expert {i}" for i in range(num_exp)]

    # 1. 无辅助损失 (塌缩发生)
    axes[0].bar(experts, dist_no_aux * 100, color='#DC2626', width=0.5)
    axes[0].set_title("[消融对比] 无辅助损失 (易发生路由塌缩)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Token 负载占比 (%)", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].axhline(y=100.0 / num_exp, color='blue', linestyle='--', label=f"理想均分基准 ({100.0/num_exp:.1f}%)")
    for i, v in enumerate(dist_no_aux):
        axes[0].text(i, v * 100 + 2, f"{v*100:.1f}%", ha='center', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 2. 开启辅助损失 (负载均衡)
    axes[1].bar(experts, dist_with_aux * 100, color='#10B981', width=0.5)
    axes[1].set_title("[基准对照] 开启辅助损失 (负载健康均衡)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Token 负载占比 (%)", fontsize=10)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(y=100.0 / num_exp, color='blue', linestyle='--', label=f"理想均分基准 ({100.0/num_exp:.1f}%)")
    for i, v in enumerate(dist_with_aux):
        axes[1].text(i, v * 100 + 2, f"{v*100:.1f}%", ha='center', fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("MoE 路由塌缩机理与辅助损失均衡实证 (Routing Collapse Ablation)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_comparison(losses_dense, losses_moe, losses_deepseek, save_path):
    """
    绘制 Dense vs Standard MoE vs DeepSeekMoE 训练收敛曲线
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    epochs = range(1, len(losses_dense) + 1)
    ax.plot(epochs, losses_dense, label="Dense FFN (参数量 P, 算力 P)", color="#6B7280", lw=2, linestyle="--")
    ax.plot(epochs, losses_moe, label="Standard MoE (参数量 4P, 算力 P)", color="#3B82F6", lw=2, marker='o')
    ax.plot(epochs, losses_deepseek, label="DeepSeekMoE 细粒度+共享 (参数量 4P, 算力 P)", color="#10B981", lw=2.5, marker='s')

    ax.set_title("相同算力预算 (FLOPs) 下三大架构收敛能力对比", fontsize=12, fontweight='bold')
    ax.set_xlabel("训练轮次 (Epoch)", fontsize=11)
    ax.set_ylabel("任务交叉熵损失 (Cross Entropy Loss)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_expert_specialization_heatmap(specialization_matrix, domains, experts, save_path):
    """
    绘制领域 Token 与专家的激活对齐热力图
    """
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    im = ax.imshow(specialization_matrix * 100, cmap="YlGnBu", interpolation="nearest")

    ax.set_xticks(range(len(experts)))
    ax.set_yticks(range(len(domains)))
    ax.set_xticklabels(experts, fontsize=10)
    ax.set_yticklabels(domains, fontsize=10)

    for i in range(len(domains)):
        for j in range(len(experts)):
            val = specialization_matrix[i, j] * 100
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight='bold', fontsize=9)

    ax.set_title("各领域语义 Token 在不同专家上的路由分发热力图 (Specialization Matrix)", fontsize=12, fontweight='bold')
    ax.set_xlabel("路由专家 (Routed Experts)", fontsize=11)
    ax.set_ylabel("输入 Token 领域属性 (Token Semantic Domain)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="分发占比 (%)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

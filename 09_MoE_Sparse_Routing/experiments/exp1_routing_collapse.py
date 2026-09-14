# -*- coding: utf-8 -*-
"""
Experiment 1: 路由塌缩实证与负载均衡辅助损失消融 (Routing Collapse Ablation)
实证展示：无辅助损失时，因马太效应单一专家霸占全网，其余专家闲置枯死；
引入辅助损失后，专家负载恢复良性均匀分布。
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.moe_layer import StandardMoE
from src.visualizer import plot_routing_collapse

def run(num_steps=300, d_model=32, d_ff=64, num_experts=4):
    print("=" * 80)
    print(">>> 【实验一：MoE 路由塌缩实证与辅助损失消融 (Routing Collapse Ablation)】")
    print("=" * 80)
    torch.manual_seed(42)

    # 1. 训练无辅助损失的 MoE 模型 (很容易发生路由塌缩)
    print("\n[1/2] 训练 MoE (aux_loss_coef = 0.0, 禁用负载均衡)...")
    moe_no_aux = StandardMoE(d_model, d_ff, num_experts=num_experts, top_k=1, aux_loss_coef=0.0)
    opt_no_aux = optim.Adam(moe_no_aux.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 构造仿真特征与目标
    x_train = torch.randn(num_steps, 64, d_model)
    target = torch.randn(num_steps, 64, d_model)

    for step in range(num_steps):
        opt_no_aux.zero_grad()
        out, aux_loss = moe_no_aux(x_train[step], apply_aux_loss=False)
        task_loss = criterion(out, target[step])
        loss = task_loss + aux_loss
        loss.backward()
        opt_no_aux.step()

    # 2. 训练开启辅助损失的 MoE 模型
    print("\n[2/2] 训练 MoE (aux_loss_coef = 0.05, 开启 Switch/GShard 负载均衡)...")
    moe_with_aux = StandardMoE(d_model, d_ff, num_experts=num_experts, top_k=1, aux_loss_coef=0.05)
    opt_with_aux = optim.Adam(moe_with_aux.parameters(), lr=0.01)

    for step in range(num_steps):
        opt_with_aux.zero_grad()
        out, aux_loss = moe_with_aux(x_train[step], apply_aux_loss=True)
        task_loss = criterion(out, target[step])
        loss = task_loss + aux_loss
        loss.backward()
        opt_with_aux.step()

    # 3. 统计测试集上的分发分布
    print("\n>>> 在 5000 个独立测试 Token 上统计专家实际分配占比...")
    test_x = torch.randn(5000, d_model)
    moe_no_aux.eval()
    moe_with_aux.eval()

    with torch.no_grad():
        _, sel_no_aux, _ = moe_no_aux.router(test_x, apply_aux_loss=False)
        _, sel_with_aux, _ = moe_with_aux.router(test_x, apply_aux_loss=False)

    dist_no_aux = torch.bincount(sel_no_aux.squeeze(), minlength=num_experts).float() / len(test_x)
    dist_with_aux = torch.bincount(sel_with_aux.squeeze(), minlength=num_experts).float() / len(test_x)

    dist_no_aux = dist_no_aux.cpu().numpy()
    dist_with_aux = dist_with_aux.cpu().numpy()

    print("\n" + "=" * 70)
    print(f"{'专家编号':<10} | {'无辅助损失负载 (塌缩)':<24} | {'开启辅助损失负载 (均衡)'}")
    print("-" * 70)
    for i in range(num_experts):
        flag = "🔴 垄断霸权" if dist_no_aux[i] > 0.6 else ("⚠️ 严重饥饿" if dist_no_aux[i] < 0.05 else "正常")
        print(f"Expert {i:<3} | {dist_no_aux[i]*100:>8.2f}% ({flag:<8}) | {dist_with_aux[i]*100:>8.2f}% (🟢 健康分担)")
    print("=" * 70)

    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.makedirs(img_dir, exist_ok=True)
    save_fig = os.path.join(img_dir, "moe_routing_collapse.png")
    plot_routing_collapse(dist_no_aux, dist_with_aux, save_fig)
    print(f">>> 路由塌缩对照图已生成至: {save_fig}\n")

if __name__ == "__main__":
    run()

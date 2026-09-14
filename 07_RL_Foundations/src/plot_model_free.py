"""
Plotting module for Model-Free algorithms: TD(0), SARSA, and Q-Learning.
Generates 3 dedicated high-resolution figures showing the complete workflows and results.
"""
import os
import sys
import pathlib

# Sandbox pathlib fix
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.grid_world import GridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_SYMBOLS

# Font setup for Chinese and clean math
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def smooth_curve(values: List[float], window: int = 50) -> np.ndarray:
    """Moving average smoothing for reward/step curves."""
    if len(values) < window:
        return np.array(values)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')

def draw_grid_cell_annotations(ax, env, V_dict: Dict, pi_dict: Optional[Dict] = None):
    """Draws grid values, markers, and policy arrows on an axis for ComplexGridWorld."""
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            val = V_dict.get(s, 0.0)
            
            if env.is_terminal(s):
                if s == (0, 3):
                    term_label = f"SUB-GOAL\n[+3.0]\n{val:.2f}"
                    fc = "#C8E6C9"
                elif s == (3, 3):
                    term_label = f"MAIN GOAL\n[+10.0]\n{val:.2f}"
                    fc = "#A5D6A7"
                else:
                    term_label = f"GOAL\n{val:.2f}"
                    fc = "lightgreen"
                ax.text(c, r, term_label, ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color='darkgreen',
                        bbox=dict(boxstyle="round,pad=0.2", fc=fc, alpha=0.85))
                continue
            elif s in getattr(env, 'hazard_states', {}) or s in getattr(env, 'trap_states', {}):
                if s == (1, 2):
                    trap_label = f"LAVA (-10)\n{val:.2f}"
                elif s == (2, 3):
                    trap_label = f"SPIKES (-6)\n{val:.2f}"
                elif s == (2, 1):
                    trap_label = f"SWAMP (-3)\n{val:.2f}"
                else:
                    trap_label = f"TRAP\n{val:.2f}"
                ax.text(c, r, trap_label, ha='center', va='center',
                        fontsize=7.0, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#FFCDD2", alpha=0.85))
            elif s == (0, 0):
                ax.text(c, r + 0.28, f"START\n{val:.2f}", ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color='#1A237E',
                        bbox=dict(boxstyle="round,pad=0.15", fc="#E8EAF6", alpha=0.7))
            else:
                ax.text(c, r + 0.28, f"{val:.2f}", ha='center', va='center',
                        fontsize=8, fontweight='bold', color='black')
                
            if pi_dict is not None and s in pi_dict:
                action_probs = pi_dict[s]
                arrow_len = 0.22
                for a, p in action_probs.items():
                    if p > 0.05:
                        dx, dy = 0.0, 0.0
                        if a == UP: dy = -arrow_len * p
                        elif a == DOWN: dy = arrow_len * p
                        elif a == LEFT: dx = -arrow_len * p
                        elif a == RIGHT: dx = arrow_len * p
                        
                        color = 'blue' if p > 0.4 else 'gray'
                        lw = 2.0 if p > 0.4 else 1.0
                        ax.arrow(c, r - 0.05, dx, dy, head_width=0.07, head_length=0.07,
                                 fc=color, ec=color, length_includes_head=True, lw=lw, alpha=0.9)

def plot_td0_workflow(td_result: Dict, true_V: Dict, env: GridWorld, save_path: str):
    """Figure 1: TD(0) Complete Workflow & Prediction Verification."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. TD(0) Learned Value Heatmap
    ax1 = axes[0]
    v_td_mat = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            v_td_mat[r, c] = td_result["V"][(r, c)]
    im1 = ax1.imshow(v_td_mat, cmap='Spectral_r', interpolation='nearest')
    ax1.set_title("TD(0) 学习得到的状态价值 V(s)\n(2500 回合随机策略采样)", fontsize=11, fontweight='bold')
    draw_grid_cell_annotations(ax1, env, td_result["V"])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. DP Ground Truth Value Heatmap
    ax2 = axes[1]
    v_dp_mat = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            v_dp_mat[r, c] = true_V[(r, c)]
    im2 = ax2.imshow(v_dp_mat, cmap='Spectral_r', interpolation='nearest')
    ax2.set_title("动态规划 (DP) 理论真值 V_true(s)\n(贝尔曼期望方程不动点解析值)", fontsize=11, fontweight='bold')
    draw_grid_cell_annotations(ax2, env, true_V)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. RMSE Learning Curve
    ax3 = axes[2]
    rmse = td_result["rmse_history"]
    smoothed_rmse = smooth_curve(rmse, window=50)
    ax3.plot(rmse, alpha=0.25, color='royalblue', label='单回合原始 RMSE')
    ax3.plot(range(25, 25 + len(smoothed_rmse)), smoothed_rmse, color='darkblue', lw=2, label='滑动平均 (窗口=50)')
    ax3.set_title("TD(0) 价值逼近真值的均方根误差 (RMSE)\n$V(S) \leftarrow V(S) + \\alpha [R + \gamma V(S') - V(S)]$", fontsize=11, fontweight='bold')
    ax3.set_xlabel("训练回合数 (Episodes)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("RMSE = $\sqrt{\mathbb{E}[(V_{TD} - V_{true})^2]}$", fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9)
    
    for ax in [ax1, ax2]:
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1.2)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ TD(0) 流程图已保存至: {save_path}")

def plot_sarsa_workflow(sarsa_result: Dict, env: GridWorld, save_path: str):
    """Figure 2: SARSA Complete Workflow & On-Policy Control."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Extract V(s) and greedy policy from Q
    V_sarsa = {}
    pi_sarsa = {}
    for s in env.states:
        q_s = sarsa_result["Q"][s]
        max_q = max(q_s.values())
        V_sarsa[s] = max_q
        best_a = [a for a, q in q_s.items() if np.isclose(q, max_q, atol=1e-6)]
        pi_sarsa[s] = {a: (1.0 / len(best_a) if a in best_a else 0.0) for a in ACTIONS}
        
    # 1. State Value & Policy Arrows
    ax1 = axes[0]
    v_mat = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            v_mat[r, c] = V_sarsa[(r, c)]
    im1 = ax1.imshow(v_mat, cmap='Spectral_r', interpolation='nearest')
    ax1.set_title("SARSA 动作价值 max_a Q(s,a) 与策略\n(同策略 On-Policy 学习)", fontsize=11, fontweight='bold')
    draw_grid_cell_annotations(ax1, env, V_sarsa, pi_sarsa)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Episode Return Curve
    ax2 = axes[1]
    returns = sarsa_result["reward_history"]
    smoothed_returns = smooth_curve(returns, window=40)
    ax2.plot(returns, alpha=0.2, color='coral')
    ax2.plot(range(20, 20 + len(smoothed_returns)), smoothed_returns, color='orangered', lw=2, label='滑动回报 (窗口=40)')
    ax2.set_title("SARSA 回合累积回报变化 (Episode Return)\n更新目标: $R + \gamma Q(S', A')$ (包含探索动作)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("训练回合数 (Episodes)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("回合总得分 (Total Reward)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)
    
    # 3. Episode Steps to Goal
    ax3 = axes[2]
    steps = sarsa_result["steps_history"]
    smoothed_steps = smooth_curve(steps, window=40)
    ax3.plot(steps, alpha=0.2, color='mediumpurple')
    ax3.plot(range(20, 20 + len(smoothed_steps)), smoothed_steps, color='indigo', lw=2, label='滑动步数 (窗口=40)')
    ax3.set_title("到达终点所需步数 (Steps to Goal)\n从盲目游走到收敛至极速寻路", fontsize=11, fontweight='bold')
    ax3.set_xlabel("训练回合数 (Episodes)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("步数 (Steps)", fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9)
    
    ax1.set_xticks(range(env.cols))
    ax1.set_yticks(range(env.rows))
    ax1.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=1.2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ SARSA 流程图已保存至: {save_path}")

def plot_qlearning_workflow(ql_result: Dict, sarsa_result: Dict, env: GridWorld, save_path: str):
    """Figure 3: Q-Learning Complete Workflow & Off-Policy Comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Extract V*(s) and greedy policy from Q
    V_ql = {}
    pi_ql = {}
    for s in env.states:
        q_s = ql_result["Q"][s]
        max_q = max(q_s.values())
        V_ql[s] = max_q
        best_a = [a for a, q in q_s.items() if np.isclose(q, max_q, atol=1e-6)]
        pi_ql[s] = {a: (1.0 / len(best_a) if a in best_a else 0.0) for a in ACTIONS}
        
    # 1. State Value & Optimal Policy Arrows
    ax1 = axes[0]
    v_mat = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            v_mat[r, c] = V_ql[(r, c)]
    im1 = ax1.imshow(v_mat, cmap='Spectral_r', interpolation='nearest')
    ax1.set_title("Q-Learning 最优价值 max_a Q(s,a) 与策略\n(异策略 Off-Policy 最优寻径)", fontsize=11, fontweight='bold')
    draw_grid_cell_annotations(ax1, env, V_ql, pi_ql)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Performance Comparison: SARSA vs Q-Learning
    ax2 = axes[1]
    ql_returns = ql_result["reward_history"]
    sarsa_returns = sarsa_result["reward_history"]
    sm_ql = smooth_curve(ql_returns, window=50)
    sm_sarsa = smooth_curve(sarsa_returns, window=50)
    
    ax2.plot(range(25, 25 + len(sm_ql)), sm_ql, color='forestgreen', lw=2.2, label='Q-Learning (异策略)')
    ax2.plot(range(25, 25 + len(sm_sarsa)), sm_sarsa, color='orangered', lw=2.0, linestyle='--', label='SARSA (同策略)')
    ax2.set_title("训练表现对比: SARSA vs Q-Learning\n(更新目标: $\max_{a'} Q$ vs 实际 $Q(S', A')$)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("训练回合数 (Episodes)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("滑动平均回报 (Smoothed Return)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    # 3. Action Values at Critical Trap-Neighbor State (1, 1)
    ax3 = axes[2]
    q_11 = ql_result["Q"][(1, 1)]
    action_labels = ["UP (↑)", "DOWN (↓)", "LEFT (←)", "RIGHT (→)\n[进入陷阱]"]
    values = [q_11[UP], q_11[DOWN], q_11[LEFT], q_11[RIGHT]]
    colors = ['skyblue', 'forestgreen', 'lightblue', 'crimson']
    
    bars = ax3.bar(action_labels, values, color=colors, edgecolor='black', lw=1.2)
    ax3.set_title("关键状态 (1, 1) 动作价值 Q((1,1), a) 剖析\n(Q-Learning 自动确立向下优势，远离右侧陷阱)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("动作价值 Q((1,1), a)", fontsize=10, fontweight='bold')
    ax3.grid(True, axis='y', linestyle="--", alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, -14 if height < 0 else 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                     
    ax1.set_xticks(range(env.cols))
    ax1.set_yticks(range(env.rows))
    ax1.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=1.2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Q-Learning 流程图已保存至: {save_path}")

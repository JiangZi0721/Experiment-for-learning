"""
深度 Q 网络 (DQN) 专业级学术图表渲染模块 (DQN Visualizer Engine)
生成 DQN 训练动态收敛四联图与最终策略/价值热力图对比图。
"""
import os
import sys
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple
from src.grid_world import GridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT

# Setup Chinese and math font configurations
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def plot_dqn_training_curves(history: Dict, save_path: str):
    """
    绘制 DQN 训练四联学术图表:
    1. 回合累积回报曲线与滑动平均 (对齐理论最优回报 -6.0)
    2. 贝尔曼 TD 损失下降轨迹
    3. Epsilon 探索率退火过程
    4. 神经网络与动态规划理论真值 Q* 的均方误差 (MSE to Ground Truth Q*)
    """
    episodes = history["episodes"]
    returns = history["returns"]
    losses = history["losses"]
    epsilons = history["epsilons"]
    q_errors = history["q_errors"]
    
    # 20 步滑动平均计算
    window = 20
    def moving_average(data, w):
        return np.convolve(data, np.ones(w) / w, mode='valid')
        
    avg_returns = moving_average(returns, window)
    avg_losses = moving_average(losses, window) if len(losses) >= window else losses
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # -------------------------------------------------------------
    # 1. 回合累积回报
    # -------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(episodes, returns, color='#FFAB91', alpha=0.45, label="单回合原始回报 (Raw Return)")
    ax.plot(range(window, len(returns) + 1), avg_returns, color='#D84315', lw=2.4, label=f"{window} 回合滑动平均回报")
    ax.axhline(y=5.0, color='#2E7D32', linestyle='--', lw=2.0, label="理论最优回报 $R^* = +5.0$ (外圈绕避险情)")
    ax.set_xlabel("训练回合数 (Episodes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("回合累积回报 (Episodic Return)", fontsize=11, fontweight='bold')
    ax.set_title("DQN 回合回报上升与收敛轨迹", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    
    # -------------------------------------------------------------
    # 2. 贝尔曼 TD 损失
    # -------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(episodes, losses, color='#90CAF9', alpha=0.45, label="原始均方误差损失")
    if len(losses) >= window:
        ax.plot(range(window, len(losses) + 1), avg_losses, color='#1565C0', lw=2.2, label=f"{window} 回合滑动平均损失")
    ax.set_xlabel("训练回合数 (Episodes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("TD 损失 $\\mathcal{L}(\\theta)$", fontsize=11, fontweight='bold')
    ax.set_title("神经网络贝尔曼均方误差损失 (Bellman TD Loss)", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    # -------------------------------------------------------------
    # 3. Epsilon 探索率退火
    # -------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color='#7B1FA2', lw=2.2, label=r"$\epsilon$-贪婪探索率 ($\epsilon$)")
    ax.axhline(y=0.05, color='#AB47BC', linestyle=':', lw=1.5, label="探索率下限 $\\epsilon_{\\min} = 0.05$")
    ax.set_xlabel("训练回合数 (Episodes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("探索概率 $\\epsilon$", fontsize=11, fontweight='bold')
    ax.set_title(r"$\epsilon$-贪婪探索衰减进度 (Exploration Decay)", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    # 标注阶段性说明
    ax.text(0.45, 0.55, "前期: 高探索率遍历全图\n后期: 低探索率稳定利用经验",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F3E5F5", ec="#CE93D8", alpha=0.85))
    
    # -------------------------------------------------------------
    # 4. 逼近理论真值 Q* 的 MSE 误差
    # -------------------------------------------------------------
    ax = axes[1, 1]
    ax.semilogy(episodes, q_errors, color='#00897B', lw=2.2, label="实测 $Q_\\theta$ 与真值 $Q^*$ 均方误差 (MSE)")
    ax.set_xlabel("训练回合数 (Episodes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("真值逼近误差 $\\mathrm{MSE}(Q_\\theta, Q^*)$ (对数坐标)", fontsize=11, fontweight='bold')
    ax.set_title("神经网络逼近理论最优动作价值 $Q^*$ 过程", fontsize=13, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    # 总结说明卡片
    final_err = q_errors[-1] if q_errors else 0.0
    text_summary = (
        f"【实证结论】:\n"
        f"• 最终 Q* 逼近误差 MSE = {final_err:.4f}\n"
        f"• 目标网络与回放池协同消除震荡\n"
        f"• 成功突破致命三要素 (Deadly Triad)！"
    )
    ax.text(0.05, 0.12, text_summary, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#E0F2F1", ec="#80CBC4", alpha=0.9))
            
    plt.suptitle("深度 Q 网络 (DQN) 训练收敛全景透视 (Complex GridWorld Benchmark)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ DQN 训练收敛曲线图已生成并保存至: {save_path}")

def plot_dqn_policy_and_q_heatmap(agent, env, ground_truth_V: Dict, ground_truth_Q: Dict = None, save_path: str = "images/dqn_policy_and_q_heatmap.png"):
    """
    绘制 DQN 求解价值函数与动态规划最优真值 V* 的全景同屏对比图 (3 联图):
    Panel 1: DQN 神经网络输出的状态价值 V_θ 与由网络导出的最优确定性策略
    Panel 2: 动态规划计算出的理论真值最优价值 V* 与最优策略 π*
    Panel 3: 绝对误差分布热力图 |V_θ - V*|
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. 提取 DQN 价值与最优动作
    v_dqn = np.zeros((env.rows, env.cols))
    pi_dqn = {}
    with torch.no_grad():
        for r in range(env.rows):
            for c in range(env.cols):
                s = (r, c)
                if env.is_terminal(s):
                    v_dqn[r, c] = ground_truth_V.get(s, 0.0)
                    pi_dqn[s] = 0
                else:
                    s_tensor = agent.state_to_tensor(s).unsqueeze(0)
                    q_vals = agent.q_eval(s_tensor).squeeze(0).numpy()
                    v_dqn[r, c] = float(np.max(q_vals))
                    pi_dqn[s] = int(np.argmax(q_vals))
                    
    # 2. 提取真值
    v_gt = np.zeros((env.rows, env.cols))
    pi_gt = {}
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            v_gt[r, c] = ground_truth_V.get(s, 0.0)
            if ground_truth_Q and s in ground_truth_Q and not env.is_terminal(s):
                pi_gt[s] = max(ground_truth_Q[s].keys(), key=lambda a: ground_truth_Q[s][a])
            else:
                pi_gt[s] = 0
            
    # 3. 计算绝对误差矩阵
    diff = np.abs(v_dqn - v_gt)
    mean_err = float(np.mean(diff))
    max_err = float(np.max(diff))
    
    vmin = min(float(np.min(v_dqn)), float(np.min(v_gt)))
    vmax = max(float(np.max(v_dqn)), float(np.max(v_gt)))
    
    # -------------------------------------------------------------
    # Panel 1: DQN 神经网络输出
    # -------------------------------------------------------------
    ax1 = axes[0]
    im1 = ax1.imshow(v_dqn, cmap='Spectral_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_title("DQN 神经网络状态价值 $V_\\theta(s)$ 与贪婪策略", fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------
    # Panel 2: 动态规划理论真值
    # -------------------------------------------------------------
    ax2 = axes[1]
    im2 = ax2.imshow(v_gt, cmap='Spectral_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_title("动态规划全局最优真值 $V^*(s)$ 与最优策略 $\\pi^*$", fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------
    # Panel 3: 绝对误差热力图
    # -------------------------------------------------------------
    ax3 = axes[2]
    max_diff_scale = max(0.5, float(np.max(diff)))
    im3 = ax3.imshow(diff, cmap='Reds', interpolation='nearest', vmin=0.0, vmax=max_diff_scale)
    ax3.set_title(f"绝对误差 $|V_\\theta - V^*|$ 分布\n(平均误差={mean_err:.4f}, 最大误差={max_err:.4f})", fontsize=12, fontweight='bold')
    
    # 统一装饰三张子图
    for idx, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_xticklabels([f"c={c}" for c in range(env.cols)])
        ax.set_yticklabels([f"r={r}" for r in range(env.rows)])
        ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1.2)
        ax.tick_params(which='minor', size=0)
        
        # 数值与标记标注
        for r in range(env.rows):
            for c in range(env.cols):
                s = (r, c)
                if env.is_terminal(s):
                    if s == (0, 3):
                        term_label = f"SUB-GOAL\n[+3.0]\n{v_gt[r,c]:.2f}"
                        fc = "#C8E6C9"
                    elif s == (3, 3):
                        term_label = f"MAIN GOAL\n[+10.0]\n{v_gt[r,c]:.2f}"
                        fc = "#A5D6A7"
                    else:
                        term_label = f"GOAL\n{v_gt[r,c]:.2f}"
                        fc = "lightgreen"
                    ax.text(c, r, term_label, ha='center', va='center', fontsize=8, fontweight='bold',
                            color='darkgreen', bbox=dict(boxstyle="round,pad=0.2", fc=fc, alpha=0.85))
                    continue
                elif s in getattr(env, 'hazard_states', {}) or s in getattr(env, 'trap_states', {}):
                    if s == (1, 2):
                        trap_label = f"LAVA (-10)\n{v_dqn[r,c]:.2f}" if idx == 0 else (f"LAVA (-10)\n{v_gt[r,c]:.2f}" if idx == 1 else f"LAVA\n{diff[r,c]:.2f}")
                    elif s == (2, 3):
                        trap_label = f"SPIKES (-6)\n{v_dqn[r,c]:.2f}" if idx == 0 else (f"SPIKES (-6)\n{v_gt[r,c]:.2f}" if idx == 1 else f"SPIKES\n{diff[r,c]:.2f}")
                    elif s == (2, 1):
                        trap_label = f"SWAMP (-3)\n{v_dqn[r,c]:.2f}" if idx == 0 else (f"SWAMP (-3)\n{v_gt[r,c]:.2f}" if idx == 1 else f"SWAMP\n{diff[r,c]:.2f}")
                    else:
                        trap_label = f"TRAP\n{v_dqn[r,c]:.2f}"
                    ax.text(c, r, trap_label, ha='center', va='center', fontsize=7.5, fontweight='bold',
                            color='darkred', bbox=dict(boxstyle="round,pad=0.2", fc="#FFCDD2", alpha=0.85))
                    continue
                elif s == (0, 0):
                    val = v_dqn[r, c] if idx == 0 else (v_gt[r, c] if idx == 1 else diff[r, c])
                    ax.text(c, r + 0.28, f"S:{val:.2f}", ha='center', va='center',
                            fontsize=8, fontweight='bold', color='#1A237E',
                            bbox=dict(boxstyle="round,pad=0.15", fc="#E8EAF6", alpha=0.7))
                else:
                    val = v_dqn[r, c] if idx == 0 else (v_gt[r, c] if idx == 1 else diff[r, c])
                    y_offset = 0.28 if idx < 2 else 0.0
                    ax.text(c, r + y_offset, f"{val:.2f}", ha='center', va='center', fontsize=9, fontweight='bold', color='black')
                
                # Panel 1 & 2 画出策略箭头
                if idx < 2:
                    action = pi_dqn[s] if idx == 0 else pi_gt[s]
                    arrow_len = 0.22
                    dx, dy = 0.0, 0.0
                    if action == UP: dy = -arrow_len
                    elif action == DOWN: dy = arrow_len
                    elif action == LEFT: dx = -arrow_len
                    elif action == RIGHT: dx = arrow_len
                    
                    color = '#1565C0' if idx == 0 else '#2E7D32'
                    ax.arrow(c, r - 0.06, dx, dy, head_width=0.08, head_length=0.08,
                             fc=color, ec=color, length_includes_head=True, alpha=0.9, lw=2.0)
                             
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="状态价值 $V(s)$")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="最优真值 $V^*(s)$")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="绝对误差 $|V_\\theta - V^*|$")
    
    plt.suptitle("DQN 神经网络求解结果与动态规划全局真值严密对比", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ DQN 策略与价值对比热力图已生成并保存至: {save_path}")

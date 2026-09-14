"""
Gamma Comparison Experiment:
Investigates gamma in [0.7, 0.9, 0.99, 1.0] under Policy Iteration & TD(0).
Compares convergence speed, value scales, and policy outcomes.
"""
import os
import sys
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Sandbox pathlib fix
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.grid_world import GridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_SYMBOLS
from src.policy_iteration import PolicyIteration

# Setup font support
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main():
    print("="*75)
    print("  🔬 折扣因子 gamma 核心对比实验：gamma in [0.7, 0.9, 0.99, 1.0]")
    print("="*75)
    
    gammas = [0.7, 0.9, 0.99, 1.0]
    results = {}
    
    for g in gammas:
        env = GridWorld(rows=4, cols=4, goal_states=[(3, 3)], trap_states={(1, 2): -5.0}, step_cost=-1.0, gamma=g)
        solver = PolicyIteration(env, theta=1e-4)
        history = solver.run(verbose_states=[])
        
        # Total evaluation sweeps across all policy iterations
        total_eval_sweeps = sum(h["sweeps"] for h in history)
        final_V = history[-1]["V"]
        final_pi = history[-1]["pi"]
        
        results[g] = {
            "history": history,
            "total_sweeps": total_eval_sweeps,
            "iter_0_sweeps": history[0]["sweeps"],
            "iter_0_deltas": history[0]["deltas"],
            "outer_iters": len(history),
            "final_V": final_V,
            "final_pi": final_pi
        }
        
        print(f"\n>>> [gamma = {g:<4}] 实验统计:")
        print(f"  - 外层策略迭代轮数: {len(history)}")
        print(f"  - 初始随机策略评估耗费扫描数: {history[0]['sweeps']} sweeps")
        print(f"  - 全流程累计评估扫描总数: {total_eval_sweeps} sweeps")
        print(f"  - 起点 (0, 0) 最终最优价值 V*(0,0): {final_V[(0, 0)]:.4f}")
        print(f"  - 陷阱 (1, 2) 最终最优价值 V*(1,2): {final_V[(1, 2)]:.4f}")
        print(f"  - 终点邻近 (2, 3) 最优价值 V*(2,3): {final_V[(2, 3)]:.4f}")
        
    # Generate Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # Subplot 1: Convergence Speed of Evaluation k=0 across gammas
    ax1 = axes[0]
    for g in gammas:
        deltas = results[g]["iter_0_deltas"]
        ax1.semilogy(range(1, len(deltas) + 1), deltas, lw=1.8, label=f"$\\gamma={g}$ ({len(deltas)} sweeps)")
    ax1.set_title("随机策略评估收敛速度对比 (k=0)\n(验证 $\\gamma$-压缩映射理论)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("评估扫描步数 (Sweeps)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("最大误差 $\\Delta$ (对数坐标)", fontsize=10, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Heatmap V*(s) for gamma = 1.0 (Physical Steps)
    ax2 = axes[1]
    env = GridWorld(rows=4, cols=4, goal_states=[(3, 3)], trap_states={(1, 2): -5.0}, step_cost=-1.0, gamma=1.0)
    v_mat_g1 = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            v_mat_g1[r, c] = results[1.0]["final_V"][(r, c)]
    im2 = ax2.imshow(v_mat_g1, cmap='Spectral_r', interpolation='nearest')
    ax2.set_title("$\\gamma = 1.0$ 最优状态价值 V*(s)\n(数值严格等于到达终点最短步数的负数)", fontsize=11, fontweight='bold')
    for r in range(4):
        for c in range(4):
            val = results[1.0]["final_V"][(r, c)]
            if (r, c) == (3, 3):
                ax2.text(c, r, f"GOAL\n{val:.1f}", ha='center', va='center', fontsize=9, fontweight='bold', color='darkgreen')
            elif (r, c) == (1, 2):
                ax2.text(c, r, f"TRAP\n{val:.1f}", ha='center', va='center', fontsize=9, fontweight='bold', color='darkred')
            else:
                ax2.text(c, r, f"{val:.1f}", ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Subplot 3: Value Decay vs Manhattan Distance to Goal
    ax3 = axes[2]
    # Path coordinates from (0,0) down to (3,3): (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3)
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]
    distances = [6, 5, 4, 3, 2, 1, 0]
    
    for g in gammas:
        path_vals = [results[g]["final_V"][s] for s in path]
        ax3.plot(distances, path_vals, marker='o', lw=2, label=f"$\\gamma={g}$")
    ax3.set_title("最优路径状态价值 vs 到终点的步数距离\n($\\gamma=1.0$ 为线性，$\\gamma<1.0$ 为指数压缩)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("距离终点的剩余步数 (Remaining Steps)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("最优状态价值 $V^*(s)$", fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=10)
    
    for ax in [ax2]:
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1.2)
        
    plt.tight_layout()
    save_path = "images/gamma_comparison_experiment.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✅ 对比实验完成！高清图已保存至: {save_path}")

if __name__ == "__main__":
    main()

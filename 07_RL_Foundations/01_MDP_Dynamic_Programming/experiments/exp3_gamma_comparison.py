"""
实验三: 折扣因子 Gamma 严格对比实验 (Discount Factor Gamma Study)
测试 gamma in [0.7, 0.9, 0.99, 1.0] 的收敛扫描次数与价值衰减规律。
输出: 生成 images/gamma_comparison_experiment.png
"""
import os
import sys
import pathlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_cache")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

from src.complex_grid_world import ComplexGridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_SYMBOLS
from src.complex_model_free import solve_dp_optimal
from src.policy_iteration import PolicyIteration

def run():
    print("="*80)
    print("  强化学习实验三：复杂网格下折扣因子 Gamma 对策略抉择与收敛速度的影响研究")
    print("  [双终点博弈: 近端次级目标 +3.0 vs 远端终极大奖 +10.0]")
    print("="*80)
    
    gammas = [0.5, 0.7, 0.85, 0.95]
    results = {}
    
    for g in gammas:
        print(f"\n>>> 正在运行 Gamma = {g} ...")
        env = ComplexGridWorld(rows=4, cols=4, gamma=g, step_cost=-1.0)
        solver = PolicyIteration(env, theta=1e-4)
        history = solver.run(verbose_states=[])
        
        # 提取起点贪婪路径
        V_opt, Q_opt = solve_dp_optimal(env, gamma=g)
        s = (0, 0)
        path = [s]
        for _ in range(12):
            if env.is_terminal(s): break
            a = max(Q_opt[s].keys(), key=lambda act: Q_opt[s][act])
            _, next_s, _ = env.get_transitions(s, a)[0]
            path.append(next_s)
            s = next_s
            
        target_reached = "次级安全目标 (0, 3) [+3.0]" if path[-1] == (0, 3) else "终极大奖目标 (3, 3) [+10.0]"
        print(f"  Gamma={g}: 评估耗时 {history[0]['sweeps']} sweeps, 最终目标: {target_reached}")
        print(f"  起点轨迹: {' -> '.join(str(p) for p in path)}")
        
        results[g] = {
            "eval_sweeps_iter0": history[0]["sweeps"],
            "total_sweeps": sum(h["sweeps"] for h in history),
            "num_policy_iters": len(history),
            "V_final": V_opt,
            "Q_final": Q_opt,
            "path": path,
            "target": target_reached
        }

    # 绘制 4 联对比学术图谱 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # -------------------------------------------------------------------------
    # 图 (a): 收敛速度柱状图
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    sweeps_iter0 = [results[g]["eval_sweeps_iter0"] for g in gammas]
    sweeps_total = [results[g]["total_sweeps"] for g in gammas]
    x = np.arange(len(gammas))
    width = 0.35
    ax.bar(x - width/2, sweeps_iter0, width, label='初始评估扫描数 (k=0)', color='#3B82F6')
    ax.bar(x + width/2, sweeps_total, width, label='全流程累计扫描总数', color='#10B981')
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\gamma={g}$" for g in gammas], fontsize=12)
    ax.set_ylabel('网格扫描次数 (Sweeps)', fontsize=11, fontweight='bold')
    ax.set_title('图 (a): 折扣因子 $\gamma$ 与算法收敛步数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i in range(len(gammas)):
        ax.text(i - width/2, sweeps_iter0[i] + 3, str(sweeps_iter0[i]), ha='center', fontweight='bold')
        ax.text(i + width/2, sweeps_total[i] + 3, str(sweeps_total[i]), ha='center', fontweight='bold')

    # -------------------------------------------------------------------------
    # 图 (b): 短视策略 Gamma = 0.70 (落袋为安次级目标)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    v_g07 = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            v_g07[r, c] = results[0.7]["V_final"][(r, c)]
    im_b = ax.imshow(v_g07, cmap='Spectral_r', origin='upper')
    plt.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('图 (b): $\gamma=0.70$ (短视策略) $\\rightarrow$ 直奔次级目标 (0, 3)\n[未来折扣剧烈，3步拿到+3优于6步拿+10]', fontsize=12, fontweight='bold')
    for r in range(4):
        for c in range(4):
            s = (r, c)
            val = v_g07[r, c]
            if s == (0, 3):
                ax.text(c, r, f"SUB-GOAL\n[+3.0]\n{val:.2f}", ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.2", fc="#C8E6C9"))
            elif s == (3, 3):
                ax.text(c, r, f"MAIN GOAL\n[+10.0]\n{val:.2f}", ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.2", fc="#A5D6A7"))
            elif s in [(1, 2), (2, 3), (2, 1)]:
                ax.text(c, r, f"HAZARD\n{val:.2f}", ha='center', va='center', fontsize=7.5, fontweight='bold', color='darkred', bbox=dict(boxstyle="round,pad=0.2", fc="#FFCDD2"))
            else:
                best_a = max(results[0.7]["Q_final"][s].keys(), key=lambda act: results[0.7]["Q_final"][s][act])
                sym = ACTION_SYMBOLS[best_a]
                ax.text(c, r - 0.15, f"{sym}", ha='center', va='center', fontsize=14, fontweight='bold', color='navy')
                ax.text(c, r + 0.25, f"{val:.2f}", ha='center', va='center', fontsize=8.5, fontweight='bold')

    # -------------------------------------------------------------------------
    # 图 (c): 远瞻策略 Gamma = 0.95 (外圈迂回追逐终极大奖)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    v_g95 = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            v_g95[r, c] = results[0.95]["V_final"][(r, c)]
    im_c = ax.imshow(v_g95, cmap='Spectral_r', origin='upper')
    plt.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('图 (c): $\gamma=0.95$ (远瞻策略) $\\rightarrow$ 外圈迂回大奖 (3, 3)\n[耐心容忍6步外围长途，追求全局最高总回报]', fontsize=12, fontweight='bold')
    for r in range(4):
        for c in range(4):
            s = (r, c)
            val = v_g95[r, c]
            if s == (0, 3):
                ax.text(c, r, f"SUB-GOAL\n[+3.0]\n{val:.2f}", ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.2", fc="#C8E6C9"))
            elif s == (3, 3):
                ax.text(c, r, f"MAIN GOAL\n[+10.0]\n{val:.2f}", ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.2", fc="#A5D6A7"))
            elif s in [(1, 2), (2, 3), (2, 1)]:
                ax.text(c, r, f"HAZARD\n{val:.2f}", ha='center', va='center', fontsize=7.5, fontweight='bold', color='darkred', bbox=dict(boxstyle="round,pad=0.2", fc="#FFCDD2"))
            else:
                best_a = max(results[0.95]["Q_final"][s].keys(), key=lambda act: results[0.95]["Q_final"][s][act])
                sym = ACTION_SYMBOLS[best_a]
                ax.text(c, r - 0.15, f"{sym}", ha='center', va='center', fontsize=14, fontweight='bold', color='navy')
                ax.text(c, r + 0.25, f"{val:.2f}", ha='center', va='center', fontsize=8.5, fontweight='bold')

    # -------------------------------------------------------------------------
    # 图 (d): 折扣回报相变解析曲线 (Theoretical Return Phase Transition)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    gamma_range = np.linspace(0.4, 0.99, 500)
    G_sub_curve = -1.0 - gamma_range - gamma_range**2 + 3.0 * (gamma_range**3)
    G_main_curve = np.array([sum(-g**i for i in range(6)) + 10.0 * (g**6) for g in gamma_range])
    
    # 相变交点
    diff = G_main_curve - G_sub_curve
    idx_cross = np.where(diff > 0)[0][0]
    gamma_star = gamma_range[idx_cross]
    
    ax.plot(gamma_range, G_sub_curve, color='#E11D48', lw=2.5, label='次级安全目标 (3步达 +3.0) $G_{sub}(\gamma)$')
    ax.plot(gamma_range, G_main_curve, color='#2563EB', lw=2.5, label='终极大奖目标 (6步达 +10.0) $G_{main}(\gamma)$')
    ax.axvline(x=gamma_star, color='purple', linestyle='--', lw=1.8, label=f'相变临界阈值 $\gamma^* \\approx {gamma_star:.2f}$')
    ax.scatter([gamma_star], [G_sub_curve[idx_cross]], color='purple', s=80, zorder=5)
    
    ax.set_xlabel('折扣因子 $\gamma$', fontsize=11, fontweight='bold')
    ax.set_ylabel('起点 (0, 0) 折扣累积回报期望 $G$', fontsize=11, fontweight='bold')
    ax.set_title('图 (d): 双终点理论回报相变临界点曲线\n[$\gamma < 0.82$: 选次要目标 | $\gamma > 0.82$: 选终极大奖]', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    images_dir = os.path.join(PROJECT_ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)
    save_path = os.path.join(images_dir, "gamma_comparison_experiment.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n[可视化输出] Gamma 对比图表已生成: {save_path}")
    print("实验三执行完毕！")

if __name__ == "__main__":
    run()

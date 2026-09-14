"""
Main Experiment Runner for Policy Iteration White-box Project.
Executes the experiment, records exact computation outputs, and saves figures.
"""
import os
import sys
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Configure local matplotlib cache inside workspace
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")

# Sandbox pathlib monkeypatch fix
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.grid_world import GridWorld, ACTIONS, ACTION_NAMES, ACTION_SYMBOLS, UP, DOWN, LEFT, RIGHT
from src.policy_iteration import PolicyIteration
from src.visualizer import render_grid_evolution, plot_convergence_loss

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def print_grid_map(env: GridWorld):
    print("\n" + "="*75)
    print("  📍 网格世界 (GridWorld 4x4) 实验拓扑与地貌图")
    print("="*75)
    print("坐标定义: (行 row, 列 col)，左上角为 (0, 0)，右下角为 (3, 3)")
    print("动作空间: 上(0), 下(1), 左(2), 右(3) | 碰壁反弹原地 | 转移为确定性概率 1.0")
    print("奖励设定: 普通移动单步惩罚 = -1.0 | 陷阱 TRAP (1, 2) 惩罚 = -5.0 | 终点 GOAL (3, 3) 吸收态 = 0.0")
    print("折扣因子: gamma = 0.9 | 收敛判据: theta = 1e-4\n")
    
    for r in range(env.rows):
        row_str = " | "
        for c in range(env.cols):
            s = (r, c)
            if env.is_terminal(s):
                cell = " 🎯 GOAL "
            elif s in env.trap_states:
                cell = " ⚠️ TRAP "
            elif s == (0, 0):
                cell = " 🚀 START"
            else:
                cell = f"  ({r},{c}) "
            row_str += cell + " | "
        print(row_str)
        print(" " + "-" * (len(row_str) - 2))

def print_grid_matrix(title: str, val_dict: dict, rows: int, cols: int):
    print(f"\n--- {title} ---")
    for r in range(rows):
        row_str = " | "
        for c in range(cols):
            row_str += f"{val_dict[(r, c)]:8.4f} | "
        print(row_str)

def print_policy_matrix(title: str, pi_dict: dict, rows: int, cols: int, env: GridWorld):
    print(f"\n--- {title} ---")
    for r in range(rows):
        row_str = " | "
        for c in range(cols):
            s = (r, c)
            if env.is_terminal(s):
                sym = "  [GOAL] "
            else:
                # Find all actions with max probability
                max_p = max(pi_dict[s].values())
                active = [ACTION_SYMBOLS[a] for a, p in pi_dict[s].items() if p > 0.05]
                sym = f"  {''.join(active):<6} "
            row_str += f"{sym} | "
        print(row_str)

def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.txt")
    sys.stdout = DualLogger(log_path)
    
    print("="*75)
    print("  强化学习白盒透视实验：策略迭代 (Policy Iteration) 全流程实证")
    print("="*75)
    
    # 1. Initialize environment
    env = GridWorld(rows=4, cols=4, goal_states=[(3, 3)], trap_states={(1, 2): -5.0}, step_cost=-1.0, gamma=0.9)
    print_grid_map(env)
    
    # 2. Run Policy Iteration
    pi_solver = PolicyIteration(env, theta=1e-4)
    
    # Track states of interest for deep white-box arithmetic logging
    verbose_states = [(0, 1), (1, 1), (1, 2), (2, 3), (3, 2)]
    
    history = pi_solver.run(verbose_states=verbose_states)
    
    # 3. Print Final Matrices
    print("\n" + "="*75)
    print("  🏆 最终收敛结果一览 (Optimal Values & Policy)")
    print("="*75)
    final_V = history[-1]["V"]
    final_pi = history[-1]["pi"]
    print_grid_matrix("最优状态价值函数 V*(s) 矩阵", final_V, env.rows, env.cols)
    print_policy_matrix("最优确定性策略 pi*(s) 动作网格 (箭头指示方向)", final_pi, env.rows, env.cols, env)
    
    # 4. Generate Images
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(img_dir, exist_ok=True)
    
    evolution_img = os.path.join(img_dir, "policy_iteration_evolution.png")
    convergence_img = os.path.join(img_dir, "policy_evaluation_convergence.png")
    
    render_grid_evolution(history, env, evolution_img)
    plot_convergence_loss(history, convergence_img)
    
    print("\n" + "="*75)
    print("  📊 实验完成！可视化图表已成功保存至 images/ 目录。")
    print("="*75)

if __name__ == "__main__":
    main()

"""
实验一: 经典动态规划对比实验 (Dynamic Programming: Policy Iteration vs Value Iteration)
运行环境: 基础 4x4 GridWorld
输出内容:
- 终端与 logs/experiment_results.txt 同步输出
- 生成 images/policy_iteration_evolution.png (策略演变全景图)
- 生成 images/policy_evaluation_convergence.png (贝尔曼误差收敛曲线)
"""
import os
import sys
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Sandbox pathlib monkeypatch
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_cache")

from src.complex_grid_world import ComplexGridWorld
from src.policy_iteration import PolicyIteration
from src.value_iteration import ValueIteration
from src.visualizer import (
    render_grid_evolution,
    plot_convergence_loss,
    render_value_iteration_evolution,
    plot_value_iteration_convergence
)

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

def run():
    log_path = os.path.join(PROJECT_ROOT, "logs", "experiment_results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    dual_logger = DualLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = dual_logger
    
    try:
        print("="*80)
        print("  强化学习实验一：复杂网格下策略迭代 (PI) 与 价值迭代 (VI) 白盒透视实证")
        print("  [双终点博弈 +3 vs +10 | 三级陷阱 -10, -6, -3]")
        print("="*80)
        
        env = ComplexGridWorld(rows=4, cols=4, gamma=0.9, step_cost=-1.0)
        
        # 1. 运行策略迭代
        print("\n" + "#"*70)
        print("  【模块 A】策略迭代 (Policy Iteration) 启动")
        print("#"*70)
        pi_solver = PolicyIteration(env, theta=1e-4)
        pi_history = pi_solver.run(verbose_states=[(0, 0), (0, 1), (1, 1), (2, 2)])
        
        # 2. 运行价值迭代
        print("\n" + "#"*70)
        print("  【模块 B】价值迭代 (Value Iteration) 启动")
        print("#"*70)
        vi_solver = ValueIteration(env, theta=1e-4)
        vi_history = vi_solver.run(verbose_states=[(0, 0), (1, 1), (1, 2), (2, 2), (3, 2)])
        
        # 3. 渲染图表
        images_dir = os.path.join(PROJECT_ROOT, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        fig1_path = os.path.join(images_dir, "policy_iteration_evolution.png")
        render_grid_evolution(pi_history, env, save_path=fig1_path)
        print(f"\n[可视化输出 1/4] 策略迭代演化全景图已生成: {fig1_path}")
        
        fig2_path = os.path.join(images_dir, "policy_evaluation_convergence.png")
        plot_convergence_loss(pi_history, save_path=fig2_path)
        print(f"[可视化输出 2/4] 策略迭代收敛误差曲线已生成: {fig2_path}")
        
        fig3_path = os.path.join(images_dir, "value_iteration_evolution.png")
        render_value_iteration_evolution(vi_history, env, save_path=fig3_path)
        print(f"[可视化输出 3/4] 价值迭代全流程 8 步演变图已生成: {fig3_path}")
        
        fig4_path = os.path.join(images_dir, "value_iteration_convergence.png")
        plot_value_iteration_convergence(vi_history, save_path=fig4_path)
        print(f"[可视化输出 4/4] 价值迭代贝尔曼最优误差收敛曲线已生成: {fig4_path}")
        
    finally:
        sys.stdout = old_stdout
        dual_logger.log.close()
        print(f"\n实验一执行完毕！详细数值日志已安全归档至: {log_path}")

if __name__ == "__main__":
    run()

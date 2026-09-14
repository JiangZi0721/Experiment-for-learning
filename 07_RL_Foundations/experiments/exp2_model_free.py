"""
实验二: 无模型三剑客对比实验 (Model-Free: TD(0) vs SARSA vs Q-Learning)
运行环境: 基础 4x4 GridWorld
输出内容:
- 终端与 logs/model_free_results.txt 同步输出
- 生成 images/td0_complete_workflow.png
- 生成 images/sarsa_complete_workflow.png
- 生成 images/qlearning_complete_workflow.png
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

from src.complex_grid_world import ComplexGridWorld, ACTIONS, ACTION_NAMES
from src.policy_iteration import PolicyIteration
from src.model_free import run_td_zero, run_sarsa, run_q_learning
from src.plot_model_free import plot_td0_workflow, plot_sarsa_workflow, plot_qlearning_workflow

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

def format_grid(val_dict, rows=4, cols=4, precision=2):
    lines = []
    for r in range(rows):
        row_str = "  ".join([f"{val_dict.get((r, c), 0.0):+6.{precision}f}" for c in range(cols)])
        lines.append(f"  Row {r}: [ {row_str} ]")
    return "\n".join(lines)

def run():
    log_path = os.path.join(PROJECT_ROOT, "logs", "model_free_results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    dual_logger = DualLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = dual_logger
    
    try:
        print("="*80)
        print("  强化学习实验二：复杂网格下无模型三剑客 TD(0)、SARSA 与 Q-Learning 全流程比对")
        print("  [双终点博弈 +3 vs +10 | 三级危险 -10, -6, -3]")
        print("="*80)
        
        env = ComplexGridWorld(rows=4, cols=4, gamma=0.9, step_cost=-1.0)
        
        # 1. DP 求解基准真值
        pi_solver = PolicyIteration(env, theta=1e-5)
        pi_solver.evaluate_policy(outer_iter=0, verbose_states=[])
        true_V_random = pi_solver.V.copy()
        
        # 2. TD(0) 估计
        print("\n" + "-"*60)
        print("1. 运行 TD(0) 状态价值估计 (评估随机策略)")
        print("-"*60)
        td0_res = run_td_zero(env, num_episodes=2500, alpha=0.05, gamma=0.9, true_V=true_V_random, seed=42)
        print("收敛价值矩阵:")
        print(format_grid(td0_res["V"]))
        
        # 3. SARSA 控制
        print("\n" + "-"*60)
        print("2. 运行 SARSA 同策略控制 (Epsilon=0.1, Alpha=0.1)")
        print("-"*60)
        sarsa_res = run_sarsa(env, num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
        sarsa_V = {s: max(sarsa_res["Q"][s].values()) for s in env.states}
        print("收敛 Max Q 价值矩阵:")
        print(format_grid(sarsa_V))
        
        # 4. Q-Learning 控制
        print("\n" + "-"*60)
        print("3. 运行 Q-Learning 异策略控制 (Epsilon=0.1, Alpha=0.1)")
        print("-"*60)
        ql_res = run_q_learning(env, num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
        ql_V = {s: max(ql_res["Q"][s].values()) for s in env.states}
        print("收敛 Max Q 价值矩阵:")
        print(format_grid(ql_V))
        
        # 5. 绘图
        images_dir = os.path.join(PROJECT_ROOT, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        p1 = os.path.join(images_dir, "td0_complete_workflow.png")
        plot_td0_workflow(td0_res, true_V_random, env, save_path=p1)
        print(f"\n[可视化输出] TD(0) 全流程图已生成: {p1}")
        
        p2 = os.path.join(images_dir, "sarsa_complete_workflow.png")
        plot_sarsa_workflow(sarsa_res, env, save_path=p2)
        print(f"[可视化输出] SARSA 全流程图已生成: {p2}")
        
        p3 = os.path.join(images_dir, "qlearning_complete_workflow.png")
        plot_qlearning_workflow(ql_res, sarsa_res, env, save_path=p3)
        print(f"[可视化输出] Q-Learning 全流程图已生成: {p3}")
        
    finally:
        sys.stdout = old_stdout
        dual_logger.log.close()
        print(f"\n实验二执行完毕！详细日志已归档至: {log_path}")

if __name__ == "__main__":
    run()

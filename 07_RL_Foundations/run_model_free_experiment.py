"""
Master Experiment Runner for Model-Free Algorithms: TD(0), SARSA, and Q-Learning.
Runs all three algorithms on the same 4x4 GridWorld, prints transparent step-by-step logs,
and generates three dedicated high-resolution workflow figures.
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

import numpy as np

from src.grid_world import GridWorld, ACTIONS, ACTION_NAMES, ACTION_SYMBOLS, UP, DOWN, LEFT, RIGHT
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

def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_free_results.txt")
    sys.stdout = DualLogger(log_path)
    
    print("="*75)
    print("  强化学习白盒透视实验：TD(0)、SARSA 与 Q-Learning 同任务全流程实证")
    print("="*75)
    
    # 1. Initialize environment
    env = GridWorld(rows=4, cols=4, goal_states=[(3, 3)], trap_states={(1, 2): -5.0}, step_cost=-1.0, gamma=0.9)
    
    # 2. Obtain Ground Truth V^pi under random policy via DP Policy Evaluation
    print("\n>>> [基准计算] 求解随机策略下的 DP 理论真值 (作为 TD(0) 的误差评估基准)...")
    pi_evaluator = PolicyIteration(env, theta=1e-5)
    pi_evaluator.evaluate_policy(outer_iter=0, verbose_states=[])
    true_V = pi_evaluator.V.copy()
    print("✅ DP 理论真值计算完毕！")
    
    # 3. Execute TD(0)
    print("\n" + "="*75)
    print("  1️⃣ 【TD(0) 状态价值估计】算法开始执行")
    print("="*75)
    td_res = run_td_zero(env, num_episodes=2500, alpha=0.05, gamma=0.9, true_V=true_V, seed=42)
    print("\n--- TD(0) 单步白盒运算展开 (前 3 步真实执行痕迹) ---")
    for step_info in td_res["white_box_steps"]:
        print(f"时间步 t={step_info['step']} | 状态 S_t={step_info['s']} | 动作 A_t={ACTION_NAMES[step_info['a']]} "
              f"| 奖励 R={step_info['r']:.1f} | 下一状态 S'={step_info['next_s']}")
        print(f"  公式: delta_t = R + gamma * V(S') - V(S)")
        print(f"  代入: delta_t = {step_info['r']:.1f} + 0.9 * {step_info['v_next']:.4f} - ({step_info['old_v']:.4f}) = {step_info['error']:.4f}")
        print(f"  更新: V(S) <- {step_info['old_v']:.4f} + 0.05 * ({step_info['error']:.4f}) = {step_info['new_v']:.4f}\n")
        
    final_rmse = td_res["rmse_history"][-1]
    print(f"🏆 TD(0) 2500 回合训练完毕！与 DP 理论真值的最终 RMSE 误差为: {final_rmse:.4f}")

    # 4. Execute SARSA
    print("\n" + "="*75)
    print("  2️⃣ 【SARSA 同策略控制】算法开始执行")
    print("="*75)
    sarsa_res = run_sarsa(env, num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    print("\n--- SARSA 单步白盒运算展开 (前 3 步真实执行痕迹) ---")
    for step_info in sarsa_res["white_box_steps"]:
        next_a_str = ACTION_NAMES[step_info['next_a']] if step_info['next_a'] is not None else "None (终止)"
        print(f"时间步 t={step_info['step']} | 状态 S_t={step_info['s']} | 动作 A_t={ACTION_NAMES[step_info['a']]} "
              f"| 奖励 R={step_info['r']:.1f} | S'={step_info['next_s']} | 下一步实际动作 A'={next_a_str}")
        print(f"  公式: Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]")
        print(f"  代入: Target = {step_info['target']:.4f}, Error = {step_info['error']:.4f}")
        print(f"  更新: Q({step_info['s']}, {ACTION_NAMES[step_info['a']]}) <- {step_info['old_q']:.4f} + 0.1 * ({step_info['error']:.4f}) = {step_info['new_q']:.4f}\n")
    print(f"🏆 SARSA 1500 回合训练完毕！最后 100 回合平均回报: {np.mean(sarsa_res['reward_history'][-100:]):.2f}")

    # 5. Execute Q-Learning
    print("\n" + "="*75)
    print("  3️⃣ 【Q-Learning 异策略控制】算法开始执行")
    print("="*75)
    ql_res = run_q_learning(env, num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    print("\n--- Q-Learning 单步白盒运算展开 (前 3 步真实执行痕迹) ---")
    for step_info in ql_res["white_box_steps"]:
        print(f"时间步 t={step_info['step']} | 状态 S_t={step_info['s']} | 动作 A_t={ACTION_NAMES[step_info['a']]} "
              f"| 奖励 R={step_info['r']:.1f} | S'={step_info['next_s']}")
        print(f"  公式: Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a' Q(S', a') - Q(S, A)]")
        print(f"  代入: Target = {step_info['target']:.4f}, Error = {step_info['error']:.4f}")
        print(f"  更新: Q({step_info['s']}, {ACTION_NAMES[step_info['a']]}) <- {step_info['old_q']:.4f} + 0.1 * ({step_info['error']:.4f}) = {step_info['new_q']:.4f}\n")
    print(f"🏆 Q-Learning 1500 回合训练完毕！最后 100 回合平均回报: {np.mean(ql_res['reward_history'][-100:]):.2f}")

    # 6. Generate the 3 dedicated high-resolution figures
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(img_dir, exist_ok=True)
    
    td0_img = os.path.join(img_dir, "td0_complete_workflow.png")
    sarsa_img = os.path.join(img_dir, "sarsa_complete_workflow.png")
    ql_img = os.path.join(img_dir, "qlearning_complete_workflow.png")
    
    plot_td0_workflow(td_res, true_V, env, td0_img)
    plot_sarsa_workflow(sarsa_res, env, sarsa_img)
    plot_qlearning_workflow(ql_res, sarsa_res, env, ql_img)
    
    print("\n" + "="*75)
    print("  📊 实验全流程圆满成功！三幅高清工作流图已保存至 images/ 目录：")
    print(f"   1. {td0_img}")
    print(f"   2. {sarsa_img}")
    print(f"   3. {ql_img}")
    print("="*75)

if __name__ == "__main__":
    main()

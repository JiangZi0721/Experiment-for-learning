"""
实验五: 深度强化学习 DQN 算法透视与真值逼近实验 (Deep Q-Network vs Tabular Ground Truth)
运行环境: 标准 4x4 GridWorld
输出内容:
- 终端与 logs/dqn_results.txt 同步全量数值输出
- 生成 images/dqn_training_curves.png (DQN 四联收敛曲线)
- 生成 images/dqn_policy_and_q_heatmap.png (DQN vs 理论真值对比图)
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_cache")

import numpy as np
from src.complex_grid_world import ComplexGridWorld, ACTIONS, ACTION_NAMES
from src.complex_model_free import solve_dp_optimal
from src.dqn import DQNAgent
from src.plot_dqn import plot_dqn_training_curves, plot_dqn_policy_and_q_heatmap

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
    log_path = os.path.join(PROJECT_ROOT, "logs", "dqn_results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    dual_logger = DualLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = dual_logger
    
    try:
        print("="*80)
        print("  强化学习实验五：复杂网格下深度 Q 网络 (DQN) 白盒透视与真值逼近实证")
        print("  [双终点博弈 +3 vs +10 | 三级危险 -10, -6, -3]")
        print("="*80)
        
        env = ComplexGridWorld(rows=4, cols=4, gamma=0.9, step_cost=-1.0)
        
        # 1. 求解理论最优基准真值 Q*(s, a) 与 V*(s)
        print("\n[基准校验] 运行价值迭代求解全局理论最优真值 V*(s) 与 Q*(s, a)...")
        ground_truth_v, ground_truth_q = solve_dp_optimal(env, gamma=0.9)
        print(f"理论最优真值计算完毕！终点价值 = {ground_truth_v[(3,3)]:.2f}, 起点价值 = {ground_truth_v[(0,0)]:.4f}")
        
        # 2. 构建并训练 DQN 智能体
        print("\n" + "#"*70)
        print("  【模块 A】构建深度 Q 网络 (DQN) 并启动经验回放训练")
        print("#"*70)
        agent = DQNAgent(
            env=env,
            lr=1e-3,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.990,
            buffer_capacity=3000,
            batch_size=32,
            target_update_freq=50,
            hidden_dim=64,
            seed=42
        )
        
        history = agent.train(
            episodes=350,
            max_steps_per_episode=60,
            ground_truth_q=ground_truth_q,
            verbose_interval=50
        )
        
        # 3. 最终策略与价值评估汇报
        print("\n" + "#"*70)
        print("  【模块 B】DQN 训练成果白盒透视报告")
        print("#"*70)
        
        print("\n[1] DQN 神经网络输出全图状态价值 V_θ(s):")
        v_mat_dqn = np.zeros((4, 4))
        v_mat_gt = np.zeros((4, 4))
        for r in range(4):
            for c in range(4):
                v_mat_dqn[r, c] = history["final_V"][(r, c)]
                v_mat_gt[r, c] = ground_truth_v[(r, c)]
        print(np.round(v_mat_dqn, 4))
        
        print("\n[2] 动态规划理论全局最优价值 V*(s):")
        print(np.round(v_mat_gt, 4))
        
        abs_diff = np.abs(v_mat_dqn - v_mat_gt)
        print(f"\n[3] 绝对误差统计: 平均绝对误差 MAE = {np.mean(abs_diff):.4f}, 最大误差 MaxError = {np.max(abs_diff):.4f}")
        
        # 测试贪婪评估路径
        print("\n[4] 从起点 (0, 0) 出发执行完全确定性贪婪策略路径验证:")
        curr_state = env.reset()
        eval_path = [curr_state]
        eval_return = 0.0
        eval_steps = 0
        while not env.is_terminal(curr_state) and eval_steps < 20:
            a = agent.select_action(curr_state, deterministic=True)
            next_s, r, done = env.step(a)
            eval_path.append(next_s)
            eval_return += r
            curr_state = next_s
            eval_steps += 1
            
        print(f"  评估轨迹: {' -> '.join(str(s) for s in eval_path)}")
        print(f"  轨迹总步数: {eval_steps} 步 | 累积回报: {eval_return:.1f}")
        is_optimal = (eval_path[-1] == (3, 3) and eval_steps == 6)
        print(f"  评定结论: {'🏆 达到全局最优外圈避险大奖轨迹！' if is_optimal else '✅ 成功避险达成目标！'}")
        
        # 4. 渲染可视化学术图表
        images_dir = os.path.join(PROJECT_ROOT, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        fig1_path = os.path.join(images_dir, "dqn_training_curves.png")
        plot_dqn_training_curves(history, save_path=fig1_path)
        print(f"\n[可视化输出 1/2] DQN 四联收敛曲线图已生成: {fig1_path}")
        
        fig2_path = os.path.join(images_dir, "dqn_policy_and_q_heatmap.png")
        plot_dqn_policy_and_q_heatmap(agent, env, ground_truth_v, ground_truth_q, save_path=fig2_path)
        print(f"[可视化输出 2/2] DQN 策略与价值对比热力图已生成: {fig2_path}")
        
    finally:
        sys.stdout = old_stdout
        dual_logger.log.close()
        print(f"\n实验五执行完毕！详细数值日志已安全归档至: {log_path}")

if __name__ == "__main__":
    run()

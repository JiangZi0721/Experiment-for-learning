"""
实验七: 深度世界模型 (Deep World Model) 白盒实证：从神经网络训练到做梦推演全流程
运行环境: 连续非线性随机单摆 (Stochastic Inverted Pendulum)

全流程实验包含四大核心阶段:
1. 数据收集与经验回放构建 (Environment Interaction -> Replay Buffer)
2. 深度世界模型概率训练 (Gaussian NLL + Reward MSE Optimization)
3. 脑内做梦推演与真实物理世界高精度对比 (Dream Imagination vs Real Physics)
4. 多模态平行宇宙发散云实验 (Sample Model Multi-modal Fan via White Noise Injection)
5. 基于脑内做梦推演的闭环模型预测控制 (Imagination-driven MPC Swing-Up Control)

输出产物:
- 同步双写控制台与 logs/deep_world_model_results.txt
- images/world_model_training_curves.png
- images/world_model_dream_vs_real.png
- images/world_model_parallel_universes.png
- images/world_model_mpc_control.png
- images/world_model_master_dashboard.png
"""
import os
import sys
import time
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_cache")

# Sandbox pathlib patch if available
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import torch
import numpy as np

from src.deep_world_model import (
    StochasticPendulumEnv,
    DeepWorldModel,
    TransitionReplayBuffer,
    WorldModelTrainer,
    DreamSimulator
)
from src.plot_world_model import (
    plot_world_model_training,
    plot_dream_vs_real,
    plot_parallel_universes,
    plot_mpc_control_comparison,
    plot_world_model_master_dashboard
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
    log_path = os.path.join(PROJECT_ROOT, "logs", "deep_world_model_results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    dual_logger = DualLogger(log_path)
    sys.stdout = dual_logger
    
    print("=" * 80)
    print("  强化学习实验七：深度世界模型 (Deep World Model) 白盒实证与做梦推演全流程")
    print("  [连续非线性倒立摆 | 高斯重参数化采样 | 潜空间白噪声平行宇宙发散云 | MPC 闭环控制]")
    print("=" * 80)
    
    # 检测硬件加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> [系统环境] 计算引擎设备: {device.upper()} | PyTorch: {torch.__version__}")
    
    # -------------------------------------------------------------
    # 阶段一：真实物理世界数据交互采集 (Data Collection)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> 【阶段一：真实世界数据采集】智能体与客观物理环境交互并沉淀历史经验...")
    print("-" * 80)
    
    env = StochasticPendulumEnv(noise_std=0.25)
    buffer = TransitionReplayBuffer(capacity=15000)
    
    total_collect_steps = 10000
    obs = env.reset()
    start_collect_time = time.perf_counter()
    
    for step in range(total_collect_steps):
        # 采用复合混合策略探索 (大范围探索 + 伪正弦激振)
        if np.random.rand() < 0.3:
            action = float(np.random.uniform(-2.0, 2.0))
        else:
            action = float(1.8 * np.sin(0.2 * step) + np.random.normal(0, 0.2))
            
        next_obs, reward, _, info = env.step(action)
        buffer.add(obs, action, reward, next_obs)
        obs = next_obs
        
        # 偶尔随机重置，覆盖更广泛的状态空间
        if (step + 1) % 200 == 0:
            obs = env.reset()
            
    collect_time = time.perf_counter() - start_collect_time
    print(f"  * 成功采集转换四元组 (s, a, r, s'): {len(buffer)} 条")
    print(f"  * 真实交互挂钟耗时: {collect_time:.2f} 秒 (平均单步交互耗时: {collect_time/total_collect_steps*1e6:.1f} 微秒)")
    print(f"  * 状态空间维度: {obs.shape} | 包含物理随机扰动: 高斯噪声 sigma = 0.25")
    
    # -------------------------------------------------------------
    # 阶段二：深度世界模型神经网络训练 (World Model Training)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> 【阶段二：深度世界模型参数优化】训练高斯概率转移网络与奖励网络...")
    print("-" * 80)
    
    world_model = DeepWorldModel(state_dim=3, action_dim=1, hidden_dim=128, device=device)
    trainer = WorldModelTrainer(world_model, lr=1.5e-3, weight_decay=1e-4)
    
    train_steps = 3000
    batch_size = 128
    train_history = {"steps": [], "total_loss": [], "nll_loss": [], "reward_loss": [], "mse_norm": []}
    
    start_train_time = time.perf_counter()
    for step in range(1, train_steps + 1):
        batch = buffer.sample(batch_size)
        loss_dict = trainer.train_step(batch)
        
        if step % 50 == 0 or step == 1:
            train_history["steps"].append(step)
            train_history["total_loss"].append(loss_dict["total_loss"])
            train_history["nll_loss"].append(loss_dict["nll_loss"])
            train_history["reward_loss"].append(loss_dict["reward_loss"])
            train_history["mse_norm"].append(loss_dict["mse_norm"])
            
        if step % 500 == 0 or step == train_steps:
            print(f"  Step [{step:4d}/{train_steps}] | Total Loss: {loss_dict['total_loss']:.4f} | "
                  f"Dynamics NLL: {loss_dict['nll_loss']:.4f} | Reward MSE: {loss_dict['reward_loss']:.4f} | "
                  f"State MSE: {loss_dict['mse_norm']:.5f}")
                  
    train_time = time.perf_counter() - start_train_time
    print(f"  * 深度世界模型训练完成！总梯度反向传播耗时: {train_time:.2f} 秒")
    
    # 绘制训练曲线
    train_fig_path = os.path.join(PROJECT_ROOT, "images", "world_model_training_curves.png")
    plot_world_model_training(train_history, train_fig_path)
    print(f"  * 训练收敛曲线已持久化至: {train_fig_path}")
    
    # -------------------------------------------------------------
    # 阶段三：脑内做梦推演 vs 真实物理单轨高精度对比 (Dream vs Real)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> 【阶段三：做梦推演实测】相同初始状态与动作序列下，脑内白日梦 vs 真实物理...")
    print("-" * 80)
    
    dreamer = DreamSimulator(world_model)
    
    # 设定固定的测试初始状态 (摆杆偏转 90 度，带微小角速度)
    init_state = env.reset(theta=np.pi / 2.0, theta_dot=0.5)
    horizon = 60
    # 连续动态正弦控制动作序列
    test_actions = [float(1.6 * np.sin(0.25 * t)) for t in range(horizon)]
    
    # 1. 真实物理世界单步推进 (Ground Truth)
    real_states = [init_state]
    real_rewards = []
    real_env_copy = StochasticPendulumEnv(noise_std=0.25)
    real_env_copy.theta = env.theta
    real_env_copy.theta_dot = env.theta_dot
    
    for a in test_actions:
        next_o, r, _, _ = real_env_copy.step(a)
        real_states.append(next_o)
        real_rewards.append(r)
        
    real_traj = {
        "states": np.array(real_states),
        "rewards": np.array(real_rewards)
    }
    
    # 2. 深度世界模型脑内闭门做梦 (Dream Rollout via Sample Model)
    dream_start_time = time.perf_counter()
    dream_traj = dreamer.rollout_dream(init_state, test_actions, deterministic=False)
    dream_time = time.perf_counter() - dream_start_time
    
    # 计算均方误差
    final_l2_drift = np.linalg.norm(real_traj["states"][-1] - dream_traj["states"][-1])
    mean_l2_drift = np.mean(np.linalg.norm(real_traj["states"] - dream_traj["states"], axis=1))
    
    print(f"  * 脑内自回归自举推演 {horizon} 步耗时: {dream_time * 1000:.2f} 毫秒 (单步推演只需 {dream_time/horizon*1e6:.1f} 微秒！)")
    print(f"  * 60 步长程自回归推演平均状态漂移误差 (Mean L2 Drift): {mean_l2_drift:.4f}")
    print(f"  * 最终第 60 步终端状态误差 (Final Step L2 Drift): {final_l2_drift:.4f}")
    
    dream_fig_path = os.path.join(PROJECT_ROOT, "images", "world_model_dream_vs_real.png")
    plot_dream_vs_real(real_traj, dream_traj, dream_fig_path)
    print(f"  * 脑内推演与真实物理对比图已持久化至: {dream_fig_path}")
    
    # -------------------------------------------------------------
    # 阶段四：多模态平行宇宙发散云实验 (Parallel Universes Fan)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> 【阶段四：样本模型核心机制】相同 (s0, a)，白噪声注入生成 60 个平行宇宙...")
    print("-" * 80)
    
    num_universes = 60
    fan_start_time = time.perf_counter()
    parallel_trajs = dreamer.parallel_dream_fan(init_state, test_actions, num_universes=num_universes)
    fan_time = time.perf_counter() - fan_start_time
    
    print(f"  * 并行演化 {num_universes} 个平行宇宙未来完成！总耗时: {fan_time * 1000:.2f} 毫秒")
    print(f"  * 物理启示：世界模型【根本不需要做多重连续积分】，每次推演只需从 N(0, I) 采样一段白噪声，")
    print(f"    成百上千次单步抽样自然而然构成一条概率分布云（粒子滤波效应）！")
    
    fan_fig_path = os.path.join(PROJECT_ROOT, "images", "world_model_parallel_universes.png")
    plot_parallel_universes(parallel_trajs, real_traj["states"], fan_fig_path)
    print(f"  * 平行宇宙分布发散云图已持久化至: {fan_fig_path}")
    
    # -------------------------------------------------------------
    # 阶段五：基于脑内做梦推演的闭环控制 (Imagination-driven MPC)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> 【阶段五：闭环实战检验】基于世界模型脑内推演的 MPC 摇摆倒立控制...")
    print("-" * 80)
    
    control_steps = 70
    mpc_env = StochasticPendulumEnv(noise_std=0.20)
    # 初始状态设在最困难的下垂死点 (theta = -pi)
    mpc_obs = mpc_env.reset(theta=-np.pi, theta_dot=0.0)
    
    mpc_log = {"thetas": [mpc_env.theta], "omegas": [mpc_env.theta_dot], "rewards": []}
    
    print("  * 智能体开始闭环控制：每一步在脑内推演 80 条候选动作序列，评估累积回报并执行最优动作...")
    mpc_start_time = time.perf_counter()
    
    for t in range(control_steps):
        # 在大脑内部推演规划最佳动作 (Horizon = 12, Candidates = 80)
        action = dreamer.plan_mpc(mpc_obs, horizon=12, num_candidates=80)
        
        # 将脑内挑出的最佳动作投入真实世界执行
        next_mpc_obs, r, _, _ = mpc_env.step(action)
        mpc_obs = next_mpc_obs
        
        mpc_log["thetas"].append(mpc_env.theta)
        mpc_log["omegas"].append(mpc_env.theta_dot)
        mpc_log["rewards"].append(r)
        
        if (t + 1) % 15 == 0 or t == control_steps - 1:
            print(f"    Step [{t+1:2d}/{control_steps}] | Current Angle theta: {mpc_env.theta:+.3f} rad | "
                  f"Angular Velocity: {mpc_env.theta_dot:+.3f} rad/s | Step Reward: {r:+.3f}")
                  
    mpc_time = time.perf_counter() - mpc_start_time
    
    # 对比随机策略基准
    rnd_env = StochasticPendulumEnv(noise_std=0.20)
    rnd_env.theta = -np.pi
    rnd_env.theta_dot = 0.0
    random_log = {"thetas": [rnd_env.theta], "omegas": [rnd_env.theta_dot], "rewards": []}
    for _ in range(control_steps):
        act = float(np.random.uniform(-2.0, 2.0))
        _, r_rnd, _, _ = rnd_env.step(act)
        random_log["thetas"].append(rnd_env.theta)
        random_log["omegas"].append(rnd_env.theta_dot)
        random_log["rewards"].append(r_rnd)
        
    print(f"  * MPC 闭环控制完成！70 步执行总耗时: {mpc_time:.2f} 秒")
    print(f"  * 脑内做梦规划控制总回报: {np.sum(mpc_log['rewards']):.2f} vs 随机策略盲目探索总回报: {np.sum(random_log['rewards']):.2f}")
    
    mpc_fig_path = os.path.join(PROJECT_ROOT, "images", "world_model_mpc_control.png")
    plot_mpc_control_comparison(mpc_log, random_log, mpc_fig_path)
    print(f"  * 闭环控制表现图已持久化至: {mpc_fig_path}")
    
    # -------------------------------------------------------------
    # 阶段六：全生命周期综合大盘看板绘制 (Master Dashboard)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> 【阶段六：综合看板绘制】生成出版级四联屏 Master Dashboard...")
    print("-" * 80)
    
    dash_fig_path = os.path.join(PROJECT_ROOT, "images", "world_model_master_dashboard.png")
    plot_world_model_master_dashboard(
        train_history, real_traj, dream_traj, parallel_trajs, mpc_log, dash_fig_path
    )
    print(f"  * 综合大盘看板已持久化至: {dash_fig_path}")
    
    print("\n" + "=" * 80)
    print("  深度世界模型全流程仿真实验圆满完成！全部数据与图像已写入 logs 与 images 目录。")
    print("=" * 80)

if __name__ == "__main__":
    run()

"""
可视化与图表绘制模块：深度世界模型 (Deep World Model) 做梦仿真全景展现
绘制高分辨率 (300 DPI) 出版级分析图表：
1. world_model_training_curves.png (世界模型训练收敛曲线)
2. world_model_dream_vs_real.png (脑内推演与真实物理世界高精度对比)
3. world_model_parallel_universes.png (样本模型核心特征：多模态平行宇宙发散云)
4. world_model_mpc_control.png (基于脑内做梦规划的闭环控制表现)
5. world_model_master_dashboard.png (深度世界模型全生命周期综合决策看板)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_world_model_training(train_history: dict, save_path: str):
    """图 1：深度世界模型训练收敛与损失演化曲线"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    steps = train_history.get("steps", np.arange(len(train_history["total_loss"])))
    
    # 1. 联合总损失与高斯负对数似然 (NLL)
    ax1.plot(steps, train_history["total_loss"], label="Total Joint Loss", color="#2C3E50", linewidth=2.0)
    ax1.plot(steps, train_history["nll_loss"], label="Dynamics Gaussian NLL", color="#E74C3C", linestyle="--", linewidth=1.8)
    ax1.set_title("World Model Training Loss Evolution\n(Gaussian NLL + Reward MSE)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Optimization Gradient Steps", fontsize=10)
    ax1.set_ylabel("Loss Magnitude", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95)
    
    # 2. 奖励预测损失
    ax2.plot(steps, train_history["reward_loss"], label="Reward Predictor MSE", color="#27AE60", linewidth=2.0)
    ax2.set_title("Reward Predictor Convergence\nMSE Loss ||r - r_hat||^2", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Optimization Gradient Steps", fontsize=10)
    ax2.set_ylabel("Reward MSE", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.95)
    
    # 3. 动力学均方根误差 (MSE Norm)
    ax3.plot(steps, train_history["mse_norm"], label="State Transition MSE", color="#2980B9", linewidth=2.0)
    ax3.set_title("Next-State Prediction Error\n||s_{t+1} - mu(s_t, a_t)||^2", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Optimization Gradient Steps", fontsize=10)
    ax3.set_ylabel("State Vector MSE", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.95)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_dream_vs_real(real_traj: dict, dream_traj: dict, save_path: str):
    """图 2：脑内做梦推演 (Dream Rollout) vs 真实物理世界 (Ground Truth) 全景对比"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    
    time_steps = np.arange(len(real_traj["states"]))
    
    real_theta = np.arctan2(real_traj["states"][:, 1], real_traj["states"][:, 0])
    dream_theta = np.arctan2(dream_traj["states"][:, 1], dream_traj["states"][:, 0])
    dream_mu_theta = np.arctan2(dream_traj["mus"][:, 1], dream_traj["mus"][:, 0])
    dream_std_theta = np.linalg.norm(dream_traj["stds"][:, :2], axis=1)
    
    # --- Subplot 1: 角度 theta 演化 ---
    ax1.plot(time_steps, real_theta, label="Real Physics Ground Truth (Env)", color="#2C3E50", linewidth=2.4)
    ax1.plot(time_steps, dream_theta, label="World Model Dream (Sample)", color="#E74C3C", linestyle="-", linewidth=2.0)
    # 绘制置信不确定度带 (+-1 sigma & +-2 sigma)
    # 适配维度 (mus 比 states 少一步)
    steps_m = min(len(time_steps)-1, len(dream_mu_theta))
    ax1.fill_between(time_steps[1:steps_m+1], 
                     dream_mu_theta[:steps_m] - dream_std_theta[:steps_m],
                     dream_mu_theta[:steps_m] + dream_std_theta[:steps_m],
                     color="#E74C3C", alpha=0.25, label="Model Predicted Uncertainty (+-1 sigma)")
    ax1.set_title("Pendulum Angle Evolution: Real World vs Dream Imagination", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Time Steps (dt = 0.05s)", fontsize=10)
    ax1.set_ylabel("Angle theta (rad)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9)
    
    # --- Subplot 2: 角速度 omega 演化 ---
    real_omega = real_traj["states"][:, 2]
    dream_omega = dream_traj["states"][:, 2]
    dream_mu_omega = dream_traj["mus"][:, 2]
    dream_std_omega = dream_traj["stds"][:, 2]
    
    ax2.plot(time_steps, real_omega, label="Real Physics Ground Truth", color="#2C3E50", linewidth=2.4)
    ax2.plot(time_steps, dream_omega, label="World Model Dream", color="#2980B9", linewidth=2.0)
    ax2.fill_between(time_steps[1:steps_m+1],
                     dream_mu_omega[:steps_m] - dream_std_omega[:steps_m],
                     dream_mu_omega[:steps_m] + dream_std_omega[:steps_m],
                     color="#2980B9", alpha=0.25, label="Predicted Velocity Uncertainty (+-1 sigma)")
    ax2.set_title("Angular Velocity Evolution: Real World vs Dream Imagination", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Time Steps (dt = 0.05s)", fontsize=10)
    ax2.set_ylabel("Angular Velocity omega (rad/s)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)
    
    # --- Subplot 3: 奖励函数预测 ---
    r_steps = np.arange(len(real_traj["rewards"]))
    ax3.plot(r_steps, real_traj["rewards"], label="Real Environment Reward", color="#2C3E50", linewidth=2.4)
    ax3.plot(r_steps, dream_traj["rewards"], label="Dream Model Predicted Reward", color="#27AE60", linestyle="--", linewidth=2.0)
    ax3.set_title("Step Reward Tracking: Real vs Model Imagined", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Time Steps", fontsize=10)
    ax3.set_ylabel("Reward Value", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower right", framealpha=0.95, fontsize=9)
    
    # --- Subplot 4: 累积自回归状态误差漂移 (Compounding Error) ---
    l2_errors = np.linalg.norm(real_traj["states"] - dream_traj["states"], axis=1)
    ax4.plot(time_steps, l2_errors, label="Compounding L2 State Error ||s_real - s_dream||", color="#8E44AD", linewidth=2.2)
    ax4.fill_between(time_steps, 0, l2_errors, color="#8E44AD", alpha=0.15)
    ax4.set_title("Autonomous Autoregressive Drift (Multi-step Compounding Error)", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Simulation Horizon Steps H", fontsize=10)
    ax4.set_ylabel("Euclidean State Drift (L2 Norm)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper left", framealpha=0.95, fontsize=9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_parallel_universes(parallel_trajs: list, real_traj: np.ndarray, save_path: str):
    """
    图 3：【样本模型核心灵魂：多模态平行宇宙发散云 (Parallel Universes Fan)】
    直观展示：从完全相同的 (s0, a0) 出发，仅通过采样标准白噪声 epsilon ~ N(0, I)，
    样本模型如何以粒子云的形式完整呈现连续概率分布！
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    num_universes = len(parallel_trajs)
    time_steps = np.arange(len(parallel_trajs[0]))
    
    # 提取所有平行宇宙的角度与角速度
    all_thetas = []
    for traj in parallel_trajs:
        theta_series = np.arctan2(traj[:, 1], traj[:, 0])
        all_thetas.append(theta_series)
        # 绘制每条微弱的平行未来线
        ax1.plot(time_steps, theta_series, color="#3498DB", alpha=0.15, linewidth=1.0)
        ax2.plot(traj[:, 0], traj[:, 2], color="#9B59B6", alpha=0.15, linewidth=1.0)
        
    all_thetas = np.array(all_thetas)
    median_theta = np.median(all_thetas, axis=0)
    q25 = np.percentile(all_thetas, 25, axis=0)
    q75 = np.percentile(all_thetas, 75, axis=0)
    
    # 真实物理世界轨迹
    real_theta = np.arctan2(real_traj[:, 1], real_traj[:, 0])
    ax1.plot(time_steps, real_theta, label="Real World Physics Outcome", color="#C0392B", linewidth=2.8, zorder=5)
    ax1.plot(time_steps, median_theta, label="Dream Ensemble Median", color="#2C3E50", linestyle="--", linewidth=2.2, zorder=4)
    ax1.fill_between(time_steps, q25, q75, color="#3498DB", alpha=0.35, label="Interquartile Prediction Cloud (25%-75%)", zorder=3)
    
    ax1.set_title(f"Sample Model Latent Imagination: {num_universes} Parallel Dreams\n(Purely Generated by White Noise Sampling epsilon ~ N(0, I))", 
                  fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Imagined Time Horizon (Steps)", fontsize=11)
    ax1.set_ylabel("Angle theta (rad)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    
    # 相空间轨迹云 (cos(theta), omega)
    ax2.scatter([parallel_trajs[0][0, 0]], [parallel_trajs[0][0, 2]], color="#E74C3C", s=100, zorder=10, label="Initial State s0")
    ax2.plot(real_traj[:, 0], real_traj[:, 2], label="Real Environment Path", color="#C0392B", linewidth=2.8, zorder=5)
    ax2.set_title("Phase Space Trajectory Fan: (cos theta, omega)\n(Sample Model Branching Distribution Cloud)", 
                  fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("cos(theta)", fontsize=11)
    ax2.set_ylabel("Angular Velocity omega (rad/s)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mpc_control_comparison(mpc_log: dict, random_log: dict, save_path: str):
    """图 4：基于脑内做梦推演的 MPC 规划控制表现对比"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    t_mpc = np.arange(len(mpc_log["thetas"]))
    t_rnd = np.arange(len(random_log["thetas"]))
    
    # 1. 角度对比 (向 0 竖直倒立收敛)
    ax1.plot(t_mpc, mpc_log["thetas"], label="World Model MPC Planning", color="#27AE60", linewidth=2.2)
    ax1.plot(t_rnd, random_log["thetas"], label="Random Action Baseline", color="#7F8C8D", linestyle="--", linewidth=1.6)
    ax1.axhline(0.0, color="#E74C3C", linestyle=":", label="Target Upright Angle (0 rad)")
    ax1.set_title("Angle Stabilization\n(Swing-up & Inverted Balance)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Execution Steps", fontsize=10)
    ax1.set_ylabel("Angle theta (rad)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95)
    
    # 2. 角速度控制
    ax2.plot(t_mpc, mpc_log["omegas"], label="World Model MPC Planning", color="#2980B9", linewidth=2.2)
    ax2.plot(t_rnd, random_log["omegas"], label="Random Action Baseline", color="#7F8C8D", linestyle="--", linewidth=1.6)
    ax2.axhline(0.0, color="#E74C3C", linestyle=":", label="Zero Velocity Target")
    ax2.set_title("Angular Velocity Damping\n(Kinetic Energy Dissipation)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Execution Steps", fontsize=10)
    ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.95)
    
    # 3. 累积回报 (Cumulative Return)
    t_rew_mpc = np.arange(len(mpc_log["rewards"]))
    t_rew_rnd = np.arange(len(random_log["rewards"]))
    ax3.plot(t_rew_mpc, np.cumsum(mpc_log["rewards"]), label=f"MPC Total: {np.sum(mpc_log['rewards']):.1f}", color="#8E44AD", linewidth=2.4)
    ax3.plot(t_rew_rnd, np.cumsum(random_log["rewards"]), label=f"Random Total: {np.sum(random_log['rewards']):.1f}", color="#7F8C8D", linestyle="--", linewidth=1.8)
    ax3.set_title("Cumulative Return Comparison\n(Higher is Better)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Execution Steps", fontsize=10)
    ax3.set_ylabel("Cumulative Sum of Rewards", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower left", framealpha=0.95)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_world_model_master_dashboard(train_history: dict, real_traj: dict, dream_traj: dict, 
                                      parallel_trajs: list, mpc_log: dict, save_path: str):
    """图 5：深度世界模型全生命周期综合决策看板 (出版级 Master Dashboard)"""
    fig = plt.figure(figsize=(20, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.20)
    
    # --- Panel A: 训练收敛与损失演化 ---
    ax_a = fig.add_subplot(gs[0, 0])
    steps = train_history.get("steps", np.arange(len(train_history["total_loss"])))
    ax_a.plot(steps, train_history["total_loss"], label="Total Joint Loss", color="#2C3E50", linewidth=2.2)
    ax_a.plot(steps, train_history["nll_loss"], label="Dynamics Gaussian NLL", color="#E74C3C", linestyle="--", linewidth=1.8)
    ax_a.plot(steps, train_history["reward_loss"], label="Reward Predictor MSE", color="#27AE60", linestyle=":", linewidth=2.0)
    ax_a.set_title("Panel A | World Model Neural Training Dynamics\n(Gaussian NLL + Reward MSE Optimization)", 
                   fontsize=13, fontweight='bold', pad=10)
    ax_a.set_xlabel("Optimization Steps", fontsize=11)
    ax_a.set_ylabel("Loss Magnitude", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    ax_a.legend(loc="upper right", framealpha=0.95, fontsize=10)
    
    # --- Panel B: 脑内做梦 vs 真实物理单轨高保真验证 ---
    ax_b = fig.add_subplot(gs[0, 1])
    time_steps = np.arange(len(real_traj["states"]))
    real_theta = np.arctan2(real_traj["states"][:, 1], real_traj["states"][:, 0])
    dream_theta = np.arctan2(dream_traj["states"][:, 1], dream_traj["states"][:, 0])
    dream_mu_theta = np.arctan2(dream_traj["mus"][:, 1], dream_traj["mus"][:, 0])
    dream_std_theta = np.linalg.norm(dream_traj["stds"][:, :2], axis=1)
    
    steps_m = min(len(time_steps)-1, len(dream_mu_theta))
    ax_b.plot(time_steps, real_theta, label="Real World Physics Path", color="#2C3E50", linewidth=2.6)
    ax_b.plot(time_steps, dream_theta, label="Neural Dream Sample", color="#E74C3C", linestyle="-", linewidth=2.0)
    ax_b.fill_between(time_steps[1:steps_m+1], 
                     dream_mu_theta[:steps_m] - dream_std_theta[:steps_m],
                     dream_mu_theta[:steps_m] + dream_std_theta[:steps_m],
                     color="#E74C3C", alpha=0.25, label="Model Uncertainty Band (+-1 sigma)")
    ax_b.set_title("Panel B | Autoregressive Rollout: Real Physics vs Neural Imagination\n(Compounding Rollout over 50 Continuous Steps)", 
                   fontsize=13, fontweight='bold', pad=10)
    ax_b.set_xlabel("Time Horizon Steps (dt = 0.05s)", fontsize=11)
    ax_b.set_ylabel("Pendulum Angle theta (rad)", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.6)
    ax_b.legend(loc="upper right", framealpha=0.95, fontsize=10)
    
    # --- Panel C: 样本模型核心机制——多模态平行宇宙发散云 ---
    ax_c = fig.add_subplot(gs[1, 0])
    all_thetas = []
    for traj in parallel_trajs:
        th = np.arctan2(traj[:, 1], traj[:, 0])
        all_thetas.append(th)
        ax_c.plot(time_steps, th, color="#3498DB", alpha=0.18, linewidth=1.0)
    all_thetas = np.array(all_thetas)
    med_th = np.median(all_thetas, axis=0)
    q25 = np.percentile(all_thetas, 25, axis=0)
    q75 = np.percentile(all_thetas, 75, axis=0)
    ax_c.plot(time_steps, real_theta, label="Real Physics Outcome", color="#C0392B", linewidth=2.6, zorder=5)
    ax_c.plot(time_steps, med_th, label="Dream Ensemble Median", color="#2C3E50", linestyle="--", linewidth=2.2, zorder=4)
    ax_c.fill_between(time_steps, q25, q75, color="#3498DB", alpha=0.35, label="Interquartile Distribution Fan (25%-75%)", zorder=3)
    ax_c.set_title(f"Panel C | Sample Model Essence: {len(parallel_trajs)} Parallel Universes Fan\n(Probability Representation via White Noise Injection epsilon ~ N(0, I))", 
                   fontsize=13, fontweight='bold', pad=10)
    ax_c.set_xlabel("Imagined Time Horizon (Steps)", fontsize=11)
    ax_c.set_ylabel("Angle theta (rad)", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.6)
    ax_c.legend(loc="upper right", framealpha=0.95, fontsize=10)
    
    # --- Panel D: 基于脑内推演的闭环 MPC 规划收益 ---
    ax_d = fig.add_subplot(gs[1, 1])
    t_mpc = np.arange(len(mpc_log["thetas"]))
    ax_d.plot(t_mpc, mpc_log["thetas"], label="Angle theta (Stabilizes near 0)", color="#27AE60", linewidth=2.4)
    ax_d.plot(t_mpc, mpc_log["omegas"], label="Angular Velocity omega", color="#2980B9", linestyle="--", linewidth=2.0)
    ax_d.axhline(0.0, color="#E74C3C", linestyle=":", label="Goal Equilibrium (0, 0)")
    ax_d.set_title("Panel D | Closed-Loop Decision Making: Imagination-Driven MPC\n(Planning Action Sequences Exclusively in Neural Dream)", 
                   fontsize=13, fontweight='bold', pad=10)
    ax_d.set_xlabel("Environment Interaction Steps", fontsize=11)
    ax_d.set_ylabel("Physical State Quantities", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.6)
    ax_d.legend(loc="upper right", framealpha=0.95, fontsize=10)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

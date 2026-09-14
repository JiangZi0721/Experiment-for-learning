"""
OpenAI Gym 真实物理环境实证与世界模型做梦推演可视化展示套件
Gymnasium Pendulum-v1: 连续非线性物理摆真实环境渲染、GIF 生成与大盘展示
"""
import os
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from src.deep_world_model import (
    StochasticPendulumEnv,
    DeepWorldModel,
    TransitionReplayBuffer,
    WorldModelTrainer,
    DreamSimulator
)

def get_or_train_world_model(checkpoint_path: str, device: str = "cuda") -> DeepWorldModel:
    """加载已有模型权重或快速预训练并持久化"""
    model = DeepWorldModel(state_dim=3, action_dim=1, hidden_dim=128, device=device)
    
    if os.path.exists(checkpoint_path):
        print(f">>> [CheckPoint] 检测到已有权重: {checkpoint_path}，正在加载...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model
        
    print(">>> [CheckPoint] 未检测到预训练权重，开始在物理环境中采集数据并训练...")
    # 采集 10,000 条转换样本
    env = StochasticPendulumEnv(noise_std=0.25)
    buffer = TransitionReplayBuffer(capacity=20000)
    
    obs = env.reset()
    for _ in range(10000):
        action = float(np.random.uniform(-2.0, 2.0))
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs)
        obs = next_obs
        
    trainer = WorldModelTrainer(model, lr=1e-3)
    for step in range(1, 3001):
        batch = buffer.sample(batch_size=128)
        loss_dict = trainer.train_step(batch)
        if step % 1000 == 0 or step == 3000:
            print(f"  Train Step [{step}/3000] | Total Loss: {loss_dict['total_loss']:.4f} | NLL: {loss_dict['nll_loss']:.4f}")
            
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f">>> [CheckPoint] 模型训练完成并成功保存至: {checkpoint_path}")
    model.eval()
    return model

def add_hud_overlay(frame: np.ndarray, title: str, step: int, total_steps: int,
                    theta: float, theta_dot: float, action: float, reward: float,
                    color_theme: str = "cyan") -> np.ndarray:
    """在 Gym 渲染帧顶部添加高质量信息 HUD 抬头显示"""
    img = Image.fromarray(frame).convert("RGBA")
    w, h = img.size
    
    # 创建透明覆盖层
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 顶部半透明背景条
    banner_height = 80
    draw.rectangle([(0, 0), (w, banner_height)], fill=(15, 23, 42, 220))
    draw.line([(0, banner_height), (w, banner_height)], fill=(56, 189, 248, 180), width=2)
    
    theme_colors = {
        "cyan": (56, 189, 248),
        "emerald": (52, 211, 153),
        "rose": (251, 113, 133),
        "amber": (251, 191, 36),
        "purple": (192, 132, 252)
    }
    tc = theme_colors.get(color_theme, (255, 255, 255))
    
    theta_deg = math.degrees(theta)
    
    # 绘制标题与步骤
    draw.text((15, 8), f"{title}", fill=tc)
    draw.text((w - 130, 8), f"Step: {step:02d}/{total_steps}", fill=(226, 232, 240))
    
    # 绘制状态参数
    info_str1 = f"Angle θ: {theta:+.3f} rad ({theta_deg:+.1f}°) | Speed: {theta_dot:+.3f} rad/s"
    info_str2 = f"Control Torque u: {action:+.3f} N·m | Step Reward: {reward:+.3f}"
    
    draw.text((15, 32), info_str1, fill=(241, 245, 249))
    draw.text((15, 54), info_str2, fill=(148, 163, 184))
    
    # 底部指示条：目标直立提示
    is_upright = abs(theta) < 0.35
    status_text = "STATUS: UPRIGHT BALANCED" if is_upright else "STATUS: PUMPING ENERGY"
    status_color = (52, 211, 153, 230) if is_upright else (251, 191, 36, 230)
    
    draw.rectangle([(w - 180, h - 35), (w - 10, h - 10)], fill=(15, 23, 42, 200))
    draw.text((w - 170, h - 30), status_text, fill=status_color)
    
    combined = Image.alpha_composite(img, overlay)
    return np.array(combined.convert("RGB"))

def stitch_frames_side_by_side(frame_left: np.ndarray, frame_right: np.ndarray,
                               label_left: str, label_right: str) -> np.ndarray:
    """横向拼接两路渲染帧 (如真实物理 vs 做梦推演)"""
    h_l, w_l, _ = frame_left.shape
    h_r, w_r, _ = frame_right.shape
    assert h_l == h_r, "Frame heights must match"
    
    stitched = np.concatenate([frame_left, frame_right], axis=1)
    img = Image.fromarray(stitched)
    draw = ImageDraw.Draw(img)
    
    # 顶部标签条
    draw.rectangle([(0, 0), (w_l, 30)], fill=(30, 41, 59, 230))
    draw.rectangle([(w_l, 0), (w_l + w_r, 30)], fill=(15, 23, 42, 230))
    draw.text((w_l // 2 - 80, 7), label_left, fill=(56, 189, 248))
    draw.text((w_l + w_r // 2 - 80, 7), label_right, fill=(251, 146, 60))
    
    # 分隔线
    draw.line([(w_l, 0), (w_l, h_l)], fill=(255, 255, 255), width=3)
    
    return np.array(img)

def run_gym_simulation(world_model: DeepWorldModel, total_steps: int = 140, seed: int = 42):
    """
    运行基于 OpenAI Gym (Gymnasium) Pendulum-v1 的真实环境实证对比
    包含三条完整轨迹：
    1. 随机策略基线 (Random Baseline)
    2. 深度世界模型 MPC 闭环控制 (World Model MPC)
    3. 世界模型脑内白日梦推演 (Brain Dream Rollout)
    """
    env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    dream_env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    simulator = DreamSimulator(world_model)
    
    # 初始状态固定在正下方垂悬点 (theta = -pi, speed = 0)
    init_theta = -math.pi
    init_speed = 0.0
    
    # -------------------------------------------------------------
    # 1. 运行随机策略基线 (Random Policy)
    # -------------------------------------------------------------
    print(">>> 正在运行 OpenAI Gym 随机探索基线...")
    env.reset(seed=seed)
    env.unwrapped.state = np.array([init_theta, init_speed])
    
    random_frames = []
    random_thetas = [init_theta]
    random_speeds = [init_speed]
    random_actions = []
    random_rewards = []
    
    for t in range(total_steps):
        th = env.unwrapped.state[0]
        th_dot = env.unwrapped.state[1]
        raw_frame = env.render()
        
        act = float(np.random.uniform(-2.0, 2.0))
        _, rew, _, _, _ = env.step(np.array([act], dtype=np.float32))
        
        hud_frame = add_hud_overlay(
            raw_frame, "OpenAI Gym [Random Baseline]", t, total_steps,
            th, th_dot, act, rew, color_theme="rose"
        )
        random_frames.append(hud_frame)
        random_actions.append(act)
        random_rewards.append(rew)
        random_thetas.append(env.unwrapped.state[0])
        random_speeds.append(env.unwrapped.state[1])
        
    # -------------------------------------------------------------
    # 2. 运行世界模型 MPC 闭环控制 (World Model MPC)
    # -------------------------------------------------------------
    print(">>> 正在运行基于深度世界模型脑内推演的 MPC 闭环摇摆控制...")
    env.reset(seed=seed)
    env.unwrapped.state = np.array([init_theta, init_speed])
    
    mpc_frames = []
    mpc_raw_frames = []
    mpc_thetas = [init_theta]
    mpc_speeds = [init_speed]
    mpc_actions = []
    mpc_rewards = []
    
    for t in range(total_steps):
        th = env.unwrapped.state[0]
        th_dot = env.unwrapped.state[1]
        raw_frame = env.render()
        mpc_raw_frames.append(raw_frame)
        
        # 观测状态向量 [cos(th), sin(th), th_dot]
        obs = np.array([math.cos(th), math.sin(th), th_dot], dtype=np.float32)
        
        # 世界模型脑内做梦 MPC 规划最优动作
        act = simulator.plan_mpc(obs, horizon=14, num_candidates=100)
        
        _, rew, _, _, _ = env.step(np.array([act], dtype=np.float32))
        
        hud_frame = add_hud_overlay(
            raw_frame, "OpenAI Gym [World Model MPC Controller]", t, total_steps,
            th, th_dot, act, rew, color_theme="cyan"
        )
        mpc_frames.append(hud_frame)
        mpc_actions.append(act)
        mpc_rewards.append(rew)
        mpc_thetas.append(env.unwrapped.state[0])
        mpc_speeds.append(env.unwrapped.state[1])
        
    # -------------------------------------------------------------
    # 3. 运行世界模型“脑内做梦”自回归仿真帧渲染 (Brain Dream Rollout)
    # -------------------------------------------------------------
    print(">>> 正在渲染世界模型脑内白日梦推演视角...")
    # 从初始状态开始，输入 MPC 相同的动作序列，纯靠世界模型自回归自举做梦
    init_obs = np.array([math.cos(init_theta), math.sin(init_theta), init_speed], dtype=np.float32)
    dream_res = simulator.rollout_dream(init_obs, mpc_actions, deterministic=True)
    dream_states = dream_res["states"]  # shape (total_steps + 1, 3)
    
    dream_frames = []
    dream_raw_frames = []
    for t in range(total_steps):
        # 提取世界模型想象出来的角度与角速度
        cos_t, sin_t, th_dot_t = dream_states[t]
        th_t = math.atan2(sin_t, cos_t)
        
        # 将想象出的状态灌入影子环境以渲染图像
        dream_env.reset()
        dream_env.unwrapped.state = np.array([th_t, th_dot_t])
        raw_dream_frame = dream_env.render()
        dream_raw_frames.append(raw_dream_frame)
        
        act = mpc_actions[t]
        rew = dream_res["rewards"][t]
        
        hud_dream_frame = add_hud_overlay(
            raw_dream_frame, "World Model [Latent Dream Imagination]", t, total_steps,
            th_t, th_dot_t, act, rew, color_theme="amber"
        )
        dream_frames.append(hud_dream_frame)
        
    env.close()
    dream_env.close()
    
    return {
        "random": {
            "frames": random_frames,
            "thetas": np.array(random_thetas),
            "speeds": np.array(random_speeds),
            "actions": np.array(random_actions),
            "rewards": np.array(random_rewards),
            "total_return": float(np.sum(random_rewards))
        },
        "mpc": {
            "frames": mpc_frames,
            "raw_frames": mpc_raw_frames,
            "thetas": np.array(mpc_thetas),
            "speeds": np.array(mpc_speeds),
            "actions": np.array(mpc_actions),
            "rewards": np.array(mpc_rewards),
            "total_return": float(np.sum(mpc_rewards))
        },
        "dream": {
            "frames": dream_frames,
            "raw_frames": dream_raw_frames,
            "states": dream_states,
            "rewards": dream_res["rewards"]
        }
    }

def generate_gym_gifs(sim_results: dict, output_dir: str):
    """生成三套高画质动态 GIF 动画资产"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 深度世界模型 MPC 闭环控制全过程 (Gym Pendulum-v1)
    mpc_gif_path = os.path.join(output_dir, "gym_pendulum_mpc_control.gif")
    print(f">>> 正在保存 MPC 闭环动画: {mpc_gif_path}")
    imageio.mimsave(mpc_gif_path, sim_results["mpc"]["frames"], fps=15, loop=0)
    
    # 2. 随机基线 vs 世界模型 MPC 对比动画 (Side-by-Side)
    comp_gif_path = os.path.join(output_dir, "gym_comparison_random_vs_mpc.gif")
    print(f">>> 正在保存对比动画: {comp_gif_path}")
    comp_frames = []
    for f_rand, f_mpc in zip(sim_results["random"]["frames"], sim_results["mpc"]["frames"]):
        stitched = stitch_frames_side_by_side(
            f_rand, f_mpc, "OpenAI Gym [Random Policy]", "OpenAI Gym [World Model MPC]"
        )
        comp_frames.append(stitched)
    imageio.mimsave(comp_gif_path, comp_frames, fps=15, loop=0)
    
    # 3. 真实物理世界 vs 脑内白日梦推演动画 (Side-by-Side)
    dream_gif_path = os.path.join(output_dir, "gym_real_vs_dream.gif")
    print(f">>> 正在保存真实与做梦对比动画: {dream_gif_path}")
    dream_comp_frames = []
    for f_real, f_dream in zip(sim_results["mpc"]["frames"], sim_results["dream"]["frames"]):
        stitched = stitch_frames_side_by_side(
            f_real, f_dream, "Real Gym Physical World", "World Model Brain Dream"
        )
        dream_comp_frames.append(stitched)
    imageio.mimsave(dream_gif_path, dream_comp_frames, fps=15, loop=0)

def plot_gym_pendulum_master_dashboard(sim_results: dict, output_path: str):
    """绘制出版级 300 DPI 全景 OpenAI Gym 物理实证与世界模型看板"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 设置现代化暗色主题
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig = plt.figure(figsize=(20, 16), facecolor="#0b0f19")
    gs = fig.add_gridspec(4, 8, height_ratios=[1.0, 1.0, 1.0, 1.2], hspace=0.35, wspace=0.15)
    
    # 关键时间步采样 (根据实际步数动态均匀采样 8 帧胶片)
    total_steps = len(sim_results["mpc"]["frames"])
    key_steps = np.linspace(0, total_steps - 1, 8, dtype=int).tolist()
    n_cols = len(key_steps)
    
    # =========================================================================
    # Row 1: 真实 OpenAI Gym 环境下的深度世界模型 MPC 倒立摆控制胶片 (Filmstrip)
    # =========================================================================
    for idx, t in enumerate(key_steps):
        ax = fig.add_subplot(gs[0, idx])
        frame = sim_results["mpc"]["raw_frames"][t]
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        th = sim_results["mpc"]["thetas"][t]
        u = sim_results["mpc"]["actions"][t]
        deg = math.degrees(th)
        ax.set_title(f"t={t}\nθ={deg:+.0f}° | u={u:+.1f}", fontsize=10, color="#38bdf8", pad=4)
        for spine in ax.spines.values():
            spine.set_color("#38bdf8" if abs(th) < 0.35 else "#475569")
            spine.set_linewidth(2 if abs(th) < 0.35 else 1)
            
    fig.text(0.015, 0.88, "【胶片一】OpenAI Gym 真实物理环境\n[深度世界模型 MPC 闭环控制]\n(蓄力摆动冲顶并稳定平衡)", 
             fontsize=12, color="#38bdf8", weight="bold", va="center", ha="left")

    # =========================================================================
    # Row 2: 真实物理世界 (上) vs 世界模型脑内白日梦推演 (下) 逐帧视觉保真度对比
    # =========================================================================
    for idx, t in enumerate(key_steps):
        ax = fig.add_subplot(gs[1, idx])
        dream_frame = sim_results["dream"]["raw_frames"][t]
        ax.imshow(dream_frame)
        ax.set_xticks([])
        ax.set_yticks([])
        cos_t, sin_t, _ = sim_results["dream"]["states"][t]
        th_dream = math.atan2(sin_t, cos_t)
        deg_dream = math.degrees(th_dream)
        ax.set_title(f"t={t} (脑内做梦)\nθ_hat={deg_dream:+.0f}°", fontsize=10, color="#fbbf24", pad=4)
        for spine in ax.spines.values():
            spine.set_color("#fbbf24")
            spine.set_linewidth(1.5)
            
    fig.text(0.015, 0.63, "【胶片二】世界模型脑内白日梦视角\n[Latent Dream Imagination]\n(脱离现实纯靠神经网络推演)", 
             fontsize=12, color="#fbbf24", weight="bold", va="center", ha="left")

    # =========================================================================
    # Row 3: 随机探索基线胶片 (失控剧烈翻滚或停滞在下方)
    # =========================================================================
    for idx, t in enumerate(key_steps):
        ax = fig.add_subplot(gs[2, idx])
        rand_frame = sim_results["random"]["frames"][t]
        ax.imshow(rand_frame)
        ax.set_xticks([])
        ax.set_yticks([])
        th = sim_results["random"]["thetas"][t]
        deg = math.degrees(th)
        ax.set_title(f"t={t}\nθ={deg:+.0f}°", fontsize=10, color="#fb7185", pad=4)
        for spine in ax.spines.values():
            spine.set_color("#fb7185")
            spine.set_linewidth(1)
            
    fig.text(0.015, 0.38, "【胶片三】随机策略探索基线\n[Random Baseline]\n(无序震荡，无法直立平衡)", 
             fontsize=12, color="#fb7185", weight="bold", va="center", ha="left")

    # =========================================================================
    # Row 4: 定量动力学指标比对曲线 (角度、角速度、力矩、累计回报)
    # =========================================================================
    steps_arr = np.arange(len(sim_results["mpc"]["actions"]))
    
    # 1. 角度对比
    ax_th = fig.add_subplot(gs[3, 0:2], facecolor="#1e293b")
    ax_th.plot(steps_arr, sim_results["mpc"]["thetas"][:len(steps_arr)], color="#38bdf8", lw=2.2, label="World Model MPC")
    ax_th.plot(steps_arr, sim_results["random"]["thetas"][:len(steps_arr)], color="#fb7185", lw=1.5, ls="--", label="Random Baseline")
    ax_th.axhline(0, color="#4ade80", ls=":", lw=1.5, label="Upright Goal (θ=0)")
    ax_th.set_title("摆杆角度 θ 演进曲线", color="#f8fafc", fontsize=12)
    ax_th.set_xlabel("Time Step t", color="#94a3b8")
    ax_th.set_ylabel("Angle θ (rad)", color="#94a3b8")
    ax_th.tick_params(colors="#cbd5e1")
    ax_th.legend(fontsize=9, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    ax_th.grid(True, color="#334155", alpha=0.5)

    # 2. 角速度对比
    ax_sp = fig.add_subplot(gs[3, 2:4], facecolor="#1e293b")
    ax_sp.plot(steps_arr, sim_results["mpc"]["speeds"][:len(steps_arr)], color="#38bdf8", lw=2.2, label="World Model MPC")
    ax_sp.plot(steps_arr, sim_results["random"]["speeds"][:len(steps_arr)], color="#fb7185", lw=1.5, ls="--", label="Random Baseline")
    ax_sp.axhline(0, color="#4ade80", ls=":", lw=1.5)
    ax_sp.set_title("角速度 dθ/dt 阻尼稳定曲线", color="#f8fafc", fontsize=12)
    ax_sp.set_xlabel("Time Step t", color="#94a3b8")
    ax_sp.set_ylabel("Angular Velocity (rad/s)", color="#94a3b8")
    ax_sp.tick_params(colors="#cbd5e1")
    ax_sp.grid(True, color="#334155", alpha=0.5)

    # 3. 控制力矩 u 序列
    ax_u = fig.add_subplot(gs[3, 4:6], facecolor="#1e293b")
    ax_u.step(steps_arr, sim_results["mpc"]["actions"], color="#c084fc", lw=2.0, where="post", label="MPC Torque u")
    ax_u.axhline(0, color="#64748b", ls="-", lw=1.0)
    ax_u.set_title("控制力矩输入 u(t) (反向蓄力与稳摆)", color="#f8fafc", fontsize=12)
    ax_u.set_xlabel("Time Step t", color="#94a3b8")
    ax_u.set_ylabel("Torque (N·m)", color="#94a3b8")
    ax_u.set_ylim(-2.2, 2.2)
    ax_u.tick_params(colors="#cbd5e1")
    ax_u.legend(fontsize=9, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    ax_u.grid(True, color="#334155", alpha=0.5)

    # 4. 累积回报曲线
    ax_rew = fig.add_subplot(gs[3, 6:8], facecolor="#1e293b")
    cum_mpc = np.cumsum(sim_results["mpc"]["rewards"])
    cum_rand = np.cumsum(sim_results["random"]["rewards"])
    ax_rew.plot(steps_arr, cum_mpc, color="#38bdf8", lw=2.5, label=f"MPC: {cum_mpc[-1]:.1f}")
    ax_rew.plot(steps_arr, cum_rand, color="#fb7185", lw=2.0, ls="--", label=f"Random: {cum_rand[-1]:.1f}")
    ax_rew.set_title("累积回报 (Cumulative Return)", color="#f8fafc", fontsize=12)
    ax_rew.set_xlabel("Time Step t", color="#94a3b8")
    ax_rew.set_ylabel("Return", color="#94a3b8")
    ax_rew.tick_params(colors="#cbd5e1")
    ax_rew.legend(fontsize=9, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    ax_rew.grid(True, color="#334155", alpha=0.5)

    # 主标题与落款
    fig.suptitle("OpenAI Gym (Gymnasium) 真实物理环境实证：深度世界模型闭环控制与做梦仿真全景观摩",
                 fontsize=18, color="#f8fafc", weight="bold", y=0.98)
    
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f">>> [Dashboard] 全景实证看板已保存至: {output_path}")


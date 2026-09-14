"""
实验八: 使用 OpenAI Gym (Gymnasium) 真实物理环境展示深度世界模型与做梦推演实验结果
运行环境: Gymnasium Pendulum-v1 (连续非线性倒立摆)

实验目标:
1. 在标准 OpenAI Gym 环境中真实步进，调用训练好的深度世界模型 MPC 控制器进行摆杆倒立控制；
2. 逐帧捕获 Gym 真实渲染输出 (rgb_array) 与状态变量，生成 HUD 抬头显示；
3. 将世界模型“脑内做梦”预测出的虚构状态灌入影子 Gym 环境，同步生成世界模型眼中的“白日梦现实”；
4. 输出三大动态 GIF 动画资产与出版级全景大盘：
   - images/gym_pendulum_mpc_control.gif (MPC 闭环倒立摆全过程)
   - images/gym_comparison_random_vs_mpc.gif (随机基线 vs MPC 规划横向对比)
   - images/gym_real_vs_dream.gif (真实物理世界 vs 脑内做梦推演对比)
   - images/gym_pendulum_master_showcase.png (300 DPI 综合全景看板)
"""
import os
import sys
import time
import shutil
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

from src.gym_world_model_showcase import (
    get_or_train_world_model,
    run_gym_simulation,
    generate_gym_gifs,
    plot_gym_pendulum_master_dashboard
)

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    log_path = os.path.join(PROJECT_ROOT, "logs", "gym_world_model_showcase.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 80)
    print("  强化学习实验八：使用 OpenAI Gym 真实物理环境展示深度世界模型与仿真结果")
    print("  [Gymnasium Pendulum-v1 | 逐帧真实渲染 | 脑内做梦投影 | 三大动态 GIF 动画]")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> [系统计算引擎] 计算设备: {device.upper()} | PyTorch: {torch.__version__}")
    
    # 1. 加载或训练深度世界模型
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "world_model.pt")
    t0 = time.time()
    world_model = get_or_train_world_model(checkpoint_path, device=device)
    t_load = time.time() - t0
    print(f">>> 世界模型就绪，耗时: {t_load:.2f} 秒")
    
    # 2. 在 OpenAI Gym 中执行真实仿真并记录帧
    print("\n" + "-" * 80)
    print(">>> 【阶段一：OpenAI Gym 环境步进与做梦推演同步渲染】...")
    print("-" * 80)
    t0 = time.time()
    total_sim_steps = 140
    sim_results = run_gym_simulation(world_model, total_steps=total_sim_steps, seed=42)
    t_sim = time.time() - t0
    print(f">>> OpenAI Gym 仿真渲染完成！{total_sim_steps} 步执行总用时: {t_sim:.2f} 秒")
    
    # 3. 统计定量对比结果
    ret_rand = sim_results["random"]["total_return"]
    ret_mpc = sim_results["mpc"]["total_return"]
    gain = ((ret_mpc - ret_rand) / abs(ret_rand)) * 100
    
    # 计算直立时间比率 (|theta| < 0.35 rad 约 20 度)
    mpc_thetas = sim_results["mpc"]["thetas"]
    upright_steps = np.sum(np.abs(mpc_thetas) < 0.35)
    upright_ratio = (upright_steps / len(mpc_thetas)) * 100
    
    rand_thetas = sim_results["random"]["thetas"]
    rand_upright = np.sum(np.abs(rand_thetas) < 0.35)
    rand_upright_ratio = (rand_upright / len(rand_thetas)) * 100
    
    print("\n" + "-" * 80)
    print(">>> 【阶段二：OpenAI Gym 真实控制指标对决】")
    print("-" * 80)
    print(f"  * 随机探索基线 {total_sim_steps} 步累积总回报: {ret_rand:.2f}")
    print(f"  * 世界模型 MPC 规划 {total_sim_steps} 步总回报: {ret_mpc:.2f} (性能飞跃: {gain:+.2f}%)")
    print(f"  * 直立维持稳定时间比例: MPC 控制器 {upright_ratio:.1f}% vs 随机基线 {rand_upright_ratio:.1f}%")
    
    # 4. 生成高画质动态 GIF 动画
    print("\n" + "-" * 80)
    print(">>> 【阶段三：生成三大 OpenAI Gym 动态动画资产 (GIF)】...")
    print("-" * 80)
    images_dir = os.path.join(PROJECT_ROOT, "images")
    generate_gym_gifs(sim_results, images_dir)
    print("  * 1. 独立闭环控制动画: images/gym_pendulum_mpc_control.gif")
    print("  * 2. 随机探索 vs MPC 对比动画: images/gym_comparison_random_vs_mpc.gif")
    print("  * 3. 真实物理 vs 脑内做梦对比动画: images/gym_real_vs_dream.gif")
    
    # 5. 绘制综合全景 300 DPI 大盘看板
    print("\n" + "-" * 80)
    print(">>> 【阶段四：绘制出版级综合全景胶片看板 (Master Dashboard)】...")
    print("-" * 80)
    dashboard_path = os.path.join(images_dir, "gym_pendulum_master_showcase.png")
    plot_gym_pendulum_master_dashboard(sim_results, dashboard_path)
    
    # 6. 将生成的重要图像和 GIF 同步到 artifact 目录以支持实时可视化
    artifact_dir = r"C:\Users\zero\.gemini\antigravity-cli\brain\fd79c765-3b16-4002-9ebb-3aa92e34085e"
    if os.path.exists(artifact_dir):
        for fname in [
            "gym_pendulum_mpc_control.gif",
            "gym_comparison_random_vs_mpc.gif",
            "gym_real_vs_dream.gif",
            "gym_pendulum_master_showcase.png"
        ]:
            src_f = os.path.join(images_dir, fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(artifact_dir, fname))
        print(">>> 资产已同步至对话 Artifacts 缓存！")
        
    print("\n" + "=" * 80)
    print("  OpenAI Gym 真实环境实证全套资产生成圆满成功！")
    print("=" * 80)

if __name__ == "__main__":
    main()

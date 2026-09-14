"""
强化学习基础实证项目统一总控入口 (Main Orchestrator for RL Foundations)
一站式运行动态规划、无模型时序差分、Gamma 对比、复杂网格透视以及深度强化学习 DQN 实验。

使用方法:
    python main.py --all       # 依次运行全部 5 大核心实验并重新生成全部图表与日志
    python main.py --dp        # 运行实验一: 策略迭代 (PI) 与 价值迭代 (VI)
    python main.py --mf        # 运行实验二: 无模型三剑客 TD(0)、SARSA 与 Q-Learning
    python main.py --gamma     # 运行实验三: 折扣因子 Gamma 严格对比实验
    python main.py --complex   # 运行实验四: 复杂多奖励网格与 6 大时间步深度透视
    python main.py --dqn       # 运行实验五: 深度 Q 网络 (DQN) 白盒实证与理论真值对比
"""
import os
import sys
import argparse
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

from experiments.exp1_dynamic_programming import run as run_exp1
from experiments.exp2_model_free import run as run_exp2
from experiments.exp3_gamma_comparison import run as run_exp3
from experiments.exp4_complex_gridworld import run as run_exp4
from experiments.exp5_dqn import run as run_exp5
from experiments.exp6_distribution_vs_sample_model import main as run_exp6
from experiments.exp7_deep_world_model_simulation import run as run_exp7
from experiments.exp8_gym_world_model_showcase import main as run_exp8

def print_banner():
    banner = r"""
================================================================================
          强化学习核心算法全透视实证项目 (RL Foundations Master Suite)
  - 理论体系: 动态规划 (PI/VI) | 表格时序差分 | DQN | 分布vs样本模型 | 深度世界模型 & Gym
  - 核心特色: 100% 真实仿真数据 | 白盒算术级回溯 | 矢量热力图 | 动态渲染 GIF
================================================================================
"""
    print(banner)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="强化学习实验统一总控平台")
    parser.add_argument("--all", action="store_true", help="连续运行全部 8 个实验并更新所有图表与日志")
    parser.add_argument("--dp", action="store_true", help="运行实验一: 动态规划 (策略迭代 vs 价值迭代)")
    parser.add_argument("--mf", action="store_true", help="运行实验二: 无模型时序差分 (TD(0), SARSA, Q-Learning)")
    parser.add_argument("--gamma", action="store_true", help="运行实验三: 折扣因子 Gamma 对比研究")
    parser.add_argument("--complex", action="store_true", help="运行实验四: 复杂多奖励网格与 6 大关键剧变步深度透视")
    parser.add_argument("--dqn", action="store_true", help="运行实验五: 深度 Q 网络 (DQN) 白盒实证与理论真值对比")
    parser.add_argument("--models", action="store_true", help="运行实验六: 分布模型 vs 样本模型实证 (Sutton Fig 8.7)")
    parser.add_argument("--worldmodel", action="store_true", help="运行实验七: 深度高斯世界模型 60 步做梦推演与 MPC 规划")
    parser.add_argument("--gym", action="store_true", help="运行实验八: OpenAI Gym 真实物理环境实证与动态 GIF 渲染")
    
    args = parser.parse_args()
    
    # 若未指定任何参数，默认执行 --all 并给出引导
    if not (args.all or args.dp or args.mf or args.gamma or args.complex or args.dqn or args.models or args.worldmodel or args.gym):
        print("未指定特定参数，默认执行全部实验 (--all)...\n")
        args.all = True

    if args.all or args.dp:
        print("\n" + "="*80)
        print(">>> 启动实验一: 经典动态规划对比实验 (Policy Iteration vs Value Iteration)")
        print("="*80)
        run_exp1()
        
    if args.all or args.mf:
        print("\n" + "="*80)
        print(">>> 启动实验二: 无模型三剑客对比实验 (TD(0) vs SARSA vs Q-Learning)")
        print("="*80)
        run_exp2()
        
    if args.all or args.gamma:
        print("\n" + "="*80)
        print(">>> 启动实验三: 折扣因子 Gamma 对比研究 (0.7 vs 0.9 vs 0.99 vs 1.0)")
        print("="*80)
        run_exp3()
        
    if args.all or args.complex:
        print("\n" + "="*80)
        print(">>> 启动实验四: 复杂多奖励网格实验与 6 大关键时间步白盒透视")
        print("="*80)
        run_exp4()

    if args.all or args.dqn:
        print("\n" + "="*80)
        print(">>> 启动实验五: 深度 Q 网络 (DQN) 白盒实证 (经验回放 + 目标网络)")
        print("="*80)
        run_exp5()

    if args.all or args.models:
        print("\n" + "="*80)
        print(">>> 启动实验六: 分布模型 vs 样本模型实证研究 (Sutton Fig 8.7 复现)")
        print("="*80)
        run_exp6()

    if args.all or args.worldmodel:
        print("\n" + "="*80)
        print(">>> 启动实验七: 深度世界模型做梦推演与 GPU 矢量化 MPC 控制")
        print("="*80)
        run_exp7()

    if args.all or args.gym:
        print("\n" + "="*80)
        print(">>> 启动实验八: OpenAI Gym 真实物理环境实证与三大动态 GIF 动画生成")
        print("="*80)
        run_exp8()
        
    print("\n" + "#"*80)
    print("  🎉 选定实验已全部顺利执行完成！")
    print("  - 查看核心主理论文档:  README.md")
    print("  - 查看 6 大专题目录:   01~06 独立子目录中的 README.md 与 docs/")
    print("  - 查看数值运行日志:    logs/ 目录下所有 .txt 文件")
    print("  - 查看高清图表卡片:    images/ 目录下所有 .png 图片与 .gif 动画")
    print("#"*80)

if __name__ == "__main__":
    main()

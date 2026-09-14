# -*- coding: utf-8 -*-
"""
White-box MoE Lab: 混合专家与稀疏门控路由实验室
一站式全景透视入口 (CLI & Interactive Dashboard)
"""
import sys
import os
import argparse
from pathlib import Path

# 强制在 Windows 控制台下使用 UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
    console = Console(force_terminal=True)
except ImportError:
    HAS_RICH = False
    console = None

from experiments.exp1_routing_collapse import run as run_exp1
from experiments.exp2_dense_vs_moe_vs_deepseek import run as run_exp2
from experiments.exp3_expert_specialization import run as run_exp3

BANNER = r"""
  __  __         ______   _           _     
 |  \/  |  ___  |  ____| | |    __ _ | |__  
 | |\/| | / _ \ | |__    | |   / _` || '_ \ 
 | |  | || (_) ||  __|   | |__| (_| || |_) |
 |_|  |_| \___/ |_|      |_____\__,_||_.__/ 
  White-Box Sparse Mixture of Experts Lab
"""

def print_welcome():
    if HAS_RICH:
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        desc = Text()
        desc.append("基于 PyTorch 底层手推纯白盒实现的混合专家模型 (MoE) 与稀疏门控路由实验室\n", style="bold green")
        desc.append("解构 Top-K 门控、辅助负载均衡损失、DeepSeekMoE 细粒度+共享专家与路由塌缩实证", style="dim white")
        console.print(Panel(desc, border_style="cyan"))
    else:
        print(BANNER)
        print("基于 PyTorch 底层手推纯白盒实现的混合专家模型 (MoE) 与稀疏门控路由实验室\n")

def list_experiments():
    table_data = [
        ("1", "exp1_routing_collapse", "路由塌缩消融实证 (无辅助损失垄断死锁 vs 开启辅助损失健康均衡)"),
        ("2", "exp2_dense_vs_moe_vs_deepseek", "算力对齐下 Dense vs Standard MoE vs DeepSeekMoE 效能对照"),
        ("3", "exp3_expert_specialization", "专家专业化分工热力图透视 (语义领域在各专家上的自发分工)"),
    ]
    if HAS_RICH:
        table = Table(title="🧪 实验套件清单 (Experiment Suites)", border_style="green", header_style="bold magenta")
        table.add_column("编号", justify="center", style="cyan", width=6)
        table.add_column("子实验模块", style="bold yellow", width=32)
        table.add_column("核心机理与评测重点", style="white")
        for row in table_data:
            table.add_row(*row)
        console.print(table)
    else:
        print("=== 实验套件清单 ===")
        for row in table_data:
            print(f"[{row[0]}] {row[1]}: {row[2]}")

def main():
    parser = argparse.ArgumentParser(description="White-box MoE 实验室")
    parser.add_argument("--exp", type=str, choices=["1", "2", "3", "all"], help="运行指定实验编号")
    args = parser.parse_args()

    print_welcome()
    list_experiments()

    if args.exp == "1":
        run_exp1()
    elif args.exp == "2":
        run_exp2()
    elif args.exp == "3":
        run_exp3()
    elif args.exp == "all":
        print("\n>>> 正在一键执行全部三大实验...")
        run_exp1()
        run_exp2()
        run_exp3()
    else:
        print("\n[提示] 使用方式: python main.py --exp [1|2|3|all]")

if __name__ == "__main__":
    main()

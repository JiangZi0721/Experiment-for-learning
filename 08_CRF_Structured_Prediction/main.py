# -*- coding: utf-8 -*-
"""
White-box CRF Lab: 条件随机场与结构化序列预测实验室
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

from experiments.exp1_label_bias_duel import run as run_exp1
from experiments.exp2_ner_benchmark import run as run_exp2
from experiments.exp3_transition_constraints import run as run_exp3

BANNER = r"""
   ____  ____   _____   _           _     
  / ___||  _ \ |  ___| | |    __ _ | |__  
 | |    | |_) || |_    | |   / _` || '_ \ 
 | |___ |  _ < |  _|   | |__| (_| || |_) |
  \____||_| \_\|_|     |_____\__,_||_.__/ 
  White-Box Linear-Chain CRF Full-Stack Lab
"""

def print_welcome():
    if HAS_RICH:
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        desc = Text()
        desc.append("基于 PyTorch 底层手推纯白盒实现的线性链条件随机场 (Linear-Chain CRF) 教学实战工程\n", style="bold green")
        desc.append("从物理场到马尔可夫随机场，深度解构配分函数对数空间累积、维特比动态规划解码与标注偏置破解机理", style="dim white")
        console.print(Panel(desc, border_style="cyan"))
    else:
        print(BANNER)
        print("基于 PyTorch 底层手推纯白盒实现的线性链条件随机场 (Linear-Chain CRF) 教学实战工程\n")

def list_experiments():
    table_data = [
        ("1", "exp1_label_bias_duel", "标注偏置反例实证 (MEMM 局部归一化死锁 vs CRF 全局归一化秒杀)"),
        ("2", "exp2_ner_benchmark", "命名实体识别三大模型横向基准实测 (HMM vs BiLSTM vs BiLSTM-CRF)"),
        ("3", "exp3_transition_constraints", "CRF 状态转移矩阵硬约束透视 (非法跳变自发物理阻断机制)"),
    ]
    if HAS_RICH:
        table = Table(title="🧪 实验套件清单 (Experiment Suites)", border_style="green", header_style="bold magenta")
        table.add_column("编号", justify="center", style="cyan", width=6)
        table.add_column("子实验模块", style="bold yellow", width=28)
        table.add_column("核心机理与评测重点", style="white")
        for row in table_data:
            table.add_row(*row)
        console.print(table)
    else:
        print("=== 实验套件清单 ===")
        for row in table_data:
            print(f"[{row[0]}] {row[1]}: {row[2]}")

def main():
    parser = argparse.ArgumentParser(description="White-box CRF 实验室")
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
        model_crf = run_exp2()
        run_exp3(model_crf)
    else:
        print("\n[提示] 使用方式: python main.py --exp [1|2|3|all]")

if __name__ == "__main__":
    main()

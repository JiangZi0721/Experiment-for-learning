# -*- coding: utf-8 -*-
"""
Seq2seq & Attention Lab: 序列建模与编解码架构全景实证实验室
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

BANNER = r"""
  ____                  ____                       
 / ___|   ___    __ _  |___ \  ___   ___   __ _   
 \___ \  / _ \  / _` |   __) |/ __| / _ \ / _` |  
  ___) ||  __/ | (_| |  / __/ \__ \|  __/| (_| |  
 |____/  \___|  \__, | |_____||___/ \___| \__, |  
                   |_|                        |_|   
  & Attention Mechanism Full-Stack Lab
"""

def print_welcome():
    if HAS_RICH:
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        desc = Text()
        desc.append("Seq2Seq 核心优化与架构演进（Reverse、Peeky 到 Attention）深度实证与机理透视\n", style="bold green")
        desc.append("全景探究输入逆序、解码器直连与注意力机制在数学加法与机器翻译双任务下的真实作用边界", style="dim white")
        console.print(Panel(desc, border_style="cyan"))
    else:
        print(BANNER)
        print("Seq2Seq 核心优化与架构演进（Reverse、Peeky 到 Attention）深度实证与机理透视\n")

def list_experiments():
    table_data = [
        ("1", "01_Math_Addition", "LSTM 字符级三位数加法实证 (Baseline vs Reverse vs Peeky vs Reverse+Peeky)"),
        ("2", "02_Machine_Translation", "LSTM 英法机器翻译端到端实证与 Corpus-BLEU 分段长短句评测"),
        ("3", "03_Vanilla_RNN_Addition", "经典基础 RNN (Vanilla RNN) 记忆崩溃与 Reverse 奇迹破解"),
        ("4", "04_Attention_Seq2seq", "Attention 注意力机制降维打击与对齐热力图/单步注意力权重探针"),
    ]
    if HAS_RICH:
        table = Table(title="🧪 实验套件清单 (Experiment Suites)", border_style="green", header_style="bold magenta")
        table.add_column("编号", justify="center", style="cyan", width=6)
        table.add_column("子实验模块", style="bold yellow", width=26)
        table.add_column("核心机理与评测重点", style="white")
        for row in table_data:
            table.add_row(*row)
        console.print(table)
    else:
        print("=== 实验套件清单 ===")
        for row in table_data:
            print(f"[{row[0]}] {row[1]}: {row[2]}")

def run_exp(choice):
    if choice == "1":
        print("\n>>> 启动实验一：LSTM 数学加法实证...")
        exp_dir = PROJECT_ROOT / "01_Math_Addition"
        os.system(f'python "{exp_dir / "visualize.py"}"')
    elif choice == "2":
        print("\n>>> 启动实验二：LSTM 英法机器翻译实证...")
        exp_dir = PROJECT_ROOT / "02_Machine_Translation"
        os.system(f'python "{exp_dir / "visualize.py"}"')
    elif choice == "3":
        print("\n>>> 启动实验三：Vanilla RNN 加法与 Reverse 逆序实证...")
        exp_dir = PROJECT_ROOT / "03_Vanilla_RNN_Addition"
        os.system(f'python "{exp_dir / "visualize.py"}"')
    elif choice == "4":
        print("\n>>> 启动实验四：Attention 注意力机制与单步探针...")
        exp_dir = PROJECT_ROOT / "04_Attention_Seq2seq"
        os.system(f'python "{exp_dir / "demo_step_by_step.py"}"')
    elif choice == "all":
        for c in ["1", "2", "3", "4"]:
            run_exp(c)

def main():
    parser = argparse.ArgumentParser(description="Seq2Seq & Attention 全景实证实验室")
    parser.add_argument("--exp", type=str, choices=["1", "2", "3", "4", "all"], help="运行指定实验编号")
    args = parser.parse_args()

    print_welcome()
    list_experiments()

    if args.exp:
        run_exp(args.exp)
    else:
        print("\n[提示] 使用方式: python main.py --exp [1|2|3|4|all]")

if __name__ == "__main__":
    main()

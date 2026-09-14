# -*- coding: utf-8 -*-
"""
LLM Foundations & Kernels Lab: 大语言模型核心算子与底层架构基准实验室
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

from experiments.exp1_rope_rotation import run as run_exp1
from experiments.exp2_kv_cache_analysis import run as run_exp2
from experiments.exp3_flash_attention_correctness import run as run_exp3

BANNER = r"""
  _     _     __  __   _  __                    _     
 | |   | |   |  \/  | | |/ /___ _ __ _ __   ___| |___ 
 | |   | |   | |\/| | | ' // _ \ '__| '_ \ / _ \ / __|
 | |___| |___| |  | | | . \  __/ |  | | | |  __/ \__ \
 |_____|_____|_|  |_| |_|\_\___|_|  |_| |_|\___|_|___/
  LLM Foundations & Core Attention Kernels Lab
"""

def print_welcome():
    if HAS_RICH:
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        desc = Text()
        desc.append("大语言模型架构基石与手撕核心算子基准评测工程 (RoPE, KV Cache, FlashAttention, LoRA)\n", style="bold green")
        desc.append("解构 RoPE 相对位置不变性、KV Cache 压缩演进（MHA/GQA/MQA/MLA）与 FlashAttention 切块在线 Softmax", style="dim white")
        console.print(Panel(desc, border_style="cyan"))
    else:
        print(BANNER)
        print("大语言模型架构基石与手撕核心算子基准评测工程\n")

def list_experiments():
    table_data = [
        ("1", "exp1_rope_rotation", "RoPE 旋转位置编码相对距离不变性检验与内积衰减实测"),
        ("2", "exp2_kv_cache_analysis", "KV Cache 演进全景建模 (MHA vs GQA vs MQA vs DeepSeek MLA 显存革命)"),
        ("3", "exp3_flash_attention_correctness", "FlashAttention Online Softmax 切块计算数学精确等价性证明"),
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
    parser = argparse.ArgumentParser(description="LLM Foundations & Kernels 实验室")
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

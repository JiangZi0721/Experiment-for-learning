# -*- coding: utf-8 -*-
"""
White-Box RNN Lab: 纯白盒可透视循环神经网络与语言模型实验室
一站式全景透视入口 (CLI & Interactive Dashboard) - 初学者亲和版
"""
import sys
import io
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

from benchmarks.grad_check import main as run_grad_check
from experiments.exp1_single_step_anatomy import run_experiment as run_exp1
from experiments.exp2_gradient_vanishing import run_experiment as run_exp2
from experiments.exp3_truncated_bptt import run_experiment as run_exp3
from experiments.exp4_train_rnnlm import run_experiment as run_exp4
from experiments.exp5_gated_rnn_duel import run_experiment as run_exp5
from src.glossary import print_glossary


BANNER = """
 ██████╗ ███╗   ██╗███╗   ██╗     ██╗     ███╗   ███╗
 ██╔══██╗████╗  ██║████╗  ██║     ██║     ████╗ ████║
 ██████╔╝██╔██╗ ██║██╔██╗ ██║     ██║     ██╔████╔██║
 ██╔══██╗██║╚██╗██║██║╚██╗██║     ██║     ██║╚██╔╝██║
 ██║  ██║██║ ╚████║██║ ╚████║     ███████╗██║ ╚═╝ ██║
 ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝     ╚══════╝╚═╝     ╚═╝
  White-Box RNN-LM Lab (白盒可透视循环神经网络语言模型实验室)
"""

def print_welcome():
    if HAS_RICH:
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        desc = Text()
        desc.append("基于纯 NumPy 底层白盒手推实现的 RNN 教学实战工程 (初学者零门槛版)\n", style="bold green")
        desc.append("拒绝黑盒调用！将记忆流动、梯度逆流、跨块接力与语言模型解构为高透明度终端看板", style="dim white")
        console.print(Panel(desc, border_style="cyan"))
    else:
        print(BANNER)
        print("基于纯 NumPy 底层白盒手推实现的 RNN 教学工程 (初学者零门槛版)")


def run_full_tour():
    """全自动全景教学巡礼"""
    if HAS_RICH:
        console.print("\n[bold green]>>> 启动【RNN 全景白盒教学大巡礼 (Full Tour)】...[/bold green]\n")
    else:
        print("\n>>> 启动【RNN 全景白盒教学大巡礼 (Full Tour)】...\n")

    print("\n[步骤 1/6] 严密数学梯度检验 (Numerical Gradient Check)...")
    run_grad_check()

    print("\n[步骤 2/6] 单步神经元内部解构 (看神经元怎么做笔记，以及为什么会麻木)...")
    run_exp1()

    print("\n[步骤 3/6] BPTT 时序反向传播 (看传话游戏为什么会消失或啸叫爆炸)...")
    run_exp2()

    print("\n[步骤 4/6] 截断反向传播跨段接力 (看十万字长文怎么用微小内存读完)...")
    run_exp3()

    print("\n[步骤 5/6] 门控 Gated RNN 终极对决 (看更新门高速公路如何终结梯度消失)...")
    run_exp5()

    print("\n[步骤 6/6] 自回归语言模型训练与生成 (看 AI 怎么从乱码到流畅接龙吐字)...")
    run_exp4(max_epoch=40, batch_size=4, time_size=20)

    if HAS_RICH:
        console.print("\n[bold green][SUCCESS] 全景大巡礼圆满完成！全部物理机制与动态透视均已印证。[/bold green]\n")
    else:
        print("\n[SUCCESS] 全景大巡礼圆满完成！全部物理机制与动态透视均已印证。\n")


def interactive_menu():
    """终端交互式菜单"""
    print_welcome()
    while True:
        if HAS_RICH:
            table = Table(title="请选择要透视的实验项目 (Interactive Menu)", show_header=True, header_style="bold magenta")
            table.add_column("选项", justify="center", style="bold yellow", width=6, overflow="fold")
            table.add_column("实验主题与透视内容", style="bold cyan", width=32, overflow="fold")
            table.add_column("初学者白话生活化比喻", style="yellow", width=34, overflow="fold")

            table.add_row("1", "单步神经元微观解构 (Anatomy)", "看神经元如何记笔记，以及信号太大为什么会听麻木")
            table.add_row("2", "BPTT 梯度逆流实验 (Vanishing)", "传话游戏逆向找责任人：声音衰减归零 vs 刺耳啸叫爆炸")
            table.add_row("3", "Truncated BPTT 跨段接力 (Relay)", "马拉松接力：记忆小本子一路传，犯错倒查不连累前人")
            table.add_row("4", "端到端 RNNLM 训练与生成 (LM)", "文字接龙演练：看选择困难症指数 (PPL) 怎样极速暴降")
            table.add_row("5", "Gated RNN 门控对决实验 (Duel)", "重置门与更新门：实测梯度高速公路直连无损穿透 30 步")
            table.add_row("6", "全网络数学梯度校验 (Grad Check)", "中心差分法验证手推导数，达到 10^-11 双精度极限")
            table.add_row("7", "全自动全景教学大巡礼 (Full Tour)", "依次执行全部 6 大实验，建立清晰完整的物理心智模型")
            table.add_row("g", "📖 专有名词'人话'对照宝典 (Glossary)", "生活化比喻速查字典：扫清所有学术黑话障碍")
            table.add_row("0", "退出系统 (Exit)", "结束本次透视之旅")
            console.print(table)
        else:
            print("\n[1] 单步神经元微观解构")
            print("[2] BPTT 梯度逆流实验")
            print("[3] Truncated BPTT 跨段接力")
            print("[4] 端到端 RNNLM 训练与生成")
            print("[5] Gated RNN 门控对决实验")
            print("[6] 全网络数学梯度校验")
            print("[7] 全自动全景教学大巡礼")
            print("[g] 专有名词'人话'对照宝典")
            print("[0] 退出系统")

        choice = input("\n请输入选项编号 [0-7 或 g]: ").strip().lower()
        if choice == "1":
            run_exp1()
        elif choice == "2":
            run_exp2()
        elif choice == "3":
            run_exp3()
        elif choice == "4":
            run_exp4(max_epoch=40, batch_size=4, time_size=20)
        elif choice == "5":
            run_exp5()
        elif choice == "6":
            run_grad_check()
        elif choice == "7":
            run_full_tour()
        elif choice == "g":
            print_glossary()
        elif choice in ("0", "q", "quit", "exit"):
            print("感谢使用 White-Box RNN-LM Lab，祝您在 NLP 学习之路上突飞猛进！")
            break
        else:
            print("[!] 无效输入，请重新输入。")


def main():
    parser = argparse.ArgumentParser(description="White-Box RNN Lab 统一入口")
    parser.add_argument("--tour", action="store_true", help="全自动执行完整全景教学巡礼")
    parser.add_argument("--grad-check", action="store_true", help="执行纯白盒计算图梯度校验")
    parser.add_argument("--glossary", action="store_true", help="查看专有名词初学者'人话'宝典")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], help="执行指定编号实验 (1-4)")
    args = parser.parse_args()

    if args.tour:
        print_welcome()
        run_full_tour()
    elif args.glossary:
        print_glossary()
    elif args.grad_check:
        run_grad_check()
    elif args.exp == 1:
        run_exp1()
    elif args.exp == 2:
        run_exp2()
    elif args.exp == 3:
        run_exp3()
    elif args.exp == 4:
        run_exp4()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()

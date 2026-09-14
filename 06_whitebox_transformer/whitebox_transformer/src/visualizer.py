# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 终端白盒可视化看板 (Visualizer)
支持 Rich 现代化色彩渲染与表格显示，同时提供自适应纯文本/ANSI 降级方案。
提供张量看板、注意力热力字符图、单步展开询问 (Active Prompting) 等能力。
"""
import sys
from typing import List, Optional, Tuple, Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

Matrix = List[List[float]]
Vector = List[float]

class WhiteBoxVisualizer:
    def __init__(self):
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None

    def print_banner(self, title: str):
        """打印一级章节横幅"""
        if HAS_RICH:
            self.console.print(f"\n[bold cyan]{'═' * 76}[/bold cyan]")
            self.console.print(f"  [bold yellow]📌 {title}[/bold yellow]")
            self.console.print(f"[bold cyan]{'═' * 76}[/bold cyan]\n")
        else:
            print("\n" + "=" * 76)
            print(f"  📌 {title}")
            print("=" * 76 + "\n")

    def print_subbanner(self, title: str):
        """打印二级小节横幅"""
        if HAS_RICH:
            self.console.print(f"\n[bold green]─── 🔹 {title} ───[/bold green]")
        else:
            print(f"\n--- 🔹 {title} ---")

    def ask_expand(self, step_name: str = "这其中的具体计算过程与数值矩阵") -> Any:
        """
        主动提问机制 (Active Prompting)
        每一步主动询问用户是否展开查看底层数值与张量运算细节
        """
        prompt = f"\n👉 是否展开查看【{step_name}】？(y: 展开查看 / n: 跳过查看公式 / q: 返回主菜单): "
        while True:
            choice = input(prompt).strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no', '']:
                return False
            elif choice in ['q', 'quit']:
                return 'quit'
            else:
                print("请输入 y (是), n (否) 或 q (退出到主菜单)。")

    def pause(self):
        """暂停等待用户确认推进"""
        input("\n[按 Enter 键继续推进到下一步...]")

    def show_matrix(self, mat: Matrix, 
                    row_labels: Optional[List[str]] = None, 
                    col_labels: Optional[List[str]] = None, 
                    precision: int = 4, 
                    name: str = ""):
        """打印多维数值矩阵"""
        if not mat or not mat[0]:
            print("(空矩阵)")
            return

        rows = len(mat)
        cols = len(mat[0])

        if HAS_RICH:
            title = f"矩阵: {name} (维度: {rows} x {cols})" if name else f"维度: ({rows} x {cols})"
            table = Table(title=title, show_lines=True, header_style="bold magenta")
            table.add_column("Row / Idx", justify="center", style="bold cyan", width=10)
            
            for j in range(cols):
                header = col_labels[j] if col_labels and j < len(col_labels) else f"dim_{j}"
                table.add_column(header, justify="right", style="green")

            for i in range(rows):
                r_lbl = row_labels[i] if row_labels and i < len(row_labels) else f"#{i}"
                row_vals = [f"{val:+.{precision}f}" for val in mat[i]]
                table.add_row(r_lbl, *row_vals)
                
            self.console.print(table)
        else:
            if name:
                print(f"\n【矩阵: {name}】 维度: ({rows} 行 x {cols} 列)")
            if col_labels:
                col_header = "       " + "   ".join([f"{col[:8]:>8}" for col in col_labels])
                print(col_header)
                print("       " + "-" * (len(col_labels) * 11))
            for idx, row in enumerate(mat):
                label = f"{row_labels[idx][:6]:>6}: " if row_labels else f"Row {idx:2d}: "
                row_str = "  ".join([f"{x:+.{precision}f}" for x in row])
                print(f"{label}[ {row_str} ]")

    def show_attention_map(self, attn_weights: Matrix, 
                           query_labels: List[str], 
                           key_labels: List[str], 
                           name: str = "注意力热力分布图 (Attention Map)"):
        """可视化注意力概率分布与字符热力图"""
        rows = len(attn_weights)
        cols = len(attn_weights[0])
        
        # 字符热力阶梯
        def get_heat_char(weight: float) -> str:
            if weight >= 0.70: return "████"
            elif weight >= 0.40: return "▓▓▓ "
            elif weight >= 0.20: return "▒▒  "
            elif weight >= 0.05: return "░   "
            else: return "    "

        if HAS_RICH:
            table = Table(title=f"🔥 {name} (数值 + 字符热力)", show_lines=True)
            table.add_column("Query \\ Key", justify="center", style="bold cyan", width=12)
            for k in key_labels:
                table.add_column(f"{k}", justify="center", style="bold yellow")

            for i in range(rows):
                q_lbl = query_labels[i]
                row_cells = []
                for j in range(cols):
                    w = attn_weights[i][j]
                    heat = get_heat_char(w)
                    row_cells.append(f"{w:.3f}\n[blue]{heat}[/blue]")
                table.add_row(q_lbl, *row_cells)
            self.console.print(table)
        else:
            print(f"\n【🔥 {name}】 (每行之和严格等于 1.0000)")
            header = "    Query \\ Key: " + "   ".join([f"{k[:6]:>6}" for k in key_labels])
            print(header)
            print("    " + "-" * (len(key_labels) * 10 + 16))
            for i in range(rows):
                q_lbl = query_labels[i]
                row_str = "  ".join([f"{attn_weights[i][j]:.3f}" for j in range(cols)])
                print(f"    {q_lbl:>10}: [ {row_str} ]")

    def show_prob_distribution(self, probs: Vector, vocab: List[str], top_k: int = 5):
        """展示输出词表概率分布榜"""
        pairs = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)
        if HAS_RICH:
            table = Table(title="🎯 当前生成步词表概率分布 Top 预测", show_lines=True)
            table.add_column("Rank", justify="center", style="bold green", width=6)
            table.add_column("Token", justify="center", style="bold cyan", width=12)
            table.add_column("概率 (Probability)", justify="right", style="bold yellow", width=18)
            table.add_column("置信度条 (Confidence Bar)", justify="left", style="blue", width=25)
            
            for rank, (idx, prob) in enumerate(pairs[:top_k], 1):
                token = vocab[idx]
                bar_len = int(prob * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                table.add_row(f"#{rank}", f"'{token}'", f"{prob*100:6.2f}%", bar)
            self.console.print(table)
        else:
            print("\n🎯 词表概率分布 Top 预测:")
            for rank, (idx, prob) in enumerate(pairs[:top_k], 1):
                token = vocab[idx]
                bar = "█" * int(prob * 20)
                print(f"  #{rank} Token: '{token:<8}' -> {prob*100:6.2f}% | {bar}")

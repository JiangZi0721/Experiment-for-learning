# -*- coding: utf-8 -*-
"""
终端白盒全景透视看板 (Visualizer) - 初学者亲和版
基于 Rich 库构建多维高信息密度控制台看板，内置【💡新手白话导读】与生活化比喻，
让初学者零门槛透视 RNN 内部流动、梯度逆流、记忆接力与文本生成。
"""
import sys
import io
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# 强制在 Windows 控制台下使用 UTF-8 编码，防止 gbk 异常
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class RNNVisualizer:
    """RNN 白盒全景透视可视化器 (初学者亲和版)"""
    def __init__(self):
        if HAS_RICH:
            self.console = Console(force_terminal=True)
        else:
            self.console = None
        self._seen_tips = set()

    def print_banner(self, title: str, subtitle: str = ""):
        """打印主标题横幅"""
        if HAS_RICH:
            content = f"[bold white]{title}[/bold white]"
            if subtitle:
                content += f"\n[dim cyan]{subtitle}[/dim cyan]"
            self.console.print(Panel(content, border_style="cyan", expand=False))
        else:
            print(f"\n{'='*20} {title} {'='*20}")
            if subtitle:
                print(f"--- {subtitle} ---")

    def print_tip(self, tip_text: str, tip_key: Optional[str] = None, once: bool = True):
        """
        打印给初学者的白话导读提示小卡片
        once=True 时相同卡片内容仅打印一次，杜绝多轮实验重复刷屏
        """
        key = tip_key if tip_key is not None else tip_text.strip()[:40]
        if once and key in self._seen_tips:
            return
        self._seen_tips.add(key)

        if HAS_RICH:
            msg = Text()
            msg.append("💡【初学者白话速懂指南】\n", style="bold yellow")
            msg.append(tip_text, style="white")
            self.console.print(Panel(msg, border_style="yellow", expand=False))
        else:
            print(f"\n[💡 初学者白话速懂指南]: {tip_text}\n")

    def show_metric_definitions(
        self,
        title: str,
        definitions: List[Tuple[str, str, str, str]]
    ):
        """
        展示实验核心指标通俗大白话词典看板 (Metric Glossary Panel)
        彻底解决初学者'只看数字不知其意'的痛点，逐一拆解每个指标的学术名、数学来源、生活化含义与正常基准区间。

        参数:
            title: 看板主标题
            definitions: 列表，每项为 (学术指标名, 数学定义/公式, 初学者人话含义, 诊断与参考区间)
        """
        if not HAS_RICH:
            print(f"\n=== [指标字典] {title} ===")
            for name, formula, meaning, diag in definitions:
                print(f"• {name} [{formula}]: {meaning} -> {diag}")
            return

        table = Table(
            title=f"【核心指标白话词典】{title}",
            show_header=True,
            header_style="bold magenta",
            border_style="dim magenta"
        )
        table.add_column("学术物理指标", style="bold cyan", width=14, overflow="fold")
        table.add_column("数学定义 / 来源", style="yellow", width=18, overflow="fold")
        table.add_column("初学者白话含义", style="white", width=25, overflow="fold")
        table.add_column("正常参考与诊断", style="green", width=17, overflow="fold")

        for name, formula, meaning, diag in definitions:
            table.add_row(name, formula, meaning, diag)

        self.console.print(table)

    def _render_bar(self, ratio: float, width: int = 14, color: str = "cyan") -> str:
        """生成文本条形可视化进度条"""
        ratio = max(0.0, min(1.0, ratio))
        filled = int(round(ratio * width))
        empty = width - filled
        if HAS_RICH:
            return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim] {ratio*100:5.1f}%"
        else:
            return f"{'#' * filled}{'.' * empty} {ratio*100:5.1f}%"

    def show_cell_anatomy(
        self,
        step_idx: int,
        probe_data: Dict[str, Any],
        show_tip: bool = True,
        show_banner: bool = True,
        scene_name: str = ""
    ):
        """
        探针 1：单步 RNN 神经元内部切片透视看板 (One-Step Cell Anatomy)
        """
        if show_banner:
            sub = f"透视前向加权融合、tanh 非线性激活与微观饱和度 ({scene_name})" if scene_name else "透视前向加权融合、tanh 非线性激活与微观饱和度"
            self.print_banner(f"【探针 1】单步 RNN 神经元解构 (时间步 t={step_idx})", sub)

        if show_tip:
            self.print_tip(
                "神经元就像一个人在边看书边做笔记：\n"
                "• 它同时接收两股信号：刚刚看到的新词（外部刺激）与之前记下的小本子（历史记忆）。\n"
                "• 重点关注【tanh 饱和度】：如果输入能量太大导致神经元'情绪过载/听麻木了'（饱和率 > 30%），\n"
                "  数学上的导数会瞬间归零，导致反向纠错时改错信号完全无法穿透进来，这就是'梯度消失'的微观根源！",
                tip_key="probe1_cell_anatomy"
            )

        if not HAS_RICH:
            print(f"x_norm: {probe_data.get('x_norm', 0):.4f}, h_prev_norm: {probe_data.get('h_prev_norm', 0):.4f}")
            print(f"Memory Ratio: {probe_data.get('memory_ratio', 0):.2%}, Saturation: {probe_data.get('saturation_ratio', 0):.2%}")
            return

        table_title = f"神经元微观工作台 (Step t={step_idx} - {scene_name})" if scene_name else f"神经元微观工作台 (Step t={step_idx})"
        table = Table(title=table_title, show_header=True, header_style="bold magenta", border_style="dim")
        table.add_column("物理组件", style="bold cyan", width=15, overflow="fold")
        table.add_column("生活比喻", style="yellow", width=13, overflow="fold")
        table.add_column("数值与能量", justify="center", width=18, overflow="fold")
        table.add_column("微观状态解读", justify="left", width=26, overflow="fold")

        # 输入项
        table.add_row(
            "输入投影 x@Wx",
            "新词外部刺激",
            f"模长: {probe_data.get('x_proj_norm', 0.0):.2f}",
            f"外部刺激占比: {1.0 - probe_data.get('memory_ratio', 0.5):.1%}"
        )
        # 历史项
        mem_ratio = probe_data.get("memory_ratio", 0.5)
        bar_mem = self._render_bar(mem_ratio, 8, "green" if mem_ratio > 0.5 else "yellow")
        table.add_row(
            "记忆循环 h@Wh",
            "前文旧笔记",
            f"模长: {probe_data.get('h_proj_norm', 0.0):.2f}",
            f"依赖历史: {bar_mem}"
        )
        # 预激活
        table.add_row(
            "预激活和 a_t",
            "激活前势能",
            f"均={probe_data.get('a_t_mean', 0.0):.2f}, 方={probe_data.get('a_t_std', 0.0):.2f}",
            "新刺激与旧笔记混合总能量"
        )
        # 饱和度
        sat_ratio = probe_data.get("saturation_ratio", 0.0)
        sat_color = "red" if sat_ratio > 0.3 else ("yellow" if sat_ratio > 0.1 else "green")
        bar_sat = self._render_bar(sat_ratio, 8, sat_color)
        sat_desc = "[bold red]极度麻木！导数趋近 0[/bold red]" if sat_ratio > 0.3 else "[green]敏锐活跃，改错顺畅[/green]"
        table.add_row(
            "tanh 饱和度",
            "耳朵麻木度",
            bar_sat,
            sat_desc
        )
        # 输出隐状态
        table.add_row(
            "新隐状态 h_t",
            "新笔记沉淀",
            f"均值: {probe_data.get('h_next_mean', 0.0):.3f}",
            "压缩至(-1,1)传给后继"
        )

        self.console.print(table)

    def show_time_unrolling_forward(
        self,
        forward_probe: Dict[str, Any],
        show_tip: bool = True,
        show_banner: bool = True
    ):
        """
        探针 2：TimeRNN 时序展开前向流动看板 (Time Unrolling Forward Flow)
        """
        if show_banner:
            self.print_banner("【探针 2】TimeRNN 时序前向展开流动透视", "观察状态向量 h_t 在时间序列 T 个时钟周期的演化轨迹")

        if show_tip:
            self.print_tip(
                "这里展示模型连续读入 T 个字的过程中，脑海中的记忆小本子 (h_t) 是怎么一步步演变的：\n"
                "• 关注【笔记饱满度】：看记忆小本子在句子开头、中间和结尾的记忆充盈程度。\n"
                "• 关注【听从历史比例】：刚开始读第 1 个字时记忆为空，随着句子拉长，模型越来越依赖前文语境。",
                tip_key="probe2_forward_unrolling"
            )

        step_h_norms = forward_probe.get("step_h_norms", [])
        step_sat = forward_probe.get("step_sat_ratios", [])
        step_mem = forward_probe.get("step_mem_ratios", [])
        T = len(step_h_norms)

        if not HAS_RICH or T == 0:
            print(f"Time steps: {T}, Initial norm: {forward_probe.get('h_start_norm', 0):.4f}, Final norm: {forward_probe.get('h_end_norm', 0):.4f}")
            return

        table = Table(title="读句推进时钟轨迹 (前向时序展开)", show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("时序", justify="center", style="bold yellow", width=8, overflow="fold")
        table.add_column("记忆模长 ||h_t||", justify="right", style="green", width=18, overflow="fold")
        table.add_column("依赖历史比重", justify="left", width=18, overflow="fold")
        table.add_column("饱和麻木比", justify="left", width=16, overflow="fold")
        table.add_column("推进状态", justify="left", width=14, overflow="fold")

        max_norm = max(step_h_norms) if max(step_h_norms) > 0 else 1.0
        for t in range(T):
            norm_val = step_h_norms[t]
            norm_bar = self._render_bar(norm_val / max_norm, 8, "cyan")
            mem_bar = self._render_bar(step_mem[t], 8, "magenta")
            sat_bar = self._render_bar(step_sat[t], 8, "red" if step_sat[t] > 0.2 else "green")

            note = "开始破题" if t == 0 else ("句意收拢" if t == T-1 else "语境累积中")
            table.add_row(f"第 {t+1:02d} 字", f"{norm_val:6.3f} {norm_bar}", mem_bar, sat_bar, note)

        self.console.print(table)

    def show_bptt_backward_flow(
        self,
        backward_probe: Dict[str, Any],
        show_tip: bool = True,
        show_banner: bool = True,
        scene_name: str = ""
    ):
        """
        探针 3：BPTT 梯度逆流与消失/爆炸透视看板 (BPTT Gradient Flow Probe)
        """
        if show_banner:
            sub = f"观察改错信号沿时间轴逆流回溯时，隐状态梯度 ||dh_t|| 的衰减与放大 ({scene_name})" if scene_name else "观察改错信号沿时间轴逆流回溯时，隐状态梯度 ||dh_t|| 的衰减与放大"
            self.print_banner("【探针 3】BPTT 梯度时序逆流透视 (倒放电影·秋后算账)", sub)

        if show_tip:
            self.print_tip(
                "秋后算账时间到！句子末尾预测出错后，模型像'倒放电影'一样从未来向过去倒查责任人：\n"
                "• 重点关注【信号放大/缩小倍数】：\n"
                "  - 若倍数飞速缩水至 0.0x（黄色）：说明发生【梯度消失】，前面的字根本分不到改错责任，永远学不会长距离语法！\n"
                "  - 若倍数狂飙到几十甚至上百倍（红色）：说明发生【梯度爆炸】，像话筒啸叫一样数字失控，参数会被踹飞崩溃！\n"
                "  - 只有稳定保持在 1.0x 附近（绿色）：误差信号才能健康穿越漫长的时间轴。",
                tip_key="probe3_bptt_backward"
            )

        step_dh = backward_probe.get("step_dh_norms", [])
        T = len(step_dh)

        if not HAS_RICH or T == 0:
            print(f"Gradient norms across time: {step_dh}")
            return

        table_title = f"误差信号逆流账本 ({scene_name})" if scene_name else "误差信号逆时序回溯账本 (从末尾倒查回开头)"
        table = Table(title=table_title, show_header=True, header_style="bold red", border_style="dim")
        table.add_column("倒查节点", justify="center", style="bold yellow", width=13, overflow="fold")
        table.add_column("梯度模长 ||dh_t||", justify="right", style="bold", width=16, overflow="fold")
        table.add_column("相比末端倍数", justify="left", width=14, overflow="fold")
        table.add_column("健康度诊断", justify="left", width=27, overflow="fold")

        terminal_grad = step_dh[-1] if step_dh[-1] > 0 else 1e-8
        for t in reversed(range(T)):
            cur_dh = step_dh[t]
            ratio = cur_dh / terminal_grad

            # 状态诊断
            if ratio < 0.05:
                diag = "[bold yellow][!] 严重梯度消失 (声音听不见了)[/bold yellow]"
                color = "yellow"
            elif ratio > 20.0:
                diag = "[bold red][FAIL] 梯度指数爆炸 (回音啸叫失控)[/bold red]"
                color = "red"
            else:
                diag = "[green][OK] 信号平稳可达 (健康传递)[/green]"
                color = "green"

            ratio_str = f"{ratio:8.2e}x" if (ratio < 0.01 or ratio > 100) else f"{ratio:6.2f}x"
            table.add_row(
                f"← 倒查第{t+1:02d}字",
                f"{cur_dh:10.6f}",
                f"[{color}]{ratio_str}[/{color}]",
                diag
            )

        self.console.print(table)

        # 打印总汇参数梯度
        summary_panel = Text()
        summary_panel.append(f"• 输入字嵌入梯度   ||dWx||: {backward_probe.get('total_dWx_norm', 0.0):.4f} (新词特征改错步幅)\n", style="cyan")
        summary_panel.append(f"• 记忆循环矩阵梯度 ||dWh||: {backward_probe.get('total_dWh_norm', 0.0):.4f} (记笔记手法改错步幅)\n", style="magenta")
        summary_panel.append(f"• 偏置基础阈值梯度 ||db||:  {backward_probe.get('total_db_norm', 0.0):.4f}\n", style="yellow")
        summary_panel.append(f"• 试图穿透到上一段的梯度 ||dh_prev||: {backward_probe.get('dh_to_prev_chunk_norm', 0.0):.4f} (在截断 BPTT 中将被直接斩断置 0！)", style="bold red")
        self.console.print(Panel(summary_panel, title="时间轴累积总账本", border_style="red"))

    def show_truncated_bptt_relay(
        self,
        chunk_idx: int,
        prev_h_norm: float,
        curr_h_norm: float,
        dh_truncated_norm: float,
        show_tip: bool = True,
        show_banner: bool = True
    ):
        """
        探针 4：Truncated BPTT 跨块记忆接力与梯度截断看板 (Relay & Truncation)
        """
        if show_banner:
            self.print_banner(f"【探针 4】Truncated BPTT 跨段接力透视 (第 #{chunk_idx+1} 分段)", "揭秘 RNN 能够学完超长小说而不挤爆电脑内存的绝招")

        if show_tip:
            self.print_tip(
                "为什么十万字的小说不会把显存撑爆？\n"
                "因为 Truncated BPTT 实行了极其聪明的【责任有限制】法则：\n"
                "1. 读课文时（前向传播）：记忆小本子 (h) 像火炬接力一样一路传下去，永远不扔掉，保住跨越整本书的长程记忆！\n"
                "2. 找责任时（反向传播）：各人自扫门前雪，改错只追究当前这一段（如 15 步），绝不跨段找前一段的麻烦（反向截断）！\n"
                "这样电脑内存永远只需要保存 15 步的账本，花极小的显存就能读完一整座图书馆！",
                tip_key="probe4_truncated_relay"
            )

        if not HAS_RICH:
            print(f"Chunk {chunk_idx}: Prev h norm = {prev_h_norm:.4f}, New h norm = {curr_h_norm:.4f}, Truncated grad norm = {dh_truncated_norm:.4f}")
            return

        table = Table(title=f"Truncated BPTT 分段接力实测 (Chunk #{chunk_idx})", show_header=True, header_style="bold blue", border_style="dim")
        table.add_column("物理方向", justify="center", style="bold", width=12, overflow="fold")
        table.add_column("交接策略", justify="center", width=16, overflow="fold")
        table.add_column("实测数值", justify="right", width=16, overflow="fold")
        table.add_column("生活化大白话解读", justify="left", width=28, overflow="fold")

        # 前向接力
        table.add_row(
            "[green]正向读书 (前向)[/green]",
            "[bold green]火炬接力 Carry[/bold green]",
            f"前段={prev_h_norm:.2f} -> 本段={curr_h_norm:.2f}",
            "[green]完整继承前文笔记，语境不丢！[/green]"
        )

        # 反向截断
        table.add_row(
            "[red]逆向算账 (反向)[/red]",
            "[bold red]边界斩断 Trunc[/bold red]",
            f"向更早追责={dh_truncated_norm:.2f} -> 归零",
            "[red]绝不连累上段，显存锁定常数！[/red]"
        )

        self.console.print(table)

    def show_gradient_clipping(
        self,
        orig_norm: float,
        final_norm: float,
        scale_rate: float,
        max_norm: float,
        show_tip: bool = True,
        show_banner: bool = True
    ):
        """
        探针 5：梯度裁剪手术看板 (Gradient Clipping Surgery)
        """
        if show_banner:
            self.print_banner("【探针 5】梯度裁剪 (Gradient Clipping) 保命刹车看板", "观察防范模型暴走崩溃的自动限速机制")

        if show_tip:
            self.print_tip(
                "在训练 RNN 时，偶尔会遇到突发的一阵大风暴（比如某个词算出的改错步子巨大）。\n"
                "如果放任这个巨大的改错步子冲进模型，参数会被一瞬间砸成乱码 (NaN)。\n"
                "【梯度裁剪】就像电闸上的空气保险丝：只要改错总步长超标，就一律按比例缩减到安全上限之内！",
                tip_key="probe5_gradient_clipping"
            )

        if not HAS_RICH:
            print(f"Orig norm: {orig_norm:.4f}, Max norm: {max_norm:.4f}, Scale: {scale_rate:.4f}, Final norm: {final_norm:.4f}")
            return

        is_clipped = orig_norm > max_norm
        status_text = "[bold red][CLIP] 触发自动刹车！已等比缩减[/bold red]" if is_clipped else "[bold green][OK] 处于安全速度内 (平稳通过)[/bold green]"

        table = Table(title="自动限速保命对比表", show_header=True, header_style="bold yellow", border_style="dim")
        table.add_column("物理指标", style="bold cyan", width=16, overflow="fold")
        table.add_column("大白话含义", style="dim cyan", width=14, overflow="fold")
        table.add_column("实测数值", justify="right", width=14, overflow="fold")
        table.add_column("安全状态与措施", justify="left", width=28, overflow="fold")

        table.add_row("全局梯度 ||g||", "全网改错总步长", f"{orig_norm:8.2f}", "所有参数梯度的欧氏总模长")
        table.add_row("限速阈值 max", "允许最大安全步长", f"{max_norm:8.2f}", "一旦超过该线必须立刻刹车")
        table.add_row("缩减倍率 eta", "减速刹车系数", f"{scale_rate:8.4f}", f"公式: min(1.0, {max_norm} / ||g||)")
        table.add_row("生效步长", "实际更新的安全步长", f"[{'yellow' if is_clipped else 'green'}]{final_norm:8.2f}[/]", status_text)

        self.console.print(table)

    def show_lm_step_prediction(
        self,
        step_idx: int,
        input_token: str,
        target_token: str,
        top_candidates: List[Tuple[str, float]],
        loss: float,
        ppl: float,
        show_tip: bool = True,
        show_banner: bool = True
    ):
        """
        探针 6：RNNLM 语言模型自回归预测看板 (Next-Token Probabilities)
        """
        if show_banner:
            self.print_banner(f"【探针 6】语言模型单步文字接龙看板 (第 #{step_idx} 步演练)", "透视模型脑海里给出的下一字概率排行榜")

        if show_tip:
            self.print_tip(
                "语言模型的核心任务只有一件事：【文字接龙】！\n"
                "看它看到当前字时，心里排在第 1 名的候选字是谁：\n"
                "• 重点关注【困惑度 PPL】：把它理解为模型的'选择困难症指数'。\n"
                "  - PPL = 200：说明模型像从 200 张卡片里瞎抓一张（极其迷茫，刚开始学）；\n"
                "  - PPL = 1.05：说明模型心里几乎百分之百肯定下一个字就是它，毫不纠结（学成了）！",
                tip_key="probe6_lm_step"
            )

        if not HAS_RICH:
            print(f"Input: '{input_token}' -> Target: '{target_token}', Loss: {loss:.4f}, PPL: {ppl:.2f}")
            print(f"Top candidates: {top_candidates}")
            return

        table = Table(
            title=f"看到: [yellow]'{input_token}'[/yellow] ──猜下一字──> 答案: [green]'{target_token}'[/green]",
            show_header=True,
            header_style="bold green"
        )
        table.add_column("名次", justify="center", width=6, overflow="fold")
        table.add_column("候选字", justify="center", style="bold cyan", width=8, overflow="fold")
        table.add_column("置信度概率 (Softmax)", justify="left", width=24, overflow="fold")
        table.add_column("判定", justify="center", width=16, overflow="fold")

        is_hit = False
        for rank, (cand, prob) in enumerate(top_candidates, 1):
            bar = self._render_bar(prob, 10, "green" if cand == target_token else "cyan")
            match = "[bold green][HIT 命中][/bold green]" if cand == target_token else "[dim]备选[/dim]"
            if cand == target_token:
                is_hit = True
            table.add_row(f"#{rank}", f"'{cand}'", bar, match)

        self.console.print(table)

        info_text = Text()
        info_text.append(f"• 单步猜错惩罚 (交叉熵损失 Loss): {loss:.4f}\n", style="bold yellow")
        info_text.append(f"• 模型迷茫指数 (困惑度 PPL = exp(Loss)): {ppl:.2f} (相当于在 {max(1, int(round(ppl)))} 个字里掷骰子猜测)\n", style="bold magenta")
        info_text.append(f"• 押宝状态: {'[green]太棒了！模型第 1 候选字完全命中真实后继字！[/green]' if is_hit else '[yellow]还在苦练中，排在第 1 的字还不是标准答案，梯度正在全力纠偏...[/yellow]'}")
        self.console.print(Panel(info_text, border_style="cyan"))

    def show_lm_training_evolution_table(self, milestones: List[Dict[str, Any]]):
        """
        全景展示训练演变里程碑表格 (Training Evolution Table)
        将模型从最初的无知胡猜到最后的精准预测浓缩在一张清爽优美的大表中，
        避免终端刷屏，适配标准 80 列控制台，极大提升阅读体验。
        """
        if not HAS_RICH:
            for m in milestones:
                print(m)
            return

        table = Table(
            title="【全景演化透视】语言模型文字接龙从混沌走向有序的蜕变历程",
            show_header=True,
            header_style="bold cyan",
            border_style="dim cyan"
        )
        table.add_column("步数", justify="center", width=7, overflow="fold")
        table.add_column("轮次", justify="center", width=7, overflow="fold")
        table.add_column("Loss", justify="right", width=8, overflow="fold")
        table.add_column("PPL", justify="right", width=8, overflow="fold")
        table.add_column("实测预测采样", justify="left", width=24, overflow="fold")
        table.add_column("演化阶段评价", justify="left", width=18, overflow="fold")

        for m in milestones:
            step = m["step"]
            epoch = m["epoch"]
            loss = m["loss"]
            ppl = m["ppl"]
            in_char = m.get("in_char", "")
            pred_char = m.get("pred_char", "")
            prob = m.get("prob", 0.0)
            hit = m.get("hit", False)
            eval_text = m.get("eval_text", "")

            # 颜色分级
            if loss > 4.0:
                loss_str = f"[bold red]{loss:.4f}[/bold red]"
                ppl_str = f"[bold red]{ppl:.1f}[/bold red]"
            elif loss > 1.5:
                loss_str = f"[bold yellow]{loss:.4f}[/bold yellow]"
                ppl_str = f"[bold yellow]{ppl:.1f}[/bold yellow]"
            else:
                loss_str = f"[bold green]{loss:.4f}[/bold green]"
                ppl_str = f"[bold green]{ppl:.2f}[/bold green]"

            hit_icon = "[bold green][HIT][/bold green]" if hit else "[dim][MISS][/dim]"
            pred_str = f"'{in_char}'->'{pred_char}' ({prob*100:4.1f}%) {hit_icon}"

            table.add_row(
                f"#{step}",
                f"Ep {epoch}",
                loss_str,
                ppl_str,
                pred_str,
                eval_text
            )

        self.console.print(table)

    def show_lm_generation_panel(
        self,
        case_idx: int,
        prompt: str,
        generated_text: str,
        strategy_desc: str,
        quality_eval: str,
        is_repetitive_trap: bool = False
    ):
        """
        自回归文本生成结果高可读性卡片
        清晰分离 Prompt 与 AI 续写内容，避免杂乱混排
        """
        if not HAS_RICH:
            print(f"\n--- 生成测试 #{case_idx} ---")
            print(f"Prompt: {prompt}")
            print(f"Continuation: {generated_text}")
            print(f"Strategy: {strategy_desc}")
            print(f"Evaluation: {quality_eval}")
            return

        body = Text()
        body.append("[*] 提示引导词 (Prompt):\n", style="bold yellow")
        body.append(f"   \"{prompt}\"\n\n", style="bold yellow")

        continuation_style = "bold green" if not is_repetitive_trap else "bold red"
        body.append("[>] 语言模型自回归续写 (AI Continuation):\n", style=continuation_style)
        body.append(f"   \"{generated_text}\"\n\n", style=continuation_style)

        body.append("[=] 解码策略: ", style="bold cyan")
        body.append(f"{strategy_desc}\n", style="white")

        body.append("[?] 质量透视: ", style="bold magenta")
        body.append(f"{quality_eval}", style="white")

        title = f"[bold cyan]自回归生成测试 #{case_idx}[/bold cyan]"
        border_color = "red" if is_repetitive_trap else "green"
        self.console.print(Panel(body, title=title, border_style=border_color, expand=False))

    def show_gated_rnn_probe(
        self,
        step_idx: int,
        reset_gate: float,
        update_gate: float,
        h_norm: float,
        highway_flow: float,
        retention_gain: float,
        show_tip: bool = True,
        show_banner: bool = True
    ):
        """
        探针 7：Gated RNN 门控机制与梯度高速公路透视看板 (Gated Units & Gradient Highway)
        """
        if show_banner:
            self.print_banner(f"【探针 7】Gated RNN 门控透视 (时钟周期 t={step_idx})", "透视重置门、更新门与梯度直连无损高速公路")

        if show_tip:
            self.print_tip(
                "为什么门控循环网络 (如 GRU/LSTM) 能彻底终结梯度消失？\n"
                "因为门控给神经元安装了两个极其智能的【智能阀门】：\n"
                "1. 🔄 重置门 (Reset Gate, r)：'旧账清空阀'。开度越小，越彻底擦除无关的前文陈旧记忆。\n"
                "2. 🔀 更新门 (Update Gate, z)：'新旧交替阀'。决定本时刻是由旧笔记主导还是写入新记忆。\n"
                "★ 最震撼的物理机制【梯度高速公路 (1 - z)】：反向传播时，误差信号通过 (1 - z) 直连通道直接穿透，\n"
                "  完全绕过了传统的权重矩阵连乘！即使回传 100 步，梯度依然毫发无损！",
                tip_key="probe7_gated_rnn"
            )

        if not HAS_RICH:
            print(f"Step {step_idx}: Reset gate = {reset_gate:.2%}, Update gate = {update_gate:.2%}, Highway = {highway_flow:.2%}, Retention = {retention_gain:.4f}")
            return

        table = Table(title=f"门控阀门动态开闭看板 (Step t={step_idx})", show_header=True, header_style="bold green", border_style="dim")
        table.add_column("门控组件", style="bold cyan", width=15, overflow="fold")
        table.add_column("生活比喻", style="yellow", width=13, overflow="fold")
        table.add_column("阀门开度", justify="left", width=18, overflow="fold")
        table.add_column("物理决策解读", justify="left", width=26, overflow="fold")

        bar_r = self._render_bar(reset_gate, 8, "cyan")
        r_desc = "[dim]保留大部分旧笔记[/dim]" if reset_gate > 0.5 else "[bold red]大幅擦除旧账，轻装上阵[/bold red]"
        table.add_row("重置门 (r)", "旧账清空阀", bar_r, r_desc)

        bar_z = self._render_bar(update_gate, 8, "magenta")
        z_desc = "[green]大幅继承历史记忆[/green]" if update_gate < 0.5 else "[yellow]以吸纳当前新词为主[/yellow]"
        table.add_row("更新门 (z)", "新旧权衡阀", bar_z, z_desc)

        bar_hw = self._render_bar(highway_flow, 8, "green")
        table.add_row("梯度公路 (1-z)", "无损直达通道", bar_hw, "[bold green]畅通直达上一时刻[/bold green]")

        table.add_row("新隐状态 ||h||", "融合后新记忆", f"模长: {h_norm:.2f}", "新旧有机融合的记忆")
        table.add_row("局部梯度增益", "误差抗衰能力", f"{retention_gain:.4f}x", "[bold green]超越 Vanilla RNN！[/bold green]")

        self.console.print(table)

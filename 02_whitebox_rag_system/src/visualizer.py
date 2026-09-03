# -*- coding: utf-8 -*-
"""
终端白盒全景透视看板 (Visualizer)
基于 Rich 库构建多维表格，实时展示初排双路对抗、RRF名次跃迁、重排颠覆轨迹与 Prompt 组装。
"""
from typing import List, Dict, Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class WhiteBoxVisualizer:
    def __init__(self):
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None

    def print_banner(self, title: str):
        if HAS_RICH:
            self.console.print(f"\n[bold cyan]══════════════ {title} ══════════════[/bold cyan]")
        else:
            print(f"\n============== {title} ==============")

    def show_first_stage_duel(self, query: str, bm25_results: List[Dict[str, Any]], dense_results: List[Dict[str, Any]]):
        """面板 1：初排双路对抗看板"""
        self.print_banner("1. 初排双路对抗看板 (BM25 稀疏 vs Dense 稠密)")

        if not HAS_RICH:
            print(f"Query: {query}")
            print("BM25 Top 5:")
            for item in bm25_results[:5]:
                print(f"  #{item['rank']} [{item['chunk_id']}] Score: {item['score']} Hits: {item.get('hit_terms', [])}")
            print("Dense Top 5:")
            for item in dense_results[:5]:
                print(f"  #{item['rank']} [{item['chunk_id']}] Cosine Sim: {item['score']}")
            return

        table = Table(title=f"Query: [yellow]{query}[/yellow]", show_lines=True)
        table.add_column("Rank", justify="center", style="bold green", width=6)
        table.add_column("BM25 召回切片 (稀疏关键词)", style="cyan", width=38)
        table.add_column("BM25 分", justify="right", style="bold", width=9)
        table.add_column("Dense 召回切片 (稠密向量)", style="magenta", width=38)
        table.add_column("余弦分", justify="right", style="bold", width=9)

        max_len = max(len(bm25_results), len(dense_results))
        for i in range(min(max_len, 7)):
            b_item = bm25_results[i] if i < len(bm25_results) else None
            d_item = dense_results[i] if i < len(dense_results) else None

            b_str = f"[{b_item['chunk_id']}]\n{b_item['heading_path'][:30]}..." if b_item else "-"
            b_sc = str(b_item['score']) if b_item else "-"
            d_str = f"[{d_item['chunk_id']}]\n{d_item['heading_path'][:30]}..." if d_item else "-"
            d_sc = str(d_item['score']) if d_item else "-"

            table.add_row(f"#{i+1}", b_str, b_sc, d_str, d_sc)

        self.console.print(table)

        # 双路对决态势诊断 (Duel Synergy Diagnosis)
        b_ids = [item['chunk_id'] for item in bm25_results[:5]]
        d_ids = [item['chunk_id'] for item in dense_results[:5]]
        overlap = set(b_ids) & set(d_ids)
        d_top = dense_results[0]['score'] if dense_results else 0.0
        d_cut = dense_results[min(4, len(dense_results)-1)]['score'] if dense_results else 0.0
        d_margin = d_top - d_cut

        top1_match = (b_ids and d_ids and b_ids[0] == d_ids[0])
        status_tag = "[bold green][双路共识 / CONSENSUS][/bold green] BM25 与 Dense 协同锁定相同黄金切片，置信度极高" if top1_match else "[bold yellow][正交互补 / COMPLEMENTARY][/bold yellow] BM25 侧重精准词频，Dense 探索语义流形，各展所长"

        diag_text = (
            f"- [bold]对决态势[/bold]：{status_tag}\n"
            f"- [bold]Top-5 候选重叠率[/bold]：共识重叠 {len(overlap)} 篇 / BM25 独占 {len(b_ids)-len(overlap)} 篇 / Dense 独占 {len(d_ids)-len(overlap)} 篇 (互补多样性)\n"
            f"- [bold]Dense 神经判别裕度[/bold]：Top-1 余弦 [cyan]{d_top:.3f}[/cyan] vs Top-5 余弦 [cyan]{d_cut:.3f}[/cyan] (置信区分鸿沟 Δ = [bold green]{d_margin:.3f}[/bold green])"
        )
        self.console.print(Panel(diag_text, title="[bold]初排双路对决与互补诊断 (Synergy Diagnosis)[/bold]", border_style="cyan"))

    def show_rrf_fusion(self, fused_results: List[Dict[str, Any]]):
        """面板 2：RRF 融合名次跃迁看板"""
        self.print_banner("2. 倒数排名融合 (RRF) 跃迁追踪")

        if not HAS_RICH:
            for item in fused_results[:5]:
                print(f"  #{item['final_rank']} [{item['chunk_id']}] RRF Score: {item['rrf_score']} | BM25: #{item['bm25_rank']} | Dense: #{item['dense_rank']} ({item['source_type']})")
            return

        table = Table(title="RRF Formula: Score = 1/(60+Rank_BM25) + 1/(60+Rank_Dense)", show_lines=True)
        table.add_column("融合位次", justify="center", style="bold yellow", width=10)
        table.add_column("切片编号与标题路径", style="white", width=42)
        table.add_column("BM25 排名", justify="center", style="cyan", width=11)
        table.add_column("Dense 排名", justify="center", style="magenta", width=11)
        table.add_column("RRF 得分", justify="right", style="bold green", width=12)
        table.add_column("跃迁特征说明", style="dim", width=26)

        for item in fused_results[:8]:
            b_rk = f"#{item['bm25_rank']}" if item['bm25_rank'] else "[red]未入围[/red]"
            d_rk = f"#{item['dense_rank']}" if item['dense_rank'] else "[red]未入围[/red]"
            
            table.add_row(
                f"#{item['final_rank']}",
                f"[{item['chunk_id']}]\n{item['heading_path']}",
                b_rk,
                d_rk,
                f"{item['rrf_score']:.5f}",
                f"[{'green' if 'Both' in item['source_type'] else 'yellow'}]{item['source_type']}[/]\n{item['lift_note']}"
            )

        self.console.print(table)

    def show_reranker_shakeup(self, reranked_results: List[Dict[str, Any]]):
        """面板 3：Cross-Encoder 精排排位颠覆看板"""
        self.print_banner("3. 交叉重排 (Cross-Encoder) 颠覆榜")

        if not HAS_RICH:
            for item in reranked_results:
                print(f"  #{item['final_rank']} (原RRF #{item['rrf_rank']} 变动:{item['rank_delta']}) [{item['chunk_id']}] 得分: {item['rerank_score']} 状态: {item['shake_status']}")
            return

        table = Table(title="Cross-Encoder 全注意力细粒度打分与名次洗牌", show_lines=True)
        table.add_column("精排名次", justify="center", style="bold green", width=10)
        table.add_column("切片出处与标题", style="white", width=40)
        table.add_column("初排RRF", justify="center", style="yellow", width=9)
        table.add_column("名次升降", justify="center", style="bold", width=10)
        table.add_column("精排得分", justify="right", style="bold cyan", width=10)
        table.add_column("重排诊断评定", style="bold", width=24)

        for item in reranked_results:
            delta = item["rank_delta"]
            delta_str = f"[green]+{delta} UP[/green]" if delta > 0 else (f"[red]{delta} DOWN[/red]" if delta < 0 else "[dim]0[/dim]")
            
            table.add_row(
                f"#{item['final_rank']}",
                f"[{item['domain'][:4]}] {item['heading_path']}",
                f"#{item['rrf_rank']}",
                delta_str,
                f"{item['rerank_score']:.4f}",
                item["shake_status"]
            )

        self.console.print(table)

    def show_ablation_comparison(self, query: str, rrf_top: List[Dict[str, Any]], reranked_top: List[Dict[str, Any]]):
        """面板 6：重排序消融实验对比看板 (Ablation Study)"""
        self.print_banner("重排序消融实验对比看板 (Ablation: No-Rerank vs With-Rerank)")

        if not HAS_RICH:
            print(f"Query: {query}")
            print("【对照组 A：无重排 (纯 RRF 融合 Top-5)】")
            for item in rrf_top[:5]:
                print(f"  #{item['final_rank']} [{item['chunk_id']}] {item['heading_path']}")
            print("\n【实验组 B：加入 Cross-Encoder 重排后 Top-5】")
            for item in reranked_top[:5]:
                print(f"  #{item['final_rank']} (原RRF #{item['rrf_rank']} {item['rank_delta']}) [{item['chunk_id']}] {item['heading_path']}")
            return

        table = Table(title=f"Ablation Query: [yellow]{query}[/yellow]", show_lines=True)
        table.add_column("Rank", justify="center", style="bold green", width=6)
        table.add_column("对照组 A：消融重排 (仅 RRF 融合)", style="yellow", width=42)
        table.add_column("实验组 B：完整系统 (Cross-Encoder 精排)", style="cyan", width=42)
        table.add_column("消融位次变化", justify="center", style="bold", width=14)

        for i in range(min(len(rrf_top), len(reranked_top), 5)):
            a_item = rrf_top[i]
            b_item = reranked_top[i]
            delta = b_item["rank_delta"]
            delta_str = f"[green]+{delta} UP[/green]" if delta > 0 else (f"[red]{delta} DOWN[/red]" if delta < 0 else "[dim]0 (持平)[/dim]")

            a_str = f"[{a_item['chunk_id']}]\n{a_item['heading_path']}"
            b_str = f"[{b_item['chunk_id']}]\n{b_item['heading_path']}\n[dim]得分: {b_item['rerank_score']}[/dim]"

            table.add_row(f"#{i+1}", a_str, b_str, delta_str)

        self.console.print(table)

    def show_prompt_payload(self, system_prompt: str, user_prompt: str):
        """面板 4：真实 Prompt 组装透视"""
        self.print_banner("4. 真实喂给 DeepSeek 的上下文 Payload 透视")
        if not HAS_RICH:
            print("\n--- System Prompt ---\n" + system_prompt)
            print("\n--- User Prompt Snippet ---\n" + user_prompt[:400] + "...\n")
            return

        self.console.print(Panel(system_prompt, title="[bold blue]System Prompt (防幻觉约束)[/bold blue]", border_style="blue"))
        self.console.print(Panel(user_prompt[:600] + "\n... [已折叠部分参考切片内容] ...", title="[bold green]User Context Payload (注入切片)[/bold green]", border_style="green"))

    def show_stream_header(self):
        """面板 5 头部"""
        self.print_banner("5. DeepSeek 智能流式生成回答 (含引用溯源 [1],[2])")
        if HAS_RICH:
            self.console.print("[dim]正在接收 DeepSeek 流式 Token 响应...[/dim]\n")
        else:
            print("正在生成回答...\n")

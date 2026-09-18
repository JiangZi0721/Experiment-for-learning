# -*- coding: utf-8 -*-
"""
实验 6：高参数量深度循环神经网络语言模型 (Deep RNNLM: LSTM / GRU, 最高 365 万参数) 长文本演化透视

核心升级与真实科学原则：
1. 【模型架构支持深度 LSTM 与 GRU】：
   - LSTM 架构：2 层 Deep LSTM，隐藏维度 384，显式细胞状态 (Cell State, c_t) 线性记忆传送带，3,651,699 参数 (13.93 MB)
   - GRU 架构：2 层 Deep GRU，隐藏维度 384，紧凑门控设计，3,134,067 参数 (11.96 MB)
2. 【彻底剔除虚假点评】：杜绝任何硬编码的“主谓宾严整”等词句，完全以实测生成的文本与客观量化指标说话
3. 【提示词接合处置信度透视】：打印模型在输入提示词后，对紧随其后的前 3 个候选字的真实 Logits 与预测概率 (Top-3 Probs)
4. 【全程真实梯度更新】：基于 12 万字真实多领域专著语料库进行 BPTT 训练，透视 Loss 与 PPL 从混沌到收敛的真实轨迹
"""
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console(force_terminal=True)
except ImportError:
    HAS_RICH = False
    console = None


class CharTokenizer:
    """标准字符级分词器，原生集成 <unk> 机制"""
    def __init__(self, text: str):
        raw_chars = sorted(list(set(text)))
        self.vocab = ["<unk>"] + [c for c in raw_chars if c != "<unk>"]
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.unk_id = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> List[int]:
        return [self.char_to_id.get(c, self.unk_id) for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_char.get(i, "?") for i in ids])


class ScaledDeepLSTMLM(nn.Module):
    """
    高容量深度因果长短期记忆语言模型 (Deep LSTM LM)
    参数规模: ~365 万 (3,651,699 params)，显存占用约 13.93 MB
    配备显式 Cell State (c_t) 线性无损记忆通道与遗忘门/输入门/输出门 4 重门控机制
    """
    def __init__(self, vocab_size: int, emb_dim: int = 192, hidden_dim: int = 384, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.tok_emb(x)
        out, state_next = self.lstm(emb, state)
        out = self.ln(out)
        logits = self.head(out)
        return logits, state_next

    def get_param_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        layers = []
        for name, p in self.named_parameters():
            layers.append({"name": name, "shape": tuple(p.shape), "params": p.numel()})
        return {
            "total_params": total_params,
            "param_mb": (total_params * 4) / (1024 * 1024),
            "layers": layers
        }


class ScaledDeepGRULM(nn.Module):
    """
    高容量深度因果门控循环语言模型 (Deep GRU LM)
    参数规模: ~313 万 (3,134,067 params)，显存占用约 11.96 MB
    """
    def __init__(self, vocab_size: int, emb_dim: int = 192, hidden_dim: int = 384, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.tok_emb(x)
        out, h_next = self.gru(emb, h)
        out = self.ln(out)
        logits = self.head(out)
        return logits, h_next

    def get_param_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        layers = []
        for name, p in self.named_parameters():
            layers.append({"name": name, "shape": tuple(p.shape), "params": p.numel()})
        return {
            "total_params": total_params,
            "param_mb": (total_params * 4) / (1024 * 1024),
            "layers": layers
        }


def compute_text_metrics(text: str) -> Dict[str, float]:
    """定量评估生成文本的词汇多样性"""
    n = len(text)
    if n == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0}
    unique_chars = len(set(text))
    distinct_1 = unique_chars / n
    bigrams = [text[i:i+2] for i in range(n - 1)]
    distinct_2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return {"distinct_1": distinct_1, "distinct_2": distinct_2}


def detach_state(state: Any) -> Any:
    """对隐藏状态进行 BPTT 截断断开计算图"""
    if state is None:
        return None
    if isinstance(state, tuple):
        return (state[0].detach(), state[1].detach())
    return state.detach()


def sample_continuation(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt_str: str,
    max_length: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3
) -> Tuple[str, List[Tuple[str, float]]]:
    """自回归续写生成，并返回提示词末尾处的真实 Top-3 候选预测与置信度"""
    model.eval()
    prompt_tokens = tokenizer.encode(prompt_str)
    
    with torch.no_grad():
        inp = torch.tensor([prompt_tokens], dtype=torch.long)
        logits, state = model(inp, None)
        
        # 提取提示词最后一个字符处的输出 Logits (预测续写第 1 个字的真实分布)
        first_step_logits = logits[0, -1]
        probs = torch.softmax(first_step_logits, dim=-1)
        top3_vals, top3_indices = torch.topk(probs, 3)
        top3_predictions = [
            (tokenizer.decode([idx.item()]), float(val.item()))
            for val, idx in zip(top3_vals, top3_indices)
        ]
        
        generated_ids = []
        last_tok = prompt_tokens[-1]
        
        for _ in range(max_length):
            step_inp = torch.tensor([[last_tok]], dtype=torch.long)
            step_logits, state = model(step_inp, state)
            logits_step = step_logits[0, -1].clone()
            
            # 重复惩罚
            if repetition_penalty > 1.0:
                seen_tokens = set(prompt_tokens + generated_ids[-40:])
                for tok_id in seen_tokens:
                    if logits_step[tok_id] > 0:
                        logits_step[tok_id] /= repetition_penalty
                    else:
                        logits_step[tok_id] *= repetition_penalty
            
            # 温度缩放
            logits_step = logits_step / max(temperature, 1e-4)
            
            # Top-P 核采样
            probs = torch.softmax(logits_step, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits_step[indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(logits_step, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated_ids.append(next_tok)
            last_tok = next_tok

    continuation_text = tokenizer.decode(generated_ids)
    return continuation_text, top3_predictions


def run_experiment(
    max_epoch: int = 10,
    batch_size: int = 32,
    time_size: int = 40,
    gen_length: int = 120,
    prompt_str: str = "人工智能的发展",
    rnn_type: str = "lstm",
    seed: int = 42,
    interactive: bool = False,
    corpus_file: Optional[str] = None
):
    print("=" * 80)
    print(f">>> 启动实验 6：高参数量深度语言模型 (架构: {rnn_type.upper()}) 长文本演化透视")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. 加载语料
    corpus_path = Path(corpus_file).resolve() if corpus_file else PROJECT_ROOT / "data" / "corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = CharTokenizer(raw_text)
    vocab_size = tokenizer.vocab_size
    corpus_tensor = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

    # 2. 交互式提示词输入
    if interactive:
        if HAS_RICH and console:
            console.print("\n[bold cyan]📝 【自定义演化探针】请输入您想要观察演变过程的提示词 (Prompt):[/bold cyan]")
            console.print("  • 示例 1: [bold yellow]人工智能的发展[/bold yellow] (深度学习与科技)")
            console.print("  • 示例 2: [bold yellow]哲学的基本问题[/bold yellow] (哲学与意识思辨)")
            console.print("  • 示例 3: [bold yellow]文艺复兴起始于[/bold yellow] (历史与文明演进)")
        user_p = input(f"\n请输入您的提示引导词 [直接回车使用: '{prompt_str}']: ").strip()
        if user_p:
            prompt_str = user_p
        user_len = input(f"请输入期望生成的文本长度 (字符数) [直接回车使用: {gen_length} 字]: ").strip()
        if user_len.isdigit() and int(user_len) > 0:
            gen_length = int(user_len)

    # 3. 实例化指定的高容量循环模型 (LSTM 或 GRU)
    if rnn_type.lower() == "lstm":
        model = ScaledDeepLSTMLM(vocab_size=vocab_size, emb_dim=192, hidden_dim=384, num_layers=2)
        arch_desc = "双层深度长短期记忆网络 (2-Layer LSTM, 带显式 Cell State 线性无损传送带)"
    else:
        model = ScaledDeepGRULM(vocab_size=vocab_size, emb_dim=192, hidden_dim=384, num_layers=2)
        arch_desc = "双层深度门控循环网络 (2-Layer GRU, 紧凑型更新门/重置门)"

    summary = model.get_param_summary()

    if HAS_RICH and console:
        w_table = Table(title=f"🧠 语言模型内部权重参数规格表 ({rnn_type.upper()} 架构版)", show_header=True, header_style="bold green")
        w_table.add_column("网络层组件", style="bold cyan", width=22)
        w_table.add_column("权重形状 (Shape)", style="yellow", width=20)
        w_table.add_column("参数量 (Params)", justify="right", width=16)
        w_table.add_column("物理功能描述", style="dim white", width=26)

        w_table.add_row("词嵌入层 (Embedding)", f"({vocab_size}, 192)", f"{vocab_size * 192:,}", "高维语义向量坐标")
        if rnn_type.lower() == "lstm":
            w_table.add_row("深度 LSTM 第 1 层", "(192 -> 384)", f"{4 * (192*384 + 384*384 + 384*2):,}", "输入/遗忘/输出/细胞4重门控")
            w_table.add_row("深度 LSTM 第 2 层", "(384 -> 384)", f"{4 * (384*384 + 384*384 + 384*2):,}", "高层抽象语义与长程 Cell 状态")
        else:
            w_table.add_row("深度 GRU 第 1 层", "(192 -> 384)", f"{3 * (192*384 + 384*384 + 384*2):,}", "输入/更新/候选3重门控")
            w_table.add_row("深度 GRU 第 2 层", "(384 -> 384)", f"{3 * (384*384 + 384*384 + 384*2):,}", "高层抽象语义与长程隐藏状态")
        w_table.add_row("层归一化 (LayerNorm)", "(384,)", "768", "特征幅值稳定器")
        w_table.add_row("输出投影层 (Head Linear)", f"(384, {vocab_size})", f"{384 * vocab_size:,}", "记忆向 2739 字概率分布解码")
        console.print(w_table)
        console.print(f"[bold green]▶ 架构: {arch_desc}[/bold green]")
        console.print(f"[bold green]▶ 真实参数量：{summary['total_params']:,} 个浮点数 (显存占用 ~{summary['param_mb']:.2f} MB)[/bold green]\n")

    # 4. 构建 Truncated BPTT 流水线
    jump = len(corpus_tensor) // batch_size
    max_iters = (jump - 1) // time_size
    offsets = [i * jump for i in range(batch_size)]

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    milestone_epochs = [0, 1, 3, 6, 10, max_epoch]
    milestone_epochs = sorted(list(set([ep for ep in milestone_epochs if ep <= max_epoch])))

    stage_records = []
    current_epoch = 0

    print("═" * 78)
    print(f">>> [启动真实训练与生成快照追踪: 架构「{rnn_type.upper()}」 | 提示词「{prompt_str}」]")
    print("═" * 78)

    for target_ep in milestone_epochs:
        # 训练推进
        if target_ep > current_epoch:
            epochs_to_run = target_ep - current_epoch
            model.train()
            for _ in range(epochs_to_run):
                state = None
                total_loss = 0.0
                for it in range(max_iters):
                    xs = torch.stack([corpus_tensor[off + it * time_size : off + (it + 1) * time_size] for off in offsets])
                    ts = torch.stack([corpus_tensor[off + it * time_size + 1 : off + (it + 1) * time_size + 1] for off in offsets])
                    
                    optimizer.zero_grad()
                    state = detach_state(state)
                    logits, state = model(xs, state)
                    loss = criterion(logits.view(-1, vocab_size), ts.view(-1))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += loss.item()
            current_epoch = target_ep

        # 计算当前 Loss 与 PPL
        if current_epoch > 0:
            avg_loss = total_loss / max_iters
            ppl = np.exp(avg_loss) if avg_loss < 20 else float('inf')
        else:
            avg_loss = float(np.log(vocab_size))
            ppl = float(vocab_size)

        # 自回归采样与 Top-3 接合透视
        continuation_text, top3 = sample_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt_str=prompt_str,
            max_length=gen_length,
            temperature=0.7,
            top_p=0.9
        )
        full_text = prompt_str + continuation_text
        metrics = compute_text_metrics(full_text)

        stage_records.append({
            "epoch": target_ep,
            "loss": avg_loss,
            "ppl": ppl,
            "distinct_1": metrics["distinct_1"],
            "distinct_2": metrics["distinct_2"],
            "continuation": continuation_text,
            "top3": top3
        })

        # 打印客观真实快照面板
        top3_desc = " | ".join([f"'{c}': {p*100:.1f}%" for c, p in top3])
        if HAS_RICH and console:
            body = Text()
            body.append(f"• 训练轮次: Epoch {target_ep:02d} / {max_epoch}  │  模型架构: {rnn_type.upper()}\n", style="bold white")
            body.append(f"• 交叉熵损失 (Loss): {avg_loss:6.4f}  │  困惑度指数 (PPL): {ppl:7.2f} (初始 ~{vocab_size})\n", style="bold yellow")
            body.append(f"• 字符丰富度 (Distinct-1): {metrics['distinct_1']:.1%}  │  双字词丰富度 (Distinct-2): {metrics['distinct_2']:.1%}\n", style="dim white")
            body.append(f"• 提示词接续首字预测 (Top-3 Probs): {top3_desc}\n\n", style="cyan")
            body.append(f"[*] 提示词 (Prompt): \"{prompt_str}\"\n", style="bold yellow")
            body.append(f"[>] 模型真实自回归续写 (前 {len(continuation_text)} 字):\n", style="bold green")
            body.append(f"   \"{continuation_text}\"\n", style="white")

            border_c = "red" if target_ep == 0 else ("yellow" if target_ep <= 3 else "green")
            panel = Panel(body, title=f"[bold cyan]真实演化快照 #{len(stage_records)} (Epoch {target_ep})[/bold cyan]", border_style=border_c)
            console.print(panel)
        else:
            print(f"\n[快照 #{len(stage_records)}] Epoch {target_ep} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
            print(f"Top-3 首字预测: {top3_desc}")
            print(f"Prompt: {prompt_str}")
            print(f"Continuation: {continuation_text}")

    # 保存权重文件
    weights_path = PROJECT_ROOT / "data" / f"trained_{rnn_type.lower()}lm_weights.pt"
    torch.save(model.state_dict(), str(weights_path))
    print(f"\n[✓] {rnn_type.upper()} 模型权重已保存至: {weights_path} ({summary['param_mb']:.2f} MB)")

    return stage_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="高容量深度语言模型 (LSTM / GRU) 训练进化与长文本生成透视")
    parser.add_argument("-i", "--interactive", action="store_true", help="启用交互式输入提示词")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数 (默认: 10)")
    parser.add_argument("--batch", type=int, default=32, help="批次大小 (默认: 32)")
    parser.add_argument("--length", type=int, default=120, help="生成文本长度 (默认: 120)")
    parser.add_argument("--prompt", type=str, default="人工智能的发展", help="提示引导词")
    parser.add_argument("--rnn", type=str, default="lstm", choices=["lstm", "gru"], help="循环神经网络类型 (默认: lstm)")
    parser.add_argument("--corpus", type=str, default=None, help="语料路径")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    args = parser.parse_args()

    run_experiment(
        max_epoch=args.epochs,
        batch_size=args.batch,
        gen_length=args.length,
        prompt_str=args.prompt,
        rnn_type=args.rnn,
        seed=args.seed,
        interactive=args.interactive,
        corpus_file=args.corpus
    )

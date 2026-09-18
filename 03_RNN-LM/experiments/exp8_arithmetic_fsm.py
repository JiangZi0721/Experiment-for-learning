# -*- coding: utf-8 -*-
"""
实验 8：自回归语言模型在算术加法任务中的“字符反转（Character Reversal）”效应严格验证实验

【待证科学假设】：
“在训练自回归语言模型处理多位数加法时，将数字字符串进行字符反转（特别是输出端反转，即低位到高位 LSD-first），
能消除自回归左到右生成与算术进位低位到高位之间的因果冲突，显著提升收敛速度、加法准确率及泛化能力。”

【三组严格对照组设计】：
1. 组 1【标准正序基线组 (Standard Plain)】：
   格式: "$123+456=579$" (Prompt: "$123+456=", Target: "579$")
   机制: 最高位先行 (MSD-first)，存在严重的因果掩码前瞻倒置 (Causal Inversion)。

2. 组 2【逆序结果实验组 (Reverse Output / LSD-first Target)】：
   格式: "$123+456=975$" (Prompt: "$123+456=", Target: "975$")
   机制: 最低位先行 (LSD-first)，输出生成流向与进位传播物理流向完全重合（Causal Alignment）。

3. 组 3【全逆序对齐实验组 (Full Reverse / Both Inputs & Output LSD-first)】：
   格式: "$321+654=975$" (Prompt: "$321+654=", Target: "975$")
   机制: 输入操作数与输出结果均按低位在前排列，单步对齐马尔可夫自动机。

【评测维度】：
1. 训练收敛速度（Loss 下降与训练耗时）
2. 序列级全匹配准确率 (Exact Match EM Accuracy, %)
3. 进位深度敏感度分析 (0次进位 / 1次进位 / 2次进位 / 3次级联进位 准确率切片)
4. 长度外推泛化测试 (Length Generalization / OOD: 3位数训练 -> 4位数与5位数测试)
"""

import sys
import os
import time
import random
import tempfile
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

def patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        os.mkdir(str(self))
    except FileExistsError:
        if not exist_ok:
            raise
    except FileNotFoundError:
        if parents:
            self.parent.mkdir(mode, parents=True, exist_ok=True)
            os.mkdir(str(self))
        else:
            raise

pathlib.Path.mkdir = patched_mkdir
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

# 固定全局随机种子以确保 100% 科学可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 字符词表与编码 (14个独立 token)
VOCAB = list("$0123456789+=\n")
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(VOCAB)

def count_carries(a: int, b: int) -> int:
    """统计两个非负整数相加时的真实进位次数"""
    carry = 0
    count = 0
    sa, sb = str(a)[::-1], str(b)[::-1]
    max_len = max(len(sa), len(sb))
    sa = sa.ljust(max_len, '0')
    sb = sb.ljust(max_len, '0')
    for da, db in zip(sa, sb):
        total = int(da) + int(db) + carry
        if total >= 10:
            count += 1
            carry = 1
        else:
            carry = 0
    return count

def generate_addition_dataset(num_samples: int, min_digits: int, max_digits: int) -> List[Tuple[int, int, int, int]]:
    """生成包含 (a, b, sum, carries) 的算术加法数据集"""
    dataset = []
    seen = set()
    attempts = 0
    while len(dataset) < num_samples and attempts < num_samples * 10:
        attempts += 1
        d_a = random.randint(min_digits, max_digits)
        d_b = random.randint(min_digits, max_digits)
        a = random.randint(10**(d_a - 1), 10**d_a - 1)
        b = random.randint(10**(d_b - 1), 10**d_b - 1)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        c = a + b
        carries = count_carries(a, b)
        dataset.append((a, b, c, carries))
    return dataset

def format_sample(a: int, b: int, c: int, mode: str) -> Tuple[str, str]:
    """格式化为指定实验组的提示词 (Prompt) 与期望输出 (Target)"""
    sa, sb, sc = str(a), str(b), str(c)
    if mode == "standard":
        prompt = f"${sa}+{sb}="
        target = f"{sc}$"
    elif mode == "reverse_output":
        prompt = f"${sa}+{sb}="
        target = f"{sc[::-1]}$"
    elif mode == "full_reverse":
        prompt = f"${sa[::-1]}+{sb[::-1]}="
        target = f"{sc[::-1]}$"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return prompt, target

class CausalTransformerLM(nn.Module):
    """
    符合现代因果自回归架构的紧凑型 Transformer 语言模型 (NanoGPT 规范)
    用于检验注意力因果掩码机制在不同序列格式下的算法对齐效应
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_len: int = 32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out = self.transformer(h, mask=causal_mask, is_causal=True)
        out = self.ln_f(out)
        return self.head(out)

def prepare_tensors(dataset: List[Tuple[int, int, int, int]], mode: str, max_seq_len: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建批处理张量，因果自回归目标只对等号之后的结果部分计算 Loss"""
    X_list, Y_list = [], []
    for a, b, c, _ in dataset:
        prompt, target = format_sample(a, b, c, mode)
        full_seq = prompt + target
        toks = [CHAR2IDX[ch] for ch in full_seq]
        x = toks[:-1]
        y = toks[1:]
        p_len = len(prompt)
        # 仅对答案 token 计算损失，提示词部分全部置为 -100 屏蔽梯度
        masked_y = [-100 if i < p_len - 1 else y[i] for i in range(len(y))]
        pad = max_seq_len - len(x)
        x = x + [0] * pad
        masked_y = masked_y + [-100] * pad
        X_list.append(x)
        Y_list.append(masked_y)
    return torch.tensor(X_list, dtype=torch.long), torch.tensor(Y_list, dtype=torch.long)

def evaluate_accuracy(model: nn.Module, test_set: List[Tuple[int, int, int, int]], mode: str, max_gen: int = 7) -> Dict[str, Any]:
    """严格贪婪搜索评测，计算序列完全匹配率 (Exact Match EM) 以及按进位次数细分准确率"""
    model.eval()
    correct_count = 0
    total_count = len(test_set)
    carry_stats = {} # carry_num -> [correct, total]

    with torch.no_grad():
        for a, b, c, carries in test_set:
            if carries not in carry_stats:
                carry_stats[carries] = [0, 0]
            carry_stats[carries][1] += 1

            prompt, expected_target = format_sample(a, b, c, mode)
            cur_tokens = [CHAR2IDX[ch] for ch in prompt]
            
            for _ in range(max_gen):
                inp = torch.tensor([cur_tokens], dtype=torch.long)
                logits = model(inp)
                pred_tok = torch.argmax(logits[0, -1]).item()
                cur_tokens.append(pred_tok)
                if IDX2CHAR[pred_tok] == '$':
                    break
            
            gen_str = "".join([IDX2CHAR[t] for t in cur_tokens])
            gen_answer = gen_str[len(prompt):]
            
            is_match = (gen_answer == expected_target)
            if is_match:
                correct_count += 1
                carry_stats[carries][0] += 1

    overall_em = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0
    carry_accs = {k: (v[0] / v[1] * 100.0) if v[1] > 0 else 0.0 for k, v in carry_stats.items()}
    return {
        "overall_em": overall_em,
        "carry_accs": carry_accs,
        "correct": correct_count,
        "total": total_count
    }

def run_experiment(epochs: int = 35, num_train: int = 3000, num_test: int = 200, num_ood: int = 150):
    print("=" * 80)
    print(">>> 启动实验 8：自回归算术加法【字符反转假设】严格对照科学验证")
    print("=" * 80)

    # 1. 准备数据集
    print(f"[*] 正在生成受控算术数据集 (训练集: {num_train} 样本 3位数; 域内测试集: {num_test} 样本 3位数)...")
    train_data = generate_addition_dataset(num_train, min_digits=3, max_digits=3)
    test_id = generate_addition_dataset(num_test, min_digits=3, max_digits=3)
    
    print(f"[*] 正在生成长度外推 OOD 数据集 ({num_ood} 样本 4~5 位数)...")
    test_ood_4d = generate_addition_dataset(num_ood, min_digits=4, max_digits=4)
    test_ood_5d = generate_addition_dataset(num_ood, min_digits=5, max_digits=5)

    groups = [
        ("standard", "组 1: 标准正序基线组 (Standard Plain)"),
        ("reverse_output", "组 2: 逆序结果对齐组 (Reverse Output)"),
        ("full_reverse", "组 3: 全逆序对齐组 (Full Reverse)")
    ]

    history = {g[0]: {"epochs": [], "loss": [], "id_em": [], "ood_4d_em": [], "ood_5d_em": [], "carry_accs": {}} for g in groups}
    trained_models = {}

    batch_size = 64
    eval_interval = 5

    for mode_key, mode_title in groups:
        print("\n" + "#" * 80)
        print(f"### 开始训练评测: {mode_title}")
        print("#" * 80)

        # 构造训练 DataLoader
        X_train, Y_train = prepare_tensors(train_data, mode_key, max_seq_len=24)
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 实例化全新的因果 Transformer
        torch.manual_seed(SEED) # 保证初始参数起点完全一致
        model = CausalTransformerLM(vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=2)
        optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        t_start = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits.view(-1, VOCAB_SIZE), by.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(bx)
            
            avg_loss = epoch_loss / len(X_train)

            if ep % eval_interval == 0 or ep == epochs:
                # 域内测试集评估
                eval_res = evaluate_accuracy(model, test_id, mode_key, max_gen=6)
                # 长度外推评估
                eval_ood_4d = evaluate_accuracy(model, test_ood_4d, mode_key, max_gen=8)
                eval_ood_5d = evaluate_accuracy(model, test_ood_5d, mode_key, max_gen=9)

                history[mode_key]["epochs"].append(ep)
                history[mode_key]["loss"].append(avg_loss)
                history[mode_key]["id_em"].append(eval_res["overall_em"])
                history[mode_key]["ood_4d_em"].append(eval_ood_4d["overall_em"])
                history[mode_key]["ood_5d_em"].append(eval_ood_5d["overall_em"])
                if ep == epochs:
                    history[mode_key]["carry_accs"] = eval_res["carry_accs"]

                print(f"  [Epoch {ep:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | "
                      f"3位EM准确率: {eval_res['overall_em']:5.1f}% | "
                      f"4位OOD: {eval_ood_4d['overall_em']:4.1f}% | "
                      f"5位OOD: {eval_ood_5d['overall_em']:4.1f}%", flush=True)

        elapsed = time.time() - t_start
        history[mode_key]["elapsed_seconds"] = elapsed
        trained_models[mode_key] = model
        print(f"[*] {mode_title} 训练完毕，总耗时: {elapsed:.2f}s")

    # 打印最终对比表
    print("\n" + "=" * 80)
    print(">>> 实验 8 最终统计与因果进位切片全景分析")
    print("=" * 80)

    if HAS_RICH and console:
        table = Table(title="[bold green]实验 8：算术加法序列反转假设对照评测结果 (3位数加法)[/bold green]", show_header=True, header_style="bold magenta")
        table.add_column("实验组别", style="cyan", justify="left")
        table.add_column("最终 Loss", justify="right")
        table.add_column("3位域内准确率 (EM)", justify="right")
        table.add_column("4位外推 (OOD)", justify="right")
        table.add_column("5位外推 (OOD)", justify="right")
        table.add_column("0次进位准确率", justify="right")
        table.add_column("1次进位准确率", justify="right")
        table.add_column("2次进位准确率", justify="right")
        table.add_column("3次进位准确率", justify="right")

        for key, title in groups:
            h = history[key]
            c_accs = h["carry_accs"]
            c0 = f"{c_accs.get(0, 0.0):.1f}%"
            c1 = f"{c_accs.get(1, 0.0):.1f}%"
            c2 = f"{c_accs.get(2, 0.0):.1f}%"
            c3 = f"{c_accs.get(3, 0.0):.1f}%"
            table.add_row(
                title.split(":")[1].strip(),
                f"{h['loss'][-1]:.4f}",
                f"[bold]{h['id_em'][-1]:.1f}%[/bold]",
                f"{h['ood_4d_em'][-1]:.1f}%",
                f"{h['ood_5d_em'][-1]:.1f}%",
                c0, c1, c2, c3
            )
        console.print(table)
    else:
        for key, title in groups:
            h = history[key]
            print(f"{title}: Final Loss={h['loss'][-1]:.4f}, 3D EM={h['id_em'][-1]:.1f}%, 4D OOD={h['ood_4d_em'][-1]:.1f}%, 5D OOD={h['ood_5d_em'][-1]:.1f}%")

    # 打印真实生成样本对照
    print("\n" + "-" * 80)
    print(">>> 真实算术算式生成对比 (随机选取 5 个真实验证用例):")
    print("-" * 80)
    for a, b, c, carries in test_id[:5]:
        print(f"算式: {a} + {b} = {c} (实际进位次数: {carries})")
        for key, title in groups:
            p, exp = format_sample(a, b, c, key)
            model = trained_models[key]
            model.eval()
            with torch.no_grad():
                cur_tokens = [CHAR2IDX[ch] for ch in p]
                for _ in range(6):
                    inp = torch.tensor([cur_tokens], dtype=torch.long)
                    pred_tok = torch.argmax(model(inp)[0, -1]).item()
                    cur_tokens.append(pred_tok)
                    if IDX2CHAR[pred_tok] == '$':
                        break
                gen_str = "".join([IDX2CHAR[t] for t in cur_tokens])
                gen_ans = gen_str[len(p):]
                status = "✓ 正确" if gen_ans == exp else "✗ 错误"
                print(f"   [{key:14s}] 输入: {p:12s} | 生成: {gen_ans:6s} (期望: {exp:6s}) -> {status}")

    # 绘制高分辨率科研对比图
    if HAS_MATPLOTLIB:
        out_img = PROJECT_ROOT / "data" / "arithmetic_reversal_comparison.png"
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        colors = {"standard": "#e74c3c", "reverse_output": "#2ecc71", "full_reverse": "#3498db"}
        labels = {"standard": "Standard Plain (MSD-first)", "reverse_output": "Reverse Output (LSD-first)", "full_reverse": "Full Reverse (LSD-first)"}

        # 子图 1: 交叉熵损失曲线
        ax1 = axes[0, 0]
        for key in ["standard", "reverse_output", "full_reverse"]:
            ax1.plot(history[key]["epochs"], history[key]["loss"], marker='o', color=colors[key], label=labels[key], linewidth=2.0)
        ax1.set_title("Training Cross-Entropy Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 子图 2: 域内 3 位数全匹配准确率 (Exact Match EM)
        ax2 = axes[0, 1]
        for key in ["standard", "reverse_output", "full_reverse"]:
            ax2.plot(history[key]["epochs"], history[key]["id_em"], marker='s', color=colors[key], label=labels[key], linewidth=2.0)
        ax2.set_title("In-Distribution 3-Digit Addition Accuracy (EM %)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Exact Match (%)", fontsize=10)
        ax2.set_ylim(-2, 102)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        # 子图 3: 进位敏感度柱状分析 (0次 vs 1次 vs 2次 vs 3次进位)
        ax3 = axes[1, 0]
        carry_keys = [0, 1, 2, 3]
        bar_width = 0.25
        x_indices = np.arange(len(carry_keys))

        for idx, key in enumerate(["standard", "reverse_output", "full_reverse"]):
            c_accs = history[key]["carry_accs"]
            vals = [c_accs.get(ck, 0.0) for ck in carry_keys]
            ax3.bar(x_indices + idx * bar_width, vals, width=bar_width, color=colors[key], label=labels[key], alpha=0.85)

        ax3.set_title("Accuracy Breakdown by Number of Carries", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Number of Carries in Addition", fontsize=10)
        ax3.set_ylabel("Exact Match Accuracy (%)", fontsize=10)
        ax3.set_xticks(x_indices + bar_width)
        ax3.set_xticklabels(["0 Carries", "1 Carry", "2 Carries", "3 Carries"])
        ax3.set_ylim(0, 105)
        ax3.grid(True, linestyle="--", alpha=0.5, axis='y')
        ax3.legend()

        # 子图 4: 长度外推泛化评测 (In-Distribution vs 4-Digit OOD vs 5-Digit OOD)
        ax4 = axes[1, 1]
        ood_cats = ["3-Digit (Train)", "4-Digit (OOD)", "5-Digit (OOD)"]
        x_ood = np.arange(len(ood_cats))

        for idx, key in enumerate(["standard", "reverse_output", "full_reverse"]):
            vals = [
                history[key]["id_em"][-1],
                history[key]["ood_4d_em"][-1],
                history[key]["ood_5d_em"][-1]
            ]
            ax4.bar(x_ood + idx * bar_width, vals, width=bar_width, color=colors[key], label=labels[key], alpha=0.85)

        ax4.set_title("Length Generalization / Out-of-Distribution (OOD)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Digit Length Evaluation", fontsize=10)
        ax4.set_ylabel("Exact Match Accuracy (%)", fontsize=10)
        ax4.set_xticks(x_ood + bar_width)
        ax4.set_xticklabels(ood_cats)
        ax4.set_ylim(0, 105)
        ax4.grid(True, linestyle="--", alpha=0.5, axis='y')
        ax4.legend()

        plt.suptitle("Causal Autoregressive LM: Plain vs Reverse Format in Arithmetic Addition", fontsize=14, fontweight="bold", y=0.98)
        fig.savefig(str(out_img), bbox_inches='tight')
        plt.close(fig)
        print(f"[✓] 高分辨率对比图已保存至: {out_img}")

    return history

if __name__ == "__main__":
    run_experiment(epochs=35, num_train=3000, num_test=200, num_ood=150)

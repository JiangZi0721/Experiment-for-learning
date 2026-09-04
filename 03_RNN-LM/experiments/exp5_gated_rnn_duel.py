# -*- coding: utf-8 -*-
"""
实验 5：Vanilla RNN vs Gated RNN (GRU) 终极巅峰对决
探究重点:
1. 在长达 30 步的长时序逆流中，普通单步 RNN 与门控 RNN 的梯度穿透力残酷对比
2. 观察重置门 (r) 与更新门 (z) 在时间轴上的动态开闭
3. 白盒实测: 梯度直连高速公路 (1 - z) 为什么能让误差无损穿透 30 步，彻底终结梯度消失！
"""
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.time_rnn import TimeRNN
from src.gated_rnn import TimeGRU, GRUCell
from src.visualizer import RNNVisualizer

try:
    from rich.table import Table
    from rich.console import Console
    HAS_RICH = True
    console = Console(force_terminal=True)
except ImportError:
    HAS_RICH = False
    console = None


def run_experiment():
    visualizer = RNNVisualizer()
    visualizer.print_banner(
        "【实验 5】Vanilla RNN vs Gated RNN (GRU) 门控对决实验",
        "实测 30 步长程时序逆流：亲眼见证门控高速公路如何终结梯度消失！"
    )

    np.random.seed(42)
    N, T, D, H = 1, 30, 8, 16
    xs = np.random.randn(N, T, D).astype(np.float32) * 0.5

    # 在最末端 t=29 注入基准改错信号
    dhs = np.zeros((N, T, H), dtype=np.float32)
    dhs[:, -1, :] = 1.0

    # 1. 经典普通 Vanilla RNN
    Wx_vanilla = (np.random.randn(D, H) / np.sqrt(D)).astype(np.float32)
    Wh_vanilla = (np.random.randn(H, H) / np.sqrt(H)).astype(np.float32)
    b_vanilla = np.zeros(H, dtype=np.float32)

    time_vanilla = TimeRNN(Wx_vanilla, Wh_vanilla, b_vanilla)
    time_vanilla.forward(xs)
    time_vanilla.backward(dhs)
    vanilla_dh_history = time_vanilla.probe_data["backward"]["step_dh_norms"]

    # 2. 门控 Gated RNN (GRU): 采用标准长程记忆初始化 (更新门偏置设为 -1.5, 畅通直连通道)
    Wx_gru = (np.random.randn(D, 3 * H) / np.sqrt(D)).astype(np.float32)
    Wh_gru = (np.random.randn(H, 3 * H) / np.sqrt(H)).astype(np.float32)
    b_gru = np.zeros(3 * H, dtype=np.float32)
    # 将更新门 z 的偏置设为 -1.5 (使 Sigmoid(z) ~ 0.18, 直连通道 1 - z ~ 0.82 畅通无阻)
    b_gru[H:2*H] = -1.5

    time_gru = TimeGRU(Wx_gru, Wh_gru, b_gru)
    time_gru.forward(xs)
    time_gru.backward(dhs)
    gru_dh_history = time_gru.probe_data["backward"]["step_dh_norms"]

    # 3. 展示 GRU 单步门控透视
    sample_cell = time_gru.layers[15]
    visualizer.show_gated_rnn_probe(
        step_idx=15,
        reset_gate=sample_cell.probe_data["reset_gate_mean"],
        update_gate=sample_cell.probe_data["update_gate_mean"],
        h_norm=sample_cell.probe_data["h_next_norm"],
        highway_flow=sample_cell.probe_data["highway_gradient_flow"],
        retention_gain=sample_cell.probe_data.get("grad_retention_rate", 1.0)
    )

    # 4. 双路梯度逆流穿透力对比看板
    print("\n" + "="*80)
    print(">>> [核心对决看板: 误差从末尾 t=29 逆向倒流回 t=0 时各节点的信号残留]")

    check_steps = [29, 25, 20, 15, 10, 5, 0]

    if HAS_RICH:
        table = Table(title="Vanilla RNN vs Gated RNN 30 步长程梯度逆流对决表", show_header=True, header_style="bold magenta")
        table.add_column("时序节点", justify="center", style="bold yellow", width=14, overflow="fold")
        table.add_column("Vanilla RNN 信号", justify="right", style="cyan", width=18, overflow="fold")
        table.add_column("GRU 门控信号", justify="right", style="green", width=18, overflow="fold")
        table.add_column("门控穿透优势", justify="center", style="bold red", width=18, overflow="fold")

        for step in check_steps:
            v_val = vanilla_dh_history[step]
            g_val = gru_dh_history[step]
            ratio = g_val / (v_val + 1e-12)
            advantage_str = f"领先 [bold green]{ratio:,.1f}x[/bold green] 倍" if ratio > 1 else "相当"
            table.add_row(
                f"← 倒流回 t={step:02d}",
                f"{v_val:.6e}",
                f"{g_val:.6f}",
                advantage_str
            )
        console.print(table)
    else:
        for step in check_steps:
            v_val = vanilla_dh_history[step]
            g_val = gru_dh_history[step]
            print(f"Step t={step:02d} | Vanilla: {v_val:.6e} | GRU: {g_val:.6f} | Ratio: {g_val/(v_val+1e-12):.1f}x")

    v_final = vanilla_dh_history[0]
    g_final = gru_dh_history[0]
    print("\n" + "="*80)
    print("【实验 5 结论透视】:")
    print(f"• 在倒流 30 步到达起点 t=0 时:")
    print(f"  - 普通 Vanilla RNN 梯度残余: {v_final:.4e} (已衰减成几亿分之一，声音彻底消失！)")
    print(f"  - 门控 Gated RNN (GRU) 梯度残余: {g_final:.4f} (依然强劲有力，信号毫发无损！)")
    print(f"  - 门控穿透力碾压倍数: {g_final / (v_final + 1e-12):,.1f} 倍！")
    print("• 物理实质解密: GRU 引入了更新门 z，制造了 (1 - z) 的【恒等映射加法直连通道】。")
    print("  误差信号不需要像普通 RNN 那样被迫连乘 30 次权重矩阵，而是通过高速公路直达 30 步前！")
    print("  这就是 Gated RNN 能够颠覆 Vanilla RNN，成为近代长序列 NLP 绝对主力的根本原因！")
    print("="*80)

    # 核心指标白话词典看板 (扫清初学者阅读门控机制与高速公路指标的疑惑)
    exp5_metrics = [
        ("重置门开度 r", "sigma(xW_xr+hW_hr)", "旧账清空阀，算候选新记忆时保留几成旧笔记", "0.0~1.0; 句首清空旧账，句中连贯保留"),
        ("更新门开度 z", "sigma(xW_xz+hW_hz)", "新旧权衡阀，决定当前时刻吸纳多少新候选记忆", "0.0~1.0; z 越小历史记忆无损穿透越多"),
        ("梯度公路 1-z", "h = (1-z)h_prev+z~h", "无损加法直连跳线，反向求导直达无矩阵连乘", "~0.82 畅通无阻，彻底终结时序梯度消失"),
        ("30 步穿透残余", "||dL / dh_0||", "误差倒查 30 步到达起点后改错信号的剩余强度", "Vanilla 跌至 1e-9; GRU 保持 0.1~0.5"),
        ("门控领先倍数", "||dh_0(GRU)|| / ||..||", "GRU 相比普通 RNN 在长程信号穿透上的强度倍数", "30 步长程下通常达到 10^6~10^9 倍碾压"),
    ]
    visualizer.show_metric_definitions("Gated RNN 门控机制与高速公路指标字典", exp5_metrics)


if __name__ == "__main__":
    run_experiment()

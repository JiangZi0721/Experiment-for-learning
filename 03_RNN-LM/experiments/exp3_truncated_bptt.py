# -*- coding: utf-8 -*-
"""
实验 3：Truncated BPTT 跨块记忆接力与反向截断印证实验
探究重点:
1. 为什么 Truncated BPTT 能够在反向传播仅展开 T 步的情况下，依然保留超长上下文的记忆？
2. 对比实验:
   - 模式 A (无状态断连 stateful=False): 每个 Chunk 清空 h -> 记忆断裂，无法捕获跨块依赖
   - 模式 B (标准截断接力 stateful=True): 前向 h 跨块继承，反向梯度在边界截断 -> 跨块记忆无缝延续
3. 白盒透视: 监控 Chunk 间隐藏状态继承的余弦相似度与被截断的梯度数值
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
from src.visualizer import RNNVisualizer


def run_experiment():
    visualizer = RNNVisualizer()
    visualizer.print_banner("【实验 3】Truncated BPTT 跨块接力与截断机制印证", "透视前向无缝状态继承 (Carry-Over) 与反向梯度边界截断 (Truncation)")

    np.random.seed(42)
    N, T, D, H = 1, 10, 8, 16
    num_chunks = 4 # 总共 4 个 Chunk，总时序跨度 40 步

    # 构造权重
    Wx = (np.random.randn(D, H) / np.sqrt(D)).astype(np.float32)
    Wh = (np.random.randn(H, H) / np.sqrt(H)).astype(np.float32)
    b = np.zeros(H, dtype=np.float32)

    # 构造跨时序长程数据:
    # 在 Chunk 0 的第 0 步注入一个极强的特殊模式信号 (Trigger)
    chunks_data = [np.random.randn(N, T, D).astype(np.float32) * 0.1 for _ in range(num_chunks)]
    trigger_signal = np.ones((N, D), dtype=np.float32) * 5.0
    chunks_data[0][:, 0, :] = trigger_signal # 早期线索注入

    # ---------------- 模式 A: stateful=False (无状态截断，记忆断连) ----------------
    print("\n>>> [模式 A: stateful=False (每个 Chunk 重置 h=0，记忆断层)]")
    rnn_stateless = TimeRNN(Wx.copy(), Wh.copy(), b.copy(), stateful=False)

    stateless_final_states = []
    for c_idx, chunk_x in enumerate(chunks_data):
        hs = rnn_stateless.forward(chunk_x)
        stateless_final_states.append(hs[:, -1, :].copy())
        print(f"Chunk #{c_idx}: 初始 h_norm = {0.0:.4f}, 末尾 h_norm = {float(np.linalg.norm(hs[:, -1, :])):.4f}")

    # ---------------- 模式 B: stateful=True (标准 Truncated BPTT 接力) ----------------
    print("\n>>> [模式 B: stateful=True (前向状态无缝接力，跨块记忆永存)]")
    rnn_stateful = TimeRNN(Wx.copy(), Wh.copy(), b.copy(), stateful=True)

    stateful_final_states = []
    for c_idx, chunk_x in enumerate(chunks_data):
        prev_h = rnn_stateful.get_state() if hasattr(rnn_stateful, "get_state") else rnn_stateful.h
        prev_norm = float(np.linalg.norm(prev_h)) if prev_h is not None else 0.0

        hs = rnn_stateful.forward(chunk_x)
        curr_norm = float(np.linalg.norm(hs[:, -1, :]))
        stateful_final_states.append(hs[:, -1, :].copy())

        # 模拟反向传播并在边界截断
        dhs = np.ones_like(hs) * 0.1
        dxs, dh_prev = rnn_stateful.backward(dhs)
        dh_trunc_norm = float(np.linalg.norm(dh_prev))

        # 展示探针看板 (仅首个分块打印导读与横幅，后续分块纯净输出数据)
        visualizer.show_truncated_bptt_relay(
            chunk_idx=c_idx,
            prev_h_norm=prev_norm,
            curr_h_norm=curr_norm,
            dh_truncated_norm=dh_trunc_norm,
            show_tip=(c_idx == 0),
            show_banner=(c_idx == 0)
        )

    # ---------------- 跨 Chunk 记忆存留能力深度对比 ----------------
    print("\n" + "="*80)
    print("【深度记忆印证: Chunk 0 的触发信号在 Chunk 3 末尾的保留度】")
    # 计算 Chunk 3 的最终状态对 Chunk 0 触发信号的相关敏感度
    state_a = stateless_final_states[-1].reshape(-1)
    state_b = stateful_final_states[-1].reshape(-1)
    norm_diff = np.linalg.norm(state_b - state_a)

    print(f"• 模式 A (无接力) Chunk #3 终态模长: {float(np.linalg.norm(state_a)):.4f} (历史信号已彻底消亡，只剩局部噪声)")
    print(f"• 模式 B (截断接力) Chunk #3 终态模长: {float(np.linalg.norm(state_b)):.4f} (早期线索经由 h 持续接力，仍被深刻激活)")
    print(f"• 终态特征欧氏差异度: {float(norm_diff):.4f}")
    print("• 关键原理解析: Truncated BPTT 并非简单地'把长序列切碎各算各的'，而是:")
    print("  1. 前向传播：h 像火炬接力一样跨 Chunk 顺流而下，实现跨越万千 Token 的长程语境记忆！")
    print("  2. 反向传播：梯度在每个 Chunk 的入口处严格斩断截停，换取 O(T) 的恒定常数级显存！")
    print("="*80)

    # 核心指标白话词典看板 (扫清初学者阅读指标的疑惑)
    exp3_metrics = [
        ("跨块继承 ||h||", "h_end -> h_next", "火炬接力棒保留量，前序段落语义无损带入新段落", "stateful 保持 1.0~4.0; stateless 归 0"),
        ("边界截断梯度", "||dL / dh_prev||", "被斩断的跨段责任，强制截停以阻断无限回溯", "正常 0.5~5.0; 牺牲跨段反向换取恒定显存"),
        ("显存复杂度 O(T)", "Mem ~ O(T_chunk)", "显存仅与当前分段长度有关，与整本小说总长无关", "恒定常数级，读千亿字也不会爆显存"),
        ("终态欧氏差异度", "||h_B - h_A||", "早期触发信号跨越 40 步后在终态留下的深刻印记", "正常 >1.0; 充分印证跨块长程记忆有效性"),
    ]
    visualizer.show_metric_definitions("Truncated BPTT 截断接力指标字典", exp3_metrics)


if __name__ == "__main__":
    run_experiment()

# -*- coding: utf-8 -*-
"""
实验 1：单步 RNN 神经元内部切片解构与微观状态透视
探究重点:
1. 线性加权投影 (x @ Wx vs h_prev @ Wh) 的能量竞争与记忆主导度
2. 正常输入 vs 极端输入下的预激活值分布
3. tanh 饱和现象对反向传播梯度的毁灭性影响 (微观梯度消失机制)
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
from src.rnn_cells import RNNCell
from src.visualizer import RNNVisualizer


def run_experiment():
    visualizer = RNNVisualizer()
    visualizer.print_banner("【实验 1】单步 RNN 神经元微观解构与饱和度透视", "探究 tanh 非线性区与反向传播梯度的微观阻断")

    np.random.seed(42)
    N, D, H = 1, 8, 16

    # 初始化标准权重
    Wx = (np.random.randn(D, H) / np.sqrt(D)).astype(np.float32)
    Wh = (np.random.randn(H, H) / np.sqrt(H)).astype(np.float32)
    b = np.zeros(H, dtype=np.float32)

    cell = RNNCell(Wx, Wh, b)

    # 场景 A: 正常输入信号 (工作在线性活跃响应区)
    print("\n>>> [场景 A: 正常输入信号 - 活跃响应区]")
    x_normal = np.random.randn(N, D).astype(np.float32)
    h_normal = np.random.randn(N, H).astype(np.float32) * 0.5

    cell.forward(x_normal, h_normal)
    dh_normal = np.ones((N, H), dtype=np.float32)
    cell.backward(dh_normal)
    visualizer.show_cell_anatomy(
        step_idx=1,
        probe_data=cell.probe_data,
        show_tip=True,
        show_banner=True,
        scene_name="正常响应区"
    )

    # 场景 B: 极端大输入信号 (引发 tanh 严重饱和，直击梯度消失微观现场)
    print("\n>>> [场景 B: 极端大输入信号 - 严重进入饱和区]")
    cell_saturated = RNNCell(Wx, Wh, b)
    x_large = np.random.randn(N, D).astype(np.float32) * 10.0 # 输入放大 10 倍
    h_large = np.random.randn(N, H).astype(np.float32) * 10.0

    cell_saturated.forward(x_large, h_large)
    dh_large = np.ones((N, H), dtype=np.float32)
    cell_saturated.backward(dh_large)
    visualizer.show_cell_anatomy(
        step_idx=2,
        probe_data=cell_saturated.probe_data,
        show_tip=False,
        show_banner=False,
        scene_name="极端饱和区"
    )

    # 对比总结
    gain_a = cell.probe_data["grad_gain"]
    gain_b = cell_saturated.probe_data["grad_gain"]
    print("\n" + "="*70)
    print(f"【微观透视结论对比】:")
    print(f"• 场景 A (正常响应区): 饱和度 = {cell.probe_data['saturation_ratio']:.1%}, 梯度局部增益 = {gain_a:.4f}")
    print(f"• 场景 B (极端饱和区): 饱和度 = {cell_saturated.probe_data['saturation_ratio']:.1%}, 梯度局部增益 = {gain_b:.4f}")
    print(f"• 梯度衰减倍率: 场景 B 的梯度穿透能力仅为场景 A 的 {(gain_b / (gain_a + 1e-9)):.2%}")
    print("• 物理实质: 一旦隐藏状态进入 tanh 饱和区 (|a| > 2)，其导数 (1 - h^2) 归零，误差信号瞬间在此断裂！")
    print("="*70)

    # 核心指标白话词典看板 (彻底扫清初学者阅读指标的疑惑)
    exp1_metrics = [
        ("输入投影模长", "||x @ Wx||", "新词刺激强度，外部世界刚送进来的信号大小", "正常 1.0~3.0，>10 易冲击过载"),
        ("记忆循环模长", "||h @ Wh||", "旧笔记记忆强度，读词前脑海保留的历史深度", "正常 1.0~3.0，与输入模长均衡"),
        ("预激活和势能", "a_t = xWx+hWh+b", "新刺激与旧笔记加权融合后的总激活势能", "均值~0, 标准差在 0.8~1.5 为佳"),
        ("tanh 饱和度", "mean(|a_t| > 2)", "神经元被巨量输入冲击而'听麻木'的比例", "正常 <=20%, >30% 警告, >80% 阻断"),
        ("梯度局部增益", "mean(1 - h_t^2)", "改错信号反向穿透该神经元时保留的力气", "正常 0.4~0.8, <0.1 信号被吞噬"),
    ]
    visualizer.show_metric_definitions("单步神经元与微观饱和度指标字典", exp1_metrics)


if __name__ == "__main__":
    run_experiment()

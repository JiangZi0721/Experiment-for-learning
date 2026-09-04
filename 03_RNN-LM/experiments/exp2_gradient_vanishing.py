# -*- coding: utf-8 -*-
"""
实验 2：BPTT 梯度消失与梯度爆炸极端物理实验 (谱半径观察与梯度裁剪护航)
探究重点:
1. 循环权重 Wh 的谱半径 (最大绝对特征值) 对时序反向传播的指数级放大与缩小效应
2. 设定 A: 小权重方差 -> 梯度以指数速度衰减归零 (Gradient Vanishing)
3. 设定 B: 大权重方差 -> 梯度以指数速度狂飙失控 (Gradient Exploding)
4. 设定 C: 正交初始化 + 梯度裁剪 (Gradient Clipping) 实现健康稳定传导
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
from src.layers import clip_grads
from src.visualizer import RNNVisualizer


def run_experiment():
    visualizer = RNNVisualizer()
    visualizer.print_banner("【实验 2】BPTT 梯度时序逆流：消失与爆炸实验室", "探究循环权重矩阵特征值谱半径与长时序反向传播的命运")

    np.random.seed(42)
    N, T, D, H = 1, 15, 10, 20
    xs = np.random.randn(N, T, D).astype(np.float32)
    dhs = np.zeros((N, T, H), dtype=np.float32)
    # 在最后一个时刻注入误差信号
    dhs[:, -1, :] = 1.0

    # ---------------- 场景 1: 梯度消失实验 (Vanishing) ----------------
    print("\n" + "="*80)
    print(">>> [场景 1: 小方差初始化 -> 谱半径 rho < 1.0 -> 梯度消失]")
    # 初始化 Wh 为较小值 (尺度 0.5)
    Wh_small = (np.random.randn(H, H) * 0.15).astype(np.float32)
    Wx_small = (np.random.randn(D, H) * 0.1).astype(np.float32)
    b = np.zeros(H, dtype=np.float32)

    eigenvals = np.linalg.eigvals(Wh_small)
    spectral_radius = float(np.max(np.abs(eigenvals)))
    print(f"[*] 循环权重 Wh 特征值谱半径 (Spectral Radius): {spectral_radius:.4f} (< 1.0)")

    rnn_small = TimeRNN(Wx_small, Wh_small, b)
    rnn_small.forward(xs)
    rnn_small.backward(dhs)
    visualizer.show_bptt_backward_flow(
        rnn_small.probe_data["backward"],
        show_tip=True,
        show_banner=True,
        scene_name="梯度消失"
    )

    # ---------------- 场景 2: 梯度爆炸实验 (Exploding) ----------------
    print("\n" + "="*80)
    print(">>> [场景 2: 大方差初始化 -> 谱半径 rho > 1.0 -> 梯度爆炸]")
    # 初始化 Wh 为较大值 (尺度 2.0)
    Wh_large = (np.random.randn(H, H) * 0.45).astype(np.float32)
    Wx_large = (np.random.randn(D, H) * 0.1).astype(np.float32)

    eigenvals_large = np.linalg.eigvals(Wh_large)
    spectral_radius_large = float(np.max(np.abs(eigenvals_large)))
    print(f"[*] 循环权重 Wh 特征值谱半径 (Spectral Radius): {spectral_radius_large:.4f} (> 1.0)")

    rnn_large = TimeRNN(Wx_large, Wh_large, b)
    rnn_large.forward(xs)
    rnn_large.backward(dhs)
    visualizer.show_bptt_backward_flow(
        rnn_large.probe_data["backward"],
        show_tip=False,
        show_banner=False,
        scene_name="梯度爆炸"
    )

    # ---------------- 场景 3: 梯度裁剪保驾护航 (Clipping) ----------------
    print("\n" + "="*80)
    print(">>> [场景 3: 梯度裁剪 (Gradient Clipping) 对爆炸梯度的物理扼制]")
    max_norm = 5.0
    orig_norm, final_norm, rate = clip_grads(rnn_large.grads, max_norm)
    visualizer.show_gradient_clipping(orig_norm, final_norm, rate, max_norm, show_tip=True, show_banner=True)

    # 核心指标白话词典看板 (扫清初学者阅读指标的疑惑)
    exp2_metrics = [
        ("循环谱半径 rho", "max |lambda(Wh)|", "记忆矩阵缩放基因，连乘 T 步的缩放命运", "健康 ~1.0; <0.9 必消失, >1.2 必爆炸"),
        ("改错信号 ||dh_t||", "||dL / dh_t||", "逆流到时刻 t 的隐藏状态改错责任强度", "正常与末端相当; 1e-6 消失, >100 爆炸"),
        ("相比末端倍数", "||dh_t|| / ||dh_T||", "改错声音的回声保留率，相比末端放大或缩小几倍", "正常 0.2x~5.0x; <0.05x 消失, >20x 爆炸"),
        ("全局梯度模长 ||g||", "sqrt(sum ||grad||^2)", "全网所有参数改错步长的联合欧氏总模长", "正常 1.0~10.0; >5.0 迈步过猛需限速"),
        ("裁剪缩放比率 eta", "min(1.0, max/||g||)", "防止模型暴走崩溃的刹车减速系数", "正常=1.00; <1.00 表明触发防爆刹车"),
    ]
    visualizer.show_metric_definitions("BPTT 梯度逆流与裁剪指标字典", exp2_metrics)


if __name__ == "__main__":
    run_experiment()

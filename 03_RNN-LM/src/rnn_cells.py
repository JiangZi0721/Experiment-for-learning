# -*- coding: utf-8 -*-
"""
RNN 神经元核心底层数学实现 (One-Step RNN Cell)
完全白盒纯 NumPy 实现，手推前向传播与反向传播 (BPTT 单步计算图)，搭载微观透视探针。
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np


class RNNCell:
    """
    单步循环神经网络单元 (Vanilla RNN Cell)

    前向数学定义:
        a_t = x_t @ W_x + h_{prev} @ W_h + b
        h_t = tanh(a_t)

    反向梯度推导:
        da_t = dh_t * (1 - h_t^2)       (tanh 导数)
        dx_t = da_t @ W_x.T             (对当前输入的偏导)
        dh_prev = da_t @ W_h.T          (对上一时刻隐藏状态的偏导)
        dW_x = x_t.T @ da_t             (对输入权重的梯度)
        dW_h = h_prev.T @ da_t          (对隐藏状态循环权重的梯度)
        db = sum(da_t, axis=0)          (对偏置的梯度)
    """
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray):
        """
        参数:
            Wx: 输入权重矩阵, shape=(D, H)
            Wh: 循环隐状态权重矩阵, shape=(H, H)
            b:  偏置向量, shape=(H,)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

        # 微观透视探针数据仓库
        self.probe_data: Dict[str, Any] = {}

    @property
    def Wx(self) -> np.ndarray:
        return self.params[0]

    @property
    def Wh(self) -> np.ndarray:
        return self.params[1]

    @property
    def b(self) -> np.ndarray:
        return self.params[2]

    @property
    def dWx(self) -> np.ndarray:
        return self.grads[0]

    @property
    def dWh(self) -> np.ndarray:
        return self.grads[1]

    @property
    def db(self) -> np.ndarray:
        return self.grads[2]

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        单步前向传播
        参数:
            x: 当前时刻输入, shape=(N, D)
            h_prev: 前一时刻隐藏状态, shape=(N, H)
        返回:
            h_next: 当前时刻输出隐藏状态, shape=(N, H)
        """
        Wx, Wh, b = self.params

        # 1. 线性变换与加权融合 (affine combination)
        x_proj = np.dot(x, Wx)          # (N, H): 输入信号投影
        h_proj = np.dot(h_prev, Wh)     # (N, H): 历史记忆传递
        a_t = x_proj + h_proj + b       # (N, H): 预激活值 (Pre-activation)

        # 2. 非线性激活
        h_next = np.tanh(a_t)           # (N, H): 归一化到 (-1, 1) 的隐状态

        # 3. 暂存前向计算图节点，用于精确反向传播
        self.cache = (x, h_prev, h_next)

        # 4. 微观探针数据采集 (用于白盒透视看板)
        # 饱和度定义: |a_t| > 2.0 时，tanh 导数 1-tanh^2 < 0.07，进入严重饱和区
        saturation_mask = np.abs(a_t) > 2.0
        sat_ratio = float(np.mean(saturation_mask))

        # 记忆主导度: 历史信号范数 / (输入信号范数 + 历史信号范数 + eps)
        norm_x = float(np.linalg.norm(x_proj))
        norm_h = float(np.linalg.norm(h_proj))
        memory_ratio = norm_h / (norm_x + norm_h + 1e-9)

        self.probe_data = {
            "x_norm": float(np.linalg.norm(x)),
            "h_prev_norm": float(np.linalg.norm(h_prev)),
            "x_proj_norm": norm_x,
            "h_proj_norm": norm_h,
            "memory_ratio": memory_ratio,
            "a_t_mean": float(np.mean(a_t)),
            "a_t_std": float(np.std(a_t)),
            "h_next_mean": float(np.mean(h_next)),
            "h_next_std": float(np.std(h_next)),
            "saturation_ratio": sat_ratio,
            "h_next_sample": h_next[0].copy() if len(h_next) > 0 else None,
        }

        return h_next

    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        单步反向传播 (BPTT 局部微积分)
        参数:
            dh_next: 来自未来或上层的隐藏状态偏导, shape=(N, H)
        返回:
            dx: 对当前输入的偏导, shape=(N, D)
            dh_prev: 对上一时刻隐藏状态的偏导, shape=(N, H)
        """
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # 1. 穿透 tanh 激活函数: d(tanh(a))/da = 1 - tanh(a)^2 = 1 - h_next^2
        # 注意: 如果处于饱和区 (h_next 接近 1 或 -1)，dtanh 趋近于 0 (梯度消失源头)
        dtanh = 1.0 - h_next ** 2
        da_t = dh_next * dtanh          # shape=(N, H)

        # 2. 穿透偏置加法节点: sum 沿 batch 维度求和
        self.grads[2][...] = np.sum(da_t, axis=0) # db: (H,)

        # 3. 穿透隐状态循环矩阵乘法节点
        self.grads[1][...] = np.dot(h_prev.T, da_t) # dWh: (H, H)
        dh_prev = np.dot(da_t, Wh.T)                # dh_prev: (N, H)

        # 4. 穿透输入投影矩阵乘法节点
        self.grads[0][...] = np.dot(x.T, da_t)      # dWx: (D, H)
        dx = np.dot(da_t, Wx.T)                     # dx: (N, D)

        # 5. 反向传播探针记录
        in_norm = float(np.linalg.norm(dh_next))
        out_norm = float(np.linalg.norm(dh_prev))
        # 局部梯度缩放系数 gamma = ||dh_{prev}|| / ||dh_{next}||
        grad_gain = out_norm / (in_norm + 1e-9)

        self.probe_data.update({
            "dh_next_norm": in_norm,
            "dh_prev_norm": out_norm,
            "da_t_norm": float(np.linalg.norm(da_t)),
            "dWx_norm": float(np.linalg.norm(self.grads[0])),
            "dWh_norm": float(np.linalg.norm(self.grads[1])),
            "db_norm": float(np.linalg.norm(self.grads[2])),
            "grad_gain": grad_gain,
        })

        return dx, dh_prev

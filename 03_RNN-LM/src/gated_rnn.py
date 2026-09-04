# -*- coding: utf-8 -*-
"""
Gated RNN 门控循环神经网络核心底层实现 (Gated Recurrent Unit - GRU)
纯 NumPy 手推前向传播与 BPTT 反向传播计算图，搭载门控开闭全景透视探针。
彻底揭秘门控机制（更新门与重置门）如何开辟梯度直连高速公路，根治 Vanilla RNN 的梯度消失问题。
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid 激活函数"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class GRUCell:
    """
    单步门控循环单元 (One-Step GRU Cell)

    前向数学推导:
        r_t = sigmoid(x_t @ W_xr + h_{prev} @ W_hr + b_r)       (重置门: 遗忘旧笔记的比率)
        z_t = sigmoid(x_t @ W_xz + h_{prev} @ W_hz + b_z)       (更新门: 记忆更替权衡开关)
        h_reset = r_t * h_{prev}                                (受控历史信息)
        h_tilde = tanh(x_t @ W_xh + h_reset @ W_hh + b_h)       (候选新记忆)
        h_t = (1 - z_t) * h_{prev} + z_t * h_tilde              (最终隐藏状态融合)

    反向梯度高速公路:
        dh_{prev} 中包含一条极其关键的直接加法支路: dh_t * (1 - z_t)
        当更新门 z_t 接近 0 时，dh_{prev} ≈ dh_t，梯度无损穿透！彻底破除矩阵连乘导致的梯度消失！
    """
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray):
        """
        参数采用拼接紧凑形式 (3 倍隐藏维度 H):
            Wx: (D, 3*H), 分割为 [W_xr, W_xz, W_xh]
            Wh: (H, 3*H), 分割为 [W_hr, W_hz, W_hh]
            b:  (3*H,),   分割为 [b_r, b_z, b_h]
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        self.probe_data: Dict[str, Any] = {}

    @property
    def Wx(self) -> np.ndarray: return self.params[0]
    @property
    def Wh(self) -> np.ndarray: return self.params[1]
    @property
    def b(self) -> np.ndarray: return self.params[2]

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        Wx, Wh, b = self.params
        H = Wh.shape[0]

        # 1. 计算门控线性投影 (r: 重置门, z: 更新门)
        # 前 2*H 维度对应 r 和 z
        gates = np.dot(x, Wx[:, :2*H]) + np.dot(h_prev, Wh[:, :2*H]) + b[:2*H]
        r = sigmoid(gates[:, :H])
        z = sigmoid(gates[:, H:2*H])

        # 2. 计算候选新记忆 h_tilde
        h_reset = r * h_prev
        a_h = np.dot(x, Wx[:, 2*H:]) + np.dot(h_reset, Wh[:, 2*H:]) + b[2*H:]
        h_tilde = np.tanh(a_h)

        # 3. 线性插值状态融合
        h_next = (1.0 - z) * h_prev + z * h_tilde

        # 4. 暂存计算图节点用于精确反向求导
        self.cache = (x, h_prev, r, z, h_reset, a_h, h_tilde, h_next)

        # 5. 门控微观探针数据
        self.probe_data = {
            "reset_gate_mean": float(np.mean(r)),       # 重置门平均开度 (0=彻底擦除旧笔记, 1=全盘保留)
            "update_gate_mean": float(np.mean(z)),      # 更新门平均开度 (0=只保旧记忆, 1=全换新记忆)
            "h_tilde_norm": float(np.linalg.norm(h_tilde)),
            "h_next_norm": float(np.linalg.norm(h_next)),
            "highway_gradient_flow": float(np.mean(1.0 - z)), # 梯度高速公路畅通度
        }

        return h_next

    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        x, h_prev, r, z, h_reset, a_h, h_tilde, h_next = self.cache

        # 1. 穿透状态融合节点 h_next = (1 - z)*h_prev + z*h_tilde
        dh_prev_direct = dh_next * (1.0 - z)            # 【核心: 梯度高速公路直连支路！】
        dh_tilde = dh_next * z
        dz = dh_next * (h_tilde - h_prev)

        # 2. 穿透 tanh 候选隐状态节点 h_tilde = tanh(a_h)
        da_h = dh_tilde * (1.0 - h_tilde ** 2)

        # 候选部分的参数梯度
        dWx_h = np.dot(x.T, da_h)
        dWh_h = np.dot(h_reset.T, da_h)
        db_h = np.sum(da_h, axis=0)

        dx_h = np.dot(da_h, Wx[:, 2*H:].T)
        dh_reset = np.dot(da_h, Wh[:, 2*H:].T)

        # 3. 穿透重置节点 h_reset = r * h_prev
        dr = dh_reset * h_prev
        dh_prev_from_reset = dh_reset * r

        # 4. 穿透更新门 z = sigmoid(a_z)
        da_z = dz * z * (1.0 - z)
        # 穿透重置门 r = sigmoid(a_r)
        da_r = dr * r * (1.0 - r)

        # 合并门控梯度 da_gates = [da_r, da_z]
        da_gates = np.hstack([da_r, da_z])

        # 门控部分的参数梯度
        dWx_rz = np.dot(x.T, da_gates)
        dWh_rz = np.dot(h_prev.T, da_gates)
        db_rz = np.sum(da_gates, axis=0)

        dx_rz = np.dot(da_gates, Wx[:, :2*H].T)
        dh_prev_from_rz = np.dot(da_gates, Wh[:, :2*H].T)

        # 5. 整合所有分支参数梯度并写入
        self.grads[0][:, :2*H] = dWx_rz
        self.grads[0][:, 2*H:] = dWx_h
        self.grads[1][:, :2*H] = dWh_rz
        self.grads[1][:, 2*H:] = dWh_h
        self.grads[2][:2*H] = db_rz
        self.grads[2][2*H:] = db_h

        # 6. 汇流总输入梯度 dx 与历史状态梯度 dh_prev
        dx = dx_h + dx_rz
        dh_prev = dh_prev_direct + dh_prev_from_reset + dh_prev_from_rz

        in_norm = float(np.linalg.norm(dh_next))
        out_norm = float(np.linalg.norm(dh_prev))
        self.probe_data.update({
            "dh_next_norm": in_norm,
            "dh_prev_norm": out_norm,
            "grad_retention_rate": out_norm / (in_norm + 1e-9),
        })

        return dx, dh_prev


class TimeGRU:
    """
    时序展开的 GRU 门控循环层 (TimeGRU)
    支持 T 步时序推演、跨块隐藏状态接力 (stateful=True) 与完整的 BPTT 时序反向传播
    """
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful: bool = False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers: List[GRUCell] = []

        self.h: Optional[np.ndarray] = None
        self.dh: Optional[np.ndarray] = None
        self.stateful = stateful
        self.probe_data: Dict[str, Any] = {}

    def set_state(self, h: np.ndarray):
        self.h = h.copy() if h is not None else None

    def reset_state(self):
        self.h = None

    def forward(self, xs: np.ndarray) -> np.ndarray:
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype=xs.dtype)

        self.layers = []
        hs = np.empty((N, T, H), dtype=xs.dtype)

        step_r_gates = []
        step_z_gates = []
        step_h_norms = []

        for t in range(T):
            layer = GRUCell(Wx, Wh, b)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

            step_r_gates.append(layer.probe_data.get("reset_gate_mean", 0.5))
            step_z_gates.append(layer.probe_data.get("update_gate_mean", 0.5))
            step_h_norms.append(layer.probe_data.get("h_next_norm", 0.0))

        self.probe_data["forward"] = {
            "step_r_gates": step_r_gates,
            "step_z_gates": step_z_gates,
            "step_h_norms": step_h_norms,
            "final_h": self.h.copy()
        }
        return hs

    def backward(self, dhs: np.ndarray, dh_future: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype=dhs.dtype)
        grad_Wx = np.zeros_like(Wx)
        grad_Wh = np.zeros_like(Wh)
        grad_b = np.zeros_like(b)

        dh = dh_future if dh_future is not None else np.zeros((N, H), dtype=dhs.dtype)
        step_dh_norms = []

        for t in reversed(range(T)):
            layer = self.layers[t]
            dh_total = dhs[:, t, :] + dh
            dx, dh = layer.backward(dh_total)
            dxs[:, t, :] = dx

            grad_Wx += layer.grads[0]
            grad_Wh += layer.grads[1]
            grad_b += layer.grads[2]

            step_dh_norms.append(float(np.linalg.norm(dh)))

        self.grads[0][...] = grad_Wx
        self.grads[1][...] = grad_Wh
        self.grads[2][...] = grad_b
        self.dh = dh

        step_dh_norms.reverse()
        self.probe_data["backward"] = {
            "step_dh_norms": step_dh_norms,
            "total_dWx_norm": float(np.linalg.norm(grad_Wx)),
            "total_dWh_norm": float(np.linalg.norm(grad_Wh)),
            "dh_to_prev_chunk_norm": float(np.linalg.norm(dh))
        }

        return dxs, dh

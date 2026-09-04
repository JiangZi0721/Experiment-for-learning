# -*- coding: utf-8 -*-
"""
TimeRNN: 时间轴展开循环层 (Unrolled RNN Layer over Time)
将 T 个单步 RNN 神经元在时序维度串联展开，实现正向时间序列推演与 BPTT (Backpropagation Through Time)。
搭载时序流动探针，支持对隐藏状态演化和时间反向梯度衰减的全局透视。
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.rnn_cells import RNNCell


class TimeRNN:
    """
    时序展开的 RNN 层 (TimeRNN)

    张量形状规范:
        输入 xs:     (N, T, D) - N=BatchSize, T=TimeSteps, D=WordVecSize
        输出 hs:     (N, T, H) - H=HiddenSize
        参数 Wx:     (D, H)
        参数 Wh:     (H, H)
        参数 b:      (H,)
        上一状态 h:  (N, H)
    """
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful: bool = False):
        """
        参数:
            Wx, Wh, b: 权重与偏置参数矩阵
            stateful: 是否在连续前向调用间保持隐藏状态 (Truncated BPTT 的前向记忆继承开关)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers: List[RNNCell] = []

        self.h: Optional[np.ndarray] = None          # 保存最后一个时间步的隐状态 (N, H)
        self.dh: Optional[np.ndarray] = None         # 保存向前一个 Chunk 回传的隐状态梯度 (N, H)
        self.stateful = stateful

        # 时序全景探针
        self.probe_data: Dict[str, Any] = {}

    def set_state(self, h: np.ndarray):
        """显式设置初始隐藏状态 (用于跨 Chunk 接力)"""
        self.h = h.copy() if h is not None else None

    def reset_state(self):
        """清空隐藏状态"""
        self.h = None

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        时序前向传播: 展开 T 个时间步推进隐藏状态

        参数:
            xs: 形状为 (N, T, D) 的输入序列张量
        返回:
            hs: 形状为 (N, T, H) 的所有时间步隐状态张量
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D_check, H = Wx.shape
        assert D == D_check, f"输入特征维度 D={D} 与权重 Wx={D_check} 不匹配"

        # 如果不是 stateful 或初次运行，初始化隐藏状态为全 0
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype=xs.dtype)

        self.layers = []
        hs = np.empty((N, T, H), dtype=xs.dtype)

        # 探针: 记录每步隐藏状态特征
        step_h_norms = []
        step_sat_ratios = []
        step_mem_ratios = []

        # 沿着时间轴从 t=0 到 T-1 顺序推进
        for t in range(T):
            layer = RNNCell(Wx, Wh, b)
            # h 是上一时刻状态，前向推进得到新的 h
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

            # 提取探针指标
            step_h_norms.append(float(np.linalg.norm(self.h)))
            step_sat_ratios.append(layer.probe_data.get("saturation_ratio", 0.0))
            step_mem_ratios.append(layer.probe_data.get("memory_ratio", 0.0))

        # 记录全局前向透视数据
        self.probe_data["forward"] = {
            "batch_size": N,
            "time_steps": T,
            "h_start_norm": float(np.linalg.norm(self.layers[0].cache[1])),
            "h_end_norm": float(np.linalg.norm(self.h)),
            "step_h_norms": step_h_norms,
            "step_sat_ratios": step_sat_ratios,
            "step_mem_ratios": step_mem_ratios,
            "final_h": self.h.copy()
        }

        return hs

    def backward(self, dhs: np.ndarray, dh_future: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        BPTT (Backpropagation Through Time) 时序反向传播
        从 t=T-1 沿时间轴倒序向前逆流，精确累加各个时间步对共享权重 (Wx, Wh, b) 的梯度。

        参数:
            dhs: 来自上层 (如 TimeAffine) 的梯度张量, 形状 (N, T, H)
            dh_future: 来自未来 Chunk 的反向梯度 (N, H)。若为 None 则从 0 开始
        返回:
            dxs: 对输入序列的梯度张量, 形状 (N, T, D)
            dh_prev_chunk: 穿透到上一个 Chunk 的梯度 (N, H) (在 Truncated BPTT 中将被截断)
        """
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H_check = Wx.shape

        dxs = np.empty((N, T, D), dtype=dhs.dtype)

        # 重置权重梯度缓冲区
        grad_Wx = np.zeros_like(Wx)
        grad_Wh = np.zeros_like(Wh)
        grad_b = np.zeros_like(b)

        # dh 保存从 t+1 时刻流向 t 时刻的隐藏状态梯度
        dh = dh_future if dh_future is not None else np.zeros((N, H), dtype=dhs.dtype)

        # 探针: 逆序记录各步梯度流动情况
        step_dh_norms = []
        step_grad_gains = []

        # 核心：逆时间顺序遍历 t = T-1, T-2, ..., 0
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 时刻 t 的隐藏状态接收两条梯度的汇合:
            # 1. 来自当前时刻上层 (TimeAffine) 的 dhs[:, t, :]
            # 2. 来自下一时刻 (t+1) 逆流回来的 dh
            dh_total = dhs[:, t, :] + dh

            dx, dh = layer.backward(dh_total)
            dxs[:, t, :] = dx

            # 累加参数梯度 (时间共享参数的所有时刻梯度之和)
            grad_Wx += layer.grads[0]
            grad_Wh += layer.grads[1]
            grad_b += layer.grads[2]

            # 采集逆序反向流动探针
            step_dh_norms.append(float(np.linalg.norm(dh)))
            step_grad_gains.append(layer.probe_data.get("grad_gain", 1.0))

        # 统一写入参数梯度
        self.grads[0][...] = grad_Wx
        self.grads[1][...] = grad_Wh
        self.grads[2][...] = grad_b

        # 最后保留的 dh 就是流向上一个时间块 (Previous Chunk) 的梯度
        self.dh = dh

        # 逆序转为正序时序列表 [t=0, ..., t=T-1] 方便观察
        step_dh_norms.reverse()
        step_grad_gains.reverse()

        self.probe_data["backward"] = {
            "step_dh_norms": step_dh_norms,
            "step_grad_gains": step_grad_gains,
            "total_dWx_norm": float(np.linalg.norm(grad_Wx)),
            "total_dWh_norm": float(np.linalg.norm(grad_Wh)),
            "total_db_norm": float(np.linalg.norm(grad_b)),
            "dh_to_prev_chunk_norm": float(np.linalg.norm(dh)),
        }

        return dxs, dh

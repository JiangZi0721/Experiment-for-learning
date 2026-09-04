# -*- coding: utf-8 -*-
"""
RNN 语言模型基础配套时间层组件
纯 NumPy 白盒实现: TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss, clip_grads
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class TimeEmbedding:
    """
    时序词嵌入层 (Time Embedding)
    将每个时间步的 Token ID 映射为稠密连续向量
    输入 xs: (N, T)
    输出 out: (N, T, D)
    参数 W: (V, D)
    """
    def __init__(self, W: np.ndarray):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.xs = None

    def forward(self, xs: np.ndarray) -> np.ndarray:
        W, = self.params
        N, T = xs.shape
        D = W.shape[1]

        out = np.empty((N, T, D), dtype=W.dtype)
        out = W[xs]  # 高级花式索引提取词向量 (N, T, D)
        self.xs = xs
        return out

    def backward(self, dout: np.ndarray) -> None:
        """
        反向传播: 使用 np.add.at 累加出现过的 Token 的梯度
        """
        W, = self.params
        dW = self.grads[0]
        dW[...] = 0.0

        # 必须使用 add.at 处理同一个词在同一个 batch/seq 中多次出现时的梯度累加
        D = W.shape[1]
        np.add.at(dW, self.xs.reshape(-1), dout.reshape(-1, D))
        return None


class TimeAffine:
    """
    时序全连接投影层 (Time Affine)
    将每个时间步的隐藏状态 H 投影至词表空间 V (计算每个候选词的 Logits)
    输入 xs: (N, T, H)
    输出 out: (N, T, V)
    参数 W: (H, V), b: (V,)
    """
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, T, H = x.shape
        W, b = self.params

        # 展平成 2D 进行高效矩阵运算 (N*T, H) @ (H, V)
        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.x
        N, T, H = x.shape
        W, b = self.params

        dout_2d = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        # 梯度推导
        db = np.sum(dout_2d, axis=0)
        dW = np.dot(rx.T, dout_2d)
        dx = np.dot(dout_2d, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class TimeSoftmaxWithLoss:
    """
    时序 Softmax 交叉熵损失层 (Time Softmax With Loss)
    计算序列在所有时间步上的概率分布与交叉熵损失，并给出精确梯度。
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.probe_data: Dict[str, Any] = {}

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        参数:
            xs: 模型输出的 Logits, 形状为 (N, T, V)
            ts: 目标 Token ID 序列, 形状为 (N, T)
        返回:
            loss: 标量平均交叉熵损失
        """
        N, T, V = xs.shape

        # 展平为二维矩阵计算数值稳定的 Softmax
        xs_2d = xs.reshape(N * T, V)
        ts_flat = ts.reshape(-1)

        # 数值稳定性技巧: 减去行最大值，防止 exp 发生浮点溢出 (Overflow)
        c = np.max(xs_2d, axis=-1, keepdims=True)
        exp_xs = np.exp(xs_2d - c)
        ys_2d = exp_xs / np.sum(exp_xs, axis=-1, keepdims=True)

        # 计算负对数似然损失 (Negative Log-Likelihood)
        total_elements = N * T
        correct_log_probs = np.log(ys_2d[np.arange(total_elements), ts_flat] + 1e-15)
        loss = -np.sum(correct_log_probs) / total_elements

        # 准确率统计
        predictions = np.argmax(ys_2d, axis=-1)
        acc = float(np.mean(predictions == ts_flat))
        ppl = float(np.exp(loss)) if loss < 50 else float("inf")

        self.cache = (ts_flat, ys_2d, total_elements, (N, T, V))

        # 探针数据收集
        self.probe_data = {
            "loss": float(loss),
            "perplexity": ppl,
            "accuracy": acc,
            "batch_size": N,
            "time_steps": T,
            "sample_logits": xs[0, 0].copy(),
            "sample_probs": ys_2d[0].copy(),
            "sample_target": int(ts[0, 0]),
            "sample_pred": int(predictions[0])
        }

        return float(loss)

    def backward(self, dout: float = 1.0) -> np.ndarray:
        """
        Softmax 交叉熵精准梯度: dx = (p - y) / (N * T)
        """
        ts_flat, ys_2d, total_elements, original_shape = self.cache

        dx = ys_2d.copy()
        # 目标类别的导数减去 1
        dx[np.arange(total_elements), ts_flat] -= 1.0
        # 乘以传入梯度并除以总元素数量
        dx *= dout / total_elements

        return dx.reshape(*original_shape)


def clip_grads(grads: List[np.ndarray], max_norm: float) -> Tuple[float, float, float]:
    """
    梯度裁剪算法 (Gradient Clipping)
    防止 RNN 时序反向传播时由于长序列引发的“梯度爆炸 (Gradient Exploding)”！

    数学原理:
        ||g|| = sqrt( sum_i ||g_i||^2 )
        if ||g|| > max_norm:
            g_i = g_i * (max_norm / ||g||)

    返回:
        (orig_norm, final_norm, scale_ratio)
    """
    total_norm = 0.0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    orig_norm = float(np.sqrt(total_norm))

    scale_ratio = 1.0
    if orig_norm > max_norm:
        scale_ratio = max_norm / (orig_norm + 1e-8)
        for grad in grads:
            grad *= scale_ratio
        final_norm = max_norm
    else:
        final_norm = orig_norm

    return orig_norm, final_norm, scale_ratio

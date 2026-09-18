"""
=============================================================================
模块名称: optimizer.py
核心功能: 神经网络参数优化器与梯度裁剪 (纯 NumPy 实现)
包含优化器:
    1. SGD          : 随机梯度下降法 (Stochastic Gradient Descent)
    2. Adam         : 自适应矩估计优化器 (Adaptive Moment Estimation)
辅助功能:
    3. clip_grads   : 梯度范数裁剪 (防止梯度爆炸)
=============================================================================
理论重点:
1. SGD 的优点是简单直观，但收敛容易陷入鞍点或局部振荡。
2. Adam 结合了动量法 (Momentum, 一阶矩) 与 RMSprop (二阶矩自适应学习率) 的优势:
   - m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (梯度的一阶有偏矩)
   - v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (梯度的二阶有偏矩)
   - \hat{m}_t = m_t / (1 - beta1^t)                    (偏差修正)
   - \hat{v}_t = v_t / (1 - beta2^t)                    (偏差修正)
   - theta_t = theta_{t-1} - lr * \hat{m}_t / (sqrt(\hat{v}_t) + eps)
   在自然语言处理与词嵌入训练中，Adam 的收敛速度与稳定性显著优于普通 SGD。
=============================================================================
"""

import numpy as np
from typing import List


def clip_grads(grads: List[np.ndarray], max_norm: float = 5.0) -> float:
    """
    梯度范数裁剪 (Gradient Clipping)

    数学原理:
        计算所有参数梯度的全局 L2 范数:
            total_norm = sqrt( sum_i ||grad_i||_2^2 )
        若 total_norm > max_norm:
            grad_i = grad_i * (max_norm / total_norm)

    参数:
        grads (List[np.ndarray]): 参数梯度列表
        max_norm (float): 最大允许的全局范数阈值

    返回:
        total_norm (float): 裁剪前的全局梯度范数
    """
    total_norm = 0.0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate

    return float(total_norm)


class SGD:
    """
    随机梯度下降优化器 (Stochastic Gradient Descent)

    更新规则:
        W = W - lr * dW

    参数:
        lr (float): 学习率 (Learning Rate)，默认 0.01
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        """
        根据梯度原地更新参数

        参数:
            params (List[np.ndarray]): 待优化的模型参数矩阵列表
            grads (List[np.ndarray]): 对应的梯度列表
        """
        for param, grad in zip(params, grads):
            param -= self.lr * grad


class Adam:
    """
    自适应矩估计优化器 (Adam)

    参数:
        lr (float): 初始学习率，默认 0.001
        beta1 (float): 一阶动量衰减因子，默认 0.9
        beta2 (float): 二阶方差衰减因子，默认 0.999
        eps (float): 防止除以零的微小常数，默认 1e-8
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = None  # 一阶矩向量缓存
        self.v = None  # 二阶矩向量缓存

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        """
        执行一步 Adam 梯度更新

        参数:
            params (List[np.ndarray]): 模型参数矩阵列表
            grads (List[np.ndarray]): 对应的梯度列表
        """
        # 初次调用时初始化一阶矩 m 与二阶矩 v 为全 0
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.iter += 1

        # 动态计算随迭代次数递减的学习率修正系数
        # lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            # 更新一阶有偏矩估计: m = beta1 * m + (1 - beta1) * grad
            self.m[i] += (1.0 - self.beta1) * (grads[i] - self.m[i])

            # 更新二阶有偏矩估计: v = beta2 * v + (1 - beta2) * (grad ^ 2)
            self.v[i] += (1.0 - self.beta2) * (grads[i] ** 2 - self.v[i])

            # 原地更新参数: param -= lr_t * m / (sqrt(v) + eps)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

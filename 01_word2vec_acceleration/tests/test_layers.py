"""
=============================================================================
测试脚本: test_layers.py
功能: 对 word2vec 高速化实现的各个关键层进行单元测试与数值梯度检验 (Gradient Check)
检验对象:
    1. Embedding 层的正向查找与多重重复索引的反向梯度累加 (np.add.at)
    2. SigmoidWithLoss 层的正向损失与反向数值梯度对比
    3. EmbeddingDot 层的点积求导与数值梯度对比
    4. NegativeSamplingLoss 的前向损失与反向形状检验
=============================================================================
"""

import sys
import os
import unittest
import numpy as np

# 将上级目录添加到系统路径以导入模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from layers import Embedding, EmbeddingDot, SigmoidWithLoss
from negative_sampling import NegativeSamplingLoss, UnigramSampler


def numerical_gradient(f, x, h=1e-4):
    """
    通用中心差分数值梯度计算工具:
    df / dx ≈ [ f(x + h) - f(x - h) ] / (2 * h)
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x + h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x - h)
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        # 差分计算
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原
        it.iternext()
    return grad


class TestLayers(unittest.TestCase):

    def test_embedding_duplicate_indices(self):
        """测试 Embedding 层在遇到 batch 内重复词索引时的梯度原子累加特性"""
        V, H = 5, 3
        W = np.arange(V * H, dtype=np.float64).reshape(V, H)
        embed = Embedding(W)

        # 索引 1 重复出现了两次
        idx = np.array([1, 0, 1])
        out = embed.forward(idx)

        self.assertEqual(out.shape, (3, H))
        np.testing.assert_array_equal(out[0], W[1])
        np.testing.assert_array_equal(out[1], W[0])
        np.testing.assert_array_equal(out[2], W[1])

        # 反向传播梯度 dout: 每个位置给 [1.0, 1.0, 1.0]
        dout = np.ones((3, H), dtype=np.float64)
        embed.backward(dout)

        dW = embed.grads[0]
        # 索引 1 出现两次，梯度必须为 2.0；索引 0 出现一次，梯度为 1.0；其余为 0.0
        np.testing.assert_array_almost_equal(dW[1], np.array([2.0, 2.0, 2.0]))
        np.testing.assert_array_almost_equal(dW[0], np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(dW[2], np.array([0.0, 0.0, 0.0]))
        print(" [PASS] Embedding 重复索引梯度累加测试通过！")

    def test_sigmoid_with_loss_gradient(self):
        """测试 SigmoidWithLoss 的反向传播梯度与中心差分数值梯度的误差"""
        np.random.seed(42)
        N = 10
        x = np.random.randn(N).astype(np.float64)
        t = np.random.randint(0, 2, size=N).astype(np.float64)

        layer = SigmoidWithLoss()
        loss = layer.forward(x, t)
        analytic_grad = layer.backward(dout=1.0)

        # 数值梯度验证
        def loss_fn(x_in):
            test_layer = SigmoidWithLoss()
            return test_layer.forward(x_in, t)

        num_grad = numerical_gradient(loss_fn, x)

        # 相对误差判定
        diff = np.linalg.norm(analytic_grad - num_grad) / (np.linalg.norm(analytic_grad) + np.linalg.norm(num_grad) + 1e-12)
        self.assertLess(diff, 1e-6, f"SigmoidWithLoss 梯度相对误差过大: {diff}")
        print(f" [PASS] SigmoidWithLoss 数值梯度检验通过！相对误差: {diff:.2e}")

    def test_embedding_dot_gradient(self):
        """测试 EmbeddingDot 对隐藏向量 h 的导数精度"""
        np.random.seed(42)
        N, H, V = 4, 6, 8
        h = np.random.randn(N, H).astype(np.float64)
        W = np.random.randn(V, H).astype(np.float64)
        idx = np.array([2, 5, 2, 7])

        embed_dot = EmbeddingDot(W)
        score = embed_dot.forward(h, idx)
        dout = np.random.randn(N).astype(np.float64)

        analytic_dh = embed_dot.backward(dout)

        # 针对 h 计算数值梯度
        def forward_h(h_in):
            target_W = W[idx]
            out = np.sum(h_in * target_W, axis=1)
            return np.sum(out * dout)

        num_dh = numerical_gradient(forward_h, h)
        diff = np.linalg.norm(analytic_dh - num_dh) / (np.linalg.norm(analytic_dh) + np.linalg.norm(num_dh) + 1e-12)
        self.assertLess(diff, 1e-6, f"EmbeddingDot 梯度相对误差过大: {diff}")
        print(f" [PASS] EmbeddingDot 数值梯度检验通过！相对误差: {diff:.2e}")

    def test_negative_sampling_loss_forward_backward(self):
        """测试负采样损失层的前向与反向维度与梯度稳定性"""
        np.random.seed(42)
        N, H, V = 4, 8, 20
        W_out = np.random.randn(V, H).astype(np.float32)
        corpus = np.random.randint(0, V, size=100)
        target = np.array([3, 7, 3, 15], dtype=np.int32)
        h = np.random.randn(N, H).astype(np.float32)

        ns_loss = NegativeSamplingLoss(W_out, corpus=corpus, sample_size=3)
        loss = ns_loss.forward(h, target)
        self.assertTrue(np.isfinite(loss), "损失值出现 NaN 或 Inf！")

        dh = ns_loss.backward(dout=1.0)
        self.assertEqual(dh.shape, (N, H))
        self.assertEqual(ns_loss.grads[0].shape, (V, H))
        self.assertTrue(np.all(np.isfinite(dh)))
        self.assertTrue(np.all(np.isfinite(ns_loss.grads[0])))
        print(" [PASS] NegativeSamplingLoss 前向与反向维度及数值稳定性测试通过！")


if __name__ == "__main__":
    unittest.main()

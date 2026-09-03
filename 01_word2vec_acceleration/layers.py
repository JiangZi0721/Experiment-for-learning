"""
=============================================================================
模块名称: layers.py
核心功能: Word2Vec 高速化所需的基础网络层实现 (纯 NumPy 实现)
包含网络层:
    1. Embedding        : 词嵌入查找层 (替代 O(V) 的 One-hot 矩阵乘法)
    2. EmbeddingDot     : 目标词嵌入与中间向量的点乘层
    3. SigmoidWithLoss  : 带有二分类交叉熵损失的 Sigmoid 层
=============================================================================
设计思想:
在原始的简单 Word2Vec (如简单 CBOW) 中:
    - 输入层通过 One-hot 向量与输入权重 W_in 做矩阵乘法: x @ W_in
      -> 当词表 V = 100,000 时，每次乘法绝大部分是 0 的无用运算。
      -> 高速化方案: Embedding 层直接通过行索引获取对应的词向量 (W_in[idx])，耗时 O(1)。
    - 输出层使用全词表的 Softmax 归一化:
      -> 分母需要对全词表 V 个词计算 exp 并求和，计算代价极高。
      -> 高速化方案: 将多分类转化为二分类，通过 EmbeddingDot + SigmoidWithLoss
         仅针对“正样本”和“少数负样本”进行点积与逻辑回归判定。
=============================================================================
"""

import numpy as np


class Embedding:
    """
    词嵌入查找层 (Embedding Layer)

    功能:
        根据词的 ID (索引) 从权重矩阵 W 中提取对应的词向量。
        等价于 One-hot 向量与权重矩阵的乘法: OneHot(idx) @ W，但无需显式构建 One-hot 向量，
        大大节约内存与计算资源。

    参数:
        W (np.ndarray): 词嵌入权重矩阵，形状为 (V, H)，其中 V 为词表大小，H 为词向量维度。
    """

    def __init__(self, W: np.ndarray):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None  # 缓存前向传播时传入的词索引，反向传播时需要使用

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        前向传播: 根据索引提取权重行

        参数:
            idx (np.ndarray): 词索引数组，形状可以是:
                              - 一维 (N,): 每个样本一个目标词或上下文词
                              - 二维 (N, C): 每个样本有 C 个上下文词
        返回:
            out (np.ndarray): 提取出来的词向量，形状为:
                              - 若 idx 为 (N,)，输出形状为 (N, H)
                              - 若 idx 为 (N, C)，输出形状为 (N, C, H)
        """
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout: np.ndarray) -> None:
        """
        反向传播: 将上游梯度传递给权重梯度 dW

        数学原理与重要细节:
            在前向传播中，输出是根据 idx 提取 W 的某些行。
            因此在反向传播中，上游传来的梯度 dout 必须累加回 dW 的相应行中。

            【关键教学点】:
            如果一个 batch 中出现重复的词 ID (例如 idx = [2, 0, 2])，
            若简单使用 `self.dW[self.idx] = dout`，会导致梯度覆盖 (后一个 2 的梯度覆盖前一个 2)！
            正确的做法是梯度累加:
            在 NumPy 中，必须使用 `np.add.at(dW, idx, dout)` 进行原地原子累加！

        参数:
            dout (np.ndarray): 上游传来的梯度，形状与 forward 的输出相同。
        """
        dW, = self.grads
        dW[...] = 0  # 每次反向传播先将当前层的梯度清零

        # 使用 np.add.at 正确处理重复索引的梯度累加
        np.add.at(dW, self.idx, dout)
        return None


class EmbeddingDot:
    """
    词嵌入点乘层 (EmbeddingDot Layer)

    功能:
        1. 提取目标词 (或负样本词) 的词向量: target_W = W[idx]
        2. 计算中间隐藏层向量 h 与 target_W 之间的点积 (内积):
           score = sum(h * target_W, axis=1)

    应用场景:
        在负采样机制中，计算特定词与上下文综合表征 h 之间的未归一化得分 (Logit)。
        该得分随后直接送入 Sigmoid 层转换为二分类概率:
        P(y=1 | h, target) = sigmoid(score)

    参数:
        W (np.ndarray): 输出层权重矩阵，形状为 (V, H)。
                        注意: 这里的 W 是输出权重矩阵 (即通常记作 W_out)。
    """

    def __init__(self, W: np.ndarray):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None  # 缓存前向变量 (h, target_W) 供反向传播求导使用

    def forward(self, h: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        前向传播: 计算隐藏向量 h 与目标词向量 target_W 的逐样本点积

        参数:
            h (np.ndarray): 隐藏层向量 (如 CBOW 中上下文词向量的平均值)，形状为 (N, H)
            idx (np.ndarray): 目标词 (或采样词) 的索引数组，形状为 (N,)

        返回:
            out (np.ndarray): 点积得分，形状为 (N,)
                              out[i] = h[i] · W[idx[i]] = sum_j (h[i, j] * W[idx[i], j])
        """
        # 从输出权重 W 中检索出对应的词向量
        target_W = self.embed.forward(idx)  # 形状 (N, H)

        # 逐元素相乘并在隐藏层维度上求和，得到标量点积
        out = np.sum(h * target_W, axis=1)  # 形状 (N,)

        # 缓存中间变量
        self.cache = (h, target_W)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播:

        数学推导:
            设标量得分 s_i = sum_j (h_{ij} * w_{ij})
            损失为 L，已知上游梯度 dout = dL / ds (形状为 (N,))
            根据链式法则:
            1) 对隐藏层向量 h 的梯度:
               dL / dh_{ij} = (dL / ds_i) * (ds_i / dh_{ij})
                            = dout[i] * w_{ij}
               写成矩阵形式 (利用广播机制):
               dh = dout[:, np.newaxis] * target_W   (形状为 (N, H))

            2) 对词向量 target_W 的梯度:
               dL / dw_{ij} = (dL / ds_i) * (ds_i / dw_{ij})
                            = dout[i] * h_{ij}
               写成矩阵形式:
               dtarget_W = dout[:, np.newaxis] * h   (形状为 (N, H))

            3) 将 dtarget_W 通过 Embedding 层的反向传播累加到整体权重矩阵 W 的梯度中。

        参数:
            dout (np.ndarray): 上游关于得分的梯度，形状为 (N,)

        返回:
            dh (np.ndarray): 对输入隐藏层向量 h 的梯度，形状为 (N, H)
        """
        h, target_W = self.cache

        # dout 的形状是 (N,)，通过扩展为 (N, 1) 实现与 (N, H) 的按行广播相乘
        dout_reshaped = dout[:, np.newaxis]

        # 1. 计算对 h 的梯度并返回
        dh = dout_reshaped * target_W

        # 2. 计算对 target_W 的梯度
        dtarget_W = dout_reshaped * h

        # 3. 反向传递给内部的 Embedding 层，累加到输出权重矩阵的梯度 dW
        self.embed.backward(dtarget_W)

        return dh


class SigmoidWithLoss:
    """
    带二分类交叉熵损失的 Sigmoid 层 (Sigmoid with Binary Cross-Entropy Loss)

    数学模型:
        将多分类问题简化为二分类判别:
        “给定上下文 h，当前词是否为真实上下文相关的词？”
        - 真实目标词 (正样本): 标签 t = 1
        - 随机采样的负词 (负样本): 标签 t = 0

    Sigmoid 预测概率:
        y = sigma(x) = 1 / (1 + exp(-x))

    二分类交叉熵损失 (Binary Cross Entropy Loss):
        L = - [ t * log(y) + (1 - t) * log(1 - y) ]
        平均损失: Loss = (1 / N) * sum(L_i)

    反向传播梯度推导:
        dL / dx = (dL / dy) * (dy / dx)
        - 当 t = 1 时:
          dL / dy = -1 / y
          dy / dx = y * (1 - y)
          dL / dx = (-1 / y) * y * (1 - y) = y - 1 = y - t
        - 当 t = 0 时:
          dL / dy = 1 / (1 - y)
          dy / dx = y * (1 - y)
          dL / dx = (1 / (1 - y)) * y * (1 - y) = y = y - 0 = y - t

        统一形式:
            dL / dx = (y - t)
        考虑 mini-batch 平均损失:
            dx = (y - t) / N
        该梯度形式极其优雅，数值稳定！

    参数:
        无显式学习参数 (params 为空列表)。
    """

    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None  # Sigmoid 预测概率
        self.t = None  # 监督标签 (0 或 1)

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        前向传播: 计算预测概率与二分类交叉熵损失

        参数:
            x (np.ndarray): 未归一化的输入得分 (Logits)，形状为 (N,)
            t (np.ndarray): 监督标签 (0 或 1)，形状与 x 相同 (N,)

        返回:
            loss (float): 标量均方/交叉熵损失值
        """
        # 数值稳定性处理: 将输入 x 截断在 [-30, 30] 范围内，防止 exp(-x) 溢出 (Overflow)
        x_clipped = np.clip(x, -30.0, 30.0)

        # 计算 Sigmoid 概率
        self.y = 1.0 / (1.0 + np.exp(-x_clipped))
        self.t = t

        # 为了防止 log(0) 产生 -inf，加入极小常数 epsilon
        eps = 1e-7
        # 计算逐样本的二分类交叉熵损失
        loss_elements = -(self.t * np.log(self.y + eps) + (1.0 - self.t) * np.log(1.0 - self.y + eps))

        # 求 batch 维度的平均损失
        loss = np.mean(loss_elements)
        return float(loss)

    def backward(self, dout: float = 1.0) -> np.ndarray:
        """
        反向传播: 计算对输入得分 x 的梯度

        参数:
            dout (float): 上游损失的缩放因子 (通常为 1.0)

        返回:
            dx (np.ndarray): 对输入得分 x 的梯度，形状为 (N,)
        """
        batch_size = self.t.shape[0]

        # dx = (y - t) / N
        dx = (self.y - self.t) * dout / batch_size
        return dx

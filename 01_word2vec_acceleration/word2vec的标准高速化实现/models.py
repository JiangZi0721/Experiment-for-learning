"""
=============================================================================
模块名称: models.py
核心功能: Word2Vec 核心模型架构实现 (包含 CBOW 与 Skip-Gram 高速化版本)
包含模型:
    1. CBOWModel    : 连续词袋模型 (Continuous Bag-of-Words) + 负采样
    2. SkipGramModel: 跳字模型 (Skip-Gram) + 负采样
=============================================================================
架构原理对比:
1. CBOW (Continuous Bag-of-Words):
   - 任务: “利用周围上下文词预测中心目标词” (Contexts -> Target)
   - 流程:
     a. 将 2*window_size 个上下文词通过输入 Embedding 层映射为词向量。
     b. 对这些词向量求平均，合成单一的上下文隐层表征向量 h:
        h = (1 / C) * sum_{c=1}^C Embedding(context_c)
     c. 将 h 输入负采样层 (NegativeSamplingLoss)，计算正目标词和采样负词的二分类损失。
   - 特点: 训练速度比 Skip-Gram 更快，对高频词表达更平滑。

2. Skip-Gram:
   - 任务: “利用中心目标词预测周围的每个上下文词” (Target -> Contexts)
   - 流程:
     a. 将中心词通过输入 Embedding 层映射为隐层向量 h。
     b. 对上下文窗口中的每个词，分别与 h 计算负采样损失并求和。
   - 特点: 训练样本数更多 (每个中心词产生 C 个监督信号)，对低频罕见词的学习效果更好。
=============================================================================
"""

import numpy as np
from typing import Union, List
from layers import Embedding
from negative_sampling import NegativeSamplingLoss


class CBOWModel:
    """
    基于负采样的高速化 CBOW (Continuous Bag of Words) 模型

    参数:
        vocab_size (int): 词汇表大小 V
        hidden_size (int): 词嵌入维度 H (例如 100)
        corpus (np.ndarray): 训练语料词 ID 序列 (用于负采样概率统计)
        window_size (int): 上下文单侧窗口大小 (总上下文词数 C = 2 * window_size)
        sample_size (int): 负采样个数 K (默认 5)
        power (float): 负采样分布平滑指数 (默认 0.75)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        corpus: Union[np.ndarray, List[int]],
        window_size: int = 2,
        sample_size: int = 5,
        power: float = 0.75
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.context_size = 2 * window_size

        # -------------------------------------------------------------
        # 1. 初始化权重矩阵
        # -------------------------------------------------------------
        # W_in: 输入词嵌入矩阵 (V, H)，使用高斯分布缩放初始化
        # W_out: 输出上下文权重矩阵 (V, H)
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_in = (scale * np.random.randn(vocab_size, hidden_size)).astype(np.float32)
        self.W_out = (scale * np.random.randn(vocab_size, hidden_size)).astype(np.float32)

        # -------------------------------------------------------------
        # 2. 构建层结构
        # -------------------------------------------------------------
        # 输入层: 单一向量化 Embedding 即可同时处理所有上下文词
        self.embed = Embedding(self.W_in)

        # 损失层: 负采样损失层 (包含正样本与负样本的二分类 Sigmoid 判定)
        self.loss_layer = NegativeSamplingLoss(
            self.W_out,
            corpus=corpus,
            power=power,
            sample_size=sample_size
        )

        # -------------------------------------------------------------
        # 3. 汇总模型可训练参数与梯度容器
        # -------------------------------------------------------------
        self.layers = [self.embed, self.loss_layer]
        self.params = [self.W_in, self.W_out]
        self.grads = [self.embed.grads[0], self.loss_layer.grads[0]]

    @property
    def word_vecs(self) -> np.ndarray:
        """
        获取训练完成的词向量 (通常取输入权重矩阵 W_in)
        """
        return self.W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        """
        前向传播计算损失

        参数:
            contexts (np.ndarray): 上下文词 ID 矩阵，形状为 (batch_size, context_size)
            target (np.ndarray): 目标词 ID 向量，形状为 (batch_size,)

        返回:
            loss (float): 标量负采样交叉熵损失
        """
        # 1. 批量检索上下文词向量: 形状 (batch_size, context_size, hidden_size)
        h_all = self.embed.forward(contexts)

        # 2. 对所有上下文词向量求平均，作为综合表征向量 h: 形状 (batch_size, hidden_size)
        h = np.mean(h_all, axis=1)

        # 3. 计算负采样损失
        loss = self.loss_layer.forward(h, target)
        return loss

    def backward(self, dout: float = 1.0) -> None:
        """
        反向传播求各权重梯度

        参数:
            dout (float): 上游损失缩放，默认为 1.0
        """
        # 1. 反向传播通过负采样损失层，获取关于隐藏向量 h 的梯度: dh 形状 (batch_size, hidden_size)
        dh = self.loss_layer.backward(dout)

        # 2. 因为前向传播进行了平均: h = (1 / C) * sum(h_all)
        #    根据求导链式法则，每个上下文词的梯度为: dh_c = (1 / C) * dh
        #    利用广播扩展为 (batch_size, context_size, hidden_size)
        dh_all = dh[:, np.newaxis, :] / self.context_size

        # 3. 反向传播通过输入 Embedding 层，累加梯度到 dW_in
        self.embed.backward(dh_all)


class SkipGramModel:
    """
    基于负采样的高速化 Skip-Gram 模型

    参数:
        vocab_size (int): 词汇表大小 V
        hidden_size (int): 词嵌入维度 H (例如 100)
        corpus (np.ndarray): 训练语料词 ID 序列 (用于负采样概率统计)
        window_size (int): 上下文单侧窗口大小 (总上下文词数 C = 2 * window_size)
        sample_size (int): 负采样个数 K (默认 5)
        power (float): 负采样分布平滑指数 (默认 0.75)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        corpus: Union[np.ndarray, List[int]],
        window_size: int = 2,
        sample_size: int = 5,
        power: float = 0.75
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.context_size = 2 * window_size

        # 初始化权重
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_in = (scale * np.random.randn(vocab_size, hidden_size)).astype(np.float32)
        self.W_out = (scale * np.random.randn(vocab_size, hidden_size)).astype(np.float32)

        # 输入层: 中心词 Embedding
        self.embed = Embedding(self.W_in)

        # 损失层: 针对每个上下文窗口位置构建负采样损失层 (共享输出权重 W_out)
        self.loss_layers = [
            NegativeSamplingLoss(
                self.W_out,
                corpus=corpus,
                power=power,
                sample_size=sample_size
            )
            for _ in range(self.context_size)
        ]

        # 汇总可训练参数
        self.params = [self.W_in, self.W_out]
        self.dW_out = np.zeros_like(self.W_out)
        self.grads = [self.embed.grads[0], self.dW_out]

    @property
    def word_vecs(self) -> np.ndarray:
        return self.W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        """
        前向传播计算损失

        参数:
            contexts (np.ndarray): 上下文词矩阵，形状 (batch_size, context_size)
            target (np.ndarray): 中心目标词，形状 (batch_size,)

        返回:
            total_loss (float): 所有上下文位置的负采样交叉熵损失之和
        """
        # 1. 查找中心词向量: 形状 (batch_size, hidden_size)
        h = self.embed.forward(target)

        # 2. 遍历上下文中的每个词，分别计算负采样预测损失
        total_loss = 0.0
        for i, layer in enumerate(self.loss_layers):
            context_i = contexts[:, i]
            loss_i = layer.forward(h, context_i)
            total_loss += loss_i

        return float(total_loss)

    def backward(self, dout: float = 1.0) -> None:
        """
        反向传播汇总各上下文位置的梯度
        """
        dh = 0
        self.dW_out[...] = 0

        # 反向传播各负采样层
        for layer in self.loss_layers:
            dh += layer.backward(dout)
            self.dW_out += layer.grads[0]

        # 反向传播输入词嵌入层
        self.embed.backward(dh)

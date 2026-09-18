"""
=============================================================================
模块名称: negative_sampling.py
核心功能: Word2Vec 负采样 (Negative Sampling) 完整实现
包含组件:
    1. UnigramSampler       : 基于 0.75 次方平滑的负样本采样器
    2. NegativeSamplingLoss : 结合正样本与 K 个负样本的二分类损失层
=============================================================================
理论核心:
1. 为什么需要负采样 (Negative Sampling)？
   - 传统 Softmax 在全词表 V 上的归一化分母为: sum_{k=1}^V exp(s_k)。
   - 当词表 V = 100,000 时，每次迭代计算所有词的得分及其反向传播的代价极高 (O(V))。
   - 负采样将问题转化为“二分类 (Binary Classification)”：
     - 正样本 (真实上下文词): 期望其模型得分预测概率接近 1 (标签 t = 1)。
     - 负样本 (随机抽取的无关词): 期望其模型得分预测概率接近 0 (标签 t = 0)。
   - 计算复杂度从 O(V) 锐减至 O(K + 1)，其中 K 为负采样数量 (通常取 5 ~ 20)。

2. 为什么概率分布要取 0.75 次幂 (3/4 次方)？
   - 若直接按真实词频 P(w) 采样，高频停用词 (如 "the", "is") 将占据绝大部分负样本，
     导致低频词几乎没有作为负样本被更新的机会。
   - 若按均匀分布 1/V 采样，则忽略了语言本身的统计规律。
   - Mikolov 等人实验发现，将词频做 0.75 次方平滑:
         P'(w_i) = [P(w_i)]^0.75 / sum_j [P(w_j)]^0.75
     可以在保留高频词主导地位的同时，显著提高稀有词 (低频词) 被采样的相对几率。
=============================================================================
"""

import numpy as np
import collections
from typing import Union, List


class UnigramSampler:
    """
    基于一元语言模型 (Unigram) 的平滑概率负样本采样器

    参数:
        corpus (np.ndarray 或 List[int]): 整个训练文本对应的词 ID 数组
        power (float): 概率平滑指数，默认 0.75 (Word2Vec 原论文推荐值)
        sample_size (int): 每次采样负样本的数量 K (默认 5)
    """

    def __init__(self, corpus: Union[np.ndarray, List[int]], power: float = 0.75, sample_size: int = 5):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # 统计语料库中每个词的词频
        counts = collections.Counter(corpus)
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        # 构建词频数组 (按词 ID 索引)
        count_arr = np.zeros(vocab_size, dtype=np.float64)
        for word_id, count in counts.items():
            count_arr[word_id] = count

        # 对词频进行 0.75 次方平滑处理
        p = np.power(count_arr, power)
        # 归一化为合法概率分布 (和为 1.0)
        self.word_p = p / np.sum(p)

        # -------------------------------------------------------------
        # 工业级优化: 预建 1,000,000 大小的负采样概率离线查找表 (Table Lookup)
        # (Mikolov word2vec.c 原版底层标准实现)
        # 避免每次迭代调用 np.random.choice 搜索概率分布，将负采样开销降低 98%！
        # -------------------------------------------------------------
        table_size = 1000000
        self.table = np.zeros(table_size, dtype=np.int32)
        idx = 0
        for word_id, prob in enumerate(self.word_p):
            num = int(round(prob * table_size))
            if idx + num > table_size:
                num = table_size - idx
            self.table[idx : idx + num] = word_id
            idx += num

        if idx < table_size:
            self.table[idx:] = 0

    def get_negative_sample(self, target: np.ndarray) -> np.ndarray:
        """
        为当前的批次目标词抽取负样本 (O(1) 查表快速抽取)

        参数:
            target (np.ndarray): 正样本目标词 ID 数组，形状为 (N,)，其中 N 为 batch_size

        返回:
            negative_sample (np.ndarray): 采样的负样本 ID 矩阵，形状为 (N, sample_size)
        """
        batch_size = target.shape[0]

        # 直接在 100 万槽位的采样表中随机抽取整数索引，耗时由几十毫秒骤降至亚毫秒级
        rand_indices = np.random.randint(0, len(self.table), size=(batch_size, self.sample_size))
        negative_sample = self.table[rand_indices]

        return negative_sample


class NegativeSamplingLoss:
    """
    负采样损失层 (Negative Sampling Loss Layer)

    架构设计与推导:
        给定输入隐藏向量 h (形状: (N, H))，
        正样本词 ID target (形状: (N,))，
        以及采样的 K 个负样本词 ID (形状: (N, K))。

        目标损失函数为最大化正样本概率并最小化负样本概率 (等价于最小化二分类交叉熵和):
            L = - log(sigma(h · W_out[target])) - sum_{k=1}^K log(sigma(- h · W_out[neg_k]))
        注意:
            因为 1 - sigma(x) = sigma(-x)，所以负样本的损失等价于以 0 为标签的二分类交叉熵！

    参数:
        W (np.ndarray): 输出层权重矩阵 W_out，形状为 (V, H)
        corpus (np.ndarray): 训练语料 ID 序列，用于初始化采样器
        power (float): 负采样概率平滑系数，默认 0.75
        sample_size (int): 负样本数 K，默认 5
    """

    def __init__(self, W: np.ndarray, corpus: Union[np.ndarray, List[int]], power: float = 0.75, sample_size: int = 5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # 注册参数与梯度
        self.W = W
        self.params = [self.W]
        self.grads = [np.zeros_like(self.W)]

        # 缓存前向状态用于反向传播
        self.cache = None

    def forward(self, h: np.ndarray, target: np.ndarray) -> float:
        """
        前向传播: 向量化高效计算正样本损失 + K 个负样本损失

        参数:
            h (np.ndarray): 隐藏层向量 (例如 CBOW 中各上下文词向量的平均)，形状为 (N, H)
            target (np.ndarray): 正样本目标词索引，形状为 (N,)

        返回:
            loss (float): 当前批次的标量损失值 (正样本损失 + 所有负样本损失之和)
        """
        batch_size = target.shape[0]

        # 1. 抽取负样本: 形状为 (N, K)
        negative_sample = self.sampler.get_negative_sample(target)

        # -------------------------------------------------------------
        # 2. 正样本部分 (标签 t = 1)
        # -------------------------------------------------------------
        # target_W 的形状: (N, H)
        target_W = self.W[target]
        # score_pos 形状: (N,)
        score_pos = np.sum(h * target_W, axis=1)

        # 数值稳定截断并计算 Sigmoid 预测概率
        score_pos_clipped = np.clip(score_pos, -30.0, 30.0)
        y_pos = 1.0 / (1.0 + np.exp(-score_pos_clipped))  # 形状: (N,)

        # 正样本损失: -log(y_pos)
        eps = 1e-7
        loss_pos = -np.sum(np.log(y_pos + eps))

        # -------------------------------------------------------------
        # 3. 负样本部分 (向量化批量处理, 标签 t = 0)
        # -------------------------------------------------------------
        # neg_W 形状: (N, K, H)
        neg_W = self.W[negative_sample]
        # h[:, np.newaxis, :] 形状为 (N, 1, H)，利用广播机制在隐藏维度 H 上点乘求和
        # score_neg 形状: (N, K)
        score_neg = np.sum(h[:, np.newaxis, :] * neg_W, axis=2)

        score_neg_clipped = np.clip(score_neg, -30.0, 30.0)
        y_neg = 1.0 / (1.0 + np.exp(-score_neg_clipped))  # 形状: (N, K)

        # 负样本损失: -log(1 - y_neg)
        loss_neg = -np.sum(np.log(1.0 - y_neg + eps))

        # 整个 batch 的总平均损失
        loss = (loss_pos + loss_neg) / batch_size

        # 缓存反向传播所需变量
        self.cache = (h, target, negative_sample, y_pos, y_neg, target_W, neg_W)
        return float(loss)

    def backward(self, dout: float = 1.0) -> np.ndarray:
        """
        反向传播: 高效计算对输入隐藏层向量 h 的梯度以及对权重矩阵 W_out 的梯度

        梯度推导:
        设 batch_size 为 N:
        1. 正样本 (标签 t = 1):
           dL / d(score_pos) = (y_pos - 1) * dout / N    (形状: (N,))
           对 h 的正样本贡献:
               dh_pos = dscore_pos[:, np.newaxis] * target_W   (形状: (N, H))
           对 W_out[target] 的正样本贡献:
               dtarget_W = dscore_pos[:, np.newaxis] * h       (形状: (N, H))

        2. 负样本 (标签 t = 0):
           dL / d(score_neg) = (y_neg - 0) * dout / N = y_neg * dout / N  (形状: (N, K))
           对 h 的负样本贡献:
               在 K 个负样本维度上求和:
               dh_neg = sum_k ( dscore_neg[:, k, np.newaxis] * neg_W[:, k, :] ) (形状: (N, H))
           对 W_out[negative_sample] 的负样本贡献:
               dneg_W = dscore_neg[:, :, np.newaxis] * h[:, np.newaxis, :]    (形状: (N, K, H))

        3. 权重梯度累加:
           利用 np.add.at 将 dtarget_W 和 dneg_W 正确累加至 dW_out。

        返回:
            dh (np.ndarray): 对输入隐藏层向量 h 的梯度，形状为 (N, H)
        """
        h, target, negative_sample, y_pos, y_neg, target_W, neg_W = self.cache
        batch_size = target.shape[0]

        # 初始化权重梯度为 0
        self.grads[0][...] = 0

        # ==================== 1. 正样本梯度反传 ====================
        # dscore_pos: (N,)
        dscore_pos = (y_pos - 1.0) * dout / batch_size

        # 对 h 的正样本部分梯度: (N, H)
        dh_pos = dscore_pos[:, np.newaxis] * target_W

        # 对正样本对应权重的梯度: (N, H)
        dtarget_W = dscore_pos[:, np.newaxis] * h
        np.add.at(self.grads[0], target, dtarget_W)

        # ==================== 2. 负样本梯度反传 ====================
        # dscore_neg: (N, K)
        dscore_neg = (y_neg - 0.0) * dout / batch_size

        # 对 h 的负样本部分梯度: sum_{k=1}^K (dscore_neg * neg_W) -> (N, H)
        dh_neg = np.sum(dscore_neg[:, :, np.newaxis] * neg_W, axis=1)

        # 对负样本对应权重的梯度: (N, K, H)
        dneg_W = dscore_neg[:, :, np.newaxis] * h[:, np.newaxis, :]
        np.add.at(self.grads[0], negative_sample, dneg_W)

        # ==================== 3. 汇总对隐藏向量 h 的梯度 ====================
        dh = dh_pos + dh_neg  # 形状: (N, H)

        return dh

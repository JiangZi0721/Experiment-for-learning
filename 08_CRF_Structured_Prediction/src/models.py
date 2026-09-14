# -*- coding: utf-8 -*-
"""
Models for Sequence Labeling:
1. Generative Baseline: HMM (隐马尔可夫模型)
2. Local Discriminative: BiLSTM-Softmax (局部独立分类)
3. Global Discriminative: BiLSTM-CRF (全局转移约束)
"""
import torch
import torch.nn as nn
import numpy as np
from src.crf import LinearChainCRF

class HMM:
    """
    经典离散隐马尔可夫模型 (Hidden Markov Model, HMM)
    生成式模型：联合概率 P(X, Y) = \prod P(y_t | y_{t-1}) P(x_t | y_t)
    """
    def __init__(self, num_states, num_vocab, laplace_alpha=1e-3):
        self.K = num_states
        self.V = num_vocab
        self.alpha = laplace_alpha
        self.start_p = np.zeros(self.K)
        self.trans_p = np.zeros((self.K, self.K))
        self.emit_p = np.zeros((self.K, self.V))

    def fit(self, sentences, tag_sequences):
        """极大似然估计 + 拉普拉斯平滑参数拟合"""
        start_counts = np.full(self.K, self.alpha)
        trans_counts = np.full((self.K, self.K), self.alpha)
        emit_counts = np.full((self.K, self.V), self.alpha)

        for sent, tags in zip(sentences, tag_sequences):
            if len(tags) == 0:
                continue
            start_counts[tags[0]] += 1.0
            emit_counts[tags[0], sent[0]] += 1.0
            for t in range(1, len(tags)):
                trans_counts[tags[t-1], tags[t]] += 1.0
                emit_counts[tags[t], sent[t]] += 1.0

        self.start_p = np.log(start_counts / np.sum(start_counts))
        self.trans_p = np.log(trans_counts / np.sum(trans_counts, axis=1, keepdims=True))
        self.emit_p = np.log(emit_counts / np.sum(emit_counts, axis=1, keepdims=True))

    def decode(self, sentence):
        """经典维特比算法解码求最优状态序列"""
        T = len(sentence)
        if T == 0:
            return []
        viterbi = np.full((T, self.K), -np.inf)
        backpointers = np.zeros((T, self.K), dtype=int)

        w0 = sentence[0]
        viterbi[0] = self.start_p + self.emit_p[:, w0]

        for t in range(1, T):
            wt = sentence[t]
            for curr_s in range(self.K):
                trans_scores = viterbi[t-1] + self.trans_p[:, curr_s]
                best_prev = np.argmax(trans_scores)
                viterbi[t, curr_s] = trans_scores[best_prev] + self.emit_p[curr_s, wt]
                backpointers[t, curr_s] = best_prev

        best_last = np.argmax(viterbi[T-1])
        best_path = [best_last]
        for t in range(T-1, 0, -1):
            best_last = backpointers[t, best_last]
            best_path.append(best_last)
        best_path.reverse()
        return best_path


class BiLSTM_Softmax(nn.Module):
    """
    BiLSTM + 局部 Softmax 分类器
    局部归一化 (Local Normalization)：每个时间步独立做交叉熵，无法施加全局合法约束
    """
    def __init__(self, vocab_size, num_tags, embedding_dim=64, hidden_dim=64):
        super(BiLSTM_Softmax, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=False
        )
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        """
        :param x: [seq_len, batch_size]
        :return: logits [seq_len, batch_size, num_tags]
        """
        embeds = self.embedding(x)  # [seq_len, batch_size, emb_dim]
        lstm_out, _ = self.lstm(embeds)  # [seq_len, batch_size, hidden_dim]
        logits = self.fc(lstm_out)  # [seq_len, batch_size, num_tags]
        return logits

    def compute_loss(self, logits, tags, mask=None):
        """
        局部 CrossEntropy 逐 Token 损失
        """
        seq_len, batch_size, num_tags = logits.shape
        loss = self.loss_fn(logits.view(-1, num_tags), tags.view(-1))  # [seq_len * batch_size]
        loss = loss.view(seq_len, batch_size)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def decode(self, logits, mask=None):
        """
        贪婪解码：每步独立选择 argmax
        """
        preds = torch.argmax(logits, dim=2)  # [seq_len, batch_size]
        seq_len, batch_size = preds.shape
        best_paths = []
        for b in range(batch_size):
            if mask is not None:
                length = int(mask[:, b].sum().item())
            else:
                length = seq_len
            best_paths.append(preds[:length, b].tolist())
        return best_paths


class BiLSTM_CRF(nn.Module):
    """
    BiLSTM + 全局线性链条件随机场 (Global Linear-Chain CRF)
    全局归一化 (Global Normalization)：打破标注偏置，严守语法转移结构约束
    """
    def __init__(self, vocab_size, num_tags, start_tag_idx, stop_tag_idx, embedding_dim=64, hidden_dim=64):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=False
        )
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = LinearChainCRF(num_tags, start_tag_idx, stop_tag_idx)

    def get_emissions(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)
        return emissions

    def forward_loss(self, x, tags, mask=None):
        emissions = self.get_emissions(x)
        return self.crf.forward_loss(emissions, tags, mask)

    def decode(self, x, mask=None):
        emissions = self.get_emissions(x)
        return self.crf.decode(emissions, mask)

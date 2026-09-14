# -*- coding: utf-8 -*-
"""
White-box Linear-Chain Conditional Random Field (CRF) Implementation
纯白盒线性链条件随机场底层实现 (PyTorch 纯手推版)

包含：
1. 对数空间配分函数 log Z(x) 计算 (Log-Sum-Exp 稳定数值技巧)
2. 真实路径能量打分 S(x, y)
3. 动态规划维特比算法 (Viterbi Decoding with Backpointers)
4. 负对数似然损失 (Negative Log-Likelihood Loss)
"""
import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class LinearChainCRF(nn.Module):
    def __init__(self, num_tags, start_tag_idx, stop_tag_idx):
        """
        初始化线性链条件随机场
        :param num_tags: 标签总数 (包括 START 与 STOP)
        :param start_tag_idx: 虚拟起始标签索引
        :param stop_tag_idx: 虚拟结束标签索引
        """
        super(LinearChainCRF, self).__init__()
        self.num_tags = num_tags
        self.start_tag_idx = start_tag_idx
        self.stop_tag_idx = stop_tag_idx

        # 状态转移矩阵 transitions[i, j] 表示从标签 j 转移到标签 i 的转移分数
        # 维度: [num_tags, num_tags]
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))

        # 强制结构化硬约束：
        # 1. 任何状态都绝对不能转移到 START
        # 2. 从 STOP 绝对不能转移到任何状态
        self.transitions.data[self.start_tag_idx, :] = -10000.0
        self.transitions.data[:, self.stop_tag_idx] = -10000.0

    def _forward_alg(self, emissions, mask=None):
        """
        前向算法 (Forward Algorithm)：计算对数配分函数 log Z(x)
        配分函数代表所有可能路径的未归一化概率总和:
        Z(x) = \sum_{y} \exp(S(x, y))
        采用 log-sum-exp 动态规划避免浮点溢出与下溢

        :param emissions: 发射分数张量 [seq_len, batch_size, num_tags]
        :param mask: 掩码张量 [seq_len, batch_size] (1 表示有效 token, 0 表示填充)
        :return: 每条序列的 log Z(x), 维度 [batch_size]
        """
        seq_len, batch_size, _ = emissions.shape
        if mask is None:
            mask = torch.ones(seq_len, batch_size, device=emissions.device, dtype=torch.bool)

        # 初始化前向变量：初始时刻前向能量全为极小值，仅 START 标签能量为 0
        init_alphas = torch.full((batch_size, self.num_tags), -10000.0, device=emissions.device)
        init_alphas[:, self.start_tag_idx] = 0.0

        forward_var = init_alphas

        for t in range(seq_len):
            emit_score = emissions[t]  # [batch_size, num_tags]
            mask_t = mask[t].unsqueeze(1)  # [batch_size, 1]

            # 广播计算一步转移能量:
            # forward_var: [batch_size, 1, num_tags] (前一步标签 j)
            # transitions: [1, num_tags, num_tags] (当前标签 i, 前一步标签 j)
            # emit_score:  [batch_size, num_tags, 1] (当前标签 i 的发射能量)
            prev_tag_var = forward_var.unsqueeze(1)  # [batch_size, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            emit_expanded = emit_score.unsqueeze(2)  # [batch_size, num_tags, 1]

            # next_tag_var: [batch_size, num_tags, num_tags]
            # 表示从前一步 j 转移到当前步 i，并加上当前步发射分数的总分数
            next_tag_var = prev_tag_var + trans_score + emit_expanded

            # 对前一步所有可能标签 j 求 logsumexp 归约
            # 得到当前步各个标签 i 的累积前向分数: [batch_size, num_tags]
            alphas_t = torch.logsumexp(next_tag_var, dim=2)

            # 若当前 token 为 padding (mask=0)，则保持前一步能量不变
            forward_var = torch.where(mask_t, alphas_t, forward_var)

        # 最终步：汇聚转移至虚拟 STOP 标签
        terminal_vars = forward_var + self.transitions[self.stop_tag_idx].unsqueeze(0)
        alpha = torch.logsumexp(terminal_vars, dim=1)
        return alpha

    def _score_sentence(self, emissions, tags, mask=None):
        """
        计算目标真实金标序列 y 的未归一化能量得分 S(x, y)
        S(x, y) = \sum_{t=0}^{T-1} Emit(x_t, y_t) + \sum_{t=1}^{T} Trans(y_{t-1}, y_t)

        :param emissions: [seq_len, batch_size, num_tags]
        :param tags: 金标序列 [seq_len, batch_size]
        :param mask: [seq_len, batch_size]
        :return: [batch_size] 真实路径得分
        """
        seq_len, batch_size, _ = emissions.shape
        if mask is None:
            mask = torch.ones(seq_len, batch_size, device=emissions.device, dtype=torch.bool)

        score = torch.zeros(batch_size, device=emissions.device)

        # 在序列首位拼接 START 标签
        padded_tags = torch.cat(
            [torch.full((1, batch_size), self.start_tag_idx, dtype=torch.long, device=tags.device), tags],
            dim=0
        )

        for t in range(seq_len):
            mask_t = mask[t]
            cur_tag = padded_tags[t + 1]
            prev_tag = padded_tags[t]

            # 累加发射得分
            emit_score = emissions[t, torch.arange(batch_size), cur_tag]
            # 累加转移得分 (从 prev_tag 转移到 cur_tag)
            trans_score = self.transitions[cur_tag, prev_tag]

            score = score + torch.where(mask_t, emit_score + trans_score, torch.zeros_like(score))

        # 找到每个 batch 样本序列的最后一个真实有效 token，追加转移到 STOP 的得分
        # 计算每个样本的实际长度
        seq_lengths = mask.sum(dim=0).long()  # [batch_size]
        for b in range(batch_size):
            last_tag = padded_tags[seq_lengths[b], b]
            score[b] = score[b] + self.transitions[self.stop_tag_idx, last_tag]

        return score

    def forward_loss(self, emissions, tags, mask=None):
        """
        计算条件随机场负对数似然损失 (Negative Log-Likelihood, NLL)
        Loss = -\log P(y|x) = -\log \frac{\exp(S(x, y))}{Z(x)} = \log Z(x) - S(x, y)
        """
        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return torch.mean(forward_score - gold_score)

    def decode(self, emissions, mask=None):
        """
        维特比算法 (Viterbi Algorithm) 解码求全局最优标签序列
        \hat{y} = \arg\max_{y} S(x, y)

        :param emissions: [seq_len, batch_size, num_tags]
        :param mask: [seq_len, batch_size]
        :return: List of best paths for each batch item
        """
        seq_len, batch_size, _ = emissions.shape
        if mask is None:
            mask = torch.ones(seq_len, batch_size, device=emissions.device, dtype=torch.bool)

        best_paths = []
        for b in range(batch_size):
            seq_len_b = int(mask[:, b].sum().item())
            if seq_len_b == 0:
                best_paths.append([])
                continue

            emit_b = emissions[:seq_len_b, b]  # [seq_len_b, num_tags]

            # 初始化维特比变量
            viterbi_vars = torch.full((1, self.num_tags), -10000.0, device=emissions.device)
            viterbi_vars[0, self.start_tag_idx] = 0.0

            backpointers = []

            for t in range(seq_len_b):
                emit_score = emit_b[t].unsqueeze(0)  # [1, num_tags]
                # viterbi_vars.unsqueeze(1): [1, 1, num_tags]
                # self.transitions: [num_tags, num_tags] (i: 当前步, j: 前一步)
                next_tag_vars = viterbi_vars.unsqueeze(1) + self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
                max_vars, bptrs = torch.max(next_tag_vars, dim=2)  # [1, num_tags]

                viterbi_vars = max_vars + emit_score
                backpointers.append(bptrs.squeeze(0).tolist())

            # 汇聚转移至 STOP 标签
            terminal_vars = viterbi_vars + self.transitions[self.stop_tag_idx].unsqueeze(0)
            best_tag_id = torch.argmax(terminal_vars, dim=1).item()

            # 回溯解码路径
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            # 剔除首部的 START 标签并翻转顺序
            start = best_path.pop()
            assert start == self.start_tag_idx
            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

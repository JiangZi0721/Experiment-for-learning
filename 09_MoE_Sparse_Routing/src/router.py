# -*- coding: utf-8 -*-
"""
White-box MoE Router Module:
1. Top-K Softmax Gating & Noisy Router
2. Auxiliary Load Balancing Loss (Switch/GShard 经典辅助损失)
3. DeepSeek-V3 式免辅助损失动态偏置算法 (Auxiliary-Loss-Free Dynamic Bias)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2, aux_loss_coef=0.01, noisy_gating=False):
        super(TopKRouter, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coef = aux_loss_coef
        self.noisy_gating = noisy_gating

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        if noisy_gating:
            self.w_noise = nn.Linear(d_model, num_experts, bias=False)

        # DeepSeek-V3 式免辅助损失动态路由偏置 (可自适应在线更新)
        self.register_buffer("dynamic_bias", torch.zeros(num_experts))
        self.use_dynamic_bias = False

    def forward(self, x, apply_aux_loss=True):
        """
        :param x: [batch_size, seq_len, d_model] 或 [num_tokens, d_model]
        :param apply_aux_loss: 是否计算负载均衡辅助损失
        :return:
            routing_weights: [num_tokens, top_k] 归一化后的门控权重
            selected_experts: [num_tokens, top_k] 选中的专家 ID
            aux_loss: 标量辅助损失
        """
        orig_shape = x.shape
        if len(orig_shape) == 3:
            num_tokens = orig_shape[0] * orig_shape[1]
            x_flat = x.view(num_tokens, self.d_model)
        else:
            num_tokens = orig_shape[0]
            x_flat = x

        logits = self.gate(x_flat)  # [num_tokens, num_experts]

        if self.noisy_gating and self.training:
            noise = torch.randn_like(logits) * F.softplus(self.w_noise(x_flat))
            logits = logits + noise

        # 如果开启了 DeepSeek-V3 动态偏置算法，将无梯度的偏置叠加至路由判决
        if self.use_dynamic_bias:
            routing_logits = logits + self.dynamic_bias.unsqueeze(0)
        else:
            routing_logits = logits

        # 全局 Softmax 概率分布 P_i (用于辅助损失计算)
        probs = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]

        # Top-K 选路 (基于 routing_logits)
        topk_scores, selected_experts = torch.topk(routing_logits, self.top_k, dim=-1)  # [num_tokens, top_k]

        # 对选出的 Top-K 权重重新进行局部 Softmax 归一化
        routing_weights = F.softmax(topk_scores, dim=-1)  # [num_tokens, top_k]

        # 计算负载均衡辅助损失 (Auxiliary Load Balancing Loss)
        # L_aux = alpha * N * \sum_{i=1}^N (f_i * P_i)
        # 其中 f_i 为实际分发到专家 i 的 Token 比例，P_i 为全 Token 对专家 i 的平均门控概率
        aux_loss = torch.tensor(0.0, device=x.device)
        if apply_aux_loss and self.training:
            # 统计分发比例 f_i
            # 统计 selected_experts 中每个专家出现的频次
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()  # [num_tokens, top_k, num_exp]
            tokens_per_expert = expert_mask.sum(dim=[0, 1])  # [num_exp]
            f_i = tokens_per_expert / (num_tokens * self.top_k)  # [num_exp]
            P_i = probs.mean(dim=0)  # [num_exp]

            aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(f_i * P_i)

        # 动态偏置在线平滑更新 (若开启)
        if self.use_dynamic_bias and self.training:
            with torch.no_grad():
                tokens_per_expert = F.one_hot(selected_experts, num_classes=self.num_experts).float().sum(dim=[0, 1])
                actual_ratio = tokens_per_expert / (num_tokens * self.top_k)
                target_ratio = 1.0 / self.num_experts
                # 如果某专家承载过多 (actual > target)，减小其偏置；反之增大
                delta = target_ratio - actual_ratio
                self.dynamic_bias.add_(0.01 * delta)

        return routing_weights, selected_experts, aux_loss

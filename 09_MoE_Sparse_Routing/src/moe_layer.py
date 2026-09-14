# -*- coding: utf-8 -*-
"""
Mixture of Experts Layer Architectures:
1. ExpertFFN / DenseFFN
2. StandardMoE (经典稀疏门控 MoE)
3. DeepSeekMoE (细粒度专家 + 隔离共享专家架构)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.router import TopKRouter

class ExpertFFN(nn.Module):
    """单专家前馈网络 (SwiGLU/GELU 风格)"""
    def __init__(self, d_model, d_ff):
        super(ExpertFFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class DenseFFN(nn.Module):
    """基准 Dense 前馈网络 (同计算量基线)"""
    def __init__(self, d_model, d_ff):
        super(DenseFFN, self).__init__()
        self.ffn = ExpertFFN(d_model, d_ff)

    def forward(self, x):
        return self.ffn(x), torch.tensor(0.0, device=x.device)

class StandardMoE(nn.Module):
    """
    经典稀疏门控 MoE (如 Switch Transformer / GShard)
    N 个独立专家，每次动态激活 Top-K 个专家
    """
    def __init__(self, d_model, d_ff, num_experts=4, top_k=1, aux_loss_coef=0.01):
        super(StandardMoE, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(d_model, num_experts, top_k=top_k, aux_loss_coef=aux_loss_coef)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x, apply_aux_loss=True):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            b, s, d = orig_shape
            x_flat = x.view(b * s, d)
        else:
            x_flat = x

        num_tokens = x_flat.shape[0]
        routing_weights, selected_experts, aux_loss = self.router(x_flat, apply_aux_loss=apply_aux_loss)

        # 稀疏路由调度：按专家聚合 Token 执行计算
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            exp_indices = selected_experts[:, k]  # [num_tokens]
            weights = routing_weights[:, k].unsqueeze(1)  # [num_tokens, 1]

            for exp_id in range(self.num_experts):
                mask = (exp_indices == exp_id)
                if mask.any():
                    tokens_in = x_flat[mask]
                    tokens_out = self.experts[exp_id](tokens_in)
                    out[mask] += tokens_out * weights[mask]

        if len(orig_shape) == 3:
            out = out.view(orig_shape)

        return out, aux_loss

class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE 架构创新 (细粒度路由专家 + 确定性隔离共享专家)
    论文：DeepSeekMoE: Towards Ultimate Expertise in Mixture-of-Experts
    核心设计：
    1. 共享专家 (Shared Experts)：固定激活，兜底承载全领域的公共常识与语法结构
    2. 细粒度路由专家 (Fine-Grained Routed Experts)：将专家切小（如分割为 8 个微专家），激活 Top-2
    公式：y = FFN_shared(x) + \sum_{i \in TopK} w_i * FFN_routed_i(x)
    """
    def __init__(self, d_model, d_ff_total, num_routed_experts=8, top_k=2, num_shared_experts=1, aux_loss_coef=0.01):
        super(DeepSeekMoE, self).__init__()
        self.d_model = d_model
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.num_shared = num_shared_experts

        # 每个细粒度专家的隐藏层维度均分
        d_ff_expert = d_ff_total // num_routed_experts
        d_ff_shared = d_ff_total // num_routed_experts * num_shared_experts

        # 1. 共享专家 (始终参与计算)
        self.shared_experts = ExpertFFN(d_model, d_ff_shared)

        # 2. 细粒度路由专家群
        self.routed_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff_expert) for _ in range(num_routed_experts)
        ])

        # 3. 细粒度路由器
        self.router = TopKRouter(d_model, num_routed_experts, top_k=top_k, aux_loss_coef=aux_loss_coef)

    def forward(self, x, apply_aux_loss=True):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            b, s, d = orig_shape
            x_flat = x.view(b * s, d)
        else:
            x_flat = x

        # 1. 共享专家计算 (公共知识高速公路)
        shared_out = self.shared_experts(x_flat)

        # 2. 细粒度路由计算
        routing_weights, selected_experts, aux_loss = self.router(x_flat, apply_aux_loss=apply_aux_loss)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            exp_indices = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(1)

            for exp_id in range(self.num_routed_experts):
                mask = (exp_indices == exp_id)
                if mask.any():
                    tokens_in = x_flat[mask]
                    tokens_out = self.routed_experts[exp_id](tokens_in)
                    routed_out[mask] += tokens_out * weights[mask]

        out = shared_out + routed_out

        if len(orig_shape) == 3:
            out = out.view(orig_shape)

        return out, aux_loss

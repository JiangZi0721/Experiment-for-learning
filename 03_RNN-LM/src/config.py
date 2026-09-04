# -*- coding: utf-8 -*-
"""
RNN-LM 全局配置与超参数控制中心
"""
from dataclasses import dataclass
from pathlib import Path

# 项目基础路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

@dataclass
class RNNConfig:
    """RNN 语言模型超参数配置"""
    # 模型架构超参数
    rnn_type: str = "vanilla"     # 循环层类型: "vanilla" (标准单步RNN) 或 "gru" (门控循环网络)
    vocab_size: int = 1000        # 词表/字符表大小（训练时由语料动态确定）
    wordvec_size: int = 64        # 词嵌入向量维度 D
    hidden_size: int = 128        # RNN 隐藏状态维度 H

    # 截断时序超参数 (Truncated BPTT)
    time_size: int = 20           # Truncated BPTT 单次展开的时间步长 T
    batch_size: int = 16          # 批大小 N

    # 训练与优化超参数
    max_epoch: int = 30           # 最大迭代轮数
    lr: float = 0.1               # 基础学习率
    max_grad_norm: float = 5.0    # 梯度裁剪阈值 (Gradient Clipping)
    weight_init_std: float = 0.01 # 权重高斯初始化标准差

    # 透视看板控制
    verbose: bool = True
    probe_step_interval: int = 50 # 探针采样步长

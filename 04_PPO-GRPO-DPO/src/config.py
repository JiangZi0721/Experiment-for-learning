import os
from pathlib import Path
import torch

class RLConfig:
    """强化学习白盒实验室全局配置"""
    # 基础路径
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    REASONING_TASKS_PATH = DATA_DIR / "reasoning_tasks.json"
    PREFERENCE_PAIRS_PATH = DATA_DIR / "preference_pairs.json"
    ALIGNMENT_PROMPTS_PATH = DATA_DIR / "alignment_prompts.json"
    BENCHMARK_CASES_PATH = DATA_DIR / "benchmark_cases.json"

    # 设备与随机种子
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # 词表与网络架构超参数 (教学级轻量模型，秒级收敛，显存友好)
    VOCAB_SIZE = 128
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    MAX_SEQ_LEN = 32

    # PPO 超参数
    PPO_EPSILON = 0.2       # Clip 截断范围 [1-eps, 1+eps]
    PPO_BETA = 0.05         # KL 罚分权重
    PPO_GAMMA = 0.99        # 折扣因子
    PPO_GAE_LAMBDA = 0.95   # GAE 平滑系数
    PPO_VF_COEF = 0.5       # Critic 损失权重
    PPO_ENTROPY_COEF = 0.01 # 动作熵探索权重
    PPO_LR_ACTOR = 1e-3
    PPO_LR_CRITIC = 2e-3

    # DPO 超参数
    DPO_BETA = 0.1          # 隐式奖励缩放系数与 KL 正则强度
    DPO_LR = 1e-3

    # GRPO 超参数 (DeepSeek 风格)
    GRPO_GROUP_SIZE = 4     # 组内采样数 G (如 4 或 8)
    GRPO_EPSILON = 0.2      # 组内 Clip 幅度
    GRPO_BETA = 0.04        # 参考模型 KL 正则系数
    GRPO_USE_UNBIASED_KL = True # 是否开启 DeepSeek-V3.2 无偏 KL (k3 估计器)
    GRPO_OFF_POLICY_MASK = True # 是否开启 Off-Policy 序列掩码
    GRPO_OFF_POLICY_DELTA = 0.5 # 序列偏离度阈值 delta
    GRPO_LR = 1e-3

    # 评估与日志
    LOG_INTERVAL = 1
    EVAL_TOP_K = 3

cfg = RLConfig()

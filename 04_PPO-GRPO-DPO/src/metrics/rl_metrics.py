from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import numpy as np

@dataclass
class RLMetricReport:
    """强化学习单轮或全流程评估指标报告"""
    algorithm: str
    step: int
    # 1. 训练动态指标
    policy_loss: float
    value_loss: Optional[float] = None
    mean_kl: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    policy_ratio: float = 1.0
    # 2. 优势与奖励指标
    mean_reward: float = 0.0
    reward_margin: Optional[float] = None # DPO 专属
    advantage_mean: float = 0.0
    advantage_std: float = 1.0
    # 3. 生成质量与对齐指标
    accuracy_pass1: float = 0.0
    format_compliance: float = 0.0
    win_rate_vs_base: float = 0.5
    avg_response_length: float = 0.0
    length_penalty_flag: bool = False
    # 4. 健康诊断与警报
    health_status: str = "HEALTHY" # HEALTHY, WARNING, CRITICAL
    diagnostic_warnings: List[str] = field(default_factory=list)

class RLEvaluationMetrics:
    """
    强化学习全维度评估指标计算与诊断引擎
    覆盖训练过程监控、生成质量评价、系统开销与异常退化防御
    """
    @staticmethod
    def evaluate_training_health(
        mean_kl: float,
        clip_fraction: float,
        entropy: float,
        avg_length: float,
        prev_avg_length: float = 0.0
    ) -> Dict[str, Any]:
        """
        自动化健康度诊断探针：
        - 检查 KL 是否爆炸 (策略脱缰漂移)
        - 检查 Clip 率是否过高/过低 (学习率是否失调)
        - 检查熵是否骤降至零 (模式坍塌 Mode Collapse)
        - 检查长度是否异常暴增 (长度作弊 Reward Hacking)
        """
        status = "HEALTHY"
        warnings = []

        # 1. KL 散度健康区间检测
        if mean_kl > 0.5:
            status = "CRITICAL"
            warnings.append(f"KL 散度过高 ({mean_kl:.3f} > 0.5)：策略已严重背离基座模型，可能发生语言能力崩溃！")
        elif mean_kl < 0.001:
            warnings.append(f"KL 散度过低 ({mean_kl:.4f})：模型更新过于保守，未有效吸收对齐偏好。")

        # 2. Clip 触发率健康区间检测 (经验区间 5% ~ 25%)
        if clip_fraction > 0.35:
            if status != "CRITICAL":
                status = "WARNING"
            warnings.append(f"Clip Fraction 截断率异常高 ({clip_fraction*100:.1f}% > 35%)：策略梯度步长过激，存在震荡风险。")
        elif clip_fraction < 0.02 and clip_fraction > 0:
            warnings.append(f"Clip Fraction 截断率过低 ({clip_fraction*100:.1f}%)：更新幅度未能触及截断边界，学习效率偏低。")

        # 3. 策略熵 (Entropy) 模式坍塌检测
        if entropy < 0.1:
            if status != "CRITICAL":
                status = "WARNING"
            warnings.append(f"动作熵骤降 ({entropy:.3f} < 0.1)：可能发生模式坍塌 (Mode Collapse)，多样性丧失。")

        # 4. 长度作弊 (Reward Hacking) 检测
        if prev_avg_length > 0 and avg_length > prev_avg_length * 2.0:
            if status != "CRITICAL":
                status = "WARNING"
            warnings.append(f"平均回答长度突增 ({prev_avg_length:.1f} -> {avg_length:.1f})：疑似发生长度偏见作弊 (Reward Hacking)！")

        return {
            "health_status": status,
            "diagnostic_warnings": warnings
        }

    @staticmethod
    def compute_dpo_metrics(
        reward_chosen: float,
        reward_rejected: float,
        pi_log_chosen: float,
        ref_log_chosen: float
    ) -> Dict[str, float]:
        """计算 DPO 专属指标"""
        margin = reward_chosen - reward_rejected
        win_rate = 1.0 / (1.0 + math.exp(-margin))
        chosen_improvement = pi_log_chosen - ref_log_chosen
        return {
            "reward_margin": margin,
            "chosen_win_rate": win_rate,
            "chosen_improvement": chosen_improvement
        }

    @staticmethod
    def compute_grpo_group_metrics(rewards: List[float]) -> Dict[str, float]:
        """计算 GRPO 组内赛马统计指标"""
        arr = np.array(rewards, dtype=np.float32)
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        max_val = float(arr.max())
        min_val = float(arr.min())
        # 组内容量区分度 (有效标准差)
        discrimination = std_val > 1e-4
        return {
            "group_mean": mean_val,
            "group_std": std_val,
            "group_max": max_val,
            "group_min": min_val,
            "has_discrimination": 1.0 if discrimination else 0.0
        }

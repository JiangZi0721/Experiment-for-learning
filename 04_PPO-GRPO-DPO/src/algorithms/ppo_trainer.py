import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

from ..config import cfg
from ..models.policy_network import WhiteBoxPolicyNetwork, ToyTokenizer
from ..models.critic_network import WhiteBoxCriticNetwork
from ..models.reward_engine import HybridRewardEngine

class WhiteBoxPPOTrainer:
    """
    白盒可透视 PPO 训练器 (Proximal Policy Optimization)
    严格实现四模型协同：Actor, Reference Model, Critic, Reward Model
    全程动态计算真实 Token 序列的 Critic 心电图时序、Advantage、Clip 截断与 KL 惩罚
    """
    def __init__(
        self,
        actor: WhiteBoxPolicyNetwork,
        critic: WhiteBoxCriticNetwork,
        reward_engine: HybridRewardEngine,
        tokenizer: ToyTokenizer,
        epsilon: float = cfg.PPO_EPSILON,
        beta_kl: float = cfg.PPO_BETA,
        gamma: float = cfg.PPO_GAMMA,
        gae_lambda: float = cfg.PPO_GAE_LAMBDA
    ):
        self.actor = actor
        self.critic = critic
        self.reward_engine = reward_engine
        self.tokenizer = tokenizer

        # 冻结的参考模型 (Reference Model)
        self.ref_model = copy.deepcopy(actor)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.epsilon = epsilon
        self.beta_kl = beta_kl
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=cfg.PPO_LR_ACTOR)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=cfg.PPO_LR_CRITIC)

    @torch.no_grad()
    def rollout(self, prompt_text: str, max_new_tokens: int = 25, temperature: float = 0.7) -> Dict[str, Any]:
        """
        真实让 Actor 策略生成回答，并由 Reward Engine 与 Critic 进行打分观测
        """
        self.actor.eval()
        self.critic.eval()

        prompt_tokens = self.tokenizer.encode(prompt_text, add_bos=True)
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long)

        full_ids, _ = self.actor.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=5,
            repetition_penalty=1.1
        )

        full_tokens = full_ids[0].tolist()
        prompt_len = len(prompt_tokens)
        resp_tokens = full_tokens[prompt_len:]

        full_text = self.tokenizer.decode(full_tokens)
        response_text = self.tokenizer.decode(resp_tokens)

        # Critic 前向预测序列各位置的价值 V(s_t)
        values = self.critic(full_ids)

        # Reward 评定
        reward_dict = self.reward_engine.compute_text_preference_reward(prompt_text, response_text)

        # 构建真实的 ECG 分段
        ecg_segments = []
        if len(response_text) > 0:
            chunk_size = max(len(response_text) // 3, 1)
            for i in range(0, len(response_text), chunk_size):
                seg_str = response_text[i : i + chunk_size]
                tok_start = min(prompt_len + i, len(full_tokens) - 1)
                tok_end = min(prompt_len + i + chunk_size, len(full_tokens))
                seg_val = values[0, tok_start:tok_end].mean().item()
                ecg_segments.append({
                    "stage": f"第 {i+1}~{min(i+len(seg_str), len(response_text))} 字",
                    "text": seg_str,
                    "value": seg_val
                })

        return {
            "prompt": prompt_text,
            "response": response_text,
            "full_text": full_text,
            "reward": reward_dict["total_reward"],
            "reward_critique": reward_dict["critique"],
            "ecg_segments": ecg_segments,
            "mean_critic_value": values[0, prompt_len:].mean().item() if len(resp_tokens) > 0 else 0.0
        }

    def train_step(
        self,
        prompt_text: str,
        response_text: str,
        use_clip: bool = True,
        beta_kl: Optional[float] = None,
        num_epochs: int = 4
    ) -> Dict[str, Any]:
        """
        真实执行 PPO / 策略梯度参数更新，并记录训练过程中的真实数值演变
        """
        self.actor.train()
        self.critic.train()

        if beta_kl is None:
            beta_kl = self.beta_kl

        prompt_tokens = self.tokenizer.encode(prompt_text, add_bos=True)
        response_tokens = self.tokenizer.encode(response_text, add_bos=False)
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        seq_len = len(full_tokens)

        full_ids = torch.tensor([full_tokens], dtype=torch.long)

        # 1. 评估旧策略基线与参考模型
        with torch.no_grad():
            old_log_probs, old_entropy = self.actor.evaluate_actions(full_ids)
            ref_log_probs, _ = self.ref_model.evaluate_actions(full_ids)
            old_gen_log = old_log_probs[:, prompt_len-1 : seq_len-1]
            ref_gen_log = ref_log_probs[:, prompt_len-1 : seq_len-1]

        # 2. 计算奖励
        reward_dict = self.reward_engine.compute_text_preference_reward(prompt_text, response_text)
        final_reward = reward_dict["total_reward"]

        policy_losses = []
        value_losses = []
        ratios = []
        clip_fractions = []
        kl_divs = []

        for _ in range(num_epochs):
            cur_log_probs, cur_entropy = self.actor.evaluate_actions(full_ids)
            cur_gen_log = cur_log_probs[:, prompt_len-1 : seq_len-1]

            # 真实 KL 惩罚
            kl_div = 0.5 * ((cur_gen_log - ref_gen_log) ** 2)
            mean_kl = kl_div.mean().item()
            kl_divs.append(mean_kl)

            # Critic 当前估计
            cur_crit_vals = self.critic(full_ids)[:, prompt_len-1 : seq_len-1]

            # 优势 Advantage = (Reward - beta * KL) - Critic Baseline
            raw_advantage = (final_reward - beta_kl * kl_div.detach()) - cur_crit_vals.detach()
            # 单序列维度优势缩放（避免扣减单序列均值导致正样本后半截被施加负惩罚）
            norm_advantage = (raw_advantage / (raw_advantage.std() + 1.0)).detach()

            # 重要性采样比率 r_t
            ratio = torch.exp(cur_gen_log - old_gen_log)
            ratios.append(ratio.mean().item())

            surr1 = ratio * norm_advantage
            if use_clip:
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * norm_advantage
                policy_loss = -torch.min(surr1, surr2).mean() - cfg.PPO_ENTROPY_COEF * cur_entropy
                clip_mask = (ratio > (1.0 + self.epsilon)) | (ratio < (1.0 - self.epsilon))
                clip_frac = clip_mask.float().mean().item()
            else:
                # 朴素策略梯度 (未做 Clip 截断)
                policy_loss = -surr1.mean() - cfg.PPO_ENTROPY_COEF * cur_entropy
                clip_frac = 0.0

            clip_fractions.append(clip_frac)

            # Critic MSE 价值损失
            target_returns = torch.full_like(cur_crit_vals, final_reward)
            value_loss = F.mse_loss(cur_crit_vals, target_returns)

            total_loss = policy_loss + cfg.PPO_VF_COEF * value_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # 训练完成后重新观测最终状态
        with torch.no_grad():
            fin_log_probs, fin_entropy = self.actor.evaluate_actions(full_ids)
            fin_gen_log = fin_log_probs[:, prompt_len-1 : seq_len-1]
            fin_ratio = torch.exp(fin_gen_log - old_gen_log)
            fin_clip_mask = (fin_ratio > (1.0 + self.epsilon)) | (fin_ratio < (1.0 - self.epsilon))
            fin_kl = (0.5 * ((fin_gen_log - ref_gen_log) ** 2)).mean().item()
            fin_values = self.critic(full_ids)[:, prompt_len-1 : seq_len-1]

        # 计算分段 ECG
        chars = list(response_text)
        chunk_size = max(len(chars) // 3, 1)
        ecg_segments = []
        i = 0
        prev_val = 0.0
        while i < len(chars):
            seg_text = "".join(chars[i : i + chunk_size])
            start_tok = prompt_len + i
            end_tok = min(prompt_len + i + chunk_size, seq_len)
            seg_val = fin_values[0, start_tok-prompt_len : end_tok-prompt_len].mean().item() if (end_tok > start_tok) else fin_values[0, -1].item()
            delta = seg_val - prev_val
            if delta > 0.2:
                trend = "↗ 稳步上升"
            elif delta < -0.2:
                trend = "↘ 明显下滑"
            else:
                trend = "→ 走势平稳"
            diag = "模型预测该段处于正向偏好区间" if seg_val > 0 else "模型预测该段可能违规或偏离"
            ecg_segments.append({
                "stage": f"第 {i+1}~{min(i+len(seg_text), len(chars))} 字",
                "text": seg_text,
                "value": seg_val,
                "trend": trend,
                "diagnosis": diag
            })
            prev_val = seg_val
            i += chunk_size

        return {
            "algorithm": "PPO",
            "prompt": prompt_text,
            "response": response_text,
            "final_reward": final_reward,
            "reward_critique": reward_dict.get("critique", ""),
            "ecg_segments": ecg_segments,
            "mean_kl": fin_kl,
            "policy_loss": policy_losses[-1],
            "value_loss": value_losses[-1],
            "mean_ratio": fin_ratio.mean().item(),
            "clip_fraction": fin_clip_mask.float().mean().item() if use_clip else 0.0,
            "mean_advantage": norm_advantage.mean().item(),
            "entropy": fin_entropy.item(),
            "use_clip": use_clip,
            "beta_kl": beta_kl
        }

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

from ..config import cfg
from ..models.policy_network import WhiteBoxPolicyNetwork, ToyTokenizer
from ..models.reward_engine import HybridRewardEngine

class WhiteBoxGRPOTrainer:
    """
    白盒可透视 GRPO 训练器 (Group Relative Policy Optimization)
    彻底告别 Critic 模型，基于组内赛马采样 (G 个候选) 与组内 z-score 归一化计算优势
    集成 DeepSeek-V3.2 无偏 KL (k3 估计器) 与 Off-Policy 序列掩码防御机制
    """
    def __init__(
        self,
        actor: WhiteBoxPolicyNetwork,
        reward_engine: HybridRewardEngine,
        tokenizer: ToyTokenizer,
        group_size: int = cfg.GRPO_GROUP_SIZE,
        epsilon: float = cfg.GRPO_EPSILON,
        beta_kl: float = cfg.GRPO_BETA,
        use_unbiased_kl: bool = cfg.GRPO_USE_UNBIASED_KL,
        use_off_policy_mask: bool = cfg.GRPO_OFF_POLICY_MASK,
        off_policy_delta: float = cfg.GRPO_OFF_POLICY_DELTA,
        lr: float = cfg.GRPO_LR
    ):
        self.actor = actor
        self.reward_engine = reward_engine
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.epsilon = epsilon
        self.beta_kl = beta_kl
        self.use_unbiased_kl = use_unbiased_kl
        self.use_off_policy_mask = use_off_policy_mask
        self.off_policy_delta = off_policy_delta

        # 参考模型 (冻结)
        self.ref_model = copy.deepcopy(actor)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # 旧策略快照 (用于重要性采样比率 r_i)
        self.old_actor = copy.deepcopy(actor)
        for p in self.old_actor.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)

    @torch.no_grad()
    def sample_group_responses(
        self,
        prompt_text: str,
        ground_truth: str,
        G: int = 4,
        temperature: float = 0.8
    ) -> Dict[str, Any]:
        """
        利用 Actor 策略对同一 Prompt 真实采样生成 G 个不同回答，并逐一计算规则奖励与组内优势
        """
        self.actor.eval()
        prompt_tokens = self.tokenizer.encode(prompt_text, add_bos=True)
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long)

        candidates = []
        reward_details = []
        raw_rewards = []

        for _ in range(G):
            full_ids, _ = self.actor.generate(
                prompt_ids,
                max_new_tokens=25,
                temperature=temperature,
                top_k=8,
                repetition_penalty=1.1
            )
            full_toks = full_ids[0].tolist()
            resp_toks = full_toks[len(prompt_tokens):]
            resp_text = self.tokenizer.decode(resp_toks)
            eval_res = self.reward_engine.compute_rule_reward(resp_text, ground_truth)

            candidates.append(resp_text)
            reward_details.append(eval_res)
            raw_rewards.append(eval_res["total_reward"])

        reward_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
        group_mean = reward_tensor.mean().item()
        group_std = reward_tensor.std(unbiased=False).item()

        if group_std < 1e-6:
            advantages = [0.0] * G
        else:
            advantages = ((reward_tensor - group_mean) / (group_std + 1e-8)).tolist()

        traces = []
        for i in range(G):
            traces.append({
                "candidate_index": i + 1,
                "response": candidates[i],
                "accuracy": reward_details[i]["accuracy_reward"],
                "format": reward_details[i]["format_reward"],
                "raw_reward": raw_rewards[i],
                "advantage": advantages[i]
            })

        return {
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "group_size": G,
            "candidates": candidates,
            "group_mean": group_mean,
            "group_std": group_std,
            "traces": traces
        }

    def train_group_step(
        self,
        prompt_text: str,
        ground_truth: str,
        candidate_responses: Optional[List[str]] = None,
        temperature: float = 0.8,
        num_epochs: int = 3
    ) -> Dict[str, Any]:
        """
        单步透视 GRPO 赛马训练：
        若未提供候选，则真实采样生成 G 个回答；随后执行梯度更新与无偏 KL 约束
        """
        if candidate_responses is None:
            sample_data = self.sample_group_responses(prompt_text, ground_truth, G=self.group_size, temperature=temperature)
            candidate_responses = sample_data["candidates"]

        self.actor.train()
        G = len(candidate_responses)
        prompt_tokens = self.tokenizer.encode(prompt_text, add_bos=True)
        prompt_len = len(prompt_tokens)

        # 1. 探针 1：规则判卷引擎针对这组候选逐一打分
        reward_details = []
        raw_rewards = []
        for resp in candidate_responses:
            eval_res = self.reward_engine.compute_rule_reward(resp, ground_truth)
            reward_details.append(eval_res)
            raw_rewards.append(eval_res["total_reward"])

        reward_tensor = torch.tensor(raw_rewards, dtype=torch.float32)

        # 2. 探针 2：组内归一化 (Group Normalization) 算 Advantage —— "全靠同行衬托"
        group_mean = reward_tensor.mean()
        group_std = reward_tensor.std(unbiased=False)
        advantages = (reward_tensor - group_mean) / (group_std + 1e-8)

        # 预先编码并提取旧策略与参考模型的序列对数似然
        cand_data = []
        with torch.no_grad():
            for i, resp in enumerate(candidate_responses):
                cand_tokens = self.tokenizer.encode(resp, add_bos=False)
                full_tokens = prompt_tokens + cand_tokens
                full_ids = torch.tensor([full_tokens], dtype=torch.long)
                seq_len = len(full_tokens)

                old_log_probs, _ = self.old_actor.evaluate_actions(full_ids)
                ref_log_probs, _ = self.ref_model.evaluate_actions(full_ids)

                old_seq_log = old_log_probs[0, prompt_len-1 : seq_len-1].mean().item()
                ref_seq_log = ref_log_probs[0, prompt_len-1 : seq_len-1].mean().item()

                cand_data.append({
                    "full_ids": full_ids,
                    "prompt_len": prompt_len,
                    "seq_len": seq_len,
                    "old_seq_log": old_seq_log,
                    "ref_seq_log": ref_seq_log,
                    "adv": advantages[i].item()
                })

        # 3. 执行真实微调更新迭代 (让模型真正依据 Advantage 产生概率迁移)
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0.0
            self.optimizer.zero_grad()

            for item in cand_data:
                full_ids = item["full_ids"]
                p_len = item["prompt_len"]
                s_len = item["seq_len"]
                adv_val = item["adv"]

                cur_log_probs, _ = self.actor.evaluate_actions(full_ids)
                cur_seq_log = cur_log_probs[0, p_len-1 : s_len-1].mean()

                ratio = torch.exp(cur_seq_log - item["old_seq_log"])
                surr1 = ratio * adv_val
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_val
                policy_surrogate = torch.min(surr1, surr2)

                # DeepSeek-V3.2 无偏 KL (k3 估计器)
                if self.use_unbiased_kl:
                    pi_ratio = ratio
                    ref_ratio = torch.exp(torch.tensor(item["ref_seq_log"]) - cur_seq_log)
                    kl_term = pi_ratio * (ref_ratio - torch.log(ref_ratio + 1e-8) - 1.0)
                else:
                    kl_term = 0.5 * (cur_seq_log - item["ref_seq_log"]) ** 2

                # Off-Policy 序列掩码 M
                mask = 1.0
                deviation = abs(item["old_seq_log"] - cur_seq_log.item())
                if self.use_off_policy_mask and (adv_val < -0.5) and (deviation > self.off_policy_delta):
                    mask = 0.0

                cand_loss = -(policy_surrogate - self.beta_kl * kl_term) * mask
                total_loss = total_loss + cand_loss

            mean_loss = total_loss / G
            mean_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()
            losses.append(mean_loss.item())

        # 4. 更新后收集最新状态，真实呈现各候选的独立比率与无偏 KL
        candidate_traces = []
        with torch.no_grad():
            for i, (resp, item) in enumerate(zip(candidate_responses, cand_data)):
                full_ids = item["full_ids"]
                p_len = item["prompt_len"]
                s_len = item["seq_len"]

                final_log_probs, _ = self.actor.evaluate_actions(full_ids)
                final_seq_log = final_log_probs[0, p_len-1 : s_len-1].mean().item()

                final_ratio = torch.exp(torch.tensor(final_seq_log - item["old_seq_log"])).item()
                ref_ratio = torch.exp(torch.tensor(item["ref_seq_log"] - final_seq_log)).item()
                final_kl = max(0.0, final_ratio * (ref_ratio - torch.log(torch.tensor(ref_ratio + 1e-8)).item() - 1.0))

                mask = 1.0
                deviation = abs(item["old_seq_log"] - final_seq_log)
                if self.use_off_policy_mask and (item["adv"] < -0.5) and (deviation > self.off_policy_delta):
                    mask = 0.0

                candidate_traces.append({
                    "candidate_index": i + 1,
                    "response": resp,
                    "raw_reward": reward_tensor[i].item(),
                    "accuracy": reward_details[i]["accuracy_reward"],
                    "format": reward_details[i]["format_reward"],
                    "advantage": item["adv"],
                    "ratio": final_ratio,
                    "kl_penalty": final_kl,
                    "off_policy_mask": mask
                })

        # 迭代后刷新旧策略快照
        self.old_actor = copy.deepcopy(self.actor)
        for p in self.old_actor.parameters():
            p.requires_grad = False

        return {
            "algorithm": "GRPO",
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "group_size": G,
            "group_mean": group_mean.item(),
            "group_std": group_std.item(),
            "mean_loss": losses[-1],
            "unbiased_kl_enabled": self.use_unbiased_kl,
            "off_policy_mask_enabled": self.use_off_policy_mask,
            "traces": candidate_traces
        }

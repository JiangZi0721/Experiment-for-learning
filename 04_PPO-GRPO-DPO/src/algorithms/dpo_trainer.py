import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from ..config import cfg
from ..models.policy_network import WhiteBoxPolicyNetwork, ToyTokenizer

class WhiteBoxDPOTrainer:
    """
    白盒可透视 DPO 训练器 (Direct Preference Optimization)
    无 Critic、无显式 Reward Model，仅需 Actor 与 Reference Model
    全真模拟偏好对梯度优化全过程，真实透视隐式奖励拔河与胜率从 50% 到 80%+ 的跃迁！
    """
    def __init__(
        self,
        actor: WhiteBoxPolicyNetwork,
        tokenizer: ToyTokenizer,
        beta: float = cfg.DPO_BETA,
        lr: float = cfg.DPO_LR
    ):
        self.actor = actor
        self.tokenizer = tokenizer
        self.ref_model = copy.deepcopy(actor)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)

    def compute_sequence_log_probs(
        self,
        model: nn.Module,
        prompt_tokens: list,
        answer_tokens: list
    ) -> torch.Tensor:
        """
        计算给定上下文条件下，答案序列各 Token 的对数似然之和 (Sum of Token Log-Probs)
        """
        full_tokens = prompt_tokens + answer_tokens
        full_ids = torch.tensor([full_tokens], dtype=torch.long)
        prompt_len = len(prompt_tokens)

        log_probs, _ = model.evaluate_actions(full_ids) # [1, P + A - 1]
        answer_log_probs = log_probs[:, prompt_len - 1 :]
        return answer_log_probs.sum(dim=-1) # 标量对数概率

    def evaluate_preference(
        self,
        prompt_text: str,
        chosen_text: str,
        rejected_text: str,
        beta: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        评估当前模型对偏好对的初始倾向 (不执行梯度反传)
        """
        if beta is None:
            beta = self.beta

        prompt_toks = self.tokenizer.encode(prompt_text, add_bos=True)
        chosen_toks = self.tokenizer.encode(chosen_text, add_bos=False)
        rejected_toks = self.tokenizer.encode(rejected_text, add_bos=False)

        with torch.no_grad():
            pi_chosen = self.compute_sequence_log_probs(self.actor, prompt_toks, chosen_toks)
            pi_rejected = self.compute_sequence_log_probs(self.actor, prompt_toks, rejected_toks)
            ref_chosen = self.compute_sequence_log_probs(self.ref_model, prompt_toks, chosen_toks)
            ref_rejected = self.compute_sequence_log_probs(self.ref_model, prompt_toks, rejected_toks)

            diff_chosen = (pi_chosen - ref_chosen).item()
            diff_rejected = (pi_rejected - ref_rejected).item()
            margin = beta * (diff_chosen - diff_rejected)
            win_rate = torch.sigmoid(torch.tensor(margin)).item()
            loss = -F.logsigmoid(torch.tensor(margin)).item()

        return {
            "pi_chosen": pi_chosen.item(),
            "pi_rejected": pi_rejected.item(),
            "ref_chosen": ref_chosen.item(),
            "ref_rejected": ref_rejected.item(),
            "diff_chosen": diff_chosen,
            "diff_rejected": diff_rejected,
            "reward_chosen": beta * diff_chosen,
            "reward_rejected": beta * diff_rejected,
            "margin": margin,
            "win_rate": win_rate,
            "loss": loss,
            "beta": beta
        }

    def train_preference_step(
        self,
        prompt_text: str,
        chosen_text: str,
        rejected_text: str,
        beta: Optional[float] = None,
        lr: Optional[float] = None,
        num_epochs: int = 6
    ) -> Dict[str, Any]:
        """
        全动态 DPO 偏好对训练：
        记录【更新前 (Step 0)】与【多次梯度更新后 (Step N)】的真实状态对比，
        严谨杜绝“未更新即硬说拉升”的伪逻辑！
        """
        if beta is None:
            beta = self.beta

        if lr is not None:
            optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        else:
            optimizer = self.optimizer

        prompt_toks = self.tokenizer.encode(prompt_text, add_bos=True)
        chosen_toks = self.tokenizer.encode(chosen_text, add_bos=False)
        rejected_toks = self.tokenizer.encode(rejected_text, add_bos=False)

        # 1. 探针 1：更新前的初始状态 (Before Training)
        init_state = self.evaluate_preference(prompt_text, chosen_text, rejected_text, beta=beta)

        with torch.no_grad():
            ref_chosen = self.compute_sequence_log_probs(self.ref_model, prompt_toks, chosen_toks)
            ref_rejected = self.compute_sequence_log_probs(self.ref_model, prompt_toks, rejected_toks)

        # 2. 执行真实的梯度优化迭代（让模型真正学会偏好！）
        self.actor.train()
        losses = []
        margins = []
        win_rates = []

        for epoch in range(num_epochs):
            pi_chosen = self.compute_sequence_log_probs(self.actor, prompt_toks, chosen_toks)
            pi_rejected = self.compute_sequence_log_probs(self.actor, prompt_toks, rejected_toks)

            log_ratio_w = pi_chosen - ref_chosen
            log_ratio_l = pi_rejected - ref_rejected

            # 隐式奖励拔河分差
            reward_margin = beta * (log_ratio_w - log_ratio_l)
            loss = -F.logsigmoid(reward_margin).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            optimizer.step()

            cur_margin = reward_margin.item()
            losses.append(loss.item())
            margins.append(cur_margin)
            win_rates.append(torch.sigmoid(torch.tensor(cur_margin)).item())

        # 3. 探针 2：更新后的最新状态 (After Training)
        final_state = self.evaluate_preference(prompt_text, chosen_text, rejected_text, beta=beta)
        r_chosen = beta * final_state["diff_chosen"]
        r_rejected = beta * final_state["diff_rejected"]

        return {
            "algorithm": "DPO",
            "prompt": prompt_text,
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
            "beta": beta,
            # 初始对齐前数值
            "initial": init_state,
            # 真实训练后数值
            "trained": {
                "diff_chosen": final_state["diff_chosen"],
                "diff_rejected": final_state["diff_rejected"],
                "reward_chosen": r_chosen,
                "reward_rejected": r_rejected,
                "margin": final_state["margin"],
                "win_rate": final_state["win_rate"],
                "loss": final_state["loss"]
            },
            "loss_trajectory": losses,
            "margin_trajectory": margins,
            "win_rate_trajectory": win_rates
        }

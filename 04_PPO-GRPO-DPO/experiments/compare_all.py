import time
import copy
import torch
from typing import Dict, Any, List

from src.config import cfg
from src.models.policy_network import WhiteBoxPolicyNetwork, ToyTokenizer, load_base_policy_and_tokenizer
from src.models.critic_network import WhiteBoxCriticNetwork
from src.models.reward_engine import HybridRewardEngine
from src.algorithms.ppo_trainer import WhiteBoxPPOTrainer
from src.algorithms.dpo_trainer import WhiteBoxDPOTrainer
from src.algorithms.grpo_trainer import WhiteBoxGRPOTrainer
from src.metrics.rl_metrics import RLEvaluationMetrics
from src.visualizer import WhiteBoxRLVisualizer

def run_cross_algorithm_comparison():
    """
    三大算法全维度横向对照实验：
    100% 取自实测运行输出，动态统计各算法实际耗时、Loss、KL、策略熵与奖励表现！
    """
    visualizer = WhiteBoxRLVisualizer()
    visualizer.console.print("\n[bold cyan]🧪 正在启动三大算法横向对照实验 (Benchmarking PPO vs DPO vs GRPO)...[/bold cyan]\n")

    base_actor, tokenizer = load_base_policy_and_tokenizer()
    reward_engine = HybridRewardEngine()

    prompt = "请帮我给重要客户写一封邮件，礼貌地解释发货延迟的原因。"
    response = "尊敬的客户：非常抱歉发货发生轻微延迟，我们已为您申请了无门槛优惠券并加急派送！"

    results = []

    # 1. 真实运行 PPO
    ppo_actor = copy.deepcopy(base_actor)
    critic = WhiteBoxCriticNetwork(vocab_size=tokenizer.vocab_size)
    ppo = WhiteBoxPPOTrainer(ppo_actor, critic, reward_engine, tokenizer)

    t0 = time.perf_counter()
    ppo_res = ppo.train_step(prompt_text=prompt, response_text=response, use_clip=True, beta_kl=cfg.PPO_BETA, num_epochs=3)
    t_ppo = (time.perf_counter() - t0) * 1000

    ppo_diag = RLEvaluationMetrics.evaluate_training_health(
        mean_kl=ppo_res["mean_kl"],
        clip_fraction=ppo_res["clip_fraction"],
        entropy=ppo_res["entropy"],
        avg_length=len(response)
    )
    results.append({
        "algorithm": "PPO",
        "policy_loss": ppo_res["policy_loss"],
        "reward": ppo_res["final_reward"],
        "kl": ppo_res["mean_kl"],
        "entropy": ppo_res["entropy"],
        "clip_rate": ppo_res["clip_fraction"],
        "step_time_ms": t_ppo,
        "health_status": ppo_diag["health_status"],
        "warnings": ppo_diag["diagnostic_warnings"]
    })

    # 2. 真实运行 DPO (包含真实梯度迭代)
    dpo_actor = copy.deepcopy(base_actor)
    dpo = WhiteBoxDPOTrainer(dpo_actor, tokenizer)
    chosen_text = "使用哈希表可以在 O(n) 时间内完成，边遍历边记录 target - num 是否在哈希表中。"
    rejected_text = "直接写双重 for 循环暴力暴力破解，时间复杂度 O(n^2) 慢是慢了点但能用。"

    t0 = time.perf_counter()
    dpo_res = dpo.train_preference_step(
        prompt_text="如何用 Python 实现两数之和（Two Sum）的最优解？",
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        beta=cfg.DPO_BETA,
        num_epochs=6
    )
    t_dpo = (time.perf_counter() - t0) * 1000

    dpo_trained = dpo_res["trained"]
    chosen_toks = tokenizer.encode(chosen_text, add_bos=False)
    chosen_len = max(len(chosen_toks), 1)
    dpo_kl = 0.5 * ((dpo_trained["diff_chosen"] / chosen_len) ** 2)

    # 计算 DPO 模型当前实际动作熵
    with torch.no_grad():
        test_ids = torch.tensor([tokenizer.encode(chosen_text, add_bos=True)], dtype=torch.long)
        _, dpo_entropy = dpo.actor.evaluate_actions(test_ids)
        real_dpo_entropy = dpo_entropy.item()

    dpo_diag = RLEvaluationMetrics.evaluate_training_health(
        mean_kl=dpo_kl,
        clip_fraction=0.0,
        entropy=real_dpo_entropy,
        avg_length=len(chosen_text)
    )
    results.append({
        "algorithm": "DPO",
        "policy_loss": dpo_trained["loss"],
        "reward": dpo_trained["margin"],
        "kl": dpo_kl,
        "entropy": real_dpo_entropy,
        "clip_rate": 0.0,
        "step_time_ms": t_dpo,
        "health_status": dpo_diag["health_status"],
        "warnings": dpo_diag["diagnostic_warnings"]
    })

    # 3. 真实运行 GRPO
    grpo_actor = copy.deepcopy(base_actor)
    grpo = WhiteBoxGRPOTrainer(grpo_actor, reward_engine, tokenizer, group_size=4)
    candidates = [
        "<think>15个苹果分1/3是5个，剩下10个；再吃2个，10-2=8个</think><answer>8</answer>",
        "<think>15乘以三分之一是5个，所以还剩5个</think><answer>5</answer>",
        "答案大概是 8 个吧，我猜的。",
        "<think>完全不知道怎么算，乱填</think><answer>99</answer>"
    ]

    t0 = time.perf_counter()
    grpo_res = grpo.train_group_step(
        prompt_text="小明有 15 个苹果，他把其中的 1/3 给了小红，又吃了剩下的 2 个，现在小明还剩多少个苹果？",
        ground_truth="8",
        candidate_responses=candidates
    )
    t_grpo = (time.perf_counter() - t0) * 1000

    grpo_mean_kl = float(sum(t["kl_penalty"] for t in grpo_res["traces"]) / len(candidates))
    grpo_clip_frac = float(sum(1 for t in grpo_res["traces"] if abs(t["ratio"] - 1.0) > cfg.GRPO_EPSILON) / len(candidates))

    with torch.no_grad():
        test_ids = torch.tensor([tokenizer.encode(candidates[0], add_bos=True)], dtype=torch.long)
        _, grpo_entropy = grpo.actor.evaluate_actions(test_ids)
        real_grpo_entropy = grpo_entropy.item()

    grpo_diag = RLEvaluationMetrics.evaluate_training_health(
        mean_kl=grpo_mean_kl,
        clip_fraction=grpo_clip_frac,
        entropy=real_grpo_entropy,
        avg_length=len(candidates[0])
    )
    results.append({
        "algorithm": "GRPO",
        "policy_loss": grpo_res["mean_loss"],
        "reward": grpo_res["group_mean"],
        "kl": grpo_mean_kl,
        "entropy": real_grpo_entropy,
        "clip_rate": grpo_clip_frac,
        "step_time_ms": t_grpo,
        "health_status": grpo_diag["health_status"],
        "warnings": grpo_diag["diagnostic_warnings"]
    })

    # 打印全动态指标面板与基于实测结果的对比解读
    visualizer.show_evaluation_metrics_dashboard(results)
    empirical_summary = {
        "t_ppo": t_ppo,
        "t_dpo": t_dpo,
        "t_grpo": t_grpo
    }
    visualizer.show_algorithm_comparison_table(empirical_summary)

if __name__ == "__main__":
    run_cross_algorithm_comparison()

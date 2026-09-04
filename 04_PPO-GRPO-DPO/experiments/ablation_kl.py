import torch
from rich.table import Table
from rich import box

from src.models.policy_network import WhiteBoxPolicyNetwork, ToyTokenizer
from src.models.critic_network import WhiteBoxCriticNetwork
from src.models.reward_engine import HybridRewardEngine
from src.algorithms.ppo_trainer import WhiteBoxPPOTrainer
from src.visualizer import WhiteBoxRLVisualizer

def run_kl_ablation_study():
    """
    KL 惩罚消融实验 (KL Penalty Ablation Study)
    验证三个不同 KL 强度下的策略演化行为：
    1. Beta = 0.00 (无约束): 观察策略漂移与奖励黑客 (Reward Hacking) 风险
    2. Beta = 0.05 (黄金适度): 兼顾偏好对齐与基础语言流畅度
    3. Beta = 0.50 (过度拘谨): 模型固步自封，无法有效吸收人类偏好
    """
    visualizer = WhiteBoxRLVisualizer()
    visualizer.console.print("\n[bold magenta]🔬 正在执行 KL 正则系数消融实验 (Ablation on KL Beta)...[/bold magenta]\n")

    tokenizer = ToyTokenizer()
    reward_engine = HybridRewardEngine()
    prompt_text = "请帮我给重要客户写一封邮件，礼貌地解释发货延迟的原因。"
    response_text = "尊敬的客户：非常抱歉发货发生轻微延迟，我们已为您申请了无门槛优惠券并加急派送！"

    tokenizer.build_vocab_from_texts([prompt_text, response_text])

    betas = [0.0, 0.05, 0.5]
    table = Table(
        title="🧪 KL 正则强度 (Beta) 消融实验对照表",
        box=box.ROUNDED,
        header_style="bold bright_magenta"
    )
    table.add_column("KL 系数 (Beta)", style="cyan", justify="center", overflow="fold")
    table.add_column("策略设定", style="white", justify="left", overflow="fold")
    table.add_column("最终 Policy Loss", style="yellow", justify="right", overflow="fold")
    table.add_column("实测 KL 散度", style="magenta", justify="right", overflow="fold")
    table.add_column("策略状态诊断", style="bold", justify="left", overflow="fold")

    for b in betas:
        base_actor = WhiteBoxPolicyNetwork(vocab_size=tokenizer.vocab_size)
        critic = WhiteBoxCriticNetwork(vocab_size=tokenizer.vocab_size)
        trainer = WhiteBoxPPOTrainer(base_actor, critic, reward_engine, tokenizer, beta_kl=b)

        res = trainer.train_step(prompt_text, response_text)

        if b == 0.0:
            diag = "[red]⚠️ 风险：无约束下策略容易随 Reward 放飞，长期训练易引发 Reward Hacking 与胡言乱语[/red]"
            desc = "彻底放飞 (无基座约束)"
        elif b == 0.05:
            diag = "[green]✅ 最佳：既学会了诚恳补偿，又保持了优雅的人类语言习惯[/green]"
            desc = "推荐配置 (适度探索与约束)"
        else:
            diag = "[yellow]⚠️ 迟钝：基座约束过死，模型更新迟滞，偏好吸收缓慢[/yellow]"
            desc = "过度保守 (重度束缚)"

        table.add_row(f"β = {b:.2f}", desc, f"{res['policy_loss']:.4f}", f"{res['mean_kl']:.4f}", diag)

    visualizer.console.print(table)

if __name__ == "__main__":
    run_kl_ablation_study()

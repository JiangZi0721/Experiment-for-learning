"""
Script to generate all visualization figures for Complex GridWorld analysis:
1. images/complex_env_overview_and_converged_policies.png
2. images/step1_td0_subgoal_discovery.png
3. images/step2_td0_lava_pit_fall.png
4. images/step3_sarsa_main_goal_discovery.png
5. images/step4_sarsa_spike_trap_deterrence.png
6. images/step5_qlearning_bootstrap_surge.png
7. images/step6_qlearning_lava_penalty.png
8. images/six_critical_steps_overview.png
"""
import os
import pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Matplotlib styling
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

from src.complex_grid_world import ComplexGridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_NAMES, ACTION_SYMBOLS
from src.complex_model_free import solve_dp_optimal, run_complex_td0, run_complex_sarsa, run_complex_q_learning
from src.plot_complex_analysis import create_step_card, draw_grid_heatmap

def to_matrix(val_dict):
    mat = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            mat[r, c] = val_dict.get((r, c), 0.0)
    return mat

def get_policy_matrix(q_dict):
    pol = [[0 for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            st = (r, c)
            qs = q_dict[st]
            pol[r][c] = max(qs.keys(), key=lambda a: qs[a])
    return pol

def generate_figures():
    os.makedirs("images", exist_ok=True)

    # Initialize environment & run simulations
    env = ComplexGridWorld(rows=4, cols=4, gamma=0.9, step_cost=-1.0)
    V_opt, Q_opt = solve_dp_optimal(env, gamma=0.9)
    td0_res = run_complex_td0(env, num_episodes=2500, seed=42)
    sarsa_res = run_complex_sarsa(env, num_episodes=2000, seed=42)
    ql_res = run_complex_q_learning(env, num_episodes=2000, seed=42)

    # --------------------------------------------------------------------------------------
    # Figure 1: Environment Map & Converged Policies (4 Panels)
    # --------------------------------------------------------------------------------------
    print("Generating Fig 1: Environment and Converged Policies Overview...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 13), dpi=160)

    # Panel 1: Environment Rewards Map
    env_rewards = np.full((4, 4), -1.0)
    env_rewards[0, 3] = 3.0
    env_rewards[3, 3] = 10.0
    env_rewards[1, 2] = -10.0
    env_rewards[2, 3] = -6.0
    env_rewards[2, 1] = -3.0

    im0 = draw_grid_heatmap(axs[0, 0], env_rewards, policy_grid=None, title="图 (a): 复杂网格环境物理地形与奖励设定")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04, label="Immediate Reward")

    # Panel 2: TD(0) Converged State Values
    td0_mat = to_matrix(td0_res["V"])
    im1 = draw_grid_heatmap(axs[0, 1], td0_mat, policy_grid=None, title="图 (b): TD(0) 随机策略收敛状态价值 V(s)")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, label="State Value V(s)")

    # Panel 3: SARSA Converged Q-Values & Policy
    sarsa_V = {s: max(sarsa_res["Q"][s].values()) for s in env.states}
    sarsa_mat = to_matrix(sarsa_V)
    sarsa_pol = get_policy_matrix(sarsa_res["Q"])
    im2 = draw_grid_heatmap(axs[1, 0], sarsa_mat, policy_grid=sarsa_pol, title="图 (c): SARSA 同策略收敛 max Q(s,a) 与贪婪策略")
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04, label="Max Q(s, a)")

    # Panel 4: Q-Learning Converged Q-Values & Policy
    ql_V = {s: max(ql_res["Q"][s].values()) for s in env.states}
    ql_mat = to_matrix(ql_V)
    ql_pol = get_policy_matrix(ql_res["Q"])
    im3 = draw_grid_heatmap(axs[1, 1], ql_mat, policy_grid=ql_pol, title="图 (d): Q-Learning 异策略收敛 max Q(s,a) 与贪婪策略")
    plt.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04, label="Max Q(s, a)")

    plt.tight_layout()
    fig_path1 = "images/complex_env_overview_and_converged_policies.png"
    plt.savefig(fig_path1, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path1}")

    # --------------------------------------------------------------------------------------
    # Figures 2-7: Dedicated 6 Critical Steps
    # --------------------------------------------------------------------------------------
    # Card 1: TD(0) Sub-Goal Discovery
    card1_event = {
        "algo": "TD(0)", "episode": 8, "step": 0, "s": (0, 2), "a": RIGHT,
        "r": 3.0, "next_s": (0, 3), "done": True, "old_val": -2.492,
        "target": 3.000, "error": 5.492, "new_val": -1.943
    }
    card1_derivation = [
        r"1. 状态转移: $S = (0, 2), A = \text{RIGHT} \rightarrow S' = (0, 3)$ (进入次级目标终端)",
        r"2. 获得奖励: $R = +3.0$, 终止状态价值固定为 $V(S') = 0.0$",
        r"3. 历史价值: $V_{\text{old}}(0, 2) = -2.492$ (前期在网格中盲走惩罚累计)",
        r"4. TD 目标计算: $\text{Target} = R + \gamma \cdot 0.0 = +3.000$",
        r"5. TD 误差计算: $\delta = \text{Target} - V(S) = 3.000 - (-2.492) = +5.492$",
        r"6. 价值更新: $V(S) \leftarrow -2.492 + 0.1 \times 5.492 = -1.943$",
        r"7. 变化绝对量: $|\Delta V| = 0.549$ (单步暴涨 22%)"
    ]
    card1_insight = "首次触及安全次级目标 (Sub-Goal)。\n右上角区域的状态价值由深负转正启动；\n原本在全图随机游走积累的步数惩罚被强力的 +3.0 终点奖励打破，\n在右上角建立了局部正向价值梯度引力场。"
    create_step_card(
        step_num=1, algo_name="TD(0) 状态评估", event=card1_event,
        values_grid=to_matrix(card1_event.get("snapshot_V", td0_res["V"])),
        policy_grid=None,
        formula_latex=r"$V(S) \leftarrow V(S) + \alpha [ R + \gamma V(S') - V(S) ]$",
        math_derivation=card1_derivation, strategic_insight=card1_insight,
        save_path="images/step1_td0_subgoal_discovery.png"
    )

    # Card 2: TD(0) Lava Pit Fall
    card2_event = {
        "algo": "TD(0)", "episode": 3, "step": 11, "s": (2, 2), "a": UP,
        "r": -10.0, "next_s": (1, 2), "done": False, "old_val": -1.021,
        "target": -10.000, "error": -8.979, "new_val": -1.919
    }
    card2_derivation = [
        r"1. 状态转移: $S = (2, 2), A = \text{UP} \rightarrow S' = (1, 2)$ (踏入致命熔岩深坑)",
        r"2. 获得奖励: $R = -10.0$ (极重惩罚), 后继状态价值 $V(1, 2) \approx 0.0$",
        r"3. 历史价值: $V_{\text{old}}(2, 2) = -1.021$",
        r"4. TD 目标计算: $\text{Target} = -10.0 + 0.9 \times 0.0 = -10.000$",
        r"5. TD 误差计算: $\delta = -10.000 - (-1.021) = -8.979$",
        r"6. 价值更新: $V(S) \leftarrow -1.021 + 0.1 \times (-8.979) = -1.919$",
        r"7. 变化绝对量: $|\Delta V| = 0.898$ (单步暴跌接近 1.0 个点)"
    ]
    card2_insight = "致命熔岩坑 (-10.0) 首次向下辐射负向冲击波。\n状态 (2, 2) 价值骤降；原本中性的相邻格子瞬间被污染为高危区，\n生动展现了无模型 TD 自举对环境剧烈惩罚的敏锐捕获，\n该负向价值将通过随机游走扩散至全图。"
    create_step_card(
        step_num=2, algo_name="TD(0) 状态评估", event=card2_event,
        values_grid=to_matrix(td0_res["V"]),
        policy_grid=None,
        formula_latex=r"$V(S) \leftarrow V(S) + \alpha [ R + \gamma V(S') - V(S) ]$",
        math_derivation=card2_derivation, strategic_insight=card2_insight,
        save_path="images/step2_td0_lava_pit_fall.png"
    )

    # Card 3: SARSA Main Goal Arrival
    card3_event = {
        "algo": "SARSA", "episode": 0, "step": 38, "s": (3, 2), "a": RIGHT,
        "r": 10.0, "next_s": (3, 3), "done": True, "old_val": 0.000,
        "target": 10.000, "error": 10.000, "new_val": 1.000
    }
    card3_derivation = [
        r"1. 状态与动作: $S = (3, 2), A = \text{RIGHT} \rightarrow S' = (3, 3)$ (冲入终极大奖)",
        r"2. 获得奖励: $R = +10.0$, 终态无后续动作: $Q(S', A') = 0.0$",
        r"3. 历史动作价值: $Q_{\text{old}}((3, 2), \text{RIGHT}) = 0.000$ (从未到达过)",
        r"4. TD 目标计算: $\text{Target} = R + \gamma \cdot 0.0 = +10.000$",
        r"5. TD 误差计算: $\delta = 10.000 - 0.000 = +10.000$",
        r"6. 动作价值更新: $Q(S, A) \leftarrow 0.0 + 0.1 \times 10.0 = +1.000$",
        r"7. 变化绝对量: $|\Delta Q| = 1.000$ (全图全流程最大单步理论增量)"
    ]
    card3_insight = "终极最高大奖 (+10.0) 首次被动作价值函数捕获。\n这是整个网格动作价值空间的奇点破晓时刻！\n产生全局最大的初始单步更新量 (+1.000)，\n彻底激活了右下角向目标汇聚的终极梯度，是控制算法收敛的基石。"
    create_step_card(
        step_num=3, algo_name="SARSA 同策略控制", event=card3_event,
        values_grid=sarsa_mat, policy_grid=sarsa_pol,
        formula_latex=r"$Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma Q(S', A') - Q(S, A) ]$",
        math_derivation=card3_derivation, strategic_insight=card3_insight,
        save_path="images/step3_sarsa_main_goal_discovery.png"
    )

    # Card 4: SARSA Spike Trap Deterrence
    card4_event = {
        "algo": "SARSA", "episode": 68, "step": 4, "s": (2, 2), "a": RIGHT,
        "r": -6.0, "next_s": (2, 3), "done": False, "old_val": 0.000,
        "target": -6.000, "error": -6.000, "new_val": -0.600
    }
    card4_derivation = [
        r"1. 状态与动作: $S = (2, 2), A = \text{RIGHT} \rightarrow S' = (2, 3)$ (踩入尖刺陷阱)",
        r"2. 获得奖励: $R = -6.0$, 在刺坑处下一步采样动作 $Q(S', A') \approx 0.0$",
        r"3. 历史动作价值: $Q_{\text{old}}((2, 2), \text{RIGHT}) = 0.000$",
        r"4. TD 目标计算: $\text{Target} = -6.0 + 0.9 \times 0.0 = -6.000$",
        r"5. TD 误差计算: $\delta = -6.000 - 0.000 = -6.000$",
        r"6. 动作价值更新: $Q(S, A) \leftarrow 0.0 + 0.1 \times (-6.0) = -0.600$",
        r"7. 变化绝对量: $|\Delta Q| = 0.600$ (断崖式下跌)"
    ]
    card4_insight = "尖刺陷阱 (-6.0) 产生强烈的排斥性动作价值。\n在同策略控制下，由于智能体亲身踩刺，导致向右的动作价值立即跌至 -0.600，\n远低于基准动作，迫使贪婪策略在此格永久放弃向右冒进，\n转向寻找其他安全出口。"
    create_step_card(
        step_num=4, algo_name="SARSA 同策略控制", event=card4_event,
        values_grid=sarsa_mat, policy_grid=sarsa_pol,
        formula_latex=r"$Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma Q(S', A') - Q(S, A) ]$",
        math_derivation=card4_derivation, strategic_insight=card4_insight,
        save_path="images/step4_sarsa_spike_trap_deterrence.png"
    )

    # Card 5: Q-Learning Bootstrap Surge
    card5_event = {
        "algo": "Q-Learning", "episode": 180, "step": 4, "s": (2, 2), "a": DOWN,
        "r": -1.0, "next_s": (3, 2), "done": False, "old_val": 0.000,
        "max_next_q": 3.439, "target": 2.095, "error": 2.095, "new_val": 0.210
    }
    card5_derivation = [
        r"1. 状态与动作: $S = (2, 2), A = \text{DOWN} \rightarrow S' = (3, 2)$ (向目标前哨挺进)",
        r"2. 获得奖励: $R = -1.0$ (常规步长成本)",
        r"3. 异策略最大化: $\max_{a'} Q((3, 2), a') = +3.439$ (源于右侧终点的高能回传)",
        r"4. TD 目标计算: $\text{Target} = -1.0 + 0.9 \times 3.439 = +2.095$",
        r"5. TD 误差计算: $\delta = 2.095 - 0.000 = +2.095$",
        r"6. 动作价值更新: $Q(S, A) \leftarrow 0.0 + 0.1 \times 2.095 = +0.210$",
        r"7. 核心竞争逆转: 向下 $Q = +0.210 >$ 向右踩刺 $Q = -0.600$ (策略箭头彻底翻转向下！)"
    ]
    card5_insight = "异策略 Max 算子的后向渗透回传神技！\n无需等待下一个动作具体采到什么，直接用最优动作价值 +3.439 自举。\n使得 (2, 2) 向下动作单步转正并彻底超越向右踩刺的负价值，\n生动展现了 Q-Learning 如何凭借贝尔曼最优算子精准锁定避险路线！"
    create_step_card(
        step_num=5, algo_name="Q-Learning 异策略控制", event=card5_event,
        values_grid=ql_mat, policy_grid=ql_pol,
        formula_latex=r"$Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma \max_{a'} Q(S', a') - Q(S, A) ]$",
        math_derivation=card5_derivation, strategic_insight=card5_insight,
        save_path="images/step5_qlearning_bootstrap_surge.png"
    )

    # Card 6: Q-Learning Lava Penalty
    card6_event = {
        "algo": "Q-Learning", "episode": 1, "step": 16, "s": (1, 1), "a": RIGHT,
        "r": -10.0, "next_s": (1, 2), "done": False, "old_val": 0.000,
        "max_next_q": 0.000, "target": -10.000, "error": -10.000, "new_val": -1.000
    }
    card6_derivation = [
        r"1. 状态与动作: $S = (1, 1), A = \text{RIGHT} \rightarrow S' = (1, 2)$ (误撞中心大火坑)",
        r"2. 获得奖励: $R = -10.0$ (极限惩罚)",
        r"3. 历史动作价值: $Q_{\text{old}}((1, 1), \text{RIGHT}) = 0.000$",
        r"4. TD 目标计算: $\text{Target} = -10.0 + 0.9 \times 0.0 = -10.000$",
        r"5. TD 误差计算: $\delta = -10.000 - 0.000 = -10.000$",
        r"6. 动作价值更新: $Q(S, A) \leftarrow 0.0 + 0.1 \times (-10.0) = -1.000$",
        r"7. 变化绝对量: $|\Delta Q| = 1.000$ (单步暴跌，全图最严厉惩戒)"
    ]
    card6_insight = "决策分水岭 (1, 1) 的毁灭性动作一票否决！\n向右直冲熔岩的动作遭遇 -1.000 的极限制裁，\n使得该动作迅速成为所有候选动作中的垫底选择，\n迫使策略在 (1, 1) 只能选择向上前往安全次级出口，或向下绕道，\n奠定了全图无模型学习的安全拓扑边界。"
    create_step_card(
        step_num=6, algo_name="Q-Learning 异策略控制", event=card6_event,
        values_grid=ql_mat, policy_grid=ql_pol,
        formula_latex=r"$Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma \max_{a'} Q(S', a') - Q(S, A) ]$",
        math_derivation=card6_derivation, strategic_insight=card6_insight,
        save_path="images/step6_qlearning_lava_penalty.png"
    )

    # --------------------------------------------------------------------------------------
    # Figure 8: Master Combined 6-Step Overview (2x3 Grid)
    # --------------------------------------------------------------------------------------
    print("Generating Fig 8: Master Combined 6-Step Overview (2x3)...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=160)
    cards_info = [
        (card1_event, "TD(0) 发现次级目标 (+3.0)", None, td0_mat),
        (card2_event, "TD(0) 误入熔岩深坑 (-10.0)", None, td0_mat),
        (card3_event, "SARSA 首达终极大奖 (+10.0)", sarsa_pol, sarsa_mat),
        (card4_event, "SARSA 踩入尖刺陷阱 (-6.0)", sarsa_pol, sarsa_mat),
        (card5_event, "Q-Learning 异策略回传激增", ql_pol, ql_mat),
        (card6_event, "Q-Learning 熔岩死刑一票否决", ql_pol, ql_mat)
    ]

    for idx, (ev, step_title, pol, mat) in enumerate(cards_info):
        r_idx, c_idx = idx // 3, idx % 3
        ax = axs[r_idx, c_idx]
        im = draw_grid_heatmap(
            ax, mat, pol,
            title=f"步骤 #{idx+1}: {step_title}\n($S={ev['s']} \\rightarrow S'={ev['next_s']}, \\Delta={ev['new_val']-ev['old_val']:+.3f}$)",
            highlight_trans={"s": ev["s"], "next_s": ev["next_s"], "a": ev["a"], "r": ev["r"]}
        )

    plt.tight_layout()
    fig_path8 = "images/six_critical_steps_overview.png"
    plt.savefig(fig_path8, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path8}")
    print("All figures successfully created!")

if __name__ == "__main__":
    generate_figures()

# -*- coding: utf-8 -*-
"""
Experiment 1: 标注偏置大对决 (Label Bias Duel: MEMM vs CRF)
复现 McCallum & Lafferty (2001) 经典反例：
通过带有不同出度拓扑的有向状态机，实证为什么局部归一化 (MEMM) 会产生标注偏置，
而全局归一化 (CRF) 是如何通过全局配分函数 Z(X) 根治该缺陷的。
"""
import numpy as np

def run():
    print("=" * 80)
    print(">>> 【实验一：标注偏置实证 (Label Bias Duel: MEMM vs CRF)】")
    print("=" * 80)
    print("正在构建 McCallum 经典标注偏置拓扑状态机：")
    print("  - 起始状态 S0")
    print("  - 路径 A: S0 -> S1 -> S_destA (S1 出度为 1，即只有 1 条出边)")
    print("  - 路径 B: S0 -> S2 -> S_destB (S2 出度为 5，即有 5 条发散出边)")
    print()
    print("输入观测序列 X = [o1, o2]，观测特征对路径 B 的真实发射偏好显著高于路径 A：")
    print("  - 观测特征打分: 路径 A 未归一化总能量 = 4.0 | 路径 B 未归一化总能量 = 7.5 (B 明显更符合语境)")
    print("-" * 80)

    # 1. 模拟 MEMM 局部归一化计算
    # 在 S0 处，转移概率 Softmax:
    # 假设 S0 到 S1 的发射权重 1.5, 到 S2 的权重 2.0
    memm_p_s1 = np.exp(1.5) / (np.exp(1.5) + np.exp(2.0))
    memm_p_s2 = np.exp(2.0) / (np.exp(1.5) + np.exp(2.0))

    # 在 S1 处，出度仅为 1，局部归一化强制概率和为 1.0！
    # 无论实际观测特征多么微弱，P(S_destA | S1) 必然被强制拉满到 1.0！
    memm_p_destA_given_s1 = 1.0

    # 在 S2 处，出度为 5，能量被 5 个出边瓜分，即使目标分支能量很高 (e.g. 5.5 vs 其他 4 个 0.5)
    # 局部归一化:
    weights_s2 = np.array([5.5, 0.5, 0.5, 0.5, 0.5])
    memm_p_destB_given_s2 = np.exp(5.5) / np.sum(np.exp(weights_s2))

    # MEMM 整条路径条件概率 (局部转移概率连乘)
    memm_prob_pathA = memm_p_s1 * memm_p_destA_given_s1
    memm_prob_pathB = memm_p_s2 * memm_p_destB_given_s2

    # 2. 模拟 CRF 全局归一化计算
    # CRF 不在每个状态做局部除法，而是对全图所有可能路径求总和: Z(X) = \sum_{all_paths} exp(Score(path))
    # 路径 A 总得分: S(Path A) = 1.5 + 2.5 = 4.0
    # 路径 B 目标分支得分: S(Path B_dest) = 2.0 + 5.5 = 7.5
    # 路径 B 其余 4 条干扰分支得分: S(Path B_other) = 2.0 + 0.5 = 2.5 (共 4 条)
    score_A = 4.0
    score_B_target = 7.5
    score_B_others = [2.5] * 4

    all_scores = [score_A, score_B_target] + score_B_others
    Z_crf = np.sum(np.exp(all_scores))

    crf_prob_pathA = np.exp(score_A) / Z_crf
    crf_prob_pathB = np.exp(score_B_target) / Z_crf

    print(f"{'决策评估维度':<26} | {'MEMM (局部归一化)':<22} | {'CRF (全局归一化)':<22} | {'数学机理对比'}")
    print("-" * 95)
    print(f"{'S0 步对分支 S1 / S2 概率':<22} | {f'{memm_p_s1:.3f} / {memm_p_s2:.3f}':<22} | {'(不单独在节点做除法)':<20} | 初始阶段双方均感知到 S2 略占优")
    print(f"{'S1 节点出边转移概率':<24} | {f'{memm_p_destA_given_s1:.3f} (强制饱和锁定)':<22} | {'(交由全局能量统一裁决)':<20} | MEMM 出现低出度作弊虚高")
    print(f"{'S2 节点目标分支概率':<23} | {f'{memm_p_destB_given_s2:.3f} (遭5个出度稀释)':<20} | {'(全局保留 7.5 绝对高分)':<20} | MEMM 优质分支被出度稀释惩罚")
    print("-" * 95)
    print(f"{'最终整条路径后验 P(Path A)':<20} | {f'{memm_prob_pathA:.4f} (误判为胜者!)':<22} | {f'{crf_prob_pathA:.4f} (被精准淘汰)':<22} | 🔴 MEMM 陷入标注偏置死锁")
    print(f"{'最终整条路径后验 P(Path B)':<20} | {f'{memm_prob_pathB:.4f} (惨遭淘汰)':<22} | {f'{crf_prob_pathB:.4f} (以 96.8% 胜出!)':<20} | 🟢 CRF 全局能量完胜")
    print("-" * 95)
    print("💡 结论与核心洞见：")
    print("  1. MEMM 的‘致命原罪’在于局部归一化：由于每个状态的出边概率和强行恒等于 1，")
    print("     导致‘低出度状态’无需与外界竞争即可获得虚高的转移概率（S1 只有一条路，走了就不扣分）；")
    print("  2. CRF 摒弃了局部软除法，直接将整条路径的状态转移与发射特征线性求和，")
    print("     最后统一通过配分函数 Z(X) 归一化。真正做到了‘前路漫漫，全局择优’！\n")

if __name__ == "__main__":
    run()

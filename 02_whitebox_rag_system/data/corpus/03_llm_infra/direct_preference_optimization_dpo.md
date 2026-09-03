# 对齐技术演进：PPO 强化学习 vs DPO 直接偏好优化

## 1. 传统 RLHF (PPO) 的四模型复杂链路
需要维护 Actor 模型、Critic 模型、Reward 模型和 Reference 模型，训练极易不稳定，显存开销巨大，超参数调试难度极高。

## 2. DPO (Direct Preference Optimization) 的数学奇迹
Rafailov 等人证明，可以通过数学推导将强化学习中的奖励函数隐式参数化为策略模型与参考模型的对数几率差（Log-Ratio）。
- **去中心化训练**：完全摒弃独立的 Reward 和 Critic 模型。
- 直接利用偏好数据对 $(x, y_w, y_l)$ 构建二元交叉熵损失，训练稳定性与收敛速度实现质的飞跃。

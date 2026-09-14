"""
深度世界模型 (Deep World Model) 白盒级教学实现与做梦推演引擎
基于 Model-Based RL (Ha & Schmidhuber, PlaNet, Dreamer) 核心架构思想

【核心理论架构与样本模型本质】:
1. 为什么它是纯正的“样本模型 (Sample Model)”？
   - 转移网络预测的是后继状态的条件均值 mu(s, a) 与方差 sigma(s, a)；
   - 在向规划算法或智能体大脑提供推演数据时，模型通过【重参数化技巧 (Reparameterization Trick)】：
     s_{t+1} = mu_theta(s_t, a_t) + sigma_theta(s_t, a_t) * epsilon,  epsilon ~ N(0, I)
   - 对外只返回单一、具象的下一个状态向量 (s_{t+1}, r_t)，而不是返回一个连续空间的联合积分公式；
   - 算法拿到的依然是“单步样本”，避开了高维连续空间全概率加权积分的维度灾难！

2. 深度世界模型的三大核心组件:
   - 概率动力学网络 (Probabilistic Transition Network): (s, a) -> (mu_{s'}, sigma_{s'});
   - 奖励预测网络 (Reward Predictor Network): (s, a, s') -> r;
   - 做梦仿真引擎 (Dream Simulation Engine): 在潜空间中利用白噪声自回归推演平行未来轨迹。
"""
import os
import sys
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. 连续物理仿真环境: 带随机扰动的倒立摆 (Stochastic Pendulum)
# =====================================================================

class StochasticPendulumEnv:
    """
    连续控制物理环境：带随机力矩扰动的连续非线性单摆 (Stochastic Inverted Pendulum).
    状态: [cos(theta), sin(theta), dtheta/dt (角速度)] in R^3
    动作: 连续力矩 u in [-2.0, 2.0] in R^1
    物理机制:
        theta_dot_dot = (3*g / (2*l)) * sin(theta) + (3 / (m*l^2)) * (u + noise)
        其中 noise ~ N(0, sigma_env^2) 模拟真实环境中的阵风与电机机械扰动。
    奖励函数:
        r = -(theta^2 + 0.1 * theta_dot^2 + 0.001 * u^2)
        当摆杆竖直倒立静止时奖励最大 (r -> 0)。
    """
    def __init__(self, g: float = 9.81, m: float = 1.0, l: float = 1.0, dt: float = 0.05, noise_std: float = 0.25):
        self.g = g
        self.m = m
        self.l = l
        self.dt = dt
        self.noise_std = noise_std
        self.max_torque = 2.0
        self.max_speed = 8.0
        
        self.theta = 0.0
        self.theta_dot = 0.0
        
    def reset(self, theta: Optional[float] = None, theta_dot: Optional[float] = None) -> np.ndarray:
        """重置环境，若未指定初始状态则随机初始化在下垂平衡点附近或全周"""
        if theta is None:
            self.theta = np.random.uniform(-np.pi, np.pi)
        else:
            self.theta = float(theta)
            
        if theta_dot is None:
            self.theta_dot = np.random.uniform(-1.0, 1.0)
        else:
            self.theta_dot = float(theta_dot)
            
        return self._get_obs()
        
    def _get_obs(self) -> np.ndarray:
        """返回 3 维连续观测 [cos(theta), sin(theta), theta_dot]"""
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot], dtype=np.float32)
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """物理世界真实推进单步 (包含环境客观物理法则与白噪声随机扰动)"""
        u = float(np.clip(action, -self.max_torque, self.max_torque))
        
        # 注入真实物理环境中的未知随机扰动 (风阻 / 摩擦波动)
        wind_gust = np.random.normal(0.0, self.noise_std)
        effective_torque = u + wind_gust
        
        # 连续角加速度计算
        angular_acc = (3.0 * self.g / (2.0 * self.l)) * np.sin(self.theta) + \
                      (3.0 / (self.m * (self.l ** 2))) * effective_torque
                      
        # 数值积分推进
        self.theta_dot = np.clip(self.theta_dot + angular_acc * self.dt, -self.max_speed, self.max_speed)
        self.theta = self.theta + self.theta_dot * self.dt
        # 规范化角位移到 [-pi, pi]
        self.theta = ((self.theta + np.pi) % (2.0 * np.pi)) - np.pi
        
        # 奖励计算 (越靠近竖直向上 theta=0 且静止，惩罚越小)
        reward = -(self.theta ** 2 + 0.1 * (self.theta_dot ** 2) + 0.001 * (u ** 2))
        
        return self._get_obs(), float(reward), False, {"wind_gust": wind_gust}


# =====================================================================
# 2. 深度世界模型神经网络架构 (Deep World Model MLP)
# =====================================================================

class ProbabilisticTransitionModel(nn.Module):
    """
    参数化高斯转移预测模型 (Probabilistic Transition Dynamics Model).
    输入当前状态 s 与动作 a，预测下一个状态相对当前状态的残差增量 Delta_s = s' - s:
    Delta_s ~ N(mu_theta(s, a), diag(sigma_theta^2(s, a)))
    采用重参数化技巧 (Reparameterization Trick) 生成具体的后继样本。
    """
    def __init__(self, state_dim: int = 3, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特征提取主干 (Shared Backbone)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 均值输出头 (预测状态残差增量 Delta_mu)
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        
        # 对数方差输出头 (预测内在不确定性 log_var)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
        
        # 方差上下界截断，防止数值溢出或方差塌缩
        self.min_logvar = -10.0
        self.max_logvar = 2.0
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算预测分布的均值与方差
        返回:
            pred_next_state_mu: 后继状态期望均值 mu(s') = s + Delta_mu
            logvar: 对数方差 log(sigma^2)
        """
        x = torch.cat([state, action], dim=-1)
        feat = self.backbone(x)
        
        delta_mu = self.mu_head(feat)
        pred_mu = state + delta_mu
        
        logvar = self.logvar_head(feat)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        
        return pred_mu, logvar
        
    def sample_step(self, state: torch.Tensor, action: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【样本模型的核心生成算子】: 重参数化采样 (Reparameterization Sampling)
        智能体在脑内推演时：
        1. 计算高斯分布参数 mu 与 sigma;
        2. 从标准正态分布中随机抽取一段白噪声 epsilon ~ N(0, I);
        3. 计算具体的后继状态样本: s_{t+1} = mu + sigma * epsilon;
        4. 仅对外吐出一个具体的具象状态向量，绝不向算法暴露高维积分！
        """
        pred_mu, logvar = self.forward(state, action)
        std = torch.exp(0.5 * logvar)
        
        if deterministic:
            next_state = pred_mu
        else:
            # 脑内掷骰子：采样标准白噪声
            epsilon = torch.randn_like(std)
            # 重参数化生成具体样本
            next_state = pred_mu + std * epsilon
            
        return next_state, pred_mu, std


class RewardPredictor(nn.Module):
    """
    奖励预测网络 (Reward Predictor Network).
    输入当前状态 s、动作 a 以及预测出的后继状态 s'，预测即时奖励 r = R_psi(s, a, s').
    """
    def __init__(self, state_dim: int = 3, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action, next_state], dim=-1)
        return self.net(x).squeeze(-1)


class DeepWorldModel(nn.Module):
    """
    一体化深度世界模型包装类 (Unified Deep World Model).
    封装了概率转移模型和奖励模型，对外提供极简的类似 gym.Env 的仿真推演接口。
    """
    def __init__(self, state_dim: int = 3, action_dim: int = 1, hidden_dim: int = 128, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.transition_model = ProbabilisticTransitionModel(state_dim, action_dim, hidden_dim).to(self.device)
        self.reward_model = RewardPredictor(state_dim, action_dim, hidden_dim=64).to(self.device)
        
    def step_simulation(self, state: np.ndarray, action: float, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        【面向规划算法的样本模型 API 接口】
        接收单步 (state, action)，在脑海中完成白噪声注入与前向推演，
        返回单次发生的具体结果 (next_state, reward)。
        外部规划算法调用它与调用真实物理环境在接口上 100% 同构！
        """
        self.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_t = torch.tensor([[action]], dtype=torch.float32, device=self.device)
            
            s_next_t, _, _ = self.transition_model.sample_step(s_t, a_t, deterministic=deterministic)
            r_t = self.reward_model(s_t, a_t, s_next_t)
            
            next_state = s_next_t.squeeze(0).cpu().numpy()
            reward = float(r_t.squeeze(0).cpu().item())
            
        return next_state, reward


# =====================================================================
# 3. 数据集回放池与世界模型训练器
# =====================================================================

class TransitionReplayBuffer:
    """真实物理世界交互数据回放池 (Offline / Online Transition Buffer)"""
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.ptr = 0
        
    def add(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
        else:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.ptr = (self.ptr + 1) % self.capacity
            
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.choice(len(self.states), size=batch_size, replace=False)
        return {
            "states": torch.tensor(np.array([self.states[i] for i in idxs]), dtype=torch.float32),
            "actions": torch.tensor(np.array([[self.actions[i]] for i in idxs]), dtype=torch.float32),
            "rewards": torch.tensor(np.array([self.rewards[i] for i in idxs]), dtype=torch.float32),
            "next_states": torch.tensor(np.array([self.next_states[i] for i in idxs]), dtype=torch.float32),
        }
        
    def __len__(self) -> int:
        return len(self.states)


class WorldModelTrainer:
    """
    世界模型训练器 (World Model Trainer)
    使用高斯负对数似然损失 (Gaussian NLL Loss) 训练转移模型，使用 MSE 训练奖励模型。
    """
    def __init__(self, world_model: DeepWorldModel, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.world_model = world_model
        self.optimizer = optim.AdamW(self.world_model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.world_model.train()
        device = self.world_model.device
        
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        next_states = batch["next_states"].to(device)
        
        # 1. 转移模型前向
        pred_mu, logvar = self.world_model.transition_model(states, actions)
        
        # 高斯负对数似然损失 (Gaussian NLL Loss):
        # 0.5 * sum [ log(sigma^2) + (y - mu)^2 / sigma^2 ]
        inv_var = torch.exp(-logvar)
        mse_term = (next_states - pred_mu) ** 2
        nll_loss = 0.5 * torch.mean(logvar + mse_term * inv_var)
        
        # 2. 奖励模型前向
        pred_rewards = self.world_model.reward_model(states, actions, next_states)
        reward_loss = 0.5 * torch.mean((pred_rewards - rewards) ** 2)
        
        # 3. 联合总损失
        total_loss = nll_loss + reward_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return {
            "total_loss": float(total_loss.item()),
            "nll_loss": float(nll_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "mse_norm": float(torch.mean(mse_term).item())
        }


# =====================================================================
# 4. 做梦推演与基于模型的预测控制引擎 (Dream Simulator & MPC)
# =====================================================================

class DreamSimulator:
    """
    做梦仿真与规划推演引擎 (Dream Simulation & Latent Imagination Engine).
    展示深度样本模型的全部仿真威力：
    1. 单轨迹推演 (Rollout Imagination);
    2. 多模态平行宇宙推演云 (Parallel Universes Branching Fan);
    3. 基于脑内推演的随机射击 MPC 规划 (Model-Predictive Control via Imagination).
    """
    def __init__(self, world_model: DeepWorldModel):
        self.world_model = world_model
        
    def rollout_dream(self, init_state: np.ndarray, action_sequence: List[float], deterministic: bool = False) -> Dict[str, np.ndarray]:
        """
        在智能体脑海内部推演一条未来轨迹 (完全不与真实物理世界发生交互！)
        """
        states = [init_state]
        rewards = []
        mus = []
        stds = []
        
        curr_state = init_state.copy()
        for a in action_sequence:
            self.world_model.eval()
            with torch.no_grad():
                s_t = torch.tensor(curr_state, dtype=torch.float32, device=self.world_model.device).unsqueeze(0)
                a_t = torch.tensor([[a]], dtype=torch.float32, device=self.world_model.device)
                
                s_next_t, mu_t, std_t = self.world_model.transition_model.sample_step(s_t, a_t, deterministic=deterministic)
                r_t = self.world_model.reward_model(s_t, a_t, s_next_t)
                
                curr_state = s_next_t.squeeze(0).cpu().numpy()
                states.append(curr_state)
                rewards.append(float(r_t.squeeze(0).cpu().item()))
                mus.append(mu_t.squeeze(0).cpu().numpy())
                stds.append(std_t.squeeze(0).cpu().numpy())
                
        return {
            "states": np.array(states),
            "rewards": np.array(rewards),
            "mus": np.array(mus),
            "stds": np.array(stds)
        }
        
    def parallel_dream_fan(self, init_state: np.ndarray, action_sequence: List[float], num_universes: int = 50) -> List[np.ndarray]:
        """
        【样本模型独有魅力：多模态平行宇宙发散云】
        从同一个初始状态 s0 出发，执行完全相同的动作序列，
        由于每次单步推演都会从标准正态分布中抽取不同的白噪声 epsilon ~ N(0, I)，
        模型将并行演化出成百上千条互不相同的“平行未来”！
        这直观证明了：样本模型不需要计算高维解析积分，仅凭单步随机采样即可完整展现未来概率云！
        """
        parallel_trajectories = []
        for _ in range(num_universes):
            dream = self.rollout_dream(init_state, action_sequence, deterministic=False)
            parallel_trajectories.append(dream["states"])
            
        return parallel_trajectories

    def plan_mpc(self, current_state: np.ndarray, horizon: int = 15, num_candidates: int = 100) -> float:
        """
        基于世界模型做梦推演的模型预测控制 (Vectorized Random Shooting MPC via Dream Imagination).
        智能体在当前真实状态下暂停，利用 GPU 张量并行在脑海中同时模拟 num_candidates 条不同候选动作序列，
        评估哪条动作序列在脑内获得的回报期望最高，并选取最优序列的第一个动作执行！
        """
        self.world_model.eval()
        device = self.world_model.device
        with torch.no_grad():
            # 候选动作张量: shape (num_candidates, horizon, 1)
            candidate_actions = torch.empty(num_candidates, horizon, 1, device=device).uniform_(-2.0, 2.0)
            
            # 初始状态在 batch 维度复制: shape (num_candidates, 3)
            curr_states = torch.tensor(current_state, dtype=torch.float32, device=device).repeat(num_candidates, 1)
            total_rewards = torch.zeros(num_candidates, device=device)
            
            for h in range(horizon):
                acts = candidate_actions[:, h, :]  # (num_candidates, 1)
                next_states, _, _ = self.world_model.transition_model.sample_step(curr_states, acts, deterministic=True)
                rews = self.world_model.reward_model(curr_states, acts, next_states)
                total_rewards += rews
                curr_states = next_states
                
            best_idx = torch.argmax(total_rewards).item()
            best_action = candidate_actions[best_idx, 0, 0].item()
            
        return float(best_action)


"""
深度 Q 网络算法白盒级教学实现 (Deep Q-Network, DQN White-box Implementation)
论文出处: Mnih et al., Nature 2015 "Human-level control through deep reinforcement learning"

【算法理论核心与突破性创新】:
1. 为什么表格型 Q-Learning 在高维/连续状态下失效？
   - 维度灾难 (Curse of Dimensionality): 状态空间庞大时无法建立 Q 表格；
   - 无法泛化 (No Generalization): 表格型算法中未见过的状态无法从相似状态迁移经验。
2. 神经网络逼近面临的“致命三要素 (The Deadly Triad)”:
   - 函数逼近 (Function Approximation) + 自举 (Bootstrapping) + 异策略 (Off-policy);
   - 直接用神经网络拟合 Q 函数会导致训练剧烈震荡甚至发散！
3. DeepMind 的两大稳定法宝:
   - 经验回放池 (Experience Replay Buffer): 打破时序连续样本的高度自相关性，恢复 I.I.D. 假设；
   - 目标网络 (Target Network): 解耦当前 Q 估计与 TD 目标，切断“自己追逐自己”的移动靶效应。
"""
import os
import sys
import random
import pathlib
from typing import Dict, List, Tuple, Optional
from collections import deque

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.grid_world import GridWorld, ACTIONS, ACTION_NAMES, UP, DOWN, LEFT, RIGHT

class QNetwork(nn.Module):
    """
    DQN Q 值函数逼近深度神经网络 (Deep Q-Network MLP).
    输入状态特征向量，输出各个离散动作的估计动作价值 Q(s, a)。
    """
    def __init__(self, state_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 正交初始化权重，提升训练稳定性
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 计算当前状态下所有动作的 Q 值.
        
        参数:
            state (torch.Tensor): 形状为 [batch_size, state_dim] 的状态特征张量
            
        返回:
            torch.Tensor: 形状为 [batch_size, action_dim] 的动作价值 Q(s, ·)
        """
        return self.net(state)

class ReplayBuffer:
    """
    经验回放池 (Experience Replay Buffer).
    利用先进先出队列 (FIFO) 存储智能体与环境交互产生的转移四元组 (s, a, r, s', done)，
    并在训练时进行均匀无偏随机采样，彻底打破时序序列的时间自相关性。
    """
    def __init__(self, capacity: int = 2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int], done: bool):
        """将一条交互经验存入回放池."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        """从回放池中均匀随机无放回采样一个批次."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    r"""
    深度 Q 网络智能体 (DQN Agent, Mnih et al. Nature 2015).
    
    【数学损失函数与半梯度反向传播】:
    当前策略评估网络权重为 θ，目标网络参数为 θ^-。
    对于采样批次中的转移样本 (s_i, a_i, r_i, s'_i, done_i)，贝尔曼最优目标为:
    
        y_i = r_i + \gamma \cdot \max_{a' \in \mathcal{A}} Q_{\theta^-}(s'_i, a') \cdot (1 - done_i)
        
    均方误差损失函数 (MSE Loss / Huber Loss):
    
        \mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( y_i - Q_\theta(s_i, a_i) \right)^2
        
    半梯度更新 (目标 y_i 不对 θ 求导，即截断梯度):
        \nabla_\theta \mathcal{L}(\theta) = - \frac{2}{B} \sum_{i=1}^B \left( y_i - Q_\theta(s_i, a_i) \right) \nabla_\theta Q_\theta(s_i, a_i)
    """
    def __init__(
        self,
        env: GridWorld,
        lr: float = 1e-3,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.992,
        buffer_capacity: int = 3000,
        batch_size: int = 32,
        target_update_freq: int = 50,
        hidden_dim: int = 64,
        seed: int = 42
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # 随机种子固化
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.state_dim = env.rows * env.cols
        self.action_dim = len(ACTIONS)
        
        # 1. 评估网络 (Q_eval / Current Q-Network)
        self.q_eval = QNetwork(self.state_dim, self.action_dim, hidden_dim)
        # 2. 目标网络 (Q_target / Target Network)
        self.q_target = QNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval() # 目标网络永久处于评估冻结模式
        
        # 优化器与损失函数 (使用 Smooth L1 / Huber Loss 增强异常值鲁棒性)
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        
        # 经验回放池
        self.memory = ReplayBuffer(buffer_capacity)
        self.replay_buffer = self.memory

    def get_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """获取指定状态在当前评估网络下的所有动作 Q 值."""
        with torch.no_grad():
            s_tensor = self.state_to_tensor(state).unsqueeze(0)
            return self.q_eval(s_tensor).squeeze(0).cpu().numpy()

    def state_to_tensor(self, state: Tuple[int, int]) -> torch.Tensor:
        """
        将网格二维离散坐标 (r, c) 转换为标准 One-Hot 特征向量张量.
        状态空间的 16 个网格对应 16 维的正交单位基向量，保证特征表达的无偏与正交性。
        """
        r, c = state
        idx = r * self.env.cols + c
        one_hot = np.zeros(self.state_dim, dtype=np.float32)
        one_hot[idx] = 1.0
        return torch.tensor(one_hot, dtype=torch.float32)

    def select_action(self, state: Tuple[int, int], deterministic: bool = False) -> int:
        """
        根据 ε-贪婪策略 (ε-greedy) 选择动作.
        
        参数:
            state: 当前网格坐标
            deterministic: 若为 True 则完全采用贪婪动作（测试/评估模式），忽略 ε 探索
        """
        if not deterministic and random.random() < self.epsilon:
            return random.choice(ACTIONS)
            
        with torch.no_grad():
            s_tensor = self.state_to_tensor(state).unsqueeze(0) # [1, state_dim]
            q_values = self.q_eval(s_tensor)                    # [1, action_dim]
            return int(torch.argmax(q_values, dim=1).item())

    def update_model(self) -> Optional[float]:
        """
        从经验回放池采样单批次数据并执行一次贝尔曼误差反向传播.
        
        返回:
            Optional[float]: 当前步的标量 Loss 值（若样本不足返回 None）
        """
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 批次张量化
        b_s = torch.stack([self.state_to_tensor(s) for s in states])              # [B, state_dim]
        b_a = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)               # [B, 1]
        b_r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)             # [B, 1]
        b_s_next = torch.stack([self.state_to_tensor(s) for s in next_states])    # [B, state_dim]
        b_done = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)            # [B, 1]
        
        # 1. 计算当前网络输出 Q_eval(s, a)
        q_eval_all = self.q_eval(b_s)                          # [B, action_dim]
        q_eval_curr = q_eval_all.gather(1, b_a)                 # 提取实际采取动作 a 的 Q 值 [B, 1]
        
        # 2. 用目标网络计算 TD 目标: y = r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next_target_all = self.q_target(b_s_next)        # [B, action_dim]
            max_q_next = q_next_target_all.max(dim=1, keepdim=True)[0] # [B, 1]
            td_target = b_r + self.gamma * max_q_next * (1.0 - b_done)
            
        # 3. 计算回归损失并执行梯度下降
        loss = self.loss_fn(q_eval_curr, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # 4. 周期性同步目标网络 (硬同步 Hard Update)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            
        return float(loss.item())

    def evaluate_q_error(self, ground_truth_q: Dict) -> float:
        """
        计算神经网络估计的 Q_θ(s, a) 与动态规划理论真值 Q*(s, a) 之间的均方误差 MSE.
        实证检验深度强化学习对理论极限的逼近精度！
        """
        errors = []
        with torch.no_grad():
            for s in self.env.states:
                if self.env.is_terminal(s):
                    continue
                s_tensor = self.state_to_tensor(s).unsqueeze(0)
                q_preds = self.q_eval(s_tensor).squeeze(0).numpy()
                for a in ACTIONS:
                    if isinstance(ground_truth_q.get(s), dict):
                        gt = ground_truth_q[s].get(a, 0.0)
                    else:
                        gt = ground_truth_q.get((s, a), 0.0)
                    pred = q_preds[a]
                    errors.append((pred - gt) ** 2)
        return float(np.mean(errors)) if errors else 0.0

    def train(
        self,
        episodes: int = 350,
        max_steps_per_episode: int = 60,
        ground_truth_q: Optional[Dict] = None,
        verbose_interval: int = 50
    ) -> Dict:
        """
        执行完整 DQN 训练循环.
        
        参数:
            episodes: 训练总回合数
            max_steps_per_episode: 单回合最大允许探索步长上限（防死循环）
            ground_truth_q: 理论最优动作价值字典（用于真值误差追踪）
            verbose_interval: 日志打印间隔
            
        返回:
            Dict: 记录详实训练曲线的字典 (returns, losses, q_errors, lengths, epsilons)
        """
        history = {
            "episodes": [],
            "returns": [],
            "lengths": [],
            "losses": [],
            "epsilons": [],
            "q_errors": []
        }
        
        print("\n" + "="*75)
        print("  >>> [深度 Q 网络 DQN (Deep Q-Network)] 算法开始训练")
        print("="*75)
        print(f"架构: One-Hot(16) -> MLP(64, 64) -> Q(4) | 批大小={self.batch_size}, 学习率={self.optimizer.param_groups[0]['lr']}")
        print(f"经验回放容量={self.memory.buffer.maxlen}, 目标网络同步频率={self.target_update_freq} 步")
        
        for ep in range(1, episodes + 1):
            state = self.env.reset()
            ep_return = 0.0
            ep_losses = []
            steps = 0
            done = False
            
            while not done and steps < max_steps_per_episode:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # 存入经验回放池
                self.memory.push(state, action, reward, next_state, done)
                
                # 神经网络参数优化更新
                step_loss = self.update_model()
                if step_loss is not None:
                    ep_losses.append(step_loss)
                    
                state = next_state
                ep_return += reward
                steps += 1
                
            # ε 退火衰减
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 计算本回合平均损失与理论真值误差
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            q_err = self.evaluate_q_error(ground_truth_q) if ground_truth_q is not None else 0.0
            
            history["episodes"].append(ep)
            history["returns"].append(ep_return)
            history["lengths"].append(steps)
            history["losses"].append(avg_loss)
            history["epsilons"].append(self.epsilon)
            history["q_errors"].append(q_err)
            
            if ep % verbose_interval == 0 or ep == 1:
                recent_return = np.mean(history["returns"][-20:])
                print(f"Episode {ep:>3d}/{episodes} | 步数: {steps:>2d} | 回报: {ep_return:>6.1f} (近20轮均值: {recent_return:>5.1f}) | 损失: {avg_loss:>6.4f} | Q*真值MSE: {q_err:>6.4f} | eps: {self.epsilon:.3f}")
                
        # 导出训练后网络给出的全图状态价值 V_θ 与确定性策略 π_θ
        final_v = {}
        final_q = {}
        final_pi = {}
        with torch.no_grad():
            for s in self.env.states:
                if self.env.is_terminal(s):
                    final_v[s] = 0.0
                    final_q[s] = {a: 0.0 for a in ACTIONS}
                    final_pi[s] = 0
                    continue
                s_tensor = self.state_to_tensor(s).unsqueeze(0)
                q_vals = self.q_eval(s_tensor).squeeze(0).numpy()
                final_q[s] = {a: float(q_vals[a]) for a in ACTIONS}
                final_v[s] = float(np.max(q_vals))
                final_pi[s] = int(np.argmax(q_vals))
                
        history["final_V"] = final_v
        history["final_Q"] = final_q
        history["final_pi"] = final_pi
        
        print("\n" + "="*75)
        print("  🎯 DQN 训练完成！已成功收敛并提取神经网络动作价值与最优策略")
        print("="*75)
        return history

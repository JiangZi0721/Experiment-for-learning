"""
无模型强化学习经典算法 (Model-Free Reinforcement Learning Algorithms)
包含算法:
1. TD(0) 状态价值估计 (时序差分策略评估)
2. SARSA 同策略控制 (On-Policy TD Control)
3. Q-Learning 异策略控制 (Off-Policy TD Control)

每个函数均配备保姆级公式解析、物理含义推导与逐行教学注释，供初学者直接通过阅读代码理解核心原理。
"""
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.grid_world import GridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_NAMES

# ======================================================================================
# 动作选择机制: Epsilon-Greedy (探索与利用平衡算子)
# ======================================================================================
def epsilon_greedy(q_values: Dict[int, float], epsilon: float) -> int:
    r"""
    ε-贪婪动作选择策略 (Epsilon-Greedy Action Selection Policy)
    
    【理论核心与数学公式】:
    在强化学习控制问题中，智能体面临“利用已知最优动作 (Exploitation)”与“探索未知动作以发现潜在更优解 (Exploration)”
    的基本权衡。ε-贪婪算法通过如下分段概率分布来兼顾二者:
    
                   ┌ 1 - ε + ε / |A|,   若 a = argmax_{a'} Q(s, a') (当前估计最优动作)
        π(a | s) = │
                   └ ε / |A|,           若 a ≠ argmax_{a'} Q(s, a') (非最优动作，探索概率)
                   
    等价物理实现:
    1. 以概率 ε 随机从所有可能动作中均匀挑选一个 (探索 Exploration)
    2. 以概率 (1 - ε) 挑选使得 Q(s, a) 最大的动作 (利用 Exploitation)
    
    【平局处理 (Tie-Breaking)】:
    当多个动作的 Q 值相同时（如初始化全为 0 时），不能偏向于字典排序的第一个动作，
    必须从所有并列最大值中随机等概率挑选一个，防止引入探索偏见。
    
    参数:
        q_values (Dict[int, float]): 当前状态下所有动作的 Q 估计值映射 {动作: Q值}
        epsilon (float): 探索率 ε ∈ [0, 1]，0 表示完全贪婪，1 表示完全随机探索
        
    返回:
        int: 选定的离散动作 (UP=0, DOWN=1, LEFT=2, RIGHT=3)
    """
    # 1. 掷骰子决定是探索还是利用:
    # random.random() 生成 [0.0, 1.0) 之间的浮点数。若小于 epsilon 则执行随机探索。
    if random.random() < epsilon:
        return random.choice(ACTIONS)
        
    # 2. 否则执行贪婪利用: 找出所有动作中 Q 值最大的值
    max_q = max(q_values.values())
    
    # 3. 找出所有与最大 Q 值在数值精度内相等的动作列表 (平局处理)
    best_actions = [a for a, q in q_values.items() if np.isclose(q, max_q, atol=1e-8)]
    
    # 4. 在所有并列最优的动作中随机挑选一个返回
    return random.choice(best_actions)


# ======================================================================================
# 算法一: TD(0) 状态价值估计 (Temporal Difference Prediction)
# ======================================================================================
def run_td_zero(
    env: GridWorld,
    num_episodes: int = 2500,
    alpha: float = 0.05,
    gamma: float = 0.9,
    true_V: Optional[Dict[Tuple[int, int], float]] = None,
    seed: int = 42
) -> Dict:
    r"""
    TD(0) 状态价值估计算法 (单步时序差分策略评估)
    
    【理论核心与数学更新公式】:
    在给定策略 π 下（本函数采用全图均匀随机策略 π(a|s) = 0.25），智能体通过与真实环境单步交互，
    利用下一时刻的估计值实时修正当前状态的估计值:
    
        V(S_t) ← V(S_t) + α · [ R_{t+1} + γ · V(S_{t+1}) - V(S_t) ]
                               \_________________________/
                                      TD 目标 (Target)
                              \____________________________________/
                                        TD 误差 δ_t (Error)
                                        
    【深度物理剖析】:
    1. 为什么是 Model-Free (无模型)？
       - 动态规划需要已知环境的转移概率矩阵 P(s'|s,a) 与奖励函数 R(s,a) 来求期望;
       - TD(0) 根本不需要任何矩阵，仅凭真实交互采样产生的一个物理四元组 (S_t, A_t, R_{t+1}, S_{t+1}) 即可更新！
    2. 为什么叫 自举 (Bootstrapping)？
       - 蒙特卡洛 (MC) 必须等到回合彻底结束，计算整个完整轨迹的真实总回报 G_t；
       - TD(0) 在走到下一步 S_{t+1} 时，直接用当前对该状态的估计值 V(S_{t+1}) 代替未来所有真实回报，走一步就能更新一步！
    3. 为什么 α 对应在线指数滑动平均 (EMA)？
       - 更新式可写为: V(S) ← (1 - α) · V(S) + α · [R + γ V(S')]
       - 历史旧估计占比 (1 - α)，新单步观测目标占比 α，历史权重呈几何级数 (1 - α)^k 指数衰减。
    4. 终止吸收态的处理:
       - 若 S_{t+1} 属于终点 (Terminal State)，由于终点后不再有未来转移，其未来价值严格锁定为 V(S_{t+1}) = 0，
         此时 TD 目标退化为纯即时奖励 R_{t+1}。
         
    参数:
        env (GridWorld): 网格世界物理环境实例
        num_episodes (int): 训练交互总回合数
        alpha (float): 学习率（更新步长） α ∈ (0, 1]
        gamma (float): 折扣因子 γ ∈ [0, 1]
        true_V (Optional[Dict]): 动态规划精确求出的真值字典（用于计算收敛均方根误差 RMSE）
        seed (int): 随机种子，确保实验 100% 可重复
        
    返回:
        Dict: 包含收敛状态价值 V、RMSE 历史曲线、以及前几个步骤的白盒计算拆解
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 状态价值函数 V(s) 初始化: 全图所有非终止状态初始估值设为 0.0
    V = {s: 0.0 for s in env.states}
    rmse_history = []
    white_box_steps = []
    
    for ep in range(num_episodes):
        # 探索起手式: 为了让全图各个角落都能被均匀访问，每回合随机选择一个非终止状态作为起点
        non_terminals = [s for s in env.states if not env.is_terminal(s)]
        s = random.choice(non_terminals)
        env.current_state = s
        
        steps = 0
        # 只要当前状态不是终点，且单局步数未超过上限，就持续交互
        while not env.is_terminal(s) and steps < 100:
            # 步骤 1: 根据被评估策略采取动作（此处为基准均匀随机策略）
            a = random.choice(ACTIONS)
            
            # 步骤 2: 与环境执行一步交互，获取下一状态 next_s、即时奖励 r、是否到达终点 done
            next_s, r, done = env.step(a)
            
            # 步骤 3: 计算时序差分目标 (TD Target)
            # 若已经到达终点 (done=True)，则未来价值为 0，目标直接等于即时奖励 r；
            # 否则自举后继状态的估计值: r + gamma * V[next_s]
            td_target = r + (0.0 if done else gamma * V[next_s])
            
            # 步骤 4: 计算时序差分误差 (TD Error): 观测到的目标 与 当前估计 之间的差值
            td_error = td_target - V[s]
            
            # 步骤 5: 在线滑动平均更新当前状态价值 V(s)
            old_v = V[s]
            V[s] = old_v + alpha * td_error
            
            # 记录第 0 回合的前 3 步用于教学日志展示
            if ep == 0 and len(white_box_steps) < 3:
                white_box_steps.append({
                    "step": len(white_box_steps) + 1,
                    "s": s, "a": a, "r": r, "next_s": next_s,
                    "old_v": old_v, "v_next": 0.0 if done else V[next_s],
                    "target": td_target, "error": td_error, "new_v": V[s]
                })
                
            # 转移至下一状态，继续步进
            s = next_s
            steps += 1
            
        # 若提供了动态规划真值，计算当前估计与真实解之间的均方根误差 (RMSE)
        if true_V is not None:
            err = np.sqrt(np.mean([(V[st] - true_V[st])**2 for st in non_terminals]))
            rmse_history.append(err)
            
    return {
        "V": V,
        "rmse_history": rmse_history,
        "white_box_steps": white_box_steps
    }


# ======================================================================================
# 算法二: SARSA 同策略控制 (State-Action-Reward-State-Action On-Policy TD Control)
# ======================================================================================
def run_sarsa(
    env: GridWorld,
    num_episodes: int = 1500,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 42
) -> Dict:
    r"""
    SARSA 同策略控制算法 (On-Policy TD Control)
    
    【理论核心与数学更新公式】:
    SARSA 求解动作价值函数 Q(s, a)。更新依赖于五元组 (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})，
    故得名 S-A-R-S-A。其核心递推公式为:
    
        Q(S_t, A_t) ← Q(S_t, A_t) + α · [ R_{t+1} + γ · Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ]
                                         \_________________________________/
                                                    SARSA TD 目标
                                        \_____________________________________________/
                                                        SARSA TD 误差
                                                        
    【同策略 (On-Policy) 的数学与物理本质】:
    1. 什么是同策略？
       - 智能体用来与环境交互采样的策略是行为策略 (Behavior Policy，即 ε-贪婪策略)；
       - 在更新公式中，自举目标使用的下一个动作 A_{t+1}，是**严格根据该行为策略实际采样出来的下一个动作**！
       - 即: 采样策略 与 评估目标策略 是同一个策略 (On-Policy)！
    2. “知行合一”的谨慎避险特性:
       - 智能体深知自己在未来有 ε 的概率会“手滑乱走”。如果一条路非常接近悬崖/火坑，
         即便沿着悬崖走是物理最短路径，但一旦手滑就会跌入悬崖遭受巨额扣分。
       - SARSA 将这种“探索风险”直接计入了期望收益中，因此会主动放弃高风险捷径，学出一条远离危险的保守安全路线！
       
    参数:
        env (GridWorld): 网格物理环境
        num_episodes (int): 训练交互总局数
        alpha (float): 学习率 α
        gamma (float): 贴现因子 γ
        epsilon (float): 探索率 ε
        seed (int): 随机种子
        
    返回:
        Dict: 包含动作价值表 Q(s, a)、每回合累积奖励回报曲线、每回合步数曲线等
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 动作价值函数 Q(s, a) 初始化: 所有状态的所有动作初值设为 0.0
    Q = {s: {a: 0.0 for a in ACTIONS} for s in env.states}
    reward_history = []
    steps_history = []
    white_box_steps = []
    
    for ep in range(num_episodes):
        # 步骤 1: 初始化起始状态 S (复位到起点 (0, 0))
        s = env.reset()
        
        # 步骤 2: 根据当前 Q 估计，使用 ε-贪婪策略为当前状态挑选第一个动作 A
        a = epsilon_greedy(Q[s], epsilon)
        ep_reward = 0.0
        steps = 0
        
        while not env.is_terminal(s) and steps < 200:
            # 步骤 3: 采取动作 A，观察转移结果 (S', R, done)
            next_s, r, done = env.step(a)
            ep_reward += r
            
            # 步骤 4: 决定下一个动作 A' 并计算 TD 目标
            if done:
                # 若到达终点，终点后无后续动作价值，目标即为即时奖励
                td_target = r
                next_a = None
            else:
                # 【同策略核心行】: 使用与行为策略完全相同的 ε-贪婪策略在 next_s 处采样真实的下一个动作 A'
                next_a = epsilon_greedy(Q[next_s], epsilon)
                # 使用该实际动作的价值自举: R + gamma * Q(S', A')
                td_target = r + gamma * Q[next_s][next_a]
                
            # 步骤 5: 计算 SARSA TD 误差
            td_error = td_target - Q[s][a]
            
            # 步骤 6: 在线更新当前状态-动作对的价值 Q(S, A)
            old_q = Q[s][a]
            Q[s][a] = old_q + alpha * td_error
            
            # 记录第 0 回合的前 3 步用于教学对比
            if ep == 0 and len(white_box_steps) < 3:
                white_box_steps.append({
                    "step": len(white_box_steps) + 1,
                    "s": s, "a": a, "r": r, "next_s": next_s, "next_a": next_a,
                    "old_q": old_q, "target": td_target, "error": td_error, "new_q": Q[s][a]
                })
                
            # 步骤 7: 状态与动作向前推进一步: S ← S', A ← A'
            s = next_s
            a = next_a
            steps += 1
            
        reward_history.append(ep_reward)
        steps_history.append(steps)
        
    return {
        "Q": Q,
        "reward_history": reward_history,
        "steps_history": steps_history,
        "white_box_steps": white_box_steps
    }


# ======================================================================================
# 算法三: Q-Learning 异策略控制 (Watkins' Off-Policy TD Control)
# ======================================================================================
def run_q_learning(
    env: GridWorld,
    num_episodes: int = 1500,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 42
) -> Dict:
    r"""
    Q-Learning 异策略控制算法 (Watkins, 1989)
    
    【理论核心与数学更新公式】:
    Q-Learning 直接逼近贝尔曼最优方程中的最优动作价值函数 Q*(s, a):
    
        Q(S_t, A_t) ← Q(S_t, A_t) + α · [ R_{t+1} + γ · max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) ]
                                         \____________________________________/
                                                    Q-Learning TD 目标
                                        \_______________________________________________/
                                                        Q-Learning TD 误差
                                                        
    【异策略 (Off-Policy) 的数学与物理本质】:
    1. 什么是异策略？
       - 行为策略 (Behavior Policy): 智能体在当前物理环境中依然使用带有探索的 ε-贪婪策略行动，保证充分访问所有状态空间;
       - 目标策略 (Target Policy): 在构造更新目标时，**完全抛弃实际可能选择的动作，假想下一时刻执行的是绝对贪婪的最优动作 max_{a'} Q(S', a')**！
       - 行为策略 ≠ 目标策略，二者分离，故称为“异策略 (Off-Policy)”！
    2. “心怀天下”的激进寻优特性:
       - Q-Learning 假设智能体一旦学会了最优策略，未来在利用阶段是绝对不会“手滑乱走”的；
       - 因此它学习的是纯粹的理论极限最优解。即便紧挨着悬崖，只要悬崖旁边的路是最短路径，
         Q-Learning 也会毫不犹豫地收敛到贴着悬崖边缘走的最优激进路径。
         
    【SARSA vs Q-Learning 核心代码行对决】:
        SARSA:      td_target = r + gamma * Q[next_s][next_a]           # 采用实际采样的动作 next_a
        Q-Learning: td_target = r + gamma * max(Q[next_s].values())     # 强制采用理论最优动作 max_{a'}
        
    参数:
        env (GridWorld): 网格物理环境
        num_episodes (int): 训练交互总局数
        alpha (float): 学习率 α
        gamma (float): 贴现因子 γ
        epsilon (float): 探索率 ε
        seed (int): 随机种子
        
    返回:
        Dict: 包含动作价值表 Q(s, a)、每回合累积奖励回报曲线、每回合步数曲线等
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 动作价值函数 Q(s, a) 初始化: 全图初值设为 0.0
    Q = {s: {a: 0.0 for a in ACTIONS} for s in env.states}
    reward_history = []
    steps_history = []
    white_box_steps = []
    
    for ep in range(num_episodes):
        # 步骤 1: 初始化起始状态 S (复位到起点 (0, 0))
        s = env.reset()
        ep_reward = 0.0
        steps = 0
        
        while not env.is_terminal(s) and steps < 200:
            # 步骤 2: 依据行为策略 (ε-贪婪) 选择动作 A 去实际执行
            a = epsilon_greedy(Q[s], epsilon)
            
            # 步骤 3: 采取动作 A，观察真实反馈 (S', R, done)
            next_s, r, done = env.step(a)
            ep_reward += r
            
            # 步骤 4: 计算 Q-Learning 异策略 TD 目标
            if done:
                # 到达终点，终点未来无收益
                td_target = r
            else:
                # 【异策略核心行】: 直接对后继状态 S' 的所有可能动作取 MAX，忽略实际探索行为！
                max_next_q = max(Q[next_s].values())
                td_target = r + gamma * max_next_q
                
            # 步骤 5: 计算 Q-Learning TD 误差
            td_error = td_target - Q[s][a]
            
            # 步骤 6: 在线更新 Q(S, A)
            old_q = Q[s][a]
            Q[s][a] = old_q + alpha * td_error
            
            # 记录第 0 回合的前 3 步用于教学对比
            if ep == 0 and len(white_box_steps) < 3:
                white_box_steps.append({
                    "step": len(white_box_steps) + 1,
                    "s": s, "a": a, "r": r, "next_s": next_s,
                    "old_q": old_q, "target": td_target, "error": td_error, "new_q": Q[s][a]
                })
                
            # 步骤 7: 仅推进状态 S ← S' (注意: Q-Learning 不需要提前决定 A'，下一动作在下个循环重新由 ε-贪婪选取)
            s = next_s
            steps += 1
            
        reward_history.append(ep_reward)
        steps_history.append(steps)
        
    return {
        "Q": Q,
        "reward_history": reward_history,
        "steps_history": steps_history,
        "white_box_steps": white_box_steps
    }

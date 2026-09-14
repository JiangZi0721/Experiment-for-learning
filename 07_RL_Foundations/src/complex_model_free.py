"""
复杂多奖励网格世界模型无关算法实现 (Complex GridWorld Model-Free Learning with Event Logging)
功能:
1. 提供理论金标准求解器 solve_dp_optimal (基于贝尔曼最优方程的价值迭代)
2. TD(0) 状态价值评估 (支持高精度单步事件追踪)
3. SARSA 同策略控制 (支持高精度单步事件追踪)
4. Q-Learning 异策略控制 (支持高精度单步事件追踪)

全模块配备全量事件录制 (Event Stream) 功能，能够精准捕获全流程中价值变动量最大的时间步，供白盒深入教学。
"""
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.complex_grid_world import ComplexGridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_NAMES, ACTION_SYMBOLS

def solve_dp_optimal(
    env: ComplexGridWorld,
    gamma: float = 0.9,
    theta: float = 1e-7
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Dict[int, float]]]:
    r"""
    理论金标准求解器: 基于价值迭代精确求解复杂网格世界的理论最优解 V*(s) 与 Q*(s, a)
    
    【数学原理】:
        V_{k+1}(s) = max_{a} ∑_{s', r} P(s', r | s, a) [ r + γ · V_k(s') ]
        Q*(s, a) = ∑_{s', r} P(s', r | s, a) [ r + γ · V*(s') ]
        
    参数:
        env (ComplexGridWorld): 复杂网格环境实例
        gamma (float): 折扣因子
        theta (float): 极高精度的收敛判据阈值 (1e-7)
        
    返回:
        Tuple[Dict, Dict]: (最优状态价值 V*, 最优动作价值矩阵 Q*)
    """
    V = {s: 0.0 for s in env.states}
    
    while True:
        delta = 0.0
        new_V = V.copy()
        for s in env.states:
            # 终止状态吸收态价值严格为 0
            if env.is_terminal(s):
                new_V[s] = 0.0
                continue
            q_vals = []
            for a in ACTIONS:
                trans = env.get_transitions(s, a)
                prob, next_s, r = trans[0]
                # 自举后继价值: 若后继为终点则无未来收益
                q_vals.append(r + (0.0 if env.is_terminal(next_s) else gamma * V[next_s]))
            # 贝尔曼最优算子
            new_V[s] = max(q_vals)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
            
    # 计算精确动作价值矩阵 Q*(s, a)
    Q = {s: {a: 0.0 for a in ACTIONS} for s in env.states}
    for s in env.states:
        if env.is_terminal(s):
            continue
        for a in ACTIONS:
            trans = env.get_transitions(s, a)
            prob, next_s, r = trans[0]
            Q[s][a] = r + (0.0 if env.is_terminal(next_s) else gamma * V[next_s])
            
    return V, Q

def epsilon_greedy(q_values: Dict[int, float], epsilon: float) -> int:
    """ε-贪婪动作选择算子，包含随机平局打破机制。"""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if np.isclose(q, max_q, atol=1e-8)]
    return random.choice(best_actions)

def run_complex_td0(
    env: ComplexGridWorld,
    num_episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    seed: int = 42
) -> Dict:
    r"""
    复杂网格环境下的 TD(0) 状态评估与全事件捕获器
    
    公式:
        V(S) ← V(S) + α · [ R + γ · V(S') - V(S) ]
        
    特性:
        每发生一次单步更新，均记录详细快照字典 (包含状态转移、奖励、旧价值、TD目标、TD误差、新价值与改变量绝对值)。
    """
    random.seed(seed)
    np.random.seed(seed)
    
    V = {s: 0.0 for s in env.states}
    all_events = []
    
    for ep in range(num_episodes):
        # 均匀采样非终止状态作为探索起点
        non_terminals = [s for s in env.states if not env.is_terminal(s)]
        s = random.choice(non_terminals)
        env.reset(start_state=s)
        
        step_idx = 0
        while not env.is_terminal(s) and step_idx < 100:
            a = random.choice(ACTIONS) # 均匀随机基准策略
            next_s, r, done = env.step(a)
            
            # 贝尔曼期望自举目标
            td_target = r + (0.0 if done else gamma * V[next_s])
            td_error = td_target - V[s]
            old_v = V[s]
            new_v = old_v + alpha * td_error
            V[s] = new_v
            
            # 记录本次单步更新的物理事件
            all_events.append({
                "algo": "TD(0)",
                "episode": ep,
                "step": step_idx,
                "s": s,
                "a": a,
                "r": r,
                "next_s": next_s,
                "done": done,
                "old_val": old_v,
                "target": td_target,
                "error": td_error,
                "new_val": new_v,
                "abs_delta": abs(new_v - old_v),
                "snapshot_V": V.copy()
            })
            
            s = next_s
            step_idx += 1
            
    return {"V": V, "events": all_events}

def run_complex_sarsa(
    env: ComplexGridWorld,
    num_episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 42
) -> Dict:
    r"""
    复杂网格环境下的 SARSA 同策略控制与全事件捕获器
    
    公式:
        Q(S, A) ← Q(S, A) + α · [ R + γ · Q(S', A') - Q(S, A) ]
        
    特性:
        采用真实采样的下一个动作 A' 进行自举，全面捕获高危陷阱在同策略探索下的恐惧惩罚机制。
    """
    random.seed(seed)
    np.random.seed(seed)
    
    Q = {s: {a: 0.0 for a in ACTIONS} for s in env.states}
    all_events = []
    reward_history = []
    
    for ep in range(num_episodes):
        s = env.reset(start_state=(0, 0))
        a = epsilon_greedy(Q[s], epsilon)
        ep_reward = 0.0
        step_idx = 0
        
        while not env.is_terminal(s) and step_idx < 150:
            next_s, r, done = env.step(a)
            ep_reward += r
            
            if done:
                td_target = r
                next_a = None
            else:
                # 同策略采样: 实际探索的下一动作
                next_a = epsilon_greedy(Q[next_s], epsilon)
                td_target = r + gamma * Q[next_s][next_a]
                
            td_error = td_target - Q[s][a]
            old_q = Q[s][a]
            new_q = old_q + alpha * td_error
            Q[s][a] = new_q
            
            # 记录本次更新事件
            all_events.append({
                "algo": "SARSA",
                "episode": ep,
                "step": step_idx,
                "s": s,
                "a": a,
                "r": r,
                "next_s": next_s,
                "next_a": next_a,
                "done": done,
                "old_val": old_q,
                "target": td_target,
                "error": td_error,
                "new_val": new_q,
                "abs_delta": abs(new_q - old_q),
                "snapshot_Q": {st: Q[st].copy() for st in env.states}
            })
            
            s = next_s
            a = next_a
            step_idx += 1
            
        reward_history.append(ep_reward)
        
    return {"Q": Q, "events": all_events, "reward_history": reward_history}

def run_complex_q_learning(
    env: ComplexGridWorld,
    num_episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 42
) -> Dict:
    r"""
    复杂网格环境下的 Q-Learning 异策略控制与全事件捕获器
    
    公式:
        Q(S, A) ← Q(S, A) + α · [ R + γ · max_{a'} Q(S', a') - Q(S, A) ]
        
    特性:
        直接利用 max 算子自举后继状态的最优估值，展现异策略回传对次优动作的逆转翻盘。
    """
    random.seed(seed)
    np.random.seed(seed)
    
    Q = {s: {a: 0.0 for a in ACTIONS} for s in env.states}
    all_events = []
    reward_history = []
    
    for ep in range(num_episodes):
        s = env.reset(start_state=(0, 0))
        ep_reward = 0.0
        step_idx = 0
        
        while not env.is_terminal(s) and step_idx < 150:
            a = epsilon_greedy(Q[s], epsilon)
            next_s, r, done = env.step(a)
            ep_reward += r
            
            if done:
                td_target = r
                max_next_q = 0.0
            else:
                # 异策略核心: 理论最优动作最大化
                max_next_q = max(Q[next_s].values())
                td_target = r + gamma * max_next_q
                
            td_error = td_target - Q[s][a]
            old_q = Q[s][a]
            new_q = old_q + alpha * td_error
            Q[s][a] = new_q
            
            all_events.append({
                "algo": "Q-Learning",
                "episode": ep,
                "step": step_idx,
                "s": s,
                "a": a,
                "r": r,
                "next_s": next_s,
                "max_next_q": max_next_q,
                "done": done,
                "old_val": old_q,
                "target": td_target,
                "error": td_error,
                "new_val": new_q,
                "abs_delta": abs(new_q - old_q),
                "snapshot_Q": {st: Q[st].copy() for st in env.states}
            })
            
            s = next_s
            step_idx += 1
            
        reward_history.append(ep_reward)
        
    return {"Q": Q, "events": all_events, "reward_history": reward_history}

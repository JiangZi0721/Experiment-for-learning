"""
标准 4x4 网格世界马尔可夫决策过程环境 (Standard 4x4 GridWorld MDP Environment)

【马尔可夫决策过程五元组 (S, A, P, R, γ) 形式化定义】:
1. 状态空间 S: 4x4 离散网格坐标集合 {(r, c) | r ∈ [0, 3], c ∈ [0, 3]}，共 16 个状态。
2. 动作空间 A: 离散 4 方向动作 {UP(0), DOWN(1), LEFT(2), RIGHT(3)}。
3. 状态转移概率 P(s' | s, a): 确定性转移（转移概率恒为 1.0），若撞墙则发生弹性碰撞反弹停留在原地。
4. 奖励函数 R(s, a, s'): 进入陷阱 (1, 2) 给予惩罚 -5.0，进入终点 (3, 3) 给予奖励 0.0，其余移动均为单步损耗 -1.0。
5. 折扣因子 γ: 默认 0.9。

【双接口设计】:
- Model-Based 接口 (get_transitions): 面向动态规划 (DP)，提供完整的状态转移概率分布 P 与期望奖励。
- Model-Free 接口 (reset, step): 面向时序差分 (TD/SARSA/Q-Learning)，仅提供与真实物理黑盒交互的单步采样能力。
"""
from typing import Dict, List, Tuple, Optional

# 离散动作常量编码
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "UP (↑)", DOWN: "DOWN (↓)", LEFT: "LEFT (←)", RIGHT: "RIGHT (→)"}
ACTION_SYMBOLS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}

class GridWorld:
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        goal_states: Optional[List[Tuple[int, int]]] = None,
        trap_states: Optional[Dict[Tuple[int, int], float]] = None,
        step_cost: float = -1.0,
        gamma: float = 0.9,
    ):
        """
        初始化网格世界环境.
        
        参数:
            rows (int): 网格行数 (默认 4)
            cols (int): 网格列数 (默认 4)
            goal_states (List[Tuple[int, int]]): 终止目标吸收态坐标列表 (默认右下角 (3, 3))
            trap_states (Dict[Tuple[int, int], float]): 陷阱网格及其惩罚值映射 (默认 (1, 2) 罚 -5.0)
            step_cost (float): 基础单步行动能耗惩罚 (默认 -1.0)
            gamma (float): 折扣贴现因子 (默认 0.9)
        """
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.step_cost = step_cost
        
        # 目标终点与陷阱配置
        self.goal_states = goal_states if goal_states is not None else [(rows - 1, cols - 1)]
        self.trap_states = trap_states if trap_states is not None else {(1, 2): -5.0}
        
        # 构建离散状态全集 S
        self.states = [(r, c) for r in range(rows) for c in range(cols)]
        self.current_state = (0, 0)
        
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """判定给定状态是否为终止吸收态 (Terminal Absorbing State)。"""
        return state in self.goal_states
        
    def get_transitions(self, state: Tuple[int, int], action: int) -> List[Tuple[float, Tuple[int, int], float]]:
        r"""
        【Model-Based 接口】获取状态转移概率与即时奖励: P(s' | s, a) 和 R(s, a, s')
        供动态规划 (策略迭代、价值迭代) 遍历计算贝尔曼期望与最优方程。
        
        返回:
            List[Tuple[prob, next_state, reward]]:
            由于本环境为确定性物理环境，列表内仅包含 1 个概率为 1.0 的元组。
        """
        # 边界吸收性: 若当前状态已是终止态，它自环转移至自身，且不再产生任何奖励
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]
            
        r, c = state
        # 确定性转移物理引擎: 碰到边界时 max/min 操作会强行截断，实现“撞墙停在原地”
        if action == UP:
            next_r, next_c = max(r - 1, 0), c
        elif action == DOWN:
            next_r, next_c = min(r + 1, self.rows - 1), c
        elif action == LEFT:
            next_r, next_c = r, max(c - 1, 0)
        elif action == RIGHT:
            next_r, next_c = r, min(c + 1, self.cols - 1)
        else:
            raise ValueError(f"无效动作编码: {action}")
            
        next_state = (next_r, next_c)
        
        # 即时奖励结算逻辑:
        # 若下一状态是陷阱，返回陷阱惩罚；否则返回标准步长成本 (-1.0)
        if next_state in self.trap_states:
            reward = self.trap_states[next_state]
        else:
            reward = self.step_cost
            
        return [(1.0, next_state, reward)]

    def reset(self) -> Tuple[int, int]:
        """
        【Model-Free 接口】复位环境，将智能体放回初始起点 (0, 0)。
        """
        self.current_state = (0, 0)
        return self.current_state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        【Model-Free 接口】真实物理单步交互.
        模拟智能体在黑盒世界中执行动作 a，观测环境吐出的下一个状态、奖励以及回合结束信号。
        
        参数:
            action (int): 智能体决策采取的动作
            
        返回:
            next_state (Tuple[int, int]): 动作发生后所处的下一个物理网格坐标
            reward (float): 动作发生后环境给予的即时数值反馈
            done (bool): 是否到达终止状态（回合结束标记）
        """
        transitions = self.get_transitions(self.current_state, action)
        _, next_state, reward = transitions[0]
        self.current_state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

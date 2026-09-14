"""
进阶多级奖惩复杂网格环境 (Complex Multi-Reward GridWorld Environment)

【理论与物理拓扑设计】:
本环境专为探究“风险与收益权衡 (Risk-Reward Trade-off)”与“局部次优陷阱”而设计，包含:
1. 双终点吸收态:
   - 次级安全目标 (Sub-Goal) (0, 3): 奖励 +3.0，安全、距离短 (3步可达)、方差极小。
   - 终极大奖目标 (Main Goal) (3, 3): 奖励 +10.0，奖金极其丰厚，但四周布满陷阱。
2. 三级梯度危险陷阱 (非终止进入惩罚):
   - 致命熔岩坑 (Lava Pit) (1, 2): 惩罚 -10.0 (中心毁灭区)。
   - 尖刺陷阱 (Spike Trap) (2, 3): 惩罚 -6.0 (大奖正上方门槛)。
   - 流沙泥潭 (Quicksand Swamp) (2, 1): 惩罚 -3.0 (中等减速带)。
3. 标准单步物理成本: -1.0。
"""
from typing import Dict, List, Tuple, Optional

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "UP (↑)", DOWN: "DOWN (↓)", LEFT: "LEFT (←)", RIGHT: "RIGHT (→)"}
ACTION_SYMBOLS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}

class ComplexGridWorld:
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        gamma: float = 0.9,
        step_cost: float = -1.0
    ):
        """
        初始化复杂网格环境.
        """
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.step_cost = step_cost
        
        # 1. 终点吸收态字典映射 {坐标: 终止奖励}
        self.terminal_states: Dict[Tuple[int, int], float] = {
            (0, 3): 3.0,   # 次级目标 (落袋为安出口)
            (3, 3): 10.0   # 终极大奖 (最高回报中心)
        }
        
        # 2. 危险陷阱惩罚字典映射 {坐标: 踏入惩罚}
        self.hazard_states: Dict[Tuple[int, int], float] = {
            (1, 2): -10.0, # 致命熔岩坑
            (2, 3): -6.0,  # 尖刺陷阱
            (2, 1): -3.0   # 流沙泥潭
        }
        
        # 兼容性别名属性
        self.trap_states = self.hazard_states
        self.goal_state = (3, 3)
        self.subgoal_state = (0, 3)
        
        # 全状态集 S
        self.states = [(r, c) for r in range(rows) for c in range(cols)]
        self.current_state = (0, 0)
        
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """判定状态是否为终点吸收态 (0, 3) 或 (3, 3)。"""
        return state in self.terminal_states
        
    def get_transitions(self, state: Tuple[int, int], action: int) -> List[Tuple[float, Tuple[int, int], float]]:
        r"""
        【Model-Based 转移概率分布接口】
        供动态规划算法查询转移概率 P(s'|s,a) 与预期奖励。
        """
        if self.is_terminal(state):
            # 终点吸收态自环且无额外奖励
            return [(1.0, state, 0.0)]
            
        r, c = state
        if action == UP:
            next_r, next_c = max(r - 1, 0), c
        elif action == DOWN:
            next_r, next_c = min(r + 1, self.rows - 1), c
        elif action == LEFT:
            next_r, next_c = r, max(c - 1, 0)
        elif action == RIGHT:
            next_r, next_c = r, min(c + 1, self.cols - 1)
        else:
            raise ValueError(f"Invalid action {action}")
            
        next_state = (next_r, next_c)
        
        # 多级奖励判定:
        if next_state in self.terminal_states:
            reward = self.terminal_states[next_state]
        elif next_state in self.hazard_states:
            reward = self.hazard_states[next_state]
        else:
            reward = self.step_cost
            
        return [(1.0, next_state, reward)]
        
    def reset(self, start_state: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
        """【Model-Free 采样接口】将智能体放置到指定起点（默认 (0, 0)）。"""
        self.current_state = start_state
        return self.current_state
        
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        【Model-Free 采样接口】执行一步物理动作，观测返回 (下一状态, 奖励, 结束标识)。
        """
        transitions = self.get_transitions(self.current_state, action)
        _, next_state, reward = transitions[0]
        self.current_state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

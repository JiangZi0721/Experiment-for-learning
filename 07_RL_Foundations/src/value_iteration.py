"""
价值迭代算法白盒级教学实现 (Value Iteration White-box Implementation)
基于贝尔曼最优算子 (Bellman Optimality Operator) 直接通过连续逼近求解最优价值函数 V*(s) 与最优策略 π*(s)。

配备保姆级数学公式推导、GPI 单步截断本质对比与逐行白盒教学注释。
"""
import os
import sys
import pathlib

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Sandbox pathlib monkeypatch
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

from typing import Dict, List, Tuple
import numpy as np
from src.grid_world import GridWorld, ACTIONS, ACTION_NAMES, ACTION_SYMBOLS, UP, DOWN, LEFT, RIGHT

class ValueIteration:
    r"""
    价值迭代算法类 (Value Iteration, Bellman 1957)
    
    【理论核心与数学更新公式】:
    价值迭代不显式维护策略 π，而是直接将【贝尔曼最优方程】当作迭代算子:
    
        V_{k+1}(s) = max_{a ∈ A} ∑_{s', r} P(s', r | s, a) [ r + γ · V_k(s') ]
        
    【与策略迭代 (Policy Iteration) 的深层本质对比】:
    1. 策略迭代是“完全评估 + 贪婪改进”:
       - 必须在内部循环中反复扫描网格数十次，直到当前策略的价值 V^π 彻底收敛，才进行一次策略调整。
    2. 价值迭代是“单步截断评估 + 即时贪婪改进” (GPI 的极限):
       - 价值迭代将策略评估截断为【仅执行 1 次网格扫描】！在这一步扫描中，直接对动作取 MAX 算子；
       - 它跳过了中间策略的冗余评估，直接在价值空间中沿着贝尔曼最优超平面极速逼近全局最优。
    3. 最优策略在哪一步产生？
       - 价值迭代在迭代过程中完全不需要保存策略矩阵！
       - 最优策略 π*(s) 是在价值函数 V_k 收敛到 V* 之后，**在最后一步通过贪婪提取一锤定音生成的**。
    """
    def __init__(self, env: GridWorld, theta: float = 1e-4, max_iters: int = 1000):
        """
        初始化价值迭代器.
        
        参数:
            env (GridWorld): 网格环境
            theta (float): 终止判定阈值 θ。当全图最大变动 max_s |V_{k+1}(s) - V_k(s)| < θ 时停止
            max_iters (int): 最大迭代步数上限
        """
        self.env = env
        self.theta = theta
        self.max_iters = max_iters
        
        # 初始状态价值全设为 0.0
        self.V = {s: 0.0 for s in self.env.states}
        self.history = []

    def _extract_policy(self, V: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], Dict[int, float]]:
        """
        根据当前状态价值函数 V(s) 提取贪婪策略 (Greedy Policy) 分布.
        若存在多个动作 Q 值并列最大，则在平局动作间均匀分配概率 (Tie-breaking).
        """
        policy = {}
        for s in self.env.states:
            if self.env.is_terminal(s):
                policy[s] = {a: 0.25 for a in ACTIONS}
                continue
            q_values = {}
            for a in ACTIONS:
                t = self.env.get_transitions(s, a)[0]
                q_values[a] = t[2] + self.env.gamma * V[t[1]]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if abs(q - max_q) < 1e-6]
            prob = 1.0 / len(best_actions)
            policy[s] = {a: (prob if a in best_actions else 0.0) for a in ACTIONS}
        return policy

    def run(self, verbose_states: List[Tuple[int, int]] = None) -> List[Dict]:
        """
        执行价值迭代主循环.
        
        参数:
            verbose_states (List[Tuple[int, int]]): 重点打印计算展开细节的状态坐标
            
        返回:
            List[Dict]: 迭代历史轨迹 (记录每步的 V 矩阵、贪婪策略 pi 与误差 delta)
        """
        if verbose_states is None:
            verbose_states = [(0, 0), (1, 1), (1, 2), (2, 2), (3, 2)]
            
        print("\n" + "="*75)
        print("  >>> [价值迭代 Value Iteration] 算法开始运行")
        print("="*75)
        print(f"收敛阈值 theta = {self.theta}, 折扣因子 gamma = {self.env.gamma}")
        
        # 记录初始状态 k=0 (全 0 价值矩阵)
        self.history.append({
            "iter": 0,
            "V": self.V.copy(),
            "pi": self._extract_policy(self.V),
            "delta": 1.0
        })
        print(f"\n--- [价值迭代 初始步 k = 0] ---")
        print(f"全图状态初始价值 V_0(s) 恒等于 0.0000")
        
        iteration = 1
        while iteration <= self.max_iters:
            delta = 0.0
            new_V = self.V.copy()
            
            print(f"\n--- [价值迭代 迭代步 k = {iteration}] 贝尔曼最优算子展开 ---")
                
            # 遍历全图所有状态 s
            for s in self.env.states:
                # 终止状态未来价值严格恒为 0
                if self.env.is_terminal(s):
                    new_V[s] = 0.0
                    continue
                    
                # 步骤 1: 计算当前状态下采取各个动作的动作价值 Q(s, a)
                q_values = {}
                for a in ACTIONS:
                    transitions = self.env.get_transitions(s, a)
                    q_val = 0.0
                    for prob_trans, next_s, r in transitions:
                        q_val += prob_trans * (r + self.env.gamma * self.V[next_s])
                    q_values[a] = q_val
                    
                # 步骤 2: 【贝尔曼最优核心算子】直接选取最大的 Q 值作为新的状态价值
                max_q = max(q_values.values())
                
                # 记录全图最大变动量
                delta = max(delta, abs(new_V[s] - max_q))
                new_V[s] = max_q
                
                # 白盒透视打印 (前两步详细展开，后续步打印关键状态概览)
                if iteration <= 2 and s in verbose_states:
                    print(f"\n[状态 {s}] 贝尔曼最优更新:")
                    for a in ACTIONS:
                        t = self.env.get_transitions(s, a)[0]
                        print(f"  动作 {ACTION_NAMES[a]}: r={t[2]:.1f} + {self.env.gamma} * V_{iteration-1}({t[1]}={self.V[t[1]]:.4f}) = {q_values[a]:.4f}")
                    print(f"  => V_{iteration}({s}) = max_a Q(s, a) = {max_q:.4f} (变化: {abs(new_V[s] - self.V[s]):.4f})")
                    
            if iteration > 2:
                # 打印重点状态在当前步的演变
                summary_items = []
                for s in verbose_states:
                    if not self.env.is_terminal(s):
                        summary_items.append(f"V({s})={new_V[s]:.4f}")
                print(f"  关键状态价值: {', '.join(summary_items)} | 最大更新误差 Delta = {delta:.6f}")
            else:
                print(f"  => 本轮网格全量更新完成，全图最大变动 Delta_{iteration} = {delta:.6f}")
                    
            # 状态价值整体同步推进更新
            self.V = new_V
            self.history.append({
                "iter": iteration,
                "V": self.V.copy(),
                "pi": self._extract_policy(self.V),
                "delta": delta
            })
            
            # 收敛判据: 若最大变动已低于阈值 theta，宣告价值迭代收敛
            if delta < self.theta:
                print(f"\n价值迭代收敛! 耗费 {iteration} 步迭代, 最终误差 Delta = {delta:.6e} < {self.theta}")
                break
                
            iteration += 1
                
        # 步骤 3: 从最终收敛的最优价值函数 V*(s) 中提取最优确定性策略 π*(s)
        optimal_policy = {}
        for s in self.env.states:
            if self.env.is_terminal(s):
                continue
            q_values = {}
            for a in ACTIONS:
                t = self.env.get_transitions(s, a)[0]
                q_values[a] = t[2] + self.env.gamma * self.V[t[1]]
            best_a = max(q_values, key=q_values.get)
            optimal_policy[s] = best_a
            
        print("\n" + "="*75)
        print(f"  🎯 价值迭代最优策略提取完成 (共经历 {len(self.history)-1} 轮贝尔曼最优扫描)")
        print("="*75)
        return self.history

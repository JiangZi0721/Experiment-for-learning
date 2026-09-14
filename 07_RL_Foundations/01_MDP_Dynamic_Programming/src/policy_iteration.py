"""
策略迭代算法白盒级教学实现 (Policy Iteration White-box Implementation)
包含核心双阶段:
1. 策略评估 (Policy Evaluation): 基于贝尔曼期望方程迭代求解当前策略的状态价值 V^π(s)
2. 策略改进 (Policy Improvement): 依据动作价值 Q^π(s, a) 贪婪更新策略得到更优策略 π'

配备保姆级公式推导、不动点收缩映射证明与逐行白盒教学日志。
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from src.grid_world import GridWorld, ACTIONS, ACTION_NAMES, ACTION_SYMBOLS, UP, DOWN, LEFT, RIGHT

class PolicyIteration:
    r"""
    策略迭代算法类 (Policy Iteration, Howard 1960, Sutton & Barto Chapter 4.1-4.3)
    
    【核心哲学与广义策略迭代 (GPI)】:
    策略迭代通过交替进行“评估”与“改进”两个对立统一的过程，逼近全局最优解:
    
        π_0  ──(评估 E)──>  V^{π_0}  ──(改进 I)──>  π_1  ──(评估 E)──>  V^{阶段1} ... ──> π*
        
    - 策略评估阶段 (E): 策略保持固定，让价值函数 V(s) 充分向当前策略的真值收敛;
    - 策略改进阶段 (I): 价值函数保持固定，利用贪婪算子产生一个严格更优或至少不差的全新策略 π'。
    """
    def __init__(self, env: GridWorld, theta: float = 1e-4, max_eval_iters: int = 1000):
        """
        初始化策略迭代器.
        
        参数:
            env (GridWorld): 网格物理环境实例 (提供状态空间、动作空间与转移模型)
            theta (float): 策略评估的终止精度阈值 θ。当全网格单轮扫描的最大价值变动 max_s |V_{k+1}(s) - V_k(s)| < θ 时停止
            max_eval_iters (int): 策略评估的最大内部扫描轮数上限 (防止无穷死循环)
        """
        self.env = env
        self.theta = theta
        self.max_eval_iters = max_eval_iters
        
        # 1. 状态价值函数 V(s) 初始化:
        # 在动态规划中，通常将所有状态价值初始化为 0.0。
        # 依据巴拿赫不动点定理，无论初值为何，柯西收缩映射均保证其收敛到唯一真实解。
        self.V = {s: 0.0 for s in self.env.states}
        
        # 2. 策略 π(a|s) 初始化:
        # 采用均匀随机策略作为起点 (每个动作被选择的概率均为 1 / |A| = 0.25)
        # 存储结构: 字典套字典 self.pi[state][action] = 概率
        self.pi = {}
        for s in self.env.states:
            self.pi[s] = {a: 0.25 for a in ACTIONS}
                
        # 历史记录，用于绘制收敛曲线与策略演化热力图
        self.history = []
        
    def evaluate_policy(self, outer_iter: int, verbose_states: List[Tuple[int, int]]) -> Tuple[int, List[float]]:
        r"""
        阶段一: 策略评估 (Policy Evaluation)
        
        【理论核心与数学公式】:
        在策略 π 保持不变的情况下，利用【贝尔曼期望方程】作为更新算子，对全网格反复扫描:
        
            V_{k+1}(s) = ∑_{a ∈ A} π(a|s) ∑_{s', r} P(s', r | s, a) [ r + γ · V_k(s') ]
            
        【数学收敛性保证 (Banach Contraction Mapping Theorem)】:
        贝尔曼期望算子 T^π 是一个无穷范数下的 γ-压缩映射:
            ||T^π u - T^π v||_∞ ≤ γ · ||u - v||_∞
        由于折扣因子 γ = 0.9 < 1，根据柯西收敛准则，经过有限次迭代 (Sweep)，V_k 必然以几何级数速度
        收敛到唯一的线性方程组不动点解 V^π。
        
        参数:
            outer_iter (int): 当前宏观策略迭代的轮数 k (0, 1, 2...)
            verbose_states (List[Tuple[int, int]]): 挑选出来需要打印单步代数代入细节的关键状态列表
            
        返回:
            Tuple[int, List[float]]: (完成评估所用的扫描总次数 sweeps, 每一次扫描的最大绝对误差 delta 列表)
        """
        sweep = 0
        deltas = []
        
        print(f"\n{'='*75}")
        print(f"  >>> [策略评估阶段 Policy Evaluation] 轮次: k = {outer_iter}")
        print(f"{'='*75}")
        print(f"收敛阈值 theta = {self.theta}, 折扣因子 gamma = {self.env.gamma}")
        
        while sweep < self.max_eval_iters:
            delta = 0.0
            new_V = self.V.copy()
            
            # 为了白盒透视展示，仅在第 1、2 次扫描打印具体代数计算过程
            print_arithmetic = (sweep == 0 or sweep == 1)
            if print_arithmetic:
                print(f"\n--- [评估内部 Sweep {sweep + 1}] 关键状态贝尔曼期望方程展开 ---")
                
            # 全网格扫描 (Full Sweep): 遍历状态空间中的每一个状态 s
            for s in self.env.states:
                # 边界条件: 终止状态 (Terminal State) 的未来价值恒为 0
                if self.env.is_terminal(s):
                    new_V[s] = 0.0
                    continue
                    
                # 贝尔曼期望方程右端项累加:
                # V(s) = sum_a pi(a|s) * Q(s, a)
                v_expected = 0.0
                action_breakdown = []
                
                for a in ACTIONS:
                    prob_a = self.pi[s][a] # 当前策略下选择动作 a 的概率
                    transitions = self.env.get_transitions(s, a) # 获取转移 (prob, next_s, r)
                    
                    # 计算动作价值 Q(s, a) = sum_{s', r} P(s', r|s, a) * [r + gamma * V(s')]
                    q_val = 0.0
                    for prob_trans, next_s, r in transitions:
                        q_val += prob_trans * (r + self.env.gamma * self.V[next_s])
                        
                    # 期望价值累加: 乘以该动作的发生概率 pi(a|s)
                    v_expected += prob_a * q_val
                    action_breakdown.append((a, prob_a, q_val, transitions[0][1], transitions[0][2]))
                    
                # 记录本轮扫描中全图最大的价值改变量 (衡量是否达到柯西收敛)
                delta = max(delta, abs(new_V[s] - v_expected))
                new_V[s] = v_expected
                
                # 白盒输出所选状态的计算展开
                if print_arithmetic and s in verbose_states:
                    print(f"\n[状态 {s}] 计算透视:")
                    print(f"  公式: V(s) = sum_a pi(a|s) * [R(s,a,s') + gamma * V_old(s')]")
                    for a, prob_a, q_val, next_s, r in action_breakdown:
                        v_next = self.V[next_s]
                        print(f"    动作 {ACTION_NAMES[a]}: pi={prob_a:.2f} | 转移至 {next_s} | 奖励 r={r:.1f} | "
                              f"计算: {r:.1f} + {self.env.gamma} * {v_next:.4f} = {q_val:.4f} "
                              f"(加权贡献: {prob_a * q_val:.4f})")
                    print(f"  => 新价值 V_new({s}) = {v_expected:.4f} (旧价值: {self.V[s]:.4f}, 变化: {abs(new_V[s] - self.V[s]):.4f})")
                    
            # 采用同步更新 (Jacobi 式)，将全图计算好的新价值覆盖到旧矩阵
            self.V = new_V
            deltas.append(delta)
            sweep += 1
            
            # 收敛判据: 若单次全网格扫描的最大绝对误差已经小于阈值 theta，判定当前价值函数已精确拟合
            if delta < self.theta:
                print(f"\n策略评估完成! 历经 {sweep} 次全网格扫描 (Sweeps), 最终最大误差 Delta = {delta:.6e} < {self.theta}")
                break
                
        return sweep, deltas

    def improve_policy(self, outer_iter: int, verbose_states: List[Tuple[int, int]]) -> bool:
        r"""
        阶段二: 策略改进 (Policy Improvement)
        
        【理论核心与数学公式】:
        根据上一阶段精确求出的状态价值函数 V^π(s)，利用贪婪算子直接选取能使得动作价值 Q^π(s, a) 最大的动作:
        
            Q^π(s, a) = ∑_{s', r} P(s', r | s, a) [ r + γ · V^π(s') ]
            
            π'(s) = argmax_{a ∈ A} Q^π(s, a)
            
        【策略改进定理 (Policy Improvement Theorem)】:
        设 π 和 π' 为任意一对确定性策略，若对所有的状态 s 满足:
            q_π(s, π'(s)) ≥ v_π(s)
        则该策略必然在全局满足:
            v_{π'}(s) ≥ v_π(s)
        这严格保证了策略在每次改进后【绝不可能变差】！
        
        【终止条件 (Policy Stability)】:
        当贪婪改进出的新策略 π' 与旧策略 π 完全相同时，说明贝尔曼最优方程已经处处成立，
        算法即可宣布完全收敛，跳出外层循环。
        
        参数:
            outer_iter (int): 当前宏观策略迭代轮次
            verbose_states (List[Tuple[int, int]]): 重点打印的状态
            
        返回:
            bool: policy_stable，若策略未发生任何变动返回 True，否则返回 False
        """
        print(f"\n{'='*75}")
        print(f"  >>> [策略改进阶段 Policy Improvement] 轮次: k = {outer_iter}")
        print(f"{'='*75}")
        
        policy_stable = True
        state_changes = []
        
        for s in self.env.states:
            if self.env.is_terminal(s):
                continue
                
            # 记录当前旧策略在此状态下的最高概率动作
            old_best_action = max(self.pi[s], key=self.pi[s].get)
            
            # 步骤 1: 计算该状态下所有可选动作的动作价值 Q(s, a)
            q_values = {}
            for a in ACTIONS:
                transitions = self.env.get_transitions(s, a)
                q_val = 0.0
                for prob_trans, next_s, r in transitions:
                    q_val += prob_trans * (r + self.env.gamma * self.V[next_s])
                q_values[a] = q_val
                
            # 步骤 2: 找出使得 Q 值最大的最优动作
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if np.isclose(q, max_q, atol=1e-8)]
            
            # 步骤 3: 构造新策略分布 (若存在多个并列最优动作，则等概率平分)
            new_pi_s = {a: 0.0 for a in ACTIONS}
            for a in best_actions:
                new_pi_s[a] = 1.0 / len(best_actions)
                
            # 步骤 4: 检查策略是否改变 (只要概率分布变动超过 1e-4，即判定策略未稳定)
            changed = any(abs(new_pi_s[a] - self.pi[s][a]) > 1e-4 for a in ACTIONS)
            if changed:
                policy_stable = False
                state_changes.append((s, old_best_action, best_actions[0]))
                
            # 打印所选状态或发生改动状态的 Q(s, a) 真实竞争情况
            if s in verbose_states or changed:
                old_act_str = ACTION_NAMES[old_best_action]
                new_act_str = "/".join([ACTION_NAMES[a] for a in best_actions])
                status = "🔄 策略更新" if changed else "✅ 策略保持"
                print(f"\n[状态 {s}] 动作价值 Q(s, a) 透视 ({status}):")
                for a in ACTIONS:
                    t = self.env.get_transitions(s, a)[0]
                    next_s, r = t[1], t[2]
                    v_next = self.V[next_s]
                    is_max = "★ [最优]" if a in best_actions else "  "
                    print(f"  {is_max} {ACTION_NAMES[a]}: r={r:.1f} + {self.env.gamma} * V({next_s}={v_next:.4f}) = Q(s, a) = {q_values[a]:.4f}")
                print(f"  决策转变: {old_act_str} ---> {new_act_str}")
                
            self.pi[s] = new_pi_s
            
        print(f"\n--- 策略稳定性检查 (Policy Stability) ---")
        if policy_stable:
            print(f"🎯 策略已完全稳定 (Policy is Stable)! 所有状态的最优动作均未改变。已收敛到全局最优策略 pi*！")
        else:
            print(f"⚠️ 本轮共有 {len(state_changes)} 个非终止状态发生动作改进，需要进入下一轮评估！")
            
        return policy_stable

    def run(self, verbose_states: List[Tuple[int, int]] = None):
        """
        执行完整的宏观策略迭代外层大循环 (E -> I -> E -> I ... 直至收敛).
        """
        if verbose_states is None:
            verbose_states = [(0, 1), (1, 1), (1, 2), (2, 2)]
            
        outer_iter = 0
        
        while True:
            # 步骤 1: 策略评估 (反复扫描直至当前策略价值函数完全拟合)
            eval_sweeps, deltas = self.evaluate_policy(outer_iter, verbose_states)
            
            # 保存当前轮次的快照供后续可视化使用
            self.history.append({
                "iter": outer_iter,
                "V": {s: self.V[s] for s in self.env.states},
                "pi": {s: {a: self.pi[s][a] for a in ACTIONS} for s in self.env.states},
                "sweeps": eval_sweeps,
                "deltas": deltas,
            })
            
            # 步骤 2: 策略改进 (根据评估出的价值函数贪婪调整动作)
            stable = self.improve_policy(outer_iter, verbose_states)
            
            outer_iter += 1
            if stable:
                break
                
        print(f"\n{'#'*75}")
        print(f"  🏆 策略迭代全流程圆满结束！共进行 {outer_iter} 轮宏观策略迭代。")
        print(f"{'#'*75}")
        return self.history

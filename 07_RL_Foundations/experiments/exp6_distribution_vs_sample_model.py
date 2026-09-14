"""
实验六: 样本模型 (Sample Model) 与 分布模型 (Distribution Model) 深度实证对比
核心实证维度:
1. 分支因子 b 与算力预算效率 (Sutton & Barto Sec 8.7 Fig 8.7 经典曲线精确复现 + 宏观 MDP 规划收敛)
2. 高分支组合扩展性下的挂钟时间 (分支因子 b 从 10 到 5000 的 O(b) vs O(1) 膨胀实测)
3. 罕见高危黑天鹅事件探测与安全性 (极小概率致命陷阱 p=0.5% 下的虚假乐观与遗憾值)
4. 生成 4 组出版级对比分析图表与综合决策大盘
"""
import os
import sys
import time
import pathlib
import numpy as np

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_cache")

from src.model_comparison import (
    compute_sutton_fig87_curves,
    SuttonBranchingMDP,
    CombinatorialScalingMDP,
    RareTrapMDP
)
from src.plot_model_comparison import (
    plot_sutton_branching_experiment,
    plot_combinatorial_time_experiment,
    plot_rare_trap_experiment,
    plot_master_dashboard
)

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_sutton_experiment(b_list=[2, 10, 50, 200], num_states=50, max_work_units=8000, num_seeds=10):
    """
    Sub-Experiment 1: Branching Factor vs. Work Units
    Part A: Sutton Fig 8.7 Analytical Exact Replication
    Part B: Macro-MDP Planning Convergence
    """
    print("\n" + "="*80)
    print("  [子实验一] 分支因子 b 对计算预算效率的影响 (Sutton & Barto Sec 8.7)")
    print("="*80)
    
    # 1. Sutton Fig 8.7 Analytical
    print(">>> 正在生成 Sutton & Barto 经典教材 Fig 8.7 精确解析曲线 (b = 1, 2, 3, 10, 100, 1000)...")
    sutton_fig_data = compute_sutton_fig87_curves(b_values=[1, 2, 3, 10, 100, 1000], max_units=100)
    
    # 2. Macro-MDP Planning
    print(f">>> 正在运行多状态宏观 MDP 规划收敛实验 (S={num_states}, 测试分支 b={b_list})...")
    mdp_data = {}
    num_eval_points = 40
    work_units_eval = np.linspace(num_states * 2, max_work_units, num_eval_points, dtype=int)
    
    for b in b_list:
        mdp = SuttonBranchingMDP(num_states=num_states, branching_factor=b, seed=42)
        true_V = mdp.true_V
        
        # Expected updates curve
        sweep_work_cost = num_states * 2 * b
        max_sweeps = int(max_work_units / sweep_work_cost) + 2
        
        V_exp = np.zeros(num_states)
        exp_history_wu = [0]
        exp_history_rmse = [np.sqrt(np.mean((V_exp - true_V)**2))]
        
        current_wu = 0
        for _ in range(1, max_sweeps + 1):
            V_new = np.zeros(num_states)
            for s in range(num_states):
                q_vals = []
                for a in range(2):
                    next_s, r, probs = mdp.get_transition_distribution(s, a)
                    q_vals.append(np.sum(probs * (r + mdp.gamma * V_exp[next_s])))
                V_new[s] = np.max(q_vals)
            V_exp = V_new
            current_wu += sweep_work_cost
            exp_history_wu.append(current_wu)
            exp_history_rmse.append(np.sqrt(np.mean((V_exp - true_V)**2)))
            if current_wu >= max_work_units:
                break
                
        expected_rmse_interp = np.interp(work_units_eval, exp_history_wu, exp_history_rmse)
        
        # Sample updates curve across seeds
        sample_rmse_all_seeds = []
        for seed_idx in range(num_seeds):
            rng = np.random.RandomState(2000 + seed_idx)
            Q_samp = np.zeros((num_states, 2))
            counts = np.zeros((num_states, 2))
            
            samp_history_wu = [0]
            samp_history_rmse = [np.sqrt(np.mean((np.max(Q_samp, axis=1) - true_V)**2))]
            
            check_interval = int(max_work_units / num_eval_points)
            
            for step in range(1, max_work_units + 1):
                s = rng.randint(0, num_states)
                a = rng.randint(0, 2)
                counts[s, a] += 1
                
                s_next, r = mdp.sample_transition(s, a)
                alpha = 1.0 / (counts[s, a] ** 0.6)
                target = r + mdp.gamma * np.max(Q_samp[s_next])
                Q_samp[s, a] += alpha * (target - Q_samp[s, a])
                
                if step % check_interval == 0 or step == max_work_units:
                    V_samp = np.max(Q_samp, axis=1)
                    rmse = np.sqrt(np.mean((V_samp - true_V)**2))
                    samp_history_wu.append(step)
                    samp_history_rmse.append(rmse)
                    
            sample_rmse_interp = np.interp(work_units_eval, samp_history_wu, samp_history_rmse)
            sample_rmse_all_seeds.append(sample_rmse_interp)
            
        sample_rmse_all_seeds = np.array(sample_rmse_all_seeds)
        sample_mean = np.mean(sample_rmse_all_seeds, axis=0)
        sample_std = np.std(sample_rmse_all_seeds, axis=0)
        
        mdp_data[b] = {
            'work_units': work_units_eval,
            'expected_rmse': expected_rmse_interp,
            'sample_rmse_mean': sample_mean,
            'sample_rmse_std': sample_std
        }
        print(f"    [b={b:3d}] 计算预算 {max_work_units} WU 完成 | 最终 Expected RMSE={expected_rmse_interp[-1]:.4f} | Sample RMSE={sample_mean[-1]:.4f} ± {sample_std[-1]:.4f}")

    return sutton_fig_data, mdp_data


def run_combinatorial_time_experiment():
    """
    Sub-Experiment 2: Combinatorial & High Branching Scalability
    Evaluates real wall-clock sweep time as branching factor b grows from 10 to 5000.
    """
    print("\n" + "="*80)
    print("  [子实验二] 高分支组合空间下的真实挂钟耗时对比 (b: 10 -> 5000)")
    print("="*80)
    
    b_values = [10, 50, 200, 1000, 5000]
    num_states = 20
    
    dist_times_ms = []
    sample_times_ms = []
    
    for b in b_values:
        env = CombinatorialScalingMDP(num_states=num_states, branching_factor=b, seed=42)
        V_dummy = np.zeros(num_states)
        
        # JIT / Cache Warmup
        for _ in range(5):
            _ = env.expected_sweep(V_dummy)
            _ = env.sample_sweep(V_dummy)
        
        # Measure Distribution Model (Expected Sweep: O(S * A * b))
        n_repeats = 20 if b <= 1000 else 10
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = env.expected_sweep(V_dummy)
        t1 = time.perf_counter()
        dist_ms = ((t1 - t0) / n_repeats) * 1000.0
        dist_times_ms.append(dist_ms)
        
        # Measure Sample Model (Sample Sweep: O(S * A * 1))
        t2 = time.perf_counter()
        for _ in range(n_repeats):
            _ = env.sample_sweep(V_dummy)
        t3 = time.perf_counter()
        samp_ms = ((t3 - t2) / n_repeats) * 1000.0
        sample_times_ms.append(samp_ms)
        
        speedup = dist_ms / max(samp_ms, 1e-6)
        print(f"    分支因子 b = {b:5d}: 分布模型耗时 {dist_ms:8.3f} ms | 样本模型耗时 {samp_ms:8.3f} ms | 样本加速比: {speedup:7.1f}x")

    return {
        'branching_factors': b_values,
        'dist_times_ms': dist_times_ms,
        'sample_times_ms': sample_times_ms
    }


def run_rare_trap_experiment(p_trap=0.005, trap_penalty=-500.0, num_trials=5000):
    """
    Sub-Experiment 3: Rare Disaster / Catastrophic Trap MDP
    Evaluates probability of miss and wrong policy rate vs sample budget N.
    """
    print("\n" + "="*80)
    print("  [子实验三] 罕见高危黑天鹅事件探测与安全性 (p_trap = 0.5%, penalty = -500)")
    print("="*80)
    
    budgets = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    mdp = RareTrapMDP(p_trap=p_trap, trap_penalty=trap_penalty, seed=42)
    
    true_q_safe = 1.0
    true_q_risky = (1 - p_trap) * 2.0 + p_trap * trap_penalty
    
    trap_miss_prob = []
    wrong_decision_rate = []
    empirical_regret = []
    
    print(f"理论期望真值: Q(安全路线) = {true_q_safe:+.2f} | Q(高危高回报路线) = {true_q_risky:+.2f}")
    print(f"理论最优决策: 严格选择安全路线 (Action 0)")
    print(f"分布模型表现: 解析期望直接识破陷阱，决策错误率恒为 0.0%，安全遗憾值 = 0.00")
    print(f"{'样本预算 N':>10} | {'理论未踩雷概率':>14} | {'样本模型错误决策率':>18} | {'期望遗憾值 (Regret)':>22}")
    print("-" * 72)
    
    for N in budgets:
        theo_miss = (1.0 - p_trap) ** N
        trap_miss_prob.append(theo_miss)
        
        rng = np.random.RandomState(42 + N)
        wrong_decisions = 0
        for _ in range(num_trials):
            draws = rng.rand(N)
            trapped_count = np.sum(draws < p_trap)
            
            if trapped_count == 0:
                sample_q_risky = 2.0
            else:
                sample_q_risky = ((N - trapped_count) * 2.0 + trapped_count * trap_penalty) / N
                
            sample_q_safe = 1.0
            if sample_q_risky > sample_q_safe:
                wrong_decisions += 1
                
        err_rate = wrong_decisions / num_trials
        wrong_decision_rate.append(err_rate)
        regret = err_rate * (true_q_safe - true_q_risky)
        empirical_regret.append(regret)
        
        print(f"{N:10d} | {theo_miss*100:13.2f}% | {err_rate*100:17.2f}% | {regret:22.4f}")

    return {
        'budgets': budgets,
        'trap_miss_prob': trap_miss_prob,
        'wrong_decision_rate': wrong_decision_rate,
        'empirical_regret': empirical_regret
    }


def main():
    log_path = os.path.join(PROJECT_ROOT, "logs", "model_comparison_results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    dual_logger = DualLogger(log_path)
    sys.stdout = dual_logger
    
    print("="*85)
    print("  强化学习环境建模实证体系：分布模型 (Distribution Model) vs 样本模型 (Sample Model)")
    print("  全维度对照基准评测 (Sutton 曲线复现 / 组合爆炸耗时 / 极端风险盲区)")
    print("="*85)
    
    # 1. Run Sutton branching experiment
    sutton_fig_data, mdp_data = run_sutton_experiment(b_list=[2, 10, 50, 200], num_states=50, max_work_units=8000, num_seeds=10)
    
    # 2. Run Combinatorial scaling experiment
    comb_res = run_combinatorial_time_experiment()
    
    # 3. Run Rare Trap experiment
    rare_res = run_rare_trap_experiment(p_trap=0.005, trap_penalty=-500.0, num_trials=5000)
    
    # 4. Generate Visualizations
    img_dir = os.path.join(PROJECT_ROOT, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    p1 = os.path.join(img_dir, "distribution_vs_sample_branching_budget.png")
    p2 = os.path.join(img_dir, "distribution_vs_sample_combinatorial_time.png")
    p3 = os.path.join(img_dir, "distribution_vs_sample_rare_trap.png")
    p4 = os.path.join(img_dir, "distribution_vs_sample_comprehensive_dashboard.png")
    
    print("\n" + "="*80)
    print("  [绘图引擎] 生成 4 组出版级对比分析图表...")
    print("="*80)
    plot_sutton_branching_experiment(sutton_fig_data, mdp_data, p1)
    plot_combinatorial_time_experiment(comb_res, p2)
    plot_rare_trap_experiment(rare_res, p3)
    plot_master_dashboard(sutton_fig_data, comb_res, rare_res, p4)
    
    print("\n" + "="*85)
    print("  所有实验顺利完成！图表与实验日志已写入 images/ 与 logs/ 目录。")
    print("="*85)


if __name__ == "__main__":
    main()

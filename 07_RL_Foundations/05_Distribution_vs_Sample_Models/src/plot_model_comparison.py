"""
Visualization module for Distribution Model vs Sample Model comparison.
Generates clear, publication-quality figures illustrating the trade-offs:
1. Sutton & Barto Fig 8.7 Classic Replication & Macro-MDP Efficiency
2. Wall-Clock Time & Combinatorial Scaling (Log-scale)
3. Rare Catastrophe Blind Spots (Safety & Risk Profile)
4. Comprehensive 4-Panel Executive Dashboard
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Use clean, universal English styling to avoid font/glyph missing issues across platforms
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_sutton_branching_experiment(sutton_fig_data: dict, mdp_data: dict, save_path: str):
    """
    Sub-Experiment 1 Figure:
    Left: Exact Sutton & Barto (2018) Figure 8.7 replication (State value estimation error vs compute units)
    Right: Macro-MDP Planning convergence (RMSE vs total compute work units)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # --- Left: Sutton Fig 8.7 Exact Reproduction ---
    colors = {1: '#7F8C8D', 2: '#2980B9', 3: '#27AE60', 10: '#E67E22', 100: '#8E44AD', 1000: '#C0392B'}
    t_vals = np.arange(1, 101)
    
    for b in [1, 2, 3, 10, 100, 1000]:
        if b in sutton_fig_data:
            d = sutton_fig_data[b]
            # Expected update (step function)
            ax1.plot(d['t'], d['expected_error'], label=f'Expected (b={b})', 
                     color=colors[b], linestyle='-', linewidth=2.4)
            # Sample update (smooth 1/sqrt(t))
            if b > 1:
                ax1.plot(d['t'], d['sample_error'], label=f'Sample (b={b})', 
                         color=colors[b], linestyle='--', linewidth=1.8, alpha=0.85)
                
    ax1.set_xlim(1, 100)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_title("Sutton & Barto Fig 8.7 Reproduction\n(Estimation Error vs. Compute Units for Branching b)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Number of Computation Units (1 unit = 1 sample update)", fontsize=11)
    ax1.set_ylabel("RMS Error in State Value Estimate", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5, ncol=2, framealpha=0.95)
    
    ax1.annotate('b=100: Sample Update drops error\nto 0.1 at t=100 while Expected\nUpdate just finished 1st pass!', 
                 xy=(50, 0.14), xytext=(35, 0.45),
                 arrowprops=dict(facecolor='#8E44AD', shrink=0.08, width=1.5, headwidth=5),
                 fontsize=9, fontweight='bold', color='#8E44AD',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#8E44AD", lw=1))

    # --- Right: Macro-MDP Planning Convergence ---
    mdp_colors = {2: '#2980B9', 10: '#27AE60', 50: '#E67E22', 200: '#C0392B'}
    for b in [2, 10, 50, 200]:
        if b in mdp_data:
            d = mdp_data[b]
            wu = d['work_units']
            ax2.plot(wu, d['expected_rmse'], label=f'Expected VI (b={b})', 
                     color=mdp_colors[b], linestyle='-', linewidth=2.5)
            ax2.plot(wu, d['sample_rmse_mean'], label=f'Sample VI (b={b})', 
                     color=mdp_colors[b], linestyle='--', linewidth=1.8, alpha=0.85)
            ax2.fill_between(wu, d['sample_rmse_mean'] - d['sample_rmse_std'], 
                             d['sample_rmse_mean'] + d['sample_rmse_std'], 
                             color=mdp_colors[b], alpha=0.12)
            
    ax2.set_title("Macro-MDP Planning Convergence\n(Value Iteration RMSE vs. Total Computation Work Units)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Total Computation Work Units (1 unit = 1 transition evaluated)", fontsize=11)
    ax2.set_ylabel("RMSE relative to True V*", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] Sutton & Macro-MDP Comparison -> {save_path}")


def plot_combinatorial_time_experiment(results: dict, save_path: str):
    """
    Sub-Experiment 2 Figure:
    Wall-Clock Time vs Branching Factor b (Log-Log scale)
    Demonstrates strict O(b) linear explosion for Distribution Model vs O(1) flat scaling for Sample Model.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    b_values = results['branching_factors']
    dist_times = results['dist_times_ms']
    sample_times = results['sample_times_ms']
    
    # Left: Absolute Wall-Clock Time per Sweep (ms)
    ax1.plot(b_values, dist_times, marker='s', linewidth=2.8, markersize=8, 
             color='#C0392B', label='Distribution Model: Full Expectation Sweep (O(b))')
    ax1.plot(b_values, sample_times, marker='o', linewidth=2.8, markersize=8, 
             color='#27AE60', label='Sample Model: Simulation Rollout Sweep (O(1))')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("Wall-Clock Time per Sweep vs. Branching Factor b\n(Log-Log Scale Demonstration)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Branching Factor b (Outcomes per State-Action)", fontsize=11)
    ax1.set_ylabel("Execution Time per Sweep (ms, Log Scale)", fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Annotation for massive speedup
    final_speedup = dist_times[-1] / sample_times[-1]
    ax1.annotate(f"Speedup: {final_speedup:,.1f}x Faster!\n(Sample Model: {sample_times[-1]:.2f}ms\nvs. Distribution: {dist_times[-1]:.2f}ms)", 
                 xy=(b_values[-1], sample_times[-1]), 
                 xytext=(b_values[-1]/40.0, sample_times[-1]*2.2),
                 arrowprops=dict(facecolor='#27AE60', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight='bold', color='#27AE60',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#27AE60", lw=1.2))

    # Right: Relative Computational Growth Multiplier (Normalized to b_min)
    dist_mult = np.array(dist_times) / dist_times[0]
    samp_mult = np.array(sample_times) / sample_times[0]
    
    ax2.plot(b_values, dist_mult, marker='s', linewidth=2.8, markersize=8, 
             color='#C0392B', label='Distribution Model Complexity Growth')
    ax2.plot(b_values, samp_mult, marker='o', linewidth=2.8, markersize=8, 
             color='#27AE60', label='Sample Model Complexity Growth')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f"Complexity Multiplier relative to Baseline (b={b_values[0]})\n(Empirical Demonstration of Computational Scalability)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Branching Factor b (Log Scale)", fontsize=11)
    ax2.set_ylabel("Growth Multiplier relative to b_min", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] Combinatorial Time Comparison -> {save_path}")


def plot_rare_trap_experiment(results: dict, save_path: str):
    """
    Sub-Experiment 3 Figure:
    Rare Catastrophe Blind Spot and Risk Profile vs Sample Budget N.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    budgets = results['budgets']
    trap_miss_prob = results['trap_miss_prob']
    wrong_decision_rate = results['wrong_decision_rate']
    empirical_regret = results['empirical_regret']
    
    # Left: Detection Failure & Suboptimal Choice Rate
    ax1.plot(budgets, [p * 100 for p in trap_miss_prob], marker='o', color='#E67E22', 
             linewidth=2.5, markersize=6, label='Theoretical Miss Probability P(No Disaster in N Draws)')
    ax1.plot(budgets, [r * 100 for r in wrong_decision_rate], marker='x', color='#C0392B', 
             linewidth=2.5, markersize=8, linestyle='--', label='Empirical Wrong Policy Selection Rate (%)')
    
    ax1.axhline(0.0, color='#2980B9', linewidth=2.5, linestyle='-', 
                label='Distribution Model Error Rate (0.0% Exact Risk Accounting)')
    
    ax1.set_xscale('log')
    ax1.set_title("Dangerous Optimism in Rare-Event Scenarios (p=0.5%)\n(Sample Model Blind Spot vs. Distribution Model Exactness)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Simulation Sample Budget N per Action (Log Scale)", fontsize=11)
    ax1.set_ylabel("Error / Miss Rate (%)", fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9.5)

    # Right: Expected Policy Regret
    ax2.plot(budgets, empirical_regret, marker='s', color='#8E44AD', linewidth=2.5, markersize=6, 
             label='Expected Real-World Regret of Learned Policy')
    ax2.axhline(0.0, color='#2980B9', linewidth=2.5, linestyle='-', 
                label='Distribution Model Regret (= 0.00)')
    
    ax2.set_xscale('log')
    ax2.set_title("Expected Value Loss / Real-World Safety Regret\n(Severe Hazard if Deployed Under Incomplete Sampling)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Simulation Sample Budget N per Action (Log Scale)", fontsize=11)
    ax2.set_ylabel("Expected Value Loss relative to Optimal Policy", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] Rare Trap Experiment -> {save_path}")


def plot_master_dashboard(sutton_fig_data: dict, comb_res: dict, rare_res: dict, save_path: str):
    """
    Creates a 4-panel master executive dashboard integrating all key insights:
    Panel A: Sutton Fig 8.7 (State Error vs Computation Units)
    Panel B: Wall-Clock Execution Time under Combinatorial Scaling
    Panel C: Rare Catastrophe Blind Spot Curve
    Panel D: Executive Strategic Decision Matrix & Applicable Scenarios Table
    """
    fig = plt.figure(figsize=(19, 13), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
    
    # ---------------- Panel A: Sutton Fig 8.7 ----------------
    ax_a = fig.add_subplot(gs[0, 0])
    sutton_colors = {1: '#7F8C8D', 2: '#2980B9', 10: '#27AE60', 100: '#E67E22', 1000: '#C0392B'}
    for b in [1, 2, 10, 100, 1000]:
        if b in sutton_fig_data:
            d = sutton_fig_data[b]
            ax_a.plot(d['t'], d['expected_error'], color=sutton_colors[b], linestyle='-', linewidth=2.2, label=f'Expected (b={b})')
            if b > 1:
                ax_a.plot(d['t'], d['sample_error'], color=sutton_colors[b], linestyle='--', linewidth=1.8, alpha=0.85, label=f'Sample (b={b})')
                
    ax_a.set_xlim(1, 100)
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.set_title("Panel A: Algorithmic Efficiency vs. Branching Factor\n(Sutton & Barto Fig 8.7: Error vs. Compute Units)", 
                   fontsize=12, fontweight='bold')
    ax_a.set_xlabel("Computation Units (1 unit = 1 sample update)", fontsize=10)
    ax_a.set_ylabel("RMS Error in Value Estimate", fontsize=10)
    ax_a.grid(True, linestyle='--', alpha=0.5)
    ax_a.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.95)
    
    # ---------------- Panel B: Combinatorial Scaling ----------------
    ax_b = fig.add_subplot(gs[0, 1])
    b_factors = comb_res['branching_factors']
    dist_t = comb_res['dist_times_ms']
    samp_t = comb_res['sample_times_ms']
    
    ax_b.plot(b_factors, dist_t, marker='s', color='#C0392B', linewidth=2.6, label='Distribution Model: Full Expectation (O(b))')
    ax_b.plot(b_factors, samp_t, marker='o', color='#27AE60', linewidth=2.6, label='Sample Model: Simulation Rollout (O(1))')
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_title("Panel B: Execution Time under Combinatorial Explosion\n(Wall-Clock Sweep Time vs. Branching Factor b)", 
                   fontsize=12, fontweight='bold')
    ax_b.set_xlabel("Branching Factor b (Log Scale)", fontsize=10)
    ax_b.set_ylabel("Time per Sweep (ms, Log Scale)", fontsize=10)
    ax_b.grid(True, which="both", ls="--", alpha=0.5)
    ax_b.legend(loc='upper left', fontsize=9.5)
    
    # ---------------- Panel C: Rare Disaster Blind Spot ----------------
    ax_c = fig.add_subplot(gs[1, 0])
    budgets = rare_res['budgets']
    wrong_dec = [r * 100 for r in rare_res['wrong_decision_rate']]
    ax_c.plot(budgets, wrong_dec, marker='o', color='#E74C3C', linewidth=2.5, 
              label='Sample Model: Selection Rate of Fatal Risky Action (%)')
    ax_c.axhline(0.0, color='#2980B9', linewidth=2.5, label='Distribution Model: Zero Error Rate (0.0%)')
    ax_c.set_xscale('log')
    ax_c.set_title("Panel C: Safety-Critical Blind Spot (Tail Risk p=0.5%)\n(Sample Model exhibits false optimism under finite budgets)", 
                   fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Simulation Sample Budget N per Action (Log Scale)", fontsize=10)
    ax_c.set_ylabel("Suboptimal Decision Rate (%)", fontsize=10)
    ax_c.set_ylim(-5, 105)
    ax_c.grid(True, which="both", ls="--", alpha=0.5)
    ax_c.legend(loc='upper right', fontsize=9.5)
    
    # ---------------- Panel D: Strategic Decision Matrix & Applicable Scenarios ----------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    
    col_labels = ["Application Domain", "Distribution Model", "Sample Model", "Recommended Selection"]
    table_data = [
        ["Small Discrete\n(b <= 5, GridWorld)", "Zero variance, rapid\nexact DP convergence", "Sampling noise, requires\nmany samples", "Distribution Model\n(Dynamic Programming)"],
        ["Large Branching\n(b >= 100, Combinatorial)", "Combinatorial explosion\nO(b) summation, slow", "O(1) step cost, scales\nacross wide state space", "Sample Model\n(Dyna-Q / Rollouts)"],
        ["Continuous Control\n& Physics Engines", "Intractable high-dim\nprobability integrals", "Trivial O(1) simulator\nstep execution", "Sample Model\n(MuJoCo / Gym / Isaac)"],
        ["Card Games / Logic\n(e.g., Blackjack, Poker)", "Tedious analytical\ncombinatorics to derive P", "Trivial black-box card\ndealing game rules", "Sample Model\n(Monte Carlo / MCTS)"],
        ["Safety-Critical &\nTail Risk (Finance, Auto)", "Captures rare disaster\np * Loss with 100% exactness", "Severe blind spots if rare\ncatastrophe is unvisited", "Distribution Model\n(Risk-Sensitive DP)"],
        ["Model Learning\n& Engineering Feasibility", "Hard to train/fit full\njoint P(s'|s,a) tensor", "Easy: neural net generator\nor physical simulator", "Sample Model\n(World Models / Dreamer)"]
    ]
    
    col_widths = [0.22, 0.28, 0.26, 0.24]
    table = ax_d.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 2.05)
    
    # Style Header
    for c_idx in range(4):
        table[0, c_idx].set_facecolor('#2C3E50')
        table[0, c_idx].set_text_props(color='white', weight='bold')
    
    # Style alternating rows
    row_bgs = ['#F8F9F9', '#FFFFFF'] * 3
    for r_idx in range(1, 7):
        bg = row_bgs[r_idx - 1]
        for c_idx in range(4):
            cell = table[r_idx, c_idx]
            cell.set_facecolor(bg)
            if c_idx == 3:
                is_dist = 'Distribution' in cell.get_text().get_text()
                cell.set_text_props(weight='bold', color='#1F618D' if is_dist else '#1E8449')
                
    ax_d.set_title("Panel D: Strategic Decision Matrix & Applicable Scenarios\n(Empirical Synthesis & Engineering Guidance)", 
                   fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle("Comparative Evaluation: Distribution Model vs. Sample Model in Reinforcement Learning", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] Master Dashboard -> {save_path}")

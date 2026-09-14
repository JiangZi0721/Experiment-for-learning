"""
Plotting script for Complex GridWorld experiments.
Generates:
1. Environment and Converged Policies overview (4 panels)
2. 6 dedicated step-by-step transition cards (Heatmap + Arrows + Formula Derivation sidebar)
3. Master combined 6-step overview figure
"""
import os
import pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Matplotlib configuration for Chinese and Math rendering
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

from src.complex_grid_world import ComplexGridWorld, ACTIONS, UP, DOWN, LEFT, RIGHT, ACTION_NAMES, ACTION_SYMBOLS

ACTION_OFFSETS = {
    UP: (0, 0.28),
    DOWN: (0, -0.28),
    LEFT: (-0.28, 0),
    RIGHT: (0.28, 0)
}

def draw_grid_heatmap(ax, values_grid, policy_grid=None, title="", highlight_trans=None, env=None):
    """
    Draws a 4x4 heatmap with cell annotations and internal policy arrows.
    highlight_trans: dict with {"s": (r, c), "a": int, "next_s": (r, c), "r": float}
    """
    rows, cols = 4, 4
    im = ax.imshow(values_grid, cmap='RdYlGn', origin='upper', aspect='equal')
    
    # Grid lines
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"C{c}" for c in range(cols)], fontsize=11, fontweight='bold')
    ax.set_yticklabels([f"R{r}" for r in range(rows)], fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    
    # Cell contents
    for r in range(rows):
        for c in range(cols):
            st = (r, c)
            val = values_grid[r, c]
            
            # Identify special states
            is_start = (st == (0, 0))
            is_subgoal = (st == (0, 3))
            is_maingoal = (st == (3, 3))
            is_lava = (st == (1, 2))
            is_spikes = (st == (2, 3))
            is_swamp = (st == (2, 1))
            
            # Badge text and value position
            badge = ""
            if is_start:
                badge = "START\n"
            elif is_subgoal:
                badge = "SUB-GOAL\n[+3.0]\n"
            elif is_maingoal:
                badge = "MAIN GOAL\n[+10.0]\n"
            elif is_lava:
                badge = "LAVA PIT\n[-10.0]\n"
            elif is_spikes:
                badge = "SPIKES\n[-6.0]\n"
            elif is_swamp:
                badge = "SWAMP\n[-3.0]\n"
                
            text_color = 'white' if (val < -3.5 or val > 6.0) else 'black'
            
            # Position text higher if there is an arrow below
            has_arrow = (policy_grid is not None and not (is_subgoal or is_maingoal))
            text_y = r - 0.15 if has_arrow else r
            
            ax.text(c, text_y, f"{badge}{val:+.2f}", ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)
                    
            # Draw policy arrow inside cell below the text
            if has_arrow:
                best_a = policy_grid[r][c]
                arrow_color = 'white' if (val < -3.5 or val > 6.0) else 'navy'
                edge_color = 'black' if arrow_color == 'white' else 'white'
                
                if best_a == UP:
                    ax.annotate('', xy=(c, r + 0.06), xytext=(c, r + 0.32),
                                arrowprops=dict(facecolor=arrow_color, edgecolor=edge_color, width=2.0, headwidth=7, shrink=0.05))
                elif best_a == DOWN:
                    ax.annotate('', xy=(c, r + 0.32), xytext=(c, r + 0.06),
                                arrowprops=dict(facecolor=arrow_color, edgecolor=edge_color, width=2.0, headwidth=7, shrink=0.05))
                elif best_a == LEFT:
                    ax.annotate('', xy=(c - 0.22, r + 0.20), xytext=(c + 0.22, r + 0.20),
                                arrowprops=dict(facecolor=arrow_color, edgecolor=edge_color, width=2.0, headwidth=7, shrink=0.05))
                elif best_a == RIGHT:
                    ax.annotate('', xy=(c + 0.22, r + 0.20), xytext=(c - 0.22, r + 0.20),
                                arrowprops=dict(facecolor=arrow_color, edgecolor=edge_color, width=2.0, headwidth=7, shrink=0.05))

    # Highlight specific transition if provided
    if highlight_trans:
        sr, sc = highlight_trans["s"]
        nr, nc = highlight_trans["next_s"]
        act_name = ACTION_NAMES[highlight_trans["a"]].split()[0]
        
        # Draw ring on source state
        circle = patches.Circle((sc, sr), 0.42, fill=False, edgecolor='cyan', linewidth=3.5, linestyle='--')
        ax.add_patch(circle)
        
        # Draw box on next state
        rect = patches.Rectangle((nc - 0.45, nr - 0.45), 0.9, 0.9, fill=False, edgecolor='magenta', linewidth=3.5)
        ax.add_patch(rect)
        
        # Draw transition vector
        ax.annotate(
            '', xy=(nc, nr), xytext=(sc, sr),
            arrowprops=dict(facecolor='gold', edgecolor='black', width=3.5, headwidth=11, shrink=0.15)
        )
        
    return im

def create_step_card(
    step_num: int,
    algo_name: str,
    event: dict,
    values_grid: np.ndarray,
    policy_grid: list,
    formula_latex: str,
    math_derivation: list,
    strategic_insight: str,
    save_path: str
):
    """
    Creates a dedicated high-resolution 2-panel figure:
    Left: Heatmap + Policy Arrows + Highlighted Transition
    Right: Mathematical formula, step-by-step arithmetic substitution, and RL insight.
    """
    fig = plt.figure(figsize=(15, 7), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2], wspace=0.25)
    
    # Left Panel: Grid Heatmap
    ax_map = fig.add_subplot(gs[0])
    im = draw_grid_heatmap(
        ax_map, values_grid, policy_grid,
        title=f"时间步 #{step_num}: {algo_name} 状态与动作转移\n({event['s']} $\\rightarrow$ {event['next_s']}, 奖励 $R={event['r']:+.1f}$)",
        highlight_trans={"s": event["s"], "next_s": event["next_s"], "a": event["a"], "r": event["r"]}
    )
    plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label="Value Scale")
    
    # Right Panel: Mathematics & Formula Derivation
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    # Card Background Box
    card_box = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=2.0
    )
    ax_text.add_patch(card_box)
    
    # Header
    ax_text.text(0.06, 0.92, f"【关键时间步 #{step_num} 深度透视卡片】", fontsize=14, fontweight='bold', color="#1E293B")
    ax_text.text(0.06, 0.86, f"算法: {algo_name} | Episode {event['episode']} | Step {event['step']}", fontsize=11, color="#64748B", fontweight='semibold')
    
    # Section 1: Update Formula
    ax_text.text(0.06, 0.78, "【核心更新公式】", fontsize=12, fontweight='bold', color="#0F172A")
    ax_text.text(0.08, 0.71, formula_latex, fontsize=12, color="#1E40AF", fontweight='bold')
    
    # Section 2: Arithmetic Substitution
    ax_text.text(0.06, 0.62, "【算术级数值代入计算】", fontsize=12, fontweight='bold', color="#0F172A")
    y_pos = 0.55
    for line in math_derivation:
        ax_text.text(0.08, y_pos, line, fontsize=10.5, color="#334155")
        y_pos -= 0.058
        
    # Section 3: Strategic RL Insight
    y_pos -= 0.02
    ax_text.text(0.06, y_pos, "【强化学习物理/策略意义】", fontsize=12, fontweight='bold', color="#B91C1C")
    y_pos -= 0.06
    
    # Split insight into lines
    words = strategic_insight.split("\n")
    for w in words:
        ax_text.text(0.08, y_pos, w, fontsize=10, color="#1F2937")
        y_pos -= 0.052
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

print("Plotting module defined successfully.")

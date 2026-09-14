"""
Main experiment runner for Complex GridWorld.
Executes DP optimal baseline, TD(0), SARSA, and Q-Learning.
Logs detailed outputs and extracts critical time steps with largest value changes.
"""
import os
import sys
import pathlib
# Pathlib mkdir patch for Windows Sandbox
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

import numpy as np
from src.complex_grid_world import ComplexGridWorld, ACTION_NAMES, ACTION_SYMBOLS, ACTIONS, UP, DOWN, LEFT, RIGHT
from src.complex_model_free import solve_dp_optimal, run_complex_td0, run_complex_sarsa, run_complex_q_learning

def format_grid(val_dict, rows=4, cols=4, precision=2):
    lines = []
    for r in range(rows):
        row_str = "  ".join([f"{val_dict.get((r, c), 0.0):+6.{precision}f}" for c in range(cols)])
        lines.append(f"  Row {r}: [ {row_str} ]")
    return "\n".join(lines)

def format_policy_grid(q_dict, rows=4, cols=4, terminals=None):
    lines = []
    terminals = terminals or []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            st = (r, c)
            if st in terminals:
                row_cells.append("  GOAL  ")
                continue
            qs = q_dict[st]
            best_a = max(qs.keys(), key=lambda a: qs[a])
            sym = ACTION_SYMBOLS[best_a]
            row_cells.append(f"   {sym}    ")
        lines.append(f"  Row {r}: [ {' '.join(row_cells)} ]")
    return "\n".join(lines)

def main():
    env = ComplexGridWorld(rows=4, cols=4, gamma=0.9, step_cost=-1.0)
    
    print("=" * 80)
    print("COMPLEX GRIDWORLD EXPERIMENT: MULTI-TIER REWARDS & MODEL-FREE DYNAMICS")
    print("=" * 80)
    print("Environment Specifications:")
    print("  Grid Size: 4x4 (16 states)")
    print("  Start: (0, 0)")
    print("  Sub-Goal: (0, 3) Terminal [Reward: +3.0] (Quick safe exit)")
    print("  Main Goal: (3, 3) Terminal [Reward: +10.0] (High-value treasure)")
    print("  Hazards: (1, 2) Lava Pit [-10.0], (2, 3) Spikes [-6.0], (2, 1) Quicksand [-3.0]")
    print("  Step Cost: -1.0, Discount Gamma: 0.9\n")
    
    # 1. Ground Truth DP Solution
    print("-" * 60)
    print("1. Ground Truth DP Optimal Solution (Value Iteration)")
    print("-" * 60)
    V_opt, Q_opt = solve_dp_optimal(env, gamma=0.9)
    print("Optimal State Values V*(s):")
    print(format_grid(V_opt))
    print("\nOptimal Policy pi*(s) derived from Q*(s, a):")
    print(format_policy_grid(Q_opt, terminals=list(env.terminal_states.keys())))
    
    # 2. TD(0) Policy Evaluation
    print("\n" + "-" * 60)
    print("2. Running TD(0) Policy Evaluation (Random Policy)")
    print("-" * 60)
    td0_res = run_complex_td0(env, num_episodes=2500, alpha=0.1, gamma=0.9, seed=42)
    print("Converged TD(0) Estimated State Values V(s):")
    print(format_grid(td0_res["V"]))
    
    # 3. SARSA On-Policy Control
    print("\n" + "-" * 60)
    print("3. Running SARSA On-Policy Control (Epsilon=0.1, Alpha=0.1)")
    print("-" * 60)
    sarsa_res = run_complex_sarsa(env, num_episodes=2000, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    sarsa_V = {s: max(sarsa_res["Q"][s].values()) for s in env.states}
    print("Converged SARSA Max Q-Values max_a Q(s, a):")
    print(format_grid(sarsa_V))
    print("\nConverged SARSA Greedy Policy:")
    print(format_policy_grid(sarsa_res["Q"], terminals=list(env.terminal_states.keys())))
    
    # 4. Q-Learning Off-Policy Control
    print("\n" + "-" * 60)
    print("4. Running Q-Learning Off-Policy Control (Epsilon=0.1, Alpha=0.1)")
    print("-" * 60)
    ql_res = run_complex_q_learning(env, num_episodes=2000, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    ql_V = {s: max(ql_res["Q"][s].values()) for s in env.states}
    print("Converged Q-Learning Max Q-Values max_a Q(s, a):")
    print(format_grid(ql_V))
    print("\nConverged Q-Learning Greedy Policy:")
    print(format_policy_grid(ql_res["Q"], terminals=list(env.terminal_states.keys())))
    
    # 5. Extract Largest Value Changes
    print("\n" + "=" * 80)
    print("ANALYSIS: EXTRACTING TIME STEPS WITH LARGEST VALUE CHANGES")
    print("=" * 80)
    
    # Sort events by abs_delta
    td0_sorted = sorted(td0_res["events"], key=lambda x: x["abs_delta"], reverse=True)
    sarsa_sorted = sorted(sarsa_res["events"], key=lambda x: x["abs_delta"], reverse=True)
    ql_sorted = sorted(ql_res["events"], key=lambda x: x["abs_delta"], reverse=True)
    
    print("\nTop 5 Largest Delta Events in TD(0):")
    for i, ev in enumerate(td0_sorted[:5]):
        print(f"  [TD0 #{i+1}] Ep {ev['episode']}, Step {ev['step']}: State {ev['s']} -> Act {ACTION_NAMES[ev['a']]} -> Rew {ev['r']:+4.1f} -> Next {ev['next_s']} (Done={ev['done']})")
        print(f"         Old V: {ev['old_val']:+6.3f} | Target: {ev['target']:+6.3f} | Error: {ev['error']:+6.3f} | New V: {ev['new_val']:+6.3f} | |Delta|: {ev['abs_delta']:6.3f}")
        
    print("\nTop 5 Largest Delta Events in SARSA:")
    for i, ev in enumerate(sarsa_sorted[:5]):
        print(f"  [SARSA #{i+1}] Ep {ev['episode']}, Step {ev['step']}: State {ev['s']} -> Act {ACTION_NAMES[ev['a']]} -> Rew {ev['r']:+4.1f} -> Next {ev['next_s']} -> NextAct {ACTION_NAMES.get(ev.get('next_a'), 'NONE')}")
        print(f"         Old Q: {ev['old_val']:+6.3f} | Target: {ev['target']:+6.3f} | Error: {ev['error']:+6.3f} | New Q: {ev['new_val']:+6.3f} | |Delta|: {ev['abs_delta']:6.3f}")

    print("\nTop 5 Largest Delta Events in Q-Learning:")
    for i, ev in enumerate(ql_sorted[:5]):
        print(f"  [QLearn #{i+1}] Ep {ev['episode']}, Step {ev['step']}: State {ev['s']} -> Act {ACTION_NAMES[ev['a']]} -> Rew {ev['r']:+4.1f} -> Next {ev['next_s']} (max Q(S')={ev['max_next_q']:+6.3f})")
        print(f"         Old Q: {ev['old_val']:+6.3f} | Target: {ev['target']:+6.3f} | Error: {ev['error']:+6.3f} | New Q: {ev['new_val']:+6.3f} | |Delta|: {ev['abs_delta']:6.3f}")

if __name__ == "__main__":
    main()

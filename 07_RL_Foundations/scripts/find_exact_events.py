"""
Locates exact events matching the 6 pedagogical critical steps.
"""
import os, pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

from src.complex_grid_world import ComplexGridWorld, ACTION_NAMES, ACTION_SYMBOLS, UP, DOWN, LEFT, RIGHT
from src.complex_model_free import run_complex_td0, run_complex_sarsa, run_complex_q_learning

env = ComplexGridWorld()
td0_res = run_complex_td0(env, num_episodes=2500, seed=42)
sarsa_res = run_complex_sarsa(env, num_episodes=2000, seed=42)
ql_res = run_complex_q_learning(env, num_episodes=2000, seed=42)

print("=== Search for Event 1: TD(0) Sub-Goal Discovery (0,2) -> (0,3) ===")
for ev in td0_res["events"]:
    if ev["s"] == (0, 2) and ev["next_s"] == (0, 3):
        print(f"Found TD0 SubGoal: Ep {ev['episode']}, Step {ev['step']}, s={ev['s']}, a={ACTION_NAMES[ev['a']]}, r={ev['r']}, next={ev['next_s']}, old={ev['old_val']:.3f}, target={ev['target']:.3f}, err={ev['error']:.3f}, new={ev['new_val']:.3f}")
        break

print("\n=== Search for Event 2: TD(0) Lava Pit Fall (1,1) -> (1,2) or (0,2) -> (1,2) ===")
for ev in td0_res["events"]:
    if ev["next_s"] == (1, 2) and ev["r"] == -10.0:
        print(f"Found TD0 Lava: Ep {ev['episode']}, Step {ev['step']}, s={ev['s']}, a={ACTION_NAMES[ev['a']]}, r={ev['r']}, next={ev['next_s']}, old={ev['old_val']:.3f}, target={ev['target']:.3f}, err={ev['error']:.3f}, new={ev['new_val']:.3f}")
        break

print("\n=== Search for Event 3: SARSA Main Goal Discovery (3,2) -> (3,3) ===")
for ev in sarsa_res["events"]:
    if ev["s"] == (3, 2) and ev["next_s"] == (3, 3):
        print(f"Found SARSA MainGoal: Ep {ev['episode']}, Step {ev['step']}, s={ev['s']}, a={ACTION_NAMES[ev['a']]}, r={ev['r']}, next={ev['next_s']}, old={ev['old_val']:.3f}, target={ev['target']:.3f}, err={ev['error']:.3f}, new={ev['new_val']:.3f}")
        break

print("\n=== Search for Event 4: SARSA Spike Trap (2,2) -> (2,3) ===")
for ev in sarsa_res["events"]:
    if ev["s"] == (2, 2) and ev["next_s"] == (2, 3):
        print(f"Found SARSA Spikes: Ep {ev['episode']}, Step {ev['step']}, s={ev['s']}, a={ACTION_NAMES[ev['a']]}, r={ev['r']}, next={ev['next_s']}, old={ev['old_val']:.3f}, target={ev['target']:.3f}, err={ev['error']:.3f}, new={ev['new_val']:.3f}")
        break

print("\n=== Search for Event 5: Q-Learning Bootstrap Surge (2,2) -> (3,2) ===")
for ev in ql_res["events"]:
    if ev["s"] == (2, 2) and ev["next_s"] == (3, 2) and ev["max_next_q"] > 0:
        print(f"Found QLearn Surge: Ep {ev['episode']}, Step {ev['step']}, s={ev['s']}, a={ACTION_NAMES[ev['a']]}, r={ev['r']}, next={ev['next_s']}, max_next={ev['max_next_q']:.3f}, old={ev['old_val']:.3f}, target={ev['target']:.3f}, err={ev['error']:.3f}, new={ev['new_val']:.3f}")
        break

print("\n=== Search for Event 6: Q-Learning Lava Trap Discovery (1,1) -> (1,2) ===")
for ev in ql_res["events"]:
    if ev["s"] == (1, 1) and ev["next_s"] == (1, 2):
        print(f"Found QLearn Lava: Ep {ev['episode']}, Step {ev['step']}, s={ev['s']}, a={ACTION_NAMES[ev['a']]}, r={ev['r']}, next={ev['next_s']}, old={ev['old_val']:.3f}, target={ev['target']:.3f}, err={ev['error']:.3f}, new={ev['new_val']:.3f}")
        break

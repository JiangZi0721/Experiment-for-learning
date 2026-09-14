"""
Core module for comparing Distribution Model vs Sample Model in Reinforcement Learning.
Includes:
1. Sutton Classic State-Update Analytical Model (Sutton & Barto 2018, Figure 8.7)
2. Parametric Branching MDP (Macro MDP Planning: Expected vs Sample VI)
3. Combinatorial Scaling MDP (Branching factor b from 10 to 5000)
4. Rare Catastrophic Trap MDP (Safety-Critical / Tail-Risk Environment)
"""
import numpy as np
import time
from typing import Dict, List, Tuple

# ==============================================================================
# 1. Sutton Classic Analytical Model (Sutton & Barto Sec 8.7)
# ==============================================================================
def compute_sutton_fig87_curves(b_values=[1, 2, 3, 10, 100, 1000], max_units=100):
    """
    Computes theoretical RMS error vs computation units for estimating state value,
    exactly reproducing Sutton & Barto (2018) Figure 8.7.
    - True state value v is the average of b successor state values (standard normal N(0, 1)).
    - Expected update: takes b units of computation, error drops to 0 at t = b.
    - Sample update: after t samples (each costing 1 unit), error = sqrt((b - t) / ((b - 1) * t))
      for sampling without replacement, or 1/sqrt(t) for sampling with replacement.
    """
    results = {}
    t_vals = np.arange(1, max_units + 1)
    
    for b in b_values:
        # Expected update curve: error = 1.0 for t < b, and 0 for t >= b
        exp_err = np.where(t_vals < b, 1.0, 0.0)
        
        # Sample update curve: error = 1 / sqrt(t)
        # With finite population correction without replacement:
        if b == 1:
            samp_err = np.zeros_like(t_vals, dtype=float)
            exp_err = np.zeros_like(t_vals, dtype=float)
        else:
            samp_err = []
            for t in t_vals:
                if t >= b:
                    samp_err.append(0.0)
                else:
                    # Finite population correction
                    err = np.sqrt(((b - t) / ((b - 1.0) * t)))
                    samp_err.append(min(1.0, err))
            samp_err = np.array(samp_err)
            
        results[b] = {
            't': t_vals,
            'expected_error': exp_err,
            'sample_error': samp_err
        }
    return results


# ==============================================================================
# 2. Parametric Branching MDP (Macro MDP Planning)
# ==============================================================================
class SuttonBranchingMDP:
    """
    Parametric MDP with n states and branching factor b.
    For each state, there are 2 actions. Under each action, transition leads to
    b next states with equal probability (1/b).
    """
    def __init__(self, num_states: int = 50, branching_factor: int = 10, gamma: float = 0.9, seed: int = 42):
        self.num_states = num_states
        self.b = branching_factor
        self.gamma = gamma
        self.num_actions = 2
        self.rng = np.random.RandomState(seed)
        
        self.transitions = self.rng.randint(0, num_states, size=(num_states, self.num_actions, self.b))
        self.rewards = self.rng.normal(0.0, 1.0, size=(num_states, self.num_actions, self.b))
        self.true_V, self.true_Q = self._compute_ground_truth()

    def _compute_ground_truth(self, tol: float = 1e-9, max_iter: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        V = np.zeros(self.num_states)
        for _ in range(max_iter):
            V_new = np.zeros(self.num_states)
            for s in range(self.num_states):
                q_vals = []
                for a in range(self.num_actions):
                    next_states = self.transitions[s, a]
                    rewards = self.rewards[s, a]
                    exp_q = np.mean(rewards + self.gamma * V[next_states])
                    q_vals.append(exp_q)
                V_new[s] = np.max(q_vals)
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        
        Q = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                next_states = self.transitions[s, a]
                rewards = self.rewards[s, a]
                Q[s, a] = np.mean(rewards + self.gamma * V[next_states])
        return V, Q

    def get_transition_distribution(self, s: int, a: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distribution model: returns (next_states, rewards, probabilities)."""
        next_states = self.transitions[s, a]
        rewards = self.rewards[s, a]
        probs = np.full(self.b, 1.0 / self.b)
        return next_states, rewards, probs

    def sample_transition(self, s: int, a: int) -> Tuple[int, float]:
        """Sample model: black-box draw O(1)."""
        idx = self.rng.randint(0, self.b)
        return self.transitions[s, a, idx], self.rewards[s, a, idx]


# ==============================================================================
# 3. Combinatorial Scaling MDP (Explicit Branching Factor b from 10 to 5000)
# ==============================================================================
class CombinatorialScalingMDP:
    """
    Evaluates real-world wall-clock computational scaling as branching factor b grows
    from low (b=10) to huge (b=5000).
    Represents complex environments (multi-agent, sensor noise, stochastic physics).
    """
    def __init__(self, num_states: int = 20, branching_factor: int = 100, gamma: float = 0.95, seed: int = 42):
        self.num_states = num_states
        self.b = branching_factor
        self.gamma = gamma
        self.num_actions = 2
        self.rng = np.random.RandomState(seed)
        
        # Branch transitions: next states and rewards
        self.transitions = self.rng.randint(0, num_states, size=(num_states, self.num_actions, self.b))
        self.rewards = self.rng.randn(num_states, self.num_actions, self.b)
        self.probs = np.full(self.b, 1.0 / self.b)

    def expected_sweep(self, V: np.ndarray) -> np.ndarray:
        """
        Distribution model update: must sum over all b branches for every (s, a).
        Time complexity: O(S * A * b).
        """
        V_new = np.zeros(self.num_states)
        for s in range(self.num_states):
            q_max = -1e9
            for a in range(self.num_actions):
                # Vectorized sum over all b branches
                next_s = self.transitions[s, a]
                rew = self.rewards[s, a]
                exp_q = np.dot(self.probs, rew + self.gamma * V[next_s])
                if exp_q > q_max:
                    q_max = exp_q
            V_new[s] = q_max
        return V_new

    def sample_sweep(self, V: np.ndarray) -> np.ndarray:
        """
        Sample model update: simulates 1 sample per (s, a).
        Time complexity: O(S * A), completely independent of branching factor b!
        """
        V_new = np.zeros(self.num_states)
        for s in range(self.num_states):
            q_max = -1e9
            for a in range(self.num_actions):
                idx = self.rng.randint(0, self.b)
                s_next = self.transitions[s, a, idx]
                r = self.rewards[s, a, idx]
                sample_q = r + self.gamma * V[s_next]
                if sample_q > q_max:
                    q_max = sample_q
            V_new[s] = q_max
        return V_new


# ==============================================================================
# 4. Rare Catastrophic Trap MDP (Safety-Critical Environment)
# ==============================================================================
class RareTrapMDP:
    """
    MDP demonstrating the dangerous optimism of sample models under rare tail events.
    Two paths:
    - Safe Action (0): 100% chance of +1.0 reward
    - Risky Action (1):
        - (1 - p_trap) chance of +2.0 reward
        - p_trap chance of -500.0 catastrophe
    
    True Expected Value:
    E[R(Safe)] = +1.00
    E[R(Risky)] = 0.995 * 2.0 - 0.005 * 500 = -0.51
    Safe is strictly optimal.
    """
    def __init__(self, p_trap: float = 0.005, trap_penalty: float = -500.0, seed: int = 42):
        self.p_trap = p_trap
        self.trap_penalty = trap_penalty
        self.rng = np.random.RandomState(seed)

    def sample_transition(self, a: int) -> float:
        if a == 0:
            return 1.0
        else:
            is_trap = (self.rng.rand() < self.p_trap)
            return self.trap_penalty if is_trap else 2.0

    def get_distribution(self, a: int) -> List[Tuple[float, float]]:
        if a == 0:
            return [(1.0, 1.0)]
        else:
            return [(2.0, 1.0 - self.p_trap), (self.trap_penalty, self.p_trap)]

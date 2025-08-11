from __future__ import annotations
import numpy as np

__all__ = ["q_learning_update"]

def q_learning_update(Q: np.ndarray, s: int, a: int, r: float, s_next: int,
                      alpha: float = 0.1, gamma: float = 0.99) -> float:
    """
    One-step Q-learning update for tabular RL.
    Q[s, a] ← Q[s, a] + α * ( r + γ * max_a' Q[s_next, a'] - Q[s, a] )
    Returns the TD error (target - current).
    """
    target = r + gamma * float(np.max(Q[s_next]))
    td_error = target - float(Q[s, a])
    Q[s, a] += alpha * td_error
    return float(td_error)
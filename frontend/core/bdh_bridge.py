"""Small mathematical teaching module for the BDH bridge.

The equations below are a simplified visualization of the published BDH
construction. They are included to teach the mechanism, not to reproduce the
full research model.
"""

from __future__ import annotations

import numpy as np


def hebbian_write(state: np.ndarray, x: np.ndarray, v: np.ndarray, decay: float = 0.02) -> np.ndarray:
    """Update synaptic state sigma <- (1-decay)sigma + x^T v."""
    return (1.0 - decay) * state + np.outer(x, v)


def read_synapses(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Read o = x sigma, matching the matrix form shown in Pathway's explainer."""
    return x @ sigma


def toy_synapse_replay(seed: int = 7, neurons: int = 12, steps: int = 8):
    rng = np.random.default_rng(seed)
    sigma = np.zeros((neurons, neurons), dtype=float)
    history = [sigma.copy()]
    for _ in range(steps):
        x = (rng.random(neurons) > 0.72).astype(float)
        v = (rng.random(neurons) > 0.72).astype(float)
        sigma = hebbian_write(sigma, x, v)
        history.append(sigma.copy())
    return np.stack(history)

"""Toy, educational latent-rule reasoner.

This is an independent teaching model inspired by the *idea* of iterative latent
state refinement. It is NOT an implementation of Coconut, HRM, TRM, BDH, or BDH-CQ.

The solver enumerates a small hypothesis space of visual transformations. Each
transform receives evidence from demonstrations. A fixed-size latent vector z
stores soft belief over hypotheses and is refined recurrently. The final prediction
is made from the current latent state without producing a textual chain of thought.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np

Grid = np.ndarray

@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    fn: Callable[[Grid], Grid]


def identity(g: Grid) -> Grid:
    return g.copy()


def rot90(g: Grid) -> Grid:
    return np.rot90(g, -1).copy()


def rot180(g: Grid) -> Grid:
    return np.rot90(g, 2).copy()


def flip_h(g: Grid) -> Grid:
    return np.fliplr(g).copy()


def flip_v(g: Grid) -> Grid:
    return np.flipud(g).copy()


def shift_right(g: Grid) -> Grid:
    out = np.zeros_like(g)
    out[:, 1:] = g[:, :-1]
    return out


def shift_down(g: Grid) -> Grid:
    out = np.zeros_like(g)
    out[1:, :] = g[:-1, :]
    return out


def swap_1_2(g: Grid) -> Grid:
    out = g.copy()
    out[g == 1] = 3
    out[g == 2] = 1
    out[g == 3] = 2
    return out


def invert_nonzero(g: Grid) -> Grid:
    out = g.copy()
    nz = out != 0
    out[nz] = 4 - out[nz]
    return out

RULES: List[Rule] = [
    Rule("Identity", "Keep the grid unchanged", identity),
    Rule("Rotate 90°", "Rotate clockwise by a quarter turn", rot90),
    Rule("Rotate 180°", "Rotate by half a turn", rot180),
    Rule("Flip horizontal", "Mirror left ↔ right", flip_h),
    Rule("Flip vertical", "Mirror top ↔ bottom", flip_v),
    Rule("Shift right", "Move every cell one step right", shift_right),
    Rule("Shift down", "Move every cell one step down", shift_down),
    Rule("Cycle colors", "Cycle 1→3, 2→1, 3→2", swap_1_2),
    Rule("Invert colors", "Swap 1↔3 while keeping 2", invert_nonzero),
]


def make_demo(rng: np.random.Generator, rule: Rule, size: int = 5) -> Tuple[Grid, Grid]:
    """Create a sparse ARC-like example whose transform stays in-frame."""
    g = np.zeros((size, size), dtype=np.int8)
    # Keep patterns away from boundaries for shift demonstrations.
    margin = 1
    for _ in range(rng.integers(2, 6)):
        r = int(rng.integers(margin, size - margin))
        c = int(rng.integers(margin, size - margin))
        g[r, c] = int(rng.integers(1, 4))
    if np.all(g == 0):
        g[size // 2, size // 2] = 1
    return g, rule.fn(g)


def corrupt_grid(target: Grid, rng: np.random.Generator, rate: float) -> Grid:
    """Randomly replace cells with another color according to rate."""
    out = target.copy()
    mask = rng.random(target.shape) < rate
    if mask.any():
        out[mask] = rng.integers(0, 4, size=mask.sum())
    return out


def mismatch(a: Grid, b: Grid) -> float:
    return float(np.mean(a != b))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-3)
    y = (x - np.max(x)) / t
    e = np.exp(y)
    return e / np.sum(e)


def score_demonstrations(demos: List[Tuple[Grid, Grid]]) -> np.ndarray:
    """Return one negative-error evidence score per hypothesis."""
    scores = []
    for rule in RULES:
        errors = [mismatch(rule.fn(x), y) for x, y in demos]
        scores.append(-float(np.mean(errors)))
    return np.asarray(scores, dtype=float)


def run_recurrent_reasoner(
    demos: List[Tuple[Grid, Grid]],
    query: Grid,
    iterations: int = 6,
    update_strength: float = 0.65,
    temperature: float = 0.50,
    noise: float = 0.0,
) -> Dict[str, object]:
    """Refine a fixed-size latent state over several recurrent steps.

    The toy latent state is a vector of hypothesis logits. Each step nudges the
    state toward evidence extracted from the demonstrations. Optional Gaussian
    noise lets the learner test robustness and saturation.
    """
    evidence = score_demonstrations(demos)
    z = np.zeros(len(RULES), dtype=float)
    trajectory = [softmax(z, temperature)]
    rng = np.random.default_rng(1234)

    for _ in range(iterations):
        z = (1.0 - update_strength) * z + update_strength * evidence
        if noise:
            z = z + rng.normal(0.0, noise, size=z.shape)
        trajectory.append(softmax(z, temperature))

    probs = trajectory[-1]
    top = int(np.argmax(probs))
    prediction = RULES[top].fn(query)

    return {
        "evidence": evidence,
        "logits": z,
        "probabilities": probs,
        "trajectory": np.vstack(trajectory),
        "rule_index": top,
        "rule": RULES[top],
        "prediction": prediction,
    }


def accuracy_curve(
    demos: List[Tuple[Grid, Grid]],
    queries: List[Tuple[Grid, Grid]],
    max_iterations: int = 12,
    update_strength: float = 0.65,
    temperature: float = 0.50,
) -> List[float]:
    acc = []
    for steps in range(1, max_iterations + 1):
        correct = 0
        for q, y in queries:
            result = run_recurrent_reasoner(demos, q, steps, update_strength, temperature)
            correct += int(np.array_equal(result["prediction"], y))
        acc.append(correct / max(len(queries), 1))
    return acc

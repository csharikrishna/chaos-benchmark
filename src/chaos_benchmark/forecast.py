"""Forecast horizon: how many steps before a tiny initial perturbation grows
into a large practical difference. This is a genuinely different question
from the Lyapunov exponent (an asymptotic rate) -- it's tied to a concrete,
practical error threshold, so it keeps its own twin-trajectory method."""

import numpy as np


def forecast_horizon(x1, x2, rel_error_thresh=0.10):
    """Steps until two twin trajectories diverge beyond rel_error_thresh times
    the base trajectory's own standard deviation. Returns None (censored) if
    they never diverge within the simulated window -- this is common for
    weakly chaotic systems (small Lyapunov exponent) where the window simply
    isn't long enough; see README."""
    n = min(len(x1), len(x2))
    scale = np.std(x1) if np.std(x1) > 0 else 1.0
    for i in range(n):
        rel_err = abs(x1[i] - x2[i]) / scale
        if rel_err > rel_error_thresh:
            return i
    return None

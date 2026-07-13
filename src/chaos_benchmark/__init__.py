"""
chaos_benchmark
================

A small, honest benchmark-dataset generator for nonlinear dynamical systems:
Logistic map, Henon map, Lorenz system, Rossler system, Mackey-Glass equation.

Each generated row = one simulated trajectory, summarized into 13 statistical/
spectral/nonlinear features, with a regime label (Stable/Periodic/Chaotic)
computed from the trajectory itself (not assumed from the sampling range),
and a forecast horizon estimated from twin-trajectory divergence.

See the README for known limitations. This package tries to be upfront about
where the labeling is fuzzy rather than hiding it.
"""

from .simulators import (
    simulate_logistic,
    simulate_henon,
    simulate_lorenz,
    simulate_rossler,
    simulate_mackey_glass,
)
from .features import extract_features
from .labeling import classify_regime, lyapunov_map, lyapunov_rosenstein
from .forecast import forecast_horizon
from .generate import build_dataset

__version__ = "0.1.0"

__all__ = [
    "simulate_logistic",
    "simulate_henon",
    "simulate_lorenz",
    "simulate_rossler",
    "simulate_mackey_glass",
    "extract_features",
    "classify_regime",
    "lyapunov_map",
    "lyapunov_rosenstein",
    "forecast_horizon",
    "build_dataset",
]

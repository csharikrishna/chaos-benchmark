"""Builds the benchmark dataset: samples parameters/initial conditions per
system from literature-informed sub-ranges, simulates, extracts features,
labels the regime from the actual trajectory, and estimates a forecast
horizon."""

import numpy as np
import pandas as pd

from .simulators import (
    simulate_logistic, simulate_henon, simulate_lorenz, simulate_rossler,
    simulate_mackey_glass,
)
from .features import extract_features
from .labeling import classify_regime, lyapunov_map, lyapunov_rosenstein
from .forecast import forecast_horizon


def _row_skeleton(system, **params):
    row = {
        "system": system, "param_r": None, "param_a": None, "param_b": None,
        "param_sigma": None, "param_rho": None, "param_beta": None,
        "param_c": None, "param_tau": None, "x0": None, "y0": None, "z0": None,
    }
    row.update(params)
    return row


def gen_logistic_row(rng, r_range=(2.0, 4.0)):
    r = rng.uniform(*r_range)
    x0 = rng.uniform(0.05, 0.95)
    x = simulate_logistic(r, x0, n_steps=500, discard=100)

    lyap = lyapunov_map(lambda xs: r * (1 - 2 * xs), x)
    x2 = simulate_logistic(r, x0 + 1e-8, n_steps=500, discard=100)
    fh = forecast_horizon(x, x2)

    label = classify_regime(lyap, x)
    row = _row_skeleton("Logistic", param_r=r, x0=x0,
                        lyapunov_exponent=lyap, label=label, forecast_horizon=fh)
    row.update(extract_features(x))
    return row


def gen_henon_row(rng, a_range=(0.2, 1.4)):
    a = rng.uniform(*a_range)
    b = rng.uniform(0.2, 0.3)
    x0, y0 = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    x = simulate_henon(a, b, x0, y0, n_steps=500, discard=100)
    if x is None:
        return None
    x2 = simulate_henon(a, b, x0 + 1e-8, y0, n_steps=500, discard=100)
    if x2 is None:
        return None

    lyap = lyapunov_rosenstein(x, m=3, tau=1, theiler_window=5)
    fh = forecast_horizon(x, x2)

    label = classify_regime(lyap, x)
    row = _row_skeleton("Henon", param_a=a, param_b=b, x0=x0, y0=y0,
                        lyapunov_exponent=lyap, label=label, forecast_horizon=fh)
    row.update(extract_features(x))
    return row


def gen_lorenz_row(rng, rho_range=(0.5, 45.0)):
    sigma, beta = 10.0, 8.0 / 3.0
    rho = rng.uniform(*rho_range)
    x0, y0, z0 = rng.uniform(-15, 15, size=3)
    kwargs = dict(sigma=sigma, rho=rho, beta=beta, x0=x0, y0=y0, z0=z0,
                  t_span=(0, 45), n_points=250, discard_frac=0.4)
    x = simulate_lorenz(**kwargs)
    if x is None:
        return None
    kwargs2 = dict(kwargs, x0=x0 + 1e-6)
    x2 = simulate_lorenz(**kwargs2)
    if x2 is None:
        return None

    lyap = lyapunov_rosenstein(x, m=3, tau=2, theiler_window=5)
    fh = forecast_horizon(x, x2)

    label = classify_regime(lyap, x)
    row = _row_skeleton("Lorenz", param_sigma=sigma, param_rho=rho, param_beta=beta,
                        x0=x0, y0=y0, z0=z0,
                        lyapunov_exponent=lyap, label=label, forecast_horizon=fh)
    row.update(extract_features(x))
    return row


def gen_rossler_row(rng, c_range=(1.0, 12.0)):
    a, b = 0.2, 0.2
    c = rng.uniform(*c_range)
    x0, y0, z0 = rng.uniform(-5, 5, size=3)
    kwargs = dict(a=a, b=b, c=c, x0=x0, y0=y0, z0=z0,
                  t_span=(0, 100), n_points=500, discard_frac=0.3)
    x = simulate_rossler(**kwargs)
    if x is None:
        return None
    kwargs2 = dict(kwargs, x0=x0 + 1e-6)
    x2 = simulate_rossler(**kwargs2)
    if x2 is None:
        return None

    lyap = lyapunov_rosenstein(x, m=3, tau=3, theiler_window=10)
    fh = forecast_horizon(x, x2)

    label = classify_regime(lyap, x)
    row = _row_skeleton("Rossler", param_a=a, param_b=b, param_c=c,
                        x0=x0, y0=y0, z0=z0,
                        lyapunov_exponent=lyap, label=label, forecast_horizon=fh)
    row.update(extract_features(x))
    return row


def gen_mackey_glass_row(rng, tau_range=(2.0, 30.0)):
    beta_mg, gamma, n_exp = 0.2, 0.1, 10
    tau = rng.uniform(*tau_range)
    x0 = rng.uniform(0.3, 1.3)
    kwargs = dict(beta=beta_mg, gamma=gamma, n_exp=n_exp, tau=tau, x0=x0,
                  n_steps=1500, dt=0.1, discard=500)
    x = simulate_mackey_glass(**kwargs)
    if x is None:
        return None
    kwargs2 = dict(kwargs, x0=x0 + 1e-6)
    x2 = simulate_mackey_glass(**kwargs2)
    if x2 is None:
        return None

    lyap = lyapunov_rosenstein(x, m=3, tau=5, theiler_window=20)
    fh = forecast_horizon(x, x2)

    # Mackey-Glass's chaotic regime has a genuinely smaller Lyapunov exponent
    # than the other four systems (see labeling.classify_regime docstring) --
    # 0.01 would misclassify real chaos here as Periodic.
    label = classify_regime(lyap, x, lyap_chaos_thresh=0.0015)
    row = _row_skeleton("MackeyGlass", param_beta=beta_mg, param_tau=tau, x0=x0,
                        lyapunov_exponent=lyap, label=label, forecast_horizon=fh)
    row.update(extract_features(x))
    return row


GENERATORS = {
    "Logistic": gen_logistic_row,
    "Henon": gen_henon_row,
    "Lorenz": gen_lorenz_row,
    "Rossler": gen_rossler_row,
    "MackeyGlass": gen_mackey_glass_row,
}

# Literature-informed sub-ranges known to produce each regime. The label is
# always re-verified from the computed trajectory, not assumed from the range.
CLASS_RANGES = {
    "Logistic":    {"Stable": {"r_range": (2.0, 2.95)},
                    "Periodic": {"r_range": (3.0, 3.55)},
                    "Chaotic": {"r_range": (3.62, 4.0)}},
    "Henon":       {"Stable": {"a_range": (0.2, 0.85)},
                    "Periodic": {"a_range": (1.0, 1.19)},
                    "Chaotic": {"a_range": (1.2, 1.4)}},
    "Lorenz":      {"Stable": {"rho_range": (0.5, 10.0)},
                    "Periodic": {"rho_range": (0.5, 45.0)},
                    "Chaotic": {"rho_range": (24.8, 40.0)}},
    "Rossler":     {"Stable": {"c_range": (1.0, 2.0)},
                    "Periodic": {"c_range": (2.5, 3.5)},
                    "Chaotic": {"c_range": (5.7, 10.0)}},
    "MackeyGlass": {"Stable": {"tau_range": (2.0, 4.0)},
                    "Periodic": {"tau_range": (6.0, 12.0)},
                    "Chaotic": {"tau_range": (17.0, 30.0)}},
}


def build_dataset(rows_per_class=100, max_attempts_per_class=500, seed=42,
                   systems=None, verbose=True):
    """Generate the benchmark dataset as a pandas DataFrame.

    Parameters
    ----------
    rows_per_class : target rows for each (system, class) combination.
    max_attempts_per_class : give up on a bucket after this many tries
        (some buckets, like Rossler-Stable, structurally can't fill --
        see README).
    seed : RNG seed for reproducibility.
    systems : optional list restricting which systems to generate
        (default: all five).
    """
    rng = np.random.default_rng(seed)
    rows = []
    system_names = systems or list(GENERATORS.keys())

    for sys_name in system_names:
        gen_fn = GENERATORS[sys_name]
        sys_counts = {"Stable": 0, "Periodic": 0, "Chaotic": 0}
        for target_class, range_kwargs in CLASS_RANGES[sys_name].items():
            attempts = 0
            while attempts < max_attempts_per_class and sys_counts[target_class] < rows_per_class:
                attempts += 1
                row = gen_fn(rng, **range_kwargs)
                if row is None:
                    continue
                label = row["label"]
                if label not in sys_counts or sys_counts[label] >= rows_per_class:
                    continue
                sys_counts[label] += 1
                rows.append(row)
        if verbose:
            print(f"{sys_name}: counts={sys_counts}")

    return pd.DataFrame(rows)

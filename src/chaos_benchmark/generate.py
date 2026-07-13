"""Builds the benchmark dataset: samples parameters/initial conditions per
system from literature-informed sub-ranges, simulates, extracts features,
labels the regime from the actual trajectory, and estimates a forecast
horizon."""

import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing

from .simulators import (
    simulate_logistic, simulate_henon, simulate_lorenz, simulate_rossler,
    simulate_mackey_glass,
)
from .features import extract_features
from .labeling import classify_regime, lyapunov_map, lyapunov_rosenstein
from .forecast import forecast_horizon

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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
    "Rossler":     {"Periodic": {"c_range": (2.5, 3.5)},
                    "Chaotic": {"c_range": (5.7, 10.0)}},
    "MackeyGlass": {"Stable": {"tau_range": (2.0, 4.0)},
                    "Periodic": {"tau_range": (6.0, 12.0)},
                    "Chaotic": {"tau_range": (17.0, 30.0)}},
}

def _evaluate_candidate(sys_name, range_kwargs, seed):
    rng = np.random.default_rng(seed)
    gen_fn = GENERATORS[sys_name]
    try:
        row = gen_fn(rng, **range_kwargs)
        return row
    except Exception:
        return None

def build_dataset(rows_per_class=100, max_attempts_per_class=500, seed=42,
                   systems=None, workers=None, verbose=True, progress_callback=None, cancel_check=None):
    """Generate the benchmark dataset as a pandas DataFrame using ProcessPoolExecutor."""
    rng = np.random.default_rng(seed)
    rows = []
    system_names = systems or list(GENERATORS.keys())

    total_expected = len(system_names) * 3 * rows_per_class
    if tqdm is not None and verbose:
        pbar = tqdm(total=total_expected, desc="Generating benchmark")
    else:
        pbar = None

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        workers = max(1, int(workers))
        
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    
    try:
        for sys_name in system_names:
            if cancel_check and cancel_check():
                break
            
            sys_counts = {"Stable": 0, "Periodic": 0, "Chaotic": 0}
            
            for target_class, range_kwargs in CLASS_RANGES[sys_name].items():
                if cancel_check and cancel_check():
                    break
                
                attempts = 0
                batch_size = max(workers * 2, rows_per_class)
                
                while attempts < max_attempts_per_class and sys_counts[target_class] < rows_per_class:
                    if cancel_check and cancel_check():
                        break
                        
                    to_submit = min(batch_size, max_attempts_per_class - attempts)
                    futures = []
                    for _ in range(to_submit):
                        task_seed = int(rng.integers(0, 2**32 - 1))
                        fut = executor.submit(_evaluate_candidate, sys_name, range_kwargs, task_seed)
                        futures.append(fut)
                        attempts += 1
                        
                    for fut in concurrent.futures.as_completed(futures):
                        if cancel_check and cancel_check():
                            for f in futures:
                                f.cancel()
                            break
                            
                        row = fut.result()
                        if row is not None:
                            label = row["label"]
                            if label in sys_counts and sys_counts[label] < rows_per_class:
                                sys_counts[label] += 1
                                rows.append(row)
                                if progress_callback:
                                    progress_callback(sys_name, label, sys_counts)
                                if pbar is not None:
                                    pbar.update(1)
                                    
                    if sys_counts[target_class] >= rows_per_class:
                        break
                        
                if cancel_check and cancel_check():
                    break
            
            if cancel_check and cancel_check():
                break

            if verbose:
                if pbar is not None:
                    pbar.write(f"{sys_name}: counts={sys_counts}")
                else:
                    print(f"{sys_name}: counts={sys_counts}")

    finally:
        # Gracefully kill all running processes immediately if Python 3.9+
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

    if pbar is not None:
        pbar.close()

    return pd.DataFrame(rows)

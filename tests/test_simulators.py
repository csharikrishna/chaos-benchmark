import numpy as np
from chaos_benchmark.simulators import (
    simulate_logistic, simulate_henon, simulate_lorenz, simulate_rossler,
    simulate_mackey_glass,
)


def test_logistic_shape_and_bounds():
    x = simulate_logistic(r=3.7, x0=0.4, n_steps=500, discard=100)
    assert len(x) == 400
    assert np.all(x >= 0) and np.all(x <= 1)  # logistic map is bounded in [0,1]


def test_logistic_low_r_converges_to_fixed_point():
    # for r < 3, logistic map should settle to a fixed point (near-zero variance)
    x = simulate_logistic(r=2.5, x0=0.4, n_steps=500, discard=200)
    assert np.var(x) < 1e-6


def test_henon_classic_chaotic_params_stay_bounded():
    x = simulate_henon(a=1.4, b=0.3, x0=0.1, y0=0.1, n_steps=500, discard=100)
    assert x is not None
    assert np.max(np.abs(x)) < 10  # classic Henon attractor stays small


def test_henon_diverges_returns_none():
    x = simulate_henon(a=5.0, b=5.0, x0=0.5, y0=0.5, n_steps=500, discard=100)
    assert x is None


def test_lorenz_low_rho_converges_near_origin():
    x = simulate_lorenz(sigma=10, rho=0.5, beta=8/3, x0=1, y0=1, z0=1)
    assert x is not None
    assert abs(np.mean(x[-20:])) < 0.5  # should settle near the origin


def test_lorenz_classic_chaotic_bounded():
    x = simulate_lorenz(sigma=10, rho=28, beta=8/3, x0=1, y0=1, z0=1)
    assert x is not None
    assert np.max(np.abs(x)) < 100  # classic butterfly attractor stays bounded


def test_rossler_runs():
    x = simulate_rossler(a=0.2, b=0.2, c=5.7, x0=1, y0=1, z0=1)
    assert x is not None
    assert len(x) > 0


def test_mackey_glass_runs_and_bounded():
    x = simulate_mackey_glass(beta=0.2, gamma=0.1, n_exp=10, tau=17, x0=1.2)
    assert x is not None
    assert np.max(np.abs(x)) < 10

"""Numerical simulators for five canonical nonlinear systems."""

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


@njit(fastmath=True)
def simulate_logistic(r, x0, n_steps=500, discard=100):
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n). Models bounded population growth."""
    x = np.empty(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[discard:]


@njit(fastmath=True)
def simulate_henon(a, b, x0, y0, n_steps=500, discard=100):
    """Henon map, a cheap 2-D stand-in for Lorenz-style chaotic attractors."""
    x = np.empty(n_steps)
    y = np.empty(n_steps)
    x[0], y[0] = x0, y0
    for i in range(1, n_steps):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
        if not np.isfinite(x[i]) or abs(x[i]) > 1e6:
            return None  # diverged
    return x[discard:]


def simulate_lorenz(sigma, rho, beta, x0, y0, z0, t_span=(0, 45), n_points=250,
                     discard_frac=0.4, blowup_bound=1e4):
    """Lorenz-63 atmospheric convection model. Default window is generous enough
    for low-rho trajectories to actually settle to their fixed point (see README
    for why this matters for correct Stable labeling)."""
    def rhs(t, s):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    def blowup_event(t, s):
        return np.linalg.norm(s) - blowup_bound
    blowup_event.terminal = True
    blowup_event.direction = 1

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, [x0, y0, z0], t_eval=t_eval, method="RK45",
                     rtol=1e-7, atol=1e-7, events=blowup_event)
    if not sol.success or sol.t[-1] < t_span[1] * 0.999:
        return None  # diverged before reaching the full window
    d = int(n_points * discard_frac)
    return sol.y[0][d:]


def simulate_rossler(a, b, c, x0, y0, z0, t_span=(0, 100), n_points=500,
                      discard_frac=0.3, blowup_bound=1e4):
    """Rossler system: deliberately the simplest possible chaotic flow.

    Note: for c in the "should be stable" range (~1.0-2.0), the fixed point
    is real but decays slowly -- it will often still register as Periodic
    within this window rather than Stable. See README known-limitations.
    """
    def rhs(t, s):
        x, y, z = s
        return [-y - z, x + a * y, b + z * (x - c)]

    def blowup_event(t, s):
        return np.linalg.norm(s) - blowup_bound
    blowup_event.terminal = True
    blowup_event.direction = 1

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, [x0, y0, z0], t_eval=t_eval, method="RK45",
                     rtol=1e-7, atol=1e-7, events=blowup_event)
    if not sol.success or sol.t[-1] < t_span[1] * 0.999:
        return None  # diverged before reaching the full window
    d = int(n_points * discard_frac)
    return sol.y[0][d:]


@njit(fastmath=True)
def simulate_mackey_glass(beta, gamma, n_exp, tau, x0, n_steps=1500, dt=0.1,
                           discard=500):
    """Mackey-Glass delay differential equation, modeling physiological feedback
    systems (e.g. blood cell production) where the body reacts to a past state,
    not the present one. Discretized with a simple fixed-step Euler method."""
    delay_steps = max(1, int(round(tau / dt)))
    total = n_steps + discard
    x = np.empty(total)
    x[:delay_steps] = x0
    for i in range(delay_steps, total - 1):
        x_tau = x[i - delay_steps]
        dx = beta * x_tau / (1 + x_tau ** n_exp) - gamma * x[i]
        x[i + 1] = x[i] + dt * dx
        if not np.isfinite(x[i + 1]) or abs(x[i + 1]) > 1e6:
            return None
    return x[discard + delay_steps:]

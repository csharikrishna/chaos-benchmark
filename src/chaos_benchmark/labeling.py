"""
Lyapunov exponent estimation and regime labeling.

The prototype version of this benchmark estimated the Lyapunov exponent for
continuous/delay systems by perturbing the initial condition once and
measuring divergence of that single pair of trajectories -- fast, but noisy
and dependent on exactly where the perturbation happened to land.

This version uses the Rosenstein et al. (1993) method instead: reconstruct
the trajectory's attractor via time-delay embedding, find each point's
nearest neighbor *within the same trajectory*, and average the divergence
rate over many such pairs. This is the standard published approach for
estimating the largest Lyapunov exponent from a single scalar time series,
and it's what most Lyapunov-exponent estimation software actually implements.
"""

import numpy as np


def lyapunov_map(deriv_fn, x_series):
    """Analytic Lyapunov exponent for 1-D maps: mean(log|f'(x)|).
    Exact (no estimation error) when you have the derivative in closed form,
    e.g. the logistic map."""
    derivs = np.abs(deriv_fn(x_series))
    derivs = derivs[derivs > 0]
    if len(derivs) == 0:
        return -np.inf
    return float(np.mean(np.log(derivs)))


def _embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return None
    return np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(n)])


def lyapunov_rosenstein(x, m=3, tau=1, theiler_window=10, max_iter=None, dt=1.0):
    """Estimate the largest Lyapunov exponent from a scalar trajectory.

    Parameters
    ----------
    x : 1-D array, the trajectory (already transient-discarded).
    m : embedding dimension.
    tau : embedding delay (in samples).
    theiler_window : excludes temporally close points from being counted as
        "nearest neighbors" (prevents trivially finding the very next point
        on the same orbit segment).
    max_iter : how many steps ahead to track divergence. Defaults to ~1/4
        of the usable series length.
    dt : time between samples, so the returned exponent is in physical units
        (nats per unit time) rather than nats per sample.

    Returns
    -------
    float or None (None if the trajectory is too short to embed reliably).

    Implementation note: nearest-neighbor search uses a KD-tree (not a brute
    force O(n^2) distance matrix), so this scales to the longer trajectories
    used for Mackey-Glass without becoming the runtime bottleneck.
    """
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float)
    embedded = _embed(x, m, tau)
    if embedded is None or len(embedded) < 30:
        return None
    n = len(embedded)
    if max_iter is None:
        max_iter = max(5, n // 4)
    max_iter = min(max_iter, n - 1)

    tree = cKDTree(embedded)
    # ask for enough neighbors to find one outside the Theiler window
    k_query = min(n, theiler_window * 2 + 10)
    dists, idxs = tree.query(embedded, k=k_query)

    neighbor_idx = np.full(n, -1, dtype=int)
    for i in range(n):
        for cand_d, cand_j in zip(dists[i], idxs[i]):
            if abs(cand_j - i) > theiler_window:
                neighbor_idx[i] = cand_j
                break

    valid = neighbor_idx >= 0
    if valid.sum() < 10:
        return None

    valid_i = np.where(valid)[0]
    valid_j = neighbor_idx[valid_i]

    log_div_by_k = []
    floor = 1e-12
    for k in range(max_iter):
        mask = (valid_i + k < n) & (valid_j + k < n)
        if mask.sum() < 10:
            break
        d = np.linalg.norm(embedded[valid_i[mask] + k] - embedded[valid_j[mask] + k], axis=1)
        log_div_by_k.append(np.mean(np.log(np.maximum(d, floor))))

    if len(log_div_by_k) < 5:
        return None

    k_vals = np.arange(len(log_div_by_k)) * dt
    slope, _ = np.polyfit(k_vals, log_div_by_k, 1)
    return float(slope)


def classify_regime(lyapunov_exp, x, stable_var_thresh=1e-6, lyap_chaos_thresh=0.01):
    """Stable / Periodic / Chaotic, from the Lyapunov exponent plus a
    variance check (near-zero variance means the trajectory settled to a
    fixed point -- Lyapunov exponent estimation gets unreliable right at
    that boundary, so the variance check takes priority).

    IMPORTANT: lyap_chaos_thresh is NOT a universal constant. Lyapunov
    exponents are not on a comparable scale across different systems --
    only the *sign* is generically meaningful (positive = chaotic). The
    magnitude depends on each system's own intrinsic timescale. E.g. in
    this benchmark, chaotic Logistic-map exponents come out ~0.4-0.5,
    Lorenz/Henon ~0.02-0.04, but Mackey-Glass's genuinely chaotic regime is
    only ~0.003-0.005 -- a real property of that equation, not an
    estimation error. Using one global threshold across systems with very
    different natural scales will silently misclassify the weaker-chaos
    system. generate.py passes a lower threshold for Mackey-Glass for this
    reason -- if you add a new system, re-check its typical chaotic-regime
    exponent magnitude before trusting the default 0.01 here.
    """
    var = np.var(x)
    if var < stable_var_thresh:
        return "Stable"
    if lyapunov_exp is None:
        return "Unknown"
    if lyapunov_exp > lyap_chaos_thresh:
        return "Chaotic"
    return "Periodic"

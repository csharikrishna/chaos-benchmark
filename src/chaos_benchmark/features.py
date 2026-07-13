"""Statistical, spectral, and nonlinear feature extraction from a 1-D trajectory."""

import numpy as np
from scipy import stats, signal


def _shannon_entropy(x, bins=20):
    if np.ptp(x) < 1e-12 * max(1.0, abs(np.mean(x))):
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _sample_entropy(x, m=2, r_frac=0.2):
    x = np.asarray(x)
    n = len(x)
    r = r_frac * np.std(x)
    if r == 0 or n < m + 2:
        return 0.0

    def _phi(m):
        templates = np.array([x[i:i + m] for i in range(n - m + 1)])
        count, total = 0, 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dists <= r) - 1
            total += len(templates) - 1
        return count, total

    cm, tm = _phi(m)
    cm1, tm1 = _phi(m + 1)
    if cm == 0 or cm1 == 0:
        return 0.0
    return float(-np.log(cm1 / tm1 / (cm / tm)))


def _spectral_entropy(x):
    freqs, psd = signal.periodogram(x - np.mean(x))
    psd = psd[1:]
    if psd.sum() == 0:
        return 0.0
    p = psd / psd.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


def _dominant_frequency(x, fs=1.0):
    freqs, psd = signal.periodogram(x - np.mean(x), fs=fs)
    if len(psd) <= 1:
        return 0.0
    idx = np.argmax(psd[1:]) + 1
    return float(freqs[idx])


def _fft_energy(x):
    spec = np.fft.rfft(x - np.mean(x))
    return float(np.sum(np.abs(spec) ** 2) / len(x))


def _autocorr_lag1(x):
    x = np.asarray(x)
    if np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _higuchi_fractal_dimension(x, k_max=10):
    x = np.asarray(x)
    n = len(x)
    lk, ks = [], range(1, k_max + 1)
    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(1, int((n - m - 1) / k) + 1)
            if len(idx) == 0:
                continue
            lmk = np.sum(np.abs(x[m + idx * k] - x[m + (idx - 1) * k]))
            norm = (n - 1) / (len(idx) * k)
            lm.append(lmk * norm / k)
        lk.append(np.mean(lm) if lm else 1e-10)
    lk = np.array(lk)
    lk[lk <= 0] = 1e-10
    log_k = np.log(1.0 / np.array(list(ks)))
    log_lk = np.log(lk)
    slope, _ = np.polyfit(log_k, log_lk, 1)
    return float(slope)


def _hurst_exponent(x):
    x = np.asarray(x)
    n = len(x)
    if n < 20:
        return 0.5
    lags = np.unique(np.logspace(0.7, np.log10(n // 2), num=15).astype(int))
    rs_vals, valid_lags = [], []
    for lag in lags:
        if lag < 2:
            continue
        segments = n // lag
        if segments < 1:
            continue
        rs_list = []
        for i in range(segments):
            seg = x[i * lag:(i + 1) * lag]
            dev = np.cumsum(seg - np.mean(seg))
            r = np.max(dev) - np.min(dev)
            s = np.std(seg)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            valid_lags.append(lag)
    if len(valid_lags) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(valid_lags), np.log(rs_vals), 1)
    return float(slope)


def extract_features(x):
    """Return a dict of 13 features summarizing a 1-D trajectory."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x)),
        "std": float(np.std(x)),
        "skewness": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "shannon_entropy": _shannon_entropy(x),
        "sample_entropy": _sample_entropy(x),
        "spectral_entropy": _spectral_entropy(x),
        "fft_energy": _fft_energy(x),
        "dominant_frequency": _dominant_frequency(x),
        "autocorr_lag1": _autocorr_lag1(x),
        "higuchi_fractal_dim": _higuchi_fractal_dimension(x),
        "hurst_exponent": _hurst_exponent(x),
    }

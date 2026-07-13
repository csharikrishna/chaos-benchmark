import numpy as np
from chaos_benchmark.features import extract_features
from chaos_benchmark.labeling import classify_regime, lyapunov_map, lyapunov_rosenstein
from chaos_benchmark.simulators import simulate_logistic


def test_extract_features_returns_all_keys():
    x = np.sin(np.linspace(0, 20, 300)) + 0.01 * np.random.randn(300)
    feats = extract_features(x)
    expected_keys = {
        "mean", "variance", "std", "skewness", "kurtosis", "shannon_entropy",
        "sample_entropy", "spectral_entropy", "fft_energy", "dominant_frequency",
        "autocorr_lag1", "higuchi_fractal_dim", "hurst_exponent",
    }
    assert expected_keys.issubset(feats.keys())
    assert all(np.isfinite(v) for v in feats.values())


def test_constant_trajectory_does_not_crash():
    x = np.ones(300) * 0.5
    feats = extract_features(x)
    assert feats["shannon_entropy"] == 0.0


def test_classify_regime_stable_from_zero_variance():
    x = np.ones(100) * 0.3
    label = classify_regime(lyapunov_exp=-0.5, x=x)
    assert label == "Stable"


def test_classify_regime_chaotic_from_positive_lyapunov():
    x = np.random.randn(300)  # nonzero variance
    label = classify_regime(lyapunov_exp=0.5, x=x)
    assert label == "Chaotic"


def test_lyapunov_map_logistic_known_chaotic_case():
    # r=4.0 logistic map has an exactly known analytic Lyapunov exponent of ln(2)
    r = 4.0
    x = simulate_logistic(r=r, x0=0.4, n_steps=5000, discard=500)
    lyap = lyapunov_map(lambda xs: r * (1 - 2 * xs), x)
    assert abs(lyap - np.log(2)) < 0.05


def test_lyapunov_rosenstein_positive_for_chaotic_logistic():
    x = simulate_logistic(r=3.9, x0=0.4, n_steps=1000, discard=200)
    lyap = lyapunov_rosenstein(x, m=3, tau=1, theiler_window=5)
    assert lyap is not None
    assert lyap > 0


def test_lyapunov_rosenstein_near_zero_or_negative_for_periodic_logistic():
    x = simulate_logistic(r=3.2, x0=0.4, n_steps=1000, discard=200)  # period-2 cycle
    lyap = lyapunov_rosenstein(x, m=3, tau=1, theiler_window=5)
    assert lyap is not None
    assert lyap < 0.05

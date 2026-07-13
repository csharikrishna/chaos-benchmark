# chaos-benchmark

A small, honest benchmark-dataset generator for nonlinear dynamical systems:
**Logistic map, Hénon map, Lorenz system, Rössler system, Mackey-Glass equation.**

Each row = one simulated trajectory, summarized into 13 features (variance,
entropy, spectral, fractal-dimension, Hurst exponent, etc.), with:
- a **regime label** (Stable / Periodic / Chaotic) computed from the trajectory's
  own Lyapunov exponent + variance — not assumed from the parameter range, and
- a **forecast horizon** — how long before a tiny initial perturbation grows
  into a practically large difference (twin-trajectory divergence).

This project is explicit about where its own labeling is fuzzy, instead of
hiding it. See **Known limitations** below before trusting the numbers.

## Repository Structure

The repository is organized following modern Python standards:
- **`src/`**: Contains the core package code (`simulators`, `features`, `labeling`, etc.) and the Web UI.
- **`examples/`**: Contains demo datasets and exploratory data analysis scripts.
- **Packaging (`pyproject.toml`, `MANIFEST.in`, `requirements.txt`)**: Standard files required to build and publish this tool to PyPI.
- **Publishing (`ROADMAP.md`, `paper.md`, `paper.bib`)**: Academic documentation for submission to the Journal of Open Source Software (JOSS).

## Install

```bash
git clone <this-repo>
cd chaos-benchmark
pip install -e .
```

## Use

As a library:

```python
from chaos_benchmark import build_dataset

df = build_dataset(rows_per_class=200, max_attempts_per_class=800, seed=42)
df.to_csv("my_dataset.csv", index=False)
```

Or run the **Web UI** for a beautiful interactive generator:

```bash
chaos-benchmark-web
```

Or from the command line (installed as a console script):

```bash
chaos-benchmark-generate --rows_per_class 200 --max_attempts 800 --out my_dataset.csv
```

Restrict to specific systems:

```bash
chaos-benchmark-generate --systems Logistic,Henon --rows_per_class 500 --out maps_only.csv
```

## Run the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

15 tests covering: simulator boundedness/divergence-detection, known analytic
Lyapunov exponents (e.g. logistic map at r=4 has an exact closed-form value of
ln(2), used as a correctness check on the estimator), feature extraction on
edge cases (constant trajectories), and regime classification logic.

## Methodology

1. Parameters/initial conditions are sampled from **literature-informed
   sub-ranges** known to sit in a given regime for each system (e.g. Lorenz
   ρ<1 → stable fixed point, ρ>24.74 → chaotic butterfly attractor).
2. **The label is always recomputed from the actual simulated trajectory**,
   never taken on faith from the sampling range. Rows land under whatever they
   actually compute to — occasional "off-target" labels near bifurcation
   boundaries are expected, not bugs.
3. **Lyapunov exponent**: the logistic map uses its exact analytic
   log-derivative formula. The other four systems use the **Rosenstein et
   al. (1993) method** — time-delay embedding, nearest-neighbor divergence
   tracking via a KD-tree, averaged over many pairs. This is the standard
   published approach for estimating the largest Lyapunov exponent from a
   single scalar time series (an earlier prototype of this code used a
   cruder single-perturbation twin-trajectory shortcut; this is more
   rigorous and was cross-checked against it).
4. **Forecast horizon** stays twin-trajectory-based (perturb the initial
   condition, measure divergence against a concrete relative-error
   threshold) — a genuinely different, more practically-flavored question
   than the asymptotic Lyapunov rate.

## Known limitations (read before trusting the numbers)

- **Lorenz "Periodic" bucket has a lower hit rate** (~85% of target) because
  the true periodic-window boundaries in ρ are narrow and not precisely
  known from memory — the current range (ρ in [0.5, 45], full sweep) is a
  best-effort net, not a precise literature citation. Worth tightening if
  you find better-sourced values.
- **Lyapunov exponent magnitude is NOT comparable across systems.** Only the
  *sign* is generically meaningful. In this benchmark, chaotic Logistic-map
  exponents come out ~0.4-0.5, Lorenz/Hénon ~0.02-0.04, but Mackey-Glass's
  genuinely chaotic regime is only ~0.003-0.005 — a real property of that
  equation, not an estimation error. `classify_regime` therefore takes a
  per-call `lyap_chaos_thresh`, and `generate.py` passes a lower one for
  Mackey-Glass specifically. **If you add a new system, re-check its
  typical chaotic-regime exponent magnitude before trusting the default.**
- **Mackey-Glass forecast horizon is mostly censored (NaN).** Its Lyapunov
  exponent is small, so a 1e-6 perturbation needs a much longer window than
  simulated here to visibly diverge to 10% of the trajectory's scale.
- **Regime boundaries are inherently fuzzy** — real bifurcation diagrams
  have periodic windows inside chaotic ranges and vice versa. Expect
  irreducible label noise near boundaries; that's a property of the
  systems, not something more code fully removes.
- **Feature/label leakage**: `lyapunov_exponent` is stored as its own
  column because it's literally what produced the label. Drop it (and
  probably `variance`/`std`, which drive the Stable check) from model
  inputs unless deliberately testing whether a model rediscovers known
  thresholds.
- `forecast_horizon` is in **simulation steps**, not physical time — the
  five systems use different step spacing, so don't compare raw step
  counts across systems without converting to simulation time.

## Runtime

Logistic and Hénon are cheap map iterations (milliseconds/row). Lorenz,
Rössler, and Mackey-Glass require real numerical integration and are the
bulk of the cost.

**Performance Optimization:** The mathematical simulation loops are JIT-compiled to machine code via `numba`, and generation is completely parallelized across all available CPU cores using `multiprocessing`. 

A 415-row run (30 rows/class target × 5 systems × 3 classes) finishes in approximately **35 seconds** on a standard modern CPU.

## Project status / roadmap

See `ROADMAP.md` for the three-level plan for turning this from "working
code" into an actual contribution (software package → empirical research
note → theory), and what's been done vs. still open at each level.

## License

MIT

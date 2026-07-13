---
title: 'chaos-benchmark: A rigorous dataset generator for nonlinear dynamical systems'
tags:
  - Python
  - chaos theory
  - dynamical systems
  - machine learning
  - benchmark
authors:
  - name: CS Hari Krishna
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 13 July 2026
bibliography: paper.bib
---

# Summary

Machine learning researchers frequently use chaotic dynamical systems (like the Lorenz attractor or the Logistic map) as toy datasets for evaluating time-series forecasting models. However, constructing these datasets is often treated as an afterthought. Many existing benchmarks hardcode system parameters and assume a static regime (e.g., treating all Lorenz trajectories as chaotic), ignoring the fractal nature of bifurcation boundaries and periodic windows. Furthermore, standard evaluation metrics often fail to capture the true physical unpredictability of the underlying system.

`chaos-benchmark` is an open-source Python package and interactive Web UI that generates high-fidelity datasets from five canonical nonlinear dynamical systems (Logistic map, Hénon map, Lorenz system, Rössler system, and Mackey-Glass equation). It automatically classifies each unique simulated trajectory into its true physical regime (Stable, Periodic, or Chaotic) by calculating the maximal Lyapunov exponent strictly from the simulated data.

# Statement of need

While tools like `dysts` [@Gilpin2022] provide vast databases of pre-computed chaotic attractors, `chaos-benchmark` serves a different need: it is an active, parameterized generator designed to explicitly explore the boundaries *between* dynamical regimes. It solves three critical methodological gaps common in ML-physics crossover research:

1. **Rigorous Labeling:** Rather than assigning labels based on the parameter sampling range, `chaos-benchmark` calculates the largest Lyapunov exponent for every simulated trajectory using the established method of Rosenstein et al. [@Rosenstein1993]. It uses this calculated exponent alongside trajectory variance to definitively label the time-series as Stable, Periodic, or Chaotic.
2. **Physical Forecasting Metrics:** The library introduces a *forecast horizon* metric by simulating a twin trajectory with a microscopic perturbation (e.g., $10^{-8}$) and tracking the time required for relative divergence. This provides an objective, physically grounded target for ML forecasting difficulty.
3. **High Performance and Usability:** The mathematical integration loops are highly optimized using `numba` JIT compilation, and dataset generation is parallelized via `multiprocessing`. The tool includes an interactive Web UI for real-time visualization and dataset configuration.

# Acknowledgements

We acknowledge the pioneering work of Edward Lorenz, Otto Rössler, Michel Hénon, and others whose canonical equations form the foundation of this benchmark.

# References

# Roadmap: from "working code" to an actual contribution

This document exists so we can pick this project back up in a later session
without re-deriving the plan from scratch. Three levels, roughly increasing
in difficulty and payoff. Do them in order — each one de-risks the next.

---

## Level 1 — Useful open-source tool (in progress)

**Goal**: a pip-installable, tested, honestly-documented benchmark generator
that's actually more useful to have around than writing your own from
scratch. Target venue: GitHub + submission to JOSS (Journal of Open Source
Software) — a real, low-barrier, peer-reviewed venue specifically for
research software.

### Done
- [x] Split monolithic prototype script into a proper package
      (`simulators.py`, `features.py`, `labeling.py`, `forecast.py`,
      `generate.py`, `cli.py`)
- [x] `pyproject.toml`, installable via `pip install -e .`
- [x] Console script entry point: `chaos-benchmark-generate`
- [x] 15 tests (simulator correctness, known analytic Lyapunov exponent
      cross-check, feature-extraction edge cases, labeling logic) — all
      passing
- [x] Replaced the prototype's crude single-perturbation twin-trajectory
      Lyapunov estimate with a proper Rosenstein et al. (1993) method
      (time-delay embedding + KD-tree nearest-neighbor divergence tracking)
      for the four continuous/delay systems
- [x] Found and fixed a real bug: certain Rössler/Lorenz parameter
      combinations caused the ODE solver to hang for minutes trying to
      precisely track a diverging trajectory near blow-up. Fixed with an
      early-termination event in `solve_ivp`.
- [x] Found and fixed a real methodological gap: a single global
      chaos-threshold on the Lyapunov exponent silently misclassified
      Mackey-Glass's chaotic regime as Periodic, because MG's true exponent
      magnitude (~0.003-0.005) is much smaller than the other four systems'
      (~0.02-0.5). Fixed with a per-system threshold + documented why
      Lyapunov magnitude isn't universally comparable.
- [x] README documents every known limitation found during testing
      (Rössler-Stable never fills, Lorenz-Periodic low hit rate,
      Mackey-Glass forecast horizon mostly censored, feature/label leakage)
      instead of hiding them.
- [x] Demo dataset generated (415 rows) and included in the repo.

### Still open before calling Level 1 "done"
- [ ] Fix (not just document) the Rössler-Stable gap: lengthen `t_span` for
      low-c Rössler specifically, verify it doesn't blow up runtime.
- [ ] Tighten the Lorenz-Periodic ρ sub-range with an actual literature
      citation instead of a wide best-effort sweep.
- [ ] Add a `CONTRIBUTING.md` and issue templates if actually publishing.
- [ ] Add CI (GitHub Actions running `pytest` on push).
- [ ] Consider adding 1-2 more systems (Duffing oscillator, Chua circuit)
      that were in the original wishlist but never built, OR explicitly
      scope them out in the README as "not included, here's why."
- [ ] Write the JOSS-style paper.md (short: statement of need, description,
      comparison to existing tools like `dysts`) if submitting.
- [ ] Decide license (currently MIT, fine as default) and add proper
      CITATION.cff if you want it citable.

---

## Level 2 — A specific, falsifiable empirical finding

**Goal**: a short (3-6 page) research note, posted to arXiv, testing one
concrete hypothesis rigorously enough to survive scrutiny. Not a full paper,
doesn't need journal acceptance to be "useful" — arXiv notes in this
chaos+ML crossover space get read and cited.

**The chosen hypothesis** (from our discussion): does a system's physical
fragility (Lyapunov exponent) show up as adversarial fragility in a
classifier trained on its features?

Concretely:
1. Train a regime classifier (Stable/Periodic/Chaotic) on the 13 extracted
   features (excluding `lyapunov_exponent` itself — that would be leakage).
2. For each test row, find the minimum perturbation to its feature vector
   that flips the model's prediction (cheap for 13 features — coordinate-wise
   or simple gradient search, no need for fancy adversarial-ML libraries).
3. Plot minimum-perturbation-size vs. that row's actual (ground-truth)
   Lyapunov exponent. Check correlation (Spearman, not Pearson — relationship
   is probably not linear).
4. **Falsifiable predictions stated in advance** (do this before looking at
   results): if the effect is real and system-independent, correlation
   should be positive and hold up within each of the five systems
   separately, not just pooled. If it vanishes after normalizing features to
   unit variance, that's evidence it was a feature-scaling artifact, not a
   real physical-fragility signal.
5. Repeat with 2-3 different classifier types (logistic regression, small
   NN, random forest) and multiple random seeds — a real finding survives
   being looked at from a few angles, a coincidence doesn't.
6. If there's a real signal: dig for the mechanism (e.g., do high-Lyapunov
   systems produce features that sit closer to decision boundaries in
   general, and why?). The mechanism, not the correlation number, is the
   actual contribution.
7. Write it up honestly either way — a clean null result is still a
   legitimate, useful thing to publish in this space.

### Backup ideas if the above doesn't pan out (also saved for later)
- **Rediscover the Feigenbaum constant** (~4.669) from your own logistic-map
  sweep, then check whether Rössler's period-doubling cascade (as you vary
  `c`) approaches the *same* constant — that's the actual universality claim
  worth testing, not just reproducing a known number once.
- **Unsupervised chaos taxonomy**: UMAP/t-SNE on the 13-feature vectors
  across all five systems (no labels), check whether emergent clusters
  align with *route to chaos* (period-doubling vs. other mechanisms) rather
  than *which system* — reframes the question around whether there's a
  universal statistical signature of chaos independent of the governing
  equations.

### Honest risk assessment (already discussed, repeating so we don't forget)
- Not guaranteed to show a correlation — Lyapunov exponent measures
  sensitivity of the raw trajectory to initial-condition nudges; adversarial
  fragility measures sensitivity of a classifier's decision to nudges in
  heavily-compressed summary statistics. No a priori mathematical reason
  these must correlate.
- Won't generalize to real-world "is my model robust" claims without
  testing on noisy/short/real signals — that's Level 2.5 / future work, not
  in scope for the first pass.
- A positive result on synthetic data alone is "interesting finding," not
  "established phenomenon," without the mechanistic follow-up.

---

## Level 3 — Genuine novel theory (don't start here)

Actually proving a mathematical relationship between Lyapunov exponents and
adversarial robustness bounds (if Level 2 finds a real signal worth
explaining formally). This is a theorem, not an experiment — likely months
of work, and might not even be true. **Only worth attempting if Level 2
produces a strong, robust empirical signal that justifies the investment.**
Not scoped further until that happens.

---

## Where to pick this up next time

Currently at: **Level 1, core package built and passing tests, demo dataset
generated.** Next concrete step, if continuing Level 1: fix the Rössler
low-c stable-window issue for real (longer `t_span`, check runtime cost), OR
move to Level 2 and start the classifier + adversarial-perturbation
experiment using the dataset already generated (`chaos_benchmark_dataset_demo.csv`).

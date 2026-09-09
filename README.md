# Stochastic IDS — A Mathematical Approach to Intrusion Detection

**HIMEUR Idris · DIAR Adem**

A from-scratch implementation of a probabilistic, multi-module Intrusion Detection System (IDS) on the **UNSW-NB15** network flow dataset, built as a course project reproducing and adapting the approach of:

> Murti, R. K., Joshi, V. V., Wagh, R., Jangale, M. S., Chaudhari, A. C., & Wagh, M. (2024).
> _"Stochastic Models for Cyber Attack Detection and Response: A Mathematical Approach to Intrusion Detection Systems."_
> Communications on Applied Nonlinear Analysis, 31(3s), 487–502.

The paper proposes combining **probability theory, Markov chains, queuing theory, Bayesian inference and stochastic optimization** into a single IDS pipeline instead of relying on a single signature- or anomaly-based detector. This repository is our concrete, working realization of that idea: every mathematical component described in the paper (Markov transition scoring, Bayes'-theorem signature matching, M/M/1 queueing congestion signal, logistic-regression fusion, online Bayesian parameter updating, and stochastic-perturbation threshold optimization) is implemented as a standalone, testable Python module and wired together into one training/evaluation pipeline.

**Note on scope:** the paper describes its approach at the level of bare theory, it states the general formulas (Bayes' theorem, the M/M/1 queue relations, the logistic sigmoid, the stochastic perturbation update, …) but does not provide an implementation, a worked derivation of how those formulas plug into concrete features, or any modelisation detail on how to turn them into a runnable pipeline. Every design decision needed to go from those formulas to working code — how to define a Markov chain state on real UNSW-NB15 fields, which features feed the statistical module and how they're weighted, which categorical features form the Bayesian "signature," how the three module scores are fused, and how the optimizer's objective and search procedure are structured — is our own modelisation, developed and implemented from scratch for this project rather than adapted from any reference implementation.

---

## 1. Why this design

Real cyberattacks don't all leave the same kind of fingerprint:

| Evidence type                | What it captures                                     | Example attack it catches             |
| ---------------------------- | ---------------------------------------------------- | ------------------------------------- |
| **Sequential**               | Abnormal `proto \| state` transition sequences       | Port scans, malformed handshakes      |
| **Statistical / volumetric** | Deviations in packet/byte rate, duration, congestion | Volumetric floods, DoS                |
| **Signature / categorical**  | Rare combinations of protocol, service, flags        | FTP exploits, application-layer abuse |

No single detector is good at all three, which is exactly the motivation in the base paper for combining Markov chains, statistical/queueing analysis and Bayesian signature matching, and then letting a **logistic regression fusion layer learn how much to trust each one**, rather than hand-tuning the combination.

---

## 2. Pipeline architecture

```
Raw UNSW-NB15 flow record
        │
        ▼
[1] Preprocessing    encode categoricals, scale numerics, resolve columns
        │
        ├───────────────────┬───────────────────┬───────────┐
        ▼                   ▼                   ▼           │  (each module scores every
[2] Markov Chain     [3] Statistical      [4] Bayesian      │   flow independently, in
    state model       + M/M/1 queue      signature model    │   parallel, from the raw
        │                   │               │               │   DataFrame)
        └───────────────────┴─┬─────────────┴───────────────┘
                              ▼
              [5] Logistic Regression Fusion
       X = [ preprocessed features | s_markov | s_stat | s_bayes ]
                              │
                              ▼
              [6] Bayesian Parameter Update
   (soft, incremental recalibration of modules 2–4 on the validation split —
    simulates an IDS adapting to freshly observed traffic without full retraining)
                              │
                              ▼
              [7] Stochastic Optimization
     perturbation search over the decision threshold τ (and, in a second phase,
     the fusion weights) maximizing F1(τ) − α·FPR(τ)
                              │
                              ▼
                    Final attack / normal decision
```

Modules 2, 3 and 4 read directly from the **original, unscaled DataFrame** (they need explicit column semantics — protocol strings, raw packet counts, etc.) and each emit one scalar score in `[0, 1]` per flow. Those three scores are concatenated onto the preprocessed, one-hot/scaled feature matrix before the fusion layer, which is the only module that consumes the fully numeric matrix (gradient descent needs stable scales).

---

## 3. Module-by-module

### Module 1 — Preprocessing (`src/preprocess.py`)

- One-hot encodes `proto`, `state`, `service`.
- Hash-encodes `srcip`/`dstip` into a bounded integer bucket (avoids blowing up dimensionality with raw IP strings).
- Derives `flow_duration` from `Stime`/`Ltime`.
- Median-imputes missing numeric values, `'unknown'` for missing categoricals (UNSW-NB15 has no missing values in practice, but the path exists for robustness).
- `StandardScaler` is **fit on train only** and applied to test — no leakage.
- Column names are resolved case-insensitively (`_resolve_label_col`, `_resolve_id_col`, …) because the two official UNSW-NB15 CSVs don't share identical casing/columns.
- Drops non-feature columns (row `id`, IP addresses, timestamps, `attack_cat`) before scaling.
- `FeatureSelector` (ANOVA F-score / `SelectKBest`) is available but the shipped pipeline uses the **full feature set** (no selection) — see §5.

### Module 2 — Markov Chain state-sequence model (`src/markov.py`)

- State definition: `Sₜ = proto|state` (e.g. `tcp|fin`, `udp|int`, `tcp|estab`).
- Builds a transition matrix from consecutive flows (ordered by `Stime` when available), Laplace-smoothed to avoid zero probabilities for unseen transitions, stored as **log-probabilities** to avoid numerical underflow.
- Anomaly score for flow _t_:

  ```
  Score_raw(t) = −[ 0.5·log P(Sₜ) + 0.5·log P(Sₜ | Sₜ₋₁) ]
  ```

  i.e. a blend of "how rare is this state on its own" and "how rare is this transition from the previous state." Raw scores are z-scored against the training distribution and passed through a sigmoid → `s_markov ∈ [0,1]`.

- Detects: port scans, repeated anomalous state transitions, never-before-seen protocol/state combinations.
- `bayesian_update()` performs an exponential-moving-average (EMA) blend of the transition matrix toward newly observed validation transitions (Module 6).

### Module 3 — Statistical + M/M/1 queueing anomaly model (`src/anomaly.py`)

- For ~20 numeric features (`spkts`, `dpkts`, `sload`, `dload`, `sbytes`, `dbytes`, `dur`, `sinpkt`, `dinpkt`, `tcprtt`, `synack`, `ackdat`, `sjit`, `djit`, `sttl`, `dttl`, `smean`, `dmean`, `rate`, `ct_srv_src`, `ct_srv_dst`), fits per-class Gaussians (μ, σ for normal and for attack).
- Feature weight = standardized effect size `|μ_attack − μ_normal| / pooled_std`, normalized so all weights sum to 1 — features that separate the two classes better get more say in the final score.
- Score is the weighted **log-likelihood ratio**: `AS(X) = Σⱼ wⱼ · [ log N(xⱼ; μ_a, σ_a) − log N(xⱼ; μ_n, σ_n) ]`, matching the paper's `AS(X) = P(X|H₀) / P(X)` in log-space.
- Adds an **M/M/1 queueing signal**: `λ = spkts/dur` (arrival rate), `μ = dpkts/dur` (service rate), `ρ = λ/μ` (utilization), `L = ρ/(1−ρ)` (expected queue length). The z-scored utilization `ρ` is added to the score to surface volumetric/asymmetric-traffic attacks that per-feature Gaussians alone might miss.
- Column names are resolved case-insensitively for the same reason as Module 1.

### Module 4 — Bayesian signature model (`src/bayesian.py`)

- Naïve-Bayes-style pattern matcher over categorical/discretized "signature" features: `proto`, `state`, `service`, `is_sm_ips_ports`, `is_ftp_login`, `ct_state_ttl`, `ct_flw_http_mthd` (continuous ones discretized into 5 bins).
- Implements Bayes' theorem directly, as in the paper:

  ```
  P(attack | X) ∝ P(attack) · Πⱼ P(xⱼ | attack)
  ```

  computed in log-space for stability, with Laplace smoothing on both class priors and per-feature likelihoods so unseen categorical values never zero out the posterior.

- Strength: catches attacks that are identifiable by categorical/protocol signature even when the numeric traffic volume looks unremarkable (e.g. certain FTP exploits).

### Module 5 — Logistic regression fusion (`src/classifier.py`)

- Logistic regression implemented **from scratch** (no `sklearn.linear_model`) with mini-batch gradient descent, L2 regularization, and `'balanced'` class weighting (UNSW-NB15's train/test attack rates differ — 68.1% vs 55.1% — so class imbalance handling matters).
- Input: preprocessed feature matrix concatenated with the three module scores — `X_fusion = [X_preprocessed | s_markov | s_stat | s_bayes]`.
- The model **learns how much to trust each evidence source** (i.e. the fusion weights on `s_markov`, `s_stat`, `s_bayes`) rather than combining them with a fixed, hand-picked rule — this is the practical realization of the paper's logistic-regression fusion step.
- Output is a probability `p̂ ∈ [0,1]`; a decision requires a threshold τ, which is _not_ assumed to be 0.5 — see Module 7.

### Module 6 — Bayesian parameter update (in `src/train.py`, using each module's `bayesian_update()`)

- After the fusion layer is trained, all three probabilistic modules are **softly recalibrated** on the held-out validation split — not re-initialized from scratch:
  - Markov: EMA on transition-matrix rows using observed validation transitions.
  - Statistical: class-conditional means/stds nudged toward validation observations.
  - Bayesian: priors and likelihoods blended with validation counts; vocabulary extended for any newly seen categorical values.
- This simulates how a deployed IDS would incrementally recalibrate itself as new traffic is observed, without a full retrain — directly implementing the paper's `P(θ|D) = P(D|θ)·P(θ) / P(D)` update step.

### Module 7 — Stochastic optimization (`src/optimizer.py`)

- Objective: `Objective(τ) = F1(τ) − α·FPR(τ)` with `α ≈ 0.3`. F1 is non-differentiable in the threshold, so gradient descent doesn't apply directly — the paper's stochastic perturbation scheme is used instead:

  ```
  x_{k+1} = x_k + δ_k     if f(x_{k+1}) ≥ f(x_k)
  x_{k+1} = x_k            otherwise
  ```

  with `δ_k ~ N(0, σ_k·I)` and `σ_k` decaying geometrically each iteration (broad exploration → fine-grained convergence).

- **Two-phase search:**
  - _Phase 1_ (first 60% of iterations): threshold-only, **unbiased** random walk. The search direction deliberately does _not_ use any knowledge of the test set's attack rate — doing so would be label leakage from the data the model is ultimately evaluated on. Only `X_val`/`y_val` ever inform the search.
  - _Phase 2_ (remaining 40%): joint refinement of the fusion weights and threshold together, with a smaller step size.
- Output: `best_params.threshold`, which replaces the default τ = 0.5 for final test-set evaluation.

---

## 4. Design choices worth calling out

These are lessons that shaped the implementation and are easy to miss when reading the paper alone:

- **Case-insensitive column resolution everywhere** (`markov.py`, `anomaly.py`, `preprocess.py`), because the two official UNSW-NB15 files don't consistently capitalize column names.
- **Class-imbalance handling.** Train and test attack rates differ substantially (68.1% vs 55.1%), so the fusion classifier uses balanced class weighting rather than assuming a fixed prior.
- **Calibration caveat.** The `[0,1]` scores produced by every module (including the final fusion probability) express _relative_ likelihood, not necessarily a true, well-calibrated posterior probability — no Platt scaling / isotonic regression is applied in this version.

---

## 5. Results

Measured on the official UNSW-NB15 test split (82,332 flows, 55.1% attack rate), fusion model trained on the training split (175,341 flows, 68.1% attack rate) with a 15% stratified validation hold-out.

**Module score statistics (training set):**

| Module      | Mean score | Std    |
| ----------- | ---------- | ------ |
| Markov      | 0.4766     | 0.1789 |
| Statistical | 0.5225     | 0.0982 |

**Learned fusion weights** (how much the logistic regression trusts each probabilistic module, relative to each other):

| Score          | Fusion weight                                                                              |
| -------------- | ------------------------------------------------------------------------------------------ |
| `bayes_score`  | **+0.382** (highest — categorical/signature evidence is the strongest single fused signal) |
| `markov_score` | +0.163                                                                                     |
| `stat_score`   | +0.107                                                                                     |

**Test-set performance, before vs. after stochastic threshold optimization:**

| Metric    | τ = 0.5 (default) | τ\* = 0.5737 (optimized) |             Δ |
| --------- | ----------------: | -----------------------: | ------------: |
| Accuracy  |            80.57% |                   82.97% | **+2.40 pts** |
| Precision |            75.76% |                   80.98% | **+5.22 pts** |
| Recall    |            95.15% |                   90.27% |     −4.88 pts |
| F1-score  |            84.36% |                   85.37% | **+1.02 pts** |

Confusion matrices (test set, TP/FP/TN/FN):

|     | τ = 0.5 | τ\* = 0.5737 |
| --- | ------- | ------------ |
| TP  | 43,134  | 40,921       |
| FP  | 13,799  | 9,610        |
| TN  | 23,201  | 27,390       |
| FN  | 2,198   | 4,411        |

**Interpretation:** the paper's objective (`F1 − α·FPR`) explicitly trades some recall for a large drop in false positives. The optimizer found exactly that trade-off: precision jumps by over 5 points and false positives drop by ~30%, at the cost of ~5 points of recall — a reasonable choice for an IDS where alert fatigue from false positives is a real operational cost. These are the same "before vs. after optimization" numbers produced by `evaluate.py`/`notebooks/experiment.ipynb`; note `evaluate.py`'s "before" run is computed by re-applying τ = 0.5 to the _already-optimized_ fusion weights (only the threshold changes between the two rows above; the weight-refinement effect of optimizer Phase 2 is folded into both).

For full classification reports, confusion-matrix heatmaps, per-module score distributions, and the optimizer's convergence curve, see `notebooks/experiment.ipynb`.

---

## 6. Limitations & future directions

Honest assessment of the current design, carried over from our project presentation (not otherwise included in this repository):

| Limitation                                                                                | Possible improvement                                                                |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Markov chain is order-1 only — no long-range sequential dependencies                      | Higher-order Markov chains or Hidden Markov Models                                  |
| Markov score degrades if flow ordering is lost or shuffled                                | —                                                                                   |
| Gaussian assumption in the statistical module may not hold for heavy-tailed features      | Kernel density estimation or Gaussian mixture models                                |
| Fused scores in `[0,1]` are not guaranteed true posterior probabilities                   | Platt scaling / isotonic regression for proper calibration                          |
| Fusion weights are static — the model needs retraining if the traffic distribution shifts | Online learning: continuous Bayesian update without full retraining                 |
| Binary classification only (normal vs. attack)                                            | Multi-class extension distinguishing attack subcategories (DoS, Probe, R2L, U2R, …) |

---

## 7. Project structure

```
.
├── data/                          # UNSW-NB15 CSVs — NOT tracked in git, see below
├── models/                        # trained pipeline artifacts (auto-created, gitignored)
├── notebooks/
│   └── experiment.ipynb           # end-to-end walkthrough with plots
├── src/
│   ├── preprocess.py              # Module 1 — preprocessing
│   ├── markov.py                  # Module 2 — Markov chain state model
│   ├── anomaly.py                 # Module 3 — statistical + M/M/1 queue
│   ├── bayesian.py                # Module 4 — Bayesian signature matching
│   ├── classifier.py              # Module 5 — logistic regression fusion
│   ├── optimizer.py               # Module 7 — stochastic optimization
│   ├── train.py                   # full training pipeline (modules 1–7)
│   └── evaluate.py                # test-set evaluation, before/after comparison
├── requirements.txt
└── README.md
```

### Dataset

This project uses **UNSW-NB15** (49 raw features, two official splits: 175,341 training flows / 82,332 testing flows). The CSVs are **not included in this repository** to keep it lightweight — download them and place them in `data/`:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Source: UNSW Sydney's Cyber Range Lab — search "UNSW-NB15 dataset" or find it mirrored on Kaggle. The files must be placed directly under `data/` for the default paths in `train.py` to work.

---

## 8. Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in data/

# 3. Train the full pipeline (modules 1–7)
cd src
python train.py

# 4. Evaluate on the test set
python evaluate.py

# 5. Or explore interactively
cd ../notebooks
jupyter notebook experiment.ipynb
```

`train.py` saves all fitted components (`preprocessor`, `markov`, `stat_model`, `bayes_model`, `fusion`, `optimizer`, `best_params`) as a single pickle in `models/ids_components.pkl`, plus the fused train/test feature matrices as `.npy` files, so `evaluate.py` can reload everything without retraining.

---

## 9. Citation

```
Murti, R. K., Joshi, V. V., Wagh, R., Jangale, M. S., Chaudhari, A. C., & Wagh, M. (2024).
Stochastic Models for Cyber Attack Detection and Response: A Mathematical Approach to
Intrusion Detection Systems. Communications on Applied Nonlinear Analysis, 31(3s), 487–502.
```

## License

See [LICENSE](LICENSE).

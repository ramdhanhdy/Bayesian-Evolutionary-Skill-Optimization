# M1 Toy Validation: Interpreting BESO Optimization Logs

This document provides a step-by-step guide on how to understand and interpret the rich, structured execution trace produced by the **Bayesian Evolutionary Skill Optimization (BESO)** framework.

When you execute a run like `run_toy_experiment.py` or `run_gsm8k_mini_experiment.py`, the console outputs real-time logs tracing pool mutation, fallback regimes, surrogate active learning, statistical gating, and multiplicity corrections.

---

## 1. Initial Evaluation (`[eval:validation_gate]`)

```text
[eval:validation_gate] z0 score=0.000 n=10
```

- **Core Action**: Before any mutations begin, the optimizer runs a baseline evaluation of the initial skill (`z0`) on the full validation gate split.
- **Log Indicators**:
  - `z0`: The unique ID of the baseline skill artifact.
  - `score=0.000`: The mean accuracy score across the evaluation batch.
  - `n=10`: The evaluation sample count (number of test instances used).
- **Interpretation**: Because the starting skill's core procedure is explicitly hardcoded to return a static `0` for every question, the accuracy score is naturally `0.000` (or around `0.100` for 10% accuracy on some tasks by random chance). This establishes the baseline from which the optimization path will evolve.

---

## 2. Iteration Setup & Candidate Pool Mutation

```text
[iteration 0] parents: ['z0']
[iteration 0] generated candidate pool: 23
  pool edit_1: The current core_procedure 'Return 0 for every question.' directly explains all 10 failure...
```

- **Core Action**: The optimizer identifies the parents to mutate, then invokes the `SkillOptReflectionProposer` to generate mutated candidate skill variants.
- **Log Indicators**:
  - `parents: ['z0']`: The source skill IDs selected from the top tiers of the evolutionary archive.
  - `generated candidate pool: 23`: The proposer queried the reflection LLM in structured JSON mode and successfully retrieved $23$ structurally valid, schema-validated edits.
  - `pool edit_1`: Previews of individual candidate edits showing their ID and the LLM's self-generated rationale.
- **System Safeguards**: If the LLM generates fractured JSON, the proposer automatically triggers a retry/repair loop (up to $2$ retries) providing the error back to the LLM. If repair fails, it retains only the individually schema-valid edits and drops any broken entries.

---

## 3. Cold-Start & Fallback Detection

```text
[surrogate] fitting and predicting candidate pool
[surrogate] bypassed: regime_pool_check
[selection] fallback selected: ['edit_1', 'edit_2']
```

- **Core Action**: The `VarianceRankRegimeDetector` decides if a surrogate model can be reliably fitted.
- **Log Indicators**:
  - `bypassed: regime_pool_check` / `fallback_precheck`: The regime detector has flagged that the optimization run is in a **cold-start regime** (meaning either too few historical runs have occurred, or there is negligible score variance across candidates).
  - `fallback selected`: Because the surrogate model is bypassed, selection defaults to fallback rules.
- **Interpretation**: Fitting a Gaussian process or bagging ensemble surrogate in a zero-variance or data-sparse environment would lead to severe overfitting on noisy data. To safeguard optimization integrity, the system falls back to a **greedy submodular diversity heuristic** to select a well-spread set of candidates (`edit_1`, `edit_2`) to test.

---

## 4. Mini-batch screening vs. Gate Evaluation

```text
[eval:optimization_minibatch] edit_1 score=0.000 n=3
[eval:validation_gate] edit_1 score=0.000 n=10
```

- **Core Action**: Candidates chosen by selection are physically compiled and evaluated.
- **Log Indicators**:
  - `optimization_minibatch`: A fast, cost-effective evaluation performed on a tiny training split ($n=3$) to gather basic feature data and fit surrogate coefficients.
  - `validation_gate`: A rigorous evaluation performed on the primary validation split ($n=10$) to collect high-fidelity accuracy and latency metrics.

---

## 5. Paired Statistical Gating

```text
[gate] raw paired decisions
  edit_1: accepted=False diff=0.000 ci=[0.000, 0.000] p=1.0000 threshold=0.000 reason=reject_ci:exact_mcnemar
```

- **Core Action**: The optimizer compares the candidate's validation performance directly against its parent's performance using paired hypothesis testing.
- **Log Indicators**:
  - `diff`: The raw difference in mean accuracy (`candidate - parent`).
  - `ci`: The confidence interval of the difference calculated via bootstrapping.
  - `p`: The one-sided paired p-value.
  - `reason`: The exact cause for acceptance or rejection (e.g. `reject_ci:exact_mcnemar` if the confidence interval lower bound goes below zero, or `accepted:exact_mcnemar` if the improvement is statistically significant).
- **System Safeguards**: Paired gating prevents the "winner's curse"—where a candidate appears better simply due to random luck on a small batch of questions—ensuring only robust improvements are accepted.

---

## 6. Multiplicity Correction (Benjamini-Hochberg)

```text
[gate] BH correction alpha=0.100
  edit_1: accepted=False p=1.0000 reason=reject_ci:exact_mcnemar
```

- **Core Action**: Gating decisions are adjusted across all candidates evaluated in a round to control the False Discovery Rate (FDR).
- **Log Indicators**:
  - `BH correction alpha`: The FDR threshold specified in configuration (e.g., `0.100` means a maximum 10% expected rate of false acceptances).
  - `bh_accept` / `bh_reject`: Appended to the final decision reason to indicate whether multiplicity correction overrode the raw paired gating result.
- **Interpretation**: If we evaluate multiple candidates in parallel, the probability of at least one candidate scoring high purely by chance increases dramatically. The Benjamini-Hochberg (BH) procedure scales individual p-value thresholds to protect against this false-discovery risk.

---

## 7. Active Learning & Surrogate Prediction (Warm Regime)

```text
[surrogate] fitting and predicting candidate pool
[acquisition] candidate scores
  E15: mu_raw=1.564 mu_bounded=1.000 sigma=0.277 acq=2.434
[selection] acquisition selected: ['E15', 'E12']
```

- **Core Action**: In a warm regime (enough evaluations and score variance logged), the surrogate fits successfully and scores candidates via active learning.
- **Log Indicators**:
  - `mu_raw`: The raw expected accuracy predicted by the bagging ensemble surrogate.
  - `mu_bounded`: The predicted accuracy clipped by `metric_bounds` (preventing ungrounded out-of-bounds score inflation, e.g. capping accuracy at $1.000$).
  - `sigma`: The model's prediction uncertainty (highest for un-scouted or highly novel mutation routes).
  - `acq`: The pool-normalized acquisition score, which balances exploitation (`mu_bounded`), exploration (`sigma`), candidate novelty, cost (token usage), and invalid format risk.
  - `acquisition selected`: The top-scoring candidates (`E15`, `E12`) selected to run through physical LLM rollouts.

---

## 8. Evolutionary Archiving & Pareto Routing

```text
best: e1 validation=1.000
```

- **Core Action**: Evaluated and accepted candidates are merged into the `EvolutionaryArchive` and assigned tiers (`BEST`, `PARETO`, `DIVERSE`, `FAILED`).
- **Archive Tiers**:
  - `BEST`: The single deployable incumbent that holds the overall highest validation accuracy.
  - `PARETO`: Holds candidates that are non-dominated (e.g., they are equal in accuracy to the best but use fewer tokens or run with lower latency).
  - `DIVERSE`: High-quality alternative strategies maintained to ensure the proposer does not get trapped in local optima.
- **System Safeguards for "Cleanup" Edits**: Neutral "cleanup" edits (which improve token size or latency without degrading accuracy) are accepted through the **Pareto Non-Inferiority Gate**. They are routed strictly to the `PARETO` or `DIVERSE` tiers and **never** overwrite the primary deployable `best` slot unless they separately undergo and pass a primary validation gate, preventing lucky fluctuations from corrupting the core model.

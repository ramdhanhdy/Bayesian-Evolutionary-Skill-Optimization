---
type: research-note
tags: [beso, methodology, bayesian-optimization, evolutionary-search, skill-optimization, llm-agents]
date: 2026-05-29
source:
  - [[Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification]]
  - [[Bayesian Evolutionary Skill Optimization (BESO) - Mathematical Breakdown]]
  - [[Bayesian Evolutionary Skill Optimization (BESO) - GEPA SkillOpt BESO Mathematical Lineage]]
status: draft
---

# Bayesian Evolutionary Skill Optimization (BESO): methodology

## 0. Purpose of this document

This note specifies the **operational methodology** for the BESO v0 implementation.
It translates the mathematical breakdown, the technical specification, and the
GEPA → SkillOpt → BESO lineage into a concrete, reproducible procedure, and it
records the **design decisions** that resolve the loose assumptions and edge
cases identified during the mathematical review.

Where the breakdown answers *what* the objects are and the specification answers
*what* the system should contain, this note answers *how* we run the optimizer,
*which* defaults we commit to for v0, and *why* those choices are statistically
defensible.

---

## 1. Problem restatement

BESO optimizes a structured natural-language skill artifact $z$ for a frozen LLM
system $\Phi$:

$$
z^* \in \arg\max_{z \in \mathcal{Z}} \; J(z)
= \arg\max_{z \in \mathcal{Z}} \;
\mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(\Phi(x; C(z), \Theta_{\mathrm{frozen}}), m\right)\right]
$$

subject to a rollout budget:

$$
\sum_{t=1}^{T} c(z_t, B_t) \le B
$$

The methodology treats this as a **budgeted, noisy, black-box, discrete**
optimization problem solved by Bayesian-guided evolutionary search over a
finite, reflection-generated candidate set at each iteration.

The model weights $\Theta_{\mathrm{frozen}}$ never change. The only trainable
state is the skill artifact $z$, deployed as a standalone document.

---

## 2. Methodological pillars

| Pillar | Role | Inherited from |
| --- | --- | --- |
| Skill-level abstraction | Optimize a typed, sectioned, linted artifact, not a raw prompt string | SkillOpt |
| Trajectory-grounded reflection | Convert *why* a rollout failed into bounded natural-language edits | GEPA |
| Bayesian experiment planning | Predict candidate utility and uncertainty, then spend rollouts where they are most informative | BESO (new) |
| Evolutionary archive | Preserve high performers, specialists, diverse variants, and informative failures | GEPA + SkillOpt |

BESO's distinct contribution is the layer **between proposal and evaluation**: a
surrogate $p(f\mid H_t)$ plus an acquisition function $a_t(z)$ that allocates a
limited rollout budget across a candidate pool $\mathcal{C}_t$.

---

## 3. Data regime

The dataset is split into four disjoint roles (Breakdown S1.2; Spec S16.1):

$$
D = D_{\mathrm{fb}} \cup D_{\mathrm{opt}} \cup D_{\mathrm{val}} \cup D_{\mathrm{test}}
$$

- $D_{\mathrm{fb}}$ — feedback/train: generates trajectories for reflection.
- $D_{\mathrm{opt}}$ — optimization minibatches $B_t$: fast, noisy candidate estimates.
- $D_{\mathrm{val}}$ — validation gate: accept/reject decisions.
- $D_{\mathrm{test}}$ — final test: untouched until reporting.

**Recommended default split** (small datasets): 40 / 20 / 20 / 20.
For benchmarks with a fixed test set, draw $D_{\mathrm{fb}}, D_{\mathrm{opt}}, D_{\mathrm{val}}$
from train and reserve the official test set for $D_{\mathrm{test}}$.

**Methodological guard (anti-overfitting).** Because text artifacts can encode
benchmark quirks, the validation gate is the only place acceptance is decided,
and the test split is never consulted during optimization. To limit progressive
overfitting to a fixed $D_{\mathrm{val}}$ (see S7), we rotate / periodically
refresh the validation subset and re-validate the incumbent.

---

## 4. The optimization loop

### 4.1 High-level procedure

```text
1.  Initialize and lint skill z_0; assert z_0 in Z.
2.  Evaluate z_0 on a validation seed; log trajectories, scores, feedback.
3.  Seed history H, archive A, rejected-edit buffer R.
repeat until budget B exhausted:
4.    Select parents P_t from A (Pareto win-count + UCB-style score).
5.    Reflect over traces to propose bounded edits; apply -> candidate pool C_t.
6.    Hard-filter infeasible candidates (lint / schema / budget / invariants).
7.    Deduplicate C_t.
8.    Featurize C_t (block-separated phi(z)); center features on the parent.
9.    Fit / update the Bayesian surrogate on H; predict mu_t, sigma_t.
10.   Score candidates with the pool-normalized acquisition a_BESO.
11.   Select an evaluation batch S_t via submodular (max-min / DPP) selection.
12.   Evaluate S_t on optimization minibatches; append observations to H.
13.   Gate survivors on D_val (paired test + multiplicity control + noise-scaled delta).
14.   Update archive A (best / Pareto / diverse / failed); push rejects to R.
15.   Prune archive to the size cap.
return best validated skill on the final validation set.
```

### 4.2 Update recurrence (formal)

$$
P_t \sim \pi_{\mathrm{parent}}(A_t, H_t)
\qquad
\mathcal{C}_t = G_{\psi}(P_t, H_t, R_t, A_t)
$$

$$
p_t(f) = p(f \mid H_t)
\qquad
S_t = \operatorname{Select}_{z\in\mathcal{C}_t}\; a_t(z)
$$

$$
\tilde{y}_z = \hat{J}_{B_t}(z) + \epsilon_z,\quad z\in S_t
\qquad
A_{t+1} = \operatorname{ArchiveUpdate}(A_t, S_t, H_{t+1})
$$

---

## 5. Candidate generation (reflection)

The reflection module is a proposal distribution over bounded edits:

$$
e_{t,j} \sim Q_{\psi}(e \mid z_p, \mathcal{R}, F, A_t, H_t, R_t),
\qquad
\mathcal{C}_t = \{e_{t,j}(z_p) : \nu(z_p, e_{t,j}) = 1\}
$$

- Reflection consumes parent skill, successful and failed trajectories,
  evaluator feedback, accepted-edit history, and the rejected-edit buffer.
- Output is **structured JSON** (diagnosis, failure modes, proposed edits with
  rationale and risk); malformed or budget-violating proposals are discarded.
- **Bounded edits act as a textual learning rate** $\Delta(z, z') \le \eta_{\mathrm{text}}$,
  enforced through per-iteration token/section caps. This prevents destructive
  rewrites and keeps the candidate distribution closer to stationary.

The Bayesian layer never generates text; it only ranks the finite pool that
reflection produces. BESO's ceiling is therefore bounded by reflection quality
(see S9 regime detector).

---

## 6. Surrogate and featurization (committed v0 design)

### 6.1 Featurization $\varphi(z)$

Block-decomposed and **kept separate until normalization**:

$$
\varphi(z) = \left[
\varphi_{\mathrm{text}}(z),\;
\varphi_{\mathrm{struct}}(z),\;
\varphi_{\mathrm{edit}}(z),\;
\varphi_{\mathrm{hist}}(z),\;
\varphi_{\mathrm{sem}}(z)
\right]
$$

Committed treatment:

- **Per-block standardization** to unit variance before assembly, so the
  high-dimensional text block cannot swamp the cheap structured signals.
- **Text block reduced** (PCA / random projection, ~16–32 dims) fit on the
  archive; reweighted by a block weight mirroring the composite-kernel weights
  $\alpha,\beta,\gamma$.
- **Parent-centered (delta) features**: structural features encode
  $\Delta\text{tokens}, \Delta\text{rules}, \Delta\text{checklist}$ relative to
  the parent, improving stationarity under a drifting proposal distribution.
- **Cold-start defaults** for undefined history features (e.g. edit-type success
  rate) at small $t$.

### 6.2 Surrogate

Target the **parent-relative improvement** (Breakdown S7.5):

$$
\Delta(z, z_p) = J(z) - J(z_p),
\qquad
\mu_t^{\mathrm{hybrid}}(z) = \mu_t^{\mathrm{abs}}(z) + \rho\,\mu_t^{\Delta}(z, z_p)
$$

Model: a **homogeneous bootstrap-bagged ensemble** over the reduced vector
(clean resampling-induced epistemic variance), with the composite-kernel GP path
available for very small $t$.

**Total predictive variance** (not raw disagreement):

$$
\sigma_t^2(z)
= \underbrace{\tfrac{1}{M-1}\sum_{m}\big(\hat{f}_m(z)-\mu_t(z)\big)^2}_{\text{epistemic}}
+ \underbrace{\hat{\sigma}_{\epsilon}^2(z, B)}_{\text{aleatoric}},
\qquad
\hat{\sigma}_{\epsilon}^2(z, B) \approx \frac{\hat{\sigma}^2(z)}{|B|}
$$

Predicted intervals are **recalibrated** on a held-out calibration slice
(isotonic on the PIT histogram or a single temperature scalar). The aleatoric
term ties to the noise decomposition
$\sigma_{\epsilon}^2 = \sigma_{\mathrm{model}}^2 + \sigma_{\mathrm{judge}}^2 + \sigma_{\mathrm{batch}}^2 + \sigma_{\mathrm{tool}}^2$.

---

## 7. Acquisition and batch selection (committed v0 design)

### 7.1 Pool-normalized acquisition

$$
a_{\mathrm{BESO}}(z)
= \tilde{\mu}_t(z)
+ \kappa\,\tilde{\sigma}_t(z)
+ \lambda\,\tilde{d}(z, A_t)
- \alpha\,\tilde{c}(z)
- \gamma\,\tilde{q}_{\mathrm{invalid}}(z)
$$

where each term $\tilde{(\cdot)}$ is **normalized over the current pool
$\mathcal{C}_t$** (z-score or min–max) so the weights $\kappa,\lambda,\alpha,\gamma$
are dimensionless and comparable across iterations. This removes the unit
mismatch between $\mu\in[0,1]$, distances, token costs, and probabilities.

### 7.2 Decisions

- **Exploration term decoupling.** $\sigma_t$ carries posterior uncertainty;
  the diversity term $d(z, A_t)$ is restricted to accepted/archive members and
  acts as anti-redundancy only, to avoid double-counting novelty.
- **Submodular batch selection.** For batch size $K$, select greedily by
  max-min / facility-location (or a DPP) and **update the reference set with
  already-selected members during the greedy pass**, preventing intra-batch
  near-duplicates.
- **Exploration schedule.** Non-decaying (or calibration-error-adaptive)
  $\kappa$, because the candidate domain $\mathcal{C}_t$ is regenerated each
  round; a decaying schedule can prematurely kill exploration when reflection
  opens a new region.
- **Deterministic vs probabilistic infeasibility.** Schema / budget / invariant
  violations are hard-filtered **before** acquisition (loop step 6); the learned
  penalty $\gamma\,\tilde{q}_{\mathrm{invalid}}$ targets only residual,
  non-deterministic risk (e.g. format drift).

---

## 8. Acceptance, gating, and rejection (committed v0 design)

### 8.1 Paired, noise-aware gate

Acceptance compares candidate and parent on the **same** validation draw, using
paired per-example differences $d_i = r_i(z) - r_i(p)$:

$$
\bar{d} = \frac{1}{|D_{\mathrm{val}}|}\sum_{i} d_i,
\qquad
\mathrm{CI}_{1-\alpha}(\bar{d}) = [L, U]
$$

Accept if the lower bound clears a **noise-scaled threshold**:

$$
\operatorname{Accept}(z) = 1
\iff
L > 0
\quad\text{and}\quad
\bar{d} \ge c \cdot \widehat{\mathrm{SE}}(\bar{d})
$$

For binary metrics with small $|D_{\mathrm{val}}|$, use an exact paired test
(McNemar) instead of a bootstrap CI.

### 8.2 Multiplicity control

Because each iteration tests $K$ candidates against the same split across many
iterations, raw thresholds inflate false acceptances (winner's curse). We apply
**Benjamini–Hochberg** correction across each round's candidate tests and
**periodically re-validate the incumbent** to avoid a lucky-but-false acceptance
permanently raising the bar.

### 8.3 Constraint gates

A candidate must also satisfy, on a **one-sided confidence bound** (not the noisy
point estimate):

$$
\mathrm{InvalidRate}(z) \le \rho_{\max},
\quad
\mathrm{Cost}(z) \le C_{\max},
\quad
\mathrm{Tokens}(z) \le T_{\max},
\quad
\mathrm{SchemaValid}(z) = \mathrm{InvariantValid}(z) = 1
$$

### 8.4 Rejections as negative evidence

Rejected edits enter the buffer $R_t$ with a reason and condition future
reflection. They may also contribute a similarity penalty to acquisition.

---

## 9. Archive management

The archive preserves four tiers (Breakdown S4.2):

$$
A_t = A_t^{\mathrm{best}} \cup A_t^{\mathrm{pareto}} \cup A_t^{\mathrm{diverse}} \cup A_t^{\mathrm{failed}}
$$

- **best** — high validation mean.
- **pareto** — instance- or objective-level specialists; win count
  $w(z_k) = \sum_i \mathbb{1}[z_k \in W_i]$.
- **diverse** — semantically distinct skills.
- **failed** — informative failures.

Parent selection mixes validation score, Pareto win count, diversity, and a cost
penalty (Breakdown S4.3). Pruning is constrained subset selection that trades
off validation value, coverage, diversity, and Pareto value, capped at
`max_archive_size`.

### 9.1 Regime detector (built-in ablation)

An online detector monitors candidate-score variance $\operatorname{Var}_{z\in\mathcal{C}_t}[J(z)]$
and surrogate calibration / rank-correlation. When the surrogate is not yet
predictive (cold start) or candidate variance is negligible, BESO **auto-falls
back** to greedy or random selection, so the Bayesian layer never adds overhead
without gain. This doubles as the "no surrogate" ablation.

---

## 10. Evaluation protocol and metrics

- **Decoding** defaults to temperature 0 for the target model; repeated
  evaluation is enabled for top candidates to shrink $\widehat{\mathrm{SE}}$.
- **Primary metric** is task-dependent (exact accuracy, pass@1, macro-F1,
  field-level F1, schema validity + content accuracy).
- **Secondary metrics**: token cost, latency, invalid-output rate, tool-call
  count, verbosity, calibration behavior, robustness on hard examples.
- **Primary curve**: optimization curve

$$
V(b) = \max_{z\in A(b)} \hat{J}_{\mathrm{val}}(z),
\qquad
\mathrm{AUC}_B = \frac{1}{B}\int_0^B V(b)\,db
$$

plus the final untouched-test score $\hat{J}_{\mathrm{test}}(z_{\mathrm{final}})$.

---

## 11. Scientific hypotheses

BESO's primary claim is **sample efficiency**, not merely a good final skill:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}}) > \mathrm{AUC}_B(V_{\mathrm{SkillOpt}}),
\qquad
\tau_{\gamma}^{\mathrm{BESO}} < \tau_{\gamma}^{\mathrm{SkillOpt}},
\quad
\tau_{\gamma} = \min\{b : V(b) \ge \gamma\}
$$

BESO is expected to win when rollouts are expensive, reflection produces many
plausible edits ($|\mathcal{C}_t| \gg K$), candidate quality varies, features
carry signal ($I(\varphi(z); J(z)) > 0$), and uncertainty-aware exploration
matters.

---

## 12. Baselines and ablations

**Baselines:** no skill; human skill; one-shot LLM skill; GEPA-style prompt
evolution; SkillOpt-style bounded editing; random candidate selection; greedy
reflection-ranked selection; bandit over edit types.

**BESO ablations:** no surrogate; mean-only acquisition (drop $\sigma$); no
diversity term; no cost penalty; embeddings-only vs structured-only features;
absolute-score vs delta model; no rejected-buffer features.

---

## 13. v0 commitments (summary)

| Decision | v0 default | Rationale |
| --- | --- | --- |
| Optimization target | Level-3 skill artifact | Reusable, modular, gateable |
| Surrogate target | Parent-relative $\Delta(z, z_p)$ hybrid | Stationarity under proposal drift |
| Surrogate model | Homogeneous bootstrap-bagged ensemble | Clean epistemic variance |
| Uncertainty | Epistemic + aleatoric, recalibrated | Calibrated $\sigma_t$ for acquisition |
| Featurization | Block-separated, standardized, parent-centered, reduced text | No scale domination |
| Acquisition | Pool-normalized $a_{\mathrm{BESO}}$ (UCB + diversity − cost − invalid) | Dimensionless, interpretable weights |
| Batch selection | Submodular max-min / DPP with in-batch updates | No intra-batch duplicates |
| Exploration schedule | Non-decaying / adaptive $\kappa$ | Non-stationary candidate domain |
| Gate | Paired test + BH multiplicity + noise-scaled $\delta$ | Controls winner's curse |
| Constraints | One-sided confidence bounds | Noisy constraint estimates |
| Fallback | Regime detector auto-disables surrogate | Avoids overhead without gain |

---

## 14. Acceptance criteria (v0)

1. End-to-end loop runs without manual intervention.
2. Skill artifacts remain schema-valid after edits.
3. Candidate evaluations are logged with trajectories and scores.
4. Surrogate selects candidates via acquisition.
5. Final skill improves over the initial skill on validation.
6. Final test result is reported on the untouched test split.
7. At least one baseline comparison is included.
8. Optimization trace is inspectable.

Minimum success: `BESO > initial skill` and `BESO > random edit search`.
Stronger success: `BESO >= SkillOpt-style editing` and
`BESO >= GEPA-style prompt evolution` under the same rollout budget.

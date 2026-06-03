---
type: agent-task-board
status: complete
date: 2026-06-03
milestone: M2
related_note: high-performance-plateau-and-archive-admission.md
---

# Agent Assignments: High-Performance Plateau and Archive Admission

This file is the coordination channel for follow-up work on the plateau/archive
admission decision. Agents should update only their assigned section unless they
need to add a cross-agent dependency.

The current source note is
[`high-performance-plateau-and-archive-admission.md`](high-performance-plateau-and-archive-admission.md).
The LLM-friendly references are under `docs/external_references-bib4llm/`.

## Shared constraints

- Do not weaken the deployable promotion gate.
- Do not allow archive-only exploration candidates to become deployable `best`.
- Do not treat the `n=32`, `31/32` GSM8K mini run as report-quality evidence.
- Do not tune on hidden test data.
- Do not silently change default experiment parameters.
- Do not add implementation complexity unless the BESO-specific rationale and
  evaluation path are documented first.

## Communication protocol

Each agent should append updates to its own `Agent log` section using:

```text
YYYY-MM-DD status=<not-started|in-progress|blocked|done>
- Finding:
- Evidence:
- Decision needed:
- Artifact:
```

If work is blocked, state the exact missing input. If a proposed change affects
another assignment, add a note under `Cross-agent dependencies`.

## Orchestrator closeout

2026-06-03 status=complete
- Finding: All five assigned agents completed their bounded research/documentation
  updates in this task board. The work remains at research-design level: no
  production code, ADR, default experiment parameter, or deployable promotion
  rule was changed.
- Evidence: Agent logs show `status=done` for Agent 1 reference synthesis, Agent
  2 archive-only admission semantics, Agent 3 plateau detection/protocol, Agent
  4 parent-selection pressure audit, and Agent 5 ADR readiness.
- Decision needed: Reviewers still need to choose concrete archive thresholds,
  plateau protocol labels, parent-selection trace names, and acceptance evidence
  before turning any proposal into an accepted ADR or implementation ticket.
- Artifact: This file is the consolidated coordination artifact.

## Agent 1: Reference Synthesis

- BESO problem being addressed: the current design review must not overstate
  what GEPA, SkillOpt, or the optimization textbook justify.
- Textbook concept being tested: exploration/exploitation, selection pressure,
  Pareto fronts, constraints, and adaptive search control.
- Expected benefit: keep the archive-admission rationale mathematically honest
  and source-grounded.
- Minimal scope: produce a short source map that lists the exact claims BESO may
  use from each reference and the claims it may not use.
- Success metric: every major design implication in the plateau/archive note has
  a source-backed rationale or is labeled as a BESO hypothesis.
- Failure/kill criterion: the reference review cannot distinguish between
  deployable promotion, archive admission, and parent selection.
- Likely files: `docs/notes/high-performance-plateau-and-archive-admission.md`,
  `docs/external_references-bib4llm/**`.
- What not to change: do not modify source code; do not create an ADR.

### Agent 1 log

2026-06-03 status=done
- Finding: GEPA, SkillOpt, and the optimization textbook support the source
  note's separation between deployable promotion, archive admission, and parent
  selection, but they do not justify unconditional admission of aggregate ties.
  The safe synthesis is: keep deployable promotion strict; treat archive-only
  admission as a BESO hypothesis that must require explicit search evidence and
  must be ablated.
- Evidence:
  - GEPA source map:
    - May claim: GEPA optimizes prompts with frozen model weights and a
      candidate pool, using reflective mutation or crossover from existing
      candidates. Source:
      `docs/external_references-bib4llm/GEPA-Reflective-Prompt-Evolution-Can-Outperform-RL/GEPA-Reflective-Prompt-Evolution-Can-Outperform-RL.md`,
      Section 3 "GEPA: Reflective Prompt Evolution" and Figure 4.
    - May claim: GEPA avoids always mutating the aggregate best candidate by
      using Pareto-based candidate selection over instance-wise best performers.
      Source: same file, Section 3.1 "Pareto-Based Candidate Selection" and
      Algorithm 2.
    - May claim: GEPA uses local minibatch improvement as a pool-entry
      condition: the new program is added only if the minibatch score improves,
      then it is evaluated on `Dpareto`. Source: same file, Figure 3 caption,
      Figure 4 lines 13-18, and surrounding Section 3 text.
    - May claim: GEPA's empirical evidence supports Pareto-based parent
      sampling as a search-trajectory choice relative to greedy or beam
      selection under GEPA's protocol. Source: same file, Observation 3 and
      Figure 6.
    - May claim: GEPA merge/crossover is conditional and sparse, using
      complementary Pareto-optimal lineages that improve over a shared ancestor.
      Source: same file, Appendix D.1 "Merge: System-Aware Crossover Strategy".
    - Must not claim: GEPA admits every score-neutral candidate, treats
      aggregate ties as sufficient archive evidence, or relaxes final selection
      to something other than best aggregate performance on `Dpareto`.
    - Must not claim: GEPA's results validate BESO's Bayesian surrogate,
      statistical deployable gate, or archive-only admission policy directly;
      those are BESO-specific mechanisms.
  - SkillOpt source map:
    - May claim: SkillOpt treats the skill document as the trainable external
      state while keeping the target model fixed, and uses rollout evidence plus
      bounded add/delete/replace edits. Source:
      `docs/external_references-bib4llm/SKILLOPT_Executive_Strategy_for_Self-Evolving_Agent_Skills/SKILLOPT_Executive_Strategy_for_Self-Evolving_Agent_Skills.md`,
      Sections 3.2-3.4 and Appendix C.3.
    - May claim: SkillOpt accepts candidate skills only through a held-out
      validation gate, and rejected updates are retained as negative feedback
      rather than deployed. Source: same file, Section 3.5 and Appendix C.4.
    - May claim: SkillOpt separates deployed skill compactness from
      optimizer-side memory: only `best_skill.md` is exported, while rejected
      edit buffers and meta skill guide later optimizer calls. Source: same
      file, Sections 3.5-3.7 and Appendix C.1.
    - May claim: SkillOpt's protocol distinguishes train/selection/test splits,
      uses selection only for model selection, and reports headline scores on
      held-out test data. Source: same file, Appendix C "Experimental Protocol
      Details".
    - Must not claim: SkillOpt supports a multi-candidate exploration archive,
      Pareto parent selection, archive-only candidates, or neutral admission.
      Its default acceptance rule is strict improvement on the selection split.
    - Must not claim: SkillOpt justifies a total BESO skill-artifact token cap;
      it supports bounded per-step edits and compact exported artifacts, not a
      universal final-document cap.
  - Optimization textbook source map:
    - May claim: BESO's plateau is an exploration/exploitation search-control
      problem: search methods must decide whether to exploit elite solutions or
      explore other regions, and exploitative local search can get trapped in
      local optima. Source:
      `docs/external_references-bib4llm/Optimization-Algorithms_AI-techniques-for-design-planning-and-control-problems/Optimization-Algorithms_AI-techniques-for-design-planning-and-control-problems.md`,
      Chapter 1, Section 1.5 "Search algorithms and the search dilemma".
    - May claim: hard constraints should remain non-negotiable while soft
      constraints can be modeled as rewards or penalties. Source: same file,
      Chapter 1, Section 1.3.3 "Constraints".
    - May claim: Pareto optimization preserves nondominated trade-off solutions
      for multi-objective decisions rather than collapsing everything into one
      scalar preference. Source: same file, Chapter 1, Section 1.3.2 and
      Chapter 8, Section 8.4 "Multi-objective optimization".
    - May claim: parent-selection pressure affects population diversity; high
      selective pressure and elitism can reduce diversity and cause premature
      convergence, while random selection has the lowest selective pressure.
      Source: same file, Chapter 7, Section 7.3.3 "Selection operators".
    - May claim: adaptive GA parameters can shift search behavior between
      exploration and exploitation as progress changes. Source: same file,
      Chapter 8, Section 8.5 "Adaptive GA".
    - Must not claim: the textbook prescribes BESO-specific thresholds,
      admission reasons, plateau detectors, archive pruning rules, or statistical
      promotion tests.
    - Must not claim: Pareto terminology alone implies every nondominated or
      tied candidate should be retained; BESO still needs constraints,
      evidence criteria, budget control, and pruning.
- Decision needed:
  - Label as source-backed in the plateau/archive note: strict deployable
    gating, bounded edits, rejected-edit memory, Pareto-aware parent selection,
    diversity/selection-pressure concerns, hard-vs-soft constraint separation,
    and adaptive search-control framing.
  - Label as BESO hypotheses until evaluated: exploration-only archive
    admission for primary-score non-inferior candidates, admissible positive
    search reasons beyond GEPA-style minibatch improvement, novelty thresholds,
    grounded-expansion admission, pruning behavior, parent-selection weights,
    and plateau-response policy.
  - Keep out of scope unless later evidence changes: weakening deployable
    promotion, allowing archive-only entries to become deployable `best`,
    admitting pure aggregate ties with no positive search evidence, and treating
    the `n=32`, `31/32` GSM8K mini run as report-quality evidence.
- Artifact: Source map above. No source code was modified and no ADR was
  created.

## Agent 2: Exploration-Archive Semantics

- BESO problem being addressed: useful score-neutral or near-neutral candidates
  can be discarded before they can act as mutation parents.
- Textbook concept being tested: population diversity under constrained search,
  selective pressure control, and hard versus soft constraints.
- Expected benefit: define an archive-only admission policy that preserves
  search value without weakening deployment safety.
- Minimal scope: specify admissible exploration reasons, required statistics,
  trace fields, and pruning behavior. Candidate reasons should include only
  explicit search evidence: optimization-minibatch improvement, per-example
  specialist value, meaningful novelty, or grounded expansion tied to observed
  failures.
- Success metric: the semantics can be tested with deterministic unit tests and
  can be ablated against the current archive policy.
- Failure/kill criterion: the policy admits pure aggregate ties with no positive
  search evidence, or cannot prevent archive-only entries from becoming `best`.
- Likely files: `beso/optimization/accept_reject.py`,
  `beso/optimization/loop.py`, `beso/archive/manager.py`,
  `beso/archive/pareto.py`, `tests/test_archive.py`,
  `tests/test_optimization_math.py`, `tests/test_optimization_loop.py`.
- What not to change: do not implement the policy until the semantics are
  reviewed; do not alter the paired promotion gate.

Proposed archive-only semantics:

- Scope: exploration admission is a search-state decision only. It may retain a
  non-inferior candidate as a mutation parent in Pareto/diversity-oriented
  archive tiers, but it must not replace the deployable incumbent, become
  `Archive.best()`, bypass Benjamini-Hochberg for primary promotion, or alter
  the paired promotion gate.
- Hard preconditions: the candidate must pass schema/edit validity, budget and
  safety constraints, invalid-rate/cost/latency caps, deterministic applicator
  checks, and paired primary non-inferiority against its parent on the same
  validation draw. Non-inferiority means `ci_low >= -margin` for an explicit
  exploration margin, with `margin=0.0` as the default unless a protocol
  document changes it.
- Rejection floor: a primary aggregate tie is not enough. If the candidate has
  no admissible positive search-evidence reason below, it is rejected for
  archive admission or kept only as diagnostic rejected-edit evidence.

Admissible exploration reasons:

- `optimization_minibatch_improvement`: the candidate beats its parent on the
  same optimization minibatch, with `optimization_mean_diff > 0`, at least one
  per-example improvement, and no hard-constraint violation. If the parent was
  not evaluated on that exact minibatch, this reason is unavailable.
- `per_example_specialist`: the candidate has explicit winning examples where
  it improves over the parent or matches the archive-best score while the
  parent fails, and validation non-inferiority still holds. The trace must list
  `winning_examples`, `improved_examples`, `regressed_examples`, and the net
  win/regression counts so tiny one-off specialists can be audited.
- `meaningful_novelty`: the candidate occupies a feature-space region far enough
  from retained entries to be useful for parent selection, measured by the
  configured archive distance over stored feature blocks. The reason requires a
  novelty score above the protocol threshold, nearest-neighbor id, feature
  blocks used, and hard non-inferiority. Novelty without valid features or
  without non-inferiority is not admissible.
- `grounded_failure_expansion`: the edit adds or replaces rules, examples, or
  recovery guidance grounded in observed feedback/optimization failures. The
  trace must identify the source trajectory ids, failure labels or feedback
  snippets, affected section, edit category, and the evaluated examples showing
  either a referenced-failure improvement or no regression on the referenced
  failure cluster.

Required statistics:

- Candidate and parent ids, parent tier, lineage depth, split names, seeds,
  batch ids, and evaluation sizes for validation and any optimization minibatch
  used as evidence.
- Paired primary statistics: mean diff, standard error, confidence interval,
  p-value/test method when available, non-inferiority margin, and hard
  constraint result.
- Per-example diff tables for validation and the evidence minibatch, including
  improved, tied, regressed, and winning example ids.
- Secondary statistics already used by cleanup policy: invalid rate, token/cost
  per task, latency, and any paired secondary gains.
- Novelty and Pareto context: novelty score, nearest archived neighbor,
  Pareto win count, objective vector inputs, feature blocks used, and whether
  the candidate is dominated by deployable-best-eligible entries.

Trace fields:

- `admission_mode`: one of `promotion`, `archive_exploration`, `cleanup`, or
  `reject`.
- `deployable_eligible`: boolean. This must be `false` for
  `archive_exploration`.
- `archive_parent_eligible`: boolean. This may be `true` only after all hard
  preconditions and reason-specific checks pass.
- `admission_reasons`: ordered list of reason strings from the admissible set.
- `rejection_reasons`: ordered list for failed hard constraints,
  non-inferiority failure, missing evidence, or pure aggregate tie.
- `evidence`: structured payload containing the required statistics above,
  including `source_trajectory_ids`, `winning_examples`,
  `improved_examples`, `regressed_examples`, `novelty_score`,
  `nearest_archive_id`, `paired_validation`, and `paired_optimization` when
  applicable.
- `best_exclusion_reason`: required for every archive-only entry, for example
  `archive_only_exploration` or `pareto_cleanup`.

Pruning behavior:

- The deployable incumbent and any strict-promotion entries are retained ahead
  of archive-only exploration entries. Exploration entries must never evict the
  sole deployable-best-eligible incumbent under a tight archive cap.
- Archive-only entries compete within Pareto/diversity parent pools, not within
  the deployable-best pool. They can be sampled as parents only while
  `archive_parent_eligible=true`.
- Prune exploration entries first when they lose hard constraints, become
  dominated without unique winning examples, lose meaningful novelty after new
  neighbors arrive, exceed token/cost/latency caps, or fail to produce accepted
  descendants within the protocol-defined patience window.
- Retain at most the configured exploration quota per reason/tier. When the
  quota is exceeded, prefer entries with descendant contribution, more unique
  winning examples, stronger non-inferiority evidence, higher novelty against
  the retained set, lower cost, and shorter lineage depth.

Success metric:

- In a controlled ablation with identical model, split, seeds, evaluator,
  rollout budget, and parent-selection policy, exploration admission is useful
  only if it improves budgeted validation AUC or hidden-test score, or if
  archive-only entries become ancestors of later strict-promotion candidates.
  The trace must also show that parent selection actually samples the retained
  exploration tiers.

Failure/kill criteria:

- The policy admits aggregate validation ties with no admissible positive
  search-evidence reason.
- Any archive-only entry can become deployable `best` without later passing the
  unchanged strict promotion gate.
- Exploration entries do not contribute accepted descendants, materially
  increase rollout cost, latency, token length, or archive churn, widen the
  validation-to-test gap, or cannot be distinguished from noise in deterministic
  unit tests and controlled ablations.
- Required evidence cannot be collected without changing default experiment
  parameters or evaluating on hidden test data.

### Agent 2 log

2026-06-03 status=done
- Finding: current code already separates strict promotion from archive-only
  cleanup, but the broader exploration archive semantics are not implemented or
  specified in enough detail for tests.
- Evidence: `PairedBootstrapAcceptanceGate.check_non_inferiority` handles only
  secondary cleanup gains, and `EvolutionaryArchive._best_eligible` excludes
  cleanup ids from deployable `best`.
- Decision needed: reviewers must choose concrete thresholds/quotas for
  non-inferiority margin, novelty, reason-specific patience, and exploration
  tier caps before implementation.
- Artifact: added proposed Agent 2 semantics for admissible reasons, required
  statistics, trace fields, pruning behavior, success metric, and kill criteria.

## Agent 3: Plateau Detection and Protocol

- BESO problem being addressed: saturated validation draws can spend rollout
  budget while making strict promotion impossible.
- Textbook concept being tested: adaptive search control under stagnation and
  limited evaluation budget.
- Expected benefit: define when BESO should stop, switch evaluation regime, or
  switch to archive-only exploration mode.
- Minimal scope: draft plateau diagnostics and an M2 protocol covering no-skill
  baseline, minimal-seed baseline, larger or repeated paired validation draws,
  rotating feedback minibatches, fixed seeds, clean provenance, and early-stop
  criteria.
- Success metric: the protocol can distinguish saturated smoke-test behavior
  from meaningful sample-efficiency improvement.
- Failure/kill criterion: the detector would fire during ordinary noisy progress
  or produce experiment traces that cannot be compared.
- Likely files: `docs/experiments/protocol.md`,
  `docs/experiments/baselines.md`, `docs/experiments/README.md`,
  experiment configs, and run-manifest docs.
- What not to change: do not rerun the saturated GSM8K mini configuration as a
  benchmark claim; do not change runtime defaults silently.

### Agent 3 log

2026-06-03 status=done
- Finding: The observed GSM8K mini plateau is a saturated validation-regime
  diagnostic, not benchmark evidence. With a minimal seed at `31/32`, the fixed
  validation draw leaves one correctable failure and cannot satisfy the strict
  paired promotion gate even if a candidate reaches `32/32` with no regressions.
- Evidence: `docs/experiments/results/M2-gsm8k-mini-high-performance-plateau.md`
  records an exploratory dirty-worktree run with five completed iterations,
  `validation_gate_size: 32`, `feedback_train_size: 8`, repeated candidate
  scores of `0.969`, one observed regression to `0.938`, no accepted
  candidates, and surrogate fallback reason `regime_pool_check`. The runner
  already reports `literal_no_skill`, `minimal_seed`, and `BESO` conditions on
  shared validation IDs, but the current mini limit remains a smoke-test size.
- Decision needed: M2 report-quality runs should use a harder evaluation regime
  before archive-admission conclusions are drawn. Do not rerun the saturated
  `BESO_GSM8K_LIMIT=32`, `31/32` configuration as benchmark evidence.
- Artifact: Draft plateau diagnostics and M2 protocol below.

#### Plateau diagnostics draft

Record these diagnostics per run and per iteration before adding runtime
behavior:

- Promotion headroom: incumbent validation errors remaining on the paired gate,
  maximum possible positive delta on the current draw, and whether that delta
  can satisfy the configured paired gate and multiplicity correction.
- Gate informativeness: candidate-parent discordant counts, exact or bootstrap
  p-values, confidence interval, accepted/rejected reason, and whether all
  candidates tie the parent on the paired validation draw.
- Score variation: variance of recent observed validation means, variance of
  optimization-minibatch means, and variance of surrogate predictions; report
  existing fallback reasons such as `regime_precheck` and `regime_pool_check`.
- Search evidence despite no promotion: per-example candidate wins and losses
  versus parent, optimization-minibatch improvement over parent, invalid-output
  rate changes, token/cost changes, and whether a candidate would qualify only
  for archive exploration once Agent 2 finalizes those semantics.
- Feedback coverage: feedback minibatch IDs, overlap with previous feedback
  batches, number of distinct failures seen by the proposer, and whether
  reflection is repeatedly seeing the same narrow failure pattern.
- Provenance hygiene: immutable run ID, package version, commit hash, dirty flag,
  config path or config snapshot, model, dataset paths or dataset fingerprint,
  split sizes, fixed seeds, validation IDs, feedback minibatch seeds, budget,
  and trace path.

Plateau classification should be diagnostic-only until implemented and tested:

- `saturated_gate`: incumbent score is so high on the paired draw that no
  candidate can pass the configured promotion gate.
- `zero_variance_pool`: evaluated candidate validation outcomes tie or nearly
  tie across the recent window and surrogate predictions have too little
  variance for acquisition to matter.
- `feedback_exhausted`: rotating feedback batches no longer expose new failures
  or candidate edits repeatedly target already-solved examples.
- `ordinary_noisy_progress`: candidate outcomes vary, discordant examples exist,
  or some candidates produce plausible positive deltas; do not early-stop this
  case.

#### M2 protocol draft

Use this protocol to distinguish saturated smoke-test behavior from meaningful
sample-efficiency improvement:

- Conditions: evaluate `literal_no_skill`, `minimal_seed`, current BESO, and
  any exploration-archive BESO variant under identical target model, scorer,
  prompt wrapper, train/validation/test split source, validation IDs, feedback
  schedule, seeds, and rollout budget. Baseline validation rollouts must be
  reported separately from the BESO optimization budget.
- Baselines: keep both frozen baselines. `literal_no_skill` tests model-only
  capability; `minimal_seed` tests whether neutral formatting guidance already
  saturates the draw. If these are near ceiling, the task/model/draw is not a
  useful M2 benchmark for improvement claims.
- Validation draw: replace the `n=32` smoke gate with either a larger paired
  validation draw or repeated paired validation draws. Each candidate-parent
  decision must remain paired on the same examples. If repeated draws are used,
  predeclare the draw seeds, aggregate rule, and stopping rule before running.
- Feedback minibatches: rotate optimization feedback minibatches with fixed,
  predeclared seeds so reflection sees more than one failure pattern. Log
  minibatch IDs and overlap. Do not rotate hidden test examples into feedback.
- Seeds: predeclare run seeds for dataset batching, proposer/optimizer
  randomness, validation draws, feedback draws, bootstrap gate sampling, and
  any target-model stochasticity. With temperature `0.0`, still log the seed
  passed through the harness for comparability.
- Budget and curves: compare budgeted validation AUC and final held-out score,
  not only best final validation. Use identical `max_rollouts`, iteration caps,
  candidate-pool sizes, batch sizes, and evaluation costs across BESO variants
  unless the protocol explicitly studies that difference.
- Provenance: report only clean-worktree runs for research claims. Each
  report-quality run needs the repository run-ID format, package version, full
  commit, dirty flag, config snapshot, model identifier, dataset fingerprint,
  split sizes, seeds, trace path, and compact metrics summary.
- Test split: reserve hidden test evaluation for final selected deployable
  skills after the protocol and seeds are fixed. Do not tune plateau thresholds,
  archive semantics, parent-selection weights, or early-stop rules on test
  outcomes.
- Defaults: any larger draw, repeated validation, minibatch rotation, plateau
  detector, or early-stop policy should be opt-in in experiment config first.
  Do not silently change runtime defaults.

#### Early-stop criteria draft

Early stopping should be conservative and auditable. A run may stop or switch
to a predeclared harder regime only when all required evidence is present:

- Hard stop for saturated smoke test: incumbent promotion headroom is
  insufficient to pass the configured gate on the current paired draw, at least
  one full recent window of selected candidates has no possible promotable
  result, and the run is explicitly marked diagnostic rather than benchmark
  evidence.
- Stop for uninformative candidate pool: recent validation outcomes and
  surrogate predictions remain below the predeclared variance thresholds for a
  fixed patience window, and optimization-minibatch results do not show
  positive search evidence that an archive-only protocol is meant to study.
- Switch to larger or repeated validation: frozen baselines or the incumbent
  exceed a predeclared saturation threshold on the smoke draw, but the task is
  still intended for M2 evaluation. The switch must start a new trace segment or
  run ID with explicit provenance rather than silently continuing as the same
  comparable run.
- Switch to archive-only exploration mode: only after Agent 2's archive
  admission reasons are finalized and logged. This mode may retain candidates
  as parents but must not update deployable `best`.
- Never early-stop ordinary noisy progress: if candidates have mixed wins and
  losses, nonzero discordant counts, meaningful optimization-minibatch
  variation, or improving validation AUC under budget, continue until the
  predeclared budget or patience rule is reached.

Kill criteria for the detector or protocol:

- It fires on runs with ordinary noisy progress.
- It produces traces that cannot be paired across conditions.
- It hides validation-to-test overfitting by stopping before the fixed final
  evaluation plan.
- It makes a smoke-test plateau look like BESO success.
- It requires default runtime changes that were not explicitly configured.

## Agent 4: Parent-Selection Pressure Audit

- BESO problem being addressed: archive diversity can be cosmetic if parent
  selection still collapses onto the aggregate validation best.
- Textbook concept being tested: selective pressure, premature convergence, and
  exploration/exploitation balance in population-based search.
- Expected benefit: make Pareto and diversity tiers operational in parent
  selection.
- Minimal scope: design instrumentation for parent-selection probabilities by
  tier, lineage, validation score, Pareto win count, diversity, and cost. Propose
  an ablation matrix for validation/Pareto/diversity/cost weights.
- Success metric: traces can show whether non-best archive tiers are actually
  sampled and whether sampled lineages produce useful descendants.
- Failure/kill criterion: lowering selection pressure only creates random drift,
  token growth, archive churn, or a larger validation-to-test gap.
- Likely files: `beso/archive/manager.py`, `beso/optimization/logger.py`,
  optimization trace schema, `tests/test_archive.py`.
- What not to change: do not make random selection the default; do not tune
  parent weights on hidden test data; do not alter deployable promotion.

### Agent 4 log

2026-06-03 status=done
- Finding: The archive already computes parent-selection weights from validation
  quality, Pareto win count, diversity novelty, and cost, but the optimizer
  trace does not expose the eligible parent pool, normalized probabilities, or
  enough lineage outcome summaries to audit whether selection pressure collapses
  onto the aggregate validation best.
- Evidence: `EvolutionaryArchive.select_parents()` samples non-failed entries
  without replacement from `_parent_weights()`. `_parent_weights()` combines
  z-scored validation/optimization quality, Pareto wins, archive novelty, and
  cost using `parent_validation_beta`, `parent_pareto_beta`,
  `parent_diversity_beta`, and `parent_cost_beta`. The JSONL iteration payload
  records selected parent ids and archive snapshots, but not the candidate parent
  probability table that produced the sample.
- Decision needed: Agent 2 should define stable archive-only tier/admission
  reason names before finalizing trace field names for exploration-only parents.
  Agent 3 should decide which protocol labels distinguish plateau mode from
  ordinary optimization so selection-pressure diagnostics can be compared across
  regimes.
- Artifact: Parent-selection pressure audit design below.

#### Parent-Selection Pressure Audit Design

Goal: make parent selection auditable without changing archive admission,
deployable promotion, or the default parent-selection policy. The audit should
answer three questions:

1. Which archive entries were eligible to become parents at each iteration?
2. What probability did each eligible entry receive, decomposed by validation,
   Pareto, diversity, and cost terms?
3. Did sampled non-best tiers produce descendants with useful search outcomes?

Instrumentation should be added as an explicit trace/debug surface, not as a
behavior change. The archive should expose the same probability calculation used
by `select_parents()` through an inspectable helper such as a parent-candidate
weight table. The optimizer logger can then record the table before sampling or
immediately after sampling with the same seed.

Recommended trace shape:

```yaml
parent_selection:
  seed: 12345
  requested_parent_count: 2
  replacement: false
  config:
    parent_validation_beta: 1.0
    parent_pareto_beta: 0.25
    parent_diversity_beta: 0.25
    parent_cost_beta: 0.1
  eligible_count: 7
  selected_ids: [skill_004, skill_009]
  eligible:
    - candidate_id: skill_004
      parent_id: skill_001
      root_lineage_id: skill_000
      lineage_depth: 3
      tier: best
      admission_reason: primary_gate
      validation_mean: 0.8125
      optimization_mean: 0.78125
      pareto_win_count: 4
      diversity_novelty: 0.32
      cost_per_task: 1850.0
      invalid_rate: 0.0
      terms:
        validation_z: 1.10
        pareto_z: 0.72
        diversity_z: -0.15
        cost_z: 0.40
        weighted_logit: 1.16
      probability: 0.41
      selected: true
```

Required parent-level fields:

- Identity and lineage: `candidate_id`, `parent_id`, `root_lineage_id`,
  `lineage_depth`, `created_at_iteration`, and the selected/not-selected flag.
- Archive state: `tier`, `admission_reason` once Agent 2 defines it, and whether
  the entry is eligible for deployable `best`.
- Selection inputs: validation quality used by `_entry_quality()`,
  `optimization_mean`, `validation_mean`, `validation_se`,
  `pareto_win_count`, diversity novelty, `cost_per_task`, `latency_seconds`,
  and `invalid_rate`.
- Weight decomposition: z-scored validation, Pareto, diversity, and cost terms;
  configured beta values; final weighted logit; normalized probability.
- Sampling metadata: seed, requested parent count, eligible count, replacement
  mode, and fallback reason if the eligible pool is empty or only failed entries
  are available.

Required child/outcome linkage:

- For every generated candidate, record `candidate_id`, `parent_id`,
  `parent_tier`, `parent_probability`, `parent_rank_by_probability`, and
  `parent_lineage_depth`.
- For evaluated descendants, record optimization score, validation score,
  gate decision reason, accepted/rejected status, cleanup/archive-only flag, and
  whether the descendant later becomes a deployable promotion.
- For lineage summaries, aggregate by `root_lineage_id` and by parent tier:
  number sampled, number of children generated, number evaluated, number
  admitted to archive, number promoted, mean validation delta versus parent,
  best validation delta, token/cost growth, and invalid-rate change.

Derived diagnostics for analysis:

- Tier exposure: total probability mass and realized sample count by `best`,
  `pareto`, `diverse`, and any future exploration-only tier/reason.
- Concentration: maximum parent probability, top-1/top-3 probability mass,
  effective parent count `1 / sum(p_i^2)`, and entropy of the parent
  distribution.
- Lineage health: probability mass by root lineage, sampled lineage count,
  descendant acceptance rate by lineage, and promoted-descendant ancestry.
- Search value: for each tier/reason, descendant validation AUC contribution,
  best observed validation improvement, hidden-test score only in held-out
  protocol analysis, and cost/token growth.
- Drift checks: archive churn rate, average artifact token growth, invalid-rate
  change, and validation-to-test gap by ablation arm.

The trace must not include hidden test outcomes during optimization. Hidden-test
results may be joined only in offline protocol analysis after weights and
admission rules are frozen.

#### Ablation Matrix

Run all arms with identical task split, model, seeds, evaluator, rollout budget,
archive admission policy, and deployable promotion gate. Treat the existing
configuration as the default control, not random selection.

| Arm | Validation beta | Pareto beta | Diversity beta | Cost beta | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 current | 1.00 | 0.25 | 0.25 | 0.10 | Baseline selection pressure already in code. |
| A1 validation-only | 1.00 | 0.00 | 0.00 | 0.00 | Measures collapse under pure aggregate exploitation. |
| A2 no-cost-penalty | 1.00 | 0.25 | 0.25 | 0.00 | Tests whether cost pressure suppresses useful longer lineages. |
| A3 Pareto-forward | 0.75 | 0.75 | 0.25 | 0.10 | Tests specialist parent pressure without discarding validation. |
| A4 diversity-forward | 0.75 | 0.25 | 0.75 | 0.10 | Tests whether novelty produces useful descendants or churn. |
| A5 balanced-low-elitism | 0.50 | 0.50 | 0.50 | 0.10 | Tests lower elitism while retaining score guidance. |
| A6 cost-constrained | 1.00 | 0.25 | 0.25 | 0.30 | Tests whether stronger cost control preserves quality per rollout. |
| A7 Pareto-diverse | 0.50 | 0.75 | 0.75 | 0.10 | Tests broad archive pressure under bounded validation guidance. |

Optional stress arms only after the matrix above is understood:

| Arm | Validation beta | Pareto beta | Diversity beta | Cost beta | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| S1 uniform-diagnostic | 0.00 | 0.00 | 0.00 | 0.00 | Diagnostic lower bound for drift; do not make this default. |
| S2 high-cost-penalty | 0.75 | 0.25 | 0.25 | 0.75 | Tests whether aggressive cost pressure causes under-exploration. |

Primary success criteria:

- Non-best tiers receive measurable probability mass and realized samples when
  they exist.
- Sampled Pareto/diverse/exploration-only lineages produce admitted descendants,
  deployable promotions, or improved budgeted validation AUC under fixed budget.
- Hidden-test evaluation, performed only after frozen selection settings, does
  not show a larger validation-to-test gap than the current control.

Kill criteria:

- Effective parent count increases but descendant quality does not improve.
- Non-best sampling mostly increases rejected edits, archive churn, invalid
  outputs, token growth, or cost per useful descendant.
- Any arm weakens deployable promotion, lets archive-only entries become
  deployable `best`, or tunes weights using hidden test outcomes.

## Agent 5: ADR Readiness

- BESO problem being addressed: the research note is open; the project needs an
  ADR only after semantics and protocol are precise enough to review.
- Textbook concept being tested: objective definition and constraint discipline
  before adding mechanism.
- Expected benefit: avoid prematurely freezing an underspecified archive policy.
- Minimal scope: prepare an ADR outline with context, decision options,
  alternatives, consequences, lifecycle metadata, and acceptance evidence needed.
  The ADR should remain `proposed` until the experiments or ablations decide.
- Success metric: the ADR outline states exactly what evidence would accept,
  reject, or narrow the exploration-archive policy.
- Failure/kill criterion: the ADR reads like a commitment to implement Pareto
  archive expansion before the evidence exists.
- Likely files: `docs/decisions/README.md`,
  `docs/decisions/v0.1/`, `docs/notes/high-performance-plateau-and-archive-admission.md`.
- What not to change: do not create an accepted ADR; do not migrate unrelated
  documentation.

### Agent 5 log

2026-06-03 status=done
- Finding: The plateau/archive note is ready for a proposed ADR outline, but not
  for an accepted ADR. The decision still depends on precise archive semantics,
  protocol coverage, and parent-selection instrumentation from Agents 1-4.
- Evidence: The source note explicitly separates deployable promotion from
  archive admission, preserves the strict paired gate, rejects pure aggregate
  ties without positive search evidence, and says the next step is an ADR only
  after exploration-archive semantics and evaluation protocol are specified.
- Decision needed: Proposed ADR title: "Separate deployable promotion from
  exploration-archive admission under high-performance plateaus." Proposed
  lifecycle metadata should be:

  ```yaml
  ---
  adr: TBD
  title: Separate deployable promotion from exploration-archive admission under high-performance plateaus
  status: proposed
  introduced_in: TBD
  applies_to: TBD
  milestones: [M2]
  supersedes: null
  superseded_by: null
  source_note: docs/notes/high-performance-plateau-and-archive-admission.md
  ---
  ```

  Context outline:
  - A saturated GSM8K mini smoke run started from a `31/32` validation score, so
    strict deployable promotion was statistically impossible on that draw even
    for a `32/32` candidate.
  - The deployable gate behaved as intended and should remain a hard safety
    constraint for replacing the incumbent.
  - The current archive behavior is promotion-oriented and can discard
    non-inferior candidates that may have search value as future mutation
    parents.
  - The open question is whether BESO should add a bounded archive-only search
    state, not whether aggregate ties should become deployable best.

  Decision options to carry into the proposed ADR:
  - Option A: Keep the current archive policy and treat saturated draws as a
    protocol issue addressed by harder or larger validation regimes.
  - Option B: Add exploration-archive admission for candidates that satisfy hard
    constraints, paired primary non-inferiority, and at least one explicit
    positive search reason.
  - Option C: Add plateau detection first, then enable exploration-archive
    admission only in declared plateau regimes.
  - Option D: Instrument parent-selection pressure before changing admission, so
    the project can confirm whether existing Pareto/diversity machinery is
    operational.

  Alternatives to reject or keep out of scope unless later evidence changes:
  - Weakening the deployable promotion gate.
  - Promoting aggregate score ties directly to deployable best.
  - Admitting every neutral candidate without minibatch, per-example, novelty,
    grounded-expansion, or lineage evidence.
  - Treating the `n=32`, `31/32` GSM8K mini run as report-quality benchmark
    evidence.
  - Migrating documentation structure as part of this decision.

  Consequences to document:
  - Positive: BESO may preserve specialist or complementary lineages that a
    strict promotion archive would discard, improving low-budget search.
  - Positive: The deployable incumbent remains protected by paired statistical
    evidence and cannot be silently replaced by archive-only entries.
  - Negative: The policy adds admission reasons, thresholds, trace fields,
    pruning rules, parent-selection weights, and ablation obligations.
  - Negative: Tiny validation specialists may be noise artifacts that increase
    overfitting, token growth, archive churn, or validation-to-test gap.

  Acceptance evidence needed before moving beyond `proposed`:
  - Agent 1: source map distinguishes supported reference claims from BESO
    hypotheses, especially around GEPA, SkillOpt, and textbook optimization
    concepts.
  - Agent 2: deterministic archive semantics specify admissible positive search
    reasons, required statistics, trace fields, pruning behavior, and the rule
    that archive-only entries cannot become deployable `best`.
  - Agent 3: M2 protocol covers no-skill baseline, minimal-seed baseline,
    larger or repeated paired validation draws, rotating feedback minibatches,
    fixed seeds, clean provenance, and early-stop or plateau-response criteria.
  - Agent 4: parent-selection traces show sampling probability by archive tier,
    lineage, validation score, Pareto win count, diversity, and cost.
  - Controlled ablation compares current archive policy against the proposed
    exploration-archive policy under identical model, splits, seeds, evaluator,
    and rollout budget.
  - Success requires improved budgeted validation AUC or hidden-test
    performance, or evidence that exploration-only candidates become ancestors
    of later deployable promotions.
  - Rejection or narrowing is required if exploration-only candidates increase
    cost or token length materially, widen the validation-to-test gap, create
    archive churn, or fail to contribute useful descendants.
- Artifact: This log entry is the ADR readiness outline. No accepted ADR was
  created, and no documentation migration was performed.

## Cross-agent dependencies

- Agent 2 depends on Agent 1 for precise source claims.
- Agent 3 depends on Agent 2 for which exploration-admission variants need
  protocol coverage.
- Agent 4 depends on Agent 2 for the archive tiers and admission reasons that
  should appear in traces.
- Agent 5 depends on Agents 1-4 before drafting a credible proposed ADR.

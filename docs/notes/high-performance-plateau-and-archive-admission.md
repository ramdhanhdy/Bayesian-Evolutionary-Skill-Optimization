---
type: research-note
status: open
date: 2026-06-01
milestone: M2
related_run: ../experiments/results/M2-gsm8k-mini-high-performance-plateau.md
---

# High-Performance Plateau and Archive Admission

## Context

The GSM8K mini runner exposed a high-performance plateau when a capable frozen
model starts from a neutral skill. In the observed exploratory run, the seed
artifact scored `31/32 = 0.96875` on the validation gate. Most evaluated
mutations tied that score, one regressed to `30/32`, and none entered the
archive.

This is not evidence that the acceptance gate or regime detector is broken. It
is evidence that BESO currently uses a promotion-oriented archive admission
policy in a regime where statistically demonstrable promotion is nearly
impossible.

## What Is Working As Intended

### Deployable Promotion Gate

The strict gate correctly rejects score-neutral edits:

```text
diff=0.000
ci=[0.000, 0.000]
p=1.0000
reason=reject_ci:exact_mcnemar
```

It also rejects observed regressions. This behavior protects the deployable
incumbent from winner's-curse fluctuations and should not be weakened casually.

### Surrogate Regime Detector

The surrogate is bypassed when candidate outcomes have negligible variance:

```text
[surrogate] bypassed: regime_pool_check
```

This is consistent with the BESO design. A surrogate cannot produce a useful
ranking when evaluated candidates repeatedly tie. Forcing Bayesian selection in
that regime would add computation without adding reliable information.

## Statistical Dead Zone

With a fixed gate of `n=32` and a baseline score of `31/32`, only one validation
failure remains. The maximum possible accuracy improvement is:

```text
1 / 32 = 0.03125
```

If a candidate fixes that one failure with no regressions, the one-sided exact
McNemar p-value is:

```text
p = 0.5
```

The current gate requires `p <= 0.10`. Therefore, no candidate can achieve
strict deployable promotion on this draw, even if it reaches `32/32`.

This is a property of the saturated validation draw, not a mutation-quality
failure.

## Search Bottleneck

BESO currently conflates two decisions:

1. Should a candidate replace the deployable incumbent?
2. Should a candidate remain in the evolutionary population as a useful search
   parent?

These decisions need different standards.

The deployable incumbent should continue to require paired statistical evidence
and multiplicity correction. However, a score-neutral candidate may still be a
useful stepping stone if it:

- wins on a different subset of examples;
- introduces a complementary specialist strategy;
- adds reusable structure for later mutations;
- expands the skill with grounded rules, examples, or recovery guidance;
- occupies a novel region of feature space without materially regressing.

At present, these candidates are usually discarded unless they improve a
secondary cleanup metric such as token count, invalid-output rate, or latency.
That biases retained lineages toward compression and cleanup rather than
comprehensive skill growth.

## Relation To GEPA

GEPA retains a Pareto frontier of candidates that excel on different task
subsets, reflects on trajectories, mutates candidates, and can merge
complementary lessons. Its population is not reduced to a single
aggregate-score hill climb.

The GEPA algorithm is not evidence for admitting every score-neutral candidate.
It first evaluates a new candidate on a feedback minibatch and only adds it to
the candidate pool when that minibatch score improves over the parent. The
Pareto machinery is then used for parent selection among candidates that already
showed local search value.

BESO should preserve its Bayesian screening and statistically strict deployable
promotion gate while recovering that broader evolutionary population behavior.
For BESO, the relevant GEPA lesson is therefore not "archive all ties." It is:
avoid always mutating only the current aggregate best, and preserve candidates
that have explicit evidence of per-example, minibatch, or lineage-level search
value.

References:

- [GEPA repository](https://github.com/gepa-ai/gepa#how-it-works)
- [GEPA paper](https://arxiv.org/abs/2507.19457)

## Textbook-Informed Decision Review

### Relevant textbook lens

- Concept: exploration versus exploitation, local optima, selection pressure,
  population diversity, Pareto or multi-objective optimization, hard versus soft
  constraints, and adaptive search control.
- Textbook location: `docs/external_references-bib4llm/Optimization-Algorithms_AI-techniques-for-design-planning-and-control-problems/Optimization-Algorithms_AI-techniques-for-design-planning-and-control-problems.md`,
  especially chapter 1 on optimization ingredients, constraints, and the search
  dilemma; chapter 7 on genetic algorithm parent-selection pressure; chapter 8
  on Pareto fronts, multi-objective optimization, and adaptive GA controls; and
  chapter 12 on UCB-style exploration under uncertainty.
- Short explanation: the plateau is not primarily a failure of mutation or the
  statistical gate. It is a search-control problem. Aggregate-score exploitation
  has saturated on the current validation draw. The textbook lens says BESO
  needs either a harder evaluation regime or an explicit, bounded mechanism for
  preserving diverse candidates without letting selection pressure collapse the
  population onto one incumbent.

### Related LLM-optimizer references

- GEPA location: `docs/external_references-bib4llm/GEPA-Reflective-Prompt-Evolution-Can-Outperform-RL/GEPA-Reflective-Prompt-Evolution-Can-Outperform-RL.md`,
  especially Section 3 and Section 3.1.
- SkillOpt location: `docs/external_references-bib4llm/SKILLOPT_Executive_Strategy_for_Self-Evolving_Agent_Skills/SKILLOPT_Executive_Strategy_for_Self-Evolving_Agent_Skills.md`,
  especially Sections 3.2-3.6 and 4.3-4.4.
- Short explanation: GEPA supports Pareto-based parent selection and diverse
  search trees, but it still requires local minibatch improvement before a new
  candidate enters the pool. SkillOpt supports strict deployable gating,
  bounded text updates, rejected-edit evidence, and compact exported skills.
  Together they argue for archive-only exploration evidence, not relaxed
  deployment acceptance.

### Mapping to BESO

- BESO component: deployable promotion, archive admission, parent selection,
  Pareto tiers, diversity tiers, rejected-edit memory, and plateau detection.
- Why it matters: the deployable incumbent is an exploitative state and should
  be protected by paired statistical evidence. The archive is a search state and
  can reasonably use weaker, archive-only standards if those standards are
  explicit, auditable, and prevented from silently replacing the deployable
  best.
- Current weakness or uncertainty: BESO already has multi-tier archive machinery
  and a cleanup path for non-inferior candidates that improve secondary metrics.
  However, candidates that are score-neutral but useful as specialists,
  grounded expansions, or novel stepping stones still usually die before Pareto
  or diversity logic can help. Conversely, admitting every neutral candidate
  would be unjustified. In textbook terms, BESO needs lower selection pressure
  and more population diversity, but only under constraints that prevent the
  archive from becoming an uncontrolled cache of noise.

### Design implication

- Recommended direction: keep strict deployable promotion unchanged, and test a
  separate exploration-archive admission path for archive-only parents.
  Admission should require hard-constraint satisfaction, primary-metric
  non-inferiority, and at least one positive search reason: optimization
  minibatch improvement over the parent, per-example specialist value,
  meaningful feature-space novelty, or grounded expansion tied to observed
  failures. Pure aggregate ties with no positive search evidence should remain
  rejected or recorded only as negative/diagnostic evidence.
- Alternative: do not add exploration admission yet. Treat this as a saturated
  benchmark artifact and move to larger validation draws, repeated paired draws,
  harder tasks, weaker target models, or early stopping before changing archive
  semantics.
- What evidence would decide: compare the current archive policy against an
  exploration-archive policy under identical model, split, seed, evaluator, and
  rollout budget. The exploration archive is useful only if it improves
  budgeted validation AUC or hidden-test performance, or if exploration-only
  candidates become ancestors of later deployable promotions. Also measure
  whether parent selection actually samples the new archive tiers rather than
  collapsing back to the validation-best incumbent.

### Risk / caveat

- Where this could be wrong: per-example specialists on a tiny validation draw
  can be noise artifacts. Preserving them may increase overfitting, prompt
  bloat, lineage complexity, and parent-selection variance without improving
  final skill quality. GEPA's evidence is about Pareto parent selection after
  local improvement, not unconditional neutral admission.
- Complexity cost: exploration admission adds another policy layer: admission
  reasons, non-inferiority margins, novelty thresholds, parent weights, pruning
  priorities, trace logging, and ablation requirements. It is justified only if
  it improves low-budget search, not merely because Pareto machinery exists.

## Artifact-Length Clarification

BESO does not need a total skill-artifact token cap. The current bounded edit
rule should be interpreted as a per-mutation limit, not a final-document limit.
A lineage should be able to accumulate multiple grounded additions over time.

The remaining limitation is archive admission: if neutral expansion edits never
survive, comprehensive artifacts cannot accumulate even when the patcher allows
them.

## Candidate Design Direction

The following direction is proposed for evaluation, not yet accepted:

1. Keep strict deployable promotion unchanged.
2. Add a separate exploration-archive admission path for non-inferior candidates
   that also show explicit search value: optimization-minibatch improvement,
   per-example Pareto specialization, meaningful novelty, or grounded expansion
   tied to observed failures.
3. Preserve specialist candidates as eligible mutation parents without letting
   them silently replace the deployable best.
4. Add explicit merge or macro-mutation proposals that can combine
   complementary lessons from retained specialists only after the archive has
   demonstrably distinct successful lineages.
5. Add plateau detection so a saturated benchmark can stop early or switch to
   a harder evaluation regime.
6. Evaluate the surrogate only when the regime detector observes enough score
   variation to justify Bayesian ranking.

## Experiment-Protocol Implications

Before making benchmark claims:

1. Measure a literal no-skill baseline, a minimal-seed baseline, and BESO under
   identical model and scoring settings.
2. Treat `BESO_GSM8K_LIMIT=32` as a smoke test only.
3. Use larger hidden validation draws or repeated paired draws when the baseline
   is near saturation.
4. Rotate feedback minibatches so reflection sees more than one narrow failure
   pattern.
5. Use a harder benchmark or a weaker task model when the research question is
   whether BESO can improve capability under budget.
6. Record an immutable run ID, clean commit, exact configuration, and compact
   summary before using a run in a report.

## Bounded Investigation Tickets

### Ticket 1: Specify exploration-archive admission semantics

- BESO problem being addressed: score-neutral, potentially useful candidates
  are discarded during high-performance plateaus before they can serve as
  mutation parents.
- Textbook concept being tested: population diversity and exploration under
  local-optimum or plateau conditions; selection pressure control; hard versus
  soft constraints.
- Expected benefit: preserve non-inferior stepping stones without weakening the
  deployable promotion gate.
- Minimal implementation scope: define archive-only admission criteria and trace
  fields. Require hard constraints, paired primary non-inferiority, and at least
  one positive search-value reason. Do not change promotion, final-skill
  selection, or default benchmark claims.
- Success metric: admitted exploration-only candidates later appear in promoted
  lineages or improve budgeted validation AUC / hidden-test score in a controlled
  ablation.
- Failure/kill criterion: exploration-only candidates do not contribute to
  later accepted descendants, increase cost or token length materially, or widen
  the validation-to-test generalization gap.
- Files or modules likely involved: `beso/optimization/accept_reject.py`,
  `beso/optimization/loop.py`, `beso/archive/manager.py`,
  `beso/archive/pareto.py`, and archive/optimization tests.
- What not to change: do not weaken the deployable gate; do not allow
  exploration-only candidates to become `best`; do not admit pure aggregate
  ties that lack positive search evidence; do not change default experiment
  parameters silently.

### Ticket 2: Add plateau detection and response policy

- BESO problem being addressed: saturated validation draws can consume rollout
  budget even when strict promotion is impossible or uninformative.
- Textbook concept being tested: adaptive search control for balancing
  exploration and exploitation as search progress changes.
- Expected benefit: stop early, switch to harder evaluation, or switch to
  exploration-archive mode when the current draw has no promotion headroom.
- Minimal implementation scope: specify plateau diagnostics and allowed policy
  responses before implementing any runtime behavior.
- Success metric: plateau detection correctly identifies saturated smoke-test
  regimes and prevents misleading benchmark interpretation.
- Failure/kill criterion: detector fires on ordinary noisy progress, masks real
  improvement opportunities, or creates incomparable experiment traces.
- Files or modules likely involved: `beso/optimization/regime.py`,
  `beso/optimization/loop.py`, experiment configs, trace logger, and M2 protocol
  docs.
- What not to change: do not force the surrogate on in zero-variance regimes;
  do not use plateau detection as evidence that BESO has succeeded.

### Ticket 3: Evaluate harder protocol before benchmark claims

- BESO problem being addressed: the `n=32`, `31/32` GSM8K mini run cannot support
  claims about BESO quality or archive policy.
- Textbook concept being tested: objective definition, constraint control, and
  fair comparison under stochastic search.
- Expected benefit: separate saturated smoke-test behavior from meaningful
  sample-efficiency evaluation.
- Minimal implementation scope: define an M2 protocol with no-skill and
  minimal-seed baselines, larger or repeated paired validation draws, rotating
  feedback minibatches, fixed seeds, and clean provenance.
- Success metric: the protocol can distinguish no-skill, minimal seed, current
  BESO, and exploration-archive BESO under identical budgets.
- Failure/kill criterion: the target model remains saturated, score variance is
  negligible, or the protocol cannot produce paired comparable traces.
- Files or modules likely involved: `docs/experiments/protocol.md`,
  `docs/experiments/baselines.md`, experiment configs, and run manifest docs.
- What not to change: do not treat smoke-test results as leaderboard results;
  do not mix dirty-worktree exploratory traces with report-quality evidence.

### Ticket 4: Audit parent-selection pressure under plateau

- BESO problem being addressed: even if exploration-only candidates enter the
  archive, parent selection may still over-sample the aggregate validation best
  and behave like a local hill climb.
- Textbook concept being tested: selective pressure in population-based search
  and premature convergence from excessive elitism.
- Expected benefit: make archive diversity operational rather than cosmetic.
- Minimal implementation scope: instrument parent-selection probabilities by
  archive tier and lineage; run an ablation of validation, Pareto, diversity,
  and cost weights without changing admission rules.
- Success metric: parent sampling includes useful Pareto/diverse lineages and
  those lineages produce descendants that improve validation AUC or hidden-test
  score.
- Failure/kill criterion: lower selection pressure only increases random drift,
  archive churn, token growth, or validation-to-test gap.
- Files or modules likely involved: `beso/archive/manager.py`,
  `beso/optimization/logger.py`, optimization trace schema, and archive tests.
- What not to change: do not make random selection the default; do not tune
  weights on the test split; do not let parent-selection changes alter
  deployable promotion criteria.

## Non-Decisions

This note does not authorize:

- weakening the deployable acceptance gate;
- forcing the surrogate on in a zero-variance regime;
- promoting aggregate ties directly to deployable best;
- treating a 32-example smoke run as a leaderboard comparison.

The next step is to turn the candidate design direction into an ADR after the
exploration-archive semantics and evaluation protocol are specified precisely.

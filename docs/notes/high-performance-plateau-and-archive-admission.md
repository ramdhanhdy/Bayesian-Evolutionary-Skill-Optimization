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

BESO should preserve its Bayesian screening and statistically strict deployable
promotion gate while recovering that broader evolutionary population behavior.

References:

- [GEPA repository](https://github.com/gepa-ai/gepa#how-it-works)
- [GEPA paper](https://arxiv.org/abs/2507.19457)

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
2. Add a separate exploration-archive admission path for non-inferior,
   per-example Pareto specialists and novel stepping stones.
3. Preserve specialist candidates as eligible mutation parents without letting
   them silently replace the deployable best.
4. Add explicit merge or macro-mutation proposals that can combine
   complementary lessons from retained specialists.
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

## Non-Decisions

This note does not authorize:

- weakening the deployable acceptance gate;
- forcing the surrogate on in a zero-variance regime;
- promoting aggregate ties directly to deployable best;
- treating a 32-example smoke run as a leaderboard comparison.

The next step is to turn the candidate design direction into an ADR after the
exploration-archive semantics and evaluation protocol are specified precisely.


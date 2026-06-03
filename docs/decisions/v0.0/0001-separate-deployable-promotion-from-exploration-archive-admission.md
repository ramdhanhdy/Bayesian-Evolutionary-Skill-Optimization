---
adr: 0001
title: Separate deployable promotion from exploration-archive admission under high-performance plateaus
status: proposed
introduced_in: 0.0.1
applies_to: "proposal only; no accepted exploration-admission runtime policy"
milestones: [M2]
supersedes: null
superseded_by: null
source_note: docs/notes/high-performance-plateau-and-archive-admission.md
task_board: docs/notes/high-performance-plateau-agent-assignments.md
---

# ADR 0001: Separate deployable promotion from exploration-archive admission under high-performance plateaus

## Context

An exploratory GSM8K mini run started from a seed skill that scored `31/32` on
the validation gate. With that saturated binary draw, even fixing the remaining
failure would produce only one discordant improvement, giving an exact one-sided
McNemar p-value of `0.5`. Under the configured gate, strict deployable promotion
was therefore impossible on that draw.

This does not imply that the deployable gate is wrong. It means BESO needs to
keep separate two decisions:

- whether a candidate may replace the deployable incumbent;
- whether a candidate may remain in the evolutionary population as an
  archive-only parent.

The source review supports strict deployable gating, Pareto-aware parent
selection, bounded edits, rejected-edit evidence, and adaptive search-control
diagnostics. It does not support unconditional admission of aggregate ties.

## Decision

This ADR is proposed, not accepted.

BESO should preserve the strict deployable promotion gate. Archive-only entries
must be explicitly marked as not deployable-best eligible, and traces must make
their admission mode, admission reasons, parent eligibility, and best-exclusion
reason auditable.

The current implementation may expose supporting metadata and diagnostics:

- archive admission metadata on retained entries;
- archive-only cleanup and exploration eligibility fields;
- parent-selection probability tables using the existing weighted sampler;
- plateau diagnostics for saturated binary validation draws.

The implementation must not automatically admit score-neutral candidates merely
because they tie aggregate validation. Exploration-archive admission remains a
future opt-in policy until reviewers choose concrete thresholds and protocols
for non-inferiority, novelty, per-example specialist value, grounded expansion,
reason quotas, and pruning patience.

## Alternatives Considered

Keep the current archive policy only. This avoids policy complexity, but it can
discard non-inferior specialist or expansion candidates before parent selection
can test their search value.

Weaken deployable promotion. Rejected. The plateau is not evidence that the
statistical gate is defective, and weakening it would make winner's-curse
promotion more likely.

Promote aggregate ties directly to deployable best. Rejected. A tie on a small
validation draw is not evidence of improvement.

Admit every neutral candidate to the archive. Rejected. GEPA-style population
search requires local search evidence; the textbook does not justify retaining
unbounded noise under a finite rollout budget.

Instrument parent-selection pressure before changing admission. Accepted as a
low-risk precursor. It can show whether existing Pareto/diversity machinery is
operational without changing admission behavior.

## Consequences

Positive consequences:

- The deployable incumbent remains protected by the unchanged paired gate.
- Archive-only entries can be represented without becoming deployable best.
- Parent-selection pressure can be audited by tier, lineage, score, Pareto wins,
  diversity, and cost.
- Saturated smoke-test validation draws can be labeled as diagnostics rather
  than benchmark evidence.

Negative consequences:

- The archive schema and trace payload become more complex.
- Exploration-admission policy still needs thresholds and ablations before it
  can be accepted.
- Per-example specialists may be noise artifacts if later admitted under weak
  evidence.

## Version Lifecycle

Status is `proposed` in `0.0.1`.

To move to `accepted`, BESO needs a controlled M2 ablation comparing current
archive behavior against an opt-in exploration-archive policy under identical
model, splits, seeds, evaluator, and rollout budget. Acceptance requires
improved budgeted validation AUC, held-out performance after frozen settings, or
evidence that archive-only entries become ancestors of later deployable
promotions.

The proposal should be rejected or narrowed if exploration-only entries increase
cost, token length, archive churn, invalid outputs, or validation-to-test gap
without producing useful descendants.

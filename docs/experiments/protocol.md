# Experiment Protocol

This document defines the minimum protocol information needed before an
experiment result can support a release note, report, or research claim.

## Minimum Run Metadata

- Run ID.
- Package version.
- Git commit SHA.
- Dirty-worktree status.
- Config path and resolved configuration.
- Dataset name and split sizes.
- Random seed.
- Target model and optimizer model.
- Evaluation metric and scoring rule.
- Rollout budget and candidate pool settings.

## Evidence Levels

Exploratory evidence:

- May use small samples or temporary settings.
- Useful for debugging and hypothesis generation.
- Must not be presented as report-quality.

Report-quality evidence:

- Uses a clean Git commit.
- Records exact config and seed.
- Uses documented splits and metrics.
- Has enough sample size for the claim being made.
- Includes compact result notes under `docs/experiments/results/`.

## Baseline Discipline

Baseline and BESO comparisons must use identical tasks, seeds, splits, target
models, decoding settings, scoring rules, and rollout budgets unless the
experiment explicitly studies a difference.


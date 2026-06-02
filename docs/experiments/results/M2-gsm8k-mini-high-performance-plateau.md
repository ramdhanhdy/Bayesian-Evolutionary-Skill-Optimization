# M2 Exploratory Result: GSM8K Mini High-Performance Plateau

## Status

Exploratory diagnostic only. This is not a report-quality benchmark result.

The run was executed from a dirty worktree and was interrupted while iteration
`5` was starting. The JSONL artifact contains five completed iterations.

## Provenance

```yaml
observed_at: 2026-06-01T18:37:27+07:00
trace: artifacts/gsm8k_mini_experiment_20260601_183727.jsonl
package_version: 0.0.1
git_commit: 95e8328
git_dirty: true
model: deepseek-v4-flash
seed_mode: baseline
completed_iterations: 5
validation_gate_size: 32
feedback_train_size: 8
```

The log excerpt is the source of truth for this exploratory summary. It should
not be compared against leaderboard results.

## Baseline

The neutral seed artifact scored:

```text
[eval:validation_gate] gsm8k_z0 score=0.969 n=32
[eval:feedback_train] gsm8k_z0 score=0.875 n=8
```

The validation result corresponds to `31/32` correct answers.

## Completed Iterations

| Iteration | Selected candidates | Validation scores | Accepted |
| --- | --- | --- | --- |
| 0 | `13`, `23` | `0.969`, `0.969` | none |
| 1 | `e12`, `e17` | `0.969`, `0.938` | none |
| 2 | `ed19`, `ed1` | `0.969`, `0.969` | none |
| 3 | `1`, `2` | `0.969`, `0.969` | none |
| 4 | `11`, `23` | `0.938`, `0.969` | none |

The surrogate was bypassed after attempted fitting because candidate-pool score
variance remained too small:

```text
[surrogate] bypassed: regime_pool_check
```

## Interpretation

The strict gate and regime detector behaved consistently with the current
mathematical contracts:

- tied mutations did not replace the incumbent;
- regressing mutations were rejected;
- the surrogate abstained from ranking nearly indistinguishable candidates.

However, the run revealed a search-policy limitation. Score-neutral variants
did not survive as mutation parents, so the evolutionary population could not
accumulate complementary specialist lessons or more comprehensive skill
artifacts.

See [High-performance plateau and archive admission](../../notes/high-performance-plateau-and-archive-admission.md)
for the design analysis and proposed next questions.

## Limitations

- The run is incomplete.
- The worktree was dirty.
- The validation gate contains only `32` examples.
- A `31/32` baseline leaves insufficient headroom for strict paired promotion.
- The result is not a full GSM8K evaluation.

## Next Experiment

Do not rerun this configuration as evidence of BESO quality. First specify the
M2 protocol for:

- literal no-skill and minimal-seed baselines;
- a larger hidden validation draw or repeated paired draws;
- rotating feedback minibatches;
- exploration-archive admission separate from deployable promotion;
- plateau detection and early stopping.


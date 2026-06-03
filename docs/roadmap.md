# BESO Roadmap

The project tracks package versions, research milestones, and experiment runs as
separate concepts.

## Version Targets

| Version | Target |
| --- | --- |
| `0.0.x` | Prototype implementation, hardening, and research scaffolding |
| `0.1.0` | Reproducible toy validation |
| `0.2.0` | Benchmark runner and baseline comparisons |
| `0.3.0` | Ablation suite and reporting |
| `0.4.0` | Transfer experiments |
| `1.0.0` | Stable public API and reproducible research release |

## Research Milestones

| Milestone | Meaning |
| --- | --- |
| `M0` | Core optimizer prototype |
| `M1` | Reproducible toy validation |
| `M2` | Benchmark evaluation |
| `M3` | Low-budget stress tests |
| `M4` | Ablations |
| `M5` | Transfer evaluation |
| `M6` | Paper-ready replication package |

## Current Focus

The current public target is to make `v0.1.0` a clean reproducibility checkpoint.

Exit criteria:

- Toy experiment runs from a documented command.
- Exact config, seed, model, commit, and dirty-worktree status are recorded.
- Run artifacts use immutable run identifiers.
- Tests pass.
- README reflects implemented behavior.
- A compact result summary exists under `docs/experiments/results/`.

## Next Research Questions

- How should BESO separate deployable promotion from exploration-only archive
  admission?
- Which plateau-response protocol is justified for high-performing frozen
  models?
- Which ablations isolate the Bayesian acquisition contribution from the
  reflection and archive components?
- Which run manifest fields are required before a result is report-quality?


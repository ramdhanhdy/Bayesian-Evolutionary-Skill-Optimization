# BESO: Bayesian Evolutionary Skill Optimization

BESO optimizes a structured natural-language **skill artifact** for a frozen LLM
agent by combining trajectory-grounded reflective mutation, a Bayesian surrogate
over candidate utility, a pool-normalized acquisition layer that allocates a
limited rollout budget, and an evolutionary archive.

It is a research fork of [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)
(MIT). BESO reuses SkillOpt-style execution plumbing and replaces the greedy
one-edit propose/evaluate/gate loop with a Bayesian experiment-planning layer.

## Objective

```text
z* in argmax_{z in Z} J(z) = E_{(x,m)~T}[mu(Phi(x; C(z), Theta_frozen), m)]
        subject to sum_t c(z_t, B_t) <= B
```

The model weights stay frozen; the only trainable state is the skill document.

## Architecture

The codebase is organized around protocol boundaries in `beso/core`:

- `beso/core/types.py`: typed contracts for skills, edits, candidates,
  observations, predictions, archive entries, and budgets.
- `beso/core/protocols.py`: shared interfaces for execution harnesses,
  evaluators, dataset providers, reflection proposers, featurizers, surrogates,
  acquisition functions, batch selectors, acceptance gates, archives, and regime
  detectors.
- `beso/adapters/skillopt.py`: SkillOpt-style adapters for skill parsing, edit
  application, dataset loading, evaluation, and LLM-driven reflection.

Module map:

- `beso/surrogate/`: predictive utility and uncertainty models.
- `beso/acquisition/`: pool-normalized scoring, diversity, and batch selection.
- `beso/archive/`: retained candidates, Pareto/diversity tiers, and parent
  selection.
- `beso/optimization/`: optimization loop, statistical gates, regime detection,
  and logging.
- `beso/experiments/`: baseline and experiment helpers.
- `examples/`: runnable toy and GSM8K-mini entry points.

## Committed v0 Design

- Surrogate models parent-relative deltas.
- Predictive variance combines epistemic and aleatoric uncertainty.
- Features are block-separated, standardized, and parent-centered.
- Acquisition is pool-normalized.
- Batch selection is diversity-aware.
- The deployable gate uses paired statistical evidence and multiplicity control.
- The regime detector disables the surrogate when candidate evidence is
  uninformative.

## Scientific Rigor

`ExecutionHarness`, `Evaluator`, and `DatasetProvider` are shared by BESO and
baseline runners where possible, so comparisons can use identical tasks, seeds,
splits, models, scoring rules, and rollout budgets. Any measured lift should be
attributable to the optimization layer, not harness differences.

## Status

Prototype layer complete: core contracts, surrogate/acquisition components,
archive management, acceptance gates, regime detection, SkillOpt-style adapters,
toy and GSM8K-mini examples, baseline helpers, and tests are in place.

The current focus is reproducibility hardening for `v0.1.0`: run provenance,
documented protocols, clean result summaries, and clearer decision records.

## Documentation

- [Documentation index](docs/README.md)
- [Roadmap](docs/roadmap.md)
- [Development workflow](docs/development.md)
- [Architecture overview](docs/architecture/overview.md)
- [Design documents](docs/design/README.md)
- [Experiment protocol and results](docs/experiments/README.md)
- [Decision records](docs/decisions/README.md)

## License

MIT.

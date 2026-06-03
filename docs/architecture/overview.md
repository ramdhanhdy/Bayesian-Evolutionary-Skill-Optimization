# Architecture Overview

BESO optimizes a structured natural-language skill artifact for a frozen LLM
agent. The trainable state is text, not model weights.

## Package Map

- `beso/core/`: shared types and protocol boundaries.
- `beso/adapters/`: bindings between BESO protocols and external systems such as
  SkillOpt-style harnesses, datasets, evaluators, and LLM calls.
- `beso/features/`: candidate featurization and normalization.
- `beso/surrogate/`: predictive models for candidate utility and uncertainty.
- `beso/acquisition/`: acquisition scoring, diversity terms, and batch selection.
- `beso/archive/`: candidate retention, Pareto/diversity tiers, and parent
  selection state.
- `beso/optimization/`: optimization loop, acceptance gates, regime detection,
  and logging.
- `beso/experiments/`: baseline helpers and experiment utilities.
- `examples/`: runnable toy and GSM8K-mini entry points.
- `configs/`: default experiment configuration.
- `tests/`: focused unit and integration tests.

## Core Separation

BESO separates:

- candidate generation from candidate selection;
- deployable promotion from search-state retention;
- model evaluation from optimizer-side evidence tracking;
- exploratory notes from accepted decisions.

This separation is important for research validity: improvements should be
attributable to the optimization layer, not accidental changes in harness,
splits, evaluation rules, or artifact visibility.


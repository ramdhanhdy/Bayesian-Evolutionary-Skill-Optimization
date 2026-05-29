# BESO: Bayesian Evolutionary Skill Optimization

BESO optimizes a structured natural-language **skill artifact** for a *frozen* LLM
agent by combining trajectory-grounded reflective mutation, a **Bayesian
surrogate** over candidate utility, a **pool-normalized acquisition** layer that
allocates a limited rollout budget, and an **evolutionary archive**.

It is a research fork of [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)
(MIT): BESO reuses SkillOpt's execution harness, deterministic SKILL.md edit
applicator, evaluators, and dataloaders ("the plumbing"), and replaces SkillOpt's
greedy 1-edit propose -> evaluate -> gate loop with a Bayesian experiment-planning
selection layer ("the brains").

## Objective

```
z* in argmax_{z in Z} J(z) = E_{(x,m)~T}[ mu(Phi(x; C(z), Theta_frozen), m) ]
        subject to  sum_t c(z_t, B_t) <= B
```

The model weights stay frozen; the only trainable state is the skill document.

## Architecture

The codebase is organized around a clean protocol boundary in `beso/core`:

- `beso/core/types.py` — typed contracts (skill, edit, candidate, observation,
  prediction, archive entry, budget).
- `beso/core/protocols.py` — the seam. **Reused-from-SkillOpt** interfaces
  (`ExecutionHarness`, `EditApplicator`, `Evaluator`, `DatasetProvider`,
  `SkillSerializer`) and **BESO-owned brains** (`ReflectionProposer`,
  `Featurizer`, `Surrogate`, `AcquisitionFunction`, `BatchSelector`,
  `AcceptanceGate`, `Archive`, `RegimeDetector`).
- `beso/adapters/skillopt.py` — thin adapters binding SkillOpt to the protocols
  (currently documented stubs; concrete binding deferred).

Module map: `surrogate/`, `acquisition/`, `archive/`, `features/`,
`reflection/`, `edits/`, `evaluation/`, `compiler/`, `optimization/`,
`trajectories/`, `store/`, `llm/`, `experiments/`.

## Committed v0 design

- Surrogate models the **parent-relative delta** with a bootstrap-bagged ensemble.
- Predictive variance = **epistemic + aleatoric**, recalibrated.
- Features are **block-separated, standardized, parent-centered**.
- Acquisition `a_BESO` is **pool-normalized** (dimensionless weights).
- Batch selection is **submodular** (max-min / DPP).
- Gate uses a **paired test + Benjamini-Hochberg + noise-scaled delta**.
- A **regime detector** disables the surrogate when uninformative.

## Scientific rigor

`ExecutionHarness`, `Evaluator`, and `DatasetProvider` are shared by both BESO
and an unmodified-SkillOpt baseline runner, so the baseline runs on identical
tasks, seeds, splits, and rollout budgets. Any measured lift is attributable to
Bayesian planning, not harness differences.

## Status

Foundational layer complete: repository layout, `core/types.py`, and
`core/protocols.py`. Concrete brains (surrogate, acquisition, archive) and
SkillOpt adapter bindings are next.

## Docs

- `docs/Bayesian Evolutionary Skill Optimization (BESO) - Methodology.md`
- `docs/Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification.md`
- `docs/Bayesian Evolutionary Skill Optimization (BESO) - Mathematical Breakdown.md`
- `docs/Bayesian Evolutionary Skill Optimization (BESO) - GEPA SkillOpt BESO Mathematical Lineage.md`

## License

MIT.

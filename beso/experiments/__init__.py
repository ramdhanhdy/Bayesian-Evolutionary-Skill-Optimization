"""Baselines, ablations, and reporting.

Hosts the unmodified-SkillOpt baseline runner (sharing the same harness/evaluator/
dataset protocols as BESO for an apples-to-apples comparison), ablation configs,
and the V(b) / AUC_B optimization-curve reporting (Breakdown S11; Spec S32).

Planned modules: ``ablations.py``, ``reporting.py``.
"""

from beso.experiments.baselines import (
    ConditionEvaluation,
    EvaluationCondition,
    evaluate_conditions,
    shared_example_ids,
)

__all__ = [
    "ConditionEvaluation",
    "EvaluationCondition",
    "evaluate_conditions",
    "shared_example_ids",
]

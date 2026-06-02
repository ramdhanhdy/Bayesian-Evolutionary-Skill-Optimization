"""Shared evaluation conditions for apples-to-apples benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from beso.core.protocols import DatasetProvider, Evaluator
from beso.core.types import EvaluationResult, SkillArtifact, SplitRole


@dataclass(frozen=True)
class EvaluationCondition:
    """One frozen benchmark condition evaluated on a shared example draw."""

    name: str
    artifact: SkillArtifact
    evaluator: Evaluator


@dataclass(frozen=True)
class ConditionEvaluation:
    """Auditable result for one benchmark condition."""

    name: str
    artifact: SkillArtifact
    evaluation: EvaluationResult


def shared_example_ids(
    dataset: DatasetProvider,
    *,
    role: SplitRole,
    batch_size: int,
    seed: int,
) -> tuple[str, ...]:
    """Draw one deterministic batch reused by every comparison condition."""

    size = min(max(int(batch_size), 0), dataset.split_size(role))
    if size == 0:
        return ()
    return tuple(dataset.batch(role, size, seed))


def evaluate_conditions(
    conditions: Sequence[EvaluationCondition],
    *,
    role: SplitRole,
    example_ids: Sequence[str],
    seed: int,
) -> list[ConditionEvaluation]:
    """Evaluate each condition without changing the shared validation draw."""

    ids = tuple(example_ids)
    return [
        ConditionEvaluation(
            name=condition.name,
            artifact=condition.artifact,
            evaluation=condition.evaluator.evaluate(
                condition.artifact,
                role,
                ids,
                seed,
            ),
        )
        for condition in conditions
    ]


__all__ = [
    "ConditionEvaluation",
    "EvaluationCondition",
    "evaluate_conditions",
    "shared_example_ids",
]

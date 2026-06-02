from __future__ import annotations

from collections.abc import Sequence

from beso.adapters import SkillOptDatasetProvider
from beso.core.types import EvaluationResult, SkillArtifact, SplitRole, Trajectory
from beso.experiments import (
    EvaluationCondition,
    evaluate_conditions,
    shared_example_ids,
)


class RecordingEvaluator:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls: list[tuple[str, SplitRole, tuple[str, ...], int]] = []

    def score_trajectory(self, trajectory: Trajectory) -> float:
        return trajectory.score

    def evaluate(
        self,
        skill: SkillArtifact,
        role: SplitRole,
        example_ids: Sequence[str],
        seed: int,
    ) -> EvaluationResult:
        ids = tuple(example_ids)
        self.calls.append((skill.skill_id, role, ids, seed))
        return EvaluationResult(
            candidate_id=skill.skill_id,
            split=role,
            per_example_scores={example_id: self.score for example_id in ids},
        )


def test_conditions_share_one_deterministic_validation_draw() -> None:
    dataset = SkillOptDatasetProvider(
        items_by_role={
            SplitRole.VALIDATION_GATE: [
                {"id": "v1"},
                {"id": "v2"},
                {"id": "v3"},
            ]
        }
    )
    no_skill_evaluator = RecordingEvaluator(0.5)
    minimal_seed_evaluator = RecordingEvaluator(0.75)
    example_ids = shared_example_ids(
        dataset,
        role=SplitRole.VALIDATION_GATE,
        batch_size=2,
        seed=7,
    )

    results = evaluate_conditions(
        [
            EvaluationCondition(
                name="literal_no_skill",
                artifact=SkillArtifact(skill_id="no_skill", name=""),
                evaluator=no_skill_evaluator,
            ),
            EvaluationCondition(
                name="minimal_seed",
                artifact=SkillArtifact(skill_id="minimal_seed", name="GSM8K"),
                evaluator=minimal_seed_evaluator,
            ),
        ],
        role=SplitRole.VALIDATION_GATE,
        example_ids=example_ids,
        seed=7,
    )

    assert [result.name for result in results] == [
        "literal_no_skill",
        "minimal_seed",
    ]
    assert [result.evaluation.mean_score for result in results] == [0.5, 0.75]
    assert no_skill_evaluator.calls[0][2] == example_ids
    assert minimal_seed_evaluator.calls[0][2] == example_ids
    assert no_skill_evaluator.calls[0][3] == minimal_seed_evaluator.calls[0][3] == 7

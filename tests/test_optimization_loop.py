from __future__ import annotations

from typing import Sequence

from beso.archive import ArchiveConfig, EvolutionaryArchive
from beso.core.protocols import GateDecision, PoolStatistics
from beso.core.types import (
    Candidate,
    CandidateFeatures,
    EditOperation,
    EditProposal,
    EvaluationResult,
    RolloutBudget,
    SkillArtifact,
    SkillMetadata,
    SplitRole,
    SurrogatePrediction,
    Trajectory,
)
from beso.optimization import BESOOptimizer, BESOOptimizerConfig


class FakeDataset:
    def batch(self, role: SplitRole, size: int, seed: int) -> Sequence[str]:
        return [f"{role.value}_{seed}_{i}" for i in range(size)]

    def split_size(self, role: SplitRole) -> int:
        return 100


class FakeEvaluator:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, SplitRole, tuple[str, ...]]] = []

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
        self.calls.append((skill.skill_id, role, ids))
        score = self.scores.get(skill.skill_id, 0.0)
        trajectories = [
            Trajectory(example_id=example_id, task_input="", score=score)
            for example_id in ids
        ]
        return EvaluationResult(
            candidate_id=skill.skill_id,
            split=role,
            per_example_scores={example_id: score for example_id in ids},
            trajectories=trajectories,
        )


class FakeProposer:
    def propose_pool(
        self,
        parent: SkillArtifact,
        trajectories: Sequence[Trajectory],
        rejected: Sequence[EditProposal],
        pool_size: int,
    ) -> list[EditProposal]:
        edits = [
            EditProposal(
                edit_id="z_good",
                parent_skill_id=parent.skill_id,
                operation=EditOperation.APPEND,
                content="good",
            ),
            EditProposal(
                edit_id="z_bad",
                parent_skill_id=parent.skill_id,
                operation=EditOperation.APPEND,
                content="bad",
            ),
        ]
        return edits[:pool_size]


class FakeApplicator:
    def apply(self, parent: SkillArtifact, edit: EditProposal) -> SkillArtifact:
        pred = 0.8 if edit.edit_id == "z_good" else 0.2
        return SkillArtifact(
            skill_id=edit.edit_id,
            name=edit.edit_id,
            document=f"{parent.document}\n{edit.content}",
            metadata=SkillMetadata(parent_id=parent.skill_id, extra={"pred": pred}),
        )

    def apply_sequence(
        self,
        parent: SkillArtifact,
        edits: Sequence[EditProposal],
    ) -> SkillArtifact:
        skill = parent
        for edit in edits:
            skill = self.apply(skill, edit)
        return skill


class FakeFeaturizer:
    def featurize(
        self,
        candidate: Candidate,
        parent: SkillArtifact | None,
        history,
    ) -> CandidateFeatures:
        pred = float(candidate.skill.metadata.extra.get("pred", 0.5))
        return CandidateFeatures(
            candidate_id=candidate.candidate_id,
            structural={"child_tokens": 10.0, "pred": pred},
            history={"parent_mean": 0.5},
        )


class FakeSurrogate:
    def __init__(self) -> None:
        self.fit_count = 0
        self.predict_count = 0

    def fit(self, features, observations) -> None:
        self.fit_count += 1

    def predict(self, features: CandidateFeatures) -> SurrogatePrediction:
        self.predict_count += 1
        return SurrogatePrediction(
            candidate_id=features.candidate_id,
            mu=float(features.structural["pred"]),
            sigma=0.1,
        )

    @property
    def is_calibrated(self) -> bool:
        return True


class FakeAcquisition:
    def __init__(self) -> None:
        self.calls = 0

    def score(
        self,
        candidate: Candidate,
        prediction: SurrogatePrediction,
        archive,
        pool_stats: PoolStatistics,
    ) -> float:
        self.calls += 1
        return prediction.mu


class SortBatchSelector:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, candidates: Sequence[Candidate], k: int) -> list[Candidate]:
        self.calls += 1
        return sorted(
            candidates,
            key=lambda c: float(c.acquisition_score or 0.0),
            reverse=True,
        )[:k]


class FakeGate:
    def decide(
        self,
        candidate_eval: EvaluationResult,
        parent_eval: EvaluationResult,
    ) -> GateDecision:
        accepted = candidate_eval.mean_score > parent_eval.mean_score
        return GateDecision(
            candidate_id=candidate_eval.candidate_id,
            accepted=accepted,
            reason="accepted" if accepted else "reject",
            mean_diff=candidate_eval.mean_score - parent_eval.mean_score,
            p_value=0.001 if accepted else 1.0,
        )


class ToggleRegime:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.calls = 0

    def use_surrogate(self, surrogate, recent_scores: Sequence[float]) -> bool:
        self.calls += 1
        return self.enabled


def _optimizer(regime_enabled: bool = True):
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=8,
            top_by_validation=4,
            top_by_pareto=2,
            top_by_diversity=2,
            top_failed_informative=0,
        )
    )
    surrogate = FakeSurrogate()
    acquisition = FakeAcquisition()
    selector = SortBatchSelector()
    regime = ToggleRegime(regime_enabled)
    opt = BESOOptimizer(
        dataset=FakeDataset(),
        evaluator=FakeEvaluator({"z0": 0.5, "z_good": 0.8, "z_bad": 0.2}),
        proposer=FakeProposer(),
        applicator=FakeApplicator(),
        featurizer=FakeFeaturizer(),
        surrogate=surrogate,
        acquisition=acquisition,
        batch_selector=selector,
        gate=FakeGate(),
        archive=archive,
        regime_detector=regime,
        config=BESOOptimizerConfig(
            max_iterations=1,
            candidate_pool_size=2,
            batch_size=1,
            optimization_batch_size=1,
            validation_batch_size=1,
            seed=11,
            fallback_strategy="greedy",
        ),
        multiplicity_correction=lambda decisions: list(decisions),
    )
    return opt, surrogate, acquisition, selector, regime


def test_optimizer_wires_surrogate_acquisition_gate_and_archive() -> None:
    opt, surrogate, acquisition, selector, regime = _optimizer(regime_enabled=True)
    initial = SkillArtifact(skill_id="z0", name="seed", document="seed")

    result = opt.optimize(initial, RolloutBudget(max_rollouts=5))

    assert result.best is not None
    assert result.best.candidate_id == "z_good"
    assert result.iterations[0].used_surrogate
    assert result.iterations[0].selected_ids == ["z_good"]
    assert result.iterations[0].accepted_ids == ["z_good"]
    assert result.budget.spent_rollouts == 3
    assert surrogate.fit_count == 1
    assert surrogate.predict_count == 2
    assert acquisition.calls == 2
    assert selector.calls == 1
    assert regime.calls == 2


def test_optimizer_regime_fallback_bypasses_bayesian_math() -> None:
    opt, surrogate, acquisition, selector, regime = _optimizer(regime_enabled=False)
    initial = SkillArtifact(skill_id="z0", name="seed", document="seed")

    result = opt.optimize(initial, RolloutBudget(max_rollouts=5))

    assert not result.iterations[0].used_surrogate
    assert result.iterations[0].fallback_reason == "regime_precheck"
    assert result.iterations[0].selected_ids == ["z_good"]
    assert surrogate.fit_count == 0
    assert surrogate.predict_count == 0
    assert acquisition.calls == 0
    assert selector.calls == 0
    assert regime.calls == 1


def test_optimizer_stops_when_rollout_budget_is_exhausted() -> None:
    opt, _, _, _, _ = _optimizer(regime_enabled=True)
    initial = SkillArtifact(skill_id="z0", name="seed", document="seed")

    result = opt.optimize(initial, RolloutBudget(max_rollouts=1))

    assert result.budget.exhausted
    assert result.budget.spent_rollouts == 1
    assert result.iterations == []

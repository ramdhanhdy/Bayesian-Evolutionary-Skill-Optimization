"""Smoke tests for the locked core contracts (types + protocols).

These verify the foundational layer is importable and internally consistent so
downstream BESO modules can be built against stable interfaces.
"""

from __future__ import annotations

from beso.core import (
    Candidate,
    EditCategory,
    EditOperation,
    EditProposal,
    EvaluationResult,
    Observation,
    RolloutBudget,
    SkillArtifact,
    SplitRole,
    SurrogatePrediction,
)
from beso.core import protocols as P


def test_skill_document_is_source_of_truth() -> None:
    skill = SkillArtifact(skill_id="z0", name="seed", document="# Goal\nSolve tasks.\n")
    assert skill.document.startswith("# Goal")
    assert skill.sections == {}


def test_edit_proposal_maps_to_skillopt_fields() -> None:
    edit = EditProposal(
        edit_id="e1",
        parent_skill_id="z0",
        operation=EditOperation.APPEND,
        content="Always verify the final answer.",
        target="",
        category=EditCategory.ADD_RULE,
    )
    assert edit.operation.value == "append"
    assert edit.category is EditCategory.ADD_RULE


def test_evaluation_result_mean_and_n() -> None:
    ev = EvaluationResult(
        candidate_id="z1",
        split=SplitRole.OPTIMIZATION_MINIBATCH,
        per_example_scores={"a": 1.0, "b": 0.0, "c": 1.0},
    )
    assert ev.n == 3
    assert abs(ev.mean_score - (2 / 3)) < 1e-9


def test_rollout_budget_accounting() -> None:
    b = RolloutBudget(max_rollouts=10)
    b.spend(4)
    assert b.remaining == 6
    assert not b.exhausted
    b.spend(6)
    assert b.exhausted


def test_protocol_surface_exists() -> None:
    for name in (
        "ExecutionHarness",
        "EditApplicator",
        "Evaluator",
        "DatasetProvider",
        "SkillSerializer",
        "ReflectionProposer",
        "Featurizer",
        "Surrogate",
        "AcquisitionFunction",
        "BatchSelector",
        "AcceptanceGate",
        "Archive",
        "RegimeDetector",
    ):
        assert hasattr(P, name), f"missing protocol {name}"


def test_prediction_and_candidate_wiring() -> None:
    pred = SurrogatePrediction(candidate_id="z1", mu=0.7, sigma=0.1, epistemic_var=0.008, aleatoric_var=0.002)
    cand = Candidate(candidate_id="z1", skill=SkillArtifact(skill_id="z1", name="c1"), prediction=pred)
    assert cand.prediction is not None
    assert cand.prediction.mu == 0.7
    _ = Observation(candidate_id="z1", batch_ids=("a", "b"), observed_score=0.66)

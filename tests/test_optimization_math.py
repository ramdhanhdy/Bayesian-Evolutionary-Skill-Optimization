from __future__ import annotations

import pytest

from beso.core.protocols import GateDecision
from beso.core.types import EvaluationResult, SplitRole, Trajectory
from beso.optimization import (
    PARETO_CLEANUP_REASON,
    AcceptanceGateConfig,
    PairedBootstrapAcceptanceGate,
    RegimeDetectorConfig,
    VarianceRankRegimeDetector,
    apply_benjamini_hochberg,
    paired_differences,
    spearman_rank_correlation,
)


class _DummySurrogate:
    def __init__(self, calibrated: bool = True) -> None:
        self._calibrated = calibrated

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


def _eval(candidate_id: str, scores: list[float]) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        split=SplitRole.VALIDATION_GATE,
        per_example_scores={f"ex_{i}": score for i, score in enumerate(scores)},
    )


def test_paired_gate_requires_identical_validation_draw() -> None:
    candidate = EvaluationResult(
        candidate_id="cand",
        split=SplitRole.VALIDATION_GATE,
        per_example_scores={"a": 1.0, "b": 0.0},
    )
    parent = EvaluationResult(
        candidate_id="parent",
        split=SplitRole.VALIDATION_GATE,
        per_example_scores={"a": 0.0, "c": 0.0},
    )

    with pytest.raises(ValueError, match="identical example ids"):
        paired_differences(candidate, parent)


def test_gate_accepts_consistent_paired_improvement() -> None:
    parent = _eval("parent", [0.5] * 16)
    candidate = _eval("cand", [0.6] * 16)
    gate = PairedBootstrapAcceptanceGate(
        AcceptanceGateConfig(bootstrap_samples=512, noise_scaled_delta_c=1.0)
    )

    decision = gate.decide(candidate, parent)

    assert decision.accepted
    assert decision.mean_diff == pytest.approx(0.1)
    assert decision.noise_scaled_threshold == pytest.approx(0.0)
    assert decision.ci_low > 0.0


def test_gate_rejects_lucky_noise_by_noise_scaled_delta() -> None:
    parent = _eval("parent", [0.5] * 20)
    diffs = [0.5] * 5 + [-0.4] * 5 + [0.0] * 10
    candidate = _eval("cand", [0.5 + d for d in diffs])
    gate = PairedBootstrapAcceptanceGate(
        AcceptanceGateConfig(
            bootstrap_samples=1024,
            bootstrap_seed=3,
            noise_scaled_delta_c=1.0,
        )
    )

    decision = gate.decide(candidate, parent)

    assert decision.mean_diff > 0.0
    assert decision.mean_diff < decision.noise_scaled_threshold
    assert not decision.accepted


def test_gate_uses_exact_mcnemar_for_small_binary_pairs() -> None:
    parent = _eval("parent", [0, 0, 0, 0, 0, 0, 1, 1])
    candidate = _eval("cand", [1, 1, 1, 1, 1, 1, 1, 1])
    gate = PairedBootstrapAcceptanceGate(
        AcceptanceGateConfig(bootstrap_samples=512, exact_mcnemar_max_n=20)
    )

    decision = gate.decide(candidate, parent)

    assert decision.accepted
    assert "exact_mcnemar" in decision.reason
    assert decision.p_value == pytest.approx(1 / 64)


def _eval_with(candidate_id: str, scores: list[float], **kwargs) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        split=SplitRole.VALIDATION_GATE,
        per_example_scores={f"ex_{i}": s for i, s in enumerate(scores)},
        **kwargs,
    )


def test_gate_accepts_pareto_cleanup_when_tokens_drop_without_accuracy_loss() -> None:
    parent = _eval_with("parent", [1.0] * 8, mean_cost_tokens=100.0)
    candidate = _eval_with("cand", [1.0] * 8, mean_cost_tokens=60.0)
    gate = PairedBootstrapAcceptanceGate(AcceptanceGateConfig(bootstrap_samples=256))

    decision = gate.decide(candidate, parent)

    assert decision.accepted
    assert decision.reason.startswith(PARETO_CLEANUP_REASON)
    assert "child_tokens" in decision.reason
    assert decision.mean_diff == pytest.approx(0.0)


def test_gate_does_not_cleanup_when_secondary_metrics_are_equal() -> None:
    parent = _eval_with("parent", [1.0] * 8, mean_cost_tokens=100.0)
    candidate = _eval_with("cand", [1.0] * 8, mean_cost_tokens=100.0)
    gate = PairedBootstrapAcceptanceGate(AcceptanceGateConfig(bootstrap_samples=256))

    decision = gate.decide(candidate, parent)

    assert not decision.accepted
    assert not decision.reason.startswith(PARETO_CLEANUP_REASON)


def test_gate_does_not_cleanup_when_primary_metric_degrades() -> None:
    parent = _eval_with("parent", [1.0] * 8, mean_cost_tokens=100.0)
    candidate = _eval_with("cand", [1.0] * 7 + [0.0], mean_cost_tokens=10.0)
    gate = PairedBootstrapAcceptanceGate(AcceptanceGateConfig(bootstrap_samples=256))

    decision = gate.decide(candidate, parent)

    assert not decision.accepted


def test_gate_does_not_cleanup_when_paired_ci_cannot_establish_noninferiority() -> None:
    parent = _eval_with(
        "parent",
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        mean_cost_tokens=100.0,
    )
    candidate = _eval_with(
        "cand",
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        mean_cost_tokens=10.0,
    )
    gate = PairedBootstrapAcceptanceGate(
        AcceptanceGateConfig(bootstrap_samples=1024, bootstrap_seed=2)
    )

    decision = gate.decide(candidate, parent)

    assert decision.ci_low < 0.0
    assert not decision.accepted
    assert not decision.reason.startswith(PARETO_CLEANUP_REASON)


def test_gate_does_not_cleanup_on_noisy_latency_point_estimate() -> None:
    ids = [f"ex_{i}" for i in range(8)]
    parent = _eval_with(
        "parent",
        [1.0] * 8,
        trajectories=[
            Trajectory(example_id=example_id, task_input="", latency_seconds=1.0)
            for example_id in ids
        ],
        mean_latency_seconds=1.0,
    )
    candidate_latencies = [0.1, 0.1, 0.1, 0.1, 1.7, 1.7, 1.7, 1.7]
    candidate = _eval_with(
        "cand",
        [1.0] * 8,
        trajectories=[
            Trajectory(
                example_id=example_id,
                task_input="",
                latency_seconds=latency,
            )
            for example_id, latency in zip(ids, candidate_latencies)
        ],
        mean_latency_seconds=sum(candidate_latencies) / len(candidate_latencies),
    )
    gate = PairedBootstrapAcceptanceGate(
        AcceptanceGateConfig(bootstrap_samples=1024, bootstrap_seed=2)
    )

    decision = gate.decide(candidate, parent)

    assert not decision.accepted


def test_benjamini_hochberg_preserves_pareto_cleanup_decisions() -> None:
    decisions = [
        GateDecision(
            candidate_id="z1",
            accepted=True,
            p_value=1.0,
            reason=f"{PARETO_CLEANUP_REASON}:child_tokens",
        ),
        GateDecision(candidate_id="z2", accepted=False, p_value=0.30, reason="reject"),
    ]

    corrected = apply_benjamini_hochberg(decisions, alpha=0.05)

    assert corrected[0].accepted
    assert corrected[0].reason == f"{PARETO_CLEANUP_REASON}:child_tokens"
    assert not corrected[1].accepted


def test_benjamini_hochberg_rejects_borderline_batch_winner() -> None:
    decisions = [
        GateDecision(candidate_id="z1", accepted=True, p_value=0.01, reason="accepted"),
        GateDecision(candidate_id="z2", accepted=True, p_value=0.04, reason="accepted"),
        GateDecision(candidate_id="z3", accepted=False, p_value=0.20, reason="reject"),
    ]

    corrected = apply_benjamini_hochberg(decisions, alpha=0.05)

    assert [d.accepted for d in corrected] == [True, False, False]
    assert "bh_reject" in corrected[1].reason


def test_regime_detector_falls_back_on_low_variance() -> None:
    detector = VarianceRankRegimeDetector(
        RegimeDetectorConfig(min_candidate_variance=1e-4, min_scores=3)
    )

    assert not detector.use_surrogate(_DummySurrogate(True), [0.5, 0.5001, 0.4999])


def test_regime_detector_falls_back_on_poor_rank_correlation() -> None:
    detector = VarianceRankRegimeDetector(
        RegimeDetectorConfig(
            min_candidate_variance=1e-4,
            min_rank_correlation=0.1,
            min_scores=3,
        )
    )
    corr = detector.update_rank_correlation([1, 2, 3, 4], [4, 3, 2, 1])

    assert corr < 0.0
    assert not detector.use_surrogate(_DummySurrogate(True), [0.1, 0.4, 0.9, 1.2])


def test_regime_detector_allows_calibrated_predictive_regime() -> None:
    detector = VarianceRankRegimeDetector(
        RegimeDetectorConfig(
            min_candidate_variance=1e-4,
            min_rank_correlation=0.1,
            min_scores=3,
        )
    )
    detector.update_rank_correlation([1, 2, 3, 4], [1, 2, 3, 4])

    assert detector.use_surrogate(_DummySurrogate(True), [0.1, 0.4, 0.9, 1.2])
    assert not detector.use_surrogate(_DummySurrogate(False), [0.1, 0.4, 0.9, 1.2])


def test_spearman_rank_correlation_handles_ties() -> None:
    corr = spearman_rank_correlation([1, 1, 2, 3], [1, 1, 2, 3])

    assert corr == pytest.approx(1.0)

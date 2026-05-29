from __future__ import annotations

import pytest

from beso.core.protocols import GateDecision
from beso.core.types import EvaluationResult, SplitRole
from beso.optimization import (
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

"""Regime detector for deciding when the surrogate should drive selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from beso.core.protocols import Surrogate
from beso.core.types import EvaluationResult


@dataclass(frozen=True)
class RegimeDetectorConfig:
    """Fallback thresholds for surrogate-driven acquisition."""

    min_candidate_variance: float = 1e-4
    min_rank_correlation: float = 0.1
    min_scores: int = 3
    require_calibrated: bool = True
    eps: float = 1e-12


@dataclass(frozen=True)
class PlateauDiagnosticConfig:
    """Thresholds for detecting saturated binary validation draws."""

    max_score: float = 1.0
    alpha: float = 0.10
    saturation_score: float = 0.95
    eps: float = 1e-12


@dataclass(frozen=True)
class PlateauDiagnostic:
    """Read-only diagnosis of validation headroom, not a runtime policy."""

    saturated: bool
    validation_n: int
    current_mean: float
    promotion_headroom: float
    improvable_count: int
    best_possible_exact_mcnemar_p: float | None
    promotion_possible_under_exact_mcnemar: bool | None
    reason: str


class VarianceRankRegimeDetector:
    """Disable Bayesian selection when the pool or surrogate is uninformative."""

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self.rank_correlation: float | None = None

    def use_surrogate(
        self,
        surrogate: Surrogate,
        recent_scores: Sequence[float],
    ) -> bool:
        scores = _finite_array(recent_scores)
        if scores.size < self.config.min_scores:
            return False
        if float(np.var(scores, ddof=0)) < self.config.min_candidate_variance:
            return False
        if self.config.require_calibrated and not surrogate.is_calibrated:
            return False
        if (
            self.rank_correlation is not None
            and self.rank_correlation < self.config.min_rank_correlation
        ):
            return False
        return True

    def update_rank_correlation(
        self,
        predicted_scores: Sequence[float],
        observed_scores: Sequence[float],
    ) -> float:
        """Record Spearman rank correlation for later regime decisions."""

        corr = spearman_rank_correlation(predicted_scores, observed_scores)
        self.rank_correlation = corr
        return corr


def spearman_rank_correlation(
    predicted_scores: Sequence[float],
    observed_scores: Sequence[float],
) -> float:
    """Dependency-free Spearman rho with average ranks for ties."""

    pred = _finite_array(predicted_scores)
    obs = _finite_array(observed_scores)
    if pred.size != obs.size:
        raise ValueError("predicted_scores and observed_scores must have equal length")
    if pred.size < 2:
        return 0.0
    pr = _average_ranks(pred)
    or_ = _average_ranks(obs)
    if float(np.std(pr, ddof=0)) <= 1e-12 or float(np.std(or_, ddof=0)) <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(pr, or_)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def diagnose_binary_validation_plateau(
    validation: EvaluationResult | Sequence[float],
    config: PlateauDiagnosticConfig | None = None,
) -> PlateauDiagnostic:
    """Diagnose whether a binary validation draw lacks promotion headroom.

    This is intentionally diagnostic only. It does not change optimizer control
    flow and should be used by experiment protocols before claiming a benchmark
    plateau.
    """

    cfg = config or PlateauDiagnosticConfig()
    scores = (
        list(validation.per_example_scores.values())
        if isinstance(validation, EvaluationResult)
        else list(validation)
    )
    arr = _finite_array(scores)
    if arr.size == 0:
        raise ValueError("plateau diagnosis requires at least one finite score")

    current_mean = float(np.mean(arr))
    promotion_headroom = max(0.0, float(cfg.max_score - current_mean))
    is_binary = bool(
        np.all(
            np.isclose(arr, 0.0, atol=cfg.eps)
            | np.isclose(arr, cfg.max_score, atol=cfg.eps)
        )
    )
    if not is_binary:
        return PlateauDiagnostic(
            saturated=False,
            validation_n=int(arr.size),
            current_mean=current_mean,
            promotion_headroom=promotion_headroom,
            improvable_count=0,
            best_possible_exact_mcnemar_p=None,
            promotion_possible_under_exact_mcnemar=None,
            reason="non_binary_scores",
        )

    improvable_count = int(np.sum(arr < cfg.max_score - cfg.eps))
    best_possible_p = (
        1.0
        if improvable_count == 0
        else float(2.0 ** (-improvable_count))
    )
    promotion_possible = improvable_count > 0 and best_possible_p <= cfg.alpha
    saturated = current_mean >= cfg.saturation_score and not promotion_possible
    if saturated and improvable_count == 0:
        reason = "perfect_validation_draw"
    elif saturated:
        reason = "saturated_binary_validation_draw"
    elif current_mean < cfg.saturation_score:
        reason = "below_saturation_threshold"
    else:
        reason = "promotion_headroom_available"

    return PlateauDiagnostic(
        saturated=saturated,
        validation_n=int(arr.size),
        current_mean=current_mean,
        promotion_headroom=promotion_headroom,
        improvable_count=improvable_count,
        best_possible_exact_mcnemar_p=best_possible_p,
        promotion_possible_under_exact_mcnemar=promotion_possible,
        reason=reason,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


__all__ = [
    "PlateauDiagnostic",
    "PlateauDiagnosticConfig",
    "RegimeDetectorConfig",
    "VarianceRankRegimeDetector",
    "diagnose_binary_validation_plateau",
    "spearman_rank_correlation",
]

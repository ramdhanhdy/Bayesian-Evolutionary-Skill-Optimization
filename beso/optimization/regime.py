"""Regime detector for deciding when the surrogate should drive selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from beso.core.protocols import Surrogate


@dataclass(frozen=True)
class RegimeDetectorConfig:
    """Fallback thresholds for surrogate-driven acquisition."""

    min_candidate_variance: float = 1e-4
    min_rank_correlation: float = 0.1
    min_scores: int = 3
    require_calibrated: bool = True
    eps: float = 1e-12


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
    "RegimeDetectorConfig",
    "VarianceRankRegimeDetector",
    "spearman_rank_correlation",
]

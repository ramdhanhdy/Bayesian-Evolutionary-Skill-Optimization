"""Shared acquisition utilities.

The concrete acquisition function is pool-normalized: every raw term is
converted to a dimensionless value using statistics from the current candidate
pool before weights are applied. This keeps UCB, diversity, cost, and invalidity
weights comparable across iterations and makes degenerate pools numerically
safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from beso.core.protocols import PoolStatistics
from beso.core.types import Candidate, SurrogatePrediction
from beso.features.featurizer import approx_tokens

TERM_MU = "mu"
TERM_SIGMA = "sigma"
TERM_DIVERSITY = "diversity"
TERM_COST = "cost"
TERM_INVALID = "invalid_risk"
DEFAULT_TERMS: tuple[str, ...] = (
    TERM_MU,
    TERM_SIGMA,
    TERM_DIVERSITY,
    TERM_COST,
    TERM_INVALID,
)


@dataclass(frozen=True)
class AcquisitionConfig:
    """Weights and numerical settings for pool-normalized acquisition."""

    kappa: float = 1.5
    diversity_lambda: float = 0.2
    cost_alpha: float = 0.1
    invalid_gamma: float = 0.1
    normalization: str = "zscore"
    eps: float = 1e-8
    # Optional (lo, hi) clip applied to the expected-score (mu) term *only* when
    # building the acquisition value. The surrogate keeps emitting raw,
    # unbounded predictions; bounding here prevents an out-of-range mu from
    # warping the pool-normalized a_BESO score (Spec: TICKET-003).
    metric_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.normalization not in {"zscore", "minmax"}:
            raise ValueError("normalization must be 'zscore' or 'minmax'")
        if self.metric_bounds is not None:
            lo, hi = self.metric_bounds
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError("metric_bounds must be finite")
            if lo > hi:
                raise ValueError("metric_bounds must satisfy lo <= hi")


def clip_to_bounds(value: float, bounds: tuple[float, float] | None) -> float:
    """Clip ``value`` into ``bounds`` (inclusive); pass through when unset."""

    if bounds is None:
        return float(value)
    lo, hi = bounds
    return float(min(max(float(value), float(lo)), float(hi)))


@dataclass(frozen=True)
class AcquisitionTerms:
    """Raw, unnormalized terms for one candidate."""

    mu: float = 0.0
    sigma: float = 0.0
    diversity: float = 0.0
    cost: float = 0.0
    invalid_risk: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            TERM_MU: float(self.mu),
            TERM_SIGMA: float(self.sigma),
            TERM_DIVERSITY: float(self.diversity),
            TERM_COST: float(self.cost),
            TERM_INVALID: float(self.invalid_risk),
        }


def build_pool_statistics(
    rows: Sequence[Mapping[str, float]],
    *,
    terms: Sequence[str] = DEFAULT_TERMS,
) -> PoolStatistics:
    """Compute per-term statistics for a candidate pool.

    Empty pools return empty dictionaries. Non-finite values are replaced by
    zero before statistics are computed, so one bad candidate cannot poison the
    whole acquisition pass.
    """

    stats = PoolStatistics()
    if not rows:
        return stats
    for term in terms:
        vals = np.asarray([float(row.get(term, 0.0)) for row in rows], dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        stats.means[term] = float(np.mean(vals))
        stats.stds[term] = float(np.std(vals, ddof=0))
        stats.mins[term] = float(np.min(vals))
        stats.maxs[term] = float(np.max(vals))
    return stats


def normalize_term(
    value: float,
    term: str,
    pool_stats: PoolStatistics,
    *,
    mode: str = "zscore",
    eps: float = 1e-8,
) -> float:
    """Normalize one acquisition term using current-pool statistics.

    Degenerate terms contribute ``0.0``. This makes zero-variance pools stable
    and avoids arbitrary ordering from numerical noise.
    """

    value = float(value)
    if not np.isfinite(value):
        value = 0.0
    if mode == "zscore":
        std = float(pool_stats.stds.get(term, 0.0))
        if std <= eps:
            return 0.0
        return (value - float(pool_stats.means.get(term, 0.0))) / std
    if mode == "minmax":
        lo = float(pool_stats.mins.get(term, 0.0))
        hi = float(pool_stats.maxs.get(term, 0.0))
        span = hi - lo
        if span <= eps:
            return 0.0
        return (value - lo) / span
    raise ValueError("mode must be 'zscore' or 'minmax'")


def normalized_terms(
    terms: AcquisitionTerms,
    pool_stats: PoolStatistics,
    config: AcquisitionConfig,
) -> dict[str, float]:
    """Return all acquisition terms after pool normalization."""

    return {
        name: normalize_term(
            value,
            name,
            pool_stats,
            mode=config.normalization,
            eps=config.eps,
        )
        for name, value in terms.as_dict().items()
    }


def compose_acquisition_score(
    terms: AcquisitionTerms,
    pool_stats: PoolStatistics,
    config: AcquisitionConfig,
) -> float:
    """Weighted a_BESO score from raw terms and pool statistics."""

    t = normalized_terms(terms, pool_stats, config)
    score = (
        t[TERM_MU]
        + config.kappa * t[TERM_SIGMA]
        + config.diversity_lambda * t[TERM_DIVERSITY]
        - config.cost_alpha * t[TERM_COST]
        - config.invalid_gamma * t[TERM_INVALID]
    )
    return float(score) if np.isfinite(score) else 0.0


def prediction_terms(prediction: SurrogatePrediction) -> tuple[float, float]:
    """Extract finite acquisition-ready mean and sigma from a prediction."""

    mu = float(prediction.mu)
    sigma = max(float(prediction.sigma), 0.0)
    return (
        mu if np.isfinite(mu) else 0.0,
        sigma if np.isfinite(sigma) else 0.0,
    )


def candidate_cost(candidate: Candidate) -> float:
    """Estimate inference-time skill cost from existing candidate metadata."""

    if candidate.features is not None:
        for block in (candidate.features.structural, candidate.features.history):
            for key in ("child_tokens", "tokens", "parent_tokens"):
                if key in block:
                    val = float(block[key])
                    return val if np.isfinite(val) and val >= 0.0 else 0.0
    token_count = float(candidate.skill.metadata.token_count or 0)
    if token_count > 0:
        return token_count
    return float(approx_tokens(candidate.skill.document))


def candidate_invalid_risk(candidate: Candidate) -> float:
    """Read an optional invalidity-risk signal from features or metadata."""

    keys = ("invalid_risk", "q_invalid", "predicted_invalid_rate", "invalid_rate")
    if candidate.features is not None:
        for block in (
            candidate.features.semantic,
            candidate.features.history,
            candidate.features.structural,
        ):
            for key in keys:
                if key in block:
                    return _finite_nonnegative(block[key])
    extra = candidate.skill.metadata.extra
    for key in keys:
        if key in extra:
            return _finite_nonnegative(extra[key])
    return 0.0


def _finite_nonnegative(value: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return float(max(0.0, val))


__all__ = [
    "AcquisitionConfig",
    "AcquisitionTerms",
    "DEFAULT_TERMS",
    "TERM_COST",
    "TERM_DIVERSITY",
    "TERM_INVALID",
    "TERM_MU",
    "TERM_SIGMA",
    "build_pool_statistics",
    "candidate_cost",
    "candidate_invalid_risk",
    "clip_to_bounds",
    "compose_acquisition_score",
    "normalize_term",
    "normalized_terms",
    "prediction_terms",
]

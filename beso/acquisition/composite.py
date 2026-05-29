"""Pool-normalized BESO acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from beso.acquisition.base import (
    AcquisitionConfig,
    AcquisitionTerms,
    build_pool_statistics,
    candidate_cost,
    candidate_invalid_risk,
    compose_acquisition_score,
    prediction_terms,
)
from beso.acquisition.diversity import FeatureLookup, candidate_archive_diversity
from beso.core.protocols import PoolStatistics
from beso.core.types import ArchiveEntry, Candidate, SurrogatePrediction


class PoolNormalizedBESOAcquisition:
    """Default a_BESO implementation.

    The protocol entry point is :meth:`score`. Optimizers should normally call
    :meth:`score_pool`, which computes the pool statistics once, stores each
    candidate's prediction and acquisition score, and returns the scored pool.
    """

    def __init__(
        self,
        config: Optional[AcquisitionConfig] = None,
        *,
        feature_lookup: FeatureLookup | None = None,
    ) -> None:
        self.config = config or AcquisitionConfig()
        self.feature_lookup = feature_lookup

    def score(
        self,
        candidate: Candidate,
        prediction: SurrogatePrediction,
        archive: Sequence[ArchiveEntry],
        pool_stats: PoolStatistics,
    ) -> float:
        terms = self.raw_terms(candidate, prediction, archive)
        return compose_acquisition_score(terms, pool_stats, self.config)

    def score_pool(
        self,
        candidates: Sequence[Candidate],
        predictions: Sequence[SurrogatePrediction] | Mapping[str, SurrogatePrediction],
        archive: Sequence[ArchiveEntry],
    ) -> list[Candidate]:
        """Score a candidate pool and mutate candidates with predictions/scores."""

        pred_by_id = _prediction_mapping(predictions)
        terms_by_id: dict[str, AcquisitionTerms] = {}
        rows: list[dict[str, float]] = []
        for candidate in candidates:
            pred = pred_by_id[candidate.candidate_id]
            candidate.prediction = pred
            terms = self.raw_terms(candidate, pred, archive)
            terms_by_id[candidate.candidate_id] = terms
            rows.append(terms.as_dict())

        stats = build_pool_statistics(rows)
        scored: list[Candidate] = []
        for candidate in candidates:
            score = compose_acquisition_score(
                terms_by_id[candidate.candidate_id],
                stats,
                self.config,
            )
            candidate.acquisition_score = score
            scored.append(candidate)
        return scored

    def raw_terms(
        self,
        candidate: Candidate,
        prediction: SurrogatePrediction,
        archive: Sequence[ArchiveEntry],
    ) -> AcquisitionTerms:
        mu, sigma = prediction_terms(prediction)
        return AcquisitionTerms(
            mu=mu,
            sigma=sigma,
            diversity=candidate_archive_diversity(
                candidate,
                archive,
                feature_lookup=self.feature_lookup,
            ),
            cost=candidate_cost(candidate),
            invalid_risk=candidate_invalid_risk(candidate),
        )


def _prediction_mapping(
    predictions: Sequence[SurrogatePrediction] | Mapping[str, SurrogatePrediction],
) -> dict[str, SurrogatePrediction]:
    if isinstance(predictions, Mapping):
        return dict(predictions)
    return {p.candidate_id: p for p in predictions}


__all__ = ["PoolNormalizedBESOAcquisition"]

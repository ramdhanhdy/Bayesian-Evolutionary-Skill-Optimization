"""Submodular-style greedy batch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from beso.acquisition.base import normalize_term
from beso.acquisition.diversity import FeatureLookup, candidate_novelty
from beso.core.protocols import PoolStatistics
from beso.core.types import ArchiveEntry, Candidate


@dataclass(frozen=True)
class BatchSelectionConfig:
    """Settings for greedy acquisition-plus-diversity batch selection."""

    diversity_weight: float = 0.5
    eps: float = 1e-8


class GreedySubmodularBatchSelector:
    """Greedy max-min novelty selector with in-batch reference updates."""

    def __init__(
        self,
        config: Optional[BatchSelectionConfig] = None,
        *,
        archive: Sequence[ArchiveEntry] | None = None,
        feature_lookup: FeatureLookup | None = None,
    ) -> None:
        self.config = config or BatchSelectionConfig()
        self.archive = list(archive or [])
        self.feature_lookup = feature_lookup

    def select(self, candidates: Sequence[Candidate], k: int) -> list[Candidate]:
        pool = list(candidates)
        if k <= 0 or not pool:
            return []
        if k >= len(pool):
            return list(pool)

        acq_stats = _acquisition_stats(pool)
        selected: list[Candidate] = []
        remaining = list(pool)
        while remaining and len(selected) < k:
            best_idx = 0
            best_score = -np.inf
            for idx, candidate in enumerate(remaining):
                acq = normalize_term(
                    float(candidate.acquisition_score or 0.0),
                    "acquisition",
                    acq_stats,
                    eps=self.config.eps,
                )
                novelty = candidate_novelty(
                    candidate,
                    selected,
                    archive=self.archive,
                    feature_lookup=self.feature_lookup,
                )
                score = acq + self.config.diversity_weight * novelty
                if score > best_score + self.config.eps:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected


def _acquisition_stats(candidates: Sequence[Candidate]) -> PoolStatistics:
    values = [float(c.acquisition_score or 0.0) for c in candidates]
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return PoolStatistics(
        means={"acquisition": float(np.mean(arr))},
        stds={"acquisition": float(np.std(arr, ddof=0))},
        mins={"acquisition": float(np.min(arr))},
        maxs={"acquisition": float(np.max(arr))},
    )


__all__ = ["BatchSelectionConfig", "GreedySubmodularBatchSelector"]

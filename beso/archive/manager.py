"""In-memory evolutionary archive manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from beso.acquisition.diversity import archive_entry_distance
from beso.archive.pareto import compute_pareto_win_counts, pareto_front
from beso.core.types import (
    ArchiveEntry,
    ArchiveTier,
    Candidate,
    CandidateFeatures,
    EvaluationResult,
    SplitRole,
)


@dataclass(frozen=True)
class ArchiveConfig:
    """Size caps and parent-selection weights for the archive."""

    max_size: int = 32
    top_by_validation: int = 8
    top_by_pareto: int = 8
    top_by_diversity: int = 8
    top_failed_informative: int = 8
    max_invalid_rate: float = 1.0
    parent_validation_beta: float = 1.0
    parent_pareto_beta: float = 0.25
    parent_diversity_beta: float = 0.25
    parent_cost_beta: float = 0.1
    eps: float = 1e-8


class EvolutionaryArchive:
    """Multi-tier archive implementing the ``Archive`` protocol."""

    def __init__(self, config: Optional[ArchiveConfig] = None) -> None:
        self.config = config or ArchiveConfig()
        self._entries: dict[str, ArchiveEntry] = {}
        self._features: dict[str, CandidateFeatures] = {}
        self._evals: dict[str, EvaluationResult] = {}
        # Candidate ids accepted as Pareto "cleanup" edits (non-inferior on the
        # primary metric, better on a secondary one). They are routed to the
        # pareto/diverse tiers and never become deployable ``best`` entries
        # through the cleanup path alone (TICKET-004).
        self._cleanup_ids: set[str] = set()

    def update(
        self,
        candidates: Sequence[Candidate],
        evals: Sequence[EvaluationResult],
        *,
        cleanup_ids: Sequence[str] | None = None,
    ) -> None:
        incoming_ids = {candidate.candidate_id for candidate in candidates}
        cleanup_set = set(cleanup_ids or ())
        self._cleanup_ids -= incoming_ids - cleanup_set
        self._cleanup_ids |= cleanup_set
        eval_by_id = {ev.candidate_id: ev for ev in evals}
        for candidate in candidates:
            ev = eval_by_id.get(candidate.candidate_id)
            if ev is None:
                continue
            if candidate.features is not None:
                self._features[candidate.candidate_id] = candidate.features
            self._evals[candidate.candidate_id] = ev
            self._entries[candidate.candidate_id] = self._build_entry(candidate, ev)
        self._refresh_pareto_counts()
        self._assign_tiers_and_prune()

    def select_parents(self, n: int, seed: int) -> list[ArchiveEntry]:
        if n <= 0:
            return []
        pool = [
            entry
            for entry in self._entries.values()
            if entry.tier is not ArchiveTier.FAILED
        ] or list(self._entries.values())
        if not pool:
            return []
        if n >= len(pool):
            return sorted(pool, key=_entry_rank_key, reverse=True)

        rng = np.random.default_rng(seed)
        weights = self._parent_weights(pool)
        idx = rng.choice(len(pool), size=n, replace=False, p=weights)
        return [pool[int(i)] for i in idx]

    def best(self) -> Optional[ArchiveEntry]:
        pool = [
            entry
            for entry in self._entries.values()
            if entry.tier is not ArchiveTier.FAILED
        ]
        if not pool:
            return None
        eligible = self._best_eligible(pool)
        return max(eligible, key=_entry_rank_key) if eligible else None

    def _best_eligible(self, pool: Sequence[ArchiveEntry]) -> list[ArchiveEntry]:
        """Pool of entries allowed to be the deployable best.

        Cleanup edits are archive-only until they separately pass the normal
        primary gate. A raw validation increase is insufficient because it may
        be a winner's-curse fluctuation (TICKET-004).
        """

        return [
            entry
            for entry in pool
            if entry.candidate_id not in self._cleanup_ids
        ]

    def entries(self) -> list[ArchiveEntry]:
        return sorted(self._entries.values(), key=lambda e: e.candidate_id)

    def feature_lookup(self, candidate_id: str) -> CandidateFeatures | None:
        """Return stored features for acquisition diversity, if available."""

        return self._features.get(candidate_id)

    def feature_map(self) -> dict[str, CandidateFeatures]:
        """Return a copy of the archive's candidate feature lookup."""

        return dict(self._features)

    def _build_entry(self, candidate: Candidate, ev: EvaluationResult) -> ArchiveEntry:
        optimization_mean = ev.mean_score if ev.split is SplitRole.OPTIMIZATION_MINIBATCH else 0.0
        validation_mean = ev.mean_score if ev.split is SplitRole.VALIDATION_GATE else 0.0
        values = list(ev.per_example_scores.values())
        validation_se = _standard_error(values) if ev.split is SplitRole.VALIDATION_GATE else 0.0
        cost_per_task = float(ev.mean_cost_tokens)
        if cost_per_task == 0.0 and candidate.features is not None:
            cost_per_task = float(candidate.features.structural.get("child_tokens", 0.0))
        tier = (
            ArchiveTier.FAILED
            if ev.invalid_rate > self.config.max_invalid_rate
            else ArchiveTier.BEST
        )
        edit_summary = candidate.edit.rationale if candidate.edit else ""
        if candidate.edit and not edit_summary:
            edit_summary = candidate.edit.expected_effect
        return ArchiveEntry(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
            artifact=candidate.skill,
            tier=tier,
            optimization_mean=float(optimization_mean),
            validation_mean=float(validation_mean),
            validation_se=float(validation_se),
            cost_per_task=cost_per_task,
            latency_seconds=float(ev.mean_latency_seconds),
            invalid_rate=float(ev.invalid_rate),
            lineage_depth=int(candidate.skill.metadata.lineage_depth),
            accepted_edit_summary=edit_summary,
            created_at_iteration=int(candidate.skill.metadata.created_at_iteration),
        )

    def _refresh_pareto_counts(self) -> None:
        entries = list(self._entries.values())
        counts = compute_pareto_win_counts(entries, self._evals)
        for entry in entries:
            entry.pareto_win_count = int(counts.get(entry.candidate_id, 0))

    def _assign_tiers_and_prune(self) -> None:
        entries = list(self._entries.values())
        failed = [e for e in entries if e.tier is ArchiveTier.FAILED]
        viable = [e for e in entries if e.tier is not ArchiveTier.FAILED]

        keep: dict[str, ArchiveEntry] = {}
        best_pool = self._best_eligible(viable)
        for entry in sorted(best_pool, key=_entry_rank_key, reverse=True)[
            : self.config.top_by_validation
        ]:
            entry.tier = ArchiveTier.BEST
            keep[entry.candidate_id] = entry

        for entry in sorted(
            pareto_front(viable),
            key=lambda e: (e.pareto_win_count, _entry_quality(e)),
            reverse=True,
        )[: self.config.top_by_pareto]:
            if entry.candidate_id not in keep:
                entry.tier = ArchiveTier.PARETO
                keep[entry.candidate_id] = entry

        for entry in self._diverse_subset(viable, self.config.top_by_diversity):
            if entry.candidate_id not in keep:
                entry.tier = ArchiveTier.DIVERSE
                keep[entry.candidate_id] = entry

        for entry in sorted(
            failed,
            key=lambda e: (e.invalid_rate, -_entry_quality(e), e.candidate_id),
            reverse=True,
        )[: self.config.top_failed_informative]:
            entry.tier = ArchiveTier.FAILED
            keep[entry.candidate_id] = entry

        incumbent = (
            max(best_pool, key=_entry_rank_key)
            if best_pool
            else None
        )
        if incumbent is not None:
            incumbent.tier = ArchiveTier.BEST
            keep[incumbent.candidate_id] = incumbent

        if len(keep) > self.config.max_size:
            retained = []
            if incumbent is not None:
                retained.append(incumbent)
            retained.extend(
                entry
                for entry in sorted(
                    keep.values(),
                    key=_entry_rank_key,
                    reverse=True,
                )
                if incumbent is None or entry.candidate_id != incumbent.candidate_id
            )
            retained = retained[: self.config.max_size]
            keep = {entry.candidate_id: entry for entry in retained}

        self._entries = keep
        self._features = {
            cid: features for cid, features in self._features.items() if cid in keep
        }
        self._evals = {cid: ev for cid, ev in self._evals.items() if cid in keep}
        self._cleanup_ids = {cid for cid in self._cleanup_ids if cid in keep}

    def _diverse_subset(
        self,
        entries: Sequence[ArchiveEntry],
        k: int,
    ) -> list[ArchiveEntry]:
        if k <= 0 or not entries:
            return []
        remaining = sorted(entries, key=_entry_rank_key, reverse=True)
        selected: list[ArchiveEntry] = []
        while remaining and len(selected) < k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_idx = 0
            best_score = -np.inf
            for idx, entry in enumerate(remaining):
                novelty = min(
                    archive_entry_distance(
                        entry,
                        other,
                        feature_lookup=self.feature_lookup,
                    )
                    for other in selected
                )
                score = _entry_quality(entry) + novelty
                if score > best_score + self.config.eps:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def _parent_weights(self, entries: Sequence[ArchiveEntry]) -> np.ndarray:
        quality = np.asarray([_entry_quality(e) for e in entries], dtype=np.float64)
        pareto = np.asarray([float(e.pareto_win_count) for e in entries], dtype=np.float64)
        costs = np.asarray([float(e.cost_per_task) for e in entries], dtype=np.float64)
        diversity = np.asarray([self._entry_novelty(e, entries) for e in entries])
        raw = (
            self.config.parent_validation_beta * _zscore(quality)
            + self.config.parent_pareto_beta * _zscore(pareto)
            + self.config.parent_diversity_beta * _zscore(diversity)
            - self.config.parent_cost_beta * _zscore(costs)
        )
        raw = np.where(np.isfinite(raw), raw, 0.0)
        raw = raw - float(np.max(raw))
        weights = np.exp(raw)
        total = float(np.sum(weights))
        if total <= self.config.eps:
            return np.full(len(entries), 1.0 / len(entries))
        return weights / total

    def _entry_novelty(
        self,
        entry: ArchiveEntry,
        entries: Sequence[ArchiveEntry],
    ) -> float:
        others = [e for e in entries if e.candidate_id != entry.candidate_id]
        if not others:
            return 1.0
        return float(
            min(
                archive_entry_distance(
                    entry,
                    other,
                    feature_lookup=self.feature_lookup,
                )
                for other in others
            )
        )


def _entry_quality(entry: ArchiveEntry) -> float:
    return (
        float(entry.validation_mean)
        if entry.validation_mean != 0.0
        else float(entry.optimization_mean)
    )


def _entry_rank_key(entry: ArchiveEntry) -> tuple[float, int, float, str]:
    return (
        _entry_quality(entry),
        int(entry.pareto_win_count),
        -float(entry.invalid_rate),
        entry.candidate_id,
    )


def _standard_error(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if values.size == 0:
        return values
    std = float(np.std(values, ddof=0))
    if std <= eps:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


__all__ = ["ArchiveConfig", "EvolutionaryArchive"]

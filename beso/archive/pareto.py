"""Pareto and instance-win utilities for archive management."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from beso.core.types import ArchiveEntry, EvaluationResult


def objective_vector(entry: ArchiveEntry) -> np.ndarray:
    """Return a larger-is-better objective vector for Pareto comparisons."""

    quality = (
        float(entry.validation_mean)
        if entry.validation_mean != 0.0
        else float(entry.optimization_mean)
    )
    return np.asarray(
        [
            quality,
            float(entry.format_score),
            -float(entry.cost_per_task),
            -float(entry.latency_seconds),
            -float(entry.invalid_rate),
        ],
        dtype=np.float64,
    )


def pareto_dominates(a: ArchiveEntry, b: ArchiveEntry, *, eps: float = 1e-12) -> bool:
    """Whether entry ``a`` Pareto-dominates entry ``b``."""

    av = objective_vector(a)
    bv = objective_vector(b)
    return bool(np.all(av >= bv - eps) and np.any(av > bv + eps))


def pareto_front(
    entries: Sequence[ArchiveEntry],
    *,
    eps: float = 1e-12,
) -> list[ArchiveEntry]:
    """Return entries not dominated by another entry in the set."""

    front: list[ArchiveEntry] = []
    for entry in entries:
        dominated = any(
            other.candidate_id != entry.candidate_id
            and pareto_dominates(other, entry, eps=eps)
            for other in entries
        )
        if not dominated:
            front.append(entry)
    return front


def compute_pareto_win_counts(
    entries: Sequence[ArchiveEntry],
    evals_by_id: Mapping[str, EvaluationResult],
    *,
    eps: float = 1e-12,
) -> dict[str, int]:
    """Count per-example wins among archived evaluated candidates."""

    scores_by_example: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for entry in entries:
        ev = evals_by_id.get(entry.candidate_id)
        if ev is None:
            continue
        for example_id, score in ev.per_example_scores.items():
            val = float(score)
            if np.isfinite(val):
                scores_by_example[str(example_id)].append((entry.candidate_id, val))

    counts = {entry.candidate_id: 0 for entry in entries}
    for rows in scores_by_example.values():
        if not rows:
            continue
        best = max(score for _, score in rows)
        for candidate_id, score in rows:
            if score >= best - eps:
                counts[candidate_id] = counts.get(candidate_id, 0) + 1
    return counts


__all__ = [
    "compute_pareto_win_counts",
    "objective_vector",
    "pareto_dominates",
    "pareto_front",
]

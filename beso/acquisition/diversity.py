"""Distance and novelty helpers for acquisition and archive pruning."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Optional

import numpy as np

from beso.core.types import ArchiveEntry, Candidate, CandidateFeatures, SkillArtifact

FeatureLookup = Mapping[str, CandidateFeatures] | Callable[[str], Optional[CandidateFeatures]]
_WORD_RE = re.compile(r"\w+")
_EPS = 1e-12


def candidate_archive_diversity(
    candidate: Candidate,
    archive: Sequence[ArchiveEntry],
    *,
    feature_lookup: FeatureLookup | None = None,
) -> float:
    """Minimum distance from a candidate to accepted/archive entries.

    The empty-archive value is ``1.0``: when there is no reference set, every
    candidate is maximally novel relative to the archive.
    """

    refs = list(archive)
    if not refs:
        return 1.0
    return float(
        min(
            candidate_archive_distance(
                candidate,
                entry,
                feature_lookup=feature_lookup,
            )
            for entry in refs
        )
    )


def candidate_novelty(
    candidate: Candidate,
    selected: Sequence[Candidate],
    *,
    archive: Sequence[ArchiveEntry] | None = None,
    feature_lookup: FeatureLookup | None = None,
) -> float:
    """Minimum distance to the current reference set.

    The reference set is ``archive + selected``. This is the dynamic novelty
    term used during greedy batch selection.
    """

    distances: list[float] = []
    for other in selected:
        distances.append(candidate_distance(candidate, other))
    for entry in archive or ():
        distances.append(
            candidate_archive_distance(candidate, entry, feature_lookup=feature_lookup)
        )
    return float(min(distances)) if distances else 1.0


def candidate_distance(a: Candidate, b: Candidate) -> float:
    """Distance between two candidates using features when available."""

    if a.features is not None and b.features is not None:
        return feature_distance(a.features, b.features)
    return artifact_distance(a.skill, b.skill)


def candidate_archive_distance(
    candidate: Candidate,
    entry: ArchiveEntry,
    *,
    feature_lookup: FeatureLookup | None = None,
) -> float:
    """Distance between a candidate and an archive entry."""

    archived_features = _lookup_features(feature_lookup, entry.candidate_id)
    if candidate.features is not None and archived_features is not None:
        return feature_distance(candidate.features, archived_features)
    return artifact_distance(candidate.skill, entry.artifact)


def archive_entry_distance(
    a: ArchiveEntry,
    b: ArchiveEntry,
    *,
    feature_lookup: FeatureLookup | None = None,
) -> float:
    """Distance between two archive entries."""

    af = _lookup_features(feature_lookup, a.candidate_id)
    bf = _lookup_features(feature_lookup, b.candidate_id)
    if af is not None and bf is not None:
        return feature_distance(af, bf)
    return artifact_distance(a.artifact, b.artifact)


def feature_distance(a: CandidateFeatures, b: CandidateFeatures) -> float:
    """Cosine distance over sparse feature blocks plus optional text vectors."""

    av = feature_to_sparse(a)
    bv = feature_to_sparse(b)
    return sparse_cosine_distance(av, bv)


def feature_to_sparse(features: CandidateFeatures) -> dict[str, float]:
    """Flatten structured feature blocks into a sparse numeric mapping."""

    out: dict[str, float] = {}
    for block_name in ("structural", "edit", "history", "semantic"):
        block = getattr(features, block_name)
        for key, value in block.items():
            val = float(value)
            if np.isfinite(val) and val != 0.0:
                out[f"{block_name}::{key}"] = val
    if features.text_embedding is not None:
        emb = np.asarray(features.text_embedding, dtype=np.float64).ravel()
        for i, value in enumerate(emb):
            val = float(value)
            if np.isfinite(val) and val != 0.0:
                out[f"text::{i}"] = val
    return out


def sparse_cosine_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine distance in ``[0, 1]`` for sparse mappings."""

    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    common = set(a).intersection(b)
    dot = sum(float(a[k]) * float(b[k]) for k in common)
    an = np.sqrt(sum(float(v) ** 2 for v in a.values()))
    bn = np.sqrt(sum(float(v) ** 2 for v in b.values()))
    if an <= _EPS or bn <= _EPS:
        return 1.0
    sim = dot / (an * bn)
    sim = min(1.0, max(-1.0, float(sim)))
    return float(1.0 - max(0.0, sim))


def artifact_distance(a: SkillArtifact, b: SkillArtifact) -> float:
    """Jaccard distance between skill documents using word shingles."""

    return jaccard_distance(_word_shingles(a.document), _word_shingles(b.document))


def jaccard_distance(a: set[str], b: set[str]) -> float:
    """Jaccard distance in ``[0, 1]`` with stable empty-set handling."""

    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return float(1.0 - (len(a & b) / len(union)))


def _word_shingles(text: str, n: int = 3) -> set[str]:
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _lookup_features(
    lookup: FeatureLookup | None,
    candidate_id: str,
) -> CandidateFeatures | None:
    if lookup is None:
        return None
    if callable(lookup):
        return lookup(candidate_id)
    return lookup.get(candidate_id)


__all__ = [
    "FeatureLookup",
    "archive_entry_distance",
    "artifact_distance",
    "candidate_archive_distance",
    "candidate_archive_diversity",
    "candidate_distance",
    "candidate_novelty",
    "feature_distance",
    "feature_to_sparse",
    "jaccard_distance",
    "sparse_cosine_distance",
]

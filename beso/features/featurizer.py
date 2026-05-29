"""Candidate featurization phi(z) (Breakdown S6; Spec S12).

Implements the ``Featurizer`` protocol with the **Two-Tier, parent-relative
delta** strategy:

Tier 1 (deterministic signals): structural, edit, and history/lineage blocks are
cheap, low-dimensional, and computed as **differences against the parent skill**
(phi_struct as child - parent deltas). These are the high-signal features the
surrogate should trust early.

Tier 2 (text): a single high-dimensional text-embedding block capturing the
*changed text* (phi_text). It is kept raw here and reduced (PCA) + separately
weighted by :mod:`beso.features.normalization`, so it cannot swamp Tier 1.

This module produces the *raw* per-block features. Standardization, PCA
reduction, block weighting, and assembly into a flat vector are the job of the
:class:`~beso.features.normalization.FeatureNormalizer`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional, Sequence

import numpy as np

from beso.core.types import (
    Candidate,
    CandidateFeatures,
    EditCategory,
    EditOperation,
    Observation,
    SkillArtifact,
    SkillSection,
)

EmbedFn = Callable[[str], np.ndarray]

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```")
_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\S+")

# Structural metric keys, in fixed order, used for parent-relative deltas.
STRUCTURAL_METRICS: tuple[str, ...] = (
    "tokens",
    "chars",
    "lines",
    "headings",
    "bullets",
    "numbered",
    "rules",
    "code_blocks",
    "examples",
    "avg_sentence_tokens",
)


def approx_tokens(text: str) -> int:
    """Whitespace-based token approximation (cheap, deterministic)."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def compute_structural_metrics(document: str) -> dict[str, float]:
    """Compute absolute structural metrics for a skill markdown document."""
    document = document or ""
    tokens = approx_tokens(document)
    headings = len(_HEADING_RE.findall(document))
    bullets = len(_BULLET_RE.findall(document))
    numbered = len(_NUMBERED_RE.findall(document))
    code_blocks = len(_CODE_FENCE_RE.findall(document)) // 2
    sentences = [s for s in _SENTENCE_RE.split(document) if s.strip()]
    avg_sentence_tokens = (
        float(np.mean([approx_tokens(s) for s in sentences])) if sentences else 0.0
    )
    return {
        "tokens": float(tokens),
        "chars": float(len(document)),
        "lines": float(document.count("\n") + 1 if document else 0),
        "headings": float(headings),
        "bullets": float(bullets),
        "numbered": float(numbered),
        "rules": float(bullets + numbered),
        "code_blocks": float(code_blocks),
        "examples": float(len(re.findall(r"(?i)\bexample\b", document))),
        "avg_sentence_tokens": avg_sentence_tokens,
    }


class HashingEmbedder:
    """Deterministic, dependency-free text embedder (char n-gram hashing).

    Provided so the feature pipeline is runnable and testable without an external
    embedding model. Production runs should inject a real sentence/embedding model
    via the ``embed_fn`` argument of :class:`FeatureExtractor`.
    """

    def __init__(self, dim: int = 128, ngram: int = 3) -> None:
        self.dim = int(dim)
        self.ngram = int(ngram)

    def __call__(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        text = (text or "").lower().strip()
        if not text:
            return vec
        padded = f"  {text}  "
        for i in range(len(padded) - self.ngram + 1):
            gram = padded[i : i + self.ngram]
            h = int.from_bytes(
                hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                "little",
                signed=False,
            )
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class FeatureExtractor:
    """Default ``Featurizer`` implementation (Two-Tier, parent-relative delta).

    Parameters
    ----------
    embed_fn:
        Callable mapping text -> 1D embedding. Defaults to :class:`HashingEmbedder`.
        Pass ``None`` explicitly to disable the text block entirely.
    text_mode:
        ``"edit_content"`` embeds the changed text (inherently a delta);
        ``"document_delta"`` embeds ``emb(child) - emb(parent)``.
    semantic_labeler:
        Optional callable mapping a :class:`SkillArtifact` to a dict of semantic
        emphasis scores (phi_sem). Defaults to no semantic features.
    edit_success_rates:
        Optional mapping ``op_value -> success_rate`` used for the cold-startable
        edit-type history feature. Missing ops default to ``cold_start_rate``.
    """

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = HashingEmbedder(),
        *,
        text_mode: str = "edit_content",
        semantic_labeler: Optional[Callable[[SkillArtifact], dict[str, float]]] = None,
        edit_success_rates: Optional[dict[str, float]] = None,
        cold_start_rate: float = 0.5,
    ) -> None:
        if text_mode not in ("edit_content", "document_delta"):
            raise ValueError(f"unknown text_mode: {text_mode!r}")
        self.embed_fn = embed_fn
        self.text_mode = text_mode
        self.semantic_labeler = semantic_labeler
        self.edit_success_rates = dict(edit_success_rates or {})
        self.cold_start_rate = float(cold_start_rate)

    # -- protocol entry point ------------------------------------------------ #
    def featurize(
        self,
        candidate: Candidate,
        parent: Optional[SkillArtifact],
        history: Sequence[Observation],
    ) -> CandidateFeatures:
        return CandidateFeatures(
            candidate_id=candidate.candidate_id,
            text_embedding=self._text_block(candidate, parent),
            structural=self._structural_block(candidate.skill, parent),
            edit=self._edit_block(candidate),
            history=self._history_block(candidate, parent, history),
            semantic=self._semantic_block(candidate.skill),
        )

    # -- Tier 1: structural deltas ------------------------------------------- #
    def _structural_block(
        self, child: SkillArtifact, parent: Optional[SkillArtifact]
    ) -> dict[str, float]:
        child_m = compute_structural_metrics(child.document)
        parent_m = (
            compute_structural_metrics(parent.document)
            if parent is not None
            else {k: 0.0 for k in STRUCTURAL_METRICS}
        )
        block: dict[str, float] = {}
        for k in STRUCTURAL_METRICS:
            c = child_m[k]
            p = parent_m[k]
            block[f"d_{k}"] = c - p  # parent-relative delta
            block[f"rel_{k}"] = (c - p) / (abs(p) + 1.0)  # scale-stable relative delta
        block["child_tokens"] = child_m["tokens"]  # one absolute size anchor
        return block

    # -- Tier 1: edit features ----------------------------------------------- #
    def _edit_block(self, candidate: Candidate) -> dict[str, float]:
        block: dict[str, float] = {}
        for op in EditOperation:
            block[f"op_{op.value}"] = 0.0
        for cat in EditCategory:
            block[f"cat_{cat.value}"] = 0.0
        for sec in SkillSection:
            block[f"sec_{sec.value}"] = 0.0
        block["edit_size_tokens"] = 0.0
        block["content_tokens"] = 0.0
        block["target_tokens"] = 0.0
        block["has_target"] = 0.0
        block["src_failure"] = 0.0
        block["src_success"] = 0.0

        edit = candidate.edit
        if edit is None:
            return block

        block[f"op_{edit.operation.value}"] = 1.0
        if edit.category is not None:
            block[f"cat_{edit.category.value}"] = 1.0
        if edit.target_section is not None:
            block[f"sec_{edit.target_section.value}"] = 1.0
        content_tokens = approx_tokens(edit.content)
        block["content_tokens"] = float(content_tokens)
        block["target_tokens"] = float(approx_tokens(edit.target))
        block["edit_size_tokens"] = float(edit.edit_size_tokens or content_tokens)
        block["has_target"] = 1.0 if edit.target else 0.0
        if edit.source_type == "failure":
            block["src_failure"] = 1.0
        elif edit.source_type == "success":
            block["src_success"] = 1.0
        return block

    # -- Tier 1: history / lineage ------------------------------------------- #
    def _history_block(
        self,
        candidate: Candidate,
        parent: Optional[SkillArtifact],
        history: Sequence[Observation],
    ) -> dict[str, float]:
        parent_id = candidate.parent_id or (parent.skill_id if parent else None)
        parent_scores = [
            o.observed_score for o in history if o.candidate_id == parent_id
        ]
        n_obs = len(parent_scores)
        parent_mean = float(np.mean(parent_scores)) if n_obs else 0.0
        parent_std = float(np.std(parent_scores, ddof=0)) if n_obs > 1 else 0.0

        lineage_depth = float(candidate.skill.metadata.lineage_depth)
        if lineage_depth == 0.0 and parent is not None:
            lineage_depth = float(parent.metadata.lineage_depth + 1)

        op_value = candidate.edit.operation.value if candidate.edit else None
        edit_success = self.edit_success_rates.get(op_value, self.cold_start_rate)

        return {
            "parent_mean": parent_mean,
            "parent_std": parent_std,
            "parent_n_obs": float(n_obs),
            "parent_observed": 1.0 if n_obs else 0.0,
            "lineage_depth": lineage_depth,
            "parent_tokens": float(
                compute_structural_metrics(parent.document)["tokens"]
                if parent is not None
                else 0.0
            ),
            "edit_type_success_rate": float(edit_success),
        }

    # -- Tier 1: semantic ---------------------------------------------------- #
    def _semantic_block(self, child: SkillArtifact) -> dict[str, float]:
        if self.semantic_labeler is None:
            return {}
        labels = self.semantic_labeler(child) or {}
        return {str(k): float(v) for k, v in labels.items()}

    # -- Tier 2: text -------------------------------------------------------- #
    def _text_block(
        self, candidate: Candidate, parent: Optional[SkillArtifact]
    ) -> Optional[np.ndarray]:
        if self.embed_fn is None:
            return None
        if self.text_mode == "document_delta":
            child_emb = self.embed_fn(candidate.skill.document)
            parent_emb = (
                self.embed_fn(parent.document)
                if parent is not None
                else np.zeros_like(child_emb)
            )
            return np.asarray(child_emb, dtype=np.float64) - np.asarray(
                parent_emb, dtype=np.float64
            )
        # edit_content mode (default): embed the changed text itself.
        changed = candidate.edit.content if candidate.edit else candidate.skill.document
        return np.asarray(self.embed_fn(changed), dtype=np.float64)


__all__ = [
    "EmbedFn",
    "STRUCTURAL_METRICS",
    "approx_tokens",
    "compute_structural_metrics",
    "HashingEmbedder",
    "FeatureExtractor",
]

"""Feature normalization and assembly (Breakdown S6.3; Spec S12.4).

Turns the per-block :class:`~beso.core.types.CandidateFeatures` produced by the
featurizer into a single standardized flat vector for the surrogate, enforcing
the two anti-swamping guarantees from the design review:

1. **Per-block standardization** — every deterministic feature is centered and
   scaled to unit variance using statistics fit on the candidate corpus, so no
   raw-scale feature (e.g. token counts) dominates.
2. **Variance-balanced block weighting** — after standardization each block is
   rescaled (when ``balance_block_dims`` is on) so its *total variance* over the
   fit corpus is equalized across blocks. This is robust to blocks that contain
   many constant one-hot features (edit/structural): the high-dimensional PCA
   text block then contributes comparable total variance to the cheap Tier-1
   blocks rather than swamping them. Naive ``1/sqrt(dim)`` scaling is avoided
   because it over-penalizes sparse one-hot blocks whose nominal dimension far
   exceeds their number of active (non-constant) features.

The text block (Tier 2) is additionally reduced by PCA before standardization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from beso.core.types import CandidateFeatures

# Deterministic dict-valued blocks, in assembly order.
_DICT_BLOCKS: tuple[str, ...] = ("structural", "edit", "history", "semantic")
_TEXT_BLOCK = "text"
_EPS = 1e-8


class _PCA:
    """Minimal PCA via NumPy SVD (no external dependency).

    Kept dependency-light so the core features layer is runnable without
    scikit-learn. Mirrors the subset of the sklearn PCA API used here
    (``fit_transform``, ``transform``, ``n_components_``).
    """

    def __init__(self, n_components: int, whiten: bool = False) -> None:
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def fit_transform(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float64)
        n_samples, n_features = mat.shape
        k = max(1, min(self.n_components, n_features, n_samples))
        self.mean_ = mat.mean(axis=0)
        centered = mat - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[:k]
        self.n_components_ = k
        denom = max(n_samples - 1, 1)
        self._scale = (s[:k] / np.sqrt(denom))
        self._scale = np.where(self._scale < _EPS, 1.0, self._scale)
        return self.transform(mat)

    def transform(self, mat: np.ndarray) -> np.ndarray:
        centered = np.asarray(mat, dtype=np.float64) - self.mean_
        reduced = centered @ self.components_.T
        if self.whiten:
            reduced = reduced / self._scale
        return reduced


@dataclass
class NormalizerConfig:
    """Configuration for :class:`FeatureNormalizer`."""

    block_weights: dict[str, float] = field(
        default_factory=lambda: {
            "structural": 1.0,
            "edit": 1.0,
            "history": 1.0,
            "semantic": 1.0,
            "text": 1.0,
        }
    )
    text_pca_dims: int = 32
    standardize: bool = True
    balance_block_dims: bool = True
    whiten_pca: bool = False


class FeatureNormalizer:
    """Fits per-block scalers + a text PCA, then assembles flat vectors.

    Call :meth:`fit` (or :meth:`fit_transform`) on a representative corpus of
    :class:`CandidateFeatures` (e.g. the current pool plus archive) before
    :meth:`transform`. Feature key ordering is frozen at fit time so transformed
    vectors are stable and inspectable via :attr:`feature_names_`.
    """

    def __init__(self, config: Optional[NormalizerConfig] = None) -> None:
        self.config = config or NormalizerConfig()
        self._fitted = False
        # Per dict-block state.
        self._keys: dict[str, list[str]] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._std: dict[str, np.ndarray] = {}
        # Text block state.
        self._has_text = False
        self._pca = None
        self._text_mean: Optional[np.ndarray] = None
        self._text_std: Optional[np.ndarray] = None
        # Per-block multiplicative scale for variance balancing.
        self._block_scale: dict[str, float] = {}
        self.feature_names_: list[str] = []

    # -- fitting ------------------------------------------------------------- #
    def fit(self, corpus: Sequence[CandidateFeatures]) -> "FeatureNormalizer":
        if not corpus:
            raise ValueError("cannot fit FeatureNormalizer on an empty corpus")

        for block in _DICT_BLOCKS:
            keys = self._collect_keys(corpus, block)
            self._keys[block] = keys
            if not keys:
                self._mean[block] = np.zeros(0)
                self._std[block] = np.ones(0)
                continue
            mat = np.stack([self._dict_to_vec(getattr(f, block), keys) for f in corpus])
            self._mean[block] = mat.mean(axis=0)
            std = mat.std(axis=0, ddof=0)
            self._std[block] = np.where(std < _EPS, 1.0, std)
            standardized = (mat - self._mean[block]) / self._std[block]
            self._block_scale[block] = self._variance_scale(standardized)

        self._fit_text(corpus)
        self._build_feature_names()
        self._fitted = True
        return self

    def fit_transform(self, corpus: Sequence[CandidateFeatures]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)

    def _fit_text(self, corpus: Sequence[CandidateFeatures]) -> None:
        embs = [
            np.asarray(f.text_embedding, dtype=np.float64)
            for f in corpus
            if f.text_embedding is not None
        ]
        self._has_text = len(embs) == len(corpus) and len(embs) > 0
        if not self._has_text:
            self._pca = None
            self._text_mean = None
            self._text_std = None
            return

        mat = np.stack(embs)
        self._pca = _PCA(
            n_components=self.config.text_pca_dims, whiten=self.config.whiten_pca
        )
        reduced = self._pca.fit_transform(mat)
        self._text_mean = reduced.mean(axis=0)
        std = reduced.std(axis=0, ddof=0)
        self._text_std = np.where(std < _EPS, 1.0, std)
        standardized = (reduced - self._text_mean) / self._text_std
        self._block_scale[_TEXT_BLOCK] = self._variance_scale(standardized)

    # -- transform ----------------------------------------------------------- #
    def transform(self, features) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.transform called before fit")
        single = isinstance(features, CandidateFeatures)
        items: Sequence[CandidateFeatures] = [features] if single else features
        rows = [self._transform_one(f) for f in items]
        out = np.stack(rows) if rows else np.zeros((0, len(self.feature_names_)))
        return out[0] if single else out

    def _transform_one(self, f: CandidateFeatures) -> np.ndarray:
        parts: list[np.ndarray] = []
        for block in _DICT_BLOCKS:
            keys = self._keys.get(block, [])
            if not keys:
                continue
            vec = self._dict_to_vec(getattr(f, block), keys)
            vec = self._standardize(vec, self._mean[block], self._std[block])
            parts.append(self._weight(vec, block))

        if self._has_text:
            parts.append(self._transform_text(f))

        if not parts:
            return np.zeros(len(self.feature_names_), dtype=np.float64)
        return np.concatenate(parts)

    def _transform_text(self, f: CandidateFeatures) -> np.ndarray:
        dim = self._pca.n_components_
        if f.text_embedding is None:
            return np.zeros(dim, dtype=np.float64)
        emb = np.asarray(f.text_embedding, dtype=np.float64).reshape(1, -1)
        reduced = self._pca.transform(emb)[0]
        reduced = self._standardize(reduced, self._text_mean, self._text_std)
        return self._weight(reduced, _TEXT_BLOCK)

    # -- helpers ------------------------------------------------------------- #
    def _standardize(
        self, vec: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        if not self.config.standardize:
            return vec
        return (vec - mean) / std

    def _weight(self, vec: np.ndarray, block: str) -> np.ndarray:
        w = float(self.config.block_weights.get(block, 1.0))
        if self.config.balance_block_dims:
            w = w * self._block_scale.get(block, 1.0)
        return vec * w

    @staticmethod
    def _variance_scale(standardized: np.ndarray) -> float:
        """Scale that normalizes a block's total variance to 1 (anti-swamping)."""
        total_var = float(np.sum(np.var(standardized, axis=0, ddof=0)))
        return 1.0 / np.sqrt(total_var) if total_var > _EPS else 1.0

    @staticmethod
    def _collect_keys(corpus: Sequence[CandidateFeatures], block: str) -> list[str]:
        keys: set[str] = set()
        for f in corpus:
            keys.update(getattr(f, block).keys())
        return sorted(keys)

    @staticmethod
    def _dict_to_vec(d: dict[str, float], keys: list[str]) -> np.ndarray:
        return np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float64)

    def _build_feature_names(self) -> None:
        names: list[str] = []
        for block in _DICT_BLOCKS:
            names.extend(f"{block}::{k}" for k in self._keys.get(block, []))
        if self._has_text:
            names.extend(f"text::pca_{i}" for i in range(self._pca.n_components_))
        self.feature_names_ = names

    # -- introspection ------------------------------------------------------- #
    def block_slices(self) -> dict[str, slice]:
        """Return the column slice occupied by each block in the output vector."""
        slices: dict[str, slice] = {}
        start = 0
        for block in _DICT_BLOCKS:
            n = len(self._keys.get(block, []))
            if n:
                slices[block] = slice(start, start + n)
                start += n
        if self._has_text:
            n = self._pca.n_components_
            slices[_TEXT_BLOCK] = slice(start, start + n)
        return slices

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)


__all__ = ["NormalizerConfig", "FeatureNormalizer"]

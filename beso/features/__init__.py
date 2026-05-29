"""Candidate featurization phi(z) (the ``Featurizer`` protocol).

Builds block-separated, parent-centered (delta) features and provides per-block
standardization + text-block dimensionality reduction so the high-dimensional
embedding block cannot swamp the cheap structured signals.

Planned modules: ``featurizer.py``, ``normalization.py``, ``embeddings.py``,
``semantic_labels.py``.
"""

from beso.features.featurizer import (
    FeatureExtractor,
    HashingEmbedder,
    approx_tokens,
    compute_structural_metrics,
)
from beso.features.normalization import FeatureNormalizer, NormalizerConfig

__all__ = [
    "FeatureExtractor",
    "HashingEmbedder",
    "approx_tokens",
    "compute_structural_metrics",
    "FeatureNormalizer",
    "NormalizerConfig",
]

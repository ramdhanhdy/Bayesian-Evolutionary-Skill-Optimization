"""Acquisition and batch selection (``AcquisitionFunction`` + ``BatchSelector``).

Implements the pool-normalized a_BESO:

    a(z) = mu~ + kappa*sigma~ + lambda*d~(z,A) - alpha*c~(z) - gamma*q~_invalid(z)

with every term normalized over the current pool C_t (dimensionless weights),
plus submodular max-min batch selection that updates the reference set with
already-selected members to avoid intra-batch near-duplicates.
"""

from beso.acquisition.base import (
    AcquisitionConfig,
    AcquisitionTerms,
    build_pool_statistics,
    candidate_cost,
    candidate_invalid_risk,
    compose_acquisition_score,
    normalize_term,
)
from beso.acquisition.batch import BatchSelectionConfig, GreedySubmodularBatchSelector
from beso.acquisition.composite import PoolNormalizedBESOAcquisition
from beso.acquisition.diversity import (
    archive_entry_distance,
    candidate_archive_diversity,
    candidate_distance,
    candidate_novelty,
    feature_distance,
)

__all__ = [
    "AcquisitionConfig",
    "AcquisitionTerms",
    "BatchSelectionConfig",
    "GreedySubmodularBatchSelector",
    "PoolNormalizedBESOAcquisition",
    "archive_entry_distance",
    "build_pool_statistics",
    "candidate_archive_diversity",
    "candidate_cost",
    "candidate_distance",
    "candidate_invalid_risk",
    "candidate_novelty",
    "compose_acquisition_score",
    "feature_distance",
    "normalize_term",
]

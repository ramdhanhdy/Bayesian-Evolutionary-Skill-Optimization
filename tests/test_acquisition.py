from __future__ import annotations

import math

import pytest

from beso.acquisition import (
    AcquisitionConfig,
    BatchSelectionConfig,
    GreedySubmodularBatchSelector,
    PoolNormalizedBESOAcquisition,
)
from beso.core.types import (
    Candidate,
    CandidateFeatures,
    SkillArtifact,
    SurrogatePrediction,
)


def _candidate(cid: str, doc: str, *, tokens: float = 10.0, invalid: float = 0.0) -> Candidate:
    skill = SkillArtifact(skill_id=cid, name=cid, document=doc)
    features = CandidateFeatures(
        candidate_id=cid,
        structural={"child_tokens": tokens},
        semantic={"invalid_risk": invalid},
    )
    return Candidate(candidate_id=cid, skill=skill, features=features)


def _prediction(cid: str, mu: float, sigma: float) -> SurrogatePrediction:
    return SurrogatePrediction(candidate_id=cid, mu=mu, sigma=sigma)


def test_pool_normalized_acquisition_is_affine_scale_invariant() -> None:
    cands = [
        _candidate("z1", "alpha beta gamma", tokens=10.0, invalid=0.1),
        _candidate("z2", "delta epsilon zeta", tokens=20.0, invalid=0.4),
        _candidate("z3", "eta theta iota", tokens=30.0, invalid=0.8),
    ]
    preds = [
        _prediction("z1", 0.10, 0.01),
        _prediction("z2", 0.20, 0.03),
        _prediction("z3", 0.35, 0.05),
    ]
    acq = PoolNormalizedBESOAcquisition(
        AcquisitionConfig(kappa=0.7, diversity_lambda=0.0, cost_alpha=0.2, invalid_gamma=0.3)
    )
    original = [c.acquisition_score for c in acq.score_pool(cands, preds, archive=[])]

    scaled_cands = [
        _candidate("z1", "alpha beta gamma", tokens=105.0, invalid=0.20),
        _candidate("z2", "delta epsilon zeta", tokens=205.0, invalid=0.35),
        _candidate("z3", "eta theta iota", tokens=305.0, invalid=0.55),
    ]
    scaled_preds = [
        _prediction("z1", 10.2, 5.04),
        _prediction("z2", 10.4, 5.12),
        _prediction("z3", 10.7, 5.20),
    ]
    scaled = [
        c.acquisition_score
        for c in acq.score_pool(scaled_cands, scaled_preds, archive=[])
    ]

    assert scaled == pytest.approx(original, abs=1e-9)


def test_zero_variance_terms_are_finite_and_neutral() -> None:
    cands = [
        _candidate("z1", "same document", tokens=10.0, invalid=0.2),
        _candidate("z2", "same document", tokens=10.0, invalid=0.2),
    ]
    preds = [_prediction("z1", 0.5, 0.1), _prediction("z2", 0.5, 0.1)]
    acq = PoolNormalizedBESOAcquisition()
    scored = acq.score_pool(cands, preds, archive=[])

    assert all(c.acquisition_score is not None for c in scored)
    assert all(math.isfinite(float(c.acquisition_score)) for c in scored)
    assert scored[0].acquisition_score == pytest.approx(scored[1].acquisition_score)


def test_acquisition_directionality() -> None:
    low = _candidate("low", "compact rule", tokens=10.0, invalid=0.1)
    high = _candidate("high", "better rule", tokens=10.0, invalid=0.1)
    acq = PoolNormalizedBESOAcquisition(AcquisitionConfig(diversity_lambda=0.0))
    scored = acq.score_pool(
        [low, high],
        [_prediction("low", 0.2, 0.05), _prediction("high", 0.4, 0.10)],
        archive=[],
    )

    assert scored[1].acquisition_score > scored[0].acquisition_score


def test_submodular_batch_selector_updates_in_batch_novelty() -> None:
    c1 = _candidate("z1", "repeat repeat repeat")
    c2 = _candidate("z2", "repeat repeat repeat")
    c3 = _candidate("z3", "novel search verification policy")
    c1.acquisition_score = 10.0
    c2.acquisition_score = 9.9
    c3.acquisition_score = 9.6
    c1.features = None
    c2.features = None
    c3.features = None

    selector = GreedySubmodularBatchSelector(
        BatchSelectionConfig(diversity_weight=3.0)
    )
    selected = selector.select([c1, c2, c3], k=2)

    assert [c.candidate_id for c in selected] == ["z1", "z3"]

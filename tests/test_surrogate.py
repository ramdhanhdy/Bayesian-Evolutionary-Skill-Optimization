"""Verification tests for the surrogate engine.

Confirms the three contracted requirements:
1. Delta target: the surrogate models improvement over the parent, and absolute
   predictions reconstruct as mu = parent_mean + mu_delta.
2. Total predictive variance = epistemic (ensemble disagreement) + aleatoric
   (observation noise), both non-negative and present.
3. Calibration: temperature/isotonic recalibration brings standardized residuals
   toward unit variance.

The surrogate consumes *normalized features* end-to-end (it owns a
FeatureNormalizer fit during ``fit``).
"""

from __future__ import annotations

import numpy as np
import pytest

from beso.core.types import (
    Candidate,
    EditCategory,
    EditOperation,
    EditProposal,
    Observation,
    SkillArtifact,
    SkillMetadata,
    SkillSection,
)
from beso.features import FeatureExtractor
from beso.features.featurizer import HashingEmbedder
from beso.surrogate import (
    BaggingEnsembleSurrogate,
    IsotonicCalibrator,
    TemperatureScaler,
)
from beso.surrogate.calibration import _pav_nondecreasing

PARENT_MEAN = 0.5
NOISE_SD = 0.05
RNG = np.random.default_rng(7)

PARENT_DOC = "# Goal\nSolve.\n\n## Core Procedure\n- Read.\n- Search.\n"


def _parent() -> SkillArtifact:
    return SkillArtifact(skill_id="z0", name="seed", document=PARENT_DOC)


def _candidate(i: int) -> Candidate:
    bullets = "\n".join(f"- Verify rule {j}." for j in range(i + 1))
    doc = PARENT_DOC + "\n## Verification Checklist\n" + bullets + "\n"
    skill = SkillArtifact(
        skill_id=f"z{i+1}",
        name=f"z{i+1}",
        document=doc,
        metadata=SkillMetadata(parent_id="z0", lineage_depth=1),
    )
    edit = EditProposal(
        edit_id=f"e{i+1}",
        parent_skill_id="z0",
        operation=EditOperation.APPEND,
        content=bullets,
        category=EditCategory.ADD_RULE,
        target_section=SkillSection.VERIFICATION_CHECKLIST,
        source_type="failure",
    )
    return Candidate(candidate_id=f"z{i+1}", skill=skill, parent_id="z0", edit=edit)


def _true_delta(features) -> float:
    # True improvement is linear in the parent-relative bullet delta.
    return 0.02 * float(features.structural["d_bullets"])


def _build_dataset(n_candidates: int = 24, reps: int = 4):
    fx = FeatureExtractor(embed_fn=HashingEmbedder(dim=96))
    parent = _parent()
    parent_history = [
        Observation(candidate_id="z0", batch_ids=("p1",), observed_score=PARENT_MEAN),
        Observation(candidate_id="z0", batch_ids=("p2",), observed_score=PARENT_MEAN),
    ]
    features = []
    observations = []
    truth = {}
    for i in range(n_candidates):
        cand = _candidate(i)
        feats = fx.featurize(cand, parent, history=parent_history)
        features.append(feats)
        d = _true_delta(feats)
        truth[cand.candidate_id] = d
        true_abs = PARENT_MEAN + d
        for r in range(reps):
            noisy = float(true_abs + RNG.normal(0.0, NOISE_SD))
            observations.append(
                Observation(
                    candidate_id=cand.candidate_id,
                    batch_ids=(f"b{r}",),
                    observed_score=noisy,
                )
            )
    return features, observations, truth


def test_delta_target_and_absolute_reconstruction() -> None:
    features, observations, truth = _build_dataset()
    surr = BaggingEnsembleSurrogate(n_members=12, alpha=1.0, random_state=1)
    surr.fit(features, observations)
    assert surr.is_fitted
    assert surr.normalizer.n_features > 0

    preds = surr.predict_many(features)
    # Absolute prediction must reconstruct from parent baseline + delta.
    for p in preds:
        assert p.mu == pytest.approx(PARENT_MEAN + p.mu_delta, abs=1e-9)

    # Predicted delta must correlate strongly with the true delta.
    pred_delta = np.array([p.mu_delta for p in preds])
    true_delta = np.array([truth[p.candidate_id] for p in preds])
    corr = float(np.corrcoef(pred_delta, true_delta)[0, 1])
    assert corr > 0.7


def test_total_variance_decomposition() -> None:
    features, observations, _ = _build_dataset()
    surr = BaggingEnsembleSurrogate(n_members=12, random_state=2)
    surr.fit(features, observations)

    preds = surr.predict_many(features)
    for p in preds:
        assert p.sigma > 0.0
        assert p.epistemic_var >= 0.0
        assert p.aleatoric_var > 0.0
        # sigma^2 == epistemic + aleatoric (consistent after calibration).
        assert p.sigma ** 2 == pytest.approx(
            p.epistemic_var + p.aleatoric_var, rel=1e-6, abs=1e-9
        )

    # Aleatoric estimate should be in the neighborhood of the true noise variance.
    assert surr._aleatoric_var == pytest.approx(NOISE_SD ** 2, rel=1.5)


def test_calibration_improves_coverage() -> None:
    features, observations, _ = _build_dataset()
    surr = BaggingEnsembleSurrogate(
        n_members=16, random_state=3, calibrator=TemperatureScaler()
    )
    surr.fit(features, observations)
    assert surr.is_calibrated

    # Standardized residuals on observations should have ~unit variance.
    feat_by_id = {f.candidate_id: f for f in features}
    pred_by_id = {p.candidate_id: p for p in surr.predict_many(features)}
    z2 = []
    for obs in observations:
        p = pred_by_id[obs.candidate_id]
        if p.sigma > 0:
            z2.append(((obs.observed_score - p.mu) / p.sigma) ** 2)
    mean_z2 = float(np.mean(z2))
    assert 0.4 < mean_z2 < 2.5


def test_isotonic_calibrator_runs() -> None:
    features, observations, _ = _build_dataset()
    surr = BaggingEnsembleSurrogate(
        n_members=12, random_state=4, calibrator=IsotonicCalibrator(min_points=8)
    )
    surr.fit(features, observations)
    assert surr.is_calibrated
    preds = surr.predict_many(features)
    assert all(np.isfinite(p.sigma) and p.sigma > 0 for p in preds)


def test_pav_is_nondecreasing() -> None:
    y = np.array([3.0, 1.0, 2.0, 0.5, 4.0])
    g = _pav_nondecreasing(y)
    assert np.all(np.diff(g) >= -1e-9)
    # PAVA preserves the total sum (mean-preserving).
    assert float(np.sum(g)) == pytest.approx(float(np.sum(y)))

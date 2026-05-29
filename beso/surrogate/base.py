"""Base surrogate: delta target, variance decomposition, calibration hook.

Implements the parts of the ``Surrogate`` protocol that are independent of the
concrete predictor (ensemble, GP, ...):

1. **Delta target** (Breakdown S7.5) - the surrogate models the parent-relative
   improvement Delta(z, z_p) = J(z) - J(z_p), not the absolute score. The parent
   baseline J(z_p) is read from the featurizer's ``history["parent_mean"]`` so no
   extra bookkeeping is needed. Absolute predictions are reconstructed as
   ``mu = parent_mean + mu_delta`` for acquisition, while ``mu_delta`` is exposed
   directly.

2. **Total predictive variance** (Breakdown S7.1) -
   ``sigma_t^2(z) = epistemic(z) + aleatoric``, where the epistemic term comes
   from the concrete predictor (ensemble disagreement) and the aleatoric term is
   the observation noise sigma_eps^2 estimated from the data.

3. **Calibration** - a :class:`~beso.surrogate.calibration.Calibrator` is fit on
   out-of-bag (residual, sigma) pairs so predicted intervals match reality.

Subclasses implement :meth:`_fit_core`, :meth:`_predict_core`, and
:meth:`_oob_predict` (all in *delta* space).
"""

from __future__ import annotations

import abc
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from beso.core.types import CandidateFeatures, Observation, SurrogatePrediction
from beso.features.normalization import FeatureNormalizer
from beso.surrogate.calibration import Calibrator, IdentityCalibrator, TemperatureScaler

PARENT_MEAN_KEY = "parent_mean"
_EPS = 1e-12


class BaseSurrogate(abc.ABC):
    """Abstract delta-target surrogate with calibrated total variance."""

    def __init__(
        self,
        *,
        normalizer: Optional[FeatureNormalizer] = None,
        calibrator: Optional[Calibrator] = None,
        min_obs_for_calibration: int = 8,
        aleatoric_floor: float = 1e-6,
    ) -> None:
        self.normalizer = normalizer if normalizer is not None else FeatureNormalizer()
        self.calibrator = calibrator if calibrator is not None else TemperatureScaler()
        self.min_obs_for_calibration = int(min_obs_for_calibration)
        self.aleatoric_floor = float(aleatoric_floor)
        self._fitted = False
        self._calibrated = False
        self._aleatoric_var = 0.0
        self._n_train = 0

    # -- protocol ------------------------------------------------------------ #
    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        features: Sequence[CandidateFeatures],
        observations: Sequence[Observation],
    ) -> None:
        if not features:
            raise ValueError("cannot fit surrogate with no candidate features")
        feat_by_id = {f.candidate_id: f for f in features}

        rows: list[CandidateFeatures] = []
        targets: list[float] = []
        for obs in observations:
            f = feat_by_id.get(obs.candidate_id)
            if f is None:
                continue
            parent_mean = float(f.history.get(PARENT_MEAN_KEY, 0.0))
            rows.append(f)
            targets.append(float(obs.observed_score) - parent_mean)  # delta target
        if not rows:
            raise ValueError(
                "no observations matched the provided candidate features; "
                "cannot build a training set"
            )

        # Normalizer is fit on the unique candidate corpus, then applied to rows.
        self.normalizer.fit(features)
        X = np.atleast_2d(self.normalizer.transform(rows))
        y = np.asarray(targets, dtype=np.float64)
        self._n_train = X.shape[0]

        self._fit_core(X, y)
        oob_mu, oob_epi = self._oob_predict(X)

        self._aleatoric_var = self._estimate_aleatoric(observations, y, oob_mu, oob_epi)
        self._fit_calibration(y, oob_mu, oob_epi)
        self._fitted = True

    def predict(self, features: CandidateFeatures) -> SurrogatePrediction:
        return self._predict_batch([features])[0]

    def predict_many(
        self, features: Sequence[CandidateFeatures]
    ) -> list[SurrogatePrediction]:
        feats = list(features)
        return self._predict_batch(feats) if feats else []

    # -- internals ----------------------------------------------------------- #
    def _predict_batch(
        self, feats: Sequence[CandidateFeatures]
    ) -> list[SurrogatePrediction]:
        if not self._fitted:
            raise RuntimeError("surrogate.predict called before fit")
        X = np.atleast_2d(self.normalizer.transform(list(feats)))
        mu_delta, epi = self._predict_core(X)
        aleatoric = float(self._aleatoric_var)
        preds: list[SurrogatePrediction] = []
        for i, f in enumerate(feats):
            parent_mean = float(f.history.get(PARENT_MEAN_KEY, 0.0))
            epistemic = max(float(epi[i]), 0.0)
            total_var = epistemic + aleatoric
            total_sigma = float(np.sqrt(total_var)) if total_var > 0 else 0.0
            cal_sigma = (
                float(self.calibrator.transform(total_sigma))
                if total_sigma > 0
                else 0.0
            )
            ratio = (cal_sigma / total_sigma) ** 2 if total_sigma > _EPS else 1.0
            d = float(mu_delta[i])
            preds.append(
                SurrogatePrediction(
                    candidate_id=f.candidate_id,
                    mu=parent_mean + d,
                    sigma=cal_sigma,
                    epistemic_var=epistemic * ratio,
                    aleatoric_var=aleatoric * ratio,
                    mu_delta=d,
                )
            )
        return preds

    def _estimate_aleatoric(
        self,
        observations: Sequence[Observation],
        y: np.ndarray,
        oob_mu: np.ndarray,
        oob_epi: np.ndarray,
    ) -> float:
        """Estimate observation noise sigma_eps^2.

        Preferred: average within-candidate sample variance across repeated
        observations (a direct, model-free noise estimate). Fallback: subtract
        the mean epistemic variance from the total OOB residual variance.
        """
        by_cand: dict[str, list[float]] = defaultdict(list)
        for obs in observations:
            by_cand[obs.candidate_id].append(float(obs.observed_score))
        within = [
            float(np.var(v, ddof=1)) for v in by_cand.values() if len(v) >= 2
        ]
        if within:
            return max(float(np.mean(within)), self.aleatoric_floor)

        resid_var = float(np.var(y - oob_mu)) if y.size else 0.0
        mean_epi = float(np.mean(oob_epi)) if oob_epi.size else 0.0
        return max(resid_var - mean_epi, self.aleatoric_floor)

    def _fit_calibration(
        self, y: np.ndarray, oob_mu: np.ndarray, oob_epi: np.ndarray
    ) -> None:
        residuals = y - oob_mu
        total_var = np.maximum(oob_epi + self._aleatoric_var, 0.0)
        sigmas = np.sqrt(total_var)
        enough = residuals.size >= self.min_obs_for_calibration
        if enough and not isinstance(self.calibrator, IdentityCalibrator):
            self.calibrator.fit(residuals, sigmas)
            self._calibrated = True
        else:
            self.calibrator = IdentityCalibrator()
            self.calibrator.fit(residuals, sigmas)
            self._calibrated = False

    # -- abstract (delta space) --------------------------------------------- #
    @abc.abstractmethod
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the concrete predictor on (X, delta-targets)."""

    @abc.abstractmethod
    def _predict_core(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu_delta, epistemic_var) arrays for rows of X."""

    @abc.abstractmethod
    def _oob_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return out-of-bag (mu_delta, epistemic_var) for the training rows."""


__all__ = ["BaseSurrogate", "PARENT_MEAN_KEY"]

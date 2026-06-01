"""Uncertainty recalibration for the surrogate (Breakdown S7.4; Spec S13.5).

The bagged ensemble produces a predictive standard deviation sigma_t(z); raw
ensemble disagreement is typically *miscalibrated* (over- or under-confident).
These calibrators learn a correction on out-of-bag (residual, sigma) pairs so the
predicted intervals match empirical coverage.

Provided strategies:
- :class:`IdentityCalibrator` - no-op (cold start / disabled).
- :class:`TemperatureScaler` - single global scalar s minimizing Gaussian NLL;
  s^2 = mean((residual / sigma)^2). Robust default.
- :class:`IsotonicCalibrator` - monotonic, sigma-dependent correction via
  pool-adjacent-violators isotonic regression of |residual| on sigma, mapped to
  a corrected standard deviation. Handles heteroscedastic miscalibration.
"""

from __future__ import annotations

import abc

import numpy as np

_EPS = 1e-9
# E|X| for X ~ N(0, 1): mean of the half-normal.
_HALF_NORMAL_MEAN = np.sqrt(2.0 / np.pi)


class Calibrator(abc.ABC):
    """Maps a raw predictive sigma to a calibrated sigma."""

    is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "Calibrator":
        """Fit on out-of-bag residuals (y - mu) and their predicted sigmas."""

    @abc.abstractmethod
    def transform(self, sigma):
        """Return calibrated sigma for a scalar or array of sigmas."""


class IdentityCalibrator(Calibrator):
    """No recalibration; returns sigma unchanged."""

    def __init__(self) -> None:
        self.is_fitted = True

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "IdentityCalibrator":
        self.is_fitted = True
        return self

    def transform(self, sigma):
        return sigma


class TemperatureScaler(Calibrator):
    """Global variance recalibration by a single temperature scalar s.

    Minimizes the Gaussian negative log-likelihood of the standardized residuals,
    giving the closed form ``s = sqrt(mean((residual / sigma)^2))``. Values s > 1
    indicate the raw model was over-confident.
    """

    def __init__(self, floor: float = _EPS) -> None:
        self.floor = float(floor)
        self.scale = 1.0
        self.is_fitted = False

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "TemperatureScaler":
        r = np.asarray(residuals, dtype=np.float64).ravel()
        s = np.asarray(sigmas, dtype=np.float64).ravel()
        s = np.where(s < self.floor, self.floor, s)
        if r.size == 0:
            self.scale = 1.0
        else:
            z2 = (r / s) ** 2
            val = float(np.sqrt(np.mean(z2)))
            self.scale = val if np.isfinite(val) and val > 0 else 1.0
        self.is_fitted = True
        return self

    def transform(self, sigma):
        scaled = np.asarray(sigma, dtype=np.float64) * self.scale
        return float(scaled) if np.ndim(sigma) == 0 else scaled


class IsotonicCalibrator(Calibrator):
    """Monotonic, sigma-dependent recalibration via isotonic regression.

    Fits a non-decreasing map g(sigma) ~ E[|residual| | sigma] using the
    pool-adjacent-violators algorithm, then corrects to a calibrated standard
    deviation ``sigma' = g(sigma) / E|N(0,1)|``. Falls back to a global scalar
    when fewer than ``min_points`` calibration samples are available.
    """

    def __init__(self, min_points: int = 8, floor: float = _EPS) -> None:
        self.min_points = int(min_points)
        self.floor = float(floor)
        self._x: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._fallback = TemperatureScaler(floor=floor)
        self.is_fitted = False

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "IsotonicCalibrator":
        r = np.abs(np.asarray(residuals, dtype=np.float64).ravel())
        s = np.asarray(sigmas, dtype=np.float64).ravel()
        s = np.where(s < self.floor, self.floor, s)
        self._fallback.fit(residuals, sigmas)
        if r.size < self.min_points:
            self._x = None
            self._g = None
            self.is_fitted = True
            return self
        order = np.argsort(s)
        xs = s[order]
        ys = r[order]
        self._x = xs
        self._g = _pav_nondecreasing(ys)
        self.is_fitted = True
        return self

    def transform(self, sigma):
        scalar = np.ndim(sigma) == 0
        s = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
        if self._x is None or self._g is None:
            out = self._fallback.transform(s)
        else:
            g = np.interp(s, self._x, self._g, left=self._g[0], right=self._g[-1])
            out = np.maximum(g / _HALF_NORMAL_MEAN, self.floor)
        out = np.asarray(out, dtype=np.float64)
        return float(out[0]) if scalar else out


def _pav_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: nearest non-decreasing fit (unit weights)."""
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n == 0:
        return y
    values = y.copy()
    weights = np.ones(n)
    # Iterative pooling.
    level_values: list[float] = []
    level_weights: list[float] = []
    level_counts: list[int] = []
    for i in range(n):
        v = values[i]
        w = weights[i]
        c = 1
        while level_values and level_values[-1] >= v:
            pv = level_values.pop()
            pw = level_weights.pop()
            pc = level_counts.pop()
            v = (v * w + pv * pw) / (w + pw)
            w = w + pw
            c = c + pc
        level_values.append(v)
        level_weights.append(w)
        level_counts.append(c)
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for v, c in zip(level_values, level_counts):
        out[pos : pos + c] = v
        pos += c
    return out


__all__ = [
    "Calibrator",
    "IdentityCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
]

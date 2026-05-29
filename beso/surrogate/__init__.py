"""Bayesian surrogate over candidate utility (the ``Surrogate`` protocol).

v0 default: a homogeneous bootstrap-bagged ensemble modeling the parent-relative
delta Delta(z, z_p), exposing calibrated mu_t and sigma_t with the variance
decomposed into epistemic (model disagreement) and aleatoric (observation noise)
components. A composite-kernel GP path is available for very small t.

Planned modules: ``base.py``, ``ensemble.py``, ``calibration.py``,
``gaussian_process.py``.
"""

from beso.surrogate.base import BaseSurrogate
from beso.surrogate.calibration import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureScaler,
)
from beso.surrogate.ensemble import BaggingEnsembleSurrogate, RidgeRegressor

__all__ = [
    "BaseSurrogate",
    "BaggingEnsembleSurrogate",
    "RidgeRegressor",
    "Calibrator",
    "IdentityCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
]

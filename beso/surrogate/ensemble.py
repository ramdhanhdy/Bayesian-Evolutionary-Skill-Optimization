"""Bootstrap-bagged ensemble surrogate (Breakdown S7.2; Spec S13.2).

The v0 surrogate is a homogeneous **bag of regressors** trained on bootstrap
resamples with **random feature subspaces**. Two design points:

- Epistemic variance is the *clean* disagreement across members (variance of
  member predictions), which the bootstrap + subspace sampling makes meaningful
  in the small-data Bayesian-optimization regime.
- Out-of-bag (OOB) predictions give a leakage-free held-out signal used by the
  base class to estimate aleatoric noise and to fit the calibrator.

The default base learner is a dependency-free ridge regressor (closed form), so
the surrogate runs without scikit-learn. A different base learner can be injected
via ``base_factory`` (e.g. an sklearn ``DecisionTreeRegressor`` for nonlinearity).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from beso.surrogate.base import BaseSurrogate

_EPS = 1e-12


class RidgeRegressor:
    """Closed-form ridge regression with an unregularized intercept."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        A = np.hstack([X, np.ones((n, 1))])
        reg = self.alpha * np.eye(d + 1)
        reg[-1, -1] = 0.0  # do not regularize the intercept
        ata = A.T @ A + reg
        aty = A.T @ y
        try:
            coef = np.linalg.solve(ata, aty)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.w = coef[:-1]
        self.b = float(coef[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.w + self.b


BaseFactory = Callable[[], object]


class BaggingEnsembleSurrogate(BaseSurrogate):
    """Bootstrap + random-subspace bag of regressors over delta targets."""

    def __init__(
        self,
        *,
        n_members: int = 8,
        alpha: float = 1.0,
        base_factory: Optional[BaseFactory] = None,
        feature_subsample: float = 0.8,
        bootstrap: bool = True,
        random_state: int = 0,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        self.n_members = int(n_members)
        self.alpha = float(alpha)
        self.base_factory: BaseFactory = base_factory or (
            lambda: RidgeRegressor(alpha=self.alpha)
        )
        self.feature_subsample = float(feature_subsample)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        # Each member: (learner, feature_indices, in_bag_index_set).
        self._members: list[tuple[object, np.ndarray, set[int]]] = []
        self._n_features = 0

    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self._n_features = d
        self._members = []
        k = max(1, int(round(self.feature_subsample * d)))
        for _ in range(self.n_members):
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            feat = np.sort(rng.choice(d, size=k, replace=False))
            learner = self.base_factory()
            learner.fit(X[idx][:, feat], y[idx])
            self._members.append((learner, feat, set(int(i) for i in idx)))

    def _predict_core(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self._member_predictions(X)  # (M, n)
        mu = preds.mean(axis=0)
        if preds.shape[0] > 1:
            epi = preds.var(axis=0, ddof=1)
        else:
            epi = np.zeros(X.shape[0])
        return mu, epi

    def _oob_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        m = len(self._members)
        preds = np.full((m, n), np.nan)
        for mi, (learner, feat, in_bag) in enumerate(self._members):
            oob_mask = np.array([i not in in_bag for i in range(n)])
            if oob_mask.any():
                preds[mi, oob_mask] = learner.predict(X[oob_mask][:, feat])

        mu = np.empty(n)
        epi = np.empty(n)
        full = None
        for i in range(n):
            col = preds[:, i]
            valid = col[~np.isnan(col)]
            if valid.size >= 2:
                mu[i] = valid.mean()
                epi[i] = valid.var(ddof=1)
            elif valid.size == 1:
                mu[i] = valid[0]
                epi[i] = 0.0
            else:
                if full is None:
                    full = self._member_predictions(X)
                col_full = full[:, i]
                mu[i] = col_full.mean()
                epi[i] = col_full.var(ddof=1) if col_full.size > 1 else 0.0
        return mu, epi

    def _member_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.stack(
            [learner.predict(X[:, feat]) for learner, feat, _ in self._members],
            axis=0,
        )


__all__ = ["BaggingEnsembleSurrogate", "RidgeRegressor"]

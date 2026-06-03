# BESO Surrogate Layer

This directory implements the Bayesian surrogate model layer for Bayesian Evolutionary Skill Optimization (BESO). The surrogate is a learned predictor that estimates candidate skill utility and uncertainty to guide selection under a limited rollout budget.

## Architectural Setup

The surrogate architecture is divided into three core responsibilities:

1. **Base Surrogate (`base.py`)**: Defines the `BaseSurrogate` abstract class which implements target transformation (delta modeling), variance decomposition, and the out-of-bag (OOB) calibration loop.
2. **Homogeneous Ensemble (`ensemble.py`)**: Implements `BaggingEnsembleSurrogate`, a bootstrap-bagged ensemble with random feature subspace projection. It defaults to a dependency-free, closed-form `RidgeRegressor` as its base learner.
3. **Uncertainty Recalibration (`calibration.py`)**: Fits a `Calibrator` on OOB residuals to correct ensemble under- or over-confidence. Supports global `TemperatureScaler` and heteroscedastic `IsotonicCalibrator` (using the Pool-Adjacent-Violators algorithm).

---

## Key Design Points & Decision Rationale

### 1. Delta-Target Modeling
* **Method**: The surrogate models the parent-relative improvement target $\Delta(z, z_p) = J(z) - J(z_p)$ rather than the absolute performance score $J(z)$.
* **Rationale**:
  * **Mitigates Covariate Shift**: The absolute score of a skill document varies drastically depending on the specific tasks or prompts being evaluated. Delta target focuses learning entirely on the impact of the *textual edit* itself relative to its direct parent.
  * **Variance Stabilization**: Normalizing by parent performance controls baseline drift as the optimization runs, leading to a much more stable training signal for the underlying regressors.

### 2. Epistemic & Aleatoric Variance Decomposition
* **Method**: Predictive variance is modeled as $\sigma_t^2(z) = \sigma_{\mathrm{epistemic}}^2(z) + \sigma_{\mathrm{aleatoric}}^2$.
  * *Epistemic uncertainty* $\sigma_{\mathrm{epistemic}}^2(z)$ is computed as the variance of member predictions in the bagged ensemble.
  * *Aleatoric uncertainty* $\sigma_{\mathrm{aleatoric}}^2$ represents observation (evaluation) noise. It is estimated via within-candidate sample variance across repeated evaluations when replicates exist, falling back to residual OOB variance subtraction.
* **Rationale**:
  * **Prevents Wasteful Rollouts**: Disentangling model disagreement (epistemic) from inherent evaluation noise (aleatoric) ensures the acquisition function (e.g., UCB) targets regions of true model ignorance rather than repeatedly sampling highly noisy but well-understood skills.

### 3. Bootstrap Bagging & Random Subspaces
* **Method**: The ensemble constructs $M$ regressors, each trained on a bootstrap sample of observations and a random subset of feature dimensions (random subspace projection).
* **Rationale**:
  * **Uncertainty Estimation in Data-Scarce Regimes**: In the early iterations of Bayesian optimization, the dataset size is extremely small. Standard ensembles overfit and collapse to zero variance. Bootstrap combined with random subspace projection forces diverse feature perspective, guaranteeing meaningful epistemic variance even under extremely low sample counts.

### 4. Zero-Dependency Default Ridge Regressor
* **Method**: Default base learner is `RidgeRegressor`, a lightweight implementation of closed-form ridge regression with an unregularized intercept.
* **Rationale**:
  * **Frictionless Portability**: Running the optimization loop without heavy runtime dependencies (such as `scikit-learn`) makes BESO highly portable and easily integratable across different cluster environments.
  * **Flexibility**: The `base_factory` pattern allows seamless injection of non-linear scikit-learn models (e.g., `DecisionTreeRegressor` or `RandomForestRegressor`) if needed for complex feature interactions.

### 5. Out-of-Bag (OOB) Residual Recalibration
* **Method**: The surrogate utilizes out-of-bag (OOB) predictions—predictions on training points from ensemble members that did not see those points during training—to learn an uncertainty correction.
  * `TemperatureScaler`: Rescales standard deviation globally by minimizing Gaussian negative log-likelihood: $s = \sqrt{\mathrm{mean}((\mathrm{residual} / \sigma)^2)}$.
  * `IsotonicCalibrator`: Fits a non-decreasing map using the pool-adjacent-violators algorithm to model $\mathbb{E}[|\mathrm{residual}| \mid \sigma]$.
* **Rationale**:
  * **Prevents Miscalibration**: Ensemble variance (disagreement) is notoriously miscalibrated, often being highly over-confident. OOB predictions provide a clean, leakage-free validation signal to train calibrators, ensuring that downstream acquisition functions receive accurate and reliable uncertainty bounds.

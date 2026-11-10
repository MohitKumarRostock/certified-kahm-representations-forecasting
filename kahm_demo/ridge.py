"""Scalar ridge regression with an unpenalized intercept and clipped inference.

Based on our research. The local definition ``matched_forecasting_readout_v1``
in ``docs/READOUT_DESIGN.md`` uses the *mean* squared training error plus alpha
times the squared coefficient norm. Consequently,
the centered SVD factors contain n*alpha.  All feature columns are retained,
including constant and linearly dependent columns.

Numerical residual checks describe this float64 computation.  They are not
verified error enclosures for an ideal real-arithmetic estimator.  No data
splits, hyperparameter selection, forecasting certificates, or oracle risks
are handled here.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from numbers import Real
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike


RIDGE_VERSION = "centered-svd-scalar-float64-v1"
SVD_CHECK_TOLERANCE = 1e-10
NORMAL_EQUATION_TOLERANCE = 1e-10
INTERCEPT_CHECK_TOLERANCE = 1e-10


def _real_array(value: ArrayLike, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating-point values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def _feature_matrix(value: ArrayLike, *, width: int | None = None,
                    allow_empty: bool = False) -> np.ndarray:
    array = _real_array(value, "features")
    if (array.ndim != 2 or array.shape[1] == 0
            or (array.shape[0] == 0 and not allow_empty)):
        raise ValueError("features must be a matrix with positive feature width")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"features must have {width} columns")
    return array


def _vector(value: ArrayLike, name: str, *, width: int | None = None) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if width is not None and array.size != width:
        raise ValueError(f"{name} must have {width} entries")
    return array


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _alpha(value: float) -> float:
    result = _finite_scalar(value, "alpha")
    if result <= 0.0:
        raise ValueError("alpha must be positive")
    return result


def _immutable(value: ArrayLike) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _finite(value, name: str):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} is nonfinite")
    return value


def _affine_value(features: np.ndarray, coefficients: np.ndarray, intercept: float) -> float:
    """One canonical scalar readout, including the intercept in math.fsum."""
    terms = [intercept]
    for feature, coefficient in zip(features, coefficients, strict=True):
        product = float(feature) * float(coefficient)
        if not math.isfinite(product):
            raise ValueError("ridge affine readout contains a nonfinite product")
        terms.append(product)
    try:
        value = math.fsum(terms)
    except (OverflowError, ValueError) as error:
        raise ValueError("ridge affine readout is not finite") from error
    if not math.isfinite(value):
        raise ValueError("ridge affine readout is not finite")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RidgeModel:
    coefficients: np.ndarray
    intercept: float
    alpha: float
    training_count: int
    feature_mean: np.ndarray
    target_mean: float
    diagnostics: dict

    def __post_init__(self) -> None:
        coefficients = _vector(self.coefficients, "coefficients")
        mean = _vector(self.feature_mean, "feature_mean", width=coefficients.size)
        intercept = _finite_scalar(self.intercept, "intercept")
        alpha = _alpha(self.alpha)
        target_mean = _finite_scalar(self.target_mean, "target_mean")
        if not 0.0 <= target_mean <= 1.0:
            raise ValueError("target_mean must lie in [0,1]")
        if type(self.training_count) is not int or self.training_count < 1:
            raise ValueError("training_count must be a positive integer")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")
        object.__setattr__(self, "coefficients", _immutable(coefficients))
        object.__setattr__(self, "feature_mean", _immutable(mean))
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "target_mean", target_mean)
        # Metadata is detached from the fitting calculation and has no effect
        # on the immutable numerical state used for prediction.
        object.__setattr__(self, "diagnostics", copy.deepcopy(self.diagnostics))

    @property
    def feature_count(self) -> int:
        return int(self.coefficients.size)

    def predict_unclipped_one(self, features: ArrayLike) -> float:
        row = _vector(features, "features", width=self.feature_count)
        return _affine_value(row, self.coefficients, self.intercept)

    def predict_one(self, features: ArrayLike) -> float:
        value = self.predict_unclipped_one(features)
        # Clipping is applied only after the finite-value check; it must not
        # turn an overflowing affine calculation into a seemingly valid result.
        return min(1.0, max(0.0, value))

    def _predict_rows(self, features: ArrayLike, *, clipped: bool) -> np.ndarray:
        rows = _feature_matrix(features, width=self.feature_count, allow_empty=True)
        cache: dict[bytes, float] = {}
        outputs = []
        for row in rows:
            key = row.tobytes()
            if key not in cache:
                value = _affine_value(row, self.coefficients, self.intercept)
                cache[key] = min(1.0, max(0.0, value)) if clipped else value
            outputs.append(cache[key])
        return _immutable(outputs)

    def predict_unclipped(self, features: ArrayLike) -> np.ndarray:
        """Canonical row predictions, retaining duplicates and their order."""
        return self._predict_rows(features, clipped=False)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Canonical clipped row predictions; query data never refit a mean."""
        return self._predict_rows(features, clipped=True)


def _prepare_design(features: ArrayLike, y: ArrayLike):
    X = _feature_matrix(features)
    targets = _vector(y, "y", width=X.shape[0])
    if np.any(targets < 0.0) or np.any(targets > 1.0):
        raise ValueError("y must lie in [0,1]")
    n, p = X.shape
    with np.errstate(over="ignore", invalid="ignore"):
        feature_mean = np.mean(X, axis=0, dtype=np.float64)
        target_mean = float(np.mean(targets, dtype=np.float64))
        Xc = X - feature_mean
        yc = targets - target_mean
    for name, value in (("feature mean", feature_mean), ("target mean", target_mean),
                        ("centered design", Xc), ("centered targets", yc)):
        _finite(value, name)
    try:
        U, singular, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("ridge SVD failed; no fallback solver was used") from error
    r = min(n, p)
    if U.shape != (n, r) or singular.shape != (r,) or Vt.shape != (r, p):
        raise ValueError("ridge SVD returned incompatible dimensions")
    for name, value in (("SVD U", U), ("SVD singular values", singular), ("SVD Vt", Vt)):
        _finite(value, name)
    if np.any(singular < 0.0) or np.any(singular[1:] > singular[:-1]):
        raise ValueError("ridge SVD singular values are invalid")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reconstructed = (U * singular) @ Vt
        reconstruction_fro = float(np.linalg.norm(reconstructed - Xc, ord="fro"))
        design_fro = float(np.linalg.norm(Xc, ord="fro"))
        u_orthogonality = float(np.linalg.norm(U.T @ U - np.eye(r), ord=np.inf))
        v_orthogonality = float(np.linalg.norm(Vt @ Vt.T - np.eye(r), ord=np.inf))
        gram = Xc.T @ Xc
        right_hand_side = Xc.T @ yc
        projected_targets = U.T @ yc
    for name, value in (("SVD reconstruction residual", reconstruction_fro),
                        ("centered design norm", design_fro),
                        ("SVD U orthogonality residual", u_orthogonality),
                        ("SVD V orthogonality residual", v_orthogonality),
                        ("normal-equation Gram matrix", gram),
                        ("normal-equation right-hand side", right_hand_side),
                        ("projected targets", projected_targets)):
        _finite(value, name)
    reconstruction_relative = (reconstruction_fro / design_fro if design_fro > 0.0
                               else 0.0 if reconstruction_fro == 0.0 else math.inf)
    if (reconstruction_relative > SVD_CHECK_TOLERANCE
            or u_orthogonality > SVD_CHECK_TOLERANCE
            or v_orthogonality > SVD_CHECK_TOLERANCE):
        raise ValueError("ridge SVD failed reconstruction or orthogonality checks")
    rank_threshold = np.finfo(np.float64).eps * max(n, p) * float(singular[0])
    diagnostics = {
        "version": RIDGE_VERSION,
        "dtype": "float64",
        "solver": "numpy.linalg.svd(full_matrices=False)",
        "svd_check_tolerance": SVD_CHECK_TOLERANCE,
        "svd_reconstruction_fro": reconstruction_fro,
        "svd_reconstruction_relative_fro": reconstruction_relative,
        "svd_u_orthogonality_inf": u_orthogonality,
        "svd_v_orthogonality_inf": v_orthogonality,
        "singular_values": [float(value) for value in singular],
        "diagnostic_numerical_rank": int(np.count_nonzero(singular > rank_threshold)),
        "diagnostic_rank_threshold": float(rank_threshold),
        "rank_threshold_used_to_drop_columns": False,
        "feature_count": p,
        "all_feature_columns_retained": True,
        "training_count": n,
        "training_rows": "all supplied occurrences, including duplicates",
        "intercept_penalized": False,
        "objective": "mean((y-b-Xw)^2) + alpha*sum(w^2)",
        "inference": "one-row math.fsum of intercept and coefficient-feature products",
        "output_clipping": "[0,1], after finite affine readout",
        "numerical_enclosure_claimed": False,
        "solver_recovery_used": False,
    }
    return (X, targets, feature_mean, target_mean, Xc, singular, Vt,
            projected_targets, gram, right_hand_side, diagnostics)


def _fit_prepared(design, alpha: float) -> RidgeModel:
    (X, targets, feature_mean, target_mean, Xc, singular, Vt,
     projected_targets, gram, right_hand_side, svd_diagnostics) = design
    n, p = X.shape
    penalty = n * alpha
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("n*alpha must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominators = singular * singular + penalty
        factors = singular / denominators
        coefficients = Vt.T @ (factors * projected_targets)
    for name, value in (("ridge denominators", denominators),
                        ("ridge SVD factors", factors), ("ridge coefficients", coefficients)):
        _finite(value, name)
    intercept = _affine_value(feature_mean, -coefficients, target_mean)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        matrix = gram + penalty * np.eye(p)
        normal_residual = float(np.linalg.norm(matrix @ coefficients - right_hand_side, ord=np.inf))
        normal_scale = (float(np.linalg.norm(matrix, ord=np.inf))
                        * float(np.linalg.norm(coefficients, ord=np.inf))
                        + float(np.linalg.norm(right_hand_side, ord=np.inf)))
        # These matrix evaluations are fitting diagnostics, not the canonical
        # row-inference implementation and not exact empirical-loss records.
        errors = X @ coefficients + intercept - targets
        intercept_residual = abs(float(np.mean(errors, dtype=np.float64)))
        training_mse = float(np.mean(errors * errors, dtype=np.float64))
        coefficient_penalty = float(alpha * np.dot(coefficients, coefficients))
        objective = training_mse + coefficient_penalty
    for name, value in (("ridge normal matrix", matrix),
                        ("normal-equation residual", normal_residual),
                        ("normal-equation scale", normal_scale),
                        ("intercept residual", intercept_residual),
                        ("training MSE diagnostic", training_mse),
                        ("coefficient penalty diagnostic", coefficient_penalty),
                        ("training objective diagnostic", objective)):
        _finite(value, name)
    normal_relative = (normal_residual / normal_scale if normal_scale > 0.0
                       else 0.0 if normal_residual == 0.0 else math.inf)
    if normal_relative > NORMAL_EQUATION_TOLERANCE:
        raise ValueError("ridge fit failed its normal-equation residual check")
    if intercept_residual > INTERCEPT_CHECK_TOLERANCE:
        raise ValueError("ridge fit failed its unpenalized-intercept residual check")
    diagnostics = dict(svd_diagnostics)
    diagnostics.update({
        "alpha": alpha,
        "normal_matrix_penalty": penalty,
        "normal_equation_residual_inf": normal_residual,
        "normal_equation_residual_scale": normal_scale,
        "normal_equation_relative_residual": normal_relative,
        "normal_equation_tolerance": NORMAL_EQUATION_TOLERANCE,
        "intercept_mean_residual_abs": intercept_residual,
        "intercept_check_tolerance": INTERCEPT_CHECK_TOLERANCE,
        "training_unclipped_mse_diagnostic": training_mse,
        "coefficient_penalty_diagnostic": coefficient_penalty,
        "training_objective_diagnostic": objective,
    })
    return RidgeModel(coefficients, intercept, alpha, int(n), feature_mean,
                      target_mean, diagnostics)


def fit_ridge_grid(features: ArrayLike, y: ArrayLike,
                   alphas: Iterable[float]) -> tuple[RidgeModel, ...]:
    """Fit a supplied penalty grid in its original order, sharing one SVD.

    No penalty is selected here.  The caller performs development-only
    validation.  A failure is raised rather than silently dropping a penalty.
    """
    if isinstance(alphas, (str, bytes)):
        raise TypeError("alphas must be an iterable of positive real scalars")
    try:
        values = tuple(_alpha(value) for value in alphas)
    except TypeError as error:
        raise TypeError("alphas must contain positive real scalars") from error
    if not values:
        raise ValueError("the ridge penalty grid must not be empty")
    design = _prepare_design(features, y)
    return tuple(_fit_prepared(design, alpha) for alpha in values)


def fit_ridge(features: ArrayLike, y: ArrayLike, alpha: float) -> RidgeModel:
    """Fit one mean-error ridge objective with an unpenalized intercept."""
    return fit_ridge_grid(features, y, (alpha,))[0]


__all__ = ["RIDGE_VERSION", "RidgeModel", "fit_ridge", "fit_ridge_grid"]

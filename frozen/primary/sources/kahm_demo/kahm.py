"""Gaussian Kernel Affine Hull Machine reconstruction for one local model.

Based on our research. The local definition ``gaussian_kahm_encoder_v1`` in
``docs/ENCODER_DESIGN.md`` specifies the Gaussian kernel and signed reconstruction.
The reference energy uses globally standardized Q, without subtracting its local
mean. The fixed-point iteration stores the inverse and lambda from its last
completed solve.

Query kernels are rescaled by a common factor before exponentiation. Signed
regularized weights are retained, including valid negative normalization sums.
An unusable query quotient falls back to the local affine reconstruction. A
rank-zero model returns its mean and evaluates no Gaussian kernel.

This module specifies a floating-point computational map. Residual diagnostics
are numerical checks, not rigorous error enclosures for an ideal map. Projected
reference collisions are retained and recorded in the regularized Gaussian
construction; distinct projected references are not required or claimed.
No future responses or calibration observations are inputs to fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import LocalPCA, fit_local_pca


KAHM_VERSION = "gaussian-signed-regularized-float64-v1"
REGULARIZATION_TOLERANCE = 1e-3
MAX_REGULARIZATION_SOLVES = 100
SIGNED_RATIO_TOLERANCE = 1e-10
SOLVE_RESIDUAL_TOLERANCE = 1e-10


class KAHMFitError(ValueError):
    """A local Gaussian fit could not satisfy its declared numerical checks."""


class NormalizationError(ValueError):
    """A query's signed reconstruction quotient is not numerically usable."""

    def __init__(self, reason: str, diagnostics: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)


def _immutable_array(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _references(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError("Q must contain real integer or floating-point numbers")
    if raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError("Q must be a nonempty two-dimensional reference matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        rows = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(rows).all():
        raise ValueError("Q must contain only finite float64 values")
    return rows


def _query(value: Any, dimension: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError("query must contain real integer or floating-point numbers")
    if raw.ndim != 1 or raw.shape[0] != dimension:
        raise ValueError(f"query must have shape ({dimension},)")
    with np.errstate(over="ignore", invalid="ignore"):
        row = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(row).all():
        raise ValueError("query must contain only finite float64 values")
    return row


def _finite_number_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _signed_reconstruction(
    Q: np.ndarray, w: np.ndarray, ratio_tol: float
) -> tuple[np.ndarray, dict]:
    """Evaluate the same signed quotient after harmless common weight scaling."""
    if not np.isfinite(ratio_tol) or not 0 < ratio_tol <= 1:
        raise ValueError("ratio_tol must lie in (0, 1]")
    if w.ndim != 1 or w.shape[0] != Q.shape[0]:
        raise ValueError("weights must have one entry per reference row")
    if not np.isfinite(w).all():
        raise NormalizationError("nonfinite_weights")
    weight_scale = float(np.max(np.abs(w)))
    if weight_scale == 0:
        raise NormalizationError("zero_weight_l1", {"weight_scale": 0.0})

    # Common scaling preserves both the mathematical quotient and cancellation
    # ratio while avoiding overflow in the sums of otherwise finite weights.
    scaled = w / weight_scale
    scaled_sum = float(np.sum(scaled, dtype=np.float64))
    scaled_l1 = float(np.sum(np.abs(scaled), dtype=np.float64))
    ratio = abs(scaled_sum) / scaled_l1
    with np.errstate(over="ignore", invalid="ignore"):
        signed_sum = float(np.float64(scaled_sum) * weight_scale)
        weight_l1 = float(np.float64(scaled_l1) * weight_scale)
    stats = {
        "signed_sum": _finite_number_or_none(signed_sum),
        "weight_l1": _finite_number_or_none(weight_l1),
        "weight_scale": weight_scale,
        "scaled_signed_sum": scaled_sum,
        "scaled_weight_l1": scaled_l1,
        "cancellation_ratio": ratio,
        "cancellation_threshold": float(ratio_tol),
    }
    if not np.isfinite(ratio) or scaled_sum == 0 or ratio < ratio_tol:
        raise NormalizationError("signed_cancellation", stats)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = scaled / scaled_sum
        reconstruction = Q.T @ normalized
    if not np.isfinite(reconstruction).all():
        raise NormalizationError("nonfinite_reconstruction", stats)
    return _immutable_array(reconstruction), stats


def signed_normalized_reconstruction(
    Q: Any, w: Any, ratio_tol: float = SIGNED_RATIO_TOLERANCE
) -> np.ndarray:
    """Return Q.T @ w / sum(w), with a scale-invariant cancellation guard.

    Negative weights and a negative signed denominator are valid. The helper
    never clips weights. ``NormalizationError`` indicates a failed finite-value
    or cancellation check, whereas malformed shapes raise ``ValueError``.
    """
    references = _references(Q)
    raw = np.asarray(w)
    if raw.dtype.kind not in "iuf":
        raise TypeError("weights must contain real integer or floating-point numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.array(raw, dtype=np.float64, copy=True)
    reconstruction, _ = _signed_reconstruction(references, weights, ratio_tol)
    return reconstruction


@dataclass(frozen=True, slots=True, eq=False)
class ReconstructionResult:
    reconstruction: np.ndarray
    diagnostics: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction", _immutable_array(self.reconstruction))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True, slots=True, eq=False)
class LocalKAHM:
    pca: LocalPCA
    Q: np.ndarray
    gram: np.ndarray
    metric_transform: np.ndarray
    C: np.ndarray
    fitted_lambda: float | None
    iterations: int
    converged: bool
    diagnostics: dict

    def __post_init__(self) -> None:
        for name in ("Q", "gram", "metric_transform", "C"):
            values = _immutable_array(getattr(self, name))
            if not np.isfinite(values).all():
                raise ValueError(f"stored {name} contains nonfinite values")
            object.__setattr__(self, name, values)
        if self.Q.ndim != 2 or min(self.Q.shape) < 1:
            raise ValueError("stored Q must be a nonempty reference matrix")
        n, d = self.Q.shape
        if self.pca.reference_count != n or self.pca.mu.shape != (d,):
            raise ValueError("stored PCA does not match Q")
        if self.metric_transform.shape != (d, self.pca.rank):
            raise ValueError("stored metric transform has the wrong shape")
        expected_square = (n, n) if self.pca.rank else (0, 0)
        if self.gram.shape != expected_square or self.C.shape != expected_square:
            raise ValueError("stored Gram/inverse shapes do not match the model rank")
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if type(self.converged) is not bool:
            raise TypeError("converged must be a boolean")
        if self.pca.rank:
            if self.fitted_lambda is None or not np.isfinite(self.fitted_lambda) or self.fitted_lambda <= 0:
                raise ValueError("a positive-rank model requires a finite positive fitted_lambda")
        elif self.fitted_lambda is not None or self.iterations != 0:
            raise ValueError("a rank-zero model has no fitted lambda or inverse iterations")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def input_dimension(self) -> int:
        return int(self.Q.shape[1])

    def log_kernel_one(self, z: Any) -> np.ndarray:
        """Unrescaled query log kernels, using explicit differences in one row.

        The stored d-by-r metric transform includes the factor 1/sqrt(r), so
        log(k(z,q)) = -0.5 * ||(z-q) @ metric_transform||**2.
        """
        row = _query(z, self.input_dimension)
        if self.pca.rank == 0:
            raise ValueError("rank-zero models do not evaluate Gaussian kernels")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            differences = row[None, :] - self.Q
            transformed = differences @ self.metric_transform
            logs = -0.5 * np.sum(transformed * transformed, axis=1, dtype=np.float64)
        if not np.isfinite(logs).all():
            raise NormalizationError("nonfinite_log_kernel")
        return _immutable_array(logs)

    def affine_one(self, z: Any) -> np.ndarray:
        """The declared fallback; a full-rank affine model is exactly identity."""
        row = _query(z, self.input_dimension)
        if self.pca.rank == 0:
            result = self.pca.mu
        elif self.pca.rank == self.input_dimension:
            result = row
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                result = self.pca.mu + self.pca.P @ (self.pca.P.T @ (row - self.pca.mu))
        if not np.isfinite(result).all():
            raise ValueError("affine fallback produced nonfinite values")
        return _immutable_array(result)

    def reconstruct_one(self, z: Any) -> ReconstructionResult:
        """Reconstruct one query with frozen arrays and an explicit outcome."""
        row = _query(z, self.input_dimension)
        if self.pca.rank == 0:
            return ReconstructionResult(
                self.pca.mu,
                {"mode": "rank_zero_mean", "status": "rank_zero_mean", "rank": 0,
                 "used_fallback": False, "fallback_reason": None,
                 "kernel_evaluated": False, "cancellation_ratio": None},
            )
        query_stats: dict = {
            "rank": self.pca.rank, "kernel_evaluated": True, "cancellation_ratio": None,
        }
        try:
            logs = self.log_kernel_one(row)
            largest = float(np.max(logs))
            with np.errstate(under="ignore", over="ignore", invalid="ignore"):
                kernel = np.exp(logs - largest)
                unscaled_all_zero = bool(np.exp(largest) == 0)
                weights = self.C @ kernel
            query_stats.update({
                "maximum_log_kernel": largest,
                "unscaled_kernel_would_be_all_zero": unscaled_all_zero,
                "kernel_rescaling": "subtract query maximum log kernel",
            })
            if not np.isfinite(kernel).all():
                raise NormalizationError("nonfinite_rescaled_kernel")
            reconstruction, normalization_stats = _signed_reconstruction(
                self.Q, weights, SIGNED_RATIO_TOLERANCE
            )
            query_stats.update(normalization_stats)
            query_stats.update({
                "mode": "kahm", "status": "kahm", "used_fallback": False,
                "fallback_reason": None,
            })
            return ReconstructionResult(reconstruction, query_stats)
        except NormalizationError as error:
            query_stats.update(error.diagnostics)
            query_stats.update({
                "mode": "affine_fallback", "status": "affine_fallback", "used_fallback": True,
                "fallback_reason": error.reason,
            })
            return ReconstructionResult(self.affine_one(row), query_stats)


def fit_local_kahm(Q: Any, cap: int) -> LocalKAHM:
    """Fit a local Gaussian KAHM from reference histories, without responses.

    PCA uses equal weights on the supplied reference rows. The caller selects
    and records distinct raw references; this function does not silently drop
    or replace rows, including those that collide after projection.
    """
    references = _references(Q)
    pca = fit_local_pca(references, cap)
    n, d = references.shape
    metadata = {
        "version": KAHM_VERSION,
        "dtype": "float64",
        "reference_count": n,
        "input_dimension": d,
        "rank": pca.rank,
        "raw_distinct_count": int(np.unique(references, axis=0).shape[0]),
        "projected_unique_count": pca.diagnostics.get("projected_unique_count"),
        "projected_collision_count": pca.diagnostics.get("projected_collision_count"),
        "minimum_projected_separation": pca.diagnostics.get("minimum_projected_separation"),
        "projected_collision_policy": "retain references in the regularized Gaussian computational map",
        "projected_reference_distinctness_claimed": False,
        "regularization_tolerance": REGULARIZATION_TOLERANCE,
        "maximum_inverse_solves": MAX_REGULARIZATION_SOLVES,
        "solve_residual_tolerance": SOLVE_RESIDUAL_TOLERANCE,
        "signed_ratio_tolerance": SIGNED_RATIO_TOLERANCE,
        "fixedpoint_reference_energy": "sum(Q**2)/(reference_count*input_dimension), before local centering",
        "fit_uses_future_targets": False,
        "numerical_enclosure_claimed": False,
    }
    if pca.rank == 0:
        metadata.update({
            "regularization_status": "not_applicable_rank_zero",
            "iteration_records": [], "converged": False, "iterations": 0,
        })
        return LocalKAHM(
            pca, references, np.empty((0, 0)), np.empty((d, 0)),
            np.empty((0, 0)), None, 0, False, metadata,
        )

    try:
        cholesky = np.linalg.cholesky(pca.S)
        metric_transform = np.linalg.solve(cholesky, pca.P.T).T / np.sqrt(pca.rank)
    except np.linalg.LinAlgError as error:
        raise KAHMFitError("projected covariance Cholesky/solve failed") from error
    if not np.isfinite(cholesky).all() or not np.isfinite(metric_transform).all():
        raise KAHMFitError("projected covariance factorization produced nonfinite values")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        covariance_residual = float(np.linalg.norm(cholesky @ cholesky.T - pca.S, ord=np.inf))
        covariance_scale = max(float(np.linalg.norm(pca.S, ord=np.inf)), np.finfo(np.float64).tiny)
        covariance_relative_residual = covariance_residual / covariance_scale
        differences = references[:, None, :] - references[None, :, :]
        transformed = differences @ metric_transform
        gram_logs = -0.5 * np.sum(transformed * transformed, axis=2, dtype=np.float64)
        gram = np.exp(gram_logs)
        nu = float(np.sum(references * references, dtype=np.float64) / (n * d))
    if not np.isfinite(covariance_relative_residual) or covariance_relative_residual > SOLVE_RESIDUAL_TOLERANCE:
        raise KAHMFitError("projected covariance failed its Cholesky residual check")
    if not np.isfinite(gram_logs).all() or not np.isfinite(gram).all():
        raise KAHMFitError("reference Gaussian kernel produced nonfinite values")
    if not np.isfinite(nu) or nu <= 0:
        raise KAHMFitError("a positive-rank fit requires finite positive reference energy")

    current_lambda = 2.5 * nu
    if not np.isfinite(current_lambda) or current_lambda <= 0:
        raise KAHMFitError("initial regularizer is not finite and strictly positive")
    identity = np.eye(n, dtype=np.float64)
    records: list[dict] = []
    converged = False
    for iteration in range(1, MAX_REGULARIZATION_SOLVES + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = gram + current_lambda * identity
        if not np.isfinite(matrix).all():
            raise KAHMFitError("regularized Gram matrix contains nonfinite values")
        try:
            inverse = np.linalg.solve(matrix, identity)
        except np.linalg.LinAlgError as error:
            raise KAHMFitError("regularized Gaussian system solve failed") from error
        if not np.isfinite(inverse).all():
            raise KAHMFitError("regularized inverse contains nonfinite values")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = float(np.linalg.norm(matrix @ inverse - identity, ord=np.inf))
            residual_scale = float(np.linalg.norm(matrix, ord=np.inf)) * float(np.linalg.norm(inverse, ord=np.inf)) + 1.0
            relative_residual = residual / residual_scale
            reconstruction_error = references - gram @ (inverse @ references)
            next_lambda = 2.0 * nu + float(np.sum(reconstruction_error * reconstruction_error, dtype=np.float64) / (n * d))
        if not np.isfinite(residual) or not np.isfinite(residual_scale) or not np.isfinite(relative_residual):
            raise KAHMFitError("regularized inverse residual is nonfinite")
        if relative_residual > SOLVE_RESIDUAL_TOLERANCE:
            raise KAHMFitError("regularized inverse failed its backward residual check")
        if not np.isfinite(next_lambda) or next_lambda <= 0:
            raise KAHMFitError("proposed regularizer is not finite and strictly positive")
        change = abs(next_lambda - current_lambda)
        records.append({
            "solve_index": iteration,
            "lambda": float(current_lambda),
            "proposed_lambda": float(next_lambda),
            "absolute_change": float(change),
            "inverse_residual_inf": residual,
            "scaled_inverse_residual_inf": relative_residual,
        })
        if change <= REGULARIZATION_TOLERANCE:
            converged = True
            break
        if iteration < MAX_REGULARIZATION_SOLVES:
            current_lambda = next_lambda

    metadata.update({
        "reference_energy": nu,
        "covariance_cholesky_residual_inf": covariance_residual,
        "covariance_cholesky_relative_residual": covariance_relative_residual,
        "metric_transform_convention": "log kernel = -0.5*norm((z-q)@metric_transform)**2; includes 1/sqrt(rank)",
        "gram_rescaling": "none",
        "regularization_status": "converged" if converged else "iteration_limit_retained_last_solve",
        "iteration_records": records,
        "iterations": len(records),
        "converged": converged,
        "fitted_lambda": float(current_lambda),
        "unused_next_lambda_proposal": records[-1]["proposed_lambda"],
    })
    return LocalKAHM(
        pca, references, gram, metric_transform, inverse,
        float(current_lambda), len(records), converged, metadata,
    )


__all__ = [
    "KAHM_VERSION", "LocalKAHM", "ReconstructionResult", "KAHMFitError",
    "NormalizationError", "fit_local_kahm", "signed_normalized_reconstruction",
]

"""Matched linear-plus-Gaussian-Nystrom features fitted on training rows.

Based on our research. The local definition ``matched_forecasting_readout_v1``
in ``docs/READOUT_DESIGN.md`` standardizes every input column with training
moments (ddof=0), retains every linear column, forms a truncated spectral
Nystrom block from caller-supplied landmark identities, standardizes that new
block on all training occurrences, and appends it to the
unchanged standardized linear block. The workflow, not this module, chooses
the same distinct RAW-history landmark row identities for every representation.
Encoded landmarks may coincide and their kernel rank may differ.

Inference is one row at a time, including in batch methods. Identical rows may
reuse the same calculated row; no observation is removed from training moments.
Moments are calculated on contiguous individual columns so an unchanged MEM
prefix has identical fitted moments when LOCAL columns are appended.

All arrays use float64 and immutable byte-backed storage. Eigenvector signs and
exactly equal numerical eigenspaces use the geometry module's canonical rule.
The saved fitted state specifies inference; bitwise agreement across different
numerical runtimes is not promised. This module accepts no targets and performs
no calibration, oracle-risk assessment, or validation-based selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import _canonical_eigenvectors


READOUT_FEATURES_VERSION = "matched-nystrom-linear-float64-v1"
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
STANDARD_DEVIATION_FLOOR = 1e-8
MIN_LANDMARK_DISTANCE_SQUARED = 1e-14
MIN_SIGMA_SQUARED = 1e-12
ABSOLUTE_EIGENVALUE_CUTOFF = 1e-12
RELATIVE_EIGENVALUE_CUTOFF = 1e-10


def _immutable_array(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _matrix(value: Any, name: str, *, width: int | None = None,
            allow_empty: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating-point numbers")
    if raw.ndim != 2 or raw.shape[1] == 0 or (not allow_empty and raw.shape[0] == 0):
        raise ValueError(f"{name} must be a two-dimensional matrix with nonempty feature width")
    if width is not None and raw.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite float64 values")
    return result


def _row(value: Any, width: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError("query must contain real integer or floating-point numbers")
    if raw.ndim != 1 or raw.shape[0] != width:
        raise ValueError(f"query must have shape ({width},)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("query must contain only finite float64 values")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _landmark_indices(value: Any, training_count: int) -> tuple[int, ...]:
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError("landmark_indices must be one-dimensional")
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError("landmark_indices must be a sequence of training row indices")
    result = []
    for index in value:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("landmark row indices must be integers, not booleans")
        index = int(index)
        if not 0 <= index < training_count:
            raise ValueError("landmark row index is outside the training matrix")
        result.append(index)
    if not result or len(set(result)) != len(result):
        raise ValueError("landmark row indices must be nonempty and distinct")
    return tuple(result)


def _bandwidth_multiplier(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("bandwidth_multiplier must be a real number")
    result = float(value)
    if not np.isfinite(result) or result not in BANDWIDTH_MULTIPLIERS:
        raise ValueError(f"bandwidth_multiplier must be one of {BANDWIDTH_MULTIPLIERS}")
    return result


def _canonical_rows(rows: np.ndarray, width: int, callback) -> np.ndarray:
    """Memoize computation only; preserve every occurrence and original order."""
    if len(rows) == 0:
        return _immutable_array(np.empty((0, width)))
    cache: dict[bytes, np.ndarray] = {}
    output = np.empty((len(rows), width), dtype=np.float64)
    for index, row in enumerate(rows):
        key = row.tobytes(order="C")
        if key not in cache:
            calculated = np.asarray(callback(row), dtype=np.float64)
            if calculated.shape != (width,) or not np.isfinite(calculated).all():
                raise ValueError("canonical feature calculation produced an invalid row")
            cache[key] = calculated
        output[index] = cache[key]
    return _immutable_array(output)


@dataclass(frozen=True, slots=True, eq=False)
class ColumnScaler:
    mean: np.ndarray
    scale: np.ndarray
    raw_std: np.ndarray
    sample_count: int
    replaced_columns: tuple[int, ...]

    def __post_init__(self) -> None:
        count = _positive_integer(self.sample_count, "sample_count")
        for name in ("mean", "scale", "raw_std"):
            array = _immutable_array(getattr(self, name))
            if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"stored scaler {name} must be a finite nonempty vector")
            object.__setattr__(self, name, array)
        if self.scale.shape != self.mean.shape or self.raw_std.shape != self.mean.shape:
            raise ValueError("stored scaler vectors have inconsistent widths")
        if np.any(self.raw_std < 0) or np.any(self.scale <= 0):
            raise ValueError("stored standard deviations/scales are invalid")
        replaced = tuple(int(index) for index in self.replaced_columns)
        expected = tuple(int(index) for index in np.flatnonzero(self.raw_std < STANDARD_DEVIATION_FLOOR))
        if replaced != expected:
            raise ValueError("stored replacement columns do not match the strict standard-deviation rule")
        if not np.array_equal(self.scale, np.where(self.raw_std < STANDARD_DEVIATION_FLOOR, 1.0, self.raw_std)):
            raise ValueError("stored scales do not match the declared replacement rule")
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "replaced_columns", replaced)

    @property
    def input_width(self) -> int:
        return int(self.mean.size)

    @property
    def diagnostics(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "feature_width": self.input_width,
            "variance_divisor": self.sample_count,
            "standard_deviation_floor": STANDARD_DEVIATION_FLOOR,
            "replacement_rule": "std < 1e-8 becomes 1; all columns retained",
            "replaced_columns": list(self.replaced_columns),
            "moment_algorithm": "one contiguous column at a time, including all occurrences",
        }

    def transform_one(self, x: Any) -> np.ndarray:
        row = _row(x, self.input_width)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = (row - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("column standardization produced nonfinite values")
        return _immutable_array(result)

    def transform(self, X: Any) -> np.ndarray:
        rows = _matrix(X, "X", width=self.input_width, allow_empty=True)
        return _canonical_rows(rows, self.input_width, self.transform_one)


def fit_column_scaler(X: Any) -> ColumnScaler:
    """Fit ddof=0 moments on every occurrence, independently by column."""
    rows = _matrix(X, "X")
    n, p = rows.shape
    mean = np.empty(p, dtype=np.float64)
    std = np.empty(p, dtype=np.float64)
    for column in range(p):
        values = np.array(rows[:, column], dtype=np.float64, order="C", copy=True)
        if np.all(values == values[0]):
            mean[column], std[column] = values[0], 0.0
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                mean[column] = np.mean(values, dtype=np.float64)
                std[column] = np.std(values, ddof=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("training column moments are nonfinite")
    replaced = std < STANDARD_DEVIATION_FLOOR
    return ColumnScaler(
        mean, np.where(replaced, 1.0, std), std, n,
        tuple(int(index) for index in np.flatnonzero(replaced)),
    )


def _raw_nystrom_one(
    u: np.ndarray, landmarks: np.ndarray, sigma_squared: float,
    eigenvalues: np.ndarray, eigenvectors: np.ndarray,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        differences = landmarks - u[None, :]
        squared_distances = np.sum(differences * differences, axis=1, dtype=np.float64)
        kernel_values = np.exp(-squared_distances / (2.0 * sigma_squared))
        result = (eigenvectors.T @ kernel_values) / np.sqrt(eigenvalues)
    if not np.isfinite(kernel_values).all() or not np.isfinite(result).all():
        raise ValueError("Nystrom query features are nonfinite")
    return _immutable_array(result)


@dataclass(frozen=True, slots=True, eq=False)
class ReadoutFeatures:
    linear_scaler: ColumnScaler
    landmark_indices: tuple[int, ...]
    landmarks: np.ndarray
    bandwidth_multiplier: float
    median_squared_distance: float
    sigma_squared: float
    landmark_gram: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nystrom_scaler: ColumnScaler
    training_count: int
    diagnostics: dict

    def __post_init__(self) -> None:
        if not isinstance(self.linear_scaler, ColumnScaler) or not isinstance(self.nystrom_scaler, ColumnScaler):
            raise TypeError("readout scalers must be ColumnScaler states")
        count = _positive_integer(self.training_count, "training_count")
        indices = _landmark_indices(self.landmark_indices, count)
        multiplier = _bandwidth_multiplier(self.bandwidth_multiplier)
        for name in ("landmarks", "landmark_gram", "eigenvalues", "eigenvectors"):
            array = _immutable_array(getattr(self, name))
            if not np.isfinite(array).all():
                raise ValueError(f"stored {name} must be finite")
            object.__setattr__(self, name, array)
        m, p = len(indices), self.linear_scaler.input_width
        if self.landmarks.shape != (m, p) or self.landmark_gram.shape != (m, m):
            raise ValueError("stored landmarks/Gram have inconsistent shapes")
        if self.eigenvalues.ndim != 1 or self.eigenvalues.size == 0 or np.any(self.eigenvalues <= 0):
            raise ValueError("retained eigenvalues must be a nonempty positive vector")
        r = int(self.eigenvalues.size)
        if self.eigenvectors.shape != (m, r) or self.nystrom_scaler.input_width != r:
            raise ValueError("stored Nystrom states have inconsistent ranks")
        if self.linear_scaler.sample_count != count or self.nystrom_scaler.sample_count != count:
            raise ValueError("both feature scalers must use every training occurrence")
        delta, sigma = float(self.median_squared_distance), float(self.sigma_squared)
        if not np.isfinite(delta) or delta <= 0 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("distance median and sigma squared must be finite and positive")
        if sigma != max(delta * multiplier ** 2, MIN_SIGMA_SQUARED):
            raise ValueError("stored sigma squared does not follow the bandwidth rule")
        object.__setattr__(self, "landmark_indices", indices)
        object.__setattr__(self, "bandwidth_multiplier", multiplier)
        object.__setattr__(self, "median_squared_distance", delta)
        object.__setattr__(self, "sigma_squared", sigma)
        object.__setattr__(self, "training_count", count)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def input_width(self) -> int:
        return self.linear_scaler.input_width

    @property
    def nystrom_rank(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def landmark_count(self) -> int:
        return len(self.landmark_indices)

    @property
    def output_width(self) -> int:
        return self.input_width + self.nystrom_rank

    def transform_one(self, x: Any) -> np.ndarray:
        u = self.linear_scaler.transform_one(x)
        raw_nystrom = _raw_nystrom_one(
            u, self.landmarks, self.sigma_squared, self.eigenvalues, self.eigenvectors
        )
        standardized_nystrom = self.nystrom_scaler.transform_one(raw_nystrom)
        return _immutable_array(np.concatenate((u, standardized_nystrom)))

    def transform(self, X: Any) -> np.ndarray:
        rows = _matrix(X, "X", width=self.input_width, allow_empty=True)
        return _canonical_rows(rows, self.output_width, self.transform_one)


def fit_readout_features(
    X: Any, landmark_indices: Any, bandwidth_multiplier: float
) -> ReadoutFeatures:
    """Fit the matched feature procedure using training features only.

    Landmark identities are chosen once from distinct RAW training histories by
    the caller. The same indices must be passed for every compared encoding;
    coinciding encoded landmark values are retained rather than deduplicated.
    """
    rows = _matrix(X, "X")
    n, p = rows.shape
    indices = _landmark_indices(landmark_indices, n)
    multiplier = _bandwidth_multiplier(bandwidth_multiplier)
    linear_scaler = fit_column_scaler(rows)
    standardized = linear_scaler.transform(rows)
    # Canonicalize the numerical operands before fitting later moments. A
    # mathematically identical Fortran-order matrix can select a different
    # matrix-vector summation path from the saved C-order inference state.
    landmarks = _immutable_array(standardized[list(indices)])
    m = len(indices)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        differences = landmarks[:, None, :] - landmarks[None, :, :]
        squared_distances = np.sum(differences * differences, axis=2, dtype=np.float64)
    if not np.isfinite(squared_distances).all():
        raise ValueError("landmark squared distances are nonfinite")
    pairs = squared_distances[np.triu_indices(m, k=1)]
    positive_pairs = pairs[pairs > MIN_LANDMARK_DISTANCE_SQUARED]
    delta = float(np.median(positive_pairs)) if positive_pairs.size else 1.0
    sigma_squared = max(delta * multiplier ** 2, MIN_SIGMA_SQUARED)
    if not np.isfinite(sigma_squared) or sigma_squared <= 0:
        raise ValueError("fitted Gaussian bandwidth is not finite and positive")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        gram = np.exp(-squared_distances / (2.0 * sigma_squared))
        gram = (gram + gram.T) / 2.0
    if not np.isfinite(gram).all():
        raise ValueError("landmark Gram matrix is nonfinite")
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError("landmark eigendecomposition is nonfinite")
    eigenvectors, repeated = _canonical_eigenvectors(eigenvalues, eigenvectors)
    largest = float(eigenvalues[0])
    cutoff = max(ABSOLUTE_EIGENVALUE_CUTOFF, RELATIVE_EIGENVALUE_CUTOFF * largest)
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise ValueError("landmark Gram matrix has no retained positive eigenvalue")
    retained_values = _immutable_array(eigenvalues[keep])
    retained_vectors = _immutable_array(eigenvectors[:, keep])
    rank = int(retained_values.size)
    raw_nystrom = _canonical_rows(
        standardized, rank,
        lambda u: _raw_nystrom_one(u, landmarks, sigma_squared, retained_values, retained_vectors),
    )
    nystrom_scaler = fit_column_scaler(raw_nystrom)
    with np.errstate(over="ignore", invalid="ignore"):
        eigen_residual = float(np.linalg.norm(
            gram @ retained_vectors - retained_vectors * retained_values[None, :], ord=np.inf
        ))
        orthogonality_residual = float(np.linalg.norm(
            retained_vectors.T @ retained_vectors - np.eye(rank), ord=np.inf
        ))
    if not np.isfinite(eigen_residual) or not np.isfinite(orthogonality_residual):
        raise ValueError("landmark eigendecomposition residual is nonfinite")
    diagnostics = {
        "version": READOUT_FEATURES_VERSION,
        "dtype": "float64",
        "training_count": n,
        "input_width": p,
        "output_width": p + rank,
        "landmark_count": m,
        "landmark_indices": list(indices),
        "encoded_landmark_distinct_count": int(np.unique(landmarks, axis=0).shape[0]),
        "landmark_identity_source": "caller-supplied common RAW-history training row indices",
        "encoded_landmark_duplicates_retained": True,
        "bandwidth_multiplier": multiplier,
        "positive_distance_threshold": MIN_LANDMARK_DISTANCE_SQUARED,
        "distance_rule": "median over i<j squared distances strictly greater than 1e-14",
        "positive_distance_pair_count": int(positive_pairs.size),
        "median_distance_fallback_used": bool(positive_pairs.size == 0),
        "median_squared_distance": delta,
        "sigma_squared": sigma_squared,
        "minimum_sigma_squared": MIN_SIGMA_SQUARED,
        "full_gram_eigenvalues": eigenvalues.tolist(),
        "eigenvalue_cutoff": cutoff,
        "eigenvalue_rule": "lambda > max(1e-12, 1e-10 * largest eigenvalue)",
        "nystrom_rank": rank,
        "eigenpair_residual_inf": eigen_residual,
        "orthogonality_residual_inf": orthogonality_residual,
        "repeated_eigenvalue_blocks": [list(block) for block in repeated],
        "eigenvector_convention": "canonical exact-equality eigenspaces and largest-absolute component positive",
        "linear_scaler": linear_scaler.diagnostics,
        "nystrom_scaler": nystrom_scaler.diagnostics,
        "training_multiplicity": "all supplied rows contribute to both moment calculations",
        "output_order": "unchanged standardized linear block, then standardized Nystrom block",
        "canonical_inference": "single-row transform; batches memoize identical rows without dropping observations",
        "fit_uses_future_targets": False,
        "numerical_enclosure_claimed": False,
    }
    return ReadoutFeatures(
        linear_scaler, indices, landmarks, multiplier, delta, sigma_squared,
        gram, retained_values, retained_vectors, nystrom_scaler, n, diagnostics,
    )


__all__ = [
    "READOUT_FEATURES_VERSION", "BANDWIDTH_MULTIPLIERS", "ColumnScaler",
    "ReadoutFeatures", "fit_column_scaler", "fit_readout_features",
]

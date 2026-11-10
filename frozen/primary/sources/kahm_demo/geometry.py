"""Training-only preprocessing and reference geometry for the demonstrator.

Based on our research. The local definition ``gaussian_kahm_encoder_v1`` in
``docs/ENCODER_DESIGN.md`` specifies preprocessing on a small history support
with repeated observations. Standardization uses every training occurrence.
Clustering compresses identical rows *with their counts*, while each local
reference set contains distinct histories. PCA weights those references equally.

There is no random initialization or reference subsampling: lexicographic
farthest-first choices, weighted Lloyd updates, and first-index ties specify the
procedure. Geometry is fitted once, independently of a later PCA cap. All
numerical arrays and stored references use float64. No future targets or
calibration data are accepted.

PCA signs and exactly equal returned eigenvalue blocks have canonical choices.
Nearly equal eigenvalues are not merged. Floating-point eigendecompositions and
clustering are deterministic within the recorded numerical runtime; bitwise
agreement across different NumPy/BLAS implementations is not promised. Save the
fitted arrays for canonical inference rather than silently refitting them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STANDARD_DEVIATION_FLOOR = 1e-12
PROJECTED_RANGE_FLOOR = 1e-3
GEOMETRY_VERSION = "weighted-distinct-farthest-first-float64-v1"


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not a boolean or other type")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _matrix(value: Any, name: str, *, width: int | None = None,
            allow_empty: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating-point numbers")
    if raw.ndim != 2 or raw.shape[1] == 0 or (not allow_empty and raw.shape[0] == 0):
        raise ValueError(f"{name} must be a two-dimensional nonempty-width matrix")
    if width is not None and raw.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite float64 values")
    return result


def _immutable_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """Copy into an immutable bytes buffer, including against setflags(write=True)."""
    array = np.array(value, dtype=dtype, order="C", copy=True)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _finite(array: np.ndarray, message: str) -> np.ndarray:
    if not np.isfinite(array).all():
        raise ValueError(message)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    sample_count: int
    replaced_columns: tuple[int, ...]
    raw_std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "scale", "raw_std"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        object.__setattr__(self, "replaced_columns", tuple(self.replaced_columns))

    @property
    def input_dimension(self) -> int:
        return int(self.mean.size)

    @property
    def diagnostics(self) -> dict:
        return {
            "version": GEOMETRY_VERSION,
            "sample_count": self.sample_count,
            "input_dimension": self.input_dimension,
            "variance_divisor": self.sample_count,
            "standard_deviation_floor": STANDARD_DEVIATION_FLOOR,
            "replacement_rule": "std <= floor becomes 1",
            "replacement_count": len(self.replaced_columns),
            "replaced_columns": list(self.replaced_columns),
            "raw_std": self.raw_std.tolist(),
            "fitting_rows": "all training occurrences, including duplicates",
            "dtype": "float64",
        }

    def transform(self, X: Any) -> np.ndarray:
        rows = _matrix(X, "X", width=self.input_dimension, allow_empty=True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = (rows - self.mean) / self.scale
        return _finite(values, "standardization produced non-finite values")

    def transform_one(self, x: Any) -> np.ndarray:
        row = np.asarray(x)
        if row.ndim != 1 or row.shape[0] != self.input_dimension:
            raise ValueError(f"x must have shape ({self.input_dimension},)")
        return self.transform(row.reshape(1, -1))[0]


def fit_standardizer(X: Any) -> Standardizer:
    """Fit population moments (ddof=0) using every supplied training row."""
    rows = _matrix(X, "X")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(rows, axis=0, dtype=np.float64)
        std = np.std(rows, axis=0, ddof=0, dtype=np.float64)
    _finite(mean, "non-finite training mean")
    _finite(std, "non-finite training standard deviation")
    replace = std <= STANDARD_DEVIATION_FLOOR
    return Standardizer(mean, np.where(replace, 1.0, std), int(rows.shape[0]),
                        tuple(int(i) for i in np.flatnonzero(replace)), std)


@dataclass(frozen=True, slots=True)
class _PartitionAttempt:
    group_count: int
    initial_reference_indices: tuple[int, ...]
    iterations: int
    status: str
    empty_groups: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_reference_indices", tuple(self.initial_reference_indices))
        object.__setattr__(self, "empty_groups", tuple(self.empty_groups))

    @property
    def diagnostics(self) -> dict:
        return {
            "group_count": self.group_count,
            "initial_reference_indices": list(self.initial_reference_indices),
            "iterations": self.iterations,
            "status": self.status,
            "empty_groups": list(self.empty_groups),
        }


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceGroup:
    partition_id: int
    group_id: int
    Q: np.ndarray
    reference_indices: np.ndarray
    reference_multiplicities: np.ndarray
    member_indices: np.ndarray
    member_multiplicities: np.ndarray
    centroid: np.ndarray

    def __post_init__(self) -> None:
        for name in ("Q", "centroid"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        for name in ("reference_indices", "reference_multiplicities",
                     "member_indices", "member_multiplicities"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), np.int64))

    @property
    def diagnostics(self) -> dict:
        return {
            "partition_id": self.partition_id,
            "group_id": self.group_id,
            "training_occurrence_count": int(self.member_multiplicities.sum()),
            "distinct_member_count": int(self.member_indices.size),
            "reference_count": int(self.Q.shape[0]),
            "reference_indices": self.reference_indices.tolist(),
            "reference_multiplicities": self.reference_multiplicities.tolist(),
            "member_indices": self.member_indices.tolist(),
            "member_multiplicities": self.member_multiplicities.tolist(),
            "centroid": self.centroid.tolist(),
            "row_identity": "first zero-based row index in the supplied training matrix",
            "reference_weighting_for_pca": "one equal weight per selected distinct reference",
        }


@dataclass(frozen=True, slots=True, eq=False)
class ReferencePartition:
    partition_id: int
    requested_groups: int
    training_count: int
    distinct_count: int
    reference_budget: int
    max_iter: int
    groups: tuple[ReferenceGroup, ...]
    attempts: tuple[_PartitionAttempt, ...]
    inertia: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def effective_groups(self) -> int:
        return len(self.groups)

    @property
    def diagnostics(self) -> dict:
        return {
            "version": GEOMETRY_VERSION,
            "partition_id": self.partition_id,
            "requested_groups": self.requested_groups,
            "initial_groups": min(self.requested_groups, self.distinct_count),
            "effective_groups": self.effective_groups,
            "unique_shortage_reduction": self.requested_groups > self.distinct_count,
            "training_count": self.training_count,
            "distinct_count": self.distinct_count,
            "reference_budget_per_group": self.reference_budget,
            "max_lloyd_iterations": self.max_iter,
            "attempts": [attempt.diagnostics for attempt in self.attempts],
            "weighted_inertia": self.inertia,
            "initialization": "first lexicographic row, then farthest from selected centers",
            "distance_ties": "first lexicographic distinct row; assignment ties use first center",
            "cluster_weighting": "all training occurrence counts",
            "max_iteration_behavior": "retain last assignments and their weighted means; mark nonconvergence",
            "reference_order": "lexicographic when all fit; farthest-first when budget binds",
            "dtype": "float64",
            "groups": [group.diagnostics for group in self.groups],
        }


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        delta = X[:, None, :] - centers[None, :, :]
        result = np.sum(delta * delta, axis=2, dtype=np.float64)
    return _finite(result, "non-finite squared distance during reference fitting")


def _farthest_first(points: np.ndarray, count: int) -> tuple[int, ...]:
    """Points are lexicographically ordered; ties retain the first row."""
    chosen = [0]
    nearest = _squared_distances(points, points[[0]])[:, 0]
    while len(chosen) < count:
        available = nearest.copy()
        available[chosen] = -np.inf
        nxt = int(np.argmax(available))
        chosen.append(nxt)
        nearest = np.minimum(nearest, _squared_distances(points, points[[nxt]])[:, 0])
    return tuple(chosen)


def _weighted_lloyd(points: np.ndarray, counts: np.ndarray, group_count: int,
                    max_iter: int) -> tuple[np.ndarray, np.ndarray, int, str, tuple[int, ...], tuple[int, ...]]:
    initial = _farthest_first(points, group_count)
    centers = points[list(initial)].copy()
    previous_labels = None
    for iteration in range(1, max_iter + 1):
        labels = np.argmin(_squared_distances(points, centers), axis=1)
        empty = tuple(int(i) for i in np.flatnonzero(np.bincount(labels, minlength=group_count) == 0))
        if empty:
            return labels, centers, iteration, "empty_group_retry", empty, initial
        updated = np.empty_like(centers)
        with np.errstate(over="ignore", invalid="ignore"):
            for group in range(group_count):
                members = labels == group
                updated[group] = np.sum(points[members] * counts[members, None], axis=0) / int(counts[members].sum())
        _finite(updated, "non-finite weighted cluster mean")
        unchanged = np.array_equal(updated, centers)
        stable_labels = previous_labels is not None and np.array_equal(labels, previous_labels)
        centers = updated
        if unchanged or stable_labels:
            return labels, centers, iteration, "converged", (), initial
        previous_labels = labels
    return labels, centers, max_iter, "max_iter_reached", (), initial


def fit_reference_partitions(Z: Any, requested_groups: tuple[int, ...] = (2, 4),
                             reference_budget: int = 24, max_iter: int = 150) -> tuple[ReferencePartition, ...]:
    """Fit deterministic partitions once; the result does not depend on a PCA cap.

    Z must already be transformed by the training-fitted standardizer. Unique
    rows are lexicographically ordered; their occurrence counts weight every
    Lloyd mean. Group reference identities denote the first original row for a
    distinct history. Multiplicities are recorded but do not duplicate Q rows.
    An empty cluster halves the attempted group count and restarts from scratch.
    """
    rows = _matrix(Z, "Z")
    _positive_integer(reference_budget, "reference_budget")
    _positive_integer(max_iter, "max_iter")
    requests = tuple(requested_groups)
    if not requests:
        raise ValueError("request at least one reference partition")
    for count in requests:
        _positive_integer(count, "requested group count")
    points, first, counts = np.unique(rows, axis=0, return_index=True, return_counts=True)
    partitions = []
    for partition_id, requested in enumerate(requests):
        count = min(requested, len(points))
        attempts = []
        while True:
            labels, centers, iterations, status, empty, initial = _weighted_lloyd(points, counts, count, max_iter)
            attempts.append(_PartitionAttempt(count, tuple(int(first[i]) for i in initial),
                                              iterations, status, empty))
            if not empty:
                break
            if count == 1:
                raise ValueError("a one-group reference partition unexpectedly has no members")
            count = max(1, count // 2)
        groups = []
        for group_id in range(count):
            members = np.flatnonzero(labels == group_id)
            local = points[members]
            if len(members) <= reference_budget:
                selected = members
            else:
                selected = members[list(_farthest_first(local, reference_budget))]
            groups.append(ReferenceGroup(
                partition_id, group_id, points[selected], first[selected], counts[selected],
                first[members], counts[members], centers[group_id],
            ))
        distance = _squared_distances(points, centers)
        with np.errstate(over="ignore", invalid="ignore"):
            inertia = float(np.sum(counts * distance[np.arange(len(points)), labels]))
        if not np.isfinite(inertia):
            raise ValueError("non-finite weighted clustering objective")
        partitions.append(ReferencePartition(
            partition_id, requested, int(len(rows)), int(len(points)), reference_budget,
            max_iter, tuple(groups), tuple(attempts), inertia,
        ))
    return tuple(partitions)


@dataclass(frozen=True, slots=True)
class _PCAAttempt:
    requested_rank: int
    positive_rank: int
    minimum_projected_range: float | None
    range_violation: bool

    @property
    def diagnostics(self) -> dict:
        return {
            "requested_rank": self.requested_rank,
            "positive_rank": self.positive_rank,
            "minimum_projected_range": self.minimum_projected_range,
            "range_violation": self.range_violation,
        }


@dataclass(frozen=True, slots=True, eq=False)
class LocalPCA:
    mu: np.ndarray
    P: np.ndarray
    S: np.ndarray
    rank: int
    cap: int
    reference_count: int
    eigenvalues: np.ndarray
    projected_ranges: np.ndarray
    initial_rank_request: int
    final_rank_request: int
    eigenvalue_threshold: float
    attempts: tuple[_PCAAttempt, ...]
    reason: str
    repeated_eigenvalue_blocks: tuple[tuple[int, int], ...]
    projected_unique_count: int
    minimum_projected_separation: float | None

    def __post_init__(self) -> None:
        for name in ("mu", "P", "S", "eigenvalues", "projected_ranges"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "repeated_eigenvalue_blocks",
                           tuple(tuple(block) for block in self.repeated_eigenvalue_blocks))

    @property
    def diagnostics(self) -> dict:
        return {
            "version": GEOMETRY_VERSION,
            "dtype": "float64",
            "reference_storage": "float64",
            "cap": self.cap,
            "reference_count": self.reference_count,
            "input_dimension": int(self.mu.size),
            "covariance_divisor": max(0, self.reference_count - 1),
            "initial_rank_request": self.initial_rank_request,
            "final_rank_request": self.final_rank_request,
            "effective_rank": self.rank,
            "eigenvalues": self.eigenvalues.tolist(),
            "positive_eigenvalue_threshold": self.eigenvalue_threshold,
            "eigenvalue_rule": "strictly greater than max(float64 epsilon, epsilon * largest eigenvalue)",
            "threshold_scaling": "maximum of absolute float64 epsilon and epsilon times largest eigenvalue",
            "projected_range_floor": PROJECTED_RANGE_FLOOR,
            "projected_ranges": self.projected_ranges.tolist(),
            "remaining_range_violation": bool(self.rank and np.min(self.projected_ranges) < PROJECTED_RANGE_FLOOR),
            "rank_attempts": [attempt.diagnostics for attempt in self.attempts],
            "reason": self.reason,
            "repeated_eigenvalue_blocks": [list(block) for block in self.repeated_eigenvalue_blocks],
            "eigenvector_convention": "coordinate-projection basis for exactly equal returned eigenvalues; largest-absolute component positive",
            "numerical_reproducibility": "recorded runtime; no cross-BLAS bitwise guarantee",
            "projected_unique_count": self.projected_unique_count,
            "projected_collision_count": self.reference_count - self.projected_unique_count,
            "projected_collision_definition": "exact equality of stored float64 projected rows",
            "minimum_projected_separation": self.minimum_projected_separation,
            "projected_collisions_remove_references": False,
            "projected_covariance_cholesky_checked": bool(self.rank),
        }


def _canonical_eigenvectors(values: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Canonicalize exactly equal numerical eigenspaces without merging near ties."""
    result = vectors.copy()
    width = result.shape[0]
    blocks = []
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[end] == values[begin]:
            end += 1
        if end - begin > 1:
            blocks.append((begin, end))
            subspace = result[:, begin:end]
            projector = subspace @ subspace.T
            basis = []
            cutoff = 64 * np.finfo(np.float64).eps * max(1, width)
            for coordinate in range(width):
                candidate = projector[:, coordinate].copy()
                # Reorthogonalize: this also controls roundoff when a coordinate
                # projection is nearly in the span of previously selected axes.
                for _ in range(2):
                    for accepted in basis:
                        candidate -= accepted * np.dot(accepted, candidate)
                norm = float(np.linalg.norm(candidate))
                if norm > cutoff:
                    basis.append(candidate / norm)
                if len(basis) == end - begin:
                    break
            if len(basis) != end - begin:
                raise ValueError("failed to construct the canonical repeated-eigenvalue basis")
            result[:, begin:end] = np.column_stack(basis)
        begin = end
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result, tuple(blocks)


def fit_local_pca(Q: Any, cap: int) -> LocalPCA:
    """Fit a local mean, orthonormal basis P (d by r), and covariance S (r by r).

    Covariance uses divisor n_refs-1. Only eigenvalues strictly above
    max(eps, eps*lambda_max) survive; the relative component is an explicit
    demonstrator safeguard. Rank requests decrease if a retained projected
    coordinate has range below 1e-3, stopping at request one and recording any
    remaining violation. Rank-zero models have P.shape=(d,0) and
    S.shape=(0,0). Positive-rank S must pass Cholesky without jitter or repair.
    """
    references = _matrix(Q, "Q")
    _positive_integer(cap, "cap")
    n, dimension = references.shape
    initial_request = min(cap, dimension, n - 1)
    constant = bool(np.all(references == references[0]))
    with np.errstate(over="ignore", invalid="ignore"):
        mu = references[0].copy() if constant else references.mean(axis=0)
        centered = references - mu
    _finite(centered, "non-finite centered reference coordinates")
    epsilon = float(np.finfo(np.float64).eps)
    eigenvalues = np.zeros(dimension, dtype=np.float64)
    vectors = np.eye(dimension, dtype=np.float64)
    repeated = ()
    if n > 1 and not constant:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            covariance = centered.T @ centered / (n - 1)
            covariance = (covariance + covariance.T) / 2
        _finite(covariance, "non-finite reference covariance")
        eigenvalues, vectors = np.linalg.eigh(covariance)
        order = np.argsort(-eigenvalues, kind="stable")
        eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        _finite(eigenvalues, "non-finite reference covariance eigenvalues")
        _finite(vectors, "non-finite reference covariance eigenvectors")
        vectors, repeated = _canonical_eigenvectors(eigenvalues, vectors)
    threshold = max(epsilon, epsilon * max(0.0, float(eigenvalues[0])))
    request = initial_request
    attempts = []
    while True:
        retained = np.flatnonzero(eigenvalues[:request] > threshold)
        P = vectors[:, retained]
        projected = centered @ P
        ranges = np.ptp(projected, axis=0)
        violation = bool(len(retained) and np.min(ranges) < PROJECTED_RANGE_FLOOR)
        attempts.append(_PCAAttempt(request, int(len(retained)),
                                    float(np.min(ranges)) if len(retained) else None,
                                    violation))
        if not violation or request <= 1:
            break
        request -= 1
    rank = P.shape[1]
    if rank:
        with np.errstate(over="ignore", invalid="ignore"):
            S = projected.T @ projected / (n - 1)
            S = (S + S.T) / 2
        _finite(S, "non-finite projected reference covariance")
        # Failure is visible; silently adding jitter would change the metric.
        np.linalg.cholesky(S)
        reason = "range_reduced" if request < initial_request else "positive_rank"
    else:
        S = np.empty((0, 0), dtype=np.float64)
        reason = "singleton" if n == 1 else "constant_references" if constant else "eigenvalue_threshold"
    projected_unique = len(np.unique(projected, axis=0)) if rank else 1
    if n < 2:
        separation = None
    else:
        distances = _squared_distances(projected, projected)
        distances[np.diag_indices(n)] = np.inf
        separation = float(np.sqrt(np.min(distances)))
    return LocalPCA(mu, P, S, int(rank), cap, int(n), eigenvalues, ranges,
                    initial_request, request, threshold, tuple(attempts), reason,
                    repeated, int(projected_unique), separation)


__all__ = [
    "GEOMETRY_VERSION", "Standardizer", "ReferenceGroup", "ReferencePartition", "LocalPCA",
    "fit_standardizer", "fit_reference_partitions", "fit_local_pca",
]

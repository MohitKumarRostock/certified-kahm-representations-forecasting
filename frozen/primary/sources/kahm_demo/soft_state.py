"""Scores, partition memberships, and shared KAHM MEM/LOCAL encodings.

Based on our research. The local definition ``gaussian_kahm_encoder_v1`` in
``docs/ENCODER_DESIGN.md`` specifies these formulas. This module does not fit
geometry or consume forecast targets. Returned encodings are the representation
coordinates before the separate decoder feature standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    """Convert real numeric data without accepting complex/string coercion."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, not {raw.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def _finite_vector(value: ArrayLike, name: str) -> FloatArray:
    array = _finite_array(value, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return array


def _readonly_copy(value: ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MembershipResult:
    """Memberships in model order and one log normalizer per partition."""

    memberships: FloatArray
    log_normalizers: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "memberships", _readonly_copy(self.memberships))
        object.__setattr__(self, "log_normalizers", _readonly_copy(self.log_normalizers))


@dataclass(frozen=True, slots=True)
class EncodingResult:
    """Unstandardized MEM and LOCAL vectors, with an identical MEM prefix."""

    mem: FloatArray
    local: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mem", _readonly_copy(self.mem))
        object.__setattr__(self, "local", _readonly_copy(self.local))


def radial_angular_score(z: ArrayLike, reconstruction: ArrayLike) -> float:
    """Return the bounded full-space radial/angular reconstruction score.

    When either vector is zero, the angular discrepancy is 1/2.  In particular,
    the score at two zero vectors is 1/(2*sqrt(2)), not zero.  The angular
    calculation scales each nonzero vector before taking its norm, avoiding
    overflow from large finite coordinates.  A distance exceeding float64's
    range gives radial discrepancy one, the limiting value of the formula.
    """
    query = _finite_vector(z, "z")
    rebuilt = _finite_vector(reconstruction, "reconstruction")
    if query.shape != rebuilt.shape:
        raise ValueError("z and reconstruction must have the same shape")

    query_scale = float(np.max(np.abs(query)))
    rebuilt_scale = float(np.max(np.abs(rebuilt)))
    if query_scale == 0.0 or rebuilt_scale == 0.0:
        cosine = 0.0
    elif np.array_equal(query, rebuilt):
        # Identical nonzero vectors have exactly zero angle.  This also avoids
        # manufacturing a small angle through normalization roundoff.
        cosine = 1.0
    else:
        query_scaled = query / query_scale
        rebuilt_scaled = rebuilt / rebuilt_scale
        query_norm = math.hypot(*(float(v) for v in query_scaled))
        rebuilt_norm = math.hypot(*(float(v) for v in rebuilt_scaled))
        cosine = math.fsum(
            (float(a) / query_norm) * (float(b) / rebuilt_norm)
            for a, b in zip(query_scaled, rebuilt_scaled)
        )
        cosine = min(1.0, max(-1.0, cosine))

    # Python subtraction may produce an infinite distance between finite
    # extreme coordinates; expm1(-inf) correctly gives the saturated radius.
    distance = math.hypot(*(float(a) - float(b) for a, b in zip(query, rebuilt)))
    radial = -math.expm1(-distance)
    angular = math.acos(cosine) / math.pi
    # hypot preserves tiny radial discrepancies whose squares would underflow.
    score = math.hypot(radial, angular) / math.sqrt(2.0)
    return min(1.0, max(0.0, score))


def _positive_parameter(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _partition_sizes(group_sizes: Sequence[int], width: int) -> tuple[int, ...]:
    if isinstance(group_sizes, (str, bytes)):
        raise TypeError("group_sizes must be a sequence of positive integers")
    try:
        sizes = tuple(group_sizes)
    except TypeError as exc:
        raise TypeError("group_sizes must be a sequence of positive integers") from exc
    if not sizes:
        raise ValueError("at least one partition is required")
    for size in sizes:
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral):
            raise TypeError("each group size must be an integer")
        if size <= 0:
            raise ValueError("each group size must be positive")
    normalized = tuple(int(size) for size in sizes)
    if sum(normalized) != width:
        raise ValueError("group sizes must sum to the number of scores")
    return normalized


def partition_memberships(
    scores: ArrayLike,
    group_sizes: Sequence[int],
    offset: float = 1e-6,
    contrast: float = 16,
) -> MembershipResult:
    """Normalize score affinities within contiguous, ordered partitions.

    ``group_sizes=(2, 4)`` means scores 0:2 and 2:6 are normalized separately.
    No small-affinity threshold, uniform replacement, or membership floor is
    applied.  Extremely small *relative* memberships may round to zero; this
    is part of floating-point evaluation, not an asserted positive lower bound.
    Parameters whose log affinities/normalizers cannot be represented as finite
    float64 values are rejected instead of returning a fabricated encoding.
    """
    score_vector = _finite_vector(scores, "scores")
    if np.any(score_vector < 0.0) or np.any(score_vector > 1.0):
        raise ValueError("scores must lie in [0, 1]")
    sizes = _partition_sizes(group_sizes, score_vector.size)
    offset_value = _positive_parameter(offset, "offset")
    contrast_value = _positive_parameter(contrast, "contrast")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_affinities = contrast_value * np.log(1.0 - score_vector + offset_value)
    if not np.all(np.isfinite(log_affinities)):
        raise ValueError("offset/contrast produce nonfinite log affinities")

    memberships = np.empty_like(score_vector)
    normalizers = np.empty(len(sizes), dtype=np.float64)
    start = 0
    for partition_index, size in enumerate(sizes):
        stop = start + size
        logs = log_affinities[start:stop]
        pivot = float(np.max(logs))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            shifted = np.exp(logs - pivot)
        total = math.fsum(float(weight) for weight in shifted)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("partition normalization failed")
        normalizer = pivot + math.log(total)
        if not math.isfinite(normalizer):
            raise ValueError("offset/contrast produce a nonfinite log normalizer")
        # The shifted quotient is algebraically the log-sum-exp softmax and
        # avoids cancellation in exp(log_affinity - a very large log_Z).
        memberships[start:stop] = shifted / total
        normalizers[partition_index] = normalizer
        start = stop
    return MembershipResult(memberships, normalizers)


def compose_encodings(
    z: ArrayLike,
    memberships: ArrayLike,
    local_means: ArrayLike,
    local_bases: Sequence[ArrayLike],
) -> EncodingResult:
    """Compose MEM and LOCAL from one shared bank evaluation.

    ``local_means`` has shape (J, d).  Basis j has shape (d, r_j), including
    (d, 0) for a rank-zero local model.  Each appended block is exactly
    phi_j * P_j.T @ (z - mu_j), in model/basis-column order.  The function
    receives no reconstruction vectors or forecasting targets.

    Membership entries are checked individually in [0,1].  Their per-partition
    normalization is the responsibility of ``partition_memberships``; the
    concatenation need not sum to one when it contains several partitions.
    """
    query = _finite_vector(z, "z")
    phi = _finite_vector(memberships, "memberships")
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("memberships must lie in [0, 1]")
    means = _finite_array(local_means, "local_means")
    if means.shape != (phi.size, query.size):
        raise ValueError("local_means must have shape (number of memberships, input dimension)")
    try:
        bases = tuple(local_bases)
    except TypeError as exc:
        raise TypeError("local_bases must contain one matrix per membership") from exc
    if len(bases) != phi.size:
        raise ValueError("local_bases must contain one matrix per membership")

    blocks = [phi]
    for j, basis in enumerate(bases):
        directions = _finite_array(basis, f"local_bases[{j}]")
        if (directions.ndim != 2 or directions.shape[0] != query.size
                or directions.shape[1] > query.size):
            raise ValueError(f"local_bases[{j}] must have shape (d, r), with 0 <= r <= d")
        if directions.shape[1] == 0:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            coordinates = directions.T @ (query - means[j])
            weighted = phi[j] * coordinates
        if not np.all(np.isfinite(weighted)):
            raise ValueError(f"local-position block {j} is not finite")
        blocks.append(weighted)

    local = np.concatenate(blocks)
    return EncodingResult(phi, local)

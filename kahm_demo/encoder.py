"""Training-only Gaussian KAHM banks and shared MEM/LOCAL feature maps.

Based on our research. The local definition ``gaussian_kahm_encoder_v1`` is
specified in ``docs/ENCODER_DESIGN.md``.
It constructs development candidates; it does not fit or select forecasters.
Canonical inference is one history at a time, including in ``transform``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import (ReferencePartition, Standardizer, fit_reference_partitions,
                       fit_standardizer)
from .kahm import LocalKAHM, fit_local_kahm
from .soft_state import compose_encodings, partition_memberships, radial_angular_score

CANONICAL_ENCODER_METHOD = "single_history_numpy_float64_v1"
ENCODING_NAMES = ("raw", "kahm_mem", "kahm_local")


def _readonly(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    # Immutable bytes prevent callers from turning the write flag back on.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    cap: int
    requested_groups: tuple[int, ...] = (2, 4)
    reference_budget: int = 24
    offset: float = 1e-6
    contrast: float = 16.0
    kmeans_max_iterations: int = 150

    def __post_init__(self):
        if type(self.cap) is not int or self.cap < 1:
            raise ValueError("PCA cap must be a positive integer")
        if not isinstance(self.requested_groups, tuple) or not self.requested_groups:
            raise ValueError("requested_groups must be a nonempty tuple")
        if any(type(k) is not int or k < 1 for k in self.requested_groups):
            raise ValueError("partition group counts must be positive integers")
        for name in ("reference_budget", "kmeans_max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("offset", "contrast"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int)):
                raise ValueError(f"{name} must be a finite positive number")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True, slots=True)
class EncodedHistory:
    raw: np.ndarray
    kahm_mem: np.ndarray
    kahm_local: np.ndarray
    scores: np.ndarray
    log_normalizers: np.ndarray
    reconstructions: np.ndarray
    model_diagnostics: tuple[dict, ...]

    def features(self, encoding: str) -> np.ndarray:
        if encoding not in ENCODING_NAMES:
            raise ValueError(f"encoding must be one of {ENCODING_NAMES}")
        return getattr(self, encoding)


@dataclass(frozen=True, slots=True)
class KAHMEncoder:
    config: EncoderConfig
    standardizer: Standardizer
    partitions: tuple[ReferencePartition, ...]
    models: tuple[LocalKAHM, ...]

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(partition.groups) for partition in self.partitions)

    @property
    def input_width(self) -> int:
        return len(self.standardizer.mean)

    @property
    def feature_widths(self) -> dict[str, int]:
        count = len(self.models)
        return {"raw": self.input_width, "kahm_mem": count,
                "kahm_local": count + sum(model.pca.rank for model in self.models)}

    def encode_one(self, history: Sequence[float]) -> EncodedHistory:
        z = self.standardizer.transform_one(history)
        reconstructed = tuple(model.reconstruct_one(z) for model in self.models)
        reconstructions = np.stack([item.reconstruction for item in reconstructed])
        scores = np.asarray([radial_angular_score(z, a) for a in reconstructions])
        affinities = partition_memberships(scores, self.group_sizes,
                                           offset=self.config.offset,
                                           contrast=self.config.contrast)
        encodings = compose_encodings(z, affinities.memberships,
                                      [model.pca.mu for model in self.models],
                                      [model.pca.P for model in self.models])
        return EncodedHistory(
            raw=_readonly(z), kahm_mem=encodings.mem, kahm_local=encodings.local,
            scores=_readonly(scores), log_normalizers=affinities.log_normalizers,
            reconstructions=_readonly(reconstructions),
            model_diagnostics=tuple(dict(item.diagnostics) for item in reconstructed))

    def transform_one(self, history: Sequence[float], encoding: str = "kahm_mem") -> np.ndarray:
        if encoding not in ENCODING_NAMES:
            raise ValueError(f"encoding must be one of {ENCODING_NAMES}")
        if encoding == "raw":
            return _readonly(self.standardizer.transform_one(history))
        return self.encode_one(history).features(encoding)

    def transform(self, histories, encoding: str = "kahm_mem") -> np.ndarray:
        """Encode rows canonically; memoization preserves every repeated episode.

        Identical float64 input rows reuse a computed feature row. This changes
        neither row order nor row count and performs no statistical deduplication.
        """
        if encoding not in ENCODING_NAMES:
            raise ValueError(f"encoding must be one of {ENCODING_NAMES}")
        histories = np.asarray(histories)
        if histories.dtype.kind not in "iuf":
            raise TypeError("histories must contain real integer or floating-point numbers")
        histories = np.asarray(histories, dtype=np.float64)
        if histories.ndim != 2 or histories.shape[1] != self.input_width:
            raise ValueError("histories must be a matrix with the fitted input width")
        if not np.isfinite(histories).all():
            raise ValueError("histories must be finite")
        if not len(histories):
            return _readonly(np.empty((0, self.feature_widths[encoding])))
        cache = {}
        rows = []
        for history in histories:
            # Bytes preserve signed zeros and avoid converting numerical keys.
            key = history.tobytes()
            if key not in cache:
                cache[key] = self.transform_one(history, encoding)
            rows.append(cache[key])
        return _readonly(np.stack(rows))


def fit_encoder_candidates(histories, *, caps: tuple[int, ...] = (1, 2),
                           requested_groups: tuple[int, ...] = (2, 4),
                           reference_budget: int = 24, offset: float = 1e-6,
                           contrast: float = 16.0,
                           kmeans_max_iterations: int = 150) -> tuple[KAHMEncoder, ...]:
    """Fit banks from training INPUTS only, sharing all reference identities.

    The order of ``caps`` is the development configuration tie order. Selecting
    the common bank requires both fitted decoders and validation loss during readout development.
    """
    if not isinstance(caps, tuple) or not caps or len(set(caps)) != len(caps):
        raise ValueError("caps must be a nonempty tuple of distinct positive integers")
    configs = tuple(EncoderConfig(cap, requested_groups, reference_budget, offset,
                                  contrast, kmeans_max_iterations) for cap in caps)
    standardizer = fit_standardizer(histories)
    z = standardizer.transform(histories)
    partitions = fit_reference_partitions(z, requested_groups=requested_groups,
                                          reference_budget=reference_budget,
                                          max_iter=kmeans_max_iterations)
    reference_groups = tuple(group for partition in partitions for group in partition.groups)
    return tuple(KAHMEncoder(config, standardizer, partitions,
                             tuple(fit_local_kahm(group.Q, config.cap)
                                   for group in reference_groups))
                 for config in configs)


def fit_encoder(histories, *, cap: int = 2, **kwargs) -> KAHMEncoder:
    """Fit one explicitly requested development bank without model selection."""
    return fit_encoder_candidates(histories, caps=(cap,), **kwargs)[0]

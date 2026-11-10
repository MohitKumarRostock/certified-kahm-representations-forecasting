"""Lossless, non-pickle storage of fitted encoder state.

Arrays and floating scalars are stored as canonical hexadecimal strings. Loading
preserves fitted values: it does not rerun PCA, clustering, or regularization.
Replay of the canonical feature table detects runtime-dependent inference drift.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from contextlib import redirect_stdout
import hashlib
import io
import math
from pathlib import Path
import platform
import sys

import numpy as np

from . import __version__
from ._util import canonical_json_bytes, read_json, write_json
from .contract import CONTRACT_SHA256, validate_contract
from .encoder import CANONICAL_ENCODER_METHOD, EncoderConfig, KAHMEncoder
from .geometry import (LocalPCA, ReferenceGroup, ReferencePartition, Standardizer,
                       _PartitionAttempt, _PCAAttempt)
from .kahm import LocalKAHM

ENCODER_SCHEMA = "kahm.encoder_bank.v1"
_CLASSES = {cls.__name__: cls for cls in (EncoderConfig, KAHMEncoder, Standardizer,
            ReferenceGroup, ReferencePartition, LocalPCA, LocalKAHM,
            _PartitionAttempt, _PCAAttempt)}


def runtime_record() -> dict:
    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    return {"python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_full_version": sys.version,
            "numpy_version": np.__version__, "machine": platform.machine(),
            "numpy_build_configuration": configuration.getvalue(),
            "canonical_method": CANONICAL_ENCODER_METHOD,
            "arithmetic": "float64; canonical one-history inference"}


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            if value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                raise ValueError("encoder floating arrays must be finite float64")
            dtype = "float64"
            items = [float(x).hex() for x in value.flat]
        elif value.dtype.kind in "iu":
            dtype = "int64"
            items = [int(x) for x in value.flat]
        else:
            raise ValueError("unsupported encoder array dtype")
        return {"_type": "array", "dtype": dtype, "shape": list(value.shape), "values": items}
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("encoder state must contain finite floating values")
        return {"_type": "float", "hex": float(value).hex()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if is_dataclass(value):
        name = type(value).__name__
        if name not in _CLASSES:
            raise ValueError(f"unregistered state class: {name}")
        return {"_type": name, "fields": {field.name: _pack(getattr(value, field.name))
                                          for field in fields(value) if field.init}}
    if isinstance(value, tuple):
        return {"_type": "tuple", "items": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key == "_type" for key in value):
            raise ValueError("encoder metadata keys must be strings other than _type")
        return {key: _pack(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError(f"unsupported encoder state value: {type(value).__name__}")


def _hex_float(value):
    if not isinstance(value, str):
        raise ValueError("floating encoder value must be hexadecimal text")
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid floating encoder value") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError("nonfinite or noncanonical floating encoder value")
    return number


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, bool):
            return value
        raise ValueError("untagged floating or unsupported encoder state")
    tag = value.get("_type")
    if tag is None:
        return {key: _unpack(item) for key, item in value.items()}
    if tag == "float":
        if set(value) != {"_type", "hex"}:
            raise ValueError("invalid floating encoder fields")
        return _hex_float(value["hex"])
    if tag == "array":
        if set(value) != {"_type", "dtype", "shape", "values"}:
            raise ValueError("invalid encoder array fields")
        shape, items = value["shape"], value["values"]
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("invalid encoder array shape")
        if not isinstance(items, list) or math.prod(shape) != len(items):
            raise ValueError("encoder array length differs from shape")
        if value["dtype"] == "float64":
            array = np.asarray([_hex_float(item) for item in items], dtype=np.float64)
        elif value["dtype"] == "int64":
            if any(type(item) is not int for item in items):
                raise ValueError("invalid integer array value")
            array = np.asarray(items, dtype=np.int64)
        else:
            raise ValueError("unsupported encoder array dtype")
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(shape)
    if tag == "tuple":
        if set(value) != {"_type", "items"} or not isinstance(value["items"], list):
            raise ValueError("invalid encoder tuple fields")
        return tuple(_unpack(item) for item in value["items"])
    if tag not in _CLASSES or set(value) != {"_type", "fields"}:
        raise ValueError("unknown encoder state class or fields")
    cls = _CLASSES[tag]
    names = {field.name for field in fields(cls) if field.init}
    if not isinstance(value["fields"], dict) or set(value["fields"]) != names:
        raise ValueError(f"invalid fields for encoder class {tag}")
    return cls(**{key: _unpack(item) for key, item in value["fields"].items()})


def validate_encoder(encoder: KAHMEncoder) -> None:
    """Check state shapes and defining numerical identities without refitting."""
    if not isinstance(encoder, KAHMEncoder) or not isinstance(encoder.config, EncoderConfig):
        raise ValueError("state does not contain a KAHM encoder")
    standardizer = encoder.standardizer
    if not isinstance(standardizer, Standardizer):
        raise ValueError("invalid standardizer state")
    d = len(standardizer.mean)
    if d < 1 or standardizer.mean.shape != (d,) or standardizer.scale.shape != (d,):
        raise ValueError("invalid standardizer dimensions")
    if not np.isfinite(standardizer.mean).all() or not np.isfinite(standardizer.scale).all():
        raise ValueError("nonfinite standardizer")
    if np.any(standardizer.scale <= 0) or standardizer.sample_count < 1:
        raise ValueError("invalid standardizer scale or sample count")
    if len(encoder.partitions) != len(encoder.config.requested_groups):
        raise ValueError("partition count differs from configuration")
    groups = tuple(group for partition in encoder.partitions for group in partition.groups)
    if not groups or len(groups) != len(encoder.models) or any(size < 1 for size in encoder.group_sizes):
        raise ValueError("local-model count differs from partitions")
    for group, model in zip(groups, encoder.models, strict=True):
        if not isinstance(model, LocalKAHM) or not isinstance(model.pca, LocalPCA):
            raise ValueError("invalid local KAHM state")
        q, pca = model.Q, model.pca
        if q.ndim != 2 or q.shape[1] != d or not len(q) or not np.isfinite(q).all():
            raise ValueError("invalid reference matrix")
        if not np.array_equal(q, group.Q) or len(np.unique(q, axis=0)) != len(q):
            raise ValueError("local references differ from distinct partition references")
        n, r = len(q), pca.rank
        if type(r) is not int or not 0 <= r <= min(encoder.config.cap, d, n - 1):
            raise ValueError("invalid stored PCA rank")
        if pca.P.shape != (d, r) or pca.S.shape != (r, r) or pca.mu.shape != (d,):
            raise ValueError("invalid PCA shapes")
        if not all(np.isfinite(a).all() for a in (pca.P, pca.S, pca.mu)):
            raise ValueError("nonfinite PCA state")
        if not np.allclose(pca.mu, np.mean(q, axis=0), rtol=1e-12, atol=1e-12):
            raise ValueError("local mean differs from references")
        if not np.allclose(pca.P.T @ pca.P, np.eye(r), rtol=1e-11, atol=1e-11):
            raise ValueError("PCA directions are not orthonormal")
        if r == 0:
            continue
        projected = (q - pca.mu) @ pca.P
        covariance = projected.T @ projected / (n - 1)
        if not np.allclose(covariance, pca.S, rtol=1e-10, atol=1e-12):
            raise ValueError("projected covariance differs from references")
        cholesky = np.linalg.cholesky(pca.S)
        if model.metric_transform.shape != (d, r):
            raise ValueError("invalid kernel metric shape")
        expected_metric = np.linalg.solve(cholesky, pca.P.T).T / np.sqrt(r)
        if not np.allclose(model.metric_transform, expected_metric, rtol=1e-10, atol=1e-12):
            raise ValueError("kernel metric differs from the stored PCA basis and covariance")
        white = (q - pca.mu) @ model.metric_transform
        # The saved metric includes 1/sqrt(rank), as in the Gaussian exponent.
        if not np.allclose(white.T @ white / (n - 1), np.eye(r) / r, rtol=1e-9, atol=1e-10):
            raise ValueError("kernel metric does not whiten reference covariance")
        differences = (q[:, None, :] - q[None, :, :]) @ model.metric_transform
        gram = np.exp(-0.5 * np.sum(differences * differences, axis=2))
        if model.gram.shape != (n, n) or not np.allclose(model.gram, gram, rtol=1e-11, atol=1e-12):
            raise ValueError("saved Gram matrix differs from Gaussian reference kernel")
        if not np.isfinite(model.fitted_lambda) or model.fitted_lambda <= 0:
            raise ValueError("invalid fitted KAHM regularizer")
        if model.C.shape != (n, n) or not np.isfinite(model.C).all():
            raise ValueError("invalid KAHM inverse shape or values")
        residual = (model.gram + model.fitted_lambda * np.eye(n)) @ model.C - np.eye(n)
        if np.max(np.abs(residual)) > 1e-9:
            raise ValueError("saved inverse does not match the saved regularizer")


def encoder_state_fingerprint(encoder: KAHMEncoder) -> str:
    return hashlib.sha256(canonical_json_bytes(_pack(encoder))).hexdigest()


def shared_geometry_fingerprint(encoder: KAHMEncoder) -> str:
    """Identity of preprocessing, partitions, references, and occurrence counts."""
    state = _pack((encoder.standardizer, encoder.partitions))
    return hashlib.sha256(canonical_json_bytes(state)).hexdigest()


def save_encoder(encoder: KAHMEncoder, path: Path, *, provenance: dict | None = None) -> dict:
    validate_contract()
    validate_encoder(encoder)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"encoder output already exists: {path}")
    state = _pack(encoder)
    record = {"schema": ENCODER_SCHEMA, "status": "development_bank_unselected",
              "contract_sha256": CONTRACT_SHA256, "package_version": __version__,
              "runtime": runtime_record(), "provenance": provenance or {},
              "state_sha256": hashlib.sha256(canonical_json_bytes(state)).hexdigest(),
              "state": state}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record


def load_encoder(path: Path) -> KAHMEncoder:
    validate_contract()
    record = read_json(Path(path))
    if not isinstance(record, dict) or record.get("schema") != ENCODER_SCHEMA:
        raise ValueError("unsupported encoder artifact schema")
    if record.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("encoder belongs to another scientific contract")
    if record.get("status") != "development_bank_unselected":
        raise ValueError("unsupported encoder development status")
    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("encoder runtime metadata must be an object")
    if runtime.get("canonical_method") != CANONICAL_ENCODER_METHOD:
        raise ValueError("unsupported canonical encoder inference convention")
    state = record.get("state")
    if hashlib.sha256(canonical_json_bytes(state)).hexdigest() != record.get("state_sha256"):
        raise ValueError("encoder state hash mismatch")
    encoder = _unpack(state)
    validate_encoder(encoder)
    return encoder

"""Lossless storage and structural validation of development forecasters.

The artifact contains the actual fitted float64 computational state. Loading
never replaces that state by a refit: kernel, spectral, and affine identities
are checked against the saved values. These numerical checks are not interval
proofs and cannot verify training moments or a fitted ridge solution without
the training data. Dataset-bound replay belongs to the development workflow.

All floating state and provenance values use canonical hexadecimal strings.
The state hash identifies the fitted object; the record hash additionally binds
its contract, supported inference conventions, recorded runtime, and provenance.
A recorded package/runtime version need not equal the currently installed one:
full-support prediction replay is the separate check for inference drift.

The encoder serializer and its class registry are reused without mutation.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from . import __version__
from ._util import canonical_json_bytes, read_json, write_json
from .contract import CONTRACT_SHA256, validate_contract
from . import encoder_state
from .encoder import CANONICAL_ENCODER_METHOD, KAHMEncoder
from .forecast import CANONICAL_FORECAST_METHOD, CANDIDATE_IDS, ForecastModel
from .geometry import Standardizer, STANDARD_DEVIATION_FLOOR as INPUT_STD_FLOOR
from .readout_features import (
    ABSOLUTE_EIGENVALUE_CUTOFF, BANDWIDTH_MULTIPLIERS,
    MIN_LANDMARK_DISTANCE_SQUARED, MIN_SIGMA_SQUARED,
    RELATIVE_EIGENVALUE_CUTOFF, STANDARD_DEVIATION_FLOOR as READOUT_STD_FLOOR,
    ColumnScaler, ReadoutFeatures,
)
from .ridge import RidgeModel


FORECAST_SCHEMA = "kahm.forecast_model.v1"
FORECAST_STATUS = "development_selected_not_frozen"
FORECAST_CANDIDATE_STATUS = "development_candidate"
_DEVELOPMENT_STATUSES = (FORECAST_STATUS, FORECAST_CANDIDATE_STATUS)
_CLASSES = {cls.__name__: cls for cls in
            (ForecastModel, ColumnScaler, ReadoutFeatures, RidgeModel)}
_RECORD_FIELDS = {
    "schema", "status", "contract_sha256", "package_version", "runtime",
    "provenance", "state_sha256", "state", "record_sha256",
}
_RUNTIME_FIELDS = {
    "python_implementation", "python_version", "python_full_version",
    "numpy_version", "machine", "numpy_build_configuration",
    "canonical_method", "arithmetic", "forecast_canonical_method",
}


def _pack(value):
    """Extend the old serializer locally, never changing its class registry."""
    if is_dataclass(value) and type(value).__name__ in _CLASSES:
        name = type(value).__name__
        if type(value) is not _CLASSES[name]:
            raise ValueError(f"unsupported forecaster state class: {name}")
        return {"_type": name, "fields": {
            field.name: _pack(getattr(value, field.name))
            for field in fields(value) if field.init
        }}
    if isinstance(value, tuple):
        return {"_type": "tuple", "items": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key == "_type" for key in value):
            raise ValueError("state metadata keys must be strings other than _type")
        return {key: _pack(item) for key, item in value.items()}
    return encoder_state._pack(value)


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        return encoder_state._unpack(value)
    tag = value.get("_type")
    if tag is None:
        return {key: _unpack(item) for key, item in value.items()}
    if tag == "tuple":
        if set(value) != {"_type", "items"} or not isinstance(value["items"], list):
            raise ValueError("invalid forecaster tuple fields")
        return tuple(_unpack(item) for item in value["items"])
    if not isinstance(tag, str):
        raise ValueError("invalid forecaster state type tag")
    if tag in _CLASSES:
        cls = _CLASSES[tag]
        names = {field.name for field in fields(cls) if field.init}
        if (set(value) != {"_type", "fields"}
                or not isinstance(value["fields"], dict)
                or set(value["fields"]) != names):
            raise ValueError(f"invalid fields for forecaster class {tag}")
        return cls(**{key: _unpack(item) for key, item in value["fields"].items()})
    return encoder_state._unpack(value)


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64")
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite float64 array of shape {shape}")
    return value


def _scalar(value, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _validate_scaler(scaler, *, input_scaler: bool, count: int | None = None) -> int:
    cls = Standardizer if input_scaler else ColumnScaler
    if type(scaler) is not cls:
        raise ValueError("invalid fitted scaler class")
    if not isinstance(scaler.mean, np.ndarray) or scaler.mean.ndim != 1:
        raise ValueError("scaler mean must be a vector")
    width = _positive_int(int(scaler.mean.size), "scaler width")
    for name in ("mean", "scale", "raw_std"):
        _array(getattr(scaler, name), (width,), f"scaler {name}")
    sample_count = _positive_int(scaler.sample_count, "scaler sample count")
    if count is not None and sample_count != count:
        raise ValueError("scaler sample count differs from the linked training count")
    if np.any(scaler.raw_std < 0) or np.any(scaler.scale <= 0):
        raise ValueError("scaler standard deviations or scales are invalid")
    mask = (scaler.raw_std <= INPUT_STD_FLOOR if input_scaler
            else scaler.raw_std < READOUT_STD_FLOOR)
    expected = tuple(int(index) for index in np.flatnonzero(mask))
    if (not isinstance(scaler.replaced_columns, tuple)
            or any(type(index) is not int for index in scaler.replaced_columns)
            or scaler.replaced_columns != expected
            or not np.array_equal(scaler.scale, np.where(mask, 1.0, scaler.raw_std))):
        raise ValueError("scaler state violates its standard-deviation replacement rule")
    return width


def _validate_readout(readout: ReadoutFeatures, representation_width: int, count: int) -> None:
    if type(readout) is not ReadoutFeatures:
        raise ValueError("invalid matched readout feature state")
    if _positive_int(readout.training_count, "readout training count") != count:
        raise ValueError("readout training count differs from preprocessing")
    if _validate_scaler(readout.linear_scaler, input_scaler=False, count=count) != representation_width:
        raise ValueError("readout linear width differs from its representation")
    indices = readout.landmark_indices
    if (not isinstance(indices, tuple) or not indices
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("landmark row indices must be distinct in-range training indices")
    m = len(indices)
    landmarks = _array(readout.landmarks, (m, representation_width), "landmarks")
    gram = _array(readout.landmark_gram, (m, m), "landmark Gram matrix")
    if not isinstance(readout.eigenvalues, np.ndarray) or readout.eigenvalues.ndim != 1:
        raise ValueError("retained eigenvalues must be a vector")
    r = int(readout.eigenvalues.size)
    if not 1 <= r <= m:
        raise ValueError("retained Nystrom rank is outside the landmark dimension")
    eigenvalues = _array(readout.eigenvalues, (r,), "retained eigenvalues")
    vectors = _array(readout.eigenvectors, (m, r), "retained eigenvectors")
    if np.any(eigenvalues <= 0) or np.any(eigenvalues[1:] > eigenvalues[:-1]):
        raise ValueError("retained eigenvalues must be positive and nonincreasing")
    if _validate_scaler(readout.nystrom_scaler, input_scaler=False, count=count) != r:
        raise ValueError("Nystrom scaler width differs from the retained spectral rank")
    multiplier = _scalar(readout.bandwidth_multiplier, "bandwidth multiplier", positive=True)
    if multiplier not in BANDWIDTH_MULTIPLIERS:
        raise ValueError("unsupported bandwidth multiplier")
    delta = _scalar(readout.median_squared_distance, "distance median", positive=True)
    sigma = _scalar(readout.sigma_squared, "squared bandwidth", positive=True)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        differences = landmarks[:, None, :] - landmarks[None, :, :]
        distances = np.sum(differences * differences, axis=2, dtype=np.float64)
    if not np.isfinite(distances).all():
        raise ValueError("landmark squared distances are nonfinite")
    pairs = distances[np.triu_indices(m, k=1)]
    positive = pairs[pairs > MIN_LANDMARK_DISTANCE_SQUARED]
    expected_delta = float(np.median(positive)) if positive.size else 1.0
    if not math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("saved distance median differs from the landmark distance rule")
    if sigma != max(delta * multiplier ** 2, MIN_SIGMA_SQUARED):
        raise ValueError("saved squared bandwidth differs from its defining rule")
    with np.errstate(under="ignore"):
        expected_gram = np.exp(-distances / (2.0 * sigma))
        expected_gram = (expected_gram + expected_gram.T) / 2.0
    if not np.allclose(gram, expected_gram, rtol=1e-11, atol=1e-12):
        raise ValueError("saved landmark Gram differs from its Gaussian kernel")
    if not np.allclose(vectors.T @ vectors, np.eye(r), rtol=1e-10, atol=1e-11):
        raise ValueError("retained landmark eigenvectors are not orthonormal")
    if not np.allclose(gram @ vectors, vectors * eigenvalues[None, :],
                       rtol=1e-9, atol=1e-11):
        raise ValueError("retained landmark eigenpairs do not match the saved Gram")
    # Re-evaluate eigenvalues only to check the saved truncation. Do not replace
    # eigenvectors, scalers, or any other fitted inference state.
    full_spectrum = np.linalg.eigvalsh(gram)[::-1]
    if not np.isfinite(full_spectrum).all():
        raise ValueError("landmark spectrum check is nonfinite")
    cutoff = max(ABSOLUTE_EIGENVALUE_CUTOFF,
                 RELATIVE_EIGENVALUE_CUTOFF * float(full_spectrum[0]))
    # eigvalsh and the fitting eigh can differ by roundoff at the cutoff. Reject
    # omitted/retained directions only when separated from it by a small audit
    # tolerance; the exact fitted cutoff decision is recorded in diagnostics.
    audit_tolerance = 16 * np.finfo(np.float64).eps * m * max(1.0, float(full_spectrum[0]))
    if (np.any(eigenvalues <= cutoff - audit_tolerance)
            or (r < m and full_spectrum[r] > cutoff + audit_tolerance)
            or not np.allclose(eigenvalues, full_spectrum[:r], rtol=1e-8,
                               atol=audit_tolerance)):
        raise ValueError("saved Nystrom truncation is inconsistent with the landmark spectrum")
    if not isinstance(readout.diagnostics, dict):
        raise ValueError("readout diagnostics must be a dictionary")


def validate_forecaster(model: ForecastModel) -> None:
    """Check internal state and family linkage, without validating a data fit."""
    if type(model) is not ForecastModel or model.candidate_id not in CANDIDATE_IDS:
        raise ValueError("state does not contain a supported complete forecaster")
    if model.candidate_id == "raw":
        if model.encoder is not None or model.bank_id is not None:
            raise ValueError("RAW must not contain a KAHM bank")
        representation_width = _validate_scaler(model.raw_standardizer, input_scaler=True)
        standardizer = model.raw_standardizer
    else:
        if model.raw_standardizer is not None or type(model.encoder) is not KAHMEncoder:
            raise ValueError("KAHM forecaster must contain its encoder and no separate input scaler")
        encoder_state.validate_encoder(model.encoder)
        standardizer = model.encoder.standardizer
        _validate_scaler(standardizer, input_scaler=True)
        if model.bank_id != f"cap_{model.encoder.config.cap}":
            raise ValueError("selected bank identity differs from its encoder configuration")
        representation_width = model.encoder.feature_widths[model.candidate_id]
        for partition in model.encoder.partitions:
            if (_positive_int(partition.training_count, "partition training count")
                    != standardizer.sample_count):
                raise ValueError("partition training count differs from input preprocessing")
            occurrence_count = 0
            for group in partition.groups:
                multiplicities = group.member_multiplicities
                if (not isinstance(multiplicities, np.ndarray)
                        or multiplicities.dtype.kind not in "iu" or multiplicities.ndim != 1
                        or multiplicities.shape != group.member_indices.shape
                        or np.any(multiplicities <= 0)):
                    raise ValueError("invalid stored partition occurrence counts")
                occurrence_count += sum(int(value) for value in multiplicities)
            if occurrence_count != standardizer.sample_count:
                raise ValueError("partition occurrences do not sum to the training count")
    if standardizer.input_dimension != 2:
        raise ValueError("this demonstrator expects a two-observation history")
    count = _positive_int(standardizer.sample_count, "training count")
    _validate_readout(model.readout, representation_width, count)
    ridge = model.ridge
    if type(ridge) is not RidgeModel:
        raise ValueError("invalid ridge decoder state")
    if _positive_int(ridge.training_count, "ridge training count") != count:
        raise ValueError("ridge training count differs from its linked feature fit")
    width = representation_width + model.readout.nystrom_rank
    coefficients = _array(ridge.coefficients, (width,), "ridge coefficients")
    mean = _array(ridge.feature_mean, (width,), "ridge feature mean")
    intercept = _scalar(ridge.intercept, "ridge intercept")
    alpha = _scalar(ridge.alpha, "ridge alpha", positive=True)
    target_mean = _scalar(ridge.target_mean, "ridge target mean")
    if not 0.0 <= target_mean <= 1.0 or not math.isfinite(count * alpha):
        raise ValueError("ridge target mean or scaled penalty is invalid")
    terms = [target_mean] + [float(x) * -float(w)
                             for x, w in zip(mean, coefficients, strict=True)]
    if not all(math.isfinite(term) for term in terms):
        raise ValueError("ridge intercept relation contains nonfinite products")
    expected_intercept = math.fsum(terms)
    if not math.isclose(intercept, expected_intercept, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("ridge intercept differs from the unpenalized mean relation")
    if not isinstance(ridge.diagnostics, dict):
        raise ValueError("ridge diagnostics must be a dictionary")
    # Also reject nonfinite or unsupported leaves in metadata and old state that
    # are not visited by the numerical identities above.
    _pack(model)


def _metadata(value) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key == "_type" for key in value):
            raise ValueError("provenance keys must be strings other than _type")
        for item in value.values():
            _metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _metadata(item)
    elif value is None or type(value) in (str, int, bool):
        return
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return
    else:
        raise ValueError("provenance must contain only finite JSON-compatible metadata")


def _hash(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def forecaster_state_fingerprint(model: ForecastModel) -> str:
    """Return a lossless identity of the stored state, excluding its envelope."""
    if type(model) is not ForecastModel:
        raise ValueError("expected a complete ForecastModel")
    return _hash(_pack(model))


def save_forecaster(model: ForecastModel, path: Path, *, provenance: dict | None = None,
                    status: str = FORECAST_STATUS) -> dict:
    validate_contract()
    validate_forecaster(model)
    if not isinstance(status, str) or status not in _DEVELOPMENT_STATUSES:
        raise ValueError("unsupported forecaster development status")
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")
    _metadata(provenance)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"forecaster output already exists: {path}")
    state = _pack(model)
    runtime = encoder_state.runtime_record()
    runtime["forecast_canonical_method"] = CANONICAL_FORECAST_METHOD
    record = {
        "schema": FORECAST_SCHEMA, "status": status,
        "contract_sha256": CONTRACT_SHA256, "package_version": __version__,
        "runtime": runtime, "provenance": _pack(provenance),
        "state_sha256": _hash(state), "state": state,
    }
    record["record_sha256"] = _hash(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record


def _validate_record(record) -> None:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("invalid forecaster artifact fields")
    if record["schema"] != FORECAST_SCHEMA or record["status"] not in _DEVELOPMENT_STATUSES:
        raise ValueError("unsupported forecaster schema or development status")
    if record["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("forecaster belongs to another scientific contract")
    if not isinstance(record["package_version"], str) or not record["package_version"].strip():
        raise ValueError("missing recorded forecaster package version")
    runtime = record["runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS
            or any(not isinstance(value, str) for value in runtime.values())):
        raise ValueError("invalid forecaster runtime record")
    if (runtime["canonical_method"] != CANONICAL_ENCODER_METHOD
            or runtime["forecast_canonical_method"] != CANONICAL_FORECAST_METHOD):
        raise ValueError("unsupported canonical forecaster inference convention")
    for name in ("state_sha256", "record_sha256"):
        digest = record[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"invalid {name}")
    envelope = {key: value for key, value in record.items() if key != "record_sha256"}
    if _hash(envelope) != record["record_sha256"]:
        raise ValueError("forecaster record hash mismatch")
    if _hash(record["state"]) != record["state_sha256"]:
        raise ValueError("forecaster state hash mismatch")
    provenance = _unpack(record["provenance"])
    if not isinstance(provenance, dict):
        raise ValueError("forecaster provenance must be a dictionary")
    _metadata(provenance)


def load_forecaster(path: Path) -> ForecastModel:
    validate_contract()
    record = read_json(Path(path))
    try:
        _validate_record(record)
        model = _unpack(record["state"])
        validate_forecaster(model)
    except (TypeError, OverflowError, np.linalg.LinAlgError) as error:
        raise ValueError("invalid numerical forecaster state") from error
    return model


__all__ = ["FORECAST_SCHEMA", "FORECAST_STATUS", "FORECAST_CANDIDATE_STATUS", "save_forecaster",
           "load_forecaster", "forecaster_state_fingerprint", "validate_forecaster"]

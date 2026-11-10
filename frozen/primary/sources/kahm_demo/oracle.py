"""Exact finite-support reference risks and lossless prediction tables.

This module evaluates the declared probability law, not a calibration sample.
It must not be used to choose fitted models, calibration seeds, or retained
runs. Binary floating-point predictions are evaluated as their exact dyadic
values; no clipping or decimal approximation is performed here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
import math
from typing import Any

from .law import (
    BENCHMARK_ID,
    HISTORY_KEYS,
    OBSERVATIONS,
    OBSERVATION_FLOATS,
    conditional_mean,
    joint_probability,
)


Prediction = int | float | Fraction
History = tuple[int, int]
PREDICTION_TABLE_SCHEMA = "kahm.prediction_table.v1"


def _validate_history(key: object) -> History:
    if type(key) is not tuple or len(key) != 2:
        raise TypeError("each history key must be a (previous_state, current_state) tuple")
    a, b = key
    if type(a) is not int or type(b) is not int:
        raise TypeError("history state labels must be integers, not booleans or other types")
    if not (0 <= a < 4 and 0 <= b < 4):
        raise ValueError("history state labels must be in {0, 1, 2, 3}")
    return a, b


def _exact_prediction(value: Prediction) -> Fraction:
    if type(value) is int:
        exact = Fraction(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("predictions must be finite")
        exact = Fraction.from_float(value)
    elif isinstance(value, Fraction):
        exact = value
    else:
        raise TypeError("predictions must be int, float, or Fraction; booleans are not accepted")
    if not Fraction(0) <= exact <= Fraction(1):
        raise ValueError("predictions must already lie in [0, 1]; the oracle does not clip")
    return exact


def _validated_mapping(predictions: Mapping[History, Prediction]) -> dict[History, Fraction]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping over the 16 ordered histories")
    if len(predictions) != len(HISTORY_KEYS):
        raise ValueError("predictions must contain exactly all 16 ordered histories")
    exact: dict[History, Fraction] = {}
    for key, value in predictions.items():
        history = _validate_history(key)
        if history in exact:
            raise ValueError("duplicate history in prediction mapping")
        exact[history] = _exact_prediction(value)
    if set(exact) != set(HISTORY_KEYS):
        raise ValueError("predictions must contain exactly all 16 ordered histories")
    return exact


def exact_risk(predictions: Mapping[History, Prediction]) -> Fraction:
    """Return exact population MSE by summing all 64 history-target triples.

    ``Fraction`` predictions permit analytical reference calculations. ``float``
    predictions are converted with ``Fraction.from_float`` so that this same
    calculation assesses the actual represented outputs of an implementation.
    """
    exact = _validated_mapping(predictions)
    return sum(
        (
            joint_probability(a, b, c) * (OBSERVATIONS[c] - exact[(a, b)]) ** 2
            for a, b in HISTORY_KEYS
            for c in range(4)
        ),
        Fraction(0),
    )


def risk_of_predictor(callback: Callable[[tuple[float, float]], Prediction]) -> Fraction:
    """Evaluate a deterministic callback once on each of the 16 histories."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    predictions = {
        (a, b): callback((OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b]))
        for a, b in HISTORY_KEYS
    }
    return exact_risk(predictions)


def _fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def reference_summary() -> dict[str, Any]:
    """Return JSON-serializable analytical checks, never fitted-model results.

    Numerator and denominator are authoritative exact integers. The decimal
    entry is a convenience approximation and may hide very small differences.
    """
    bayes = {(a, b): conditional_mean(a, b) for a, b in HISTORY_KEYS}
    rounded_bayes = {key: float(value) for key, value in bayes.items()}
    bayes_risk = exact_risk(bayes)
    rounded_risk = exact_risk(rounded_bayes)
    summary: dict[str, Any] = {
        "schema": "kahm.reference_summary.v1",
        "benchmark_id": BENCHMARK_ID,
        "scope": "analytical reference functions; not fitted-model or calibration results",
        "support_histories": len(HISTORY_KEYS),
        "support_history_target_triples": 4 * len(HISTORY_KEYS),
        "bayes_mse": _fraction_record(bayes_risk),
        "best_constant_prediction": _fraction_record(Fraction(1, 2)),
        "best_constant_mse": _fraction_record(exact_risk({key: 0.5 for key in HISTORY_KEYS})),
        "constant_zero_mse": _fraction_record(exact_risk({key: 0 for key in HISTORY_KEYS})),
        "constant_one_mse": _fraction_record(exact_risk({key: 1 for key in HISTORY_KEYS})),
        "previous_observation_mse": _fraction_record(
            exact_risk({(a, b): OBSERVATIONS[a] for a, b in HISTORY_KEYS})
        ),
        "current_observation_mse": _fraction_record(
            exact_risk({(a, b): OBSERVATIONS[b] for a, b in HISTORY_KEYS})
        ),
        "nominal_successor_mse": _fraction_record(
            exact_risk({(a, b): OBSERVATIONS[(a + b) % 4] for a, b in HISTORY_KEYS})
        ),
        "rounded_bayes_mse": _fraction_record(rounded_risk),
        "rounded_bayes_excess_mse": _fraction_record(rounded_risk - bayes_risk),
        "conditional_mean_table": [
            {
                "history_id": 4 * a + b,
                "previous_state": a,
                "current_state": b,
                "conditional_mean": _fraction_record(bayes[(a, b)]),
            }
            for a, b in HISTORY_KEYS
        ],
        "arithmetic_note": (
            "Exact fraction fields are authoritative. Binary float predictions are their "
            "exact dyadic values, not rationalized decimal strings."
        ),
    }
    return summary


def _validate_candidate_id(candidate_id: object) -> str:
    if type(candidate_id) is not str:
        raise TypeError("candidate_id must be a string")
    if not candidate_id or candidate_id.strip() != candidate_id:
        raise ValueError("candidate_id must be nonempty with no leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in candidate_id):
        raise ValueError("candidate_id must not contain control characters")
    return candidate_id


def prediction_table_from_mapping(
    predictions: Mapping[History, float], candidate_id: str
) -> dict[str, Any]:
    """Serialize actual Python float outputs using canonical hexadecimal strings.

    The table format deliberately requires binary float predictions. Use
    ``exact_risk`` directly for analytical rational reference functions.
    """
    _validate_candidate_id(candidate_id)
    _validated_mapping(predictions)
    for value in predictions.values():
        if type(value) is not float:
            raise TypeError("prediction tables require actual float values")
    return {
        "schema": PREDICTION_TABLE_SCHEMA,
        "candidate_id": candidate_id,
        "predictions": [
            {
                "history_id": 4 * a + b,
                "previous_state": a,
                "current_state": b,
                "prediction_hex": predictions[(a, b)].hex(),
            }
            for a, b in HISTORY_KEYS
        ],
    }


def parse_prediction_table(payload: object) -> dict[History, float]:
    """Validate a complete prediction table and recover its exact binary outputs.

    Hexadecimal strings must have the canonical form emitted by ``float.hex``.
    Reordered entries are accepted, but missing, extra, or duplicate histories
    and history-ID disagreements are rejected. No values are clipped.
    """
    if type(payload) is not dict:
        raise TypeError("prediction table must be a JSON object")
    if set(payload) != {"schema", "candidate_id", "predictions"}:
        raise ValueError("prediction table has missing or unexpected fields")
    if payload["schema"] != PREDICTION_TABLE_SCHEMA:
        raise ValueError("unsupported prediction table schema")
    _validate_candidate_id(payload["candidate_id"])
    records = payload["predictions"]
    if type(records) is not list or len(records) != len(HISTORY_KEYS):
        raise ValueError("prediction table must contain exactly 16 prediction entries")

    recovered: dict[History, float] = {}
    entry_fields = {"history_id", "previous_state", "current_state", "prediction_hex"}
    for entry in records:
        if type(entry) is not dict or set(entry) != entry_fields:
            raise ValueError("prediction entry has missing or unexpected fields")
        a, b = _validate_history((entry["previous_state"], entry["current_state"]))
        history_id = entry["history_id"]
        if type(history_id) is not int:
            raise TypeError("history_id must be an integer, not a boolean or other type")
        if history_id != 4 * a + b:
            raise ValueError("history_id does not agree with the ordered state labels")
        history = (a, b)
        if history in recovered:
            raise ValueError("prediction table contains a duplicate history")
        encoded = entry["prediction_hex"]
        if type(encoded) is not str:
            raise TypeError("prediction_hex must be a canonical float.hex string")
        try:
            value = float.fromhex(encoded)
        except (ValueError, OverflowError) as exc:
            raise ValueError("prediction_hex is not a finite hexadecimal float") from exc
        _exact_prediction(value)
        if encoded != value.hex():
            raise ValueError("prediction_hex must use the canonical float.hex representation")
        recovered[history] = value

    if set(recovered) != set(HISTORY_KEYS):
        raise ValueError("prediction table must cover exactly all 16 ordered histories")
    return {history: recovered[history] for history in HISTORY_KEYS}


__all__ = [
    "PREDICTION_TABLE_SCHEMA",
    "exact_risk",
    "risk_of_predictor",
    "reference_summary",
    "prediction_table_from_mapping",
    "parse_prediction_table",
]

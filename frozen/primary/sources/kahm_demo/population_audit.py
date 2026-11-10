"""Exact finite-support risk audit of supplied actual Python-float forecasts.

The independent route below writes out the declared benchmark law: histories
(a,b) have mass 1/16 and successor c has conditional mass 17/20 when
c=(a+b)%4, otherwise 1/20. It does not call the law module's joint-probability
or conditional-mean helpers. The resulting expectation is checked exactly
against the unchanged ``oracle.exact_risk`` implementation.

Every numerical risk, moment, probability, and decomposition is a Fraction.
Forecasts are converted once from their actual binary64 values; no decimal
approximation, floating arithmetic on predictions, or clipping is used.

These functions do not load models, predictions, calibration data, or issued
certificates. They neither select forecasts nor alter an issued bound. The
caller supplies a complete prediction table and binds the returned audit to
the already frozen artifacts. Only the immutable scientific contract is read.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from .contract import CONTRACT_SHA256, load_contract
from . import oracle


POPULATION_AUDIT_SCHEMA = "kahm.exact_population_audit.v1"
BENCHMARK_ID = "four_state_order2_modular_v1"
_OBSERVATIONS = (Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1))
_HISTORIES = tuple((a, b) for a in range(4) for b in range(4))
_HISTORY_MASS = Fraction(1, 16)
_BAYES_REFERENCE = Fraction(9, 160)
_CONSTANT_PREDICTION = Fraction(1, 2)
_CONSTANT_REFERENCE = Fraction(5, 32)
_NOMINAL_REFERENCE = Fraction(1, 16)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fraction_record(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _contract_fraction(value: Any) -> Fraction:
    if (type(value) is not dict or set(value) != {"numerator", "denominator"}
            or type(value["numerator"]) is not int or type(value["denominator"]) is not int
            or value["denominator"] <= 0):
        raise ValueError("the declared benchmark contains an invalid exact rational fact")
    return Fraction(value["numerator"], value["denominator"])


def _validate_declared_benchmark() -> None:
    """Tie the separately implemented law to the declared benchmark facts."""
    contract = load_contract()  # The existing loader checks its fixed byte hash.
    try:
        benchmark = contract["benchmark"]
        expected = {
            "id": BENCHMARK_ID,
            "state_labels": [0, 1, 2, 3],
            "history_order": 2,
            "horizon": 1,
            "initial_state_pair_distribution": "uniform over all 16 ordered pairs",
            "nominal_successor": "(previous_state + current_state) mod 4",
            "transition_formula": "P(c|a,b) = (4/5) * indicator[c == (a+b) mod 4] + 1/20",
            "joint_probability_formula": "p(a,b,c) = P(c|a,b)/16",
            "support_histories": 16,
            "support_history_target_triples": 64,
        }
        for key, value in expected.items():
            _require(type(benchmark[key]) is type(value) and benchmark[key] == value,
                     f"independent population-audit law differs from the scientific contract: {key}")
        _require(all(type(state) is int for state in benchmark["state_labels"]),
                 "benchmark state labels must be strict integers")
        _require(tuple(_contract_fraction(item) for item in benchmark["observation_values"])
                 == _OBSERVATIONS, "independent observation map differs from the scientific contract")
        for key, value in (
            ("initial_state_pair_probability", _HISTORY_MASS),
            ("nominal_successor_probability", Fraction(17, 20)),
            ("each_other_successor_probability", Fraction(1, 20)),
            ("bayes_mse", _BAYES_REFERENCE),
            ("best_constant_prediction", _CONSTANT_PREDICTION),
            ("best_constant_mse", _CONSTANT_REFERENCE),
        ):
            _require(_contract_fraction(benchmark[key]) == value,
                     f"independent population-audit reference differs from the scientific contract: {key}")
    except (KeyError, TypeError) as error:
        raise ValueError("missing or malformed benchmark facts") from error


def _validated_predictions(predictions: Mapping) -> tuple[dict, dict]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must map all 16 ordered state histories to Python floats")
    if len(predictions) != 16:
        raise ValueError("predictions must cover exactly all 16 ordered state histories")
    observed: dict[tuple[int, int], float] = {}
    exact: dict[tuple[int, int], Fraction] = {}
    for history, value in predictions.items():
        if type(history) is not tuple or len(history) != 2:
            raise TypeError("history keys must be two-state tuples")
        a, b = history
        if type(a) is not int or type(b) is not int:
            raise TypeError("history states must be strict integers, not booleans")
        if not 0 <= a < 4 or not 0 <= b < 4:
            raise ValueError("history states must lie in {0,1,2,3}")
        if history in observed:
            raise ValueError("prediction mapping repeats an ordered history")
        if type(value) is not float:
            raise TypeError("population auditing requires actual Python float predictions")
        try:
            represented = Fraction.from_float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError("prediction values must be finite") from error
        if not 0 <= represented <= 1:
            raise ValueError("predictions must already lie in [0,1]; the audit does not clip")
        observed[history], exact[history] = value, represented
    _require(set(observed) == set(_HISTORIES), "prediction mapping lacks full ordered-history support")
    return ({history: observed[history] for history in _HISTORIES},
            {history: exact[history] for history in _HISTORIES})


def build_population_audit(predictions: Mapping[tuple[int, int], float]) -> dict:
    """Build the exact 64-triple and 16-history audit, with no artifact access.

    Only actual Python floats are accepted as predictions. Analytical rational
    conditional means appear as benchmark references, not replacements for the
    rounded forecast values being audited.
    """
    observed, exact = _validated_predictions(predictions)
    _validate_declared_benchmark()
    histories, triples = [], []
    mass = population_mse = bayes_from_triples = constant_mse = nominal_mse = Fraction(0)
    history_mse_sum = bayes_from_histories = excess_sum = target_mean = Fraction(0)

    for a, b in _HISTORIES:
        history_id = 4 * a + b
        nominal = (a + b) % 4
        forecast = exact[(a, b)]
        # This formula is independently derived from the declared transition:
        # (4/5)*z_nominal + (1/20)*sum(z_c), and sum(z_c)=2.
        mean_formula = Fraction(1, 10) + Fraction(4, 5) * _OBSERVATIONS[nominal]
        conditional_mass = history_mass = mean_from_sum = second_moment = conditional_mse = Fraction(0)
        conditional_bayes_from_sum = Fraction(0)
        for c, target in enumerate(_OBSERVATIONS):
            ticket_count = 17 if c == nominal else 1
            conditional_probability = Fraction(ticket_count, 20)
            # Explicit joint weights avoid both existing probability helpers.
            joint_probability = Fraction(ticket_count, 320)
            loss = (target - forecast) ** 2
            weighted_loss = joint_probability * loss
            triples.append({
                "triple_id": 4 * history_id + c,
                "history_id": history_id,
                "previous_state": a, "current_state": b, "next_state": c,
                "is_nominal_successor": c == nominal,
                "conditional_probability": _fraction_record(conditional_probability),
                "joint_probability": _fraction_record(joint_probability),
                "target": _fraction_record(target),
                "squared_loss": _fraction_record(loss),
                "weighted_squared_loss": _fraction_record(weighted_loss),
            })
            conditional_mass += conditional_probability
            history_mass += joint_probability
            mean_from_sum += conditional_probability * target
            second_moment += conditional_probability * target ** 2
            conditional_mse += conditional_probability * loss
            conditional_bayes_from_sum += conditional_probability * (target - mean_formula) ** 2
            population_mse += weighted_loss
            bayes_from_triples += joint_probability * (target - mean_formula) ** 2
            constant_mse += joint_probability * (target - _CONSTANT_PREDICTION) ** 2
            nominal_mse += joint_probability * (target - _OBSERVATIONS[nominal]) ** 2
            target_mean += joint_probability * target
            mass += joint_probability

        _require(conditional_mass == 1, "conditional transition masses do not sum to one")
        _require(history_mass == _HISTORY_MASS, "joint history mass differs from 1/16")
        _require(mean_from_sum == mean_formula, "conditional-mean formula disagrees with transition enumeration")
        variance = second_moment - mean_formula ** 2
        conditional_excess = (forecast - mean_formula) ** 2
        _require(variance == conditional_bayes_from_sum and variance >= 0,
                 "conditional variance disagrees with centered-loss enumeration")
        _require(conditional_mse == variance + conditional_excess,
                 "per-history bias-variance identity failed")
        weighted_mse = _HISTORY_MASS * conditional_mse
        weighted_bayes = _HISTORY_MASS * variance
        weighted_excess = _HISTORY_MASS * conditional_excess
        histories.append({
            "history_id": history_id,
            "previous_state": a, "current_state": b,
            "input": [_fraction_record(_OBSERVATIONS[a]), _fraction_record(_OBSERVATIONS[b])],
            "history_probability": _fraction_record(history_mass),
            "nominal_successor": nominal,
            "prediction_hex": observed[(a, b)].hex(),
            "prediction_exact": _fraction_record(forecast),
            "conditional_mean": _fraction_record(mean_formula),
            "conditional_second_moment": _fraction_record(second_moment),
            "conditional_variance": _fraction_record(variance),
            "conditional_mse": _fraction_record(conditional_mse),
            "conditional_excess_mse": _fraction_record(conditional_excess),
            "weighted_mse": _fraction_record(weighted_mse),
            "weighted_bayes_mse": _fraction_record(weighted_bayes),
            "weighted_excess_mse": _fraction_record(weighted_excess),
        })
        history_mse_sum += weighted_mse
        bayes_from_histories += weighted_bayes
        excess_sum += weighted_excess

    _require(len(histories) == 16 and len(triples) == 64, "finite support enumeration is incomplete")
    _require(mass == 1, "joint benchmark masses do not sum to one")
    _require(population_mse == history_mse_sum, "triple and history risk sums disagree")
    _require(bayes_from_triples == bayes_from_histories == _BAYES_REFERENCE,
             "enumerated Bayes risk differs from the 9/160 reference")
    _require(target_mean == _CONSTANT_PREDICTION and constant_mse == _CONSTANT_REFERENCE,
             "enumerated best-constant reference differs from the scientific contract")
    _require(nominal_mse == _NOMINAL_REFERENCE,
             "enumerated nominal-successor risk differs from 1/16")
    _require(population_mse == bayes_from_histories + excess_sum,
             "exact population bias-variance decomposition failed")

    # The unchanged oracle follows its own probability helper path. Pass the
    # copied actual floats, so it independently performs its exact conversion.
    oracle_mse = oracle.exact_risk(observed)
    _require(type(oracle_mse) is Fraction and oracle_mse == population_mse,
             "independent population expectation disagrees with oracle.exact_risk")

    return {
        "schema": POPULATION_AUDIT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "contract_sha256": CONTRACT_SHA256,
        "arithmetic": "exact Fraction arithmetic on the exact represented binary64 prediction values",
        "history_count": 16,
        "triple_count": 64,
        "joint_probability_mass": _fraction_record(mass),
        "exact_population_mse": _fraction_record(population_mse),
        "bayes_mse": _fraction_record(bayes_from_histories),
        "excess_over_bayes": _fraction_record(excess_sum),
        "references": {
            "bayes_mse": _fraction_record(bayes_from_histories),
            "best_constant_prediction": _fraction_record(target_mean),
            "best_constant_mse": _fraction_record(constant_mse),
            "nominal_successor_mse": _fraction_record(nominal_mse),
        },
        "histories": histories,
        "triples": triples,
        "checks": {
            "contract_facts_match": True,
            "full_history_and_triple_support": True,
            "joint_mass_is_one": mass == 1,
            "all_history_masses_are_one_sixteenth": True,
            "all_conditional_masses_are_one": True,
            "conditional_means_match_transition_enumeration": True,
            "per_history_bias_variance_identities": True,
            "triple_and_history_risk_sums_agree": population_mse == history_mse_sum,
            "bayes_reference_matches": bayes_from_triples == _BAYES_REFERENCE,
            "best_constant_reference_matches": constant_mse == _CONSTANT_REFERENCE,
            "nominal_successor_reference_matches": nominal_mse == _NOMINAL_REFERENCE,
            "oracle_exact_population_mse": _fraction_record(oracle_mse),
            "oracle_exact_equality": oracle_mse == population_mse,
            "population_minus_oracle": _fraction_record(population_mse - oracle_mse),
            "population_minus_bayes_minus_excess": _fraction_record(
                population_mse - bayes_from_histories - excess_sum),
        },
    }


def _compare_exact_record(actual: Any, expected: Any, path: str = "audit") -> None:
    """Strict JSON-shape equality, including bool/int and int/float distinctions."""
    if type(actual) is not type(expected):
        raise ValueError(f"population audit record has a different type at {path}")
    if type(expected) is dict:
        if any(type(key) is not str for key in actual) or set(actual) != set(expected):
            raise ValueError(f"population audit record has different fields at {path}")
        for key in expected:
            _compare_exact_record(actual[key], expected[key], f"{path}.{key}")
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(f"population audit record has a different list length at {path}")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _compare_exact_record(a, b, f"{path}[{index}]")
    elif actual != expected:
        raise ValueError(f"population audit record differs at {path}")


def verify_population_audit(predictions: Mapping[tuple[int, int], float], record: dict) -> dict:
    """Rebuild all exact contributions and require the complete canonical record.

    Return the independently rebuilt, identical record. The caller retains its
    own artifact identity and certificate-comparison responsibilities.
    """
    rebuilt = build_population_audit(predictions)
    _compare_exact_record(record, rebuilt)
    return rebuilt


__all__ = ["POPULATION_AUDIT_SCHEMA", "build_population_audit", "verify_population_audit"]

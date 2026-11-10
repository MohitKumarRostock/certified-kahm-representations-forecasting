"""Read the fixed public calibration protocol; perform no calibration.

The resource fixes the sampling, confidence allocation, exact-loss arithmetic,
verified logarithm recipe, bounded root-search policy, and reporting rules
used by the certifier. Its exact bytes, including these numerical choices,
are identified by a SHA-256 bound into the frozen model-family manifest.

This module does not implement the certifier, generate observations, evaluate
predictors, calculate fitted-model oracle risks, or change any artifact.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
from importlib.resources import files
import json

from .contract import (
    CANDIDATE_IDS, CONTRACT_SHA256, PRIMARY_CALIBRATION_COUNT, validate_contract,
)


CALIBRATION_PROTOCOL_SCHEMA = "kahm.calibration_protocol.v1"
CALIBRATION_PROTOCOL_SHA256 = "26d62b5c737e67f4cdb08250f10eddc8bf1c06c996b034c6bc6e1356f4e19c04"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fraction(value, name: str) -> Fraction:
    if (not isinstance(value, dict) or set(value) != {"numerator", "denominator"}
            or type(value["numerator"]) is not int or type(value["denominator"]) is not int
            or value["denominator"] <= 0):
        raise ValueError(f"{name} must be an exact rational record")
    return Fraction(value["numerator"], value["denominator"])


def _integer_equal(value, expected: int, name: str) -> None:
    _require(type(value) is int and value == expected, f"{name} differs from the fixed protocol")


def _validate_protocol(protocol: dict, contract: dict) -> None:
    """Check critical invariants and agreement with the declared benchmark law.

    The resource digest binds all prose and mathematical recurrences as well.
    This structural check is deliberately separate so contract changes cannot
    pass merely because the protocol resource itself is unchanged.
    """
    try:
        _require(isinstance(protocol, dict), "invalid calibration protocol")
        _require(protocol["schema"] == CALIBRATION_PROTOCOL_SCHEMA,
                 "unsupported calibration protocol schema")
        _require(protocol["protocol_id"] == "iid_three_forecaster_exact_mse_upper_v1",
                 "unsupported calibration protocol identity")
        _require(protocol["protocol_version"] == "1.0.0"
                 and protocol["status"] == "fixed_before_primary_calibration",
                 "unsupported calibration protocol version or status")
        _require(protocol["contract_sha256"] == CONTRACT_SHA256,
                 "calibration protocol refers to a different scientific contract")
        scope, sampling = protocol["scope"], protocol["sampling"]
        confidence, loss = protocol["confidence"], protocol["loss"]
        endpoint = protocol["endpoint"]
        logarithm, root = protocol["verified_logarithm"], protocol["upper_root_algorithm"]
        limits, reporting = protocol["work_limits"], protocol["reporting"]
        cert, data = contract["certificate"], contract["data_protocol"]
        _require(scope["candidate_ids"] == list(CANDIDATE_IDS)
                 == [candidate["id"] for candidate in contract["candidates"]],
                 "calibration candidate family differs from the scientific contract")
        _integer_equal(scope["family_size"], 3, "candidate family size")
        _integer_equal(cert["family_size"], 3, "contract candidate family size")
        _require(scope["certified_object"] == contract["claim_scope"]["certified_object"]
                 == "actual frozen deterministic computational predictor",
                 "certified object differs from the scientific contract")
        _require(scope["fitted_risk_upper_bounds_only"] is True
                 and scope["class_risk_lower_bound_claimed"] is False
                 and scope["ideal_real_arithmetic_KAHM_equivalence_claimed"] is False
                 and scope["oracle_risk_used_for_endpoint_or_selection"] is False,
                 "unsupported calibration claim scope")
        _require(sampling["benchmark_id"] == contract["benchmark"]["id"]
                 == "four_state_order2_modular_v1", "evaluation law differs from the scientific contract")
        for value, name in (
            (sampling["episode_count"], "episode count"),
            (confidence["statistical_independent_count"], "statistical count"),
            (endpoint["confidence_divisor"], "confidence divisor"),
            (cert["independent_unit_count"], "contract statistical count"),
            (data["primary_calibration_episodes"], "contract calibration count"),
        ):
            _integer_equal(value, PRIMARY_CALIBRATION_COUNT, name)
        _integer_equal(sampling["observations_per_episode"], 1, "observations per episode")
        _integer_equal(data["calibration_pairs_per_episode"], 1, "contract observations per episode")
        _require(sampling["independent_unit"] == data["calibration_independent_unit"]
                 == "one independently sampled episode", "independent unit differs from the scientific contract")
        _require(sampling["sampling_law"]
                 == "independent episodes with replacement, independent of development",
                 "unsupported sampling premise")
        _require(sampling["shared_calibration_across_candidates"] is True
                 and data["calibration_is_shared_across_candidates"] is True,
                 "shared calibration differs from the scientific contract")
        _require(sampling["retain_all_occurrences"] is True
                 and data["retain_repeated_histories_and_triples"] is True,
                 "calibration duplicates must be retained")
        _require(sampling["single_dependent_trajectory"] is False
                 and data["single_dependent_trajectory_used_for_primary_certificate"] is False,
                 "the iid protocol cannot certify a dependent-trajectory design")
        _require(sampling["calibration_randomness_selected_before_outcomes"] is True
                 and data["calibration_stream_selected_before_outcomes"] is True,
                 "calibration randomness must be chosen before outcomes")
        per = _fraction(confidence["per_candidate_failure_probability"], "per-candidate alpha")
        total = _fraction(confidence["total_failure_probability"], "total alpha")
        coverage = _fraction(confidence["simultaneous_coverage"], "simultaneous coverage")
        _require(per == _fraction(cert["per_candidate_failure_probability"], "contract per-candidate alpha")
                 == Fraction(1, 60), "per-candidate allocation differs from the scientific contract")
        _require(total == _fraction(cert["total_failure_probability"], "contract total alpha")
                 == Fraction(1, 20) and 3 * per == total,
                 "family confidence allocation differs from the scientific contract")
        _require(coverage == _fraction(contract["claim_scope"]["simultaneous_confidence"],
                                       "contract simultaneous coverage")
                 == 1 - total == Fraction(19, 20), "simultaneous coverage differs from the scientific contract")
        _integer_equal(confidence["upper_statements"], 3, "upper statement count")
        _integer_equal(endpoint["confidence_log_argument"], 60, "confidence logarithm argument")
        _require(confidence["between_candidate_independence_required"] is False
                 and cert["between_candidate_endpoint_independence_required"] is False
                 and confidence["allocation_reassigned_after_outcomes"] is False,
                 "unsupported joint-confidence or outcome-dependent allocation rule")
        _require(_fraction(confidence["numerical_failure_probability"], "numerical alpha") == 0
                 and _fraction(data["dependence_failure_allowance"], "dependence allowance") == 0,
                 "this protocol has no additional numerical or dependence allocation")
        for name in ("target_range", "prediction_range", "loss_range"):
            _require(loss[name] == contract["claim_scope"][name] == [0, 1],
                     f"{name} differs from the bounded-loss premise")
        _require(loss["name"] == "ordinary_scalar_squared_error"
                 and contract["claim_scope"]["loss"] == "squared_error"
                 and loss["variance_normalization"] is False
                 and contract["claim_scope"]["empirical_variance_normalization"] is False,
                 "loss definition differs from the scientific contract")
        _require(loss["empirical_mean_is_exact"] is True
                 and loss["empirical_mean_rounding"] == "none"
                 and loss["further_clipping_during_evaluation"] is False,
                 "exact represented-value loss arithmetic is required")
        _require(logarithm["algorithm_id"] == "range_reduced_atanh_exact_directed_dyadic_v1"
                 and logarithm["precision_bits_schedule"] == [128, 256, 512]
                 and all(type(value) is int for value in logarithm["precision_bits_schedule"])
                 and logarithm["series_term_count"] == "M=b at working precision b bits",
                 "verified logarithm algorithm or precision schedule changed")
        _require(root["algorithm_id"] == "exact_mean_verified_interval_bisection_v1",
                 "unsupported upper-root algorithm")
        _require(_fraction(root["target_bracket_width"], "root width") == Fraction(1, 2 ** 80),
                 "root width differs from 2^-80")
        for value, expected, name in (
            (root["target_bracket_width_power_of_two"], -80, "root-width exponent"),
            (root["maximum_successful_bisections"], 80, "bisection limit"),
            (limits["maximum_precision_bits"], 512, "precision limit"),
            (limits["maximum_series_terms_per_logarithm"], 512, "series limit"),
            (limits["maximum_root_trials_per_candidate"], 80, "root trial limit"),
            (limits["maximum_precision_attempts_per_trial"], 3, "refinement limit"),
            (limits["maximum_root_interval_evaluations_per_candidate"], 240, "root evaluation limit"),
            (limits["maximum_integer_bit_length"], 32768, "integer-size limit"),
            (limits["maximum_series_term_iterations_for_family_and_checker"], 1000000,
             "family series-work limit"),
            (limits["exact_empirical_risk_denominator_exponent_bound"], 2160,
             "empirical denominator exponent bound"),
            (reporting["decimal_places"], 18, "outward decimal precision"),
        ):
            _integer_equal(value, expected, name)
        _require(root["maximum_precision_does_not_imply_success"] is True
                 and limits["fallback"] == "certificate_unavailable"
                 and limits["automatic_looser_endpoint_or_replacement_sample"] is False
                 and protocol["independent_numerical_check"]["failed_check_action"]
                 == "certificate_unavailable", "unverified numerical fallback is prohibited")
        _require(_fraction(reporting["primary_tolerance"], "tolerance")
                 == _fraction(contract["claim_scope"]["primary_mse_tolerance"], "contract tolerance")
                 == Fraction(1, 10), "primary tolerance differs from the scientific contract")
        _require(set(reporting["statuses"]) == set(cert["statuses"])
                 == {"requirement_met", "unresolved", "certificate_unavailable"},
                 "certificate statuses differ from the scientific contract")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed calibration protocol or scientific contract") from error


def load_calibration_protocol() -> dict:
    """Read the exact packaged resource and verify contract consistency."""
    contents = files("kahm_demo").joinpath("calibration_protocol.json").read_bytes()
    if hashlib.sha256(contents).hexdigest() != CALIBRATION_PROTOCOL_SHA256:
        raise ValueError("the fixed calibration protocol resource has changed")
    protocol = json.loads(contents)
    _validate_protocol(protocol, validate_contract())
    return protocol


def calibration_protocol_sha256() -> str:
    """Return the fixed resource identity; this function does not read or write."""
    return CALIBRATION_PROTOCOL_SHA256


__all__ = ["CALIBRATION_PROTOCOL_SCHEMA", "CALIBRATION_PROTOCOL_SHA256",
           "load_calibration_protocol", "calibration_protocol_sha256"]

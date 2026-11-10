"""Verified numerical upper endpoints for the fixed calibration protocol.

The search and direct endpoint checker use exact rational arithmetic and the
declared logarithm enclosures. They neither inspect observations nor establish
the iid or frozen-predictor premises. The statistical statement is conditional
on development under those premises, with the unchanged per-candidate allocation
1/60. Numerical availability and requirement acceptance are not additional
conditioning events, and an unavailable candidate receives no reported endpoint.
"""

from __future__ import annotations

from fractions import Fraction
import math
import re

from .calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol
from .verified_arithmetic import (
    NumericalFailure, RationalInterval, WorkBudget,
    confidence_log_interval, kl_interval,
)


BOUND_SCHEMA = "kahm.risk_upper_bound.v1"
VERIFICATION_SCHEMA = "kahm.risk_upper_bound_verification.v1"
ALGORITHM_ID = "exact_mean_verified_interval_bisection_v1"
PRECISIONS = (128, 256, 512)
TARGET_WIDTH = Fraction(1, 2 ** 80)
TOLERANCE = Fraction(1, 10)
MAXIMUM_BISECTIONS = 80
MAXIMUM_EVALUATIONS = 240
MAXIMUM_Q_DENOMINATOR_EXPONENT = 2160
_DECIMAL_SCALE = 10 ** 18
_STATUSES = {"requirement_met", "unresolved", "certificate_unavailable"}
_RECORD_FIELDS = {
    "schema", "protocol_sha256", "algorithm_id", "empirical_mean", "status",
    "endpoint", "endpoint_decimal_18", "tolerance", "bracket", "search",
    "independent_check", "failure", "work",
}
_WORK_FIELDS = {"maximum_series_terms", "maximum_integer_bits",
                "series_terms", "maximum_integer_bits_seen"}


def _require(condition, code, message):
    if not condition:
        raise NumericalFailure(code, message)


def _failure(error):
    return {"code": error.code if isinstance(error, NumericalFailure) else "invalid_numerical_record",
            "message": str(error) or type(error).__name__}


def _fraction_record(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _parse_fraction(record, budget):
    _require(type(record) is dict and set(record) == {"numerator", "denominator"},
             "invalid_fraction_record", "an exact fraction needs precisely two integer strings")
    numerator, denominator = record["numerator"], record["denominator"]
    _require(type(numerator) is str and type(denominator) is str
             and re.fullmatch(r"0|-?[1-9][0-9]*", numerator) is not None
             and re.fullmatch(r"[1-9][0-9]*", denominator) is not None,
             "invalid_fraction_record", "fraction strings must be canonical decimal integers")
    # Actual protocol witnesses are far smaller. Reject oversized input text
    # before asking Python to convert it, independently of its process-wide
    # decimal-conversion limit.
    _require(len(numerator.lstrip("-")) <= 10000 and len(denominator) <= 10000,
             "integer_bit_limit", "fraction record exceeds the declared integer-size envelope")
    try:
        n, d = int(numerator), int(denominator)
    except ValueError as error:
        raise NumericalFailure("invalid_fraction_record", "fraction integer conversion failed") from error
    budget.check_int(n)
    budget.check_int(d)
    value = budget.check_fraction(Fraction(n, d))
    _require(_fraction_record(value) == record, "invalid_fraction_record",
             "fraction record must be reduced with a positive denominator")
    return value


def _add(left, right, budget):
    """Check intermediate integer products as well as the reduced result."""
    common = math.gcd(left.denominator, right.denominator)
    left_factor = left.denominator // common
    right_factor = right.denominator // common
    first = budget.check_int(left.numerator * right_factor)
    second = budget.check_int(right.numerator * left_factor)
    numerator = budget.check_int(first + second)
    denominator = budget.check_int(left_factor * right.denominator)
    return budget.check_fraction(Fraction(numerator, denominator))


def _subtract(left, right, budget):
    return _add(left, -right, budget)


def _scale(value, integer, budget):
    numerator = budget.check_int(value.numerator * integer)
    return budget.check_fraction(Fraction(numerator, value.denominator))


def _midpoint(lower, upper, budget):
    total = _add(lower, upper, budget)
    denominator = budget.check_int(2 * total.denominator)
    return budget.check_fraction(Fraction(total.numerator, denominator))


def _validate_q(q, budget=None):
    _require(type(q) is Fraction, "unsupported_empirical_mean", "empirical mean must be an exact Fraction")
    _require(0 <= q <= 1, "empirical_mean_out_of_range", "empirical mean must lie in [0,1]")
    denominator = q.denominator
    _require(denominator & (denominator - 1) == 0, "nondyadic_empirical_mean",
             "the fixed binary64-loss protocol requires a dyadic empirical mean")
    _require(denominator.bit_length() - 1 <= MAXIMUM_Q_DENOMINATOR_EXPONENT,
             "empirical_mean_denominator_limit", "empirical denominator exponent exceeds 2160")
    if budget is not None:
        budget.check_fraction(q)


def _work_snapshot(budget):
    return {name: getattr(budget, name) for name in _WORK_FIELDS}


def _work_record(before, budget):
    after = _work_snapshot(budget)
    return {"before": before, "after": after,
            "series_terms_used": after["series_terms"] - before["series_terms"]}


def _validate_work(record, *, available, failure_code=None):
    _require(type(record) is dict and set(record) == {"before", "after", "series_terms_used"},
             "invalid_work_record", "invalid numerical work record")
    for state in (record["before"], record["after"]):
        _require(type(state) is dict and set(state) == _WORK_FIELDS
                 and all(type(value) is int and value >= 0 for value in state.values()),
                 "invalid_work_record", "work counters and limits must be nonnegative integers")
        _require(state["maximum_series_terms"] <= 1000000
                 and 1 <= state["maximum_integer_bits"] <= 32768
                 and state["series_terms"] <= state["maximum_series_terms"],
                 "invalid_work_record", "work record exceeds the fixed limits")
        if available:
            _require(state["maximum_integer_bits_seen"] <= state["maximum_integer_bits"],
                     "invalid_work_record", "an available bound cannot have exceeded its integer-size limit")
        elif state["maximum_integer_bits_seen"] > state["maximum_integer_bits"]:
            _require(failure_code == "integer_bit_limit", "invalid_work_record",
                     "an observed integer-size violation requires its corresponding unavailable reason")
    before, after = record["before"], record["after"]
    _require(all(before[name] == after[name] for name in ("maximum_series_terms", "maximum_integer_bits"))
             and before["series_terms"] <= after["series_terms"]
             and before["maximum_integer_bits_seen"] <= after["maximum_integer_bits_seen"]
             and type(record["series_terms_used"]) is int
             and record["series_terms_used"] == after["series_terms"] - before["series_terms"],
             "invalid_work_record", "work limits or cumulative counters are inconsistent")


def _interval_record(interval):
    return {"low": _fraction_record(interval.low), "high": _fraction_record(interval.high)}


def _parse_interval(record, budget):
    _require(type(record) is dict and set(record) == {"low", "high"},
             "invalid_interval_witness", "interval witness needs low and high fractions")
    low, high = (_parse_fraction(record[name], budget) for name in ("low", "high"))
    _require(low <= high, "invalid_interval_witness", "interval endpoints are reversed")
    return RationalInterval(low, high)


def _analytic_witness(kind, point):
    return {"kind": kind, "point": _fraction_record(point)}


def _evaluate(q, point, bits, budget):
    divergence = kl_interval(q, point, bits, budget)
    confidence = confidence_log_interval(bits, budget)
    low = _subtract(_scale(divergence.low, 4096, budget), confidence.high, budget)
    high = _subtract(_scale(divergence.high, 4096, budget), confidence.low, budget)
    _require(low <= high, "invalid_interval_witness", "computed root interval is reversed")
    return {"kind": "interval", "point": _fraction_record(point), "precision_bits": bits,
            "kl_interval": _interval_record(divergence),
            "confidence_log_interval": _interval_record(confidence),
            "f_interval": _interval_record(RationalInterval(low, high))}


def _parse_witness(witness, point, budget):
    _require(type(witness) is dict and witness.get("kind") == "interval"
             and set(witness) == {"kind", "point", "precision_bits", "kl_interval",
                                  "confidence_log_interval", "f_interval"},
             "invalid_interval_witness", "invalid interval comparison witness")
    _require(_parse_fraction(witness["point"], budget) == point,
             "witness_point_mismatch", "interval witness belongs to a different endpoint")
    bits = witness["precision_bits"]
    _require(type(bits) is int and bits in PRECISIONS, "unsupported_precision",
             "witness precision is outside the fixed schedule")
    divergence = _parse_interval(witness["kl_interval"], budget)
    confidence = _parse_interval(witness["confidence_log_interval"], budget)
    residual = _parse_interval(witness["f_interval"], budget)
    _require(confidence.low > 0, "invalid_interval_witness", "log(60) witness must have a positive lower bound")
    expected_low = _subtract(_scale(divergence.low, 4096, budget), confidence.high, budget)
    expected_high = _subtract(_scale(divergence.high, 4096, budget), confidence.low, budget)
    _require((residual.low, residual.high) == (expected_low, expected_high),
             "invalid_interval_witness", "F witness does not follow from its KL and log(60) intervals")
    return residual


def _move_from_interval(interval):
    # The protocol gives the upper move precedence when both equalities hold.
    if interval.low >= 0:
        return "upper"
    if interval.high <= 0:
        return "lower"
    return "none"


def _bracket_record(lower, upper, lower_witness, upper_witness):
    return {"lower": _fraction_record(lower), "upper": _fraction_record(upper),
            "width": _fraction_record(upper - lower),
            "lower_witness": lower_witness, "upper_witness": upper_witness}


def _parse_bracket(q, record, budget, *, require_width):
    _require(type(record) is dict and set(record) ==
             {"lower", "upper", "width", "lower_witness", "upper_witness"},
             "invalid_bracket", "invalid saved endpoint bracket")
    lower, upper, width = (_parse_fraction(record[name], budget) for name in ("lower", "upper", "width"))
    _require(0 <= q <= lower <= upper <= 1 and width == _subtract(upper, lower, budget),
             "invalid_bracket", "saved bracket fails its exact order or width identity")
    if require_width:
        _require(width <= TARGET_WIDTH, "bracket_too_wide", "saved bracket is wider than 2^-80")
    if lower == q:
        _require(record["lower_witness"] == _analytic_witness("analytic_empirical_mean", q),
                 "invalid_lower_witness", "L=q requires its analytic lower-side witness")
    else:
        _require(q < lower < 1, "invalid_lower_witness", "nonanalytic lower endpoint must lie in (q,1)")
        _require(_parse_witness(record["lower_witness"], lower, budget).high <= 0,
                 "invalid_lower_witness", "saved lower endpoint inequality is not proved")
    if upper == 1:
        _require(record["upper_witness"] == _analytic_witness("analytic_unit_boundary", Fraction(1)),
                 "invalid_upper_witness", "H=1 requires its analytic upper-side witness")
    else:
        _require(q < upper < 1, "invalid_upper_witness", "nonanalytic upper endpoint must lie in (q,1)")
        _require(_parse_witness(record["upper_witness"], upper, budget).low >= 0,
                 "invalid_upper_witness", "saved upper endpoint inequality is not proved")
    return lower, upper


def _empty_check():
    return {"status": "not_attempted", "lower_witness": None, "upper_witness": None,
            "endpoint_evaluations": 0, "failure": None}


def _direct_endpoint_check(q, bracket, budget, check):
    """Check only saved endpoints; never call or replay the root search."""
    check["status"] = "running"
    budget.clear_log_caches()
    budget.check_int(0)  # A failed shared resource budget remains failed.
    lower, upper = _parse_bracket(q, bracket, budget, require_width=True)
    for side, point, analytic in (("lower", lower, lower == q), ("upper", upper, upper == 1)):
        saved = bracket[f"{side}_witness"]
        if analytic:
            fresh = saved.copy()
        else:
            check["endpoint_evaluations"] += 1
            fresh = _evaluate(q, point, saved["precision_bits"], budget)
            _require(fresh == saved, "endpoint_witness_mismatch",
                     f"fresh {side} endpoint enclosure differs from its saved witness")
            residual = _parse_witness(fresh, point, budget)
            proved = residual.high <= 0 if side == "lower" else residual.low >= 0
            _require(proved, f"invalid_{side}_witness", f"fresh {side} endpoint inequality is not proved")
        check[f"{side}_witness"] = fresh
    check["status"] = "passed"
    return upper


def _outward_decimal_18(value, budget):
    numerator = budget.check_int(value.numerator * _DECIMAL_SCALE)
    ceiling = budget.check_int(-(-numerator // value.denominator))
    whole, fraction = divmod(ceiling, _DECIMAL_SCALE)
    return f"{whole}.{fraction:018d}"


def _decision(value, budget):
    return "requirement_met" if budget.check_int(10 * value.numerator) <= value.denominator else "unresolved"


def _validate_failure(record):
    _require(type(record) is dict and set(record) == {"code", "message"}
             and type(record["code"]) is str and re.fullmatch(r"[a-z][a-z0-9_]*", record["code"]) is not None
             and type(record["message"]) is str and bool(record["message"]),
             "invalid_failure_record", "unavailable diagnostics need a reason code and message")


def _validate_trace(q, record, bracket, budget, *, available):
    _require(type(record) is dict and set(record) == {"successful_bisections", "interval_evaluations", "trace"},
             "invalid_search_trace", "invalid root-search diagnostics")
    successes, evaluations, trace = record["successful_bisections"], record["interval_evaluations"], record["trace"]
    _require(type(successes) is int and 0 <= successes <= MAXIMUM_BISECTIONS
             and type(evaluations) is int and 0 <= evaluations <= MAXIMUM_EVALUATIONS
             and type(trace) is list and len(trace) <= MAXIMUM_BISECTIONS,
             "invalid_search_trace", "root-search work exceeds its declared limits")
    lower, upper = q, Fraction(1)
    low_proof = _analytic_witness("analytic_empirical_mean", q)
    high_proof = _analytic_witness("analytic_unit_boundary", Fraction(1))
    counted_successes = counted_evaluations = 0
    for index, trial in enumerate(trace, 1):
        _require(type(trial) is dict and set(trial) ==
                 {"trial_index", "lower_before", "upper_before", "midpoint", "attempts", "move"},
                 "invalid_search_trace", "invalid bisection-trial fields")
        _require(type(trial["trial_index"]) is int and trial["trial_index"] == index
                 and _parse_fraction(trial["lower_before"], budget) == lower
                 and _parse_fraction(trial["upper_before"], budget) == upper
                 and _subtract(upper, lower, budget) > TARGET_WIDTH,
                 "invalid_search_trace", "trial order or preceding bracket is inconsistent")
        point = _midpoint(lower, upper, budget)
        _require(_parse_fraction(trial["midpoint"], budget) == point,
                 "invalid_search_trace", "trial does not use its exact bracket midpoint")
        attempts = trial["attempts"]
        _require(type(attempts) is list and 1 <= len(attempts) <= len(PRECISIONS),
                 "invalid_search_trace", "trial needs one to three bounded precision attempts")
        decision = "none"
        for attempt_index, witness in enumerate(attempts):
            counted_evaluations += 1
            _require(type(witness) is dict and witness.get("precision_bits") == PRECISIONS[attempt_index],
                     "invalid_search_trace", "precision attempts do not follow the fixed schedule")
            if witness.get("kind") == "evaluation_failure":
                _require(not available and set(witness) == {"kind", "point", "precision_bits", "failure"}
                         and attempt_index == len(attempts) - 1 and index == len(trace)
                         and _parse_fraction(witness["point"], budget) == point,
                         "invalid_search_trace", "an arithmetic failure must terminate an unavailable search")
                _validate_failure(witness["failure"])
                decision = "none"
            else:
                decision = _move_from_interval(_parse_witness(witness, point, budget))
                if attempt_index < len(attempts) - 1:
                    _require(decision == "none", "invalid_search_trace",
                             "precision was increased after an inequality had already been proved")
        _require(trial["move"] == decision, "invalid_search_trace", "recorded bracket move differs from its witness")
        if decision == "none":
            _require(not available and index == len(trace), "invalid_search_trace",
                     "an unresolved comparison must terminate an unavailable search")
            final = attempts[-1]
            _require(final.get("kind") == "evaluation_failure" or len(attempts) == len(PRECISIONS),
                     "invalid_search_trace", "an ambiguous comparison stopped before the final precision")
        elif decision == "upper":
            upper, high_proof = point, attempts[-1]
            counted_successes += 1
        else:
            lower, low_proof = point, attempts[-1]
            counted_successes += 1
    _require((successes, evaluations) == (counted_successes, counted_evaluations),
             "invalid_search_trace", "search counters do not equal the retained attempts")
    if bracket is not None:
        _require(bracket == _bracket_record(lower, upper, low_proof, high_proof),
                 "invalid_search_trace", "saved final bracket differs from its retained trace")
    else:
        _require(not available and not trace, "invalid_search_trace", "a searched certificate must retain its diagnostic bracket")


def _minimum_variable_log_terms(q, point, bits):
    """A cache-independent lower bound, derived without evaluating logarithms.

    A non-power-of-two argument needs its full b-term range-reduced series.
    Arguments 1, powers of two, and the separately cached constant 60 can need
    no new variable-series terms. Constant-cache setup is handled separately.
    """
    if q == point:
        return 0
    if q == 0:
        arguments = (1 - point,)
    elif q == 1:
        arguments = (point,)
    else:
        arguments = (q / point, (1 - q) / (1 - point))
    count = 0
    for argument in arguments:
        numerator, denominator = argument.numerator, argument.denominator
        power_of_two = (numerator > 0 and numerator & (numerator - 1) == 0
                        and denominator & (denominator - 1) == 0)
        if not power_of_two and argument != 60:
            count += bits
    return count


def _validate_minimum_work(q, record):
    """Reject impossible historical diagnostics without reconstructing a search.

    Exact historical cache contents are not replayed. The bound counts required
    variable series and, for completed independent endpoint evaluations, the
    constant series that followed the checker's mandatory cache clear.
    """
    minimum_terms = minimum_size = 0

    def saved_sizes(value):
        if type(value) is dict:
            if set(value) == {"numerator", "denominator"}:
                return max(int(value["numerator"]).bit_length(), int(value["denominator"]).bit_length())
            return max((saved_sizes(child) for child in value.values()), default=0)
        if type(value) is list:
            return max((saved_sizes(child) for child in value), default=0)
        return 0

    for trial in record["search"]["trace"]:
        point = Fraction(int(trial["midpoint"]["numerator"]), int(trial["midpoint"]["denominator"]))
        for witness in trial["attempts"]:
            if witness["kind"] == "interval":
                bits = witness["precision_bits"]
                minimum_terms += _minimum_variable_log_terms(q, point, bits)
                minimum_size = max(minimum_size, bits + 1, saved_sizes(witness))
    checked_precisions = set()
    for side in ("lower", "upper"):
        witness = record["independent_check"][f"{side}_witness"]
        if type(witness) is dict and witness.get("kind") == "interval":
            bits = witness["precision_bits"]
            point = Fraction(int(witness["point"]["numerator"]), int(witness["point"]["denominator"]))
            minimum_terms += _minimum_variable_log_terms(q, point, bits)
            checked_precisions.add(bits)
            minimum_size = max(minimum_size, bits + 1, saved_sizes(witness))
    minimum_terms += 2 * sum(checked_precisions)  # Fresh log(2) and log(60).
    if record["status"] != "certificate_unavailable":
        endpoint_numerator = int(record["endpoint"]["numerator"])
        minimum_size = max(minimum_size, saved_sizes(record["empirical_mean"]),
                           saved_sizes(record["bracket"]), saved_sizes(record["endpoint"]),
                           (endpoint_numerator * _DECIMAL_SCALE).bit_length(),
                           (10 * endpoint_numerator).bit_length())
    work = record["work"]
    _require(work["series_terms_used"] >= minimum_terms
             and work["after"]["maximum_integer_bits_seen"] >= minimum_size,
             "invalid_work_record", "recorded work is smaller than the retained numerical evidence requires")


def certify_upper_bound(q: Fraction, budget: WorkBudget) -> dict:
    """Run the fixed bounded search and a fresh, separate endpoint check.

    One WorkBudget must be shared across the three candidates. Failures preserve
    diagnostics, leave the endpoint null, and do not substitute a looser bound.
    """
    if not isinstance(budget, WorkBudget):
        raise TypeError("budget must be a WorkBudget shared by the calibrated family")
    before = _work_snapshot(budget)
    record = {"schema": BOUND_SCHEMA, "protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
              "algorithm_id": ALGORITHM_ID, "empirical_mean": None,
              "status": "certificate_unavailable", "endpoint": None, "endpoint_decimal_18": None,
              "tolerance": _fraction_record(TOLERANCE), "bracket": None,
              "search": {"successful_bisections": 0, "interval_evaluations": 0, "trace": []},
              "independent_check": _empty_check(), "failure": None, "work": None}
    lower = upper = low_proof = high_proof = None
    try:
        load_calibration_protocol()
        # Preserve the exact supported input even when an earlier candidate
        # exhausted the shared family budget. Its unavailable record must still
        # bind to the independently recomputed empirical mean.
        _validate_q(q)
        record["empirical_mean"] = _fraction_record(q)
        budget.check_fraction(q)
        lower, upper = q, Fraction(1)
        low_proof = _analytic_witness("analytic_empirical_mean", q)
        high_proof = _analytic_witness("analytic_unit_boundary", Fraction(1))
        search = record["search"]
        while _subtract(upper, lower, budget) > TARGET_WIDTH:
            _require(search["successful_bisections"] < MAXIMUM_BISECTIONS,
                     "root_trial_limit", "root search reached the fixed 80-trial limit")
            point = _midpoint(lower, upper, budget)
            trial = {"trial_index": len(search["trace"]) + 1,
                     "lower_before": _fraction_record(lower), "upper_before": _fraction_record(upper),
                     "midpoint": _fraction_record(point), "attempts": [], "move": "none"}
            search["trace"].append(trial)
            for bits in PRECISIONS:
                search["interval_evaluations"] += 1
                try:
                    witness = _evaluate(q, point, bits, budget)
                except NumericalFailure as error:
                    trial["attempts"].append({"kind": "evaluation_failure", "point": _fraction_record(point),
                                              "precision_bits": bits, "failure": _failure(error)})
                    raise
                trial["attempts"].append(witness)
                decision = _move_from_interval(_parse_witness(witness, point, budget))
                if decision == "upper":
                    upper, high_proof = point, witness
                elif decision == "lower":
                    lower, low_proof = point, witness
                else:
                    continue
                trial["move"] = decision
                search["successful_bisections"] += 1
                break
            else:
                raise NumericalFailure("ambiguous_root_comparison",
                                       "the same midpoint remained unresolved at 128, 256, and 512 bits")
        record["bracket"] = _bracket_record(lower, upper, low_proof, high_proof)
        endpoint = _direct_endpoint_check(q, record["bracket"], budget, record["independent_check"])
        record["endpoint_decimal_18"] = _outward_decimal_18(endpoint, budget)
        record["status"] = _decision(endpoint, budget)
        record["endpoint"] = _fraction_record(endpoint)
    except (NumericalFailure, ValueError, ArithmeticError, TypeError) as error:
        record["failure"] = _failure(error)
        record["status"] = "certificate_unavailable"
        record["endpoint"] = record["endpoint_decimal_18"] = None
        if lower is not None:
            record["bracket"] = _bracket_record(lower, upper, low_proof, high_proof)
        if record["independent_check"]["status"] == "running":
            record["independent_check"]["status"] = "failed"
            record["independent_check"]["failure"] = _failure(error)
    record["work"] = _work_record(before, budget)
    return record


def _validate_saved_check(check, bracket, *, available, failure):
    _require(type(check) is dict and set(check) ==
             {"status", "lower_witness", "upper_witness", "endpoint_evaluations", "failure"}
             and type(check["endpoint_evaluations"]) is int and 0 <= check["endpoint_evaluations"] <= 2,
             "invalid_independent_check", "invalid independent-check diagnostics")
    if available:
        _require(check["status"] == "passed" and check["failure"] is None
                 and check["lower_witness"] == bracket["lower_witness"]
                 and check["upper_witness"] == bracket["upper_witness"],
                 "invalid_independent_check", "available endpoint lacks matching direct-check evidence")
    elif check["status"] == "not_attempted":
        _require(check == _empty_check(), "invalid_independent_check", "unattempted check has unexpected evidence")
    elif check["status"] == "failed":
        _validate_failure(check["failure"])
        _require(check["failure"] == failure and bracket is not None,
                 "invalid_independent_check", "failed direct check must retain its bracket and the reported failure")
        for side in ("lower", "upper"):
            proof = check[f"{side}_witness"]
            _require(proof is None or proof == bracket[f"{side}_witness"],
                     "invalid_independent_check", "partial direct-check evidence differs from the retained bracket")
        _require(check["upper_witness"] is None or check["lower_witness"] is not None,
                 "invalid_independent_check", "upper-side evidence cannot precede the lower-side check")
    elif check["status"] == "passed":
        _require(bracket is not None and check["failure"] is None
                 and check["lower_witness"] == bracket["lower_witness"]
                 and check["upper_witness"] == bracket["upper_witness"],
                 "invalid_independent_check", "direct-check diagnostics differ from the retained bracket")
    else:
        raise NumericalFailure("invalid_independent_check", "unsupported saved direct-check status")


def verify_upper_bound(q: Fraction, record: dict, budget: WorkBudget) -> dict:
    """Validate a record and freshly check its two endpoints without searching.

    A consistent unavailable record remains verified=False with endpoint=None.
    Trace checks verify its exact bookkeeping; only final endpoint witnesses
    are recomputed numerically, using fresh logarithm caches and the same bounded
    arithmetic. This function never calls ``certify_upper_bound``.
    """
    if not isinstance(budget, WorkBudget):
        raise TypeError("budget must be a WorkBudget")
    before = _work_snapshot(budget)
    result = {"schema": VERIFICATION_SCHEMA, "protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
              "status": "certificate_unavailable", "verified": False,
              "unavailable_record_consistent": False, "empirical_mean": None,
              "endpoint": None, "endpoint_decimal_18": None, "failure": None,
              "numerical_check": _empty_check(), "work": None}
    try:
        load_calibration_protocol()
        _validate_q(q, budget)
        result["empirical_mean"] = _fraction_record(q)
        _require(type(record) is dict and set(record) == _RECORD_FIELDS,
                 "invalid_certificate_record", "upper-bound record has missing or unexpected fields")
        _require(record["schema"] == BOUND_SCHEMA and record["protocol_sha256"] == CALIBRATION_PROTOCOL_SHA256
                 and record["algorithm_id"] == ALGORITHM_ID and record["status"] in _STATUSES,
                 "invalid_certificate_record", "record schema, algorithm, protocol, or status is unsupported")
        _require(_parse_fraction(record["empirical_mean"], budget) == q,
                 "empirical_mean_mismatch", "saved empirical mean differs from the recomputed exact mean")
        _require(_parse_fraction(record["tolerance"], budget) == TOLERANCE,
                 "tolerance_mismatch", "saved tolerance differs from the fixed 1/10 requirement")
        available = record["status"] != "certificate_unavailable"
        if not available:
            _validate_failure(record["failure"])
        _validate_work(record["work"], available=available,
                       failure_code=None if available else record["failure"]["code"])
        bracket = record["bracket"]
        if bracket is not None:
            _parse_bracket(q, bracket, budget, require_width=available)
        else:
            _require(not available, "invalid_bracket", "available endpoint has no bracket")
        _validate_trace(q, record["search"], bracket, budget, available=available)
        _validate_saved_check(record["independent_check"], bracket, available=available,
                              failure=record["failure"])
        _validate_minimum_work(q, record)
        if not available:
            _require(record["endpoint"] is None and record["endpoint_decimal_18"] is None,
                     "invalid_unavailable_record", "unavailable candidate must not report an endpoint")
            trace = record["search"]["trace"]
            if trace and trace[-1]["move"] == "none":
                last = trace[-1]["attempts"][-1]
                if last["kind"] == "evaluation_failure":
                    _require(record["failure"] == last["failure"], "invalid_failure_record",
                             "reported failure differs from the terminal arithmetic attempt")
                else:
                    _require(record["failure"]["code"] == "ambiguous_root_comparison",
                             "invalid_failure_record", "unresolved final interval needs the ambiguity reason")
            result["failure"] = record["failure"].copy()
            result["unavailable_record_consistent"] = True
        else:
            _require(record["failure"] is None, "invalid_certificate_record", "available endpoint has a failure record")
            endpoint = _parse_fraction(record["endpoint"], budget)
            _require(endpoint == _parse_fraction(bracket["upper"], budget),
                     "endpoint_mismatch", "reported endpoint differs from the saved upper bracket")
            _require(record["status"] == _decision(endpoint, budget),
                     "decision_mismatch", "requirement decision differs from the exact rational comparison")
            decimal = _outward_decimal_18(endpoint, budget)
            _require(type(record["endpoint_decimal_18"]) is str and record["endpoint_decimal_18"] == decimal,
                     "display_mismatch", "reported decimal is not the exact outward 18-place ceiling")
            checked_endpoint = _direct_endpoint_check(q, bracket, budget, result["numerical_check"])
            _require(checked_endpoint == endpoint and result["numerical_check"] == record["independent_check"],
                     "independent_check_mismatch", "fresh endpoint checks differ from the saved direct-check evidence")
            result.update(status=record["status"], verified=True,
                          endpoint=_fraction_record(endpoint), endpoint_decimal_18=decimal)
    except (NumericalFailure, ValueError, ArithmeticError, TypeError, KeyError) as error:
        result["failure"] = _failure(error)
        if result["numerical_check"]["status"] == "running":
            result["numerical_check"]["status"] = "failed"
            result["numerical_check"]["failure"] = _failure(error)
    result["work"] = _work_record(before, budget)
    return result


__all__ = ["BOUND_SCHEMA", "VERIFICATION_SCHEMA", "ALGORITHM_ID",
           "certify_upper_bound", "verify_upper_bound"]

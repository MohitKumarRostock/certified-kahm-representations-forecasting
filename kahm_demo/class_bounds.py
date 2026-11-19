"""Verified binary-KL endpoints for the separate class-risk protocol.

The input is an exact mean of quantities in [0, 1]. This module accepts general
rational means and confidence allocations; it does not inherit the historical
three-predictor calibration protocol. Only the existing directed logarithm
arithmetic and its resource ceilings are reused. A numerical certificate does
not establish the sampling, independence, or optimization-lower-bound premises.

Both inverse branches are bracketed with exact rational arithmetic. A fresh
checker evaluates only the final endpoints, never repeats the search. The
checker establishes the numerical statement stored in the record; callers must
bind q, n, alpha and side to their independently checked statistical protocol.
"""

from __future__ import annotations

from fractions import Fraction
import math
import re

from .verified_arithmetic import (
    MAXIMUM_INTEGER_BITS, MAXIMUM_SERIES_TERMS, PRECISION_BITS,
    NumericalFailure, RationalInterval, WorkBudget, kl_interval, log_interval,
)


SCHEMA = "kahm.class_mean_endpoint.v1"
VERIFICATION_SCHEMA = "kahm.class_mean_endpoint_verification.v1"
ALGORITHM = "general_rational_kl_bracket_v1"
TARGET_WIDTH = Fraction(1, 2 ** 80)
MAXIMUM_BISECTIONS = 80
MAXIMUM_EVALUATIONS = MAXIMUM_BISECTIONS * len(PRECISION_BITS)
_DECIMAL_SCALE = 10 ** 18
_WORK_FIELDS = (
    "maximum_series_terms", "maximum_integer_bits", "series_terms",
    "maximum_integer_bits_seen",
)
_FIELDS = {
    "schema", "algorithm", "q", "n", "alpha", "side", "status",
    "endpoint", "endpoint_decimal_18", "bracket", "search",
    "independent_check", "failure", "work",
}


def _require(condition, code, message):
    if not condition:
        raise NumericalFailure(code, message)


def fraction_record(value: Fraction) -> dict:
    """Serialize an exact rational without conversion through a float."""
    _require(type(value) is Fraction, "invalid_fraction", "an exact Fraction is required")
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def parse_fraction(record: dict, budget: WorkBudget | None = None) -> Fraction:
    """Read a canonical reduced fraction, with bounded textual input."""
    _require(type(record) is dict and set(record) == {"numerator", "denominator"},
             "invalid_fraction_record", "a fraction needs numerator and denominator")
    num, den = record["numerator"], record["denominator"]
    _require(type(num) is str and type(den) is str
             and len(num) <= 10000 and len(den) <= 10000
             and re.fullmatch(r"0|-?[1-9][0-9]*", num) is not None
             and re.fullmatch(r"[1-9][0-9]*", den) is not None,
             "invalid_fraction_record", "fraction fields must be canonical integer strings")
    try:
        numerator, denominator = int(num), int(den)
    except ValueError as error:
        raise NumericalFailure("invalid_fraction_record", "fraction integer conversion failed") from error
    _require(numerator.bit_length() <= MAXIMUM_INTEGER_BITS
             and denominator.bit_length() <= MAXIMUM_INTEGER_BITS,
             "integer_bit_limit", "fraction operands exceed the arithmetic ceiling")
    if budget is not None:
        budget.check_int(numerator)
        budget.check_int(denominator)
    value = Fraction(numerator, denominator)
    _require(fraction_record(value) == record, "invalid_fraction_record", "fraction must be reduced")
    return value


def _add(left, right, budget):
    budget.check_fraction(left)
    budget.check_fraction(right)
    common = math.gcd(left.denominator, right.denominator)
    a, b = left.denominator // common, right.denominator // common
    numerator = budget.check_int(budget.check_int(left.numerator * b)
                                 + budget.check_int(right.numerator * a))
    denominator = budget.check_int(left.denominator * b)
    return budget.check_fraction(Fraction(numerator, denominator))


def _scale(value, factor, budget):
    budget.check_fraction(value)
    budget.check_int(factor)
    return budget.check_fraction(Fraction(budget.check_int(value.numerator * factor), value.denominator))


def _difference(left, right, budget):
    return _add(left, _scale(right, -1, budget), budget)


def _midpoint(left, right, budget):
    total = _add(left, right, budget)
    return budget.check_fraction(Fraction(total.numerator, budget.check_int(2 * total.denominator)))


def _le(left, right, budget):
    return (budget.check_int(left.numerator * right.denominator)
            <= budget.check_int(right.numerator * left.denominator))


def _validate_inputs(q, n, alpha, side, budget):
    _require(type(q) is Fraction and type(alpha) is Fraction,
             "invalid_fraction", "q and alpha must be exact Fractions")
    budget.check_fraction(q)
    budget.check_fraction(alpha)
    _require(0 <= q <= 1, "invalid_mean", "q must lie in [0, 1]")
    _require(type(n) is int and n >= 1, "invalid_sample_count", "n must be a positive integer")
    budget.check_int(n)
    _require(0 < alpha < 1, "invalid_allocation", "alpha must lie in (0, 1)")
    _require(type(side) is str and side in ("lower", "upper"),
             "invalid_side", "side must be lower or upper")


def _snapshot(budget):
    return {name: getattr(budget, name) for name in _WORK_FIELDS}


def _work(before, budget):
    after = _snapshot(budget)
    return {"before": before, "after": after,
            "series_terms_used": after["series_terms"] - before["series_terms"]}


def _failure(error):
    return {"code": error.code if isinstance(error, NumericalFailure) else "invalid_numerical_record",
            "message": str(error) or type(error).__name__}


def _interval_record(interval):
    return {"low": fraction_record(interval.low), "high": fraction_record(interval.high)}


def _analytic(q, point, side):
    if (side == "lower" and q == 0) or (side == "upper" and q == 1):
        reason = "degenerate_root"
    elif point == q:
        reason = "empirical_mean"
    elif (side == "lower" and point == 0) or (side == "upper" and point == 1):
        reason = "infinite_kl_boundary"
    else:
        return None
    return {"kind": "analytic", "point": fraction_record(point), "reason": reason}


def _evaluate(q, point, n, alpha, bits, budget, confidence_cache):
    divergence = kl_interval(q, point, bits, budget)
    if bits not in confidence_cache:
        inverse_alpha = budget.check_fraction(Fraction(alpha.denominator, alpha.numerator))
        confidence_cache[bits] = log_interval(inverse_alpha, bits, budget)
    confidence = confidence_cache[bits]
    residual = RationalInterval(
        _difference(_scale(divergence.low, n, budget), confidence.high, budget),
        _difference(_scale(divergence.high, n, budget), confidence.low, budget),
    )
    witness = {"kind": "interval", "point": fraction_record(point), "precision_bits": bits,
               "kl": _interval_record(divergence),
               "log_inverse_alpha": _interval_record(confidence),
               "residual": _interval_record(residual)}
    return witness, residual


def _decimal(value, side, budget):
    scaled = budget.check_int(value.numerator * _DECIMAL_SCALE)
    integer = scaled // value.denominator
    if side == "upper" and scaled % value.denominator:
        integer = budget.check_int(integer + 1)
    whole, part = divmod(integer, _DECIMAL_SCALE)
    return f"{whole}.{part:018d}"


def _empty_check():
    return {"status": "not_attempted", "endpoint_evaluations": 0,
            "lower_witness": None, "upper_witness": None, "failure": None}


def _check_bracket(q, n, alpha, side, bracket, budget, check=None):
    """Fresh endpoint-only verification, with a new confidence cache."""
    if check is None:
        check = _empty_check()
    check["status"] = "running"
    _require(type(bracket) is dict and set(bracket) ==
             {"lower", "upper", "lower_witness", "upper_witness"},
             "invalid_bracket", "bracket fields are incomplete or unexpected")
    lower = parse_fraction(bracket["lower"], budget)
    upper = parse_fraction(bracket["upper"], budget)
    valid_domain = (0 <= lower <= upper <= q) if side == "lower" else (q <= lower <= upper <= 1)
    _require(valid_domain, "invalid_bracket", "bracket is outside its inverse branch")
    _require(_le(_difference(upper, lower, budget), TARGET_WIDTH, budget),
             "invalid_bracket_width", "bracket is wider than 2^-80")
    budget.clear_log_caches()
    budget.check_int(0)
    cache = {}
    for name, point in (("lower", lower), ("upper", upper)):
        saved = bracket[name + "_witness"]
        fresh = _analytic(q, point, side)
        if fresh is None:
            _require(type(saved) is dict and saved.get("precision_bits") in PRECISION_BITS
                     and type(saved.get("precision_bits")) is int,
                     "invalid_endpoint_witness", "endpoint needs a supported precision")
            check["endpoint_evaluations"] += 1
            fresh, residual = _evaluate(q, point, n, alpha, saved["precision_bits"], budget, cache)
            # F is increasing on the upper branch and decreasing on the lower.
            outside = (name == "lower") == (side == "lower")
            proved = residual.low >= 0 if outside else residual.high <= 0
            _require(proved, "endpoint_direction_unproved", "endpoint residual does not prove its direction")
        _require(saved == fresh, "endpoint_witness_mismatch", "fresh endpoint evidence differs from the record")
        check[name + "_witness"] = fresh
    check["status"] = "passed"
    return lower if side == "lower" else upper, check


def bounded_mean_endpoint(q: Fraction, n: int, alpha: Fraction, side: str,
                          *, budget: WorkBudget | None = None) -> dict:
    """Construct a conservative inverse-KL endpoint or retain unavailability.

    The default work budget is per endpoint. An explicitly shared WorkBudget
    makes its failures sticky across the caller's family. This routine does not
    choose or reallocate alpha and does not decide a scientific risk threshold.
    """
    if budget is None:
        budget = WorkBudget()
    if type(budget) is not WorkBudget:
        raise TypeError("budget must be a WorkBudget")
    before = _snapshot(budget)
    record = {"schema": SCHEMA, "algorithm": ALGORITHM, "q": None, "n": None,
              "alpha": None, "side": None, "status": "certificate_unavailable",
              "endpoint": None, "endpoint_decimal_18": None, "bracket": None,
              "search": {"successful_bisections": 0, "interval_evaluations": 0,
                         "attempts": []},
              "independent_check": _empty_check(), "failure": None, "work": None}
    lower = upper = low_witness = high_witness = None
    try:
        # Bind valid inputs before touching a possibly exhausted shared budget.
        input_budget = WorkBudget(maximum_series_terms=0)
        _validate_inputs(q, n, alpha, side, input_budget)
        record.update(q=fraction_record(q), n=n, alpha=fraction_record(alpha), side=side)
        _validate_inputs(q, n, alpha, side, budget)
        lower, upper = (Fraction(0), q) if side == "lower" else (q, Fraction(1))
        low_witness, high_witness = _analytic(q, lower, side), _analytic(q, upper, side)
        cache = {}
        search = record["search"]
        while not _le(_difference(upper, lower, budget), TARGET_WIDTH, budget):
            _require(search["successful_bisections"] < MAXIMUM_BISECTIONS,
                     "root_trial_limit", "root search reached the 80-bisection limit")
            point = _midpoint(lower, upper, budget)
            for bits in PRECISION_BITS:
                search["interval_evaluations"] += 1
                attempt = {"point": fraction_record(point), "precision_bits": bits, "move": "failed"}
                search["attempts"].append(attempt)
                witness, residual = _evaluate(q, point, n, alpha, bits, budget, cache)
                if residual.low >= 0:
                    move = "lower" if side == "lower" else "upper"
                elif residual.high <= 0:
                    move = "upper" if side == "lower" else "lower"
                else:
                    attempt["move"] = "ambiguous"
                    continue
                attempt["move"] = move
                if move == "lower":
                    lower, low_witness = point, witness
                else:
                    upper, high_witness = point, witness
                search["successful_bisections"] += 1
                break
            else:
                raise NumericalFailure("ambiguous_root_comparison",
                                       "the midpoint remained ambiguous at every declared precision")
        bracket = {"lower": fraction_record(lower), "upper": fraction_record(upper),
                   "lower_witness": low_witness, "upper_witness": high_witness}
        record["bracket"] = bracket
        endpoint, check = _check_bracket(q, n, alpha, side, bracket, budget,
                                         record["independent_check"])
        record.update(status="available", endpoint=fraction_record(endpoint),
                      endpoint_decimal_18=_decimal(endpoint, side, budget), independent_check=check)
    except (NumericalFailure, ValueError, ArithmeticError, TypeError) as error:
        record.update(status="certificate_unavailable", endpoint=None,
                      endpoint_decimal_18=None, failure=_failure(error))
        if lower is not None:
            record["bracket"] = {"lower": fraction_record(lower), "upper": fraction_record(upper),
                                 "lower_witness": low_witness, "upper_witness": high_witness}
        if record["independent_check"]["status"] == "running":
            record["independent_check"].update(status="failed", failure=_failure(error))
    record["work"] = _work(before, budget)
    return record


def _validate_work(record, available, failure):
    _require(type(record) is dict and set(record) == {"before", "after", "series_terms_used"},
             "invalid_work_record", "work record fields are invalid")
    for state in (record["before"], record["after"]):
        _require(type(state) is dict and set(state) == set(_WORK_FIELDS)
                 and all(type(v) is int and v >= 0 for v in state.values()),
                 "invalid_work_record", "work counters must be nonnegative integers")
        _require(state["maximum_series_terms"] <= MAXIMUM_SERIES_TERMS
                 and 1 <= state["maximum_integer_bits"] <= MAXIMUM_INTEGER_BITS
                 and state["series_terms"] <= state["maximum_series_terms"],
                 "invalid_work_record", "work record exceeds the arithmetic ceilings")
        if state["maximum_integer_bits_seen"] > state["maximum_integer_bits"]:
            _require(not available and failure["code"] == "integer_bit_limit",
                     "invalid_work_record", "integer-size overrun must be recorded as unavailable")
    before, after = record["before"], record["after"]
    _require(all(before[name] == after[name] for name in _WORK_FIELDS[:2])
             and before["series_terms"] <= after["series_terms"]
             and before["maximum_integer_bits_seen"] <= after["maximum_integer_bits_seen"]
             and type(record["series_terms_used"]) is int
             and record["series_terms_used"] == after["series_terms"] - before["series_terms"],
             "invalid_work_record", "work limits or counter differences are inconsistent")


def _validate_search(record, q, side, bracket, available, budget):
    _require(type(record) is dict and set(record) ==
             {"successful_bisections", "interval_evaluations", "attempts"},
             "invalid_search_record", "search fields are invalid")
    count, evaluations, attempts = (record[k] for k in
                                   ("successful_bisections", "interval_evaluations", "attempts"))
    _require(type(count) is int and 0 <= count <= MAXIMUM_BISECTIONS
             and type(evaluations) is int and 0 <= evaluations <= MAXIMUM_EVALUATIONS
             and type(attempts) is list and len(attempts) == evaluations,
             "invalid_search_record", "search counters exceed their limits or disagree")
    lower, upper = (Fraction(0), q) if side == "lower" else (q, Fraction(1))
    successful = precision_index = 0
    for index, attempt in enumerate(attempts):
        _require(type(attempt) is dict and set(attempt) == {"point", "precision_bits", "move"},
                 "invalid_search_record", "invalid search attempt")
        _require(not _le(_difference(upper, lower, budget), TARGET_WIDTH, budget),
                 "invalid_search_record", "search continued after reaching target width")
        point = _midpoint(lower, upper, budget)
        _require(parse_fraction(attempt["point"], budget) == point
                 and type(attempt["precision_bits"]) is int
                 and attempt["precision_bits"] == PRECISION_BITS[precision_index],
                 "invalid_search_record", "search midpoint or precision schedule is inconsistent")
        move = attempt["move"]
        if move in ("lower", "upper"):
            if move == "lower":
                lower = point
            else:
                upper = point
            successful += 1
            precision_index = 0
        elif move == "ambiguous":
            precision_index += 1
            _require(precision_index < len(PRECISION_BITS) or
                     (not available and index == len(attempts) - 1),
                     "invalid_search_record", "ambiguity at maximum precision must terminate")
        else:
            _require(move == "failed" and not available and index == len(attempts) - 1,
                     "invalid_search_record", "failed evaluation must terminate an unavailable search")
    _require(successful == count, "invalid_search_record", "successful bisection count disagrees")
    if bracket is not None:
        _require(type(bracket) is dict and parse_fraction(bracket["lower"], budget) == lower
                 and parse_fraction(bracket["upper"], budget) == upper,
                 "invalid_search_record", "final bracket differs from recorded search decisions")
    else:
        _require(not available and not attempts, "invalid_search_record", "searched result needs a bracket")


def verify_mean_endpoint(record: dict, *, budget: WorkBudget | None = None,
                         expected_q: Fraction | None = None, expected_n: int | None = None,
                         expected_alpha: Fraction | None = None,
                         expected_side: str | None = None) -> dict:
    """Check stored input bindings, bookkeeping and endpoint signs, without search.

    A consistent unavailable result is retained as verified=False and
    unavailable_record_consistent=True; it never supplies a numerical bound.
    Optional expected_* arguments bind the record to an independently checked
    statistical input. Hash/provenance checks remain the caller's responsibility.
    """
    if budget is None:
        budget = WorkBudget()
    if type(budget) is not WorkBudget:
        raise TypeError("budget must be a WorkBudget")
    before = _snapshot(budget)
    result = {"schema": VERIFICATION_SCHEMA, "verified": False, "available": False,
              "unavailable_record_consistent": False, "endpoint": None,
              "failure": None, "numerical_check": _empty_check(), "work": None}
    try:
        _require(type(record) is dict and set(record) == _FIELDS,
                 "invalid_certificate_record", "certificate fields are incomplete or unexpected")
        _require(record["schema"] == SCHEMA and record["algorithm"] == ALGORITHM
                 and record["status"] in ("available", "certificate_unavailable"),
                 "invalid_certificate_record", "unsupported certificate schema, algorithm or status")
        q, n, alpha, side = (parse_fraction(record["q"], budget), record["n"],
                             parse_fraction(record["alpha"], budget), record["side"])
        _validate_inputs(q, n, alpha, side, budget)
        for name, actual, expected in (("q", q, expected_q), ("n", n, expected_n),
                                      ("alpha", alpha, expected_alpha), ("side", side, expected_side)):
            if expected is not None:
                _require(type(expected) is type(actual) and expected == actual,
                         "input_binding_mismatch", f"saved {name} differs from its independently checked value")
        available = record["status"] == "available"
        failure = record["failure"]
        if available:
            _require(failure is None, "invalid_failure_record", "available endpoint cannot record failure")
        else:
            _require(type(failure) is dict and set(failure) == {"code", "message"}
                     and type(failure["code"]) is str
                     and re.fullmatch(r"[a-z][a-z0-9_]*", failure["code"]) is not None
                     and type(failure["message"]) is str and bool(failure["message"]),
                     "invalid_failure_record", "unavailable endpoint needs a reason code and message")
        _validate_work(record["work"], available, failure)
        _validate_search(record["search"], q, side, record["bracket"], available, budget)
        if available:
            endpoint, check = _check_bracket(q, n, alpha, side, record["bracket"], budget,
                                             result["numerical_check"])
            _require(record["independent_check"] == check, "independent_check_mismatch",
                     "saved direct check differs from fresh endpoint verification")
            _require(parse_fraction(record["endpoint"], budget) == endpoint,
                     "endpoint_mismatch", "reported endpoint differs from its outward bracket end")
            _require(record["endpoint_decimal_18"] == _decimal(endpoint, side, budget),
                     "display_mismatch", "reported display does not use directed rounding")
            _require(record["work"]["series_terms_used"] >=
                     budget.series_terms - before["series_terms"],
                     "invalid_work_record", "construction work is less than its required fresh check")
            result.update(verified=True, available=True, endpoint=fraction_record(endpoint),
                          numerical_check=check)
        else:
            _require(record["endpoint"] is None and record["endpoint_decimal_18"] is None,
                     "invalid_unavailable_record", "unavailable record must not contain an endpoint")
            check = record["independent_check"]
            _require(type(check) is dict and set(check) == set(_empty_check())
                     and type(check["endpoint_evaluations"]) is int
                     and 0 <= check["endpoint_evaluations"] <= 2,
                     "invalid_independent_check", "unavailable record has invalid check diagnostics")
            if check["status"] == "not_attempted":
                _require(check == _empty_check(), "invalid_independent_check",
                         "unattempted check has unexpected evidence")
            else:
                _require(check["status"] in ("failed", "passed") and record["bracket"] is not None,
                         "invalid_independent_check", "a completed or failed check needs its bracket")
                _require(check["failure"] == (failure if check["status"] == "failed" else None),
                         "invalid_independent_check", "check failure differs from certificate failure")
                for name in ("lower_witness", "upper_witness"):
                    _require(check[name] == record["bracket"][name]
                             or (check["status"] == "failed" and check[name] is None),
                             "invalid_independent_check", "check evidence differs from its saved bracket")
            result.update(unavailable_record_consistent=True, failure=failure.copy())
    except (NumericalFailure, ValueError, ArithmeticError, TypeError, KeyError, IndexError) as error:
        result["failure"] = _failure(error)
        if result["numerical_check"]["status"] == "running":
            result["numerical_check"].update(status="failed", failure=_failure(error))
    result["work"] = _work(before, budget)
    return result


__all__ = ["SCHEMA", "VERIFICATION_SCHEMA", "ALGORITHM", "TARGET_WIDTH",
           "bounded_mean_endpoint", "verify_mean_endpoint", "fraction_record", "parse_fraction"]

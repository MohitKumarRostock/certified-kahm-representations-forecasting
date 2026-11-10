"""Exact logarithm and binary-KL enclosures under the frozen calibration protocol.

Only integer and Fraction arithmetic is used. At each supported precision b,
the range-reduced atanh series has exactly b terms, its powers and terms are
enclosed by directed integer division on the 2**b grid, and a rigorous positive
tail is added to the upper endpoint. Negative range-reduction exponents reverse
the multiplied log(2) interval. Zero-weight KL terms are never evaluated.

The two constant caches belong to one WorkBudget instance. Clearing them never
resets work counters; a resource-limit failure remains sticky. No predictor,
calibration data, empirical risk, root search, or oracle risk is handled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol


_protocol = load_calibration_protocol()
PRECISION_BITS = tuple(_protocol["verified_logarithm"]["precision_bits_schedule"])
MAXIMUM_SERIES_TERMS = _protocol["work_limits"]["maximum_series_term_iterations_for_family_and_checker"]
MAXIMUM_INTEGER_BITS = _protocol["work_limits"]["maximum_integer_bit_length"]
ARITHMETIC_PROTOCOL_SHA256 = CALIBRATION_PROTOCOL_SHA256
del _protocol


class NumericalFailure(ValueError):
    """A numerical certificate cannot be established under the fixed rules."""

    def __init__(self, code: str, message: str | None = None):
        if not isinstance(code, str) or not code:
            raise ValueError("a numerical failure needs a nonempty string code")
        self.code = code
        super().__init__(message if message is not None else code)


@dataclass(frozen=True, slots=True)
class RationalInterval:
    low: Fraction
    high: Fraction

    def __post_init__(self) -> None:
        if type(self.low) is not Fraction or type(self.high) is not Fraction:
            raise NumericalFailure("invalid_interval", "interval endpoints must be exact Fractions")
        if self.low > self.high:
            raise NumericalFailure("invalid_interval", "interval endpoints are out of order")


class WorkBudget:
    """One bounded family/checker calculation, including its constant caches.

    Limits may be stricter than the frozen protocol for testing, never larger.
    ``series_terms`` counts consumed iterations; a denied request is not added.
    ``maximum_integer_bits_seen`` includes a violating observed integer before
    the corresponding exception is raised. Limits and counters are read-only
    public properties. Private state is not a security boundary against callers.
    """

    __slots__ = (
        "_maximum_series_terms", "_maximum_integer_bits", "_series_terms",
        "_maximum_integer_bits_seen", "_failure", "_log2_cache", "_log60_cache",
    )

    def __init__(self, maximum_series_terms: int = MAXIMUM_SERIES_TERMS,
                 maximum_integer_bits: int = MAXIMUM_INTEGER_BITS):
        if (type(maximum_series_terms) is not int
                or not 0 <= maximum_series_terms <= MAXIMUM_SERIES_TERMS
                or type(maximum_integer_bits) is not int
                or not 1 <= maximum_integer_bits <= MAXIMUM_INTEGER_BITS):
            raise NumericalFailure("invalid_work_budget", "work limits must not exceed the frozen protocol")
        self._maximum_series_terms = maximum_series_terms
        self._maximum_integer_bits = maximum_integer_bits
        self._series_terms = 0
        self._maximum_integer_bits_seen = 0
        self._failure: tuple[str, str] | None = None
        self._log2_cache: dict[int, RationalInterval] = {}
        self._log60_cache: dict[int, RationalInterval] = {}

    @property
    def maximum_series_terms(self) -> int:
        return self._maximum_series_terms

    @property
    def maximum_integer_bits(self) -> int:
        return self._maximum_integer_bits

    @property
    def series_terms(self) -> int:
        return self._series_terms

    @property
    def maximum_integer_bits_seen(self) -> int:
        return self._maximum_integer_bits_seen

    def _active(self) -> None:
        if self._failure is not None:
            raise NumericalFailure(*self._failure)

    def _fail(self, code: str, message: str) -> None:
        self._failure = (code, message)
        raise NumericalFailure(code, message)

    def check_int(self, value: int) -> int:
        self._active()
        if type(value) is not int:
            raise NumericalFailure("invalid_integer", "integer arithmetic rejects booleans and nonintegers")
        size = value.bit_length()
        self._maximum_integer_bits_seen = max(self._maximum_integer_bits_seen, size)
        if size > self._maximum_integer_bits:
            self._fail("integer_bit_limit", f"observed {size}-bit integer exceeds the work limit")
        return value

    def check_fraction(self, value: Fraction) -> Fraction:
        self._active()
        if type(value) is not Fraction:
            raise NumericalFailure("invalid_fraction", "rational arithmetic requires an exact Fraction")
        self.check_int(value.numerator)
        self.check_int(value.denominator)
        return value

    def consume_terms(self, count: int = 1) -> None:
        self._active()
        if type(count) is not int or count < 0:
            raise NumericalFailure("invalid_work_count", "a consumed term count must be a nonnegative integer")
        self.check_int(count)
        proposed = self.check_int(self._series_terms + count)
        if proposed > self._maximum_series_terms:
            self._fail("series_work_limit", "series-term budget exhausted")
        self._series_terms = proposed

    def clear_log_caches(self) -> None:
        """Clear only constants; retain all counters, limits, and failures."""
        self._log2_cache.clear()
        self._log60_cache.clear()


def _prepare(bits: int, budget: WorkBudget) -> None:
    if type(budget) is not WorkBudget:
        raise NumericalFailure("invalid_work_budget", "a WorkBudget instance is required")
    budget._active()
    if type(bits) is not int or bits not in PRECISION_BITS:
        raise NumericalFailure("invalid_precision", "precision must be 128, 256, or 512 bits")
    budget.check_int(bits)


def _add_int(a: int, b: int, budget: WorkBudget) -> int:
    budget.check_int(a)
    budget.check_int(b)
    return budget.check_int(a + b)


def _mul_int(a: int, b: int, budget: WorkBudget) -> int:
    budget.check_int(a)
    budget.check_int(b)
    return budget.check_int(a * b)


def _positive_integer_power(base: int, exponent: int, budget: WorkBudget) -> int:
    """Exact powering with every multiplication explicitly budgeted."""
    budget.check_int(base)
    budget.check_int(exponent)
    if base < 1 or exponent < 0:
        raise NumericalFailure("arithmetic_invariant", "positive integer power received invalid arguments")
    result = budget.check_int(1)
    while exponent:
        if exponent & 1:
            result = _mul_int(result, base, budget)
        exponent = budget.check_int(exponent // 2)
        if exponent:
            base = _mul_int(base, base, budget)
    return result


def _quotients(numerator: int, denominator: int, budget: WorkBudget) -> tuple[int, int]:
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator <= 0:
        raise NumericalFailure("arithmetic_invariant", "directed division needs a positive denominator")
    quotient, remainder = divmod(numerator, denominator)
    budget.check_int(quotient)
    budget.check_int(remainder)
    return quotient, remainder


def _floor_div(numerator: int, denominator: int, budget: WorkBudget) -> int:
    return _quotients(numerator, denominator, budget)[0]


def _ceil_div(numerator: int, denominator: int, budget: WorkBudget) -> int:
    quotient, remainder = _quotients(numerator, denominator, budget)
    return _add_int(quotient, int(remainder != 0), budget)


def _fraction(numerator: int, denominator: int, budget: WorkBudget) -> Fraction:
    # Check the unreduced operands, not only their possibly tiny reduced ratio.
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator <= 0:
        raise NumericalFailure("arithmetic_invariant", "a rational denominator must be positive")
    return budget.check_fraction(Fraction(numerator, denominator))


def _add_fraction(a: Fraction, b: Fraction, budget: WorkBudget) -> Fraction:
    budget.check_fraction(a)
    budget.check_fraction(b)
    numerator = _add_int(_mul_int(a.numerator, b.denominator, budget),
                         _mul_int(b.numerator, a.denominator, budget), budget)
    denominator = _mul_int(a.denominator, b.denominator, budget)
    return _fraction(numerator, denominator, budget)


def _mul_fraction(a: Fraction, b: Fraction, budget: WorkBudget) -> Fraction:
    budget.check_fraction(a)
    budget.check_fraction(b)
    return _fraction(_mul_int(a.numerator, b.numerator, budget),
                     _mul_int(a.denominator, b.denominator, budget), budget)


def _divide_fraction(a: Fraction, b: Fraction, budget: WorkBudget) -> Fraction:
    budget.check_fraction(a)
    budget.check_fraction(b)
    if b <= 0:
        raise NumericalFailure("arithmetic_invariant", "this ratio needs a positive divisor")
    return _fraction(_mul_int(a.numerator, b.denominator, budget),
                     _mul_int(a.denominator, b.numerator, budget), budget)


def _negate(a: Fraction, budget: WorkBudget) -> Fraction:
    budget.check_fraction(a)
    return _fraction(budget.check_int(-a.numerator), a.denominator, budget)


def _checked_interval(low: Fraction, high: Fraction, budget: WorkBudget) -> RationalInterval:
    budget.check_fraction(low)
    budget.check_fraction(high)
    # Fraction ordering uses cross products; budget those intermediates too.
    left = _mul_int(low.numerator, high.denominator, budget)
    right = _mul_int(high.numerator, low.denominator, budget)
    if left > right:
        raise NumericalFailure("invalid_interval", "computed interval endpoints are out of order")
    return RationalInterval(low, high)


def _cached_interval(interval: RationalInterval, budget: WorkBudget) -> RationalInterval:
    # Cache reuse saves series work, not validation or budget checks.
    budget._active()
    if type(interval) is not RationalInterval:
        raise NumericalFailure("arithmetic_invariant", "invalid constant-cache interval")
    budget.check_fraction(interval.low)
    budget.check_fraction(interval.high)
    return interval


def _series_log_interval(z: Fraction, bits: int, budget: WorkBudget) -> RationalInterval:
    """Enclose log((1+z)/(1-z)) for 0<=z<=1/3 by the frozen recurrence."""
    budget.check_fraction(z)
    if z.numerator < 0 or _mul_int(3, z.numerator, budget) > z.denominator:
        raise NumericalFailure("arithmetic_invariant", "atanh-series argument is outside [0,1/3]")
    if z == 0:
        return _checked_interval(Fraction(0), Fraction(0), budget)
    denominator_grid = budget.check_int(1 << bits)
    scaled_z = _mul_int(denominator_grid, z.numerator, budget)
    power_low = _floor_div(scaled_z, z.denominator, budget)
    power_high = _ceil_div(scaled_z, z.denominator, budget)
    z_squared_numerator = _mul_int(z.numerator, z.numerator, budget)
    z_squared_denominator = _mul_int(z.denominator, z.denominator, budget)
    lower_sum = upper_sum = 0
    for j in range(bits):
        budget.consume_terms()
        odd = _add_int(_mul_int(2, j, budget), 1, budget)
        lower_term = _floor_div(_mul_int(2, power_low, budget), odd, budget)
        upper_term = _ceil_div(_mul_int(2, power_high, budget), odd, budget)
        lower_sum = _add_int(lower_sum, lower_term, budget)
        upper_sum = _add_int(upper_sum, upper_term, budget)
        if j + 1 < bits:
            power_low = _floor_div(_mul_int(power_low, z_squared_numerator, budget),
                                   z_squared_denominator, budget)
            power_high = _ceil_div(_mul_int(power_high, z_squared_numerator, budget),
                                   z_squared_denominator, budget)
    power_nine = _positive_integer_power(9, bits, budget)
    tail_denominator = _mul_int(
        _mul_int(4, _add_int(_mul_int(2, bits, budget), 1, budget), budget),
        power_nine, budget,
    )
    tail_grid_units = _ceil_div(_mul_int(3, denominator_grid, budget), tail_denominator, budget)
    upper_with_tail = _add_int(upper_sum, tail_grid_units, budget)
    return _checked_interval(_fraction(lower_sum, denominator_grid, budget),
                             _fraction(upper_with_tail, denominator_grid, budget), budget)


def _log2_interval(bits: int, budget: WorkBudget) -> RationalInterval:
    if bits in budget._log2_cache:
        return _cached_interval(budget._log2_cache[bits], budget)
    result = _series_log_interval(Fraction(1, 3), bits, budget)
    budget._log2_cache[bits] = result
    return result


def _range_reduce(x: Fraction, budget: WorkBudget) -> tuple[int, Fraction]:
    numerator, denominator = x.numerator, x.denominator
    exponent = budget.check_int(numerator.bit_length() - denominator.bit_length())

    def scaled_pair(a: int) -> tuple[int, int]:
        budget.check_int(a)
        if a >= 0:
            return numerator, budget.check_int(denominator << a)
        return budget.check_int(numerator << -a), denominator

    scaled_numerator, scaled_denominator = scaled_pair(exponent)
    if scaled_numerator < scaled_denominator:
        exponent = _add_int(exponent, -1, budget)
        scaled_numerator, scaled_denominator = scaled_pair(exponent)
    # Avoid an unnecessary doubling of a large denominator in the upper check.
    quotient, _ = _quotients(scaled_numerator, scaled_denominator, budget)
    if quotient != 1:
        raise NumericalFailure("arithmetic_invariant", "integer range reduction failed to give 1<=u<2")
    return exponent, _fraction(scaled_numerator, scaled_denominator, budget)


def _log_general(x: Fraction, bits: int, budget: WorkBudget) -> RationalInterval:
    exponent, reduced = _range_reduce(x, budget)
    z = _fraction(
        _add_int(reduced.numerator, -reduced.denominator, budget),
        _add_int(reduced.numerator, reduced.denominator, budget), budget,
    )
    unit = _series_log_interval(z, bits, budget)
    if exponent == 0:
        return unit
    log2 = _log2_interval(bits, budget)
    coefficient = _fraction(exponent, 1, budget)
    if exponent > 0:
        low2, high2 = log2.low, log2.high
    else:
        low2, high2 = log2.high, log2.low
    return _checked_interval(
        _add_fraction(unit.low, _mul_fraction(coefficient, low2, budget), budget),
        _add_fraction(unit.high, _mul_fraction(coefficient, high2, budget), budget), budget,
    )


def log_interval(x: Fraction, bits: int, budget: WorkBudget) -> RationalInterval:
    """Enclose the natural logarithm of a strictly positive exact rational."""
    _prepare(bits, budget)
    budget.check_fraction(x)
    if x <= 0:
        raise NumericalFailure("invalid_log_input", "the logarithm needs a positive rational argument")
    if x == 1:
        return _checked_interval(Fraction(0), Fraction(0), budget)
    if x == 2:
        return _log2_interval(bits, budget)
    if x == 60:
        return confidence_log_interval(bits, budget)
    return _log_general(x, bits, budget)


def confidence_log_interval(bits: int, budget: WorkBudget) -> RationalInterval:
    """Enclose log(60), caching only within this work budget and precision."""
    _prepare(bits, budget)
    x = budget.check_fraction(Fraction(60))
    if bits in budget._log60_cache:
        return _cached_interval(budget._log60_cache[bits], budget)
    result = _log_general(x, bits, budget)
    budget._log60_cache[bits] = result
    return result


def kl_interval(q: Fraction, p: Fraction, bits: int, budget: WorkBudget) -> RationalInterval:
    """Enclose finite binary kl(q||p), including its continuous finite limits.

    Infinite boundary cases raise ``infinite_kl_boundary``; the root/checker
    handles its analytic upper boundary p=1 without requesting such an interval.
    """
    _prepare(bits, budget)
    budget.check_fraction(q)
    budget.check_fraction(p)
    if not 0 <= q <= 1 or not 0 <= p <= 1:
        raise NumericalFailure("invalid_probability", "KL arguments must lie in [0,1]")
    if q == p:
        return _checked_interval(Fraction(0), Fraction(0), budget)
    if p == 0 or p == 1:
        raise NumericalFailure("infinite_kl_boundary", "the requested binary KL divergence is infinite")
    if q == 0:
        complement = _add_fraction(Fraction(1), _negate(p, budget), budget)
        logarithm = log_interval(complement, bits, budget)
        return _checked_interval(_negate(logarithm.high, budget),
                                 _negate(logarithm.low, budget), budget)
    if q == 1:
        logarithm = log_interval(p, bits, budget)
        return _checked_interval(_negate(logarithm.high, budget),
                                 _negate(logarithm.low, budget), budget)
    complement_q = _add_fraction(Fraction(1), _negate(q, budget), budget)
    complement_p = _add_fraction(Fraction(1), _negate(p, budget), budget)
    first = log_interval(_divide_fraction(q, p, budget), bits, budget)
    second = log_interval(_divide_fraction(complement_q, complement_p, budget), bits, budget)
    return _checked_interval(
        _add_fraction(_mul_fraction(q, first.low, budget),
                      _mul_fraction(complement_q, second.low, budget), budget),
        _add_fraction(_mul_fraction(q, first.high, budget),
                      _mul_fraction(complement_q, second.high, budget), budget), budget,
    )


__all__ = [
    "ARITHMETIC_PROTOCOL_SHA256", "PRECISION_BITS", "NumericalFailure",
    "WorkBudget", "RationalInterval", "log_interval", "confidence_log_interval",
    "kl_interval",
]

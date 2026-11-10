"""Exact enclosure, boundary, and resource checks without calibration data."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction
import unittest

from kahm_demo.calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol
from kahm_demo.verified_arithmetic import (
    ARITHMETIC_PROTOCOL_SHA256, PRECISION_BITS, NumericalFailure, RationalInterval,
    WorkBudget, confidence_log_interval, kl_interval, log_interval,
)


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _reference_log(value: Fraction) -> Fraction:
    # Decimal is confined to this independent test reference. Its precision is
    # much higher than the narrowest tested production interval (~150 digits).
    with localcontext() as context:
        context.prec = 240
        return Fraction(_decimal(value).ln())


def _reference_kl(q: Fraction, p: Fraction) -> Fraction:
    with localcontext() as context:
        context.prec = 420
        dq, dp = _decimal(q), _decimal(p)
        if q == p:
            return Fraction(0)
        if q == 0:
            return Fraction(-(Decimal(1) - dp).ln())
        if q == 1:
            return Fraction(-dp.ln())
        return Fraction(dq * (dq / dp).ln()
                        + (Decimal(1) - dq) * ((Decimal(1) - dq) / (Decimal(1) - dp)).ln())


def _exact_series_reference(z: Fraction, terms: int = 200) -> tuple[Fraction, Fraction]:
    # Independent exact rational partial sums, with no dyadic power or term
    # rounding. This gives a much narrower rigorous interval for small z.
    power = z
    total = Fraction(0)
    for j in range(terms):
        total += 2 * power / (2 * j + 1)
        power *= z * z
    tail = 2 * power / ((2 * terms + 1) * (1 - z * z))
    return total, total + tail


class VerifiedLogarithmTests(unittest.TestCase):
    def assertContains(self, interval, value):
        self.assertLessEqual(interval.low, value)
        self.assertGreaterEqual(interval.high, value)

    def test_implementation_uses_the_frozen_precision_and_budget_parameters(self):
        protocol = load_calibration_protocol()
        self.assertEqual(ARITHMETIC_PROTOCOL_SHA256, CALIBRATION_PROTOCOL_SHA256)
        self.assertEqual(PRECISION_BITS, tuple(protocol["verified_logarithm"]["precision_bits_schedule"]))
        budget = WorkBudget()
        self.assertEqual(budget.maximum_series_terms,
                         protocol["work_limits"]["maximum_series_term_iterations_for_family_and_checker"])
        self.assertEqual(budget.maximum_integer_bits,
                         protocol["work_limits"]["maximum_integer_bit_length"])

    def test_exact_zero_and_interval_immutability(self):
        budget = WorkBudget(maximum_series_terms=0)
        result = log_interval(Fraction(1), 128, budget)
        self.assertEqual(result, RationalInterval(Fraction(0), Fraction(0)))
        self.assertEqual(budget.series_terms, 0)
        with self.assertRaises(FrozenInstanceError):
            result.low = Fraction(-1)
        with self.assertRaises(NumericalFailure):
            RationalInterval(Fraction(1), Fraction(0))
        with self.assertRaises(NumericalFailure):
            RationalInterval(0, Fraction(0))

    def test_rigorous_independent_rational_series_enclosed_at_all_precisions(self):
        log2 = _exact_series_reference(Fraction(1, 3))
        log_three_halves = _exact_series_reference(Fraction(1, 5))
        for bits in PRECISION_BITS:
            with self.subTest(bits=bits):
                for x, reference in ((Fraction(2), log2), (Fraction(3, 2), log_three_halves)):
                    interval = log_interval(x, bits, WorkBudget())
                    self.assertContains(interval, reference[0])
                    self.assertContains(interval, reference[1])

    def test_exact_power_of_two_recombination_and_negative_exponents(self):
        for bits in PRECISION_BITS:
            budget = WorkBudget()
            log2 = log_interval(Fraction(2), bits, budget)
            for exponent in (-1074, -9, -1, 1, 7, 1023):
                x = (Fraction(1 << exponent) if exponent >= 0
                     else Fraction(1, 1 << -exponent))
                actual = log_interval(x, bits, budget)
                expected = (RationalInterval(exponent * log2.low, exponent * log2.high)
                            if exponent > 0 else
                            RationalInterval(exponent * log2.high, exponent * log2.low))
                self.assertEqual(actual, expected)
            self.assertEqual(budget.series_terms, bits)

    def test_range_reduction_across_both_power_boundaries_and_log60(self):
        values = (
            Fraction(1, 3), Fraction(7, 3), Fraction(99, 100), Fraction(100, 99),
            Fraction(255, 128), Fraction(257, 128), Fraction(255, 256),
            Fraction(257, 256), Fraction(60),
        )
        for bits in PRECISION_BITS:
            budget = WorkBudget()
            for x in values:
                with self.subTest(bits=bits, x=x):
                    interval = log_interval(x, bits, budget)
                    self.assertContains(interval, _reference_log(x))
                    self.assertGreater(interval.high - interval.low, 0)
        widths = []
        for bits in PRECISION_BITS:
            interval = confidence_log_interval(bits, WorkBudget())
            widths.append(interval.high - interval.low)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])

    def test_extreme_exact_binary64_ratios_and_empirical_precision(self):
        values = (
            Fraction.from_float(float.fromhex("0x0.0000000000001p-1022")),
            Fraction.from_float(float.fromhex("0x1.fffffffffffffp+1023")),
            Fraction.from_float(float.fromhex("0x1.0000000000001p+0")),
            Fraction.from_float(float.fromhex("0x1.fffffffffffffp-1")),
            Fraction(1, 1 << 2160),
        )
        for x in values:
            with self.subTest(x=x):
                budget = WorkBudget()
                actual = log_interval(x, 512, budget)
                self.assertContains(actual, _reference_log(x))
                self.assertLessEqual(budget.maximum_integer_bits_seen, budget.maximum_integer_bits)

    def test_rejects_unverified_inputs_and_unsupported_precision(self):
        for x in (0.5, float("nan"), float("inf"), True, 1, Decimal("0.5"), "1/2"):
            with self.subTest(x=x), self.assertRaises(NumericalFailure) as caught:
                log_interval(x, 128, WorkBudget())
            self.assertEqual(caught.exception.code, "invalid_fraction")
        for x in (Fraction(0), Fraction(-1)):
            with self.subTest(x=x), self.assertRaises(NumericalFailure) as caught:
                log_interval(x, 128, WorkBudget())
            self.assertEqual(caught.exception.code, "invalid_log_input")
        for bits in (64, 129, 1024, True, 128.0):
            with self.subTest(bits=bits), self.assertRaises(NumericalFailure) as caught:
                log_interval(Fraction(2), bits, WorkBudget())
            self.assertEqual(caught.exception.code, "invalid_precision")


class VerifiedKLTests(unittest.TestCase):
    def assertContains(self, interval, value):
        self.assertLessEqual(interval.low, value)
        self.assertGreaterEqual(interval.high, value)

    def test_finite_boundary_cases_and_negated_log_order(self):
        for bits in PRECISION_BITS:
            budget = WorkBudget()
            for q in (Fraction(0), Fraction(1), Fraction(3, 7)):
                self.assertEqual(kl_interval(q, q, bits, budget),
                                 RationalInterval(Fraction(0), Fraction(0)))
            p = Fraction(3, 8)
            complement_log = log_interval(1 - p, bits, budget)
            self.assertEqual(kl_interval(Fraction(0), p, bits, budget),
                             RationalInterval(-complement_log.high, -complement_log.low))
            probability_log = log_interval(p, bits, budget)
            self.assertEqual(kl_interval(Fraction(1), p, bits, budget),
                             RationalInterval(-probability_log.high, -probability_log.low))

    def test_general_kl_contains_high_precision_independent_reference(self):
        examples = ((Fraction(1, 10), Fraction(1, 5)),
                    (Fraction(3, 4), Fraction(1, 4)),
                    (Fraction(1, 2), Fraction(1, 2) + Fraction(1, 2 ** 70)),
                    (Fraction(0), Fraction(1, 4096)),
                    (Fraction(1), Fraction(63, 64)))
        for bits in PRECISION_BITS:
            budget = WorkBudget()
            for q, p in examples:
                with self.subTest(bits=bits, q=q, p=p):
                    interval = kl_interval(q, p, bits, budget)
                    self.assertContains(interval, _reference_kl(q, p))

    def test_extreme_exact_empirical_mean_ratio_remains_within_budget(self):
        q = Fraction(1, 2 ** 2160)
        p = (q + 1) / 2
        budget = WorkBudget()
        interval = kl_interval(q, p, 512, budget)
        # At 420 decimal digits the tiny q contribution is below the reference
        # resolution; at this precision it is also far below the wider verified
        # interval, so its high-precision limiting reference remains enclosed.
        self.assertContains(interval, _reference_kl(q, p))
        self.assertLess(budget.maximum_integer_bits_seen, budget.maximum_integer_bits)

    def test_infinite_and_out_of_range_kl_arguments_are_rejected(self):
        for q, p in ((Fraction(1, 2), Fraction(0)), (Fraction(1, 2), Fraction(1)),
                     (Fraction(0), Fraction(1)), (Fraction(1), Fraction(0))):
            with self.subTest(q=q, p=p), self.assertRaises(NumericalFailure) as caught:
                kl_interval(q, p, 128, WorkBudget())
            self.assertEqual(caught.exception.code, "infinite_kl_boundary")
        for q, p in ((Fraction(-1), Fraction(1, 2)), (Fraction(1, 2), Fraction(2))):
            with self.subTest(q=q, p=p), self.assertRaises(NumericalFailure) as caught:
                kl_interval(q, p, 128, WorkBudget())
            self.assertEqual(caught.exception.code, "invalid_probability")
        with self.assertRaises(NumericalFailure) as caught:
            kl_interval(0.1, Fraction(1, 5), 128, WorkBudget())
        self.assertEqual(caught.exception.code, "invalid_fraction")


class ArithmeticBudgetTests(unittest.TestCase):
    def test_integer_boundary_and_violating_size_are_recorded(self):
        budget = WorkBudget(maximum_integer_bits=8)
        self.assertEqual(budget.check_int(255), 255)
        self.assertEqual(budget.check_int(-255), -255)
        self.assertEqual(budget.check_fraction(Fraction(255, 127)), Fraction(255, 127))
        self.assertEqual(budget.maximum_integer_bits_seen, 8)
        with self.assertRaises(NumericalFailure) as caught:
            budget.check_int(256)
        self.assertEqual(caught.exception.code, "integer_bit_limit")
        self.assertEqual(budget.maximum_integer_bits_seen, 9)
        with self.assertRaises(NumericalFailure):
            budget.check_int(1)
        with self.assertRaises(AttributeError):
            budget.maximum_integer_bits = 100

    def test_term_limit_denies_without_increasing_consumed_count(self):
        budget = WorkBudget(maximum_series_terms=2)
        budget.consume_terms(2)
        self.assertEqual(budget.series_terms, 2)
        with self.assertRaises(NumericalFailure) as caught:
            budget.consume_terms()
        self.assertEqual(caught.exception.code, "series_work_limit")
        self.assertEqual(budget.series_terms, 2)
        budget.clear_log_caches()
        with self.assertRaises(NumericalFailure) as caught:
            budget.check_int(1)
        self.assertEqual(caught.exception.code, "series_work_limit")

    def test_constant_caches_are_per_budget_and_per_precision(self):
        first = WorkBudget()
        intervals = []
        for bits in PRECISION_BITS:
            interval = confidence_log_interval(bits, first)
            intervals.append(interval)
            count = first.series_terms
            self.assertIs(confidence_log_interval(bits, first), interval)
            self.assertIs(log_interval(Fraction(60), bits, first), interval)
            log_interval(Fraction(2), bits, first)
            self.assertEqual(first.series_terms, count)
        self.assertEqual(first.series_terms, 2 * sum(PRECISION_BITS))
        other = WorkBudget()
        self.assertEqual(confidence_log_interval(128, other), intervals[0])
        self.assertEqual(other.series_terms, 256)
        before = first.series_terms
        first.clear_log_caches()
        self.assertEqual(first.series_terms, before)
        self.assertEqual(confidence_log_interval(128, first), intervals[0])
        self.assertEqual(first.series_terms, before + 256)

    def test_cached_constants_cannot_bypass_a_failed_budget(self):
        budget = WorkBudget(maximum_series_terms=256)
        confidence_log_interval(128, budget)
        self.assertEqual(budget.series_terms, 256)
        confidence_log_interval(128, budget)  # Reuse itself consumes no terms.
        with self.assertRaises(NumericalFailure) as caught:
            log_interval(Fraction(3, 2), 128, budget)
        self.assertEqual(caught.exception.code, "series_work_limit")
        self.assertEqual(budget.series_terms, 256)
        with self.assertRaises(NumericalFailure) as caught:
            confidence_log_interval(128, budget)
        self.assertEqual(caught.exception.code, "series_work_limit")

    def test_intermediate_integer_growth_is_checked_before_any_series_term(self):
        # The input itself fits, but squaring the range-reduced z denominator
        # exceeds the smaller test budget. Checking only returned fractions
        # would miss this intermediate and incorrectly report success.
        x = Fraction((1 << 2000) + 1, (1 << 2000) - 1)
        budget = WorkBudget(maximum_integer_bits=3000)
        with self.assertRaises(NumericalFailure) as caught:
            log_interval(x, 128, budget)
        self.assertEqual(caught.exception.code, "integer_bit_limit")
        self.assertEqual(budget.maximum_integer_bits_seen, 4001)
        self.assertEqual(budget.series_terms, 0)

    def test_full_integer_limit_boundary_for_exact_power_of_two(self):
        budget = WorkBudget()
        actual = log_interval(Fraction(1, 1 << 32767), 128, budget)
        log2 = log_interval(Fraction(2), 128, budget)
        self.assertEqual(actual, RationalInterval(-32767 * log2.high, -32767 * log2.low))
        self.assertEqual(budget.maximum_integer_bits_seen, 32768)
        too_large = WorkBudget()
        with self.assertRaises(NumericalFailure) as caught:
            log_interval(Fraction(1, 1 << 32768), 128, too_large)
        self.assertEqual(caught.exception.code, "integer_bit_limit")
        self.assertEqual(too_large.maximum_integer_bits_seen, 32769)
        self.assertEqual(too_large.series_terms, 0)

    def test_budget_configuration_and_primitive_input_validation(self):
        for kwargs in ({"maximum_series_terms": -1}, {"maximum_series_terms": True},
                       {"maximum_series_terms": 1000001}, {"maximum_integer_bits": 0},
                       {"maximum_integer_bits": 32769}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NumericalFailure) as caught:
                WorkBudget(**kwargs)
            self.assertEqual(caught.exception.code, "invalid_work_budget")
        for method, value in (("check_int", True), ("check_fraction", 0.5),
                              ("consume_terms", -1), ("consume_terms", 1.0)):
            with self.subTest(method=method), self.assertRaises(NumericalFailure):
                getattr(WorkBudget(), method)(value)


if __name__ == "__main__":
    unittest.main()

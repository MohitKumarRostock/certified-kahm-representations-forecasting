"""Exact endpoint directions, independent checking, and bounded failure paths."""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import unittest
from unittest import mock

from kahm_demo import risk_bounds as bounds
from kahm_demo.risk_bounds import certify_upper_bound, verify_upper_bound
from kahm_demo.verified_arithmetic import NumericalFailure, RationalInterval, WorkBudget


def rational(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def decimal_reference_root(q):
    """Independent high-precision test reference; never used by production."""
    with localcontext() as context:
        context.prec = 110
        empirical = decimal_fraction(q)
        if q == 1:
            return Decimal(1)
        confidence = Decimal(60).ln() / Decimal(4096)
        if q == 0:
            return Decimal(1) - (-confidence).exp()
        low, high = empirical, Decimal(1)
        for _ in range(340):
            point = (low + high) / 2
            if point == low or point == high:
                break
            divergence = (empirical * (empirical / point).ln()
                          + (1 - empirical) * ((1 - empirical) / (1 - point)).ln())
            if divergence >= confidence:
                high = point
            else:
                low = point
        return (low + high) / 2


class RiskBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {
            q: certify_upper_bound(q, WorkBudget())
            for q in (Fraction(0), Fraction(1, 16), Fraction(1, 2), Fraction(1))
        }
        if any(record["status"] == "certificate_unavailable" for record in cls.records.values()):
            raise AssertionError("ordinary numerical test fixtures must be available")

    def test_brackets_enclose_independent_decimal_references(self):
        for q, record in self.records.items():
            with self.subTest(q=q), localcontext() as context:
                context.prec = 110
                lower = rational(record["bracket"]["lower"])
                upper = rational(record["bracket"]["upper"])
                reference = decimal_reference_root(q)
                self.assertLessEqual(decimal_fraction(lower), reference)
                self.assertLessEqual(reference, decimal_fraction(upper))
                self.assertLessEqual(upper - lower, Fraction(1, 2 ** 80))
                self.assertEqual(rational(record["endpoint"]), upper)
                self.assertGreaterEqual(Decimal(record["endpoint_decimal_18"]), decimal_fraction(upper))
                self.assertLess(Decimal(record["endpoint_decimal_18"]) - decimal_fraction(upper), Decimal("1e-18"))
                self.assertLessEqual(record["search"]["successful_bisections"], 80)
                self.assertLessEqual(record["search"]["interval_evaluations"], 240)

    def test_q_one_and_tiny_initial_width_use_analytic_boundaries_without_logs(self):
        for q in (Fraction(1), Fraction(1) - Fraction(1, 2 ** 100)):
            with self.subTest(q=q), \
                 mock.patch.object(bounds, "kl_interval", side_effect=AssertionError("unexpected KL call")), \
                 mock.patch.object(bounds, "confidence_log_interval", side_effect=AssertionError("unexpected logarithm")):
                record = certify_upper_bound(q, WorkBudget(maximum_series_terms=0))
                self.assertEqual(record["status"], "unresolved")
                self.assertEqual(rational(record["endpoint"]), 1)
                self.assertEqual(record["search"]["successful_bisections"], 0)
                self.assertEqual(record["work"]["series_terms_used"], 0)
                self.assertTrue(verify_upper_bound(q, record, WorkBudget()) ["verified"])

    def test_q_zero_never_evaluates_an_infinite_or_zero_log_boundary(self):
        real = bounds.kl_interval
        points = []

        def checked(q, point, bits, budget):
            self.assertEqual(q, 0)
            self.assertGreater(point, 0)
            self.assertLess(point, 1)
            points.append(point)
            return real(q, point, bits, budget)

        with mock.patch.object(bounds, "kl_interval", side_effect=checked):
            record = certify_upper_bound(Fraction(0), WorkBudget())
        self.assertEqual(record["status"], "requirement_met")
        self.assertEqual(record["search"]["successful_bisections"], 80)
        self.assertTrue(points)

    def test_actual_empirical_domain_and_denominator_limit(self):
        invalid = (
            (0.0, "unsupported_empirical_mean"), (0, "unsupported_empirical_mean"),
            (Fraction(-1, 2), "empirical_mean_out_of_range"),
            (Fraction(3, 2), "empirical_mean_out_of_range"),
            (Fraction(1, 10), "nondyadic_empirical_mean"),
            (Fraction(1, 2 ** 2161), "empirical_mean_denominator_limit"),
        )
        for q, reason in invalid:
            with self.subTest(q=str(q)):
                record = certify_upper_bound(q, WorkBudget())
                self.assertEqual(record["status"], "certificate_unavailable")
                self.assertEqual(record["failure"]["code"], reason)
                self.assertIsNone(record["endpoint"])
                self.assertIsNone(record["endpoint_decimal_18"])
        for q in (Fraction(1, 2 ** 2160), Fraction(1) - Fraction(1, 2 ** 2160)):
            with self.subTest(boundary=q == 1):
                record = certify_upper_bound(q, WorkBudget())
                self.assertNotEqual(record["status"], "certificate_unavailable", record["failure"])
                self.assertTrue(verify_upper_bound(q, record, WorkBudget())["verified"])

    def test_reporting_uses_exact_tolerance_and_integer_ceiling(self):
        below = Fraction(1, 10) - Fraction(1, 10 ** 40)
        equal = Fraction(1, 10)
        above = Fraction(1, 10) + Fraction(1, 10 ** 40)
        self.assertEqual(float(below), float(above))
        self.assertEqual(bounds._decision(below, WorkBudget()), "requirement_met")
        self.assertEqual(bounds._decision(equal, WorkBudget()), "requirement_met")
        self.assertEqual(bounds._decision(above, WorkBudget()), "unresolved")
        self.assertEqual(bounds._outward_decimal_18(below, WorkBudget()), "0.100000000000000000")
        self.assertEqual(bounds._outward_decimal_18(equal, WorkBudget()), "0.100000000000000000")
        self.assertEqual(bounds._outward_decimal_18(above, WorkBudget()), "0.100000000000000001")
        self.assertEqual(bounds._outward_decimal_18(Fraction(1, 3), WorkBudget()), "0.333333333333333334")
        self.assertEqual(bounds._outward_decimal_18(Fraction(0), WorkBudget()), "0.000000000000000000")
        self.assertEqual(bounds._outward_decimal_18(Fraction(1), WorkBudget()), "1.000000000000000000")

    def test_direct_checker_evaluates_endpoints_only_and_never_searches(self):
        q = Fraction(1, 16)
        record = self.records[q]
        real = bounds.kl_interval
        visited = []

        def track(empirical, point, bits, budget):
            visited.append((point, bits))
            return real(empirical, point, bits, budget)

        with mock.patch.object(bounds, "certify_upper_bound", side_effect=AssertionError("checker reran search")), \
             mock.patch.object(bounds, "kl_interval", side_effect=track):
            checked = verify_upper_bound(q, record, WorkBudget())
        self.assertTrue(checked["verified"], checked["failure"])
        self.assertEqual(visited, [
            (rational(record["bracket"][side]), record["bracket"][f"{side}_witness"]["precision_bits"])
            for side in ("lower", "upper")
        ])
        self.assertLess(checked["work"]["series_terms_used"], record["work"]["series_terms_used"])

    def test_near_root_ambiguity_keeps_the_same_midpoint_and_refines(self):
        q = Fraction(1, 16)
        point = rational(self.records[q]["search"]["trace"][-1]["midpoint"])
        real = bounds.kl_interval
        refined = []

        def widen(empirical, value, bits, budget):
            result = real(empirical, value, bits, budget)
            if value == point:
                refined.append(bits)
                if bits < 512:
                    confidence = bounds.confidence_log_interval(bits, budget)
                    return RationalInterval(min(result.low, confidence.low / 4096),
                                            max(result.high, confidence.high / 4096))
            return result

        with mock.patch.object(bounds, "kl_interval", side_effect=widen):
            record = certify_upper_bound(q, WorkBudget())
        self.assertNotEqual(record["status"], "certificate_unavailable", record["failure"])
        final = record["search"]["trace"][-1]
        self.assertEqual(rational(final["midpoint"]), point)
        self.assertEqual([witness["precision_bits"] for witness in final["attempts"]], [128, 256, 512])
        self.assertEqual(refined[:3], [128, 256, 512])
        self.assertEqual(record["search"]["successful_bisections"], 80)
        self.assertEqual(record["search"]["interval_evaluations"], 82)
        self.assertTrue(verify_upper_bound(q, record, WorkBudget())["verified"])

    def test_ambiguity_at_final_precision_reports_no_looser_endpoint(self):
        q = Fraction(1, 16)
        point = rational(self.records[q]["search"]["trace"][-1]["midpoint"])
        real = bounds.kl_interval

        def widen(empirical, value, bits, budget):
            result = real(empirical, value, bits, budget)
            if value == point:
                confidence = bounds.confidence_log_interval(bits, budget)
                return RationalInterval(min(result.low, confidence.low / 4096),
                                        max(result.high, confidence.high / 4096))
            return result

        with mock.patch.object(bounds, "kl_interval", side_effect=widen):
            record = certify_upper_bound(q, WorkBudget())
        self.assertEqual(record["status"], "certificate_unavailable")
        self.assertEqual(record["failure"]["code"], "ambiguous_root_comparison")
        self.assertEqual(record["search"]["successful_bisections"], 79)
        self.assertGreater(rational(record["bracket"]["width"]), Fraction(1, 2 ** 80))
        self.assertIsNone(record["endpoint"])
        self.assertIsNone(record["endpoint_decimal_18"])
        checked = verify_upper_bound(q, record, WorkBudget())
        self.assertFalse(checked["verified"])
        self.assertTrue(checked["unavailable_record_consistent"], checked["failure"])
        self.assertIsNone(checked["endpoint"])

    def test_all_240_allowed_root_evaluations_stay_bounded(self):
        q = Fraction(1, 16)
        real = bounds.kl_interval

        def widen(empirical, point, bits, budget):
            result = real(empirical, point, bits, budget)
            if bits < 512:
                confidence = bounds.confidence_log_interval(bits, budget)
                return RationalInterval(min(result.low, confidence.low / 4096),
                                        max(result.high, confidence.high / 4096))
            return result

        budget = WorkBudget()
        with mock.patch.object(bounds, "kl_interval", side_effect=widen):
            record = certify_upper_bound(q, budget)
        self.assertEqual(record["status"], "requirement_met", record["failure"])
        self.assertEqual(record["search"]["successful_bisections"], 80)
        self.assertEqual(record["search"]["interval_evaluations"], 240)
        self.assertLessEqual(budget.series_terms, 1000000)
        self.assertLessEqual(budget.maximum_integer_bits_seen, 32768)
        self.assertTrue(verify_upper_bound(q, record, WorkBudget())["verified"])

    def test_resource_failures_and_later_family_candidates_remain_unavailable(self):
        for options, code in (({"maximum_series_terms": 0}, "series_work_limit"),
                              ({"maximum_integer_bits": 64}, "integer_bit_limit")):
            budget = WorkBudget(**options)
            for q in (Fraction(1, 16), Fraction(1, 8), Fraction(1)):
                with self.subTest(options=options, q=q):
                    record = certify_upper_bound(q, budget)
                    self.assertEqual(record["status"], "certificate_unavailable")
                    self.assertEqual(record["failure"]["code"], code)
                    self.assertEqual(rational(record["empirical_mean"]), q)
                    self.assertIsNone(record["endpoint"])
                    checked = verify_upper_bound(q, record, WorkBudget())
                    self.assertTrue(checked["unavailable_record_consistent"], checked["failure"])
                    self.assertFalse(checked["verified"])
                    self.assertIsNone(checked["endpoint"])
            if code == "integer_bit_limit":
                self.assertGreater(budget.maximum_integer_bits_seen, budget.maximum_integer_bits)

    def test_direct_check_failure_does_not_issue_the_search_endpoint(self):
        q = Fraction(1, 16)
        real = bounds.kl_interval
        seen = set()

        def fail_on_recheck(empirical, point, bits, budget):
            key = (point, bits)
            if key in seen:
                raise NumericalFailure("forced_direct_check_failure", "injected independent-check failure")
            seen.add(key)
            return real(empirical, point, bits, budget)

        with mock.patch.object(bounds, "kl_interval", side_effect=fail_on_recheck):
            record = certify_upper_bound(q, WorkBudget())
        self.assertEqual(record["status"], "certificate_unavailable")
        self.assertEqual(record["failure"]["code"], "forced_direct_check_failure")
        self.assertEqual(record["independent_check"]["status"], "failed")
        self.assertLessEqual(rational(record["bracket"]["width"]), Fraction(1, 2 ** 80))
        self.assertIsNone(record["endpoint"])
        self.assertTrue(verify_upper_bound(q, record, WorkBudget())["unavailable_record_consistent"])

    def test_one_budget_accumulates_all_search_and_internal_check_work(self):
        budget = WorkBudget()
        previous = 0
        total = 0
        for q in (Fraction(0), Fraction(1, 32), Fraction(3, 16)):
            record = certify_upper_bound(q, budget)
            self.assertNotEqual(record["status"], "certificate_unavailable", record["failure"])
            self.assertEqual(record["work"]["before"]["series_terms"], previous)
            self.assertEqual(record["work"]["after"]["series_terms"], budget.series_terms)
            total += record["work"]["series_terms_used"]
            previous = budget.series_terms
        self.assertEqual(total, budget.series_terms)
        self.assertLessEqual(total, 1000000)

    def test_wrong_q_endpoint_display_status_and_fraction_encoding_are_rejected(self):
        q = Fraction(1, 16)
        original = self.records[q]
        self.assertFalse(verify_upper_bound(Fraction(1, 8), original, WorkBudget())["verified"])
        changes = (
            ("endpoint", lambda record: record.__setitem__("endpoint", {"numerator": "1", "denominator": "1"})),
            ("display", lambda record: record.__setitem__("endpoint_decimal_18", "0.000000000000000000")),
            ("decision", lambda record: record.__setitem__("status", "unresolved")),
            ("noncanonical numerator", lambda record: record["empirical_mean"].__setitem__("numerator", "01")),
            ("nonreduced fraction", lambda record: record.__setitem__("empirical_mean", {"numerator": "2", "denominator": "32"})),
            ("JSON numeric fraction", lambda record: record["endpoint"].__setitem__("numerator", 1)),
            ("count", lambda record: record["search"].__setitem__("interval_evaluations", 241)),
        )
        for name, change in changes:
            with self.subTest(case=name):
                record = copy.deepcopy(original)
                change(record)
                checked = verify_upper_bound(q, record, WorkBudget())
                self.assertFalse(checked["verified"])
                self.assertEqual(checked["status"], "certificate_unavailable")
                self.assertIsNone(checked["endpoint"])

    def test_direct_check_rejects_consistently_rewritten_endpoint_witness(self):
        q = Fraction(1, 16)
        record = copy.deepcopy(self.records[q])
        point = record["bracket"]["upper"]
        altered = copy.deepcopy(record["bracket"]["upper_witness"])
        # Widen only the upper sides, preserving the certified lower sign and
        # the algebraic F=4096*KL-log(60) relation. A direct re-evaluation must
        # still reject a witness not emitted by the prescribed logarithm recipe.
        delta = Fraction(1, 2 ** 128)
        high_kl = rational(altered["kl_interval"]["high"]) + delta
        high_f = rational(altered["f_interval"]["high"]) + 4096 * delta
        altered["kl_interval"]["high"] = bounds._fraction_record(high_kl)
        altered["f_interval"]["high"] = bounds._fraction_record(high_f)
        record["bracket"]["upper_witness"] = altered
        record["independent_check"]["upper_witness"] = copy.deepcopy(altered)
        for trial in record["search"]["trace"]:
            if trial["midpoint"] == point and trial["move"] == "upper":
                trial["attempts"][-1] = copy.deepcopy(altered)
        checked = verify_upper_bound(q, record, WorkBudget())
        self.assertFalse(checked["verified"])
        self.assertEqual(checked["failure"]["code"], "endpoint_witness_mismatch")

    def test_bad_unavailable_diagnostics_do_not_become_consistent_evidence(self):
        q = Fraction(1, 16)
        original = certify_upper_bound(q, WorkBudget(maximum_integer_bits=64))
        changes = (
            lambda record: record.__setitem__("endpoint", {"numerator": "1", "denominator": "1"}),
            lambda record: record["failure"].__setitem__("code", "ambiguous_root_comparison"),
            lambda record: record["work"].__setitem__("series_terms_used", -1),
            lambda record: record["independent_check"].__setitem__("lower_witness", "invented proof"),
        )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                record = copy.deepcopy(original)
                change(record)
                checked = verify_upper_bound(q, record, WorkBudget())
                self.assertFalse(checked["verified"])
                self.assertFalse(checked["unavailable_record_consistent"])
                self.assertIsNone(checked["endpoint"])

    def test_impossible_zero_work_for_numerical_trials_is_rejected(self):
        q = Fraction(1, 16)
        record = copy.deepcopy(self.records[q])
        for phase in ("before", "after"):
            record["work"][phase]["series_terms"] = 0
            record["work"][phase]["maximum_integer_bits_seen"] = 0
        record["work"]["series_terms_used"] = 0
        checked = verify_upper_bound(q, record, WorkBudget())
        self.assertFalse(checked["verified"])
        self.assertEqual(checked["failure"]["code"], "invalid_work_record")

    def test_saved_records_round_trip_without_any_float_leaf(self):
        def visit(value):
            self.assertNotIsInstance(value, float)
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        for q, record in self.records.items():
            visit(record)
            restored = json.loads(json.dumps(record, allow_nan=False))
            self.assertEqual(restored, record)
            self.assertTrue(verify_upper_bound(q, restored, WorkBudget())["verified"])


if __name__ == "__main__":
    unittest.main()

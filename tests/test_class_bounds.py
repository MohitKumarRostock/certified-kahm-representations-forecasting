"""General rational inverse-KL branches and endpoint-only verification."""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
import unittest
from unittest import mock

from kahm_demo import class_bounds as bounds
from kahm_demo.verified_arithmetic import RationalInterval, WorkBudget


def dec(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def reference_root(q, n, alpha, side):
    """Independent Decimal oracle used only by tests."""
    with localcontext() as context:
        context.prec = 100
        if side == "lower":
            return 1 - reference_root(1 - q, n, alpha, "upper")
        if q == 1:
            return Decimal(1)
        empirical = dec(q)
        confidence = -dec(alpha).ln() / n
        if q == 0:
            return 1 - (-confidence).exp()
        low, high = empirical, Decimal(1)
        for _ in range(320):
            point = (low + high) / 2
            kl = (empirical * (empirical / point).ln()
                  + (1 - empirical) * ((1 - empirical) / (1 - point)).ln())
            if kl >= confidence:
                high = point
            else:
                low = point
        return (low + high) / 2


class ClassBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = [(F(1, 3), 37, F(1, 100)), (F(2, 7), 1, F(1, 13)),
                      (F(0), 4096, F(1, 600)), (F(1), 4096, F(1, 600))]
        cls.records = {}
        for q, n, alpha in cls.inputs:
            for side in ("lower", "upper"):
                record = bounds.bounded_mean_endpoint(q, n, alpha, side)
                if record["status"] != "available":
                    raise AssertionError(record["failure"])
                cls.records[(q, n, alpha, side)] = record

    def test_both_branches_enclose_independent_decimal_roots(self):
        for (q, n, alpha, side), record in self.records.items():
            with self.subTest(q=q, n=n, side=side), localcontext() as context:
                context.prec = 100
                lo = bounds.parse_fraction(record["bracket"]["lower"])
                hi = bounds.parse_fraction(record["bracket"]["upper"])
                reference = reference_root(q, n, alpha, side)
                self.assertLessEqual(dec(lo), reference)
                self.assertLessEqual(reference, dec(hi))
                self.assertLessEqual(hi - lo, bounds.TARGET_WIDTH)
                endpoint = bounds.parse_fraction(record["endpoint"])
                self.assertEqual(endpoint, lo if side == "lower" else hi)
                display = Decimal(record["endpoint_decimal_18"])
                if side == "lower":
                    self.assertLessEqual(display, dec(endpoint))
                else:
                    self.assertGreaterEqual(display, dec(endpoint))
                self.assertLess(abs(display - dec(endpoint)), Decimal("1e-18"))
                checked = bounds.verify_mean_endpoint(
                    record, expected_q=q, expected_n=n,
                    expected_alpha=alpha, expected_side=side)
                self.assertTrue(checked["verified"], checked["failure"])
                self.assertTrue(checked["available"])

    def test_boundaries_and_short_brackets_do_not_need_logarithms(self):
        for q, side, expected in [(F(0), "lower", 0), (F(1), "upper", 1),
                                  (F(1, 2**100), "lower", 0),
                                  (1-F(1, 2**100), "upper", 1)]:
            with self.subTest(q=q, side=side), \
                 mock.patch.object(bounds, "log_interval", side_effect=AssertionError("unexpected log")), \
                 mock.patch.object(bounds, "kl_interval", side_effect=AssertionError("unexpected KL")):
                record = bounds.bounded_mean_endpoint(q, 1, F(1, 7), side,
                                                       budget=WorkBudget(maximum_series_terms=0))
                self.assertEqual(record["status"], "available")
                self.assertEqual(bounds.parse_fraction(record["endpoint"]), expected)
                self.assertEqual(record["work"]["series_terms_used"], 0)
                self.assertTrue(bounds.verify_mean_endpoint(record)["verified"])

    def test_checker_only_recomputes_the_two_final_endpoints(self):
        record = self.records[(F(1, 3), 37, F(1, 100), "lower")]
        visited = []
        real = bounds.kl_interval

        def track(q, point, bits, budget):
            visited.append(point)
            return real(q, point, bits, budget)

        with mock.patch.object(bounds, "bounded_mean_endpoint", side_effect=AssertionError("search called")), \
             mock.patch.object(bounds, "kl_interval", side_effect=track):
            result = bounds.verify_mean_endpoint(record)
        self.assertTrue(result["verified"], result["failure"])
        self.assertEqual(visited, [bounds.parse_fraction(record["bracket"][name])
                                  for name in ("lower", "upper")])
        self.assertLess(result["work"]["series_terms_used"], record["work"]["series_terms_used"])

    def test_expected_inputs_bind_records_to_independent_protocol(self):
        record = self.records[(F(1, 3), 37, F(1, 100), "lower")]
        for changed in ({"expected_q": F(1, 4)}, {"expected_n": 38},
                        {"expected_alpha": F(1, 99)}, {"expected_side": "upper"},
                        {"expected_n": True}):
            with self.subTest(changed=changed):
                checked = bounds.verify_mean_endpoint(record, **changed)
                self.assertFalse(checked["verified"])
                self.assertEqual(checked["failure"]["code"], "input_binding_mismatch")

    def test_tampering_cannot_change_the_issued_statement(self):
        base = self.records[(F(1, 3), 37, F(1, 100), "lower")]
        mutations = [
            lambda r: r.update(q=bounds.fraction_record(F(1, 4))),
            lambda r: r.update(n=38),
            lambda r: r.update(alpha=bounds.fraction_record(F(1, 99))),
            lambda r: r.update(side="upper"),
            lambda r: r.update(endpoint=bounds.fraction_record(F(1, 2))),
            lambda r: r.update(endpoint_decimal_18="0.999999999999999999"),
            lambda r: r["bracket"]["lower_witness"]["residual"].update(low=bounds.fraction_record(F(0))),
            lambda r: r["search"].update(interval_evaluations=0),
            lambda r: r["search"]["attempts"][0].update(precision_bits=512),
            lambda r: r["work"].update(series_terms_used=0),
            lambda r: r.update(independent_check={}),
            lambda r: r.update(status="requirement_met"),
            lambda r: r.update(unknown_field="not permitted"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                record = copy.deepcopy(base)
                mutate(record)
                checked = bounds.verify_mean_endpoint(record)
                self.assertFalse(checked["verified"], record)
                self.assertIsNotNone(checked["failure"])

    def test_budget_exhaustion_is_retained_without_an_endpoint(self):
        budget = WorkBudget(maximum_series_terms=128)
        record = bounds.bounded_mean_endpoint(F(1, 3), 37, F(1, 100), "lower", budget=budget)
        self.assertEqual(record["status"], "certificate_unavailable")
        self.assertEqual(record["failure"]["code"], "series_work_limit")
        self.assertIsNone(record["endpoint"])
        self.assertIsNone(record["endpoint_decimal_18"])
        checked = bounds.verify_mean_endpoint(record)
        self.assertFalse(checked["verified"])
        self.assertFalse(checked["available"])
        self.assertTrue(checked["unavailable_record_consistent"], checked["failure"])
        again = bounds.bounded_mean_endpoint(F(1, 7), 10, F(1, 9), "upper", budget=budget)
        self.assertEqual(again["status"], "certificate_unavailable")
        self.assertEqual(again["q"], bounds.fraction_record(F(1, 7)))
        self.assertEqual(again["work"]["series_terms_used"], 0)
        self.assertTrue(bounds.verify_mean_endpoint(again)["unavailable_record_consistent"])

    def test_unavailable_record_cannot_smuggle_a_bound(self):
        record = bounds.bounded_mean_endpoint(F(1, 3), 37, F(1, 100), "lower",
                                               budget=WorkBudget(maximum_series_terms=0))
        record["endpoint"] = bounds.fraction_record(F(0))
        checked = bounds.verify_mean_endpoint(record)
        self.assertFalse(checked["verified"])
        self.assertFalse(checked["unavailable_record_consistent"])
        self.assertEqual(checked["failure"]["code"], "invalid_unavailable_record")

    def test_resource_failure_in_direct_check_remains_visible(self):
        ordinary = self.records[(F(1, 3), 37, F(1, 100), "lower")]
        limit = ordinary["work"]["series_terms_used"] - 1
        record = bounds.bounded_mean_endpoint(F(1, 3), 37, F(1, 100), "lower",
                                               budget=WorkBudget(maximum_series_terms=limit))
        self.assertEqual(record["status"], "certificate_unavailable")
        self.assertEqual(record["independent_check"]["status"], "failed")
        self.assertEqual(record["independent_check"]["failure"], record["failure"])
        self.assertTrue(bounds.verify_mean_endpoint(record)["unavailable_record_consistent"])

    def test_integer_size_budget_failure_is_not_hidden_by_reduction(self):
        record = bounds.bounded_mean_endpoint(F(1, 3), 37, F(1, 100), "lower",
                                               budget=WorkBudget(maximum_integer_bits=128))
        self.assertEqual(record["status"], "certificate_unavailable")
        self.assertEqual(record["failure"]["code"], "integer_bit_limit")
        self.assertGreater(record["work"]["after"]["maximum_integer_bits_seen"], 128)
        self.assertTrue(bounds.verify_mean_endpoint(record)["unavailable_record_consistent"])

    def test_ambiguous_sign_never_becomes_a_guessed_endpoint(self):
        def ambiguous(*args):
            return RationalInterval(F(-100), F(100))

        with mock.patch.object(bounds, "kl_interval", side_effect=ambiguous):
            record = bounds.bounded_mean_endpoint(F(1, 3), 37, F(1, 100), "upper")
        self.assertEqual(record["status"], "certificate_unavailable")
        self.assertEqual(record["failure"]["code"], "ambiguous_root_comparison")
        self.assertEqual([a["precision_bits"] for a in record["search"]["attempts"]], [128, 256, 512])
        self.assertIsNone(record["endpoint"])
        self.assertTrue(bounds.verify_mean_endpoint(record)["unavailable_record_consistent"])

    def test_invalid_inputs_are_explicit_unavailable_json_records(self):
        valid = [F(1, 3), 37, F(1, 100), "lower"]
        invalids = [(0, 0.3), (0, F(-1)), (0, F(2)), (1, 0), (1, True),
                    (1, 1.5), (2, F(0)), (2, F(1)), (2, 0.05), (3, "left")]
        for index, value in invalids:
            with self.subTest(index=index, value=value):
                args = valid.copy()
                args[index] = value
                record = bounds.bounded_mean_endpoint(*args)
                self.assertEqual(record["status"], "certificate_unavailable")
                self.assertIsNone(record["endpoint"])
                json.dumps(record, allow_nan=False)

    def test_fraction_serialization_rejects_noncanonical_and_oversized_values(self):
        for record in [{"numerator": "2", "denominator": "6"},
                       {"numerator": "01", "denominator": "3"},
                       {"numerator": "-0", "denominator": "3"},
                       {"numerator": "1", "denominator": "0"},
                       {"numerator": 1, "denominator": "3"},
                       {"numerator": "1" * 10001, "denominator": "3"}]:
            with self.subTest(record=str(record)[:80]), self.assertRaises(ValueError):
                bounds.parse_fraction(record)


if __name__ == "__main__":
    unittest.main()

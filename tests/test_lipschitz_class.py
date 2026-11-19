"""Analytic optima and exact witness checks for finite Lipschitz classes."""

from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kahm_demo.lipschitz_class import (
    as_fraction, certify_empirical_minimum, certify_weighted_minimum,
    fraction_record, make_geometry, membership, parse_fraction_record,
    sqrt_enclosure, verify_empirical_minimum, verify_weighted_minimum,
)


ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_sqrt_direction_and_exact_endpoints(self):
        for x in (F(0), F(1, 4), F(2), F(1, 3), F(1, 10**100), F(10**100)):
            lo, hi = sqrt_enclosure(x, 80)
            self.assertLessEqual(lo * lo, x)
            self.assertGreaterEqual(hi * hi, x)
            self.assertLessEqual(hi - lo, F(1, 2**80))
        self.assertEqual(sqrt_enclosure(F(1, 4)), (F(1, 2), F(1, 2)))
        for args in ((F(-1), 80), (F(1), True), (F(1), 15)):
            with self.assertRaises(ValueError):
                sqrt_enclosure(*args)

    def test_exact_binary64_inputs_and_identity(self):
        a = make_geometry([[0.1], [0.2]], F(2))
        b = make_geometry([[0.1.hex()], [0.2.hex()]], 2)
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.features[0][0], F(0.1))
        self.assertNotEqual(a.features[0][0], F(1, 10))
        self.assertNotEqual(a.fingerprint, make_geometry([[0.1], [0.2]], 3).fingerprint)
        for value in (True, float("inf"), float("nan"), "nan", "0.1"):
            with self.assertRaises((ValueError, TypeError)):
                as_fraction(value)

    def test_membership_exact_boundary_zero_distance_and_output_range(self):
        g = make_geometry([[0], [F(1, 2)], [F(1, 2)]], 1)
        self.assertTrue(membership(g, [0, F(1, 2), F(1, 2)])["admissible"])
        self.assertFalse(membership(g, [0, F(1, 2), F(1, 2) + F(1, 10**30)])["admissible"])
        result = membership(g, [-1, F(-1, 2), F(-1, 2)])
        self.assertFalse(result["admissible"])
        self.assertFalse(result["output_range_valid"])
        self.assertEqual(result["violations"], [])
        irrational = make_geometry([[0, 0], [1, 1]], 1)
        self.assertTrue(membership(irrational, [0, 1])["admissible"])

    def test_invalid_tables_and_fraction_records(self):
        for rows, kappa in (([], 1), ([[]], 1), ([[0], [0, 1]], 1), ([[0]], -1)):
            with self.assertRaises(ValueError):
                make_geometry(rows, kappa)
        for record in ({"numerator": "2", "denominator": "4"},
                       {"numerator": "0", "denominator": "0"},
                       {"numerator": "+1", "denominator": "2"}):
            with self.assertRaises(ValueError):
                parse_fraction_record(record)


class OptimizationTests(unittest.TestCase):
    def check_record(self, g, ids, targets, **kwargs):
        record = certify_empirical_minimum(g, ids, targets, **kwargs)
        lower = verify_empirical_minimum(g, ids, targets, record)
        upper = parse_fraction_record(record["upper_sse"])
        self.assertLessEqual(lower, upper)
        json.dumps(record, allow_nan=False)
        return record, lower, upper

    def test_identical_features_preserve_all_occurrences_exactly(self):
        g = make_geometry([[0], [0], [1]], 4)
        ids, ys = [0, 1, 0, 2], [0, 1, 0, 1]
        record, lower, upper = self.check_record(g, ids, ys)
        self.assertEqual(lower, F(2, 3))
        self.assertEqual(upper, F(2, 3))
        self.assertEqual(record["sample_count"], 4)
        self.assertEqual(record["groups"][0]["weight"], fraction_record(F(3)))

    def test_zero_kappa_is_exact_constant_least_squares(self):
        g = make_geometry([[0], [10], [100]], 0)
        _, lower, upper = self.check_record(g, [0, 1, 2], [0, F(1, 2), 1])
        self.assertEqual((lower, upper), (F(1, 2), F(1, 2)))

    def test_two_point_analytic_optimum(self):
        g = make_geometry([[0], [F(1, 4)]], 1)
        _, lower, upper = self.check_record(g, [0, 1], [0, 1])
        self.assertEqual(lower, F(9, 32))
        self.assertEqual(upper, F(9, 32))

    def test_weighted_two_point_analytic_optimum(self):
        g = make_geometry([[0], [F(1, 4)]], 1)
        w, m = [F(1, 3), F(2, 3)], [F(0), F(1)]
        result = certify_weighted_minimum(g, w, m, within_sse=F(1, 10))
        lower = verify_weighted_minimum(g, w, m, result, within_sse=F(1, 10))
        # w1*w2/(w1+w2)*(1-1/4)^2 + constant.
        optimum = F(1, 8) + F(1, 10)
        self.assertLessEqual(lower, optimum)
        self.assertLess(optimum - lower, F(1, 10**28))
        self.assertGreaterEqual(parse_fraction_record(result["upper_sse"]), optimum)

    def test_unconstrained_distinct_states_and_missing_states(self):
        g = make_geometry([[0], [1], [F(1, 2)]], 1)
        _, lower, upper = self.check_record(g, [0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual((lower, upper), (F(1), F(1)))
        # The absent midpoint can be assigned 1/2 by a Lipschitz extension.
        _, lower, upper = self.check_record(g, [0, 1], [0, 1])
        self.assertEqual((lower, upper), (F(0), F(0)))

    def test_zero_sweeps_still_gives_honest_certificates(self):
        g = make_geometry([[0], [F(1, 4)]], 1)
        record, lower, upper = self.check_record(g, [0, 1], [0, 1], sweeps=0)
        self.assertEqual(lower, 0)
        self.assertGreaterEqual(upper, F(9, 32))
        self.assertEqual(record["optimization"]["termination"], "sweep_limit")

    def test_verification_does_not_run_optimizer(self):
        g = make_geometry([[0], [F(1, 4)]], 1)
        record = certify_empirical_minimum(g, [0, 1], [0, 1])
        with patch("kahm_demo.lipschitz_class._propose", side_effect=AssertionError("optimizer called")):
            self.assertEqual(verify_empirical_minimum(g, [0, 1], [0, 1], record), F(9, 32))

    def test_tampering_with_data_geometry_dual_and_primal_is_rejected(self):
        g = make_geometry([[0], [F(1, 4)]], 1)
        original = certify_empirical_minimum(g, [0, 1], [0, 1])
        variants = []
        r = deepcopy(original); r["lower_sse"] = fraction_record(F(1)); variants.append(r)
        r = deepcopy(original); r["sample_count"] = 1; variants.append(r)
        r = deepcopy(original); r["groups"][0]["weight"] = fraction_record(F(2)); variants.append(r)
        r = deepcopy(original); r["edges"][0]["multiplier_hex"] = (100.0).hex(); variants.append(r)
        r = deepcopy(original); r["edges"].append(deepcopy(r["edges"][0])); variants.append(r)
        r = deepcopy(original); r["primal_predictions"][0]["prediction"] = fraction_record(F(0)); variants.append(r)
        r = deepcopy(original); r["geometry_sha256"] = "0" * 64; variants.append(r)
        for r in variants:
            with self.subTest(record=r), self.assertRaises(ValueError):
                verify_empirical_minimum(g, [0, 1], [0, 1], r)
        with self.assertRaises(ValueError):
            verify_empirical_minimum(g, [0, 1], [0, F(1, 2)], original)
        with self.assertRaises(ValueError):
            verify_empirical_minimum(make_geometry([[0], [F(1, 4)]], 2), [0, 1], [0, 1], original)

    def test_invalid_empirical_inputs(self):
        g = make_geometry([[0], [1]], 1)
        for ids, ys in (([], []), ([0], []), ([True], [0]), ([2], [0]), ([0], [-1])):
            with self.assertRaises((ValueError, TypeError)):
                certify_empirical_minimum(g, ids, ys)
        for sweeps in (-1, True, 10001):
            with self.assertRaises(ValueError):
                certify_empirical_minimum(g, [0], [0], sweeps=sweeps)

    def test_archived_kahm_geometry_separates_regular_decoder_classes(self):
        rows = json.loads((ROOT / "encoders/primary/cap_2_features.json").read_text())["entries"]
        values = [F(0), F(1, 4), F(3, 4), F(1)]
        mu = [F(1, 10) + F(4, 5) * values[(r["previous_state"] + r["current_state"]) % 4] for r in rows]
        for name in ("raw", "kahm_mem", "kahm_local"):
            g = make_geometry([r[name + "_hex"] for r in rows], 2)
            result = certify_weighted_minimum(g, [F(1, 16)] * 16, mu, within_sse=F(9, 160), sweeps=400)
            lower = verify_weighted_minimum(g, [F(1, 16)] * 16, mu, result, within_sse=F(9, 160))
            upper = parse_fraction_record(result["upper_sse"])
            if name == "kahm_mem":
                self.assertGreater(lower, F(1, 10))
                self.assertFalse(membership(g, mu)["admissible"])
            else:
                self.assertEqual(lower, F(9, 160))
                self.assertEqual(upper, F(9, 160))
                self.assertTrue(membership(g, mu)["admissible"])


if __name__ == "__main__":
    unittest.main()

"""Analytical controls and independent checks of population class audits."""

from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kahm_demo.class_oracle import (
    audit_class, exact_population_risk, hierarchy_reference, irreducible_risk,
    population_means, verify_class_audit,
)
from kahm_demo.law import HISTORY_KEYS, OBSERVATIONS, joint_probability
from kahm_demo.lipschitz_class import make_geometry


ROOT = Path(__file__).resolve().parents[1]


def value(record):
    return F(int(record["numerator"]), int(record["denominator"]))


def full_features():
    return tuple((OBSERVATIONS[a], OBSERVATIONS[b]) for a, b in HISTORY_KEYS)


def short_features():
    return tuple((OBSERVATIONS[b],) for _, b in HISTORY_KEYS)


class PopulationReferenceTests(unittest.TestCase):
    def test_risks_by_direct_population_enumeration(self):
        self.assertEqual(irreducible_risk(), F(9, 160))
        self.assertEqual(exact_population_risk(population_means()), F(9, 160))
        self.assertEqual(exact_population_risk((F(1, 2),) * 16), F(5, 32))
        self.assertEqual(exact_population_risk((F(0),) * 16), F(13, 32))
        for index, (a, b) in enumerate(HISTORY_KEYS):
            mean = sum((joint_probability(a, b, c) * OBSERVATIONS[c] * 16
                        for c in range(4)), F(0))
            self.assertEqual(population_means()[index], mean)

    def test_population_inputs_require_complete_exact_bounded_values(self):
        for invalid in [(F(0),) * 15, (F(2),) * 16]:
            with self.assertRaises(ValueError):
                exact_population_risk(invalid)
        for invalid in [(0.5,) * 16, (False,) * 16, "0" * 16]:
            with self.assertRaises(TypeError):
                exact_population_risk(invalid)

    def test_pair_hierarchy_matches_full_joint_enumeration(self):
        # Directly average the optimized two-observation score over all 4096
        # pairs of outcomes.  Distinct features allow zero block error here;
        # identical features must use a common prediction.
        outcomes = tuple(product(range(4), repeat=3))
        for name, feature in [("short", lambda a, b: (b,)),
                              ("full", lambda a, b: (a, b))]:
            expectation = F(0)
            for a, b, c in outcomes:
                for aa, bb, cc in outcomes:
                    if feature(a, b) == feature(aa, bb):
                        score = (OBSERVATIONS[c] - OBSERVATIONS[cc]) ** 2 / 2
                        expectation += joint_probability(a, b, c) * joint_probability(aa, bb, cc) * score
            self.assertEqual(value(hierarchy_reference(2)[name]["expected_block_score"]), expectation)

    def test_short_three_observation_hierarchy_by_enumerating_group_variances(self):
        # Under the short representation, feature and target are independent
        # uniform four-valued variables.  Enumerate all 16**3 triples.
        outcomes = tuple(product(range(4), repeat=2))
        total = F(0)
        for batch in product(outcomes, repeat=3):
            groups = {}
            for feature, target in batch:
                groups.setdefault(feature, []).append(OBSERVATIONS[target])
            sse = F(0)
            for ys in groups.values():
                mean = sum(ys, F(0)) / len(ys)
                sse += sum(((y - mean) ** 2 for y in ys), F(0))
            total += sse / 2
        self.assertEqual(value(hierarchy_reference(3)["short"]["expected_block_score"]), total / 16 ** 3)

    def test_block_hierarchy_resolves_short_representation_failure(self):
        values = [value(hierarchy_reference(b)["short"]["expected_block_score"])
                  for b in (2, 4, 8, 16, 32, 64)]
        self.assertEqual(values[0], F(5, 128))
        self.assertEqual(values, sorted(values))
        self.assertLess(values[2], F(1, 10))
        self.assertGreater(values[3], F(1, 10))
        self.assertLess(values[-1], F(5, 32))
        for invalid in [True, 1, 0, -1, 2.0, 100001]:
            with self.assertRaises(ValueError):
                hierarchy_reference(invalid)


class PopulationClassAuditTests(unittest.TestCase):
    def test_exact_short_and_full_classes(self):
        for features, expected, groups in [(short_features(), F(5, 32), 4),
                                            (full_features(), F(9, 160), 16)]:
            geometry = make_geometry(features, F(4))
            record = audit_class(geometry, block_sizes=(2, 16), sweeps=20)
            self.assertEqual(record["optimal_risk_status"], "exact")
            self.assertEqual(value(record["optimal_risk_lower"]), expected)
            self.assertEqual(value(record["optimal_risk_upper"]), expected)
            self.assertEqual(record["analytic_partition"]["feature_support_size"], groups)
            self.assertTrue(verify_class_audit(geometry, record)["verified"])

    def test_constant_features_preserve_conditional_variance(self):
        geometry = make_geometry(((F(0),),) * 16, F(0))
        record = audit_class(geometry, block_sizes=(2, 16), sweeps=5)
        self.assertEqual(value(record["optimal_risk_lower"]), F(5, 32))
        self.assertEqual(value(record["optimal_risk_upper"]), F(5, 32))
        for block in record["analytic_partition"]["hierarchy"]:
            self.assertEqual(value(block["expected_block_score"]), F(5, 32))

    def test_nonuniform_partition_has_exact_risk_without_uniform_hierarchy(self):
        # One feature represents one history, the other represents fifteen.
        features = ((F(0),),) + ((F(1),),) * 15
        geometry = make_geometry(features, F(1))
        record = audit_class(geometry, block_sizes=(2, 16), sweeps=5)
        self.assertEqual(record["optimal_risk_status"], "exact")
        self.assertFalse(record["analytic_partition"]["uniform_feature_masses"])
        self.assertEqual(record["analytic_partition"]["hierarchy"], [])
        means = population_means()
        reference = (means[0],) + (sum(means[1:], F(0)) / 15,) * 15
        self.assertEqual(value(record["optimal_risk_lower"]), exact_population_risk(reference))

    def test_audit_reverification_does_not_optimize(self):
        geometry = make_geometry(full_features(), F(4))
        record = audit_class(geometry, population_means(), sweeps=10)
        with patch("kahm_demo.lipschitz_class.certify_weighted_minimum",
                   side_effect=AssertionError("verifier must not optimize")):
            self.assertTrue(verify_class_audit(geometry, record, population_means())["verified"])
        corrupted = deepcopy(record)
        corrupted["optimal_risk_lower"] = {"numerator": "1", "denominator": "1"}
        self.assertFalse(verify_class_audit(geometry, corrupted, population_means())["verified"])
        wrong_geometry = make_geometry(full_features(), F(3))
        self.assertFalse(verify_class_audit(wrong_geometry, record, population_means())["verified"])
        self.assertFalse(verify_class_audit(geometry, record)["verified"])

    def test_archive_memberships_reveal_a_class_constraint(self):
        table = json.loads((ROOT / "encoders/primary/cap_2_features.json").read_text())
        entries = sorted(table["entries"], key=lambda row: row["history_id"])
        features = tuple(tuple(F(float.fromhex(token)) for token in row["kahm_mem_hex"])
                         for row in entries)
        saved = json.loads((ROOT / "frozen/primary/predictions/kahm_mem.json").read_text())
        predictions = tuple(F(float.fromhex(row["prediction_hex"]))
                            for row in sorted(saved["predictions"], key=lambda row: row["history_id"]))
        geometry = make_geometry(features, F(2))
        record = audit_class(geometry, predictions, sweeps=60, block_sizes=(2, 16))
        self.assertEqual(value(record["four_pair_reference"]["lower_risk"]), F(29513, 250000))
        self.assertGreater(value(record["optimal_risk_lower"]), F(1, 10))
        self.assertLess(value(record["fitted_risk"]), F(1, 10))
        self.assertFalse(record["fitted_admissibility"]["admissible"])
        self.assertNotIn("fitted_decoder", record["upper_sources"])
        self.assertLessEqual(value(record["optimal_risk_lower"]), value(record["optimal_risk_upper"]))
        self.assertTrue(verify_class_audit(geometry, record, predictions)["verified"])

    def test_wrong_population_support_and_repeated_sizes_are_rejected(self):
        geometry = make_geometry(((F(0),), (F(1),)), F(1))
        with self.assertRaises(ValueError):
            audit_class(geometry)
        geometry = make_geometry(full_features(), F(4))
        with self.assertRaises(ValueError):
            audit_class(geometry, block_sizes=(2, 2))


if __name__ == "__main__":
    unittest.main()

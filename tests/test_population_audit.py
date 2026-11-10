"""Synthetic exact population-audit checks; no fitted artifact is accessed."""

from collections import Counter
from collections.abc import Mapping
import copy
from fractions import Fraction
import json
import unittest
from unittest.mock import patch

from kahm_demo.contract import CONTRACT_SHA256, load_contract
from kahm_demo import oracle
from kahm_demo.population_audit import (
    POPULATION_AUDIT_SCHEMA, build_population_audit, verify_population_audit,
)


HISTORIES = tuple((a, b) for a in range(4) for b in range(4))
OBSERVATIONS = (Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1))


def fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def constant(value):
    return {history: value for history in HISTORIES}


class PopulationAuditTests(unittest.TestCase):
    def test_analytical_constant_and_nominal_references(self):
        nominal = {(a, b): float(OBSERVATIONS[(a + b) % 4]) for a, b in HISTORIES}
        for predictions, risk, excess in (
            (constant(0.5), Fraction(5, 32), Fraction(1, 10)),
            (constant(0.0), Fraction(13, 32), Fraction(7, 20)),
            (constant(1.0), Fraction(13, 32), Fraction(7, 20)),
            (nominal, Fraction(1, 16), Fraction(1, 160)),
        ):
            with self.subTest(risk=risk, excess=excess):
                audit = build_population_audit(predictions)
                self.assertEqual(audit["schema"], POPULATION_AUDIT_SCHEMA)
                self.assertEqual(audit["contract_sha256"], CONTRACT_SHA256)
                self.assertEqual(fraction(audit["exact_population_mse"]), risk)
                self.assertEqual(fraction(audit["bayes_mse"]), Fraction(9, 160))
                self.assertEqual(fraction(audit["excess_over_bayes"]), excess)
                self.assertEqual(fraction(audit["references"]["best_constant_prediction"]), Fraction(1, 2))
                self.assertEqual(fraction(audit["references"]["best_constant_mse"]), Fraction(5, 32))
                self.assertEqual(fraction(audit["references"]["nominal_successor_mse"]), Fraction(1, 16))

    def test_rounded_conditional_means_are_not_the_ideal_bayes_function(self):
        ideal = {(a, b): Fraction(1, 10) + Fraction(4, 5) * OBSERVATIONS[(a + b) % 4]
                 for a, b in HISTORIES}
        rounded = {history: float(mean) for history, mean in ideal.items()}
        audit = build_population_audit(rounded)
        expected_excess = sum(((Fraction.from_float(rounded[h]) - ideal[h]) ** 2
                               for h in HISTORIES), Fraction(0)) / 16
        self.assertGreater(expected_excess, 0)
        self.assertEqual(fraction(audit["excess_over_bayes"]), expected_excess)
        self.assertEqual(fraction(audit["exact_population_mse"]), Fraction(9, 160) + expected_excess)
        self.assertTrue(all(fraction(row["prediction_exact"]) != fraction(row["conditional_mean"])
                            for row in audit["histories"]))
        with self.assertRaises(TypeError):
            build_population_audit(ideal)

    def test_nonsymmetric_predictions_match_an_independent_mixture_formula(self):
        predictions = {(a, b): (2 * a + 3 * b + 1) / 18 for a, b in HISTORIES}
        audit = build_population_audit(predictions)
        mixture_risk = Fraction(0)
        bias_sum = Fraction(0)
        for a, b in HISTORIES:
            forecast = Fraction.from_float(predictions[(a, b)])
            nominal = OBSERVATIONS[(a + b) % 4]
            # An 80% deterministic transition plus a 20% uniform mixture is
            # algebraically separate from summing the 17/320 and 1/320 cells.
            mixture_risk += (Fraction(4, 5) * (nominal - forecast) ** 2
                             + Fraction(1, 20) * sum(((y - forecast) ** 2 for y in OBSERVATIONS), Fraction(0))) / 16
            mean = Fraction(1, 10) + Fraction(4, 5) * nominal
            bias_sum += (forecast - mean) ** 2 / 16
        self.assertEqual(fraction(audit["exact_population_mse"]), mixture_risk)
        self.assertEqual(fraction(audit["excess_over_bayes"]), bias_sum)
        self.assertEqual(mixture_risk, Fraction(9, 160) + bias_sum)
        self.assertEqual(fraction(audit["checks"]["population_minus_oracle"]), 0)
        self.assertEqual(fraction(audit["checks"]["population_minus_bayes_minus_excess"]), 0)

    def test_all_64_positive_mass_cells_and_16_histories_are_retained(self):
        audit = build_population_audit(constant(0.5))
        triples, histories = audit["triples"], audit["histories"]
        self.assertEqual(audit["history_count"], len(histories))
        self.assertEqual(audit["triple_count"], len(triples))
        self.assertEqual([row["history_id"] for row in histories], list(range(16)))
        self.assertEqual([row["triple_id"] for row in triples], list(range(64)))
        self.assertEqual(Counter(row["history_id"] for row in triples), {index: 4 for index in range(16)})
        masses = Counter(fraction(row["joint_probability"]) for row in triples)
        self.assertEqual(masses, {Fraction(17, 320): 16, Fraction(1, 320): 48})
        self.assertEqual(sum((fraction(row["joint_probability"]) for row in triples), Fraction(0)), 1)
        self.assertEqual(fraction(audit["joint_probability_mass"]), 1)
        for history in histories:
            rows = [row for row in triples if row["history_id"] == history["history_id"]]
            self.assertEqual([row["next_state"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(sum((fraction(row["joint_probability"]) for row in rows), Fraction(0)), Fraction(1, 16))
            self.assertEqual(sum((fraction(row["conditional_probability"]) for row in rows), Fraction(0)), 1)
            self.assertEqual(sum((fraction(row["weighted_squared_loss"]) for row in rows), Fraction(0)),
                             fraction(history["weighted_mse"]))
            for row in rows:
                self.assertEqual(row["is_nominal_successor"],
                                 row["next_state"] == (row["previous_state"] + row["current_state"]) % 4)
                self.assertEqual(fraction(row["weighted_squared_loss"]),
                                 fraction(row["joint_probability"]) * fraction(row["squared_loss"]))

    def test_every_history_decomposition_and_prediction_binding_is_exact(self):
        predictions = {(a, b): (a * 4 + b) / 16 for a, b in HISTORIES}
        audit = build_population_audit(predictions)
        for row in audit["histories"]:
            a, b = row["previous_state"], row["current_state"]
            self.assertEqual(row["prediction_hex"], predictions[(a, b)].hex())
            self.assertEqual(fraction(row["prediction_exact"]), Fraction.from_float(predictions[(a, b)]))
            mean, variance = fraction(row["conditional_mean"]), fraction(row["conditional_variance"])
            self.assertEqual(variance, fraction(row["conditional_second_moment"]) - mean ** 2)
            self.assertEqual(fraction(row["conditional_mse"]),
                             variance + (fraction(row["prediction_exact"]) - mean) ** 2)
            self.assertEqual(fraction(row["weighted_mse"]), fraction(row["conditional_mse"]) / 16)
            self.assertEqual(fraction(row["weighted_bayes_mse"]), variance / 16)
            self.assertEqual(fraction(row["weighted_excess_mse"]), fraction(row["conditional_excess_mse"]) / 16)

    def test_signed_zero_is_preserved_without_changing_its_exact_value(self):
        predictions = constant(0.0)
        predictions[(0, 0)] = -0.0
        audit = build_population_audit(predictions)
        self.assertEqual(audit["histories"][0]["prediction_hex"], "-0x0.0p+0")
        self.assertEqual(fraction(audit["histories"][0]["prediction_exact"]), 0)
        self.assertEqual(fraction(audit["exact_population_mse"]), Fraction(13, 32))
        with self.assertRaises(ValueError):
            verify_population_audit(constant(0.0), audit)

    def test_subnormal_float_keeps_its_full_exact_dyadic_value(self):
        predictions = constant(0.5)
        predictions[(0, 0)] = float.fromhex("0x0.0000000000001p-1022")
        audit = build_population_audit(predictions)
        tiny = Fraction(1, 2 ** 1074)
        self.assertEqual(fraction(audit["histories"][0]["prediction_exact"]), tiny)
        expected = Fraction(5, 32) + ((tiny - Fraction(1, 10)) ** 2
                                      - (Fraction(1, 2) - Fraction(1, 10)) ** 2) / 16
        self.assertEqual(fraction(audit["exact_population_mse"]), expected)
        self.assertEqual(verify_population_audit(predictions, json.loads(json.dumps(audit))), audit)

    def test_input_order_does_not_change_canonical_audit_order(self):
        predictions = {(a, b): (a + b) / 8 for a, b in HISTORIES}
        reverse = dict(reversed(list(predictions.items())))
        self.assertEqual(build_population_audit(predictions), build_population_audit(reverse))

    def test_actual_oracle_crosscheck_is_called_on_copied_python_floats(self):
        predictions = constant(0.5)
        with patch("kahm_demo.population_audit.oracle.exact_risk", wraps=oracle.exact_risk) as checked:
            audit = build_population_audit(predictions)
        checked.assert_called_once()
        supplied = checked.call_args.args[0]
        self.assertIsNot(supplied, predictions)
        self.assertEqual(supplied, predictions)
        self.assertTrue(all(type(value) is float for value in supplied.values()))
        self.assertTrue(audit["checks"]["oracle_exact_equality"])

    def test_oracle_disagreement_or_wrong_return_type_is_not_silently_recorded(self):
        for wrong in (Fraction(0), 0.15625):
            with self.subTest(wrong=wrong), \
                 patch("kahm_demo.population_audit.oracle.exact_risk", return_value=wrong):
                with self.assertRaisesRegex(ValueError, "disagrees with oracle"):
                    build_population_audit(constant(0.5))

    def test_independent_route_does_not_use_existing_probability_or_mean_helpers(self):
        with (patch("kahm_demo.population_audit.oracle.exact_risk", return_value=Fraction(5, 32)),
              patch("kahm_demo.oracle.joint_probability", side_effect=AssertionError("shared helper")),
              patch("kahm_demo.oracle.conditional_mean", side_effect=AssertionError("shared helper")),
              patch("kahm_demo.law.joint_probability", side_effect=AssertionError("shared helper")),
              patch("kahm_demo.law.conditional_mean", side_effect=AssertionError("shared helper"))):
            audit = build_population_audit(constant(0.5))
        self.assertEqual(fraction(audit["exact_population_mse"]), Fraction(5, 32))

    def test_independent_benchmark_constants_must_agree_with_step1(self):
        contract = load_contract()
        changes = (
            ("id", "different-law"), ("support_histories", 15),
            ("nominal_successor_probability", {"numerator": 4, "denominator": 5}),
            ("bayes_mse", {"numerator": 1, "denominator": 16}),
            ("observation_values", [{"numerator": 0, "denominator": 1}] * 4),
        )
        for key, value in changes:
            changed = copy.deepcopy(contract)
            changed["benchmark"][key] = value
            with self.subTest(key=key), patch("kahm_demo.population_audit.load_contract", return_value=changed):
                with self.assertRaises(ValueError):
                    build_population_audit(constant(0.5))

    def test_missing_extra_or_malformed_history_support_is_rejected(self):
        missing = constant(0.5)
        missing.pop((3, 3))
        extra = constant(0.5)
        extra[(4, 0)] = 0.5
        for invalid in (missing, extra, list(constant(0.5).items())):
            with self.subTest(kind=type(invalid)), self.assertRaises((TypeError, ValueError)):
                build_population_audit(invalid)
        for bad_key in ("0,0", (0,), (4, 0), ("0", 0), (False, 0)):
            invalid = constant(0.5)
            invalid.pop((0, 0))
            invalid[bad_key] = 0.5
            with self.subTest(key=bad_key), self.assertRaises((TypeError, ValueError)):
                build_population_audit(invalid)

    def test_a_mapping_cannot_hide_duplicate_items(self):
        class RepeatedItemMapping(Mapping):
            def __len__(self):
                return 16

            def __iter__(self):
                return iter(HISTORIES)

            def __getitem__(self, key):
                return 0.5

            def items(self):
                return [(history, 0.5) for history in HISTORIES[:-1]] + [(HISTORIES[0], 0.5)]

        with self.assertRaisesRegex(ValueError, "repeats"):
            build_population_audit(RepeatedItemMapping())

    def test_predictions_must_be_finite_in_range_actual_python_floats(self):
        class FloatSubclass(float):
            pass

        for invalid in (0, True, Fraction(1, 2), DecimalLike(), FloatSubclass(.5),
                        float("nan"), float("inf"), -float("inf"), -.1, 1.1):
            predictions = constant(0.5)
            predictions[(0, 0)] = invalid
            with self.subTest(value=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                build_population_audit(predictions)

    def test_json_roundtrip_has_only_exact_fraction_strings_and_no_float_leaf(self):
        predictions = constant(0.5)
        audit = build_population_audit(predictions)

        def inspect(value):
            self.assertIsNot(type(value), float)
            if type(value) is dict:
                if set(value) == {"numerator", "denominator"}:
                    self.assertIs(type(value["numerator"]), str)
                    self.assertIs(type(value["denominator"]), str)
                    exact = fraction(value)
                    self.assertEqual(value, {"numerator": str(exact.numerator), "denominator": str(exact.denominator)})
                for item in value.values():
                    inspect(item)
            elif type(value) is list:
                for item in value:
                    inspect(item)

        inspect(audit)
        restored = json.loads(json.dumps(audit, allow_nan=False))
        self.assertEqual(verify_population_audit(predictions, restored), audit)

    def test_tampering_with_any_contribution_or_decomposition_is_rejected(self):
        predictions = constant(0.5)
        audit = build_population_audit(predictions)
        changes = (
            lambda r: r["exact_population_mse"].__setitem__("numerator", "0"),
            lambda r: r["triples"][0]["weighted_squared_loss"].__setitem__("numerator", "0"),
            lambda r: r["histories"][0]["conditional_mean"].__setitem__("numerator", "2"),
            lambda r: r["histories"][0].__setitem__("prediction_hex", 0.25.hex()),
            lambda r: r["checks"].__setitem__("oracle_exact_equality", False),
            lambda r: r["references"]["bayes_mse"].__setitem__("denominator", "80"),
            lambda r: r.__setitem__("schema", "kahm.exact_population_audit.v0"),
            lambda r: r.__setitem__("unexpected", "extra"),
        )
        for index, change in enumerate(changes):
            record = copy.deepcopy(audit)
            change(record)
            with self.subTest(index=index), self.assertRaises(ValueError):
                verify_population_audit(predictions, record)
        changed_predictions = predictions.copy()
        changed_predictions[(0, 0)] = 0.25
        with self.assertRaises(ValueError):
            verify_population_audit(changed_predictions, audit)

    def test_noncanonical_types_fractions_or_row_order_cannot_pass_equal_comparison(self):
        predictions = constant(0.5)
        audit = build_population_audit(predictions)
        changes = (
            lambda r: r["histories"][0].__setitem__("history_id", False),
            lambda r: r["checks"].__setitem__("oracle_exact_equality", 1),
            lambda r: r.__setitem__("triple_count", 64.0),
            lambda r: r["histories"][0].__setitem__("conditional_mean", {"numerator": "2", "denominator": "20"}),
            lambda r: r["histories"][0]["conditional_mean"].__setitem__("numerator", "01"),
            lambda r: r["histories"].reverse(),
            lambda r: r["triples"].reverse(),
            lambda r: r["triples"].pop(),
        )
        for index, change in enumerate(changes):
            record = copy.deepcopy(audit)
            change(record)
            with self.subTest(index=index), self.assertRaises(ValueError):
                verify_population_audit(predictions, record)


class DecimalLike:
    """An arbitrary convertible object must not become an actual float output."""

    def __float__(self):
        return 0.5


if __name__ == "__main__":
    unittest.main()

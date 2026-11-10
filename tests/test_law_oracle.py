"""Tests of scientific identities and the oracle's numerical object boundary."""

import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import math
import unittest

from kahm_demo.law import (
    BENCHMARK_ID,
    Episode,
    HISTORY_KEYS,
    OBSERVATIONS,
    OBSERVATION_FLOATS,
    conditional_mean,
    joint_probability,
    stationary_history_probability,
    transition_probability,
)
from kahm_demo.oracle import (
    exact_risk,
    parse_prediction_table,
    prediction_table_from_mapping,
    reference_summary,
    risk_of_predictor,
)


class LawTests(unittest.TestCase):
    def test_support_and_observation_values(self):
        self.assertEqual(BENCHMARK_ID, "four_state_order2_modular_v1")
        self.assertEqual(HISTORY_KEYS, tuple((a, b) for a in range(4) for b in range(4)))
        self.assertEqual(OBSERVATIONS, (Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1)))
        self.assertEqual(tuple(Fraction.from_float(v) for v in OBSERVATION_FLOATS), OBSERVATIONS)

    def test_transition_counts_and_total_probability(self):
        for a, b in HISTORY_KEYS:
            row = [transition_probability(a, b, c) for c in range(4)]
            self.assertEqual(sum(row), 1)
            self.assertEqual(row.count(Fraction(17, 20)), 1)
            self.assertEqual(row[(a + b) % 4], Fraction(17, 20))
            self.assertEqual(row.count(Fraction(1, 20)), 3)
        self.assertEqual(
            sum(joint_probability(a, b, c) for a, b in HISTORY_KEYS for c in range(4)),
            1,
        )

    def test_stationary_pair_law_and_output_moments(self):
        for b in range(4):
            for c in range(4):
                self.assertEqual(
                    sum(joint_probability(a, b, c) for a in range(4)),
                    stationary_history_probability(b, c),
                )
        marginal = [sum(joint_probability(a, b, c) for a, b in HISTORY_KEYS) for c in range(4)]
        self.assertEqual(marginal, [Fraction(1, 4)] * 4)
        mean = sum(p * y for p, y in zip(marginal, OBSERVATIONS))
        variance = sum(p * (y - mean) ** 2 for p, y in zip(marginal, OBSERVATIONS))
        self.assertEqual(mean, Fraction(1, 2))
        self.assertEqual(variance, Fraction(5, 32))

    def test_conditional_mean_matches_transition_enumeration(self):
        for a, b in HISTORY_KEYS:
            self.assertEqual(
                conditional_mean(a, b),
                sum(transition_probability(a, b, c) * OBSERVATIONS[c] for c in range(4)),
            )
        self.assertEqual(
            [conditional_mean(0, b) for b in range(4)],
            [Fraction(1, 10), Fraction(3, 10), Fraction(7, 10), Fraction(9, 10)],
        )

    def test_state_validation_rejects_boolean_and_noninteger_labels(self):
        for invalid in (True, False, 1.0, "1", Fraction(1), None):
            for call in (
                lambda: transition_probability(invalid, 0, 0),
                lambda: transition_probability(0, invalid, 0),
                lambda: transition_probability(0, 0, invalid),
                lambda: joint_probability(0, invalid, 0),
                lambda: conditional_mean(invalid, 0),
                lambda: stationary_history_probability(0, invalid),
            ):
                with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                    call()
        for invalid in (-1, 4, 100):
            with self.assertRaises(ValueError):
                transition_probability(0, 0, invalid)
            with self.assertRaises(ValueError):
                conditional_mean(invalid, 0)

    def test_episode_order_id_and_immutability(self):
        episode = Episode(7, 1, 3, 2)
        self.assertEqual(episode.x, (0.25, 1.0))
        self.assertEqual(episode.y, 0.75)
        self.assertEqual(episode.history_id, 7)
        self.assertEqual(episode.episode_id, 7)
        with self.assertRaises(FrozenInstanceError):
            episode.next_state = 1
        for arguments in ((True, 0, 0, 0), (0, False, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, "1")):
            with self.subTest(arguments=arguments), self.assertRaises(TypeError):
                Episode(*arguments)
        for arguments in ((-1, 0, 0, 0), (0, 4, 0, 0), (0, 0, -1, 0), (0, 0, 0, 4)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                Episode(*arguments)


class OracleTests(unittest.TestCase):
    def test_exact_analytical_reference_risks(self):
        bayes = {(a, b): conditional_mean(a, b) for a, b in HISTORY_KEYS}
        self.assertEqual(exact_risk(bayes), Fraction(9, 160))
        self.assertEqual(exact_risk({key: 0.5 for key in HISTORY_KEYS}), Fraction(5, 32))
        self.assertEqual(exact_risk({key: 0 for key in HISTORY_KEYS}), Fraction(13, 32))
        self.assertEqual(exact_risk({key: 1 for key in HISTORY_KEYS}), Fraction(13, 32))
        self.assertEqual(
            exact_risk({(a, b): OBSERVATIONS[b] for a, b in HISTORY_KEYS}), Fraction(5, 16)
        )
        self.assertEqual(
            exact_risk({(a, b): OBSERVATIONS[(a + b) % 4] for a, b in HISTORY_KEYS}),
            Fraction(1, 16),
        )

    def test_actual_float_is_not_decimal_rationalization(self):
        float_predictions = {key: 0.1 for key in HISTORY_KEYS}
        rational_predictions = {key: Fraction(1, 10) for key in HISTORY_KEYS}
        p = Fraction.from_float(0.1)
        self.assertEqual(exact_risk(float_predictions), Fraction(13, 32) - p + p * p)
        self.assertNotEqual(exact_risk(float_predictions), exact_risk(rational_predictions))

    def test_rounded_bayes_means_have_exact_positive_excess(self):
        predictions = {(a, b): float(conditional_mean(a, b)) for a, b in HISTORY_KEYS}
        excess = sum(
            (Fraction.from_float(predictions[(a, b)]) - conditional_mean(a, b)) ** 2
            for a, b in HISTORY_KEYS
        ) / 16
        self.assertGreater(excess, 0)
        self.assertEqual(exact_risk(predictions), Fraction(9, 160) + excess)

    def test_nonsymmetric_predictions_against_two_independent_calculations(self):
        predictions = {
            (a, b): ((3 * a + 5 * b + a * b + 1) % 31) / 32
            for a, b in HISTORY_KEYS
        }
        self.assertNotEqual(predictions[(1, 2)], predictions[(2, 1)])
        exact = {key: Fraction.from_float(value) for key, value in predictions.items()}
        # Independent integer-ticket construction: 17 nominal and 3 alternative
        # tickets for each uniformly weighted ordered history.
        ticket_risk = Fraction(0)
        for a, b in HISTORY_KEYS:
            j = (a + b) % 4
            for ticket in range(20):
                c = j if ticket < 17 else (j + ticket - 16) % 4
                ticket_risk += (OBSERVATIONS[c] - exact[(a, b)]) ** 2 / 320
        decomposition = Fraction(9, 160) + sum(
            (exact[(a, b)] - conditional_mean(a, b)) ** 2 for a, b in HISTORY_KEYS
        ) / 16
        self.assertEqual(exact_risk(predictions), ticket_risk)
        self.assertEqual(exact_risk(predictions), decomposition)

    def test_callback_receives_canonical_histories_exactly_once(self):
        calls = []

        def predictor(history):
            calls.append(history)
            return 0.5

        self.assertEqual(risk_of_predictor(predictor), Fraction(5, 32))
        self.assertEqual(
            calls, [(OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b]) for a, b in HISTORY_KEYS]
        )
        self.assertEqual(len(set(calls)), 16)

    def test_full_support_required(self):
        predictions = {key: 0.5 for key in HISTORY_KEYS}
        missing = dict(predictions)
        del missing[(3, 3)]
        with self.assertRaises(ValueError):
            exact_risk(missing)
        extra = dict(predictions)
        extra[(4, 0)] = 0.5
        with self.assertRaises(ValueError):
            exact_risk(extra)
        wrong = dict(missing)
        wrong[(4, 0)] = 0.5
        with self.assertRaises(ValueError):
            exact_risk(wrong)
        boolean_key = dict(predictions)
        del boolean_key[(0, 0)]
        boolean_key[(False, 0)] = 0.5
        with self.assertRaises(TypeError):
            exact_risk(boolean_key)
        with self.assertRaises(TypeError):
            exact_risk([0.5] * 16)

    def test_bad_predictions_are_rejected_never_clipped(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), -0.01, 1.01, -1, 2, Fraction(3, 2)):
            predictions = {key: 0.5 for key in HISTORY_KEYS}
            predictions[(0, 0)] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                exact_risk(predictions)
        for invalid in (True, False, "0.5", None):
            predictions = {key: 0.5 for key in HISTORY_KEYS}
            predictions[(0, 0)] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                exact_risk(predictions)

    def test_summary_is_serializable_and_identifies_exact_reference_values(self):
        summary = reference_summary()
        restored = json.loads(json.dumps(summary, allow_nan=False))
        self.assertEqual(restored["benchmark_id"], BENCHMARK_ID)
        self.assertEqual(restored["bayes_mse"]["numerator"], 9)
        self.assertEqual(restored["bayes_mse"]["denominator"], 160)
        self.assertEqual(restored["best_constant_mse"]["numerator"], 5)
        self.assertEqual(restored["best_constant_mse"]["denominator"], 32)
        excess = restored["rounded_bayes_excess_mse"]
        self.assertGreater(Fraction(excess["numerator"], excess["denominator"]), 0)
        self.assertEqual(len(restored["conditional_mean_table"]), 16)


class PredictionTableTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {(a, b): (4 * a + b) / 15 for a, b in HISTORY_KEYS}
        self.predictions[(0, 0)] = -0.0
        self.table = prediction_table_from_mapping(self.predictions, "test_candidate")

    def test_lossless_hex_round_trip_including_signed_zero(self):
        restored = parse_prediction_table(json.loads(json.dumps(self.table, allow_nan=False)))
        self.assertEqual(list(restored), list(HISTORY_KEYS))
        for key in HISTORY_KEYS:
            self.assertEqual(restored[key].hex(), self.predictions[key].hex())
        self.assertEqual(exact_risk(restored), exact_risk(self.predictions))
        self.assertEqual(self.table["predictions"][0]["prediction_hex"], "-0x0.0p+0")

    def test_serializer_orders_histories_and_parser_accepts_reordering(self):
        reversed_mapping = dict(reversed(list(self.predictions.items())))
        table = prediction_table_from_mapping(reversed_mapping, "test_candidate")
        self.assertEqual([entry["history_id"] for entry in table["predictions"]], list(range(16)))
        table["predictions"].reverse()
        self.assertEqual(list(parse_prediction_table(table)), list(HISTORY_KEYS))

    def test_serializer_requires_actual_floats(self):
        for value in (0, Fraction(1, 2), True):
            predictions = dict(self.predictions)
            predictions[(0, 0)] = value
            with self.subTest(value=value), self.assertRaises(TypeError):
                prediction_table_from_mapping(predictions, "candidate")

    def test_schema_and_candidate_identity_validation(self):
        for value in ("", " candidate", "candidate\n", "bad\x00id"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prediction_table_from_mapping(self.predictions, value)
        for value in (None, True, 2):
            with self.subTest(value=value), self.assertRaises(TypeError):
                prediction_table_from_mapping(self.predictions, value)
        for change in ({"schema": "wrong"}, {"extra": 1}):
            table = copy.deepcopy(self.table)
            table.update(change)
            with self.assertRaises(ValueError):
                parse_prediction_table(table)
        table = copy.deepcopy(self.table)
        del table["candidate_id"]
        with self.assertRaises(ValueError):
            parse_prediction_table(table)
        with self.assertRaises(TypeError):
            parse_prediction_table([])

    def test_full_support_and_id_state_agreement(self):
        table = copy.deepcopy(self.table)
        table["predictions"].pop()
        with self.assertRaises(ValueError):
            parse_prediction_table(table)
        table = copy.deepcopy(self.table)
        table["predictions"][-1] = copy.deepcopy(table["predictions"][0])
        with self.assertRaises(ValueError):
            parse_prediction_table(table)
        for field, value, exception in (
            ("history_id", 1, ValueError),
            ("history_id", False, TypeError),
            ("previous_state", 4, ValueError),
            ("previous_state", False, TypeError),
            ("current_state", "0", TypeError),
        ):
            table = copy.deepcopy(self.table)
            table["predictions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(exception):
                parse_prediction_table(table)

    def test_malformed_prediction_entries(self):
        for encoded in ("nan", "inf", "-inf", "0.1", " 0x1.0000000000000p-1", "0x1.0000000000000p+1024", "-0x1.0000000000000p-1", "0x1.8000000000000p+0"):
            table = copy.deepcopy(self.table)
            table["predictions"][0]["prediction_hex"] = encoded
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                parse_prediction_table(table)
        table = copy.deepcopy(self.table)
        table["predictions"][0]["prediction_hex"] = 0.5
        with self.assertRaises(TypeError):
            parse_prediction_table(table)
        for remove in (True, False):
            table = copy.deepcopy(self.table)
            if remove:
                del table["predictions"][0]["prediction_hex"]
            else:
                table["predictions"][0]["extra"] = 0
            with self.assertRaises(ValueError):
                parse_prediction_table(table)


if __name__ == "__main__":
    unittest.main()

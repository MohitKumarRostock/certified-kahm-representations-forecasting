"""Exact development-selection tests; no model population risks or calibration."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import unittest

import numpy as np

from kahm_demo.geometry import fit_standardizer
from kahm_demo.law import Episode
from kahm_demo.selection import (empirical_counts, empirical_mse, empirical_mse_from_counts,
                                fraction_record, parse_fraction_record, select_landmarks,
                                select_trial_records)


class LandmarkTests(unittest.TestCase):
    def test_all_distinct_histories_retain_first_identity_and_occurrence_counts(self):
        X = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 0.], [0., 1.]])
        plan = select_landmarks(X, fit_standardizer(X))
        self.assertEqual(plan.row_indices, (3, 1, 0))
        self.assertEqual(plan.occurrence_counts, (1, 2, 2))
        np.testing.assert_array_equal(plan.raw_histories, [[0., 0.], [0., 1.], [1., 0.]])
        self.assertEqual((plan.training_count, plan.distinct_count, plan.budget), (5, 3, 16))
        self.assertTrue(plan.record()["shortage_reduction"])
        # All candidate encodings consume these identities even when their
        # encoded values collide; the plan contains no encoder-dependent input.
        collapsed_encoding = np.zeros((len(plan.row_indices), 2))
        self.assertEqual(len(collapsed_encoding), 3)
        self.assertFalse(plan.record()["encoded_landmark_duplicates_removed"])

    def test_budget_binding_farthest_first_and_ties_are_deterministic(self):
        X = np.array([[3.], [0.], [1.], [2.], [4.], [5.]])
        scaler = fit_standardizer(X)
        plan = select_landmarks(X, scaler, budget=3)
        self.assertEqual(plan.row_indices, (1, 5, 3))
        np.testing.assert_array_equal(plan.raw_histories[:, 0], [0., 5., 2.])
        self.assertEqual(plan.record(), select_landmarks(X, scaler, budget=3).record())

    def test_farthest_distances_use_standardized_raw_coordinates(self):
        # The large raw scale would select (0,1000) second. Training scaling
        # makes (1,900) the farthest row from (0,0).
        X = np.array([[0., 0.], [0., 1000.], [1., 900.]])
        plan = select_landmarks(X, fit_standardizer(X), budget=2)
        self.assertEqual(plan.row_indices, (0, 2))

    def test_plan_is_copied_immutable_and_losslessly_recorded(self):
        X = np.array([[0., 0.], [1., 1.]])
        plan = select_landmarks(X, fit_standardizer(X))
        X[:] = 99.
        np.testing.assert_array_equal(plan.raw_histories, [[0., 0.], [1., 1.]])
        with self.assertRaises(ValueError):
            plan.raw_histories.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            plan.budget = 4
        record = json.loads(json.dumps(plan.record(), allow_nan=False))
        restored = [[float.fromhex(value) for value in row] for row in record["raw_histories_hex"]]
        np.testing.assert_array_equal(restored, plan.raw_histories)

    def test_bad_inputs_are_rejected(self):
        X = [[0., 0.], [1., 1.]]
        scaler = fit_standardizer(X)
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select_landmarks(X, scaler, budget)
        with self.assertRaises(ValueError):
            select_landmarks([[0., 0.]], scaler)
        with self.assertRaises(TypeError):
            select_landmarks(X, None)


class EmpiricalLossTests(unittest.TestCase):
    def test_aggregate_matches_direct_per_episode_exact_float_loss(self):
        triples = [(0, 1, 0), (0, 1, 3), (0, 1, 3), (1, 0, 1), (2, 3, 2), (2, 3, 2)]
        episodes = [Episode(i, *triple) for i, triple in enumerate(triples)]
        predictions = {(0, 1): 0.1, (1, 0): 0.7, (2, 3): 0.25}
        direct = sum((Fraction.from_float(e.y) - Fraction.from_float(predictions[(e.previous_state, e.current_state)])) ** 2
                     for e in episodes) / len(episodes)
        counts = empirical_counts(episodes)
        self.assertEqual(sum(counts.values()), len(episodes))
        self.assertEqual(empirical_mse(episodes, predictions), direct)
        self.assertEqual(empirical_mse_from_counts(counts, predictions), direct)

    def test_duplicates_receive_full_weight(self):
        episodes = [Episode(0, 0, 0, 0)] + [Episode(i, 0, 0, 3) for i in range(1, 10)]
        self.assertEqual(empirical_counts(episodes), {(0, 0, 0): 1, (0, 0, 3): 9})
        self.assertEqual(empirical_mse(episodes, {(0, 0): 0.0}), Fraction(9, 10))
        self.assertNotEqual(empirical_mse(episodes, {(0, 0): 0.0}), Fraction(1, 2))

    def test_actual_binary_float_is_not_rationalized_decimal(self):
        risk = empirical_mse([Episode(0, 0, 0, 0)], {(0, 0): 0.1})
        self.assertEqual(risk, Fraction.from_float(0.1) ** 2)
        self.assertNotEqual(risk, Fraction(1, 100))

    def test_empty_malformed_counts_missing_and_invalid_predictions(self):
        with self.assertRaises(ValueError):
            empirical_counts([])
        for counts in ({}, {(0, 0, 0): 0}, {(0, 0, 0): -1}, {(0, 0, 0): True}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                empirical_mse_from_counts(counts, {(0, 0): 0.5})
        with self.assertRaises(TypeError):
            empirical_mse_from_counts({(False, 0, 0): 1}, {(0, 0): 0.5})
        with self.assertRaises(ValueError):
            empirical_mse_from_counts({(0, 0, 4): 1}, {(0, 0): 0.5})
        with self.assertRaises(ValueError):
            empirical_mse_from_counts({(0, 0, 0): 1}, {})
        for value, error in ((-0.1, ValueError), (1.1, ValueError), (float("nan"), ValueError),
                             (float("inf"), ValueError), (1, TypeError), (Fraction(1, 2), TypeError)):
            with self.subTest(value=value), self.assertRaises(error):
                empirical_mse_from_counts({(0, 0, 0): 1}, {(0, 0): value})


class FractionRecordTests(unittest.TestCase):
    def test_round_trip_retains_tiny_exact_difference(self):
        for value in (Fraction(0), Fraction(-1, 3), Fraction(9, 10), Fraction(1, 10) + Fraction(1, 2 ** 1000)):
            record = json.loads(json.dumps(fraction_record(value), allow_nan=False))
            self.assertEqual(parse_fraction_record(record), value)

    def test_noncanonical_records_and_inconsistent_display_are_rejected(self):
        for change in ({"numerator": "+1"}, {"numerator": "01"}, {"denominator": "02"},
                       {"numerator": "2", "denominator": "4"}, {"denominator": "0"},
                       {"denominator": "-2"}, {"decimal_approximation": 0.6},
                       {"decimal_approximation": float("nan")}, {"decimal_approximation": 0},
                       {"extra": 1}):
            record = fraction_record(Fraction(1, 2))
            record.update(change)
            with self.subTest(change=change), self.assertRaises((TypeError, ValueError)):
                parse_fraction_record(record)
        record = fraction_record(Fraction(0))
        record["decimal_approximation"] = -0.0
        with self.assertRaises(ValueError):
            parse_fraction_record(record)


def trial(name, candidate, bank, grid, loss):
    return {"trial_id": name, "candidate_id": candidate, "bank_id": bank,
            "grid_index": grid, "validation_mse": fraction_record(Fraction(loss))}


class SelectionTests(unittest.TestCase):
    def records(self):
        return [trial("raw", "raw", None, 0, Fraction(1, 5)),
                trial("mem1", "kahm_mem", "cap_1", 0, Fraction(1, 100)),
                trial("local1", "kahm_local", "cap_1", 0, Fraction(49, 100)),
                trial("mem2", "kahm_mem", "cap_2", 0, Fraction(1, 10)),
                trial("local2", "kahm_local", "cap_2", 0, Fraction(1, 10))]

    def test_shared_bank_uses_mean_of_within_bank_minima(self):
        result = select_trial_records(self.records(), ("cap_1", "cap_2"))
        self.assertEqual(result["shared_bank_id"], "cap_2")
        self.assertEqual((result["mem_trial_id"], result["local_trial_id"]), ("mem2", "local2"))
        self.assertEqual(parse_fraction_record(result["per_bank"][0]["mean_validation_mse"]), Fraction(1, 4))
        self.assertEqual(parse_fraction_record(result["per_bank"][1]["mean_validation_mse"]), Fraction(1, 10))

    def test_decoder_ties_use_grid_index_not_record_order(self):
        records = self.records()
        records.insert(0, trial("raw_later", "raw", None, 10, Fraction(1, 5)))
        records.insert(0, trial("mem2_later", "kahm_mem", "cap_2", 20, Fraction(1, 10)))
        result = select_trial_records(records, ("cap_1", "cap_2"))
        self.assertEqual(result["raw_trial_id"], "raw")
        self.assertEqual(result["mem_trial_id"], "mem2")

    def test_bank_ties_use_explicit_bank_order(self):
        records = self.records()
        records[1]["validation_mse"] = fraction_record(Fraction(1, 10))
        records[2]["validation_mse"] = fraction_record(Fraction(1, 10))
        result = select_trial_records(list(reversed(records)), ("cap_2", "cap_1"))
        self.assertEqual(result["shared_bank_id"], "cap_2")

    def test_near_ties_compare_exact_rationals_not_float_displays(self):
        records = self.records()
        base = Fraction(1, 5)
        records[0]["validation_mse"] = fraction_record(base + Fraction(1, 2 ** 1000))
        records.append(trial("raw_exactly_better", "raw", None, 44, base))
        self.assertEqual(records[0]["validation_mse"]["decimal_approximation"],
                         records[-1]["validation_mse"]["decimal_approximation"])
        self.assertEqual(select_trial_records(records, ("cap_1", "cap_2"))["raw_trial_id"], "raw_exactly_better")

    def test_raw_selection_is_independent_of_kahm_bank_scores(self):
        records = self.records()
        records.append(trial("raw_better", "raw", None, 3, Fraction(1, 1000)))
        result = select_trial_records(records, ("cap_1", "cap_2"))
        self.assertEqual(result["raw_trial_id"], "raw_better")
        self.assertEqual(result["shared_bank_id"], "cap_2")

    def test_best_decoder_is_selected_within_each_bank_before_bank_comparison(self):
        records = self.records()
        records.extend([trial("mem1_best", "kahm_mem", "cap_1", 4, Fraction(1, 200)),
                        trial("local1_best", "kahm_local", "cap_1", 8, Fraction(1, 100))])
        result = select_trial_records(records, ("cap_1", "cap_2"))
        self.assertEqual(result["shared_bank_id"], "cap_1")
        self.assertEqual((result["mem_trial_id"], result["local_trial_id"]), ("mem1_best", "local1_best"))

    def test_duplicate_ids_grid_states_and_invalid_variant_bank_pairs_are_rejected(self):
        for extra in (self.records()[0], trial("raw_duplicate_grid", "raw", None, 0, 0),
                      trial("bad_raw", "raw", "cap_1", 1, 0),
                      trial("missing_bank", "kahm_mem", None, 1, 0),
                      trial("foreign_bank", "kahm_mem", "cap_3", 1, 0),
                      trial("bad_grid", "raw", None, 45, 0),
                      trial("bad_loss", "raw", None, 1, Fraction(-1, 10))):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                select_trial_records(self.records() + [extra], ("cap_1", "cap_2"))
        with self.assertRaises(ValueError):
            select_trial_records(self.records()[:-1], ("cap_1", "cap_2"))
        with self.assertRaises(ValueError):
            select_trial_records(self.records(), ("cap_1", "cap_1"))


if __name__ == "__main__":
    unittest.main()

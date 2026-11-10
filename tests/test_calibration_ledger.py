"""Exact loss-ledger tests use only dedicated synthetic calibration streams."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kahm_demo import calibration_ledger as ledger
from kahm_demo.contract import CANDIDATE_IDS, PRIMARY_CALIBRATION_COUNT
from kahm_demo.datasets import Dataset
from kahm_demo.law import Episode, HISTORY_KEYS, OBSERVATIONS
from kahm_demo.sampling import sample_episodes


def rational(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


class CalibrationLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.dataset = Dataset(
            "calibration", "test-calibration-ledger-v1",
            sample_episodes(PRIMARY_CALIBRATION_COUNT, 8675309, "calibration",
                            "test-calibration-ledger-v1"),
        )
        nearby = (0.1, math.nextafter(0.25, 1.0), math.nextafter(0.75, 0.0), 0.5)
        cls.predictions = {
            "raw": {history: 0.0 for history in HISTORY_KEYS},
            "kahm_mem": {history: 1.0 for history in HISTORY_KEYS},
            "kahm_local": {history: nearby[(4 * history[0] + history[1]) % 4]
                           for history in HISTORY_KEYS},
        }
        cls.fixture_path = Path(cls.fixture_directory.name) / "ledger.jsonl"
        cls.fixture_summary = ledger.write_calibration_ledger(
            cls.dataset, cls.predictions, cls.fixture_path
        )
        cls.fixture_lines = cls.fixture_path.read_text().splitlines(keepends=True)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_directory.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "ledger.jsonl"

    def tearDown(self):
        self.directory.cleanup()

    def changed_row(self, change, index=0):
        rows = list(self.fixture_lines)
        row = json.loads(rows[index])
        change(row)
        rows[index] = json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n"
        self.path.write_text("".join(rows))

    def assert_ledger_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            ledger.verify_calibration_ledger(self.dataset, self.predictions, self.path)

    def test_roundtrip_retains_every_occurrence_and_exact_weighted_loss(self):
        summary = ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
        self.assertEqual(summary, ledger.verify_calibration_ledger(
            self.dataset, self.predictions, self.path))
        self.assertEqual(set(summary), {
            "schema", "run_id", "count", "candidate_ids", "loss_sums", "empirical_risks",
        })
        self.assertEqual(summary["schema"], ledger.LEDGER_SUMMARY_SCHEMA)
        self.assertEqual(summary["run_id"], self.dataset.run_id)
        self.assertEqual(summary["count"], 4096)
        self.assertEqual(summary["candidate_ids"], list(CANDIDATE_IDS))

        # Independent grouped arithmetic uses actual occurrence counts and
        # as_integer_ratio, not the writer's row-by-row Fraction.from_float path.
        counts = Counter((e.previous_state, e.current_state, e.next_state)
                         for e in self.dataset.episodes)
        self.assertLessEqual(len(counts), 64)
        self.assertEqual(sum(counts.values()), 4096)
        for candidate in CANDIDATE_IDS:
            expected_sum = Fraction(0)
            for (a, b, c), count in counts.items():
                predicted = Fraction(*self.predictions[candidate][(a, b)].as_integer_ratio())
                difference = OBSERVATIONS[c] - predicted
                expected_sum += count * difference * difference
            self.assertEqual(rational(summary["loss_sums"][candidate]), expected_sum)
            self.assertEqual(rational(summary["empirical_risks"][candidate]), expected_sum / 4096)
            for kind in ("loss_sums", "empirical_risks"):
                self.assertTrue(all(type(value) is str for value in summary[kind][candidate].values()))

        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([row["episode_id"] for row in rows], list(range(4096)))
        self.assertEqual(len(rows), len(self.dataset.episodes))
        for index in (0, 101, 4095):
            episode, row = self.dataset.episodes[index], rows[index]
            self.assertEqual(row["schema"], ledger.LEDGER_ROW_SCHEMA)
            self.assertEqual(row["history_id"], 4 * episode.previous_state + episode.current_state)
            self.assertEqual(rational(row["target"]), OBSERVATIONS[episode.next_state])
            self.assertEqual(tuple(row["prediction_hex"]), tuple(CANDIDATE_IDS))
            self.assertEqual(tuple(row["squared_losses"]), tuple(CANDIDATE_IDS))

    def test_subnormal_near_one_and_signed_zero_keep_their_exact_values(self):
        dataset = Dataset("calibration", "test-ledger-edge-values", tuple(
            Episode(i, (i // 4) % 4, i % 4, 0) for i in range(4096)
        ))
        values = {"raw": -0.0, "kahm_mem": math.nextafter(0.0, 1.0),
                  "kahm_local": math.nextafter(1.0, 0.0)}
        predictions = {candidate: {history: value for history in HISTORY_KEYS}
                       for candidate, value in values.items()}
        summary = ledger.write_calibration_ledger(dataset, predictions, self.path)
        self.assertEqual(summary, ledger.verify_calibration_ledger(dataset, predictions, self.path))
        self.assertEqual(rational(summary["empirical_risks"]["raw"]), 0)
        self.assertEqual(rational(summary["empirical_risks"]["kahm_mem"]), Fraction(1, 2 ** 2148))
        self.assertEqual(rational(summary["empirical_risks"]["kahm_local"]),
                         Fraction((2 ** 53 - 1) ** 2, 2 ** 106))
        with self.path.open() as stream:
            first = json.loads(stream.readline())
        self.assertEqual(first["prediction_hex"]["raw"], "-0x0.0p+0")
        self.assertEqual(first["squared_losses"]["raw"], {"numerator": "0", "denominator": "1"})

    def test_float_one_tenth_is_not_reinterpreted_as_rational_one_tenth(self):
        dataset = Dataset("calibration", "test-ledger-decimal-values", tuple(
            Episode(i, 0, 0, 0) for i in range(4096)
        ))
        predictions = {candidate: {history: 0.1 for history in HISTORY_KEYS}
                       for candidate in CANDIDATE_IDS}
        summary = ledger.write_calibration_ledger(dataset, predictions, self.path)
        expected = Fraction(*0.1.as_integer_ratio()) ** 2
        self.assertNotEqual(expected, Fraction(1, 100))
        for candidate in CANDIDATE_IDS:
            self.assertEqual(rational(summary["empirical_risks"][candidate]), expected)

    def test_input_candidate_order_is_normalized(self):
        predictions = {candidate: self.predictions[candidate]
                       for candidate in reversed(CANDIDATE_IDS)}
        summary = ledger.write_calibration_ledger(self.dataset, predictions, self.path)
        self.assertEqual(summary, self.fixture_summary)
        self.assertEqual(self.path.read_bytes(), self.fixture_path.read_bytes())
        self.assertEqual(summary, ledger.verify_calibration_ledger(self.dataset, predictions, self.path))

    def test_invalid_dataset_role_count_and_ids_fail_before_output_creation(self):
        episodes = self.dataset.episodes
        first = episodes[0]
        wrong_id = Episode(1, first.previous_state, first.current_state, first.next_state)
        cases = [
            None,
            Dataset("training", "test-ledger-invalid", episodes),
            Dataset("validation", "test-ledger-invalid", episodes),
            Dataset("calibration", "", episodes),
            Dataset("calibration", "test-ledger-invalid", episodes[:-1]),
            Dataset("calibration", "test-ledger-invalid", episodes + (episodes[-1],)),
            Dataset("calibration", "test-ledger-invalid", list(episodes)),
            Dataset("calibration", "test-ledger-invalid", (wrong_id,) + episodes[1:]),
            Dataset("calibration", "test-ledger-invalid", (object(),) + episodes[1:]),
        ]
        output = self.root / "must-not-be-created" / "ledger.jsonl"
        for dataset in cases:
            with self.subTest(dataset_type=type(dataset), role=getattr(dataset, "role", None)):
                with self.assertRaises((TypeError, ValueError)):
                    ledger.write_calibration_ledger(dataset, self.predictions, output)
                with self.assertRaises((TypeError, ValueError)):
                    ledger.verify_calibration_ledger(dataset, self.predictions, self.fixture_path)
                self.assertFalse(output.parent.exists())

    def test_invalid_prediction_families_and_history_keys_are_rejected(self):
        missing_candidate = dict(self.predictions)
        del missing_candidate["raw"]
        extra_candidate = {**self.predictions, "other": self.predictions["raw"]}
        tables = []
        missing_history = dict(self.predictions["raw"])
        del missing_history[(0, 0)]
        tables.append(missing_history)
        for key in ((False, 0), (0.0, 0), "0,0", (4, 0), (0, 0, 0)):
            changed = dict(self.predictions["raw"])
            del changed[(0, 0)]
            changed[key] = 0.0
            tables.append(changed)
        cases = [missing_candidate, extra_candidate]
        cases.extend({**self.predictions, "raw": table} for table in tables)
        for predictions in cases:
            with self.subTest(predictions=repr(predictions)[:80]):
                with self.assertRaises((TypeError, ValueError)):
                    ledger.write_calibration_ledger(self.dataset, predictions, self.path)
                with self.assertRaises((TypeError, ValueError)):
                    ledger.verify_calibration_ledger(self.dataset, predictions, self.fixture_path)
                self.assertFalse(self.path.exists())

    def test_predictions_require_finite_bounded_actual_floats(self):
        for value in (0, True, Fraction(1, 2), "0.5", float("nan"), float("inf"),
                      -float("inf"), -math.ulp(0.0), math.nextafter(1.0, 2.0)):
            predictions = {candidate: dict(table) for candidate, table in self.predictions.items()}
            predictions["raw"][(0, 0)] = value
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    ledger.write_calibration_ledger(self.dataset, predictions, self.path)
                with self.assertRaises((TypeError, ValueError)):
                    ledger.verify_calibration_ledger(self.dataset, predictions, self.fixture_path)
                self.assertFalse(self.path.exists())

    def test_existing_file_and_dangling_symlink_are_not_replaced(self):
        self.path.write_bytes(b"existing owner's bytes\n")
        with self.assertRaises(FileExistsError):
            ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
        self.assertEqual(self.path.read_bytes(), b"existing owner's bytes\n")
        self.path.unlink()
        self.path.symlink_to(self.root / "missing-target")
        with self.assertRaises(FileExistsError):
            ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
        self.assertTrue(self.path.is_symlink())

    def test_atomic_commit_preserves_a_concurrent_destination(self):
        real_link = ledger.os.link

        def race(source, destination):
            Path(destination).write_bytes(b"concurrent writer\n")
            return real_link(source, destination)

        with patch.object(ledger.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
        self.assertEqual(self.path.read_bytes(), b"concurrent writer\n")
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_failed_write_or_commit_removes_temporary_files(self):
        real_dumps = ledger.json.dumps
        counter = 0

        def fail_during_write(*args, **kwargs):
            nonlocal counter
            counter += 1
            if counter == 4:
                raise OSError("test-only write failure")
            return real_dumps(*args, **kwargs)

        with patch.object(ledger.json, "dumps", side_effect=fail_during_write):
            with self.assertRaisesRegex(OSError, "write failure"):
                ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
        self.assertEqual(list(self.root.iterdir()), [])
        with patch.object(ledger.os, "link", side_effect=OSError("test-only commit failure")):
            with self.assertRaisesRegex(OSError, "commit failure"):
                ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_changed_loss_or_prediction_is_rejected(self):
        def change_loss(row):
            value = row["squared_losses"]["raw"]
            value["numerator"] = str(int(value["numerator"]) + 1)

        self.changed_row(change_loss)
        self.assert_ledger_rejected()
        for encoded in (0.25.hex(), "0x0p+0", "-0x0.0p+0", "nan", "inf", "0x1.0p+8", 0.0):
            with self.subTest(encoded=encoded):
                self.changed_row(lambda row: row["prediction_hex"].__setitem__("raw", encoded))
                self.assert_ledger_rejected()

    def test_changed_occurrence_states_ids_or_run_are_rejected(self):
        first = json.loads(self.fixture_lines[0])
        cases = {
            "episode_id": 1,
            "history_id": (first["history_id"] + 1) % 16,
            "previous_state": (first["previous_state"] + 1) % 4,
            "current_state": (first["current_state"] + 1) % 4,
            "next_state": (first["next_state"] + 1) % 4,
            "run_id": "test-another-run",
        }
        for key, value in cases.items():
            with self.subTest(field=key):
                self.changed_row(lambda row: row.__setitem__(key, value))
                self.assert_ledger_rejected()
        for value in (False, 0.0):
            self.changed_row(lambda row: row.__setitem__("episode_id", value))
            self.assert_ledger_rejected()

    def test_dropped_duplicated_reordered_or_extra_rows_are_rejected(self):
        reordered = list(self.fixture_lines)
        reordered[-2], reordered[-1] = reordered[-1], reordered[-2]
        cases = [
            self.fixture_lines[:-1],
            [self.fixture_lines[0]] + self.fixture_lines[:-1],
            reordered,
            self.fixture_lines + [self.fixture_lines[-1]],
            self.fixture_lines + ["\n"],
        ]
        for rows in cases:
            with self.subTest(rows=len(rows)):
                self.path.write_text("".join(rows))
                self.assert_ledger_rejected()

    def test_rational_records_must_be_canonical_strings_in_lowest_terms(self):
        target = json.loads(self.fixture_lines[0])["target"]
        numerator, denominator = int(target["numerator"]), int(target["denominator"])
        cases = [
            {"numerator": numerator, "denominator": str(denominator)},
            {"numerator": str(numerator), "denominator": denominator},
            {"numerator": f"+{numerator}", "denominator": str(denominator)},
            {"numerator": f"0{numerator}", "denominator": str(denominator)},
            {"numerator": str(2 * numerator), "denominator": str(2 * denominator)},
            {"numerator": str(numerator), "denominator": "0"},
            {"numerator": str(numerator), "denominator": "-1"},
            {"numerator": "0.0", "denominator": "1"},
            {"numerator": "NaN", "denominator": "1"},
            {"numerator": "0", "denominator": "1", "decimal": 0.0},
        ]
        for record in cases:
            with self.subTest(record=record):
                self.changed_row(lambda row: row.__setitem__("target", record))
                self.assert_ledger_rejected()
        self.changed_row(lambda row: row["squared_losses"].__setitem__(
            "raw", {"numerator": "0", "denominator": "2"}))
        self.assert_ledger_rejected()

    def test_json_shape_duplicates_nonfinite_values_and_candidate_order_are_checked(self):
        for change in (
            lambda row: row.__setitem__("unexpected", 1),
            lambda row: row.pop("target"),
            lambda row: row.__setitem__("episode_id", float("nan")),
            lambda row: row.__setitem__("prediction_hex", dict(reversed(list(row["prediction_hex"].items())))),
            lambda row: row.__setitem__("squared_losses", dict(reversed(list(row["squared_losses"].items())))),
        ):
            self.changed_row(change)
            self.assert_ledger_rejected()
        rows = list(self.fixture_lines)
        rows[0] = rows[0].replace('"episode_id":0', '"episode_id":0,"episode_id":0', 1)
        self.path.write_text("".join(rows))
        self.assert_ledger_rejected()
        for first in ("null\n", "{}\n", "invalid JSON\n"):
            self.path.write_text(first + "".join(self.fixture_lines[1:]))
            self.assert_ledger_rejected()

    def test_verifier_binds_to_supplied_data_and_predictions(self):
        first = self.dataset.episodes[0]
        modified = Episode(0, first.previous_state, first.current_state, (first.next_state + 1) % 4)
        changed_data = Dataset("calibration", self.dataset.run_id,
                               (modified,) + self.dataset.episodes[1:])
        with self.assertRaises(ValueError):
            ledger.verify_calibration_ledger(changed_data, self.predictions, self.fixture_path)
        changed_predictions = {candidate: dict(table) for candidate, table in self.predictions.items()}
        changed_predictions["raw"][(first.previous_state, first.current_state)] = 0.5
        with self.assertRaises(ValueError):
            ledger.verify_calibration_ledger(self.dataset, changed_predictions, self.fixture_path)

    def test_no_sampling_loading_or_population_risk_calls(self):
        forbidden = AssertionError("ledger must not access calibration or population-risk services")
        with (patch("kahm_demo.sampling.sample_episodes", side_effect=forbidden),
              patch("kahm_demo.datasets.sample_episodes", side_effect=forbidden),
              patch("kahm_demo.datasets.load_calibration", side_effect=forbidden),
              patch("kahm_demo.oracle.exact_risk", side_effect=forbidden),
              patch("kahm_demo.oracle.risk_of_predictor", side_effect=forbidden)):
            expected = ledger.write_calibration_ledger(self.dataset, self.predictions, self.path)
            self.assertEqual(expected, ledger.verify_calibration_ledger(
                self.dataset, self.predictions, self.path))


if __name__ == "__main__":
    unittest.main()

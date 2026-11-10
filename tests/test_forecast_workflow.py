"""Development-search integration, artifact integrity, and evaluation boundaries."""

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from kahm_demo._util import read_json, sha256_file, write_json
from kahm_demo.datasets import generate_bundle, load_development
from kahm_demo.encoder_state import encoder_state_fingerprint
from kahm_demo.encoder_workflow import fit_development_encoders
from kahm_demo.forecast_state import forecaster_state_fingerprint, load_forecaster
from kahm_demo.forecast_workflow import (fit_development_forecasters, verify_forecast_bundle,
                                        _check_training_binding)
from kahm_demo.selection import parse_fraction_record, select_landmarks


class ForecastWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.data, cls.encoders, cls.models = [cls.root / name for name in ("data", "encoders", "models")]
        generate_bundle(cls.data, master_seed=198443, run_id="test-step4-integration",
                        training_count=128, validation_count=64)
        fit_development_encoders(cls.data, cls.encoders)
        forbidden = AssertionError("calibration or population-risk access during development")
        with (mock.patch("kahm_demo.oracle.exact_risk", side_effect=forbidden),
              mock.patch("kahm_demo.oracle.risk_of_predictor", side_effect=forbidden),
              mock.patch("kahm_demo.datasets.load_calibration", side_effect=forbidden),
              mock.patch("kahm_demo.datasets.materialize_calibration", side_effect=forbidden),
              mock.patch("kahm_demo.datasets.sample_episodes", side_effect=forbidden),
              mock.patch("kahm_demo.sampling.sample_episodes", side_effect=forbidden)):
            cls.result = fit_development_forecasters(cls.data, cls.encoders, cls.models)
        cls.manifest = read_json(cls.models / "manifest.json")

    def copied_models(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        target = Path(temporary.name) / "models"
        shutil.copytree(self.models, target)
        return target

    def rewrite_search(self, output, callback):
        manifest = read_json(output / "manifest.json")
        path = output / manifest["search_path"]
        search = read_json(path)
        callback(search)
        write_json(path, search)
        manifest["search_sha256"] = sha256_file(path)
        write_json(output / "manifest.json", manifest)

    def verify(self, output):
        return verify_forecast_bundle(output, dataset_bundle=self.data, encoder_bundle=self.encoders)

    def test_complete_roster_states_and_reserved_evaluation_boundary(self):
        self.assertEqual(self.result["search_count"], 225)
        self.assertEqual(len(list((self.models / "search_states").glob("*.json"))), 225)
        search = read_json(self.models / "search.json")
        self.assertEqual({candidate: sum(row["candidate_id"] == candidate for row in search["records"])
                          for candidate in ("raw", "kahm_mem", "kahm_local")},
                         {"raw": 45, "kahm_mem": 90, "kahm_local": 90})
        self.assertFalse((self.data / "calibration").exists())
        self.assertFalse(self.manifest["final_freeze_created"])
        self.assertFalse(self.manifest["statistical_certificate_issued"])
        self.assertFalse(self.manifest["population_risk_evaluated"])
        self.assertEqual(self.manifest["status"], "development_selected_not_frozen")
        first = self.models / search["records"][0]["model_state_path"]
        self.assertEqual(read_json(first)["status"], "development_candidate")

    def test_selected_models_share_bank_landmarks_and_membership_scaling(self):
        raw, mem, local = [load_forecaster(self.models / f"{name}.json")
                           for name in ("raw", "kahm_mem", "kahm_local")]
        self.assertEqual(mem.bank_id, local.bank_id)
        self.assertEqual(encoder_state_fingerprint(mem.encoder), encoder_state_fingerprint(local.encoder))
        self.assertEqual(raw.readout.landmark_indices, mem.readout.landmark_indices)
        self.assertEqual(mem.readout.landmark_indices, local.readout.landmark_indices)
        width = mem.readout.input_width
        for key in ("mean", "scale", "raw_std"):
            np.testing.assert_array_equal(getattr(mem.readout.linear_scaler, key),
                                          getattr(local.readout.linear_scaler, key)[:width])

    def test_exact_selected_validation_losses_equal_direct_episode_sum(self):
        validation = load_development(self.data, "validation")
        for record in self.manifest["selected_models"]:
            model = load_forecaster(self.models / record["model_path"])
            losses = [(Fraction.from_float(model.predict_one(episode.x)) - Fraction.from_float(episode.y)) ** 2
                      for episode in validation.episodes]
            direct = sum(losses, Fraction()) / len(losses)
            self.assertEqual(direct, parse_fraction_record(record["validation_mse"]))

    def test_scalar_batch_inference_preserves_rows_without_refitting(self):
        histories = [[0.25, 0.75], [0, 1], [0.25, 0.75], [1, 0]]
        for name in ("raw", "kahm_mem", "kahm_local"):
            model = load_forecaster(self.models / f"{name}.json")
            before = forecaster_state_fingerprint(model)
            expected = [model.predict_one(row) for row in histories]
            np.testing.assert_array_equal(model.predict(histories), expected)
            self.assertEqual(forecaster_state_fingerprint(model), before)
            self.assertTrue(all(0 <= value <= 1 for value in expected))
        with self.assertRaises(TypeError):
            model.predict(np.asarray([[1j, 0j]]))

    def test_model_is_self_contained_for_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            single = Path(directory) / "model.json"
            shutil.copyfile(self.models / "kahm_local.json", single)
            with mock.patch("kahm_demo.forecast_workflow.load_development", side_effect=AssertionError("data access")):
                model = load_forecaster(single)
                value = model.predict_one([0.25, 0.75])
            self.assertTrue(0 <= value <= 1)

    def test_missing_search_candidate_is_rejected_even_with_updated_hash(self):
        output = self.copied_models()
        self.rewrite_search(output, lambda search: search["records"].pop())
        with self.assertRaisesRegex(ValueError, "roster"):
            self.verify(output)

    def test_wrong_grid_configuration_is_rejected(self):
        output = self.copied_models()
        self.rewrite_search(output, lambda search: search["records"][0].update(ridge_alpha=0.123))
        with self.assertRaisesRegex(ValueError, "roster"):
            self.verify(output)

    def test_unselected_candidate_rank_summary_is_bound_to_its_state(self):
        output = self.copied_models()
        self.rewrite_search(output, lambda search: search["records"][0].update(nystrom_rank=999))
        with self.assertRaisesRegex(ValueError, "summary/diagnostics"):
            self.verify(output)

    def test_exact_validation_loss_record_is_recomputed(self):
        output = self.copied_models()
        def alter(search):
            search["records"][0]["validation_mse"] = {"numerator": "0", "denominator": "1", "decimal_approximation": 0.0}
        self.rewrite_search(output, alter)
        with self.assertRaisesRegex(ValueError, "validation MSE"):
            self.verify(output)

    def test_raw_linked_training_counts_must_match_actual_data(self):
        model = load_forecaster(self.models / "raw.json")
        x, y = load_development(self.data, "training").xy()
        incorrect_count = len(x) + 1
        readout = replace(model.readout,
                          linear_scaler=replace(model.readout.linear_scaler, sample_count=incorrect_count),
                          nystrom_scaler=replace(model.readout.nystrom_scaler, sample_count=incorrect_count),
                          training_count=incorrect_count)
        changed = replace(model, raw_standardizer=replace(model.raw_standardizer, sample_count=incorrect_count),
                          readout=readout, ridge=replace(model.ridge, training_count=incorrect_count))
        self.assertEqual(model.predict_one([0.25, 0.75]).hex(), changed.predict_one([0.25, 0.75]).hex())
        landmarks = select_landmarks(x, model.raw_standardizer)
        with self.assertRaisesRegex(ValueError, "training counts"):
            _check_training_binding(changed, np.asarray(x), np.asarray(y), landmarks)

    def test_historical_calibration_and_certificate_claims_are_checked(self):
        for key, value, message in (("calibration_status", "used_for_selection", "calibration"),
                                    ("statistical_certificate_issued", True, "certificate")):
            with self.subTest(key=key):
                output = self.copied_models()
                manifest = read_json(output / "manifest.json")
                manifest[key] = value
                write_json(output / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify(output)

    def test_existing_outputs_and_materialized_calibration_prevent_development(self):
        with self.assertRaises(FileExistsError):
            fit_development_forecasters(self.data, self.encoders, self.models)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            shutil.copytree(self.data, data)
            (data / "calibration").mkdir()
            with self.assertRaisesRegex(ValueError, "calibration"):
                fit_development_forecasters(data, self.encoders, Path(directory) / "new_models")

    def test_forecast_cli_matches_the_importable_model(self):
        path = self.models / "kahm_local.json"
        result = subprocess.run([sys.executable, "-m", "kahm_demo", "forecast", "--model", str(path),
                                 "--history", "0.25", "0.75"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        expected = load_forecaster(path).predict_one([0.25, 0.75])
        self.assertEqual(output["prediction_hex"], expected.hex())
        self.assertEqual(output["status"], "development_selected_not_frozen")


if __name__ == "__main__":
    unittest.main()

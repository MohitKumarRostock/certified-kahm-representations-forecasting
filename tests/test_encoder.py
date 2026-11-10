"""End-to-end encoder checks for shared geometry, canonical inference, and storage."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from kahm_demo._util import canonical_json_bytes, read_json, sha256_file, write_json
from kahm_demo.datasets import generate_bundle, load_development
from kahm_demo.encoder import fit_encoder, fit_encoder_candidates
from kahm_demo.encoder_state import (encoder_state_fingerprint, load_encoder,
                                    save_encoder, validate_encoder)
from kahm_demo.encoder_workflow import (fit_development_encoders, support_feature_table,
                                       verify_encoder_bundle)
from kahm_demo.law import HISTORY_KEYS, OBSERVATION_FLOATS
from kahm_demo.soft_state import radial_angular_score


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = np.asarray([(OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b])
                                  for a, b in HISTORY_KEYS])
        # An asymmetric empirical distribution, still with the complete support.
        cls.training = np.concatenate([cls.support, cls.support[:5], cls.support[:3]])
        cls.banks = fit_encoder_candidates(cls.training)

    def test_caps_share_preprocessing_and_reference_objects(self):
        low, high = self.banks
        self.assertIs(low.standardizer, high.standardizer)
        self.assertIs(low.partitions, high.partitions)
        np.testing.assert_array_equal(low.standardizer.mean, self.training.mean(axis=0))
        for left, right in zip(low.models, high.models, strict=True):
            np.testing.assert_array_equal(left.Q, right.Q)
        self.assertEqual(low.feature_widths, {"raw": 2, "kahm_mem": 6, "kahm_local": 12})
        self.assertEqual(high.feature_widths, {"raw": 2, "kahm_mem": 6, "kahm_local": 18})

    def test_mem_is_exact_local_prefix_on_full_support(self):
        for bank in self.banks:
            for x in self.support:
                row = bank.encode_one(x)
                np.testing.assert_array_equal(row.kahm_mem, row.kahm_local[:6])
                self.assertTrue(np.isfinite(row.kahm_local).all())
                self.assertAlmostEqual(float(np.sum(row.kahm_mem[:2])), 1.0, places=14)
                self.assertAlmostEqual(float(np.sum(row.kahm_mem[2:])), 1.0, places=14)
                expected_blocks = [row.kahm_mem]
                for weight, model in zip(row.kahm_mem, bank.models, strict=True):
                    expected_blocks.append(weight * (model.pca.P.T @ (row.raw - model.pca.mu)))
                np.testing.assert_array_equal(row.kahm_local, np.concatenate(expected_blocks))

    def test_batch_equals_canonical_rows_and_preserves_duplicate_occurrences(self):
        bank = self.banks[1]
        x = self.training[[0, 3, 0, 7, 3, 8, 0]]
        for kind in ("raw", "kahm_mem", "kahm_local"):
            expected = np.stack([bank.transform_one(row, kind) for row in x])
            actual = bank.transform(x, kind)
            self.assertEqual(len(actual), len(x))
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(bank.transform(x[::-1], kind), expected[::-1])

    def test_encoding_does_not_refit_training_statistics(self):
        bank = self.banks[1]
        before = encoder_state_fingerprint(bank)
        bank.transform([[2.0, -1.0], [3.0, -2.0]], "kahm_local")
        self.assertEqual(before, encoder_state_fingerprint(bank))

    def test_rank_zero_bank_has_memberships_only_and_origin_score(self):
        bank = fit_encoder([[3.0, 4.0]] * 7, cap=2)
        row = bank.encode_one([3.0, 4.0])
        self.assertEqual(bank.feature_widths, {"raw": 2, "kahm_mem": 2, "kahm_local": 2})
        np.testing.assert_array_equal(row.kahm_mem, [1.0, 1.0])
        np.testing.assert_array_equal(row.kahm_local, row.kahm_mem)
        np.testing.assert_allclose(row.scores, [1 / (2 * np.sqrt(2))] * 2, rtol=1e-15)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            save_encoder(bank, path)
            reloaded = load_encoder(path)
            np.testing.assert_array_equal(reloaded.transform_one([3, 4], "kahm_local"), row.kahm_local)

    def test_affine_fallback_recomputes_complete_score(self):
        bank = self.banks[0]
        failing = replace(bank.models[0], C=np.zeros_like(bank.models[0].C))
        changed = replace(bank, models=(failing,) + bank.models[1:])
        row = changed.encode_one([0.25, 1.0])
        expected_reconstruction = failing.affine_one(row.raw)
        self.assertTrue(row.model_diagnostics[0]["used_fallback"])
        np.testing.assert_array_equal(row.reconstructions[0], expected_reconstruction)
        self.assertEqual(row.scores[0], radial_angular_score(row.raw, expected_reconstruction))

    def test_batch_input_validation_matches_single_row(self):
        bank = self.banks[0]
        for value in ([[True, False]], [["0", "1"]], np.asarray([[1j, 0j]])):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    bank.transform(value)
                with self.assertRaises((ValueError, TypeError)):
                    bank.transform_one(value[0])
        for value in ([[np.nan, 0]], [[np.inf, 1]], [[0, 1, 2]], [0, 1]):
            with self.assertRaises((ValueError, TypeError)):
                bank.transform(value)
        self.assertEqual(bank.transform(np.empty((0, 2))).shape, (0, 6))

    def test_saved_banks_replay_full_support_exactly_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            for bank in self.banks:
                path = Path(directory) / f"cap{bank.config.cap}.json"
                save_encoder(bank, path)
                with mock.patch("kahm_demo.geometry.fit_local_pca", side_effect=AssertionError("refit")):
                    loaded = load_encoder(path)
                self.assertEqual(encoder_state_fingerprint(bank), encoder_state_fingerprint(loaded))
                expected, diagnostics = support_feature_table(bank, "test")
                actual, replay_diagnostics = support_feature_table(loaded, "test")
                self.assertEqual(actual, expected)
                self.assertEqual(replay_diagnostics, diagnostics)
                with self.assertRaises(FileExistsError):
                    save_encoder(bank, path)

    def test_loader_rejects_changed_state_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            save_encoder(self.banks[0], path)
            record = read_json(path)
            record["state_sha256"] = "0" * 64
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_encoder(path)

    def test_loader_rejects_bad_inverse_even_with_recomputed_state_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            save_encoder(self.banks[1], path)
            record = read_json(path)
            model = record["state"]["fields"]["models"]["items"][0]
            values = model["fields"]["C"]["values"]
            values[0] = (float.fromhex(values[0]) + 0.1).hex()
            record["state_sha256"] = hashlib.sha256(canonical_json_bytes(record["state"])).hexdigest()
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "inverse"):
                load_encoder(path)

    def test_loaded_metric_must_match_stored_pca_direction(self):
        bank = self.banks[0]
        model = bank.models[0]
        changed_metric = model.metric_transform[::-1].copy()
        if np.array_equal(changed_metric, model.metric_transform):
            changed_metric[0] *= -1
        changed_model = replace(model, metric_transform=changed_metric)
        changed_bank = replace(bank, models=(changed_model,) + bank.models[1:])
        with self.assertRaisesRegex(ValueError, "metric"):
            validate_encoder(changed_bank)


class EncoderWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.output = self.root / "encoders"
        generate_bundle(self.data, master_seed=77411, run_id="test-step3-artifacts",
                        training_count=128, validation_count=32)

    def fit(self):
        return fit_development_encoders(self.data, self.output)

    def test_workflow_uses_only_training_inputs_and_leaves_calibration_reserved(self):
        from kahm_demo import encoder_workflow
        with mock.patch.object(encoder_workflow, "load_development", wraps=load_development) as loader:
            result = self.fit()
        self.assertEqual([call.args[1] for call in loader.call_args_list], ["training"])
        self.assertFalse((self.data / "calibration").exists())
        self.assertEqual(result["calibration_status"], "reserved_not_accessed")
        manifest = read_json(self.output / "manifest.json")
        self.assertIsNone(manifest["selected_bank_id"])
        self.assertFalse(manifest["forecast_models_fitted"])
        self.assertFalse(manifest["statistical_certificate_issued"])
        checked = verify_encoder_bundle(self.output, dataset_bundle=self.data)
        self.assertEqual(checked["status"], "passed")
        self.assertTrue(checked["candidate_geometry_identity_matches"])
        self.assertTrue(checked["numerical_source_identity_matches"])

    def test_validation_file_is_not_needed_for_encoder_fitting(self):
        (self.data / "development" / "validation.jsonl").unlink()
        self.fit()
        self.assertTrue((self.output / "cap_2.json").exists())

    def test_same_training_input_replays_identical_artifact_bytes(self):
        self.fit()
        other = self.root / "replay"
        fit_development_encoders(self.data, other)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         sorted(path.name for path in other.iterdir()))
        for path in self.output.iterdir():
            self.assertEqual(path.read_bytes(), (other / path.name).read_bytes())
        with self.assertRaises(FileExistsError):
            self.fit()

    def test_forecast_target_changes_do_not_change_encoder_states(self):
        self.fit()
        original_states = [encoder_state_fingerprint(load_encoder(self.output / f"cap_{cap}.json")) for cap in (1, 2)]
        training_path = self.data / "development" / "training.jsonl"
        rows = [json.loads(line) for line in training_path.read_text().splitlines()]
        for row in rows:
            row["next_state"] = (row["next_state"] + 1) % 4
        training_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = read_json(self.data / "manifest.json")
        manifest["development"]["training"]["sha256"] = sha256_file(training_path)
        write_json(self.data / "manifest.json", manifest)
        other = self.root / "changed_targets"
        fit_development_encoders(self.data, other)
        changed_states = [encoder_state_fingerprint(load_encoder(other / f"cap_{cap}.json")) for cap in (1, 2)]
        self.assertEqual(changed_states, original_states)

    def test_feature_replay_rejects_changed_table_even_if_file_hash_is_updated(self):
        self.fit()
        manifest = read_json(self.output / "manifest.json")
        record = manifest["banks"][0]
        path = self.output / record["feature_table_path"]
        table = read_json(path)
        table["entries"][0]["kahm_mem_hex"][0] = (0.5).hex()
        write_json(path, table)
        record["feature_table_sha256"] = sha256_file(path)
        write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "replay"):
            verify_encoder_bundle(self.output)

    def test_bundle_provenance_must_match_each_bank(self):
        self.fit()
        manifest = read_json(self.output / "manifest.json")
        manifest["training_provenance"]["run_id"] = "test-some-other-run"
        write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_encoder_bundle(self.output)

    def test_manifest_rank_summary_must_match_the_saved_model(self):
        self.fit()
        manifest = read_json(self.output / "manifest.json")
        manifest["banks"][0]["effective_ranks"] = [999] * 6
        write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "effective_ranks"):
            verify_encoder_bundle(self.output)


if __name__ == "__main__":
    unittest.main()

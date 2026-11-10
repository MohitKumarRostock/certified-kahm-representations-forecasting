"""Protocol tests use dedicated test streams, never the primary calibration."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from kahm_demo import datasets
from kahm_demo._util import read_json, sha256_file, write_json
from kahm_demo.contract import CANDIDATE_IDS, CONTRACT_SHA256, validate_contract
from kahm_demo.datasets import (FREEZE_SCHEMA, generate_bundle, load_calibration,
                               load_development, materialize_calibration,
                               read_episode_file, validate_freeze_record, verify_bundle)
from kahm_demo.law import HISTORY_KEYS
from kahm_demo.oracle import prediction_table_from_mapping


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "data"

    def tearDown(self):
        self.tmp.cleanup()

    def make_bundle(self, path=None, **kwargs):
        return generate_bundle(path or self.bundle, master_seed=424242,
                               run_id="test-dataset-protocol", training_count=96,
                               validation_count=48, **kwargs)

    def make_freeze(self):
        base = self.root / "test-freeze"
        base.mkdir()
        source = base / "inference.py"
        source.write_text("def predict_one(history):\n    return 0.5\n")
        models = []
        for candidate in CANDIDATE_IDS:
            artifact = base / f"{candidate}.json"
            write_json(artifact, {"test_fixture": True, "candidate": candidate, "constant": 0.5})
            table_path = base / f"{candidate}_predictions.json"
            write_json(table_path, prediction_table_from_mapping({h: 0.5 for h in HISTORY_KEYS}, candidate))
            models.append({"candidate_id": candidate,
                           "artifact_path": artifact.name, "artifact_sha256": sha256_file(artifact),
                           "prediction_table_path": table_path.name,
                           "prediction_table_sha256": sha256_file(table_path)})
        manifest = read_json(self.bundle / "manifest.json")
        freeze = {"schema": FREEZE_SCHEMA, "status": "frozen",
                  "contract_sha256": CONTRACT_SHA256, "run_id": manifest["run_id"],
                  "dataset_manifest_sha256": sha256_file(self.bundle / "manifest.json"),
                  "calibration_plan_sha256": manifest["calibration_plan"]["sha256"],
                  "inference": {"runtime": {"purpose": "test-only constant fixtures"},
                                "canonical_method": "one float per canonical history",
                                "source_files": [{"path": source.name, "sha256": sha256_file(source)}]},
                  "models": models}
        path = base / "freeze.json"
        write_json(path, freeze)
        return path

    def test_contract_identity_and_counts(self):
        contract = validate_contract()
        self.assertEqual(contract["certificate"]["independent_unit_count"], 4096)
        self.assertEqual(len(contract["candidates"]), 3)

    def test_generation_reserves_calibration_without_sampling_it(self):
        from kahm_demo import datasets
        actual = datasets.sample_episodes
        roles = []
        def observe(count, seed, role, run_id):
            roles.append(role)
            return actual(count, seed, role, run_id)
        with patch.object(datasets, "sample_episodes", side_effect=observe):
            result = self.make_bundle()
        self.assertEqual(roles, ["training", "validation"])
        self.assertEqual(result["calibration_state"], "reserved_not_materialized")
        self.assertFalse((self.bundle / "calibration").exists())
        self.assertEqual(read_json(self.bundle / "calibration_plan.json")["episode_count"], 4096)

    def test_bundle_bytes_are_reproducible(self):
        self.make_bundle()
        other = self.root / "other"
        self.make_bundle(other)
        files = [p.relative_to(self.bundle) for p in self.bundle.rglob("*") if p.is_file()]
        for relative in files:
            self.assertEqual((self.bundle / relative).read_bytes(), (other / relative).read_bytes())

    def test_counts_duplicate_retention_and_feature_interface(self):
        self.make_bundle()
        train = load_development(self.bundle, "training")
        val = load_development(self.bundle, "validation")
        self.assertEqual((len(train.episodes), len(val.episodes)), (96, 48))
        self.assertEqual(train.episode_key(0), ("test-dataset-protocol", "training", 0))
        self.assertNotEqual(train.episode_key(0), val.episode_key(0))
        triples = {(e.previous_state, e.current_state, e.next_state) for e in train.episodes}
        self.assertLess(len(triples), len(train.episodes))  # 96 rows, at most 64 triples.
        X, y = train.xy()
        self.assertEqual(len(X), 96)
        self.assertTrue(all(len(x) == 2 for x in X))
        self.assertTrue(all(v in (0.0, 0.25, 0.75, 1.0) for x in X for v in x))
        self.assertEqual(y, [e.y for e in train.episodes])
        with self.assertRaises(ValueError):
            load_development(self.bundle, "calibration")

    def test_existing_data_are_not_overwritten(self):
        self.make_bundle()
        before = (self.bundle / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.make_bundle()
        self.assertEqual(before, (self.bundle / "manifest.json").read_bytes())

    def test_corrupt_rows_are_rejected_by_hash(self):
        self.make_bundle()
        path = self.bundle / "development/training.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"next_state":', b'"future_state":', 1))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_bundle(self.bundle)

    def test_row_ids_and_fields_are_strict(self):
        path = self.root / "bad.jsonl"
        cases = [
            {"episode_id": 1, "previous_state": 1, "current_state": 3, "next_state": 0},
            {"episode_id": 0, "previous_state": True, "current_state": 3, "next_state": 0},
            {"episode_id": 0, "previous_state": 1, "current_state": 3, "next_state": 0, "extra": 9},
        ]
        for row in cases:
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises((TypeError, ValueError)):
                read_episode_file(path, 1)
        path.write_text('{"episode_id":0,"episode_id":0,"previous_state":1,"current_state":3,"next_state":0}\n')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_episode_file(path, 1)

    def test_historical_runtime_metadata_does_not_block_reading(self):
        self.make_bundle()
        manifest_path = self.bundle / "manifest.json"
        manifest = read_json(manifest_path)
        for record in manifest["development"].values():
            record["stream"]["rng"]["python_full_version"] = "archived supported build"
        plan_path = self.bundle / "calibration_plan.json"
        plan = read_json(plan_path)
        plan["stream"]["rng"]["python_full_version"] = "archived supported build"
        write_json(plan_path, plan)
        manifest["calibration_plan"]["sha256"] = sha256_file(plan_path)
        write_json(manifest_path, manifest)
        self.assertEqual(verify_bundle(self.bundle)["status"], "passed")

    @patch("kahm_demo.datasets._model_freeze_preflight", return_value={"status": "passed"})
    def test_missing_or_incomplete_freeze_cannot_materialize_calibration(self, preflight):
        self.make_bundle()
        missing = self.root / "absent-freeze" / "freeze.json"
        with self.assertRaises(OSError):
            materialize_calibration(self.bundle, missing)
        incomplete = self.root / "incomplete-freeze" / "freeze.json"
        incomplete.parent.mkdir()
        write_json(incomplete, {"schema": FREEZE_SCHEMA, "status": "frozen"})
        with self.assertRaises(ValueError):
            materialize_calibration(self.bundle, incomplete)
        self.assertFalse((self.bundle / "calibration").exists())
        self.assertEqual(preflight.call_count, 2)

    @patch("kahm_demo.datasets._model_freeze_preflight", return_value={"status": "passed"})
    def test_test_stream_calibration_binds_to_freeze_and_rejects_replacement(self, preflight):
        # Isolate data binding with structural constant fixtures. Full numerical
        # freeze validation belongs to the model-specific integration tests.
        self.make_bundle()
        freeze = self.make_freeze()
        receipt = materialize_calibration(self.bundle, freeze)
        self.assertEqual(receipt["episode_count"], 4096)
        self.assertEqual(receipt["freeze_record_sha256"], sha256_file(freeze))
        data = load_calibration(self.bundle, freeze)
        self.assertEqual(len(data.episodes), 4096)
        self.assertEqual(data.role, "calibration")
        self.assertEqual(data.run_id, "test-dataset-protocol")
        with self.assertRaises(FileExistsError):
            materialize_calibration(self.bundle, freeze)
        self.assertEqual(preflight.call_count, 2)

    def test_modified_inference_source_or_table_invalidates_freeze(self):
        self.make_bundle()
        freeze = self.make_freeze()
        source = freeze.parent / "inference.py"
        source.write_text("def predict_one(history):\n    return 0.25\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_freeze_record(self.bundle, freeze)
        self.assertFalse((self.bundle / "calibration").exists())

    @patch("kahm_demo.datasets._model_freeze_preflight", return_value={"status": "passed"})
    def test_calibration_loader_returns_the_same_path_it_verified(self, preflight):
        self.make_bundle()
        freeze = self.make_freeze()
        materialize_calibration(self.bundle, freeze)
        directory = self.bundle / "calibration"
        original = directory / "episodes.jsonl"
        renamed = directory / "verified_episodes.jsonl"
        original.rename(renamed)
        # An unchecked file at the old conventional path must not be returned.
        original.write_text("invalid unchecked content\n")
        receipt_path = directory / "receipt.json"
        receipt = read_json(receipt_path)
        receipt["data_path"] = renamed.name
        write_json(receipt_path, receipt)
        self.assertEqual(len(load_calibration(self.bundle, freeze).episodes), 4096)
        self.assertEqual(preflight.call_count, 2)

    @patch("kahm_demo.datasets._model_freeze_preflight", return_value={"status": "passed"})
    def test_materialization_respects_manifest_plan_path(self, preflight):
        self.make_bundle()
        source = self.bundle / "calibration_plan.json"
        target = self.bundle / "reserved_plan.json"
        source.rename(target)
        manifest_path = self.bundle / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["calibration_plan"]["path"] = target.name
        write_json(manifest_path, manifest)
        freeze = self.make_freeze()
        receipt = materialize_calibration(self.bundle, freeze)
        self.assertEqual(receipt["calibration_plan_sha256"], sha256_file(target))
        self.assertEqual(verify_bundle(self.bundle)["status"], "passed")
        preflight.assert_called_once_with(freeze, dataset_bundle=self.bundle)

    @patch("kahm_demo.datasets._model_freeze_preflight", return_value={"status": "passed"})
    def test_receipt_requires_freeze_binding_fields(self, preflight):
        self.make_bundle()
        freeze = self.make_freeze()
        materialize_calibration(self.bundle, freeze)
        path = self.bundle / "calibration/receipt.json"
        original = read_json(path)
        for key in ("status", "freeze_record_sha256", "candidate_ids"):
            changed = dict(original)
            del changed[key]
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "receipt is inconsistent"):
                verify_bundle(self.bundle)
        write_json(path, original)
        self.assertEqual(verify_bundle(self.bundle)["status"], "passed")
        preflight.assert_called_once_with(freeze, dataset_bundle=self.bundle)

    def test_model_preflight_delegates_to_the_full_freeze_verifier(self):
        # A module-local stub tests the lazy import/delegation without supplying
        # a production bypass or requiring model fitting in this data-layer test.
        module = types.ModuleType("kahm_demo.freeze")
        report = {"status": "passed", "model_specific_replay": True}
        verifier = Mock(return_value=report)
        module.verify_frozen_bundle = verifier
        freeze = self.root / "test-freeze.json"
        with patch.dict(sys.modules, {"kahm_demo.freeze": module}):
            actual = datasets._model_freeze_preflight(freeze, dataset_bundle=self.bundle)
        self.assertIs(actual, report)
        verifier.assert_called_once_with(freeze, dataset_bundle=self.bundle)

    def test_materialization_runs_model_preflight_before_sampling_or_verification(self):
        self.make_bundle()
        freeze = self.make_freeze()
        events = []
        actual_sample, actual_verify = datasets.sample_episodes, datasets.verify_bundle
        actual_structural = datasets.validate_freeze_record

        def preflight(freeze_path, *, dataset_bundle):
            self.assertEqual((freeze_path, dataset_bundle), (freeze, self.bundle))
            self.assertFalse((self.bundle / "calibration").exists())
            events.append("preflight")
            return {"status": "passed"}

        def verify(bundle):
            self.assertEqual(events, ["preflight", "structural"])
            events.append("verify")
            return actual_verify(bundle)

        def structural(bundle, freeze_path):
            self.assertEqual(events, ["preflight"])
            events.append("structural")
            return actual_structural(bundle, freeze_path)

        def sample(count, master_seed, role, run_id):
            self.assertEqual(events, ["preflight", "structural", "verify"])
            self.assertEqual((master_seed, role, run_id),
                             (424242, "calibration", "test-dataset-protocol"))
            events.append("sample")
            return actual_sample(count, master_seed, role, run_id)

        with (patch.object(datasets, "_model_freeze_preflight", side_effect=preflight),
              patch.object(datasets, "validate_freeze_record", side_effect=structural),
              patch.object(datasets, "verify_bundle", side_effect=verify),
              patch.object(datasets, "sample_episodes", side_effect=sample)):
            receipt = materialize_calibration(self.bundle, freeze)
        self.assertEqual(events, ["preflight", "structural", "verify", "sample"])
        self.assertEqual(receipt["episode_count"], 4096)

    def test_failed_model_preflight_leaves_no_materialization_output(self):
        self.make_bundle()
        freeze = self.make_freeze()
        before = {path.relative_to(self.bundle) for path in self.bundle.rglob("*")}
        with (patch.object(datasets, "_model_freeze_preflight",
                           side_effect=ValueError("model replay failed")) as preflight,
              patch.object(datasets, "validate_freeze_record") as structural,
              patch.object(datasets, "verify_bundle") as verify,
              patch.object(datasets, "sample_episodes") as sample,
              patch.object(datasets, "read_episode_file") as read,
              patch.object(datasets.tempfile, "mkdtemp") as stage):
            with self.assertRaisesRegex(ValueError, "model replay failed"):
                materialize_calibration(self.bundle, freeze)
        preflight.assert_called_once_with(freeze, dataset_bundle=self.bundle)
        structural.assert_not_called()
        verify.assert_not_called()
        sample.assert_not_called()
        read.assert_not_called()
        stage.assert_not_called()
        self.assertEqual(before, {path.relative_to(self.bundle) for path in self.bundle.rglob("*")})
        self.assertFalse((self.bundle / "calibration").exists())

    def test_existing_calibration_is_rejected_before_any_preflight_or_outcome_read(self):
        self.make_bundle()
        existing = self.bundle / "calibration"
        existing.mkdir()
        sentinel = existing / "episodes.jsonl"
        sentinel.write_text("existing content must not be read or replaced\n")
        with (patch.object(datasets, "validate_freeze_record") as structural,
              patch.object(datasets, "_model_freeze_preflight") as preflight,
              patch.object(datasets, "verify_bundle") as verify,
              patch.object(datasets, "read_episode_file") as read,
              patch.object(datasets, "sample_episodes") as sample):
            with self.assertRaises(FileExistsError):
                materialize_calibration(self.bundle, self.root / "absent-freeze.json")
        for function in (structural, preflight, verify, read, sample):
            function.assert_not_called()
        self.assertEqual(sentinel.read_text(), "existing content must not be read or replaced\n")

    def test_loader_model_preflight_failure_blocks_all_outcome_access(self):
        self.make_bundle()
        freeze = self.make_freeze()
        with patch.object(datasets, "_model_freeze_preflight", return_value={"status": "passed"}):
            materialize_calibration(self.bundle, freeze)
        with (patch.object(datasets, "_model_freeze_preflight",
                           side_effect=ValueError("inference source changed")) as preflight,
              patch.object(datasets, "validate_freeze_record") as structural,
              patch.object(datasets, "verify_bundle") as verify,
              patch.object(datasets, "read_episode_file") as read):
            with self.assertRaisesRegex(ValueError, "inference source changed"):
                load_calibration(self.bundle, freeze)
        preflight.assert_called_once_with(freeze, dataset_bundle=self.bundle)
        structural.assert_not_called()
        verify.assert_not_called()
        read.assert_not_called()

    def test_materialization_rejects_snapshot_internal_output_before_preflight(self):
        frozen_root = self.root / "frozen"
        archived_bundle = frozen_root / "development/data"
        archived_bundle.mkdir(parents=True)
        freeze = frozen_root / "freeze.json"
        alias = self.root / "outside-alias"
        alias.symlink_to(archived_bundle, target_is_directory=True)
        for bundle in (archived_bundle, alias):
            with self.subTest(bundle=bundle):
                with (patch.object(datasets, "validate_freeze_record") as structural,
                      patch.object(datasets, "_model_freeze_preflight") as preflight,
                      patch.object(datasets, "verify_bundle") as verify,
                      patch.object(datasets, "read_episode_file") as read,
                      patch.object(datasets, "sample_episodes") as sample,
                      patch.object(datasets.tempfile, "mkdtemp") as stage):
                    with self.assertRaisesRegex(ValueError, "outside the frozen snapshot"):
                        materialize_calibration(bundle, freeze)
                for function in (structural, preflight, verify, read, sample, stage):
                    function.assert_not_called()
                self.assertFalse((archived_bundle / "calibration").exists())

    def test_loader_rejects_snapshot_internal_outcomes_before_preflight(self):
        frozen_root = self.root / "frozen"
        archived_bundle = frozen_root / "development/data"
        directory = archived_bundle / "calibration"
        directory.mkdir(parents=True)
        sentinel = directory / "episodes.jsonl"
        sentinel.write_text("outcomes must not be read from the snapshot\n")
        with (patch.object(datasets, "validate_freeze_record") as structural,
              patch.object(datasets, "_model_freeze_preflight") as preflight,
              patch.object(datasets, "verify_bundle") as verify,
              patch.object(datasets, "read_episode_file") as read,
              patch.object(datasets, "read_json") as read_record):
            with self.assertRaisesRegex(ValueError, "outside the frozen snapshot"):
                load_calibration(archived_bundle, frozen_root / "freeze.json")
        for function in (structural, preflight, verify, read, read_record):
            function.assert_not_called()
        self.assertEqual(sentinel.read_text(), "outcomes must not be read from the snapshot\n")

    def test_dangling_calibration_symlink_is_not_replaced(self):
        self.make_bundle()
        output = self.bundle / "calibration"
        output.symlink_to(self.root / "missing-calibration-target", target_is_directory=True)
        self.assertFalse(output.exists())
        with (patch.object(datasets, "_model_freeze_preflight") as preflight,
              patch.object(datasets, "sample_episodes") as sample):
            with self.assertRaises(FileExistsError):
                materialize_calibration(self.bundle, self.root / "absent-freeze/freeze.json")
        preflight.assert_not_called()
        sample.assert_not_called()
        self.assertTrue(output.is_symlink())

    def test_loader_runs_model_preflight_before_reading_calibration_episodes(self):
        self.make_bundle()
        freeze = self.make_freeze()
        with patch.object(datasets, "_model_freeze_preflight", return_value={"status": "passed"}):
            materialize_calibration(self.bundle, freeze)
        events = []
        actual_read = datasets.read_episode_file

        def preflight(freeze_path, *, dataset_bundle):
            self.assertEqual((freeze_path, dataset_bundle), (freeze, self.bundle))
            self.assertEqual(events, [])
            events.append("preflight")
            return {"status": "passed"}

        def read(path, count):
            self.assertTrue(events and events[0] == "preflight")
            if path.parent == self.bundle / "calibration":
                events.append("calibration_read")
            return actual_read(path, count)

        with (patch.object(datasets, "_model_freeze_preflight", side_effect=preflight),
              patch.object(datasets, "read_episode_file", side_effect=read)):
            loaded = load_calibration(self.bundle, freeze)
        self.assertEqual(events[0], "preflight")
        self.assertIn("calibration_read", events)
        self.assertEqual(len(loaded.episodes), 4096)

    def test_different_freeze_receipt_binding_blocks_outcome_access(self):
        self.make_bundle()
        freeze = self.make_freeze()
        with patch.object(datasets, "_model_freeze_preflight", return_value={"status": "passed"}):
            materialize_calibration(self.bundle, freeze)
        changed = read_json(freeze)
        changed["inference"]["runtime"]["purpose"] = "different structural fixture"
        write_json(freeze, changed)
        # Simulate another model-valid freeze; receipt identity must still bind
        # access to the exact freeze under which these test outcomes were drawn.
        with (patch.object(datasets, "_model_freeze_preflight",
                           return_value={"status": "passed"}) as preflight,
              patch.object(datasets, "verify_bundle") as verify,
              patch.object(datasets, "read_episode_file") as read):
            with self.assertRaisesRegex(ValueError, "different freeze record"):
                load_calibration(self.bundle, freeze)
        preflight.assert_called_once_with(freeze, dataset_bundle=self.bundle)
        verify.assert_not_called()
        read.assert_not_called()

    def test_cli_generation_and_required_freeze(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "kahm_demo", "generate", "--output", str(self.bundle),
                                 "--master-seed", "101", "--run-id", "test-cli", "--training-count", "7",
                                 "--validation-count", "5"], cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["development_counts"], {"training": 7, "validation": 5})
        blocked = subprocess.run([sys.executable, "-m", "kahm_demo", "materialize-calibration", "--bundle", str(self.bundle)],
                                 cwd=root, text=True, capture_output=True)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("--freeze", blocked.stderr)
        self.assertFalse((self.bundle / "calibration").exists())


if __name__ == "__main__":
    unittest.main()

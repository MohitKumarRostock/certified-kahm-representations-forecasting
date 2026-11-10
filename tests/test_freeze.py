"""Portable frozen-family integration and pre-outcome integrity regressions."""

from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kahm_demo import datasets
from kahm_demo import freeze as freeze_module
from kahm_demo._util import canonical_json_bytes, read_json, sha256_file, write_json
from kahm_demo.contract import CANDIDATE_IDS
from kahm_demo.encoder_workflow import fit_development_encoders
from kahm_demo.forecast_state import forecaster_state_fingerprint, load_forecaster
from kahm_demo.forecast_workflow import fit_development_forecasters
from kahm_demo.freeze import freeze_forecasters, load_frozen_forecaster, verify_frozen_bundle
from kahm_demo.law import HISTORY_KEYS, OBSERVATION_FLOATS
from kahm_demo.oracle import parse_prediction_table


@contextmanager
def prohibit_evaluation_and_refitting(*, refitting=True, calibration_apis=True):
    """Guard actual entry points, while leaving all verifiers operational."""
    targets = [
        "kahm_demo.oracle.exact_risk", "kahm_demo.oracle.risk_of_predictor",
        "kahm_demo.datasets.sample_episodes", "kahm_demo.sampling.sample_episodes",
    ]
    if calibration_apis:
        targets += ["kahm_demo.datasets.load_calibration",
                    "kahm_demo.datasets.materialize_calibration"]
    if refitting:
        targets += [
            "kahm_demo.encoder_workflow.fit_development_encoders",
            "kahm_demo.encoder_workflow.fit_encoder_candidates",
            "kahm_demo.encoder.fit_encoder_candidates",
            "kahm_demo.forecast_workflow.fit_development_forecasters",
            "kahm_demo.forecast_workflow.fit_readout_features",
            "kahm_demo.forecast_workflow.fit_ridge_grid",
            "kahm_demo.readout_features.fit_readout_features",
            "kahm_demo.ridge.fit_ridge_grid",
        ]
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(mock.patch(target, side_effect=AssertionError(
                f"frozen verification called prohibited entry point: {target}")))
        yield


@contextmanager
def prohibit_open_under(directory):
    """Detect byte reads, including hashing, below an outcome directory."""
    directory = Path(directory).resolve()
    original = Path.open

    def guarded(path, *args, **kwargs):
        if Path(path).resolve().is_relative_to(directory):
            raise AssertionError("calibration outcome bytes were opened before authorization")
        return original(path, *args, **kwargs)

    with mock.patch.object(Path, "open", guarded):
        yield


def write_rehashed_record(root, record):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json(root / "freeze.json", record)


def refresh_file_identity(root, record, relative, *, archived_source=False):
    path = root / relative
    row = next(item for item in record["files"] if item["path"] == relative)
    row.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    if archived_source:
        source = next(item for item in record["inference"]["source_files"]
                      if item["path"] == relative)
        source["sha256"] = row["sha256"]


class FreezeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kahm-test-step5-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.data, cls.encoders, cls.models, cls.frozen = [
            cls.root / name for name in ("data", "encoders", "models", "frozen")
        ]
        # The fixture is wholly separate from the primary run and creates only
        # training/validation episodes. No test samples a calibration stream.
        real_sample = datasets.sample_episodes

        def development_only(count, seed, role, run_id):
            if (role not in ("training", "validation")
                    or seed != 58190425 or run_id != "test-step5-freeze-integration"):
                raise AssertionError("fixture attempted an unexpected random stream")
            return real_sample(count, seed, role, run_id)

        with mock.patch.object(datasets, "sample_episodes", side_effect=development_only):
            datasets.generate_bundle(cls.data, master_seed=58190425,
                                     run_id="test-step5-freeze-integration",
                                     training_count=128, validation_count=64)
        with prohibit_evaluation_and_refitting(refitting=False):
            fit_development_encoders(cls.data, cls.encoders)
            fit_development_forecasters(cls.data, cls.encoders, cls.models)
        with prohibit_evaluation_and_refitting():
            cls.creation = freeze_forecasters(cls.data, cls.encoders, cls.models, cls.frozen)
        cls.record = read_json(cls.frozen / "freeze.json")

    def temporary_directory(self):
        temporary = tempfile.TemporaryDirectory(prefix="kahm-test-freeze-case-")
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def copy_frozen(self):
        target = self.temporary_directory() / "frozen"
        shutil.copytree(self.frozen, target)
        return target

    def copy_data(self):
        target = self.temporary_directory() / "data"
        shutil.copytree(self.data, target)
        return target

    def verify(self, root, *, dataset_bundle=None):
        with prohibit_evaluation_and_refitting():
            return verify_frozen_bundle(root / "freeze.json", dataset_bundle=dataset_bundle)

    def test_freeze_preserves_selected_bytes_and_full_support(self):
        self.assertEqual(self.creation["status"], "frozen")
        self.assertFalse((self.data / "calibration").exists())
        self.assertFalse(self.record["refit_during_freeze"])
        self.assertFalse(self.record["model_population_risk_evaluated"])
        self.assertFalse(self.record["statistical_certificate_issued"])
        self.assertEqual(self.record["development"]["search_count"], 225)
        archived = self.frozen / "development/forecasters/search_states"
        self.assertEqual(len(list(archived.glob("*.json"))), 225)
        source_manifest = read_json(self.models / "manifest.json")
        for frozen, selected in zip(self.record["models"], source_manifest["selected_models"], strict=True):
            with self.subTest(candidate=frozen["candidate_id"]):
                artifact = self.frozen / frozen["artifact_path"]
                predictions = self.frozen / frozen["prediction_table_path"]
                self.assertEqual(artifact.read_bytes(), (self.models / selected["model_path"]).read_bytes())
                self.assertEqual(predictions.read_bytes(),
                                 (self.models / selected["prediction_table_path"]).read_bytes())
                self.assertEqual(frozen["status"], "frozen")
                self.assertEqual(read_json(artifact)["status"], "development_selected_not_frozen")
                with prohibit_evaluation_and_refitting():
                    model = load_forecaster(artifact)
                    expected = parse_prediction_table(read_json(predictions))
                    for a, b in HISTORY_KEYS:
                        actual = model.predict_one([OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b]])
                        self.assertEqual(actual.hex(), expected[(a, b)].hex())
                    self.assertEqual(forecaster_state_fingerprint(model), frozen["state_sha256"])

    def test_self_contained_copy_verifies_without_external_data(self):
        root = self.copy_frozen()
        result = self.verify(root)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["self_contained_development_snapshot_checked"])
        self.assertTrue(result["all_225_development_candidate_states_replayed"])
        self.assertTrue(result["exact_development_losses_and_selection_checked"])
        self.assertFalse(result["external_dataset_identity_checked"])
        self.assertFalse(result["calibration_outcomes_accessed"])
        self.assertEqual(result["candidate_ids"], list(CANDIDATE_IDS))

    def test_safe_loader_returns_the_actual_preflighted_instance(self):
        root = self.copy_frozen()
        captured = {}
        real_loader = freeze_module.load_forecaster

        def capture(path):
            model = real_loader(path)
            captured[model.candidate_id] = model
            return model

        with prohibit_evaluation_and_refitting(), mock.patch.object(
                freeze_module, "load_forecaster", side_effect=capture):
            wrapper = load_frozen_forecaster(root / "freeze.json", "kahm_local")
        self.assertEqual(set(captured), set(CANDIDATE_IDS))
        self.assertIs(wrapper.model, captured["kahm_local"])
        self.assertEqual(wrapper.freeze_record_sha256, sha256_file(root / "freeze.json"))
        histories = [[0.25, 0.75], [1.0, 0.0], [0.25, 0.75]]
        with prohibit_evaluation_and_refitting(), mock.patch.object(
                freeze_module, "load_forecaster", side_effect=AssertionError("query reloaded a model")):
            scalar = [wrapper.predict_one(history).hex() for history in histories]
            batch = [float(value).hex() for value in wrapper.predict(histories)]
        self.assertEqual(batch, scalar)

    def test_changed_record_without_updated_payload_hash_is_rejected(self):
        root = self.copy_frozen()
        record = read_json(root / "freeze.json")
        record["run_id"] = "test-forged-run"
        write_json(root / "freeze.json", record)
        with self.assertRaisesRegex(ValueError, "payload identity"):
            self.verify(root)

    def test_runtime_method_and_dependency_roster_fail_closed(self):
        cases = (
            ("runtime", lambda record: record["inference"]["runtime"].update(numpy_version="0.invalid"), "runtime"),
            ("method", lambda record: record["inference"].update(canonical_method="unknown"), "method"),
            ("roster", lambda record: record["inference"]["execution_source_roster"].pop(), "roster"),
        )
        for name, mutate, message in cases:
            with self.subTest(case=name):
                root = self.copy_frozen()
                record = read_json(root / "freeze.json")
                mutate(record)
                write_rehashed_record(root, record)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify(root)

    def test_rehashed_archived_source_still_must_match_executed_code(self):
        root = self.copy_frozen()
        relative = "sources/kahm_demo/forecast.py"
        path = root / relative
        path.write_text(path.read_text(encoding="utf-8") + "\n# altered archive\n", encoding="utf-8")
        record = read_json(root / "freeze.json")
        refresh_file_identity(root, record, relative, archived_source=True)
        write_rehashed_record(root, record)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: forecast.py"):
            self.verify(root)

    def test_rehashed_protocol_change_is_rejected(self):
        root = self.copy_frozen()
        path = root / "calibration_protocol.json"
        protocol = read_json(path)
        protocol["test_only_altered_protocol"] = True
        write_json(path, protocol)
        record = read_json(root / "freeze.json")
        refresh_file_identity(root, record, "calibration_protocol.json")
        record["calibration_protocol"]["sha256"] = sha256_file(path)
        write_rehashed_record(root, record)
        with self.assertRaisesRegex(ValueError, "calibration protocol"):
            self.verify(root)

    def test_recorded_version_is_bound_to_the_archived_initializer(self):
        root = self.copy_frozen()
        record = read_json(root / "freeze.json")
        record["package_version"] = "test.incorrect.historical.version"
        write_rehashed_record(root, record)
        with self.assertRaisesRegex(ValueError, "archived package version"):
            self.verify(root)

    def test_missing_or_undeclared_snapshot_file_is_rejected(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                root = self.copy_frozen()
                if missing:
                    (root / "models/raw.json").unlink()
                else:
                    (root / "undeclared.txt").write_text("unexpected", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    self.verify(root)

    def test_wrong_external_development_identity_is_rejected(self):
        wrong = self.temporary_directory() / "different_data"
        datasets.generate_bundle(wrong, master_seed=58190426, run_id="test-step5-different-data",
                                 training_count=8, validation_count=4)
        with self.assertRaisesRegex(ValueError, "external data bundle differs"):
            self.verify(self.frozen, dataset_bundle=wrong)

    def test_existing_freeze_and_premature_calibration_prevent_creation(self):
        before = sha256_file(self.frozen / "freeze.json")
        with prohibit_evaluation_and_refitting(), self.assertRaises(FileExistsError):
            freeze_forecasters(self.data, self.encoders, self.models, self.frozen)
        self.assertEqual(sha256_file(self.frozen / "freeze.json"), before)
        data = self.copy_data()
        calibration = data / "calibration"
        calibration.mkdir()
        (calibration / "episodes.jsonl").write_bytes(b"outcome bytes must not be read")
        output = self.temporary_directory() / "new_freeze"
        with prohibit_open_under(calibration), prohibit_evaluation_and_refitting():
            with self.assertRaisesRegex(ValueError, "after calibration"):
                freeze_forecasters(data, self.encoders, self.models, output)
        self.assertFalse(output.exists())

    def test_snapshot_calibration_bytes_are_rejected_before_opening(self):
        for relative in ("development/data/calibration", "unlisted/nested/calibration"):
            with self.subTest(path=relative):
                root = self.copy_frozen()
                calibration = root / relative
                calibration.mkdir(parents=True)
                (calibration / "episodes.jsonl").write_bytes(b"never read this outcome sentinel")
                with prohibit_open_under(calibration), self.assertRaisesRegex(ValueError, "calibration outcomes"):
                    self.verify(root)

    def test_existing_external_calibration_is_ignored_by_freeze_verification(self):
        data = self.copy_data()
        calibration = data / "calibration"
        calibration.mkdir()
        (calibration / "receipt.json").write_bytes(b"not a readable calibration receipt")
        (calibration / "episodes.jsonl").write_bytes(b"not readable calibration episodes")
        with prohibit_open_under(calibration):
            result = self.verify(self.frozen, dataset_bundle=data)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["calibration_outcomes_accessed"])

    def test_declared_and_symlinked_outcome_paths_are_rejected_before_reading(self):
        for symlinked in (False, True):
            with self.subTest(symlinked=symlinked):
                data = self.copy_data()
                calibration = data / "calibration"
                calibration.mkdir()
                (calibration / "episodes.jsonl").write_bytes(b"do not inspect these outcomes")
                relative = "calibration/episodes.jsonl"
                if symlinked:
                    relative = "development/calibration_alias.jsonl"
                    (data / relative).symlink_to("../calibration/episodes.jsonl")
                manifest = read_json(data / "manifest.json")
                manifest["development"]["training"]["path"] = relative
                write_json(data / "manifest.json", manifest)
                with prohibit_open_under(calibration), self.assertRaisesRegex(ValueError, "calibration outcomes"):
                    self.verify(self.frozen, dataset_bundle=data)

    def test_calibration_public_apis_cannot_mutate_or_load_inside_snapshot(self):
        root = self.copy_frozen()
        data = root / "development/data"
        before = sha256_file(root / "freeze.json")
        for operation in (datasets.materialize_calibration, datasets.load_calibration):
            with self.subTest(operation=operation.__name__):
                with prohibit_evaluation_and_refitting(calibration_apis=False):
                    with self.assertRaisesRegex(ValueError, "outside the frozen snapshot"):
                        operation(data, root / "freeze.json")
        self.assertFalse((data / "calibration").exists())
        self.assertEqual(sha256_file(root / "freeze.json"), before)

    def test_real_calibration_gates_reject_a_structurally_valid_false_model_table(self):
        root = self.copy_frozen()
        record = read_json(root / "freeze.json")
        raw = next(row for row in record["models"] if row["candidate_id"] == "raw")
        relative = raw["prediction_table_path"]
        path = root / relative
        table = read_json(path)
        first = table["predictions"][0]
        replacement = 0.125 if first["prediction_hex"] != (0.125).hex() else 0.875
        first["prediction_hex"] = replacement.hex()
        write_json(path, table)
        raw["prediction_table_sha256"] = sha256_file(path)
        refresh_file_identity(root, record, relative)
        write_rehashed_record(root, record)
        data = self.copy_data()
        # Hashes, support, boundedness and family names alone still pass the
        # independent structural checker. Actual model execution must stop it.
        datasets.validate_freeze_record(data, root / "freeze.json")
        with prohibit_evaluation_and_refitting(calibration_apis=False):
            with self.assertRaisesRegex(ValueError, "execution differs.*prediction table"):
                datasets.materialize_calibration(data, root / "freeze.json")
        self.assertFalse((data / "calibration").exists())
        calibration = data / "calibration"
        calibration.mkdir()
        (calibration / "receipt.json").write_bytes(b"must not read this receipt")
        (calibration / "episodes.jsonl").write_bytes(b"must not read these outcomes")
        with prohibit_open_under(calibration), prohibit_evaluation_and_refitting(calibration_apis=False):
            with self.assertRaisesRegex(ValueError, "execution differs.*prediction table"):
                datasets.load_calibration(data, root / "freeze.json")


if __name__ == "__main__":
    unittest.main()

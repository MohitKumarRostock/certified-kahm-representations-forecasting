"""Read-only onboarding checks against the included immutable primary example.

The delivered project includes the primary evidence this tutorial explains.
Negative cases use a private copy. No fixture trains, samples, certifies, or
creates another population audit.
"""

from contextlib import contextmanager, ExitStack
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kahm_demo import tutorial


PROJECT = Path(__file__).resolve().parents[1]
CANDIDATES = ["raw", "kahm_mem", "kahm_local"]
EVIDENCE_TREES = ("frozen/primary", "certificates/primary", "audits/primary", "data/primary")
FORBIDDEN_EXECUTION = (
    "kahm_demo.datasets.generate_bundle",
    "kahm_demo.datasets.materialize_calibration",
    "kahm_demo.datasets.sample_episodes",
    "kahm_demo.sampling.sample_episodes",
    "kahm_demo.oracle.risk_of_predictor",
    "kahm_demo.oracle.reference_summary",
    "kahm_demo.freeze.freeze_forecasters",
    "kahm_demo.encoder_workflow.fit_development_encoders",
    "kahm_demo.encoder_workflow.fit_encoder_candidates",
    "kahm_demo.encoder.fit_encoder_candidates",
    "kahm_demo.encoder.fit_encoder",
    "kahm_demo.forecast_workflow.fit_development_forecasters",
    "kahm_demo.forecast_workflow.fit_readout_features",
    "kahm_demo.forecast_workflow.fit_ridge_grid",
    "kahm_demo.readout_features.fit_readout_features",
    "kahm_demo.ridge.fit_ridge",
    "kahm_demo.ridge.fit_ridge_grid",
    "kahm_demo.certificate_workflow.certify_frozen_family",
    "kahm_demo.certificate_workflow.materialize_calibration",
    "kahm_demo.certificate_workflow.certify_upper_bound",
    "kahm_demo.risk_bounds.certify_upper_bound",
    "kahm_demo.certificate_workflow.write_calibration_ledger",
    "kahm_demo.calibration_ledger.write_calibration_ledger",
    "kahm_demo.audit_workflow.audit_certificate_family",
)


def evidence_hashes(project):
    return {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for tree in EVIDENCE_TREES for path in (project / tree).rglob("*") if path.is_file()}


def mock_audit_verification():
    record = json.loads((PROJECT / "audits/primary/audit.json").read_text())
    return {
        "status": "passed", "freeze_record_sha256": tutorial.PRIMARY_FREEZE_SHA256,
        "certificate_sha256": tutorial.PRIMARY_CERTIFICATE_SHA256,
        "audit_sha256": tutorial.PRIMARY_AUDIT_SHA256, "candidate_ids": CANDIDATES,
        "candidates": record["candidates"], "exact_population_risks_recomputed": True,
        "preserved_issued_endpoints": True, "source_identity_matches": True,
        "runtime_metadata_matches": True, "all_64_triples_per_candidate_checked": True,
        "independent_conditional_decomposition_checked": True,
        "calibration_resampled": False, "models_refitted": False, "root_search_repeated": False,
        "issued_certificate_modified": False, "model_selection_performed": False,
    }


class ObservationHistoryTests(unittest.TestCase):
    def test_all_exact_support_pairs_round_trip(self):
        observations = (0, Fraction(1, 4), 0.75, Fraction(1))
        for a in range(4):
            for b in range(4):
                self.assertEqual(tutorial.history_from_observations((observations[a], observations[b])), (a, b))

    def test_no_rounding_an_off_support_value_into_the_support(self):
        almost_quarter = Fraction(1, 4) + Fraction(1, 2 ** 200)
        self.assertEqual(float(almost_quarter), 0.25)
        with self.assertRaisesRegex(ValueError, "exactly"):
            tutorial.history_from_observations((almost_quarter, Fraction(3, 4)))

    def test_bad_observation_length_type_and_nonfinite_values(self):
        for values in ((), (0,), (0, 0.25, 0.75)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tutorial.history_from_observations(values)
        for values in ((True, 0.75), (0.25, False), ("0.25", 0.75), (0.25 + 0j, 0.75), None):
            with self.subTest(values=values), self.assertRaises(TypeError):
                tutorial.history_from_observations(values)
        for value in (float("nan"), float("inf"), -float("inf"), -0.25, 0.5, 2, 10 ** 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tutorial.history_from_observations((value, 0.75))


class PrimaryInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kahm_tutorial_test_")
        cls.project = Path(cls.temporary.name) / "project"
        cls.project.mkdir()
        for relative in EVIDENCE_TREES:
            shutil.copytree(PROJECT / relative, cls.project / relative)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def changed_file(self, relative, *, remove=False):
        path = self.project / relative
        before = path.read_bytes()
        try:
            if remove:
                path.unlink()
            else:
                path.write_bytes(before + b"\n")
            yield path
        finally:
            if path.is_symlink():
                path.unlink()
            path.write_bytes(before)

    def test_default_mode_reports_recorded_values_and_exact_histories(self):
        with mock.patch.object(tutorial.importlib, "import_module", side_effect=AssertionError("runtime import")):
            result = tutorial.inspect_primary(self.project)
        self.assertEqual(result["mode"], "archive_identity_only")
        self.assertTrue(result["archive_identity_checked"])
        for field in ("model_replayed", "certificate_reverified", "population_risk_recomputed",
                      "calibration_losses_recomputed", "calibration_resampled", "models_refitted", "root_search_repeated"):
            self.assertIs(result[field], False)
        self.assertEqual(result["candidate_ids"], CANDIDATES)
        self.assertEqual(result["history"]["state_labels"], [1, 2])
        self.assertEqual(result["history"]["history_id"], 6)
        self.assertEqual(result["history"]["observations"], [0.25, 0.75])
        self.assertEqual(result["inventory"]["freeze"]["payload_count"], 286)
        self.assertEqual(result["inventory"]["certificate"]["payload_count"], 10)
        self.assertEqual(result["inventory"]["audit"]["payload_count"], 6)
        self.assertEqual(result["inventory"]["external_dataset_file_count"], 6)
        certificate = json.loads((self.project / "certificates/primary/certificate.json").read_text())
        audit = json.loads((self.project / "audits/primary/audit.json").read_text())
        for row, issued, audited in zip(result["table"], certificate["candidates"], audit["candidates"], strict=True):
            self.assertEqual(row["empirical_mse"], issued["empirical_mse"])
            self.assertEqual(row["exact_population_mse"], audited["exact_population_mse"])
            self.assertEqual(row["issued_upper_endpoint"], issued["upper_endpoint"])
            self.assertEqual(row["issued_upper_endpoint_decimal_18"], issued["upper_endpoint_decimal_18"])
            self.assertEqual(row["certificate_status"], issued["status"])
            self.assertEqual(row["realized_coverage"], audited["realized_coverage"])
        features = result["history"]["recorded_encoder"]
        membership_width = sum(row["group_count"] for row in features["partition_groups"])
        self.assertEqual(len(features["raw"]), 2)
        self.assertEqual(len(features["memberships"]), membership_width)
        self.assertEqual(features["memberships_hex"], features["enriched_hex"][:membership_width])
        self.assertGreater(len(features["enriched"]), membership_width)
        self.assertEqual(len(features["enriched"]), features["feature_widths"]["kahm_local"])
        self.assertEqual(features["bank_id"], result["table"][1]["bank_id"])
        self.assertEqual(features["bank_id"], result["table"][2]["bank_id"])
        self.assertEqual([row["group_count"] for row in features["partition_groups"]], [2, 4])
        self.assertEqual([(row["membership_start"], row["membership_stop"])
                          for row in features["partition_groups"]], [(0, 2), (2, 6)])
        self.assertTrue(features["recorded_values_only"])
        json.dumps(result, allow_nan=False)

    def test_inspection_imports_no_numpy_or_notebook_packages_in_a_fresh_process(self):
        code = """
import builtins, json, pathlib, sys
project = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(project))
original = builtins.__import__
forbidden = {'numpy', 'IPython', 'jupyter', 'nbformat', 'ipywidgets', 'matplotlib'}
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('inspection imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked_import
from kahm_demo.tutorial import inspect_primary
result = inspect_primary(project)
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
print(json.dumps({'mode': result['mode'], 'model_replayed': result['model_replayed']}))
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PROJECT)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {"mode": "archive_identity_only", "model_replayed": False})

    def test_selected_history_changes_only_the_recorded_history_view(self):
        first = tutorial.inspect_primary(self.project, (0, 3))
        second = tutorial.inspect_primary(self.project, (3, 0))
        self.assertEqual(first["table"], second["table"])
        self.assertEqual(first["history"]["history_id"], 3)
        self.assertEqual(second["history"]["history_id"], 12)
        self.assertEqual(first["history"]["observations"], [0.0, 1.0])
        self.assertEqual(second["history"]["observations"], [1.0, 0.0])
        for inspected in (first, second):
            for row in inspected["history"]["recorded_predictions"]:
                original = json.loads((self.project / f'frozen/primary/predictions/{row["candidate_id"]}.json').read_text())
                self.assertEqual(row["prediction_hex"], original["predictions"][inspected["history"]["history_id"]]["prediction_hex"])

    def test_history_labels_require_two_strict_supported_integers(self):
        for history in ([1, 2], (True, 2), (1.0, 2), (0.25, 0.75), (1,), "12", None):
            with self.subTest(history=history), self.assertRaises(TypeError):
                tutorial.inspect_primary(self.project, history)
        for history in ((-1, 2), (1, 4)):
            with self.subTest(history=history), self.assertRaises(ValueError):
                tutorial.inspect_primary(self.project, history)

    def test_each_fixed_primary_anchor_is_required(self):
        for relative in ("frozen/primary/freeze.json", "certificates/primary/certificate.json", "audits/primary/audit.json"):
            with self.subTest(relative=relative), self.changed_file(relative), self.assertRaisesRegex(ValueError, "identity mismatch"):
                tutorial.inspect_primary(self.project)

    def test_each_bound_payload_inventory_is_checked(self):
        for relative in ("frozen/primary/predictions/raw.json", "certificates/primary/loss_ledger.jsonl", "audits/primary/population/raw.json"):
            with self.subTest(relative=relative), self.changed_file(relative), self.assertRaisesRegex(ValueError, "identity mismatch"):
                tutorial.inspect_primary(self.project)

    def test_linked_calibration_data_bytes_are_checked_without_loss_recomputation(self):
        with self.changed_file("data/primary/calibration/episodes.jsonl"), self.assertRaisesRegex(ValueError, "identity mismatch"):
            tutorial.inspect_primary(self.project)

    def test_missing_and_extra_payloads_are_rejected(self):
        with self.changed_file("frozen/primary/predictions/raw.json", remove=True), self.assertRaisesRegex(ValueError, "inventory differs"):
            tutorial.inspect_primary(self.project)
        extra = self.project / "audits/primary/unrecorded.json"
        try:
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                tutorial.inspect_primary(self.project)
        finally:
            extra.unlink()

    def test_payload_symlinks_are_rejected_even_if_target_bytes_match(self):
        relative = "frozen/primary/predictions/raw.json"
        with self.changed_file(relative, remove=True) as path:
            path.symlink_to(PROJECT / relative)
            with self.assertRaisesRegex(ValueError, "symlink"):
                tutorial.inspect_primary(self.project)

    def test_runtime_precheck_reports_missing_numpy_without_replay(self):
        with mock.patch.object(tutorial.importlib, "import_module",
                               side_effect=ModuleNotFoundError("NumPy is absent in this test", name="numpy")) as imported:
            result = tutorial.runtime_compatibility(self.project)
        imported.assert_called_once_with("numpy")
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["compatible"])
        self.assertFalse(result["dependency_status"]["numpy_available"])
        self.assertEqual(result["dependency_status"]["error"]["type"], "ModuleNotFoundError")
        self.assertIsNone(result["runtime_metadata_matches"])
        for field in ("model_replayed", "certificate_reverified", "population_risk_recomputed", "full_verification_performed"):
            self.assertIs(result[field], False)


class RuntimeAndExplicitVerificationTests(unittest.TestCase):
    def test_primary_runtime_precheck_matches_without_running_verification(self):
        audit = importlib.import_module("kahm_demo.audit_workflow")
        freeze = importlib.import_module("kahm_demo.freeze")
        with mock.patch.object(audit, "verify_audit_bundle", side_effect=AssertionError("audit verification called")), \
             mock.patch.object(freeze, "_verify_frozen_bundle", side_effect=AssertionError("model replay called")):
            result = tutorial.runtime_compatibility(PROJECT)
        self.assertTrue(result["compatible"], result)
        self.assertEqual(result["mode"], "runtime_precheck_only")
        self.assertTrue(result["runtime_metadata_matches"])
        self.assertTrue(result["source_identity_matches"])
        self.assertFalse(result["full_verification_performed"])
        self.assertFalse(result["model_replayed"])

    def test_runtime_and_source_differences_are_explicit(self):
        forecast = importlib.import_module("kahm_demo.forecast_workflow")
        certificate = importlib.import_module("kahm_demo.certificate_workflow")
        changed_runtime = {**forecast.forecast_runtime_record(), "numpy_version": "test-mismatched-version"}
        changed_sources = {**certificate._source_identities(), "risk_bounds.py": "0" * 64}
        with mock.patch.object(forecast, "forecast_runtime_record", return_value=changed_runtime), \
             mock.patch.object(certificate, "_source_identities", return_value=changed_sources):
            result = tutorial.runtime_compatibility(PROJECT)
        self.assertEqual(result["status"], "incompatible")
        self.assertFalse(result["compatible"])
        self.assertFalse(result["runtime_metadata_matches"])
        self.assertFalse(result["source_identity_matches"])
        self.assertEqual({row["field"] for row in result["runtime_differences"]}, {"numpy_version"})
        self.assertEqual([(row["record"], row["path"]) for row in result["source_differences"]], [("certificate", "risk_bounds.py")])
        self.assertFalse(result["certificate_reverified"])

    def test_explicit_verification_propagates_native_failure_without_fallback(self):
        audit = importlib.import_module("kahm_demo.audit_workflow")
        freeze = importlib.import_module("kahm_demo.freeze")
        with mock.patch.object(audit, "verify_audit_bundle", side_effect=ValueError("strict verifier sentinel")), \
             mock.patch.object(freeze, "_verify_frozen_bundle", side_effect=AssertionError("replay after failure")), \
             self.assertRaisesRegex(ValueError, "strict verifier sentinel"):
            tutorial.verify_primary(PROJECT)

    def test_explicit_verification_does_not_fallback_when_runtime_is_missing(self):
        with mock.patch.object(tutorial.importlib, "import_module",
                               side_effect=ModuleNotFoundError("missing runtime sentinel", name="numpy")), \
             self.assertRaisesRegex(ModuleNotFoundError, "missing runtime sentinel"):
            tutorial.verify_primary(PROJECT)

    def test_native_result_must_establish_every_claimed_verification_flag(self):
        audit = importlib.import_module("kahm_demo.audit_workflow")
        freeze = importlib.import_module("kahm_demo.freeze")
        result = mock_audit_verification()
        result["exact_population_risks_recomputed"] = False
        with mock.patch.object(audit, "verify_audit_bundle", return_value=result), \
             mock.patch.object(freeze, "_verify_frozen_bundle", side_effect=AssertionError("unverified replay")), \
             self.assertRaisesRegex(ValueError, "exact_population_risks_recomputed"):
            tutorial.verify_primary(PROJECT)

    def test_selected_history_replay_must_match_the_recorded_float_exactly(self):
        audit = importlib.import_module("kahm_demo.audit_workflow")
        freeze = importlib.import_module("kahm_demo.freeze")
        models = {candidate: SimpleNamespace(candidate_id=candidate, predict_one=lambda history: 0.0)
                  for candidate in CANDIDATES}
        preflight = {"status": "passed", "freeze_record_sha256": tutorial.PRIMARY_FREEZE_SHA256}
        with mock.patch.object(audit, "verify_audit_bundle", return_value=mock_audit_verification()), \
             mock.patch.object(freeze, "_verify_frozen_bundle", return_value=(preflight, models)), \
             self.assertRaisesRegex(ValueError, "replay differs"):
            tutorial.verify_primary(PROJECT)

    def test_actual_full_verification_replays_all_three_without_creation_or_refitting(self):
        before = evidence_hashes(PROJECT)
        audit = importlib.import_module("kahm_demo.audit_workflow")
        with ExitStack() as stack:
            for target in FORBIDDEN_EXECUTION:
                stack.enter_context(mock.patch(target, side_effect=AssertionError("forbidden tutorial action: " + target)))
            verified = stack.enter_context(mock.patch.object(audit, "verify_audit_bundle", wraps=audit.verify_audit_bundle))
            result = tutorial.verify_primary(PROJECT)
        verified.assert_called_once()
        self.assertEqual(result["mode"], "full_read_only_verification")
        self.assertEqual(result["status"], "passed")
        for field in ("archive_identity_checked", "model_replayed", "certificate_reverified",
                      "population_risk_recomputed", "calibration_losses_recomputed"):
            self.assertIs(result[field], True)
        for field in ("calibration_resampled", "models_refitted", "root_search_repeated"):
            self.assertIs(result[field], False)
        self.assertEqual([row["candidate_id"] for row in result["replayed_predictions"]], CANDIDATES)
        for replayed, recorded in zip(result["replayed_predictions"], result["history"]["recorded_predictions"], strict=True):
            self.assertTrue(replayed["matches_recorded"])
            self.assertEqual(replayed["prediction_hex"], recorded["prediction_hex"])
        self.assertEqual(result["audit_verification"]["audit_sha256"], tutorial.PRIMARY_AUDIT_SHA256)
        self.assertEqual(evidence_hashes(PROJECT), before)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

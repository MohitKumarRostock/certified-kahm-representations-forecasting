"""Fresh-run orchestration tests with explicitly mocked scientific stages.

These tests exercise ordering, file retention, source/plan binding, output
guards, and unchanged result reporting. They perform no fits, calibration
sampling, numerical certification, or population-risk calculations.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kahm_demo import reproduce
from kahm_demo._util import read_json, sha256_file, write_json
from kahm_demo.contract import CANDIDATE_IDS


class ReproduceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.output = self.base / "tutorial"
        self.calls = []
        self.failure_stage = None
        self.failure = RuntimeError("synthetic stage interruption")
        self.statuses = ["requirement_met"] * 3
        self.coverage = ["covered"] * 3
        self.cert_verified = True
        self.audit_verified = True
        self.search_count = 225
        self.bank_order = ["cap_1", "cap_2"]
        self.calibration_action = "materialize_reserved_stream"
        self.calibration_count = 4096
        self.mocks = {}
        stack = self.enterContext(ExitStack())
        boundaries = {
            "generate_bundle": self.generate,
            "fit_development_encoders": self.encoders,
            "fit_development_forecasters": self.forecasters,
            "freeze_forecasters": self.freeze,
            "certify_frozen_family": self.certify,
            "audit_certificate_family": self.audit,
        }
        for name, function in boundaries.items():
            self.mocks[name] = stack.enter_context(mock.patch.object(reproduce, name, side_effect=function))

    def stage(self, name, destination):
        self.assertEqual(self.calls, list(reproduce.STAGES[:len(self.calls)]))
        self.assertEqual(name, reproduce.STAGES[len(self.calls)])
        plan = read_json(self.output / "reproduction_plan.json")
        self.assertEqual(plan["schema"], reproduce.REPRODUCTION_PLAN_SCHEMA)
        self.assertEqual(plan["stage_order"], list(reproduce.STAGES))
        self.assertEqual(plan["source_sha256"], {"reproduce.py": sha256_file(Path(reproduce.__file__))})
        self.assertFalse((self.output / "reproduction.json").exists())
        self.calls.append(name)
        destination.mkdir()
        if self.failure_stage == name:
            (destination / "partial.txt").write_text("retain this synthetic partial stage")
            raise self.failure

    def generate(self, output, **kwargs):
        self.stage("generate_data", output)
        self.parameters = kwargs
        write_json(output / "manifest.json", {"schema": "kahm.dataset_bundle.v1", "run_id": kwargs["run_id"]})
        return {"status": "passed"}

    def encoders(self, bundle, output, *, caps):
        self.assertEqual(bundle, self.output / "data")
        self.assertEqual(caps, (1, 2))
        self.stage("fit_encoders", output)
        write_json(output / "manifest.json", {"schema": "kahm.encoder_development_bundle.v1"})
        return {"status": "completed", "bank_order": self.bank_order}

    def forecasters(self, bundle, encoders, output):
        self.assertEqual((bundle, encoders), (self.output / "data", self.output / "encoders"))
        self.stage("fit_forecasters", output)
        write_json(output / "manifest.json", {"schema": "kahm.forecast_development_bundle.v1"})
        return {"status": "completed", "search_count": self.search_count}

    def freeze(self, bundle, encoders, models, output):
        self.assertEqual((bundle, encoders, models), tuple(self.output / name for name in ("data", "encoders", "models")))
        self.stage("freeze_models", output)
        self.assertFalse((bundle / "calibration").exists())
        write_json(output / "freeze.json", {"schema": "kahm.model_freeze.v1"})
        return {"status": "frozen", "verification": {"status": "passed"},
                "freeze_record_sha256": sha256_file(output / "freeze.json")}

    def certify(self, bundle, freeze_path, output):
        self.assertEqual((bundle, freeze_path), (self.output / "data", self.output / "frozen/freeze.json"))
        self.stage("certify", output)
        (bundle / "calibration").mkdir()
        (bundle / "calibration/observations.txt").write_text("synthetic calibration marker")
        self.cert_rows = [{"candidate_id": candidate, "status": status, "exact_sample_identity": str(i)}
                          for i, (candidate, status) in enumerate(zip(CANDIDATE_IDS, self.statuses, strict=True))]
        write_json(output / "certificate.json", {"schema": "kahm.fitted_risk_certificate_family.v1",
                                                   "candidates": self.cert_rows})
        return {"status": "assessment_complete", "verification": {"status": "passed" if self.cert_verified else "failed"},
                "calibration_action": self.calibration_action, "calibration_count": self.calibration_count,
                "certificate_sha256": sha256_file(output / "certificate.json"), "candidates": deepcopy(self.cert_rows)}

    def audit(self, certificate_path, *, dataset_bundle, freeze_path, output):
        self.assertEqual((certificate_path, dataset_bundle, freeze_path),
                         (self.output / "certificates/certificate.json", self.output / "data", self.output / "frozen/freeze.json"))
        self.assertTrue(self.cert_verified)
        self.assertTrue((dataset_bundle / "calibration/observations.txt").is_file())
        self.stage("audit", output)
        self.audit_rows = [{"candidate_id": candidate, "certificate_status": status, "realized_coverage": coverage,
                            "exact_population_mse": {"numerator": str(i + 1), "denominator": "100"}}
                           for i, (candidate, status, coverage) in enumerate(zip(CANDIDATE_IDS, self.statuses, self.coverage, strict=True))]
        available = sum(value != "endpoint_unavailable" for value in self.coverage)
        self.audit_summary = {"candidate_count": 3, "available_endpoint_count": available,
                              "covered_endpoint_count": self.coverage.count("covered"),
                              "coverage_miss_count": self.coverage.count("coverage_miss"),
                              "unavailable_endpoint_count": 3 - available,
                              "all_available_endpoints_cover": None if available == 0 else "coverage_miss" not in self.coverage,
                              "all_three_covered": self.coverage == ["covered"] * 3}
        write_json(output / "audit.json", {"schema": "kahm.certified_family_population_audit.v1",
                                           "candidates": self.audit_rows, "summary": self.audit_summary})
        return {"status": "audit_complete", "verification": {"status": "passed" if self.audit_verified else "failed"},
                "audit_sha256": sha256_file(output / "audit.json"),
                "candidates": deepcopy(self.audit_rows), "summary": deepcopy(self.audit_summary)}

    def test_complete_order_plan_defaults_paths_hashes_and_no_printing(self):
        progress = []
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = reproduce.reproduce_demo(self.output, progress=progress.append)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertEqual(self.calls, list(reproduce.STAGES))
        self.assertEqual(progress, list(reproduce.PROGRESS_STAGES))
        self.assertEqual(self.parameters, {"master_seed": 20261110, "run_id": "tutorial-v0.1",
                                           "training_count": 4096, "validation_count": 1024})
        self.assertEqual(result["status"], "pipeline_complete")
        self.assertTrue(result["pipeline_complete"])
        self.assertEqual(result["candidates"], self.audit_rows)
        saved = read_json(self.output / "reproduction.json")
        self.assertEqual(saved["schema"], reproduce.REPRODUCTION_SCHEMA)
        self.assertEqual(saved["record_sha256"], reproduce._digest(saved))
        self.assertEqual(result["reproduction_sha256"], sha256_file(self.output / "reproduction.json"))
        self.assertEqual(saved["reproduction_plan"]["sha256"], sha256_file(self.output / "reproduction_plan.json"))
        for artifact in saved["artifacts"].values():
            self.assertEqual(artifact["sha256"], sha256_file(self.output / artifact["path"]))
        self.assertEqual(set(saved["artifacts"]), {"dataset_manifest", "encoder_manifest", "model_manifest", "freeze", "certificate", "audit"})
        self.assertFalse((self.output / "reproduction_failure.json").exists())
        for boundary in self.mocks.values():
            self.assertEqual(boundary.call_count, 1)

    def test_custom_development_counts_do_not_change_calibration_or_menu(self):
        result = reproduce.reproduce_demo(self.output, master_seed=817821, run_id="test-step8-wrapper",
                                           training_count=32, validation_count=8)
        self.assertEqual(self.parameters["training_count"], 32)
        self.assertEqual(self.parameters["validation_count"], 8)
        self.assertEqual(result["calibration_count"], 4096)
        self.assertEqual(result["forecast_search_count"], 225)
        plan = read_json(self.output / "reproduction_plan.json")
        self.assertEqual(plan["master_seed_integer"], "817821")
        self.assertEqual(plan["run_id"], "test-step8-wrapper")
        self.assertEqual({v["role"] for v in plan["streams"].values()}, {"training", "validation", "calibration"})

    def test_unfavorable_outcomes_complete_without_repair_or_filtering(self):
        self.statuses = ["requirement_met", "unresolved", "certificate_unavailable"]
        self.coverage = ["coverage_miss", "covered", "endpoint_unavailable"]
        result = reproduce.reproduce_demo(self.output)
        self.assertTrue(result["pipeline_complete"])
        self.assertFalse(result["certificate_summary"]["all_three_requirements_met"])
        self.assertFalse(result["certificate_summary"]["any_kahm_requirement_met"])
        self.assertEqual(result["certificate_summary"]["status_counts"], dict.fromkeys(self.statuses, 1))
        self.assertEqual(result["candidates"], self.audit_rows)
        self.assertEqual(result["audit_summary"]["coverage_miss_count"], 1)
        self.assertFalse(result["unfavorable_outcomes_filtered"])
        self.assertEqual(self.calls, list(reproduce.STAGES))

    def test_all_unavailable_is_completed_and_retained(self):
        self.statuses = ["certificate_unavailable"] * 3
        self.coverage = ["endpoint_unavailable"] * 3
        result = reproduce.reproduce_demo(self.output)
        self.assertTrue(result["pipeline_complete"])
        self.assertEqual(result["certificate_summary"]["status_counts"]["certificate_unavailable"], 3)
        self.assertIsNone(result["audit_summary"]["all_available_endpoints_cover"])
        self.assertEqual(len(result["candidates"]), 3)

    def test_failure_keeps_prior_partial_files_and_stops_without_retry(self):
        self.failure_stage = "certify"
        with self.assertRaisesRegex(RuntimeError, "synthetic stage interruption"):
            reproduce.reproduce_demo(self.output)
        self.assertEqual(self.calls, list(reproduce.STAGES[:5]))
        self.assertTrue((self.output / "frozen/freeze.json").is_file())
        self.assertTrue((self.output / "certificates/partial.txt").is_file())
        self.assertFalse((self.output / "reproduction.json").exists())
        failure = read_json(self.output / "reproduction_failure.json")
        self.assertEqual(failure["phase"], "certify")
        self.assertEqual(failure["completed_stages"], list(reproduce.STAGES[:4]))
        self.assertFalse(failure["automatic_retries_performed"])
        self.assertEqual(self.mocks["certify_frozen_family"].call_count, 1)
        self.mocks["audit_certificate_family"].assert_not_called()
        original = (self.output / "reproduction_failure.json").read_bytes()
        with self.assertRaises(FileExistsError):
            reproduce.reproduce_demo(self.output)
        self.assertEqual((self.output / "reproduction_failure.json").read_bytes(), original)

    def test_interrupt_is_recorded_without_losing_partial_evidence(self):
        self.failure_stage = "fit_encoders"
        self.failure = KeyboardInterrupt("synthetic interrupt")
        with self.assertRaises(KeyboardInterrupt):
            reproduce.reproduce_demo(self.output)
        failure = read_json(self.output / "reproduction_failure.json")
        self.assertEqual(failure["error_type"], "KeyboardInterrupt")
        self.assertTrue((self.output / "encoders/partial.txt").exists())
        self.mocks["fit_development_forecasters"].assert_not_called()

    def test_unverified_certificate_prevents_population_audit(self):
        self.cert_verified = False
        with self.assertRaisesRegex(ValueError, "artifact verification"):
            reproduce.reproduce_demo(self.output)
        self.mocks["audit_certificate_family"].assert_not_called()
        self.assertTrue((self.output / "data/calibration/observations.txt").exists())
        self.assertEqual(read_json(self.output / "reproduction_failure.json")["phase"], "certify")

    def test_unverified_audit_does_not_publish_completion(self):
        self.audit_verified = False
        with self.assertRaisesRegex(ValueError, "artifact verification"):
            reproduce.reproduce_demo(self.output)
        self.assertTrue((self.output / "audits/audit.json").exists())
        self.assertFalse((self.output / "reproduction.json").exists())

    def test_incomplete_search_stops_before_freeze(self):
        self.search_count = 45
        with self.assertRaisesRegex(ValueError, "225-fit"):
            reproduce.reproduce_demo(self.output)
        self.mocks["freeze_forecasters"].assert_not_called()
        self.mocks["certify_frozen_family"].assert_not_called()

    def test_missing_bank_stops_before_forecast_fitting(self):
        self.bank_order = ["cap_2"]
        with self.assertRaisesRegex(ValueError, "both declared banks"):
            reproduce.reproduce_demo(self.output)
        self.mocks["fit_development_forecasters"].assert_not_called()

    def test_reused_calibration_is_not_a_fresh_reproduction(self):
        self.calibration_action = "reuse_matching_archived_calibration"
        with self.assertRaisesRegex(ValueError, "unused reserved"):
            reproduce.reproduce_demo(self.output)
        self.mocks["audit_certificate_family"].assert_not_called()

    def test_calibration_count_cannot_be_changed(self):
        self.calibration_count = 16
        with self.assertRaisesRegex(ValueError, "calibration count"):
            reproduce.reproduce_demo(self.output)
        self.mocks["audit_certificate_family"].assert_not_called()

    def test_existing_output_file_directory_or_dangling_symlink_is_rejected(self):
        locations = [self.base / "existing-dir", self.base / "existing-file", self.base / "existing-link"]
        locations[0].mkdir()
        locations[1].write_text("retain me")
        locations[2].symlink_to(self.base / "missing")
        for path in locations:
            with self.subTest(path=path), self.assertRaises(FileExistsError):
                reproduce.reproduce_demo(path)
        self.assertEqual(locations[1].read_text(), "retain me")
        self.assertTrue(locations[2].is_symlink())
        self.assertEqual(self.calls, [])

    def test_each_existing_bundle_marker_blocks_nested_output(self):
        for index, (marker_name, schemas) in enumerate(reproduce._BUNDLE_MARKERS.items()):
            for j, schema in enumerate(schemas):
                parent = self.base / f"protected-{index}-{j}"
                parent.mkdir()
                write_json(parent / marker_name, {"schema": schema})
                target = parent / "new-level" / "new-run"
                with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "outside existing bundles"):
                    reproduce.reproduce_demo(target)
                self.assertFalse((parent / "new-level").exists())
        self.assertEqual(self.calls, [])

    def test_symlink_into_existing_bundle_is_blocked(self):
        protected = self.base / "protected"
        protected.mkdir()
        write_json(protected / "manifest.json", {"schema": "kahm.dataset_bundle.v1"})
        (protected / "nested").mkdir()
        alias = self.base / "alias"
        alias.symlink_to(protected / "nested", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside existing bundles"):
            reproduce.reproduce_demo(alias / "fresh")
        self.assertFalse((protected / "nested/fresh").exists())

    def test_symlink_out_of_existing_bundle_is_also_blocked(self):
        protected, external = self.base / "protected", self.base / "external"
        protected.mkdir()
        external.mkdir()
        write_json(protected / "audit_plan.json", {"schema": "kahm.population_audit_plan.v1"})
        (protected / "exit").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside existing bundles"):
            reproduce.reproduce_demo(protected / "exit/fresh")
        self.assertFalse((external / "fresh").exists())

    def test_ordinary_parent_manifest_is_not_mistaken_for_kahm_bundle(self):
        write_json(self.base / "manifest.json", {"schema": "other.project.v1"})
        result = reproduce.reproduce_demo(self.output)
        self.assertTrue(result["pipeline_complete"])

    def test_malformed_ancestor_marker_is_not_ignored(self):
        (self.base / "manifest.json").write_text('{"schema":')
        with self.assertRaisesRegex(ValueError, "cannot inspect possible bundle marker"):
            reproduce.reproduce_demo(self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_parameters_have_no_output_or_stage_calls(self):
        cases = [{"training_count": 0}, {"training_count": True}, {"validation_count": -1},
                 {"validation_count": 1.0}, {"master_seed": True}, {"master_seed": -1},
                 {"run_id": "   "}, {"run_id": 4}, {"progress": "print"}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                reproduce.reproduce_demo(self.output, **kwargs)
            self.assertFalse(self.output.exists())
        self.assertEqual(self.calls, [])

    def test_progress_exception_retains_plan_and_does_not_start_stage(self):
        def broken_progress(stage):
            if stage == "fit_encoders":
                raise RuntimeError("synthetic progress failure")
        with self.assertRaisesRegex(RuntimeError, "progress failure"):
            reproduce.reproduce_demo(self.output, progress=broken_progress)
        self.mocks["fit_development_encoders"].assert_not_called()
        self.assertTrue((self.output / "data/manifest.json").exists())
        self.assertEqual(read_json(self.output / "reproduction_failure.json")["phase"], "fit_encoders")

    def test_changed_plan_is_rejected_before_next_stage(self):
        def tamper(stage):
            if stage == "fit_encoders":
                path = self.output / "reproduction_plan.json"
                plan = read_json(path)
                plan["calibration_count"] = 1
                write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "plan changed"):
            reproduce.reproduce_demo(self.output, progress=tamper)
        self.mocks["fit_development_encoders"].assert_not_called()
        self.assertTrue((self.output / "reproduction_failure.json").exists())

    def test_stage_destination_injected_by_callback_is_never_overwritten(self):
        def inject(stage):
            if stage == "fit_encoders":
                (self.output / "encoders").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(FileExistsError, "not fresh"):
            reproduce.reproduce_demo(self.output, progress=inject)
        self.mocks["fit_development_encoders"].assert_not_called()
        self.assertTrue((self.output / "encoders").is_symlink())

    def test_changed_completed_artifact_prevents_success_record(self):
        def tamper(stage):
            if stage == "complete":
                write_json(self.output / "models/manifest.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "completed artifact changed"):
            reproduce.reproduce_demo(self.output, progress=tamper)
        self.assertFalse((self.output / "reproduction.json").exists())
        self.assertEqual(read_json(self.output / "reproduction_failure.json")["completed_stages"], list(reproduce.STAGES))


if __name__ == "__main__":
    unittest.main()

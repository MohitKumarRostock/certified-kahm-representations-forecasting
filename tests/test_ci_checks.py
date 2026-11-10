"""CI policy tests; scientific subprocesses use explicit synthetic boundaries."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import ci_checks as ci


def named_case(test_id, action=None):
    class Case(unittest.TestCase):
        def id(self):
            return test_id

        def runTest(self):
            if action is not None:
                action(self)
    return Case()


def small_suite(portable_action=None):
    tests = [named_case(name) for name in ci.PRIMARY_RUNTIME_ONLY_TESTS]
    tests.append(named_case("test_tutorial.PrimaryInspectionTests.test_portable_inspection", portable_action))
    return unittest.TestSuite([unittest.TestSuite(tests[:2]), unittest.TestSuite(tests[2:])])


class SelectionTests(unittest.TestCase):
    def test_portable_excludes_exactly_three_and_retains_archive_inspection(self):
        suite, report = ci.select_tests(small_suite())
        self.assertEqual(report["selected_count"], 1)
        self.assertEqual(set(report["excluded_ids"]), set(ci.PRIMARY_RUNTIME_ONLY_TESTS))
        self.assertIn("PrimaryInspectionTests", ci.flatten_tests(suite)[0].id())
        self.assertTrue(all(row["reason"] for row in report["primary_runtime_tests"]))

    def test_strict_gate_selects_the_three_named_primary_cases(self):
        suite, report = ci.select_tests(small_suite(), "strict-primary")
        self.assertEqual({test.id() for test in ci.flatten_tests(suite)}, set(ci.PRIMARY_RUNTIME_ONLY_TESTS))
        self.assertEqual(report["selected_count"], 3)

    def test_stale_exclusion_names_fail_instead_of_silently_shrinking_coverage(self):
        with self.assertRaisesRegex(ValueError, "missing from discovery"):
            ci.select_tests(unittest.TestSuite([named_case("test_portable.case")]))

    def test_duplicate_test_ids_and_unknown_modes_fail(self):
        cases = ci.flatten_tests(small_suite())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ci.select_tests(unittest.TestSuite(cases + [named_case(cases[0].id())]))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ci.select_tests(small_suite(), "automatic-fallback")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        (self.project / "kahm_demo").mkdir(parents=True)
        (self.project / "kahm_demo/__init__.py").write_text('__version__ = "0.1.0"\n')
        (self.project / "pyproject.toml").write_text('[project]\nname="kahm-demonstrator"\nversion="0.1.0"\n')
        (self.project / "README.md").write_text("Synthetic CI boundary fixture.\n")
        self.report = self.root / "portable.json"

    def test_test_failure_is_retained_and_not_retried(self):
        calls = []
        def failing(test):
            calls.append("called")
            test.fail("synthetic unit failure")
        with mock.patch.object(ci, "discover_tests", return_value=small_suite(failing)), \
             mock.patch.object(ci, "_runtime_precheck", side_effect=AssertionError("portable primary precheck")):
            result = ci.run_test_gate(self.project, mode="portable", report=self.report)
        self.assertEqual(calls, ["called"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failures"], 1)
        self.assertFalse(result["original_primary_verification_performed"])
        self.assertEqual(ci._read(self.report), result)
        self.assertIn("synthetic unit failure", self.report.with_suffix(".log").read_text())

    def test_unexpected_unittest_skip_is_a_failed_gate(self):
        with mock.patch.object(ci, "discover_tests", return_value=small_suite(lambda test: test.skipTest("unapproved skip"))):
            result = ci.run_test_gate(self.project, mode="portable", report=self.report)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["skipped"]), 1)

    def test_primary_runtime_mismatch_fails_before_any_tests_without_fallback(self):
        with mock.patch.object(ci, "discover_tests", return_value=small_suite()), \
             mock.patch.object(ci, "_runtime_precheck", return_value={"compatible": False, "status": "incompatible"}), \
             mock.patch.object(ci.unittest.TextTestRunner, "run", side_effect=AssertionError("tests after mismatch")) as run:
            result = ci.run_test_gate(self.project, mode="strict-primary", report=self.report)
        run.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertIn("no skip, fallback", result["error"]["message"])
        self.assertFalse(result["runtime_identity_checks_relaxed"])

    def test_primary_native_precheck_failure_is_not_swallowed_as_a_skip(self):
        with mock.patch.object(ci, "discover_tests", return_value=small_suite()), \
             mock.patch.object(ci, "_runtime_precheck", side_effect=ModuleNotFoundError("missing runtime")):
            result = ci.run_test_gate(self.project, mode="strict-primary", report=self.report)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["type"], "ModuleNotFoundError")
        self.assertNotIn("skipped", result)

    def test_existing_report_is_preserved(self):
        self.report.write_text("retained earlier failure")
        with self.assertRaises(FileExistsError):
            ci.run_test_gate(self.project, mode="portable", report=self.report)
        self.assertEqual(self.report.read_text(), "retained earlier failure")

    def test_fresh_output_inside_checkout_or_existing_output_is_rejected(self):
        with mock.patch.object(ci, "_command", side_effect=AssertionError("unexpected command")):
            with self.assertRaisesRegex(ValueError, "outside the checkout"):
                ci.fresh_wheel(self.project, output=self.project / "new-run")
            existing = self.root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                ci.fresh_wheel(self.project, output=existing)

    def test_fresh_output_cannot_nest_in_an_existing_experiment(self):
        prior = self.root / "prior"
        prior.mkdir()
        (prior / "reproduction_plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "existing scientific/experiment"):
            ci.fresh_wheel(self.project, output=prior / "nested")
        self.assertFalse((prior / "nested").exists())

    def test_external_data_encoder_and_model_manifests_protect_both_outputs(self):
        schemas = ("kahm.dataset_bundle.v1", "kahm.encoder_development_bundle.v1", "kahm.forecast_development_bundle.v1")
        for index, schema in enumerate(schemas):
            prior = self.root / f"existing-bundle-{index}"
            prior.mkdir()
            ci._write_new(prior / "manifest.json", {"schema": schema})
            for guard, relative in ((ci._report_path, "nested/report.json"), (ci._fresh_output, "nested/fresh")):
                with self.subTest(schema=schema, guard=guard.__name__), \
                     self.assertRaisesRegex(ValueError, "existing scientific/experiment"):
                    guard(prior / relative, self.project)
            self.assertFalse((prior / "nested").exists())

    def test_calibration_reservation_and_receipt_protect_both_outputs(self):
        for index, (name, schema) in enumerate((("calibration_plan.json", "kahm.calibration_plan.v1"),
                                               ("receipt.json", "kahm.calibration_receipt.v1"))):
            prior = self.root / f"calibration-bundle-{index}"
            prior.mkdir()
            ci._write_new(prior / name, {"schema": schema})
            for guard, relative in ((ci._report_path, "new-report.json"), (ci._fresh_output, "new-run")):
                with self.subTest(marker=name, guard=guard.__name__), \
                     self.assertRaisesRegex(ValueError, "existing scientific/experiment"):
                    guard(prior / relative, self.project)

    def test_preserved_onboarding_run_blocks_reports_but_allows_a_fresh_sibling(self):
        prior = self.project / "reports/onboarding_run"
        prior.mkdir(parents=True)
        ci._write_new(prior / "reproduction_plan.json", {"schema": "kahm.demo_reproduction_plan.v1"})
        with self.assertRaisesRegex(ValueError, "existing scientific/experiment"):
            ci._report_path(prior / "nested/new-report.json", self.project)
        sibling = self.project / "reports/new-ci-report.json"
        self.assertEqual(ci._report_path(sibling, self.project), sibling)
        external = self.root / "fresh-sibling"
        self.assertEqual(ci._fresh_output(external, self.project), external)
        self.assertFalse((prior / "nested").exists())
        self.assertFalse(sibling.exists())

    def test_symlink_into_scientific_bundle_is_blocked_for_reports_and_runs(self):
        prior = self.root / "protected"
        (prior / "nested").mkdir(parents=True)
        ci._write_new(prior / "manifest.json", {"schema": "kahm.dataset_bundle.v1"})
        alias = self.root / "alias"
        alias.symlink_to(prior / "nested", target_is_directory=True)
        for guard, name in ((ci._report_path, "report.json"), (ci._fresh_output, "fresh")):
            with self.subTest(guard=guard.__name__), self.assertRaisesRegex(ValueError, "existing scientific/experiment"):
                guard(alias / name, self.project)
        self.assertEqual(list((prior / "nested").iterdir()), [])

    def test_lexical_route_out_of_a_scientific_bundle_is_also_blocked(self):
        prior, external = self.root / "protected", self.root / "external"
        prior.mkdir()
        external.mkdir()
        ci._write_new(prior / "audit_plan.json", {"schema": "kahm.population_audit_plan.v1"})
        alias = prior / "exit"
        alias.symlink_to(external, target_is_directory=True)
        for guard, name in ((ci._report_path, "report.json"), (ci._fresh_output, "fresh")):
            with self.subTest(guard=guard.__name__), self.assertRaisesRegex(ValueError, "existing scientific/experiment"):
                guard(alias / name, self.project)
        self.assertEqual(list(external.iterdir()), [])

    def test_unreadable_marker_and_dangling_parent_are_not_bypassed(self):
        malformed = self.root / "malformed"
        malformed.mkdir()
        (malformed / "manifest.json").write_text('{"schema":')
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing", target_is_directory=True)
        for guard in (ci._report_path, ci._fresh_output):
            with self.subTest(guard=guard.__name__), self.assertRaisesRegex(ValueError, "cannot inspect"):
                guard(malformed / "new", self.project)
            with self.subTest(guard=guard.__name__), self.assertRaisesRegex(ValueError, "dangling symlink"):
                guard(dangling / "new", self.project)

    def test_unrelated_manifest_does_not_block_a_fresh_destination(self):
        unrelated = self.root / "ordinary-directory"
        unrelated.mkdir()
        ci._write_new(unrelated / "manifest.json", {"schema": "other.project.v1"})
        self.assertEqual(ci._report_path(unrelated / "report.json", self.project), unrelated / "report.json")
        self.assertEqual(ci._fresh_output(unrelated / "new-run", self.project), unrelated / "new-run")

    def test_build_failure_keeps_plan_and_partial_logs_without_starting_experiment(self):
        output = self.root / "fresh"
        def fail(arguments, *, cwd, logs, label):
            (logs / "partial.log").write_text("retained build error")
            raise RuntimeError("synthetic build failure")
        with mock.patch.object(ci, "_command", side_effect=fail) as command:
            result = ci.fresh_wheel(self.project, output=output)
        self.assertEqual(command.call_count, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["fresh_pipeline_invocation_count"], 0)
        self.assertTrue((output / "ci_plan.json").exists())
        self.assertTrue((output / "logs/partial.log").exists())
        self.assertEqual(ci._read(output / "ci_report.json"), result)

    def test_fresh_pipeline_is_invoked_once_and_failure_is_retained(self):
        output = self.root / "fresh"
        phases = []
        def command(arguments, *, cwd, logs, label):
            phases.append(label)
            if label == "build_wheel":
                (output / "wheelhouse/kahm_demonstrator-0.1.0-py3-none-any.whl").write_bytes(b"synthetic wheel")
            elif label == "installed_origin_probe":
                return json.dumps({"status": "passed", "checkout_shadowed_import": False})
            elif label == "installed_archive_inspection":
                return json.dumps({"status": "passed", "mode": "archive_identity_only"})
            elif label == "fresh_pipeline":
                (output / "run").mkdir()
                (output / "run/reproduction_failure.json").write_text('{"status":"pipeline_failed"}')
                (logs / "fresh_pipeline.stderr.log").write_text("synthetic pipeline failure")
                raise RuntimeError("synthetic pipeline failure")
            return ""
        with mock.patch.object(ci, "_command", side_effect=command):
            result = ci.fresh_wheel(self.project, output=output, run_id="test-step9-no-retry")
            with self.assertRaises(FileExistsError):
                ci.fresh_wheel(self.project, output=output, run_id="test-step9-no-retry")
        self.assertEqual(phases.count("fresh_pipeline"), 1)
        self.assertEqual(result["fresh_pipeline_invocation_count"], 1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["automatic_retries_performed"])
        self.assertTrue((output / "run/reproduction_failure.json").exists())

    def test_subprocess_failure_keeps_stdout_stderr_and_calls_once(self):
        logs = self.root / "logs"
        logs.mkdir()
        response = mock.Mock(returncode=7, stdout="partial output", stderr="native failure")
        with mock.patch.object(ci.subprocess, "run", return_value=response) as run:
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                ci._command(["test-only"], cwd=self.root, logs=logs, label="stage")
        run.assert_called_once()
        self.assertEqual((logs / "stage.stdout.log").read_text(), "partial output")
        self.assertEqual((logs / "stage.stderr.log").read_text(), "native failure")

    def test_cli_exit_status_follows_execution_not_a_success_claim(self):
        with mock.patch.object(ci, "fresh_wheel", return_value={"status": "failed", "error": "native failure"}) as call, \
             redirect_stdout(io.StringIO()) as stdout:
            code = ci.main(["fresh-wheel", "--project", str(self.project), "--output", str(self.root / "new")])
        self.assertEqual(code, 1)
        call.assert_called_once()
        self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")


class OutcomeRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, statuses, coverages):
        for name in ("data", "encoders", "models", "frozen", "certificates", "audits"):
            (self.root / name).mkdir()
        issued = [{"candidate_id": candidate, "status": status,
                   "empirical_mse": {"numerator": "1", "denominator": "20"},
                   "upper_endpoint": None if status == "certificate_unavailable" else {"numerator": "1", "denominator": "10"}}
                  for candidate, status in zip(ci.CANDIDATES, statuses, strict=True)]
        audited = [{"candidate_id": candidate, "certificate_status": row["status"],
                    "calibration_empirical_mse": row["empirical_mse"], "issued_upper_endpoint": row["upper_endpoint"],
                    "realized_coverage": coverage}
                   for candidate, row, coverage in zip(ci.CANDIDATES, issued, coverages, strict=True)]
        summary = {"candidate_count": 3, "covered_endpoint_count": coverages.count("covered"),
                   "coverage_miss_count": coverages.count("coverage_miss"),
                   "unavailable_endpoint_count": coverages.count("endpoint_unavailable")}
        for relative in ci.PRINCIPAL_ARTIFACTS.values():
            value = ({"candidates": issued} if relative == "certificates/certificate.json" else
                     {"candidates": audited, "summary": summary} if relative == "audits/audit.json" else
                     {"schema": "test.synthetic_principal_artifact"})
            ci._write_new(self.root / relative, value)
        ci._write_new(self.root / "reproduction_plan.json", {"schema": "test.synthetic_reproduction_plan"})
        counts = {status: statuses.count(status) for status in ci.CERTIFICATE_STATUSES}
        record = {
            "schema": "kahm.demo_reproduction.v1", "status": "pipeline_complete", "pipeline_complete": True,
            "completed_stages": ci.PIPELINE_STAGES, "candidate_ids": ci.CANDIDATES,
            "training_count": 4096, "validation_count": 1024, "calibration_count": 4096,
            "forecast_search_count": 225, "encoder_caps": [1, 2],
            "automatic_retries_performed": False, "unfavorable_outcomes_filtered": False,
            "artifacts": {name: {"path": relative, "sha256": ci._sha(self.root / relative)}
                          for name, relative in ci.PRINCIPAL_ARTIFACTS.items()},
            "reproduction_plan": {"path": "reproduction_plan.json", "sha256": ci._sha(self.root / "reproduction_plan.json")},
            "candidates": audited, "audit_summary": summary,
            "certificate_summary": {"all_three_requirements_met": counts["requirement_met"] == 3,
                                    "any_kahm_requirement_met": "requirement_met" in statuses[1:],
                                    "status_counts": counts},
            "run_id": "test-ci-synthetic-outcomes", "master_seed_integer": "90210",
        }
        record["record_sha256"] = ci._digest(record)
        ci._write_new(self.root / "reproduction.json", record)
        return {**deepcopy(record), "reproduction_sha256": ci._sha(self.root / "reproduction.json")}

    def test_unresolved_unavailable_and_coverage_miss_remain_successful_execution(self):
        returned = self.fixture(["requirement_met", "unresolved", "certificate_unavailable"],
                                ["coverage_miss", "covered", "endpoint_unavailable"])
        result = ci.summarize_reproduction(self.root, returned)
        self.assertTrue(result["pipeline_complete"])
        self.assertTrue(result["all_outcomes_retained"])
        self.assertFalse(result["certificate_summary"]["all_three_requirements_met"])
        self.assertEqual(result["audit_summary"]["coverage_miss_count"], 1)
        self.assertEqual(len(result["candidates"]), 3)

    def test_all_unavailable_is_retained_without_changing_completion(self):
        returned = self.fixture(["certificate_unavailable"] * 3, ["endpoint_unavailable"] * 3)
        result = ci.summarize_reproduction(self.root, returned)
        self.assertTrue(result["pipeline_complete"])
        self.assertEqual(result["certificate_summary"]["status_counts"]["certificate_unavailable"], 3)

    def test_omitted_returned_candidate_is_rejected(self):
        returned = self.fixture(["requirement_met"] * 3, ["covered"] * 3)
        returned["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "outcomes differ"):
            ci.summarize_reproduction(self.root, returned)

    def test_changed_principal_artifact_or_retained_failure_rejects_completion(self):
        returned = self.fixture(["requirement_met"] * 3, ["covered"] * 3)
        (self.root / "reproduction_failure.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "failure record"):
            ci.summarize_reproduction(self.root, returned)
        (self.root / "reproduction_failure.json").unlink()
        with (self.root / "models/manifest.json").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "artifact identity"):
            ci.summarize_reproduction(self.root, returned)


if __name__ == "__main__":
    unittest.main()

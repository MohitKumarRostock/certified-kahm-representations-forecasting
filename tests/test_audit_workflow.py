"""Post-certificate audit orchestration with a small, explicitly synthetic fixture.

Only the expensive prior-certificate verification boundary is replaced. The
real contract checks, prediction-table parser, exact population calculations,
source snapshots, artifact writer, and independent audit verifier all run. The
full fitted-model/certificate prerequisite is covered by the separate
calibration assessment integration tests; this fixture does not claim that its illustrative endpoints
are outputs of that numerical certifier. No primary data or models are used.
"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kahm_demo import audit_workflow as workflow
from kahm_demo._util import canonical_json_bytes, read_json, sha256_file, write_json
from kahm_demo.calibration_protocol import CALIBRATION_PROTOCOL_SHA256
from kahm_demo.contract import CANDIDATE_IDS, CONTRACT_SHA256
from kahm_demo.law import HISTORY_KEYS, OBSERVATION_FLOATS
from kahm_demo.oracle import prediction_table_from_mapping


TEST_RUN_ID = "test-step7-synthetic-audit-boundary"


def ratio(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def rehash_record(path, record):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json(path, record)


def refresh_audit_envelope(root, record=None):
    """Repair outer hashes so a negative case reaches semantic verification."""
    path = root / "audit.json"
    record = read_json(path) if record is None else record
    for row in record["candidates"]:
        population_path = root / row["population_audit_path"]
        if population_path.is_file():
            row["population_audit_sha256"] = sha256_file(population_path)
    record["audit_plan_sha256"] = sha256_file(root / "audit_plan.json")
    record["files"] = [
        {"path": item.relative_to(root).as_posix(), "bytes": item.stat().st_size,
         "sha256": sha256_file(item)}
        for item in sorted(root.rglob("*")) if item.is_file() and item != path
    ]
    rehash_record(path, record)


def file_identities(root):
    return {path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, sha256_file(path))
            for path in root.rglob("*") if path.is_file()}


@contextmanager
def prohibit_sampling_fitting_and_endpoint_search():
    targets = (
        "kahm_demo.datasets.generate_bundle",
        "kahm_demo.datasets.sample_episodes", "kahm_demo.sampling.sample_episodes",
        "kahm_demo.datasets.materialize_calibration",
        "kahm_demo.certificate_workflow.materialize_calibration",
        "kahm_demo.certificate_workflow.certify_frozen_family",
        "kahm_demo.risk_bounds.certify_upper_bound",
        "kahm_demo.certificate_workflow.certify_upper_bound",
        "kahm_demo.encoder_workflow.fit_development_encoders",
        "kahm_demo.encoder_workflow.fit_encoder_candidates",
        "kahm_demo.encoder.fit_encoder_candidates",
        "kahm_demo.forecast_workflow.fit_development_forecasters",
        "kahm_demo.forecast_workflow.fit_readout_features",
        "kahm_demo.forecast_workflow.fit_ridge_grid",
        "kahm_demo.readout_features.fit_readout_features",
        "kahm_demo.ridge.fit_ridge_grid", "kahm_demo.ridge.fit_ridge",
        "kahm_demo.oracle.risk_of_predictor",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(mock.patch(target, side_effect=AssertionError(
                f"post-certificate auditing called a prohibited entry point: {target}")))
        yield


class AuditWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="kahm-test-step7-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.frozen, self.assessment, self.audit = [
            self.root / name for name in ("data", "frozen", "assessment", "audit")
        ]
        for directory in (self.data, self.frozen, self.assessment):
            directory.mkdir()
        (self.frozen / "predictions").mkdir()
        (self.frozen / "models").mkdir()
        (self.assessment / "bounds").mkdir()
        write_json(self.data / "test_fixture.json", {
            "purpose": "synthetic audit-boundary test; no sampled episodes or fitted models",
            "run_id": TEST_RUN_ID,
        })

        mem_predictions = {(a, b): OBSERVATION_FLOATS[(a + b) % 4]
                           for a, b in HISTORY_KEYS}
        # Each change adds exactly 1/80 to the nominal predictor's risk 1/16.
        # Three changes therefore put the risk exactly at tolerance 1/10.
        for history in ((0, 1), (1, 0), (2, 3)):
            mem_predictions[history] = 0.75
        self.predictions = {
            "raw": {history: 0.5 for history in HISTORY_KEYS},
            "kahm_mem": mem_predictions,
            "kahm_local": {history: 0.1 for history in HISTORY_KEYS},
        }
        local_value = Fraction.from_float(0.1)
        self.expected_risks = {
            "raw": Fraction(5, 32), "kahm_mem": Fraction(1, 10),
            "kahm_local": Fraction(5, 32) + (local_value - Fraction(1, 2)) ** 2,
        }
        self.freeze_path = self.frozen / "freeze.json"
        self.certificate_path = self.assessment / "certificate.json"
        models = []
        for candidate in CANDIDATE_IDS:
            state_path = self.frozen / "models" / f"{candidate}.json"
            write_json(state_path, {"schema": "test.synthetic_model_identity",
                                    "candidate_id": candidate})
            prediction_path = self.frozen / "predictions" / f"{candidate}.json"
            write_json(prediction_path,
                       prediction_table_from_mapping(self.predictions[candidate], candidate))
            models.append({
                "candidate_id": candidate,
                "bank_id": None if candidate == "raw" else "test-shared-bank",
                "state_sha256": sha256_file(state_path),
                "prediction_table_path": f"predictions/{candidate}.json",
                "prediction_table_sha256": sha256_file(prediction_path),
            })
        self.freeze_record = {
            "schema": "test.synthetic_verified_freeze_boundary",
            "run_id": TEST_RUN_ID, "contract_sha256": CONTRACT_SHA256, "models": models,
        }
        write_json(self.freeze_path, self.freeze_record)
        candidates = []
        for candidate, q, upper, upper_display, status in (
            ("raw", Fraction(1, 10), Fraction(1, 8), "0.125000000000000000", "unresolved"),
            ("kahm_mem", Fraction(1, 20), Fraction(1, 10), "0.100000000000000000", "requirement_met"),
            ("kahm_local", Fraction(1, 4), None, None, "certificate_unavailable"),
        ):
            failure = ({"code": "series_work_limit", "message": "synthetic retained unavailable record"}
                       if upper is None else None)
            bound_path = self.assessment / "bounds" / f"{candidate}.json"
            write_json(bound_path, {"schema": "test.synthetic_issued_bound_boundary",
                                    "candidate_id": candidate, "status": status,
                                    "empirical_mean": ratio(q),
                                    "endpoint": None if upper is None else ratio(upper),
                                    "endpoint_decimal_18": upper_display, "failure": failure})
            frozen_row = models[len(candidates)]
            candidates.append({
                "candidate_id": candidate, "bank_id": frozen_row["bank_id"],
                "frozen_state_sha256": frozen_row["state_sha256"],
                "empirical_mse": ratio(q), "status": status,
                "upper_endpoint": None if upper is None else ratio(upper),
                "upper_endpoint_decimal_18": upper_display, "failure": failure,
                "bound_path": f"bounds/{candidate}.json", "bound_sha256": sha256_file(bound_path),
            })
        self.certificate_record = {
            "schema": "test.synthetic_verified_certificate_boundary",
            "contract_sha256": CONTRACT_SHA256,
            "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
            "freeze_record_sha256": sha256_file(self.freeze_path),
            "candidates": candidates,
        }
        write_json(self.certificate_path, self.certificate_record)
        self.access_events = []
        patcher = mock.patch.object(workflow, "verify_certificate_bundle",
                                    side_effect=self.verified_prerequisite)
        self.prerequisite = patcher.start()
        self.addCleanup(patcher.stop)
        guard = prohibit_sampling_fitting_and_endpoint_search()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def verified_prerequisite(self, certificate_path, *, dataset_bundle, freeze_path):
        self.assertEqual(Path(certificate_path), self.certificate_path)
        self.assertEqual(Path(dataset_bundle), self.data)
        self.assertEqual(Path(freeze_path), self.freeze_path)
        self.access_events.append("prior_certificate_verified")
        return {
            "status": "passed", "certificate_sha256": sha256_file(self.certificate_path),
            "freeze_record_sha256": sha256_file(self.freeze_path),
        }

    def create_audit(self, output=None):
        return workflow.audit_certificate_family(
            self.certificate_path, dataset_bundle=self.data, freeze_path=self.freeze_path,
            output=self.audit if output is None else output)

    def verify_audit(self, root=None):
        return workflow.verify_audit_bundle(
            (self.audit if root is None else root) / "audit.json",
            certificate_path=self.certificate_path, dataset_bundle=self.data,
            freeze_path=self.freeze_path)

    def copy_audit(self, name):
        destination = self.root / name
        shutil.copytree(self.audit, destination)
        return destination

    def test_complete_audit_retains_miss_exact_boundary_and_unavailable_endpoint(self):
        prior = {name: file_identities(path) for name, path in (
            ("data", self.data), ("frozen", self.frozen), ("certificate", self.assessment))}
        creation = self.create_audit()
        self.assertEqual(creation["status"], "audit_complete")
        self.assertEqual(creation["verification"]["status"], "passed")
        record = read_json(self.audit / "audit.json")
        self.assertEqual(record["candidate_ids"], list(CANDIDATE_IDS))
        self.assertEqual([row["candidate_id"] for row in record["candidates"]], list(CANDIDATE_IDS))
        self.assertEqual(record["freeze_record_sha256"], sha256_file(self.freeze_path))
        self.assertEqual(record["certificate_sha256"], sha256_file(self.certificate_path))
        for row, certificate, frozen in zip(record["candidates"], self.certificate_record["candidates"],
                                             self.freeze_record["models"], strict=True):
            expected = self.expected_risks[row["candidate_id"]]
            self.assertEqual(fraction(row["exact_population_mse"]), expected)
            self.assertEqual(fraction(row["fitted_excess_over_bayes"]), expected - Fraction(9, 160))
            self.assertEqual(row["calibration_empirical_mse"], certificate["empirical_mse"])
            self.assertEqual(row["certificate_status"], certificate["status"])
            self.assertEqual(row["issued_upper_endpoint"], certificate["upper_endpoint"])
            self.assertEqual(row["issued_upper_endpoint_decimal_18"], certificate["upper_endpoint_decimal_18"])
            self.assertEqual(row["frozen_state_sha256"], frozen["state_sha256"])
            self.assertEqual(row["prediction_table_sha256"], frozen["prediction_table_sha256"])
            self.assertEqual(fraction(row["empirical_minus_population_risk"]),
                             fraction(certificate["empirical_mse"]) - expected)
            self.assertEqual(row["population_requirement_met"], expected <= Fraction(1, 10))
            population = read_json(self.audit / row["population_audit_path"])
            self.assertEqual((len(population["histories"]), len(population["triples"])), (16, 64))
        raw, mem, local = record["candidates"]
        self.assertEqual(raw["realized_coverage"], "coverage_miss")
        self.assertEqual(fraction(raw["upper_minus_population_risk"]), Fraction(-1, 32))
        self.assertFalse(raw["population_requirement_met"])
        self.assertEqual(mem["realized_coverage"], "covered")
        self.assertEqual(fraction(mem["upper_minus_population_risk"]), 0)
        self.assertTrue(mem["population_requirement_met"])
        self.assertEqual(local["realized_coverage"], "endpoint_unavailable")
        self.assertIsNone(local["upper_minus_population_risk"])
        self.assertIsNone(local["upper_minus_empirical_risk"])
        self.assertNotEqual(self.expected_risks["kahm_local"], Fraction(5, 32) + Fraction(2, 5) ** 2)
        self.assertEqual(record["summary"], {
            "candidate_count": 3, "available_endpoint_count": 2,
            "covered_endpoint_count": 1, "coverage_miss_count": 1,
            "unavailable_endpoint_count": 1, "all_three_endpoints_available": False,
            "all_available_endpoints_cover": False, "all_three_covered": False,
            "population_requirement_met_count": 1, "all_three_population_requirements_met": False,
            "coverage_outcome": "coverage_miss_retained",
        })
        self.assertFalse(record["scope"]["winner_selected"])
        self.assertFalse(record["scope"]["general_representation_superiority_claimed"])
        self.assertTrue(record["scope"]["all_outcomes_retained"])
        self.assertEqual(prior, {name: file_identities(path) for name, path in (
            ("data", self.data), ("frozen", self.frozen), ("certificate", self.assessment))})

    def test_plan_and_source_snapshot_are_committed_before_any_exact_risk(self):
        real_build = workflow.build_population_audit

        def after_commit(predictions):
            self.assertIn("prior_certificate_verified", self.access_events)
            self.assertTrue((self.audit / "audit_plan.json").is_file())
            plan = read_json(self.audit / "audit_plan.json")
            self.assertEqual(plan["status"],
                             "committed_after_certificate_verification_before_this_population_audit")
            self.assertEqual(plan["candidate_ids"], list(CANDIDATE_IDS))
            self.assertEqual(plan["certificate_sha256"], sha256_file(self.certificate_path))
            self.assertEqual(plan["freeze_record_sha256"], sha256_file(self.freeze_path))
            self.assertEqual(plan["source_roster"], list(workflow.AUDIT_SOURCE_FILES))
            for name, digest in plan["source_sha256"].items():
                self.assertEqual(sha256_file(self.audit / "sources" / name), digest)
            self.access_events.append("exact_population_risk")
            return real_build(predictions)

        with mock.patch.object(workflow, "build_population_audit", side_effect=after_commit) as build:
            self.create_audit()
        self.assertEqual(build.call_count, 3)
        self.assertEqual(self.access_events[:4],
                         ["prior_certificate_verified"] + ["exact_population_risk"] * 3)

    def test_verification_is_read_only_and_recomputes_all_exact_contributions(self):
        self.create_audit()
        before = file_identities(self.root)
        with mock.patch.object(workflow, "audit_certificate_family",
                               side_effect=AssertionError("verification attempted audit creation")):
            verification = self.verify_audit()
        self.assertEqual(verification["status"], "passed")
        for key in ("exact_population_risks_recomputed", "all_64_triples_per_candidate_checked",
                    "independent_conditional_decomposition_checked", "preserved_issued_endpoints",
                    "source_identity_matches", "runtime_metadata_matches"):
            self.assertIs(verification[key], True)
        for key in ("calibration_resampled", "models_refitted", "root_search_repeated",
                    "model_selection_performed", "issued_certificate_modified"):
            self.assertIs(verification[key], False)
        self.assertEqual(before, file_identities(self.root))

    def test_failed_prerequisite_precedes_input_reads_risk_and_output_creation(self):
        self.prerequisite.side_effect = ValueError("test prerequisite rejected")
        with (mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read before prerequisite")),
              mock.patch.object(Path, "read_text", side_effect=AssertionError("read before prerequisite")),
              mock.patch.object(workflow, "build_population_audit", side_effect=AssertionError("early risk"))):
            with self.assertRaisesRegex(ValueError, "test prerequisite rejected"):
                self.create_audit()
            with self.assertRaisesRegex(ValueError, "test prerequisite rejected"):
                self.verify_audit()
        self.assertFalse(self.audit.exists())
        self.assertFalse(list(self.root.glob(".kahm-audit-*")))

    def test_nonpassed_prerequisite_is_rejected_without_reading_artifacts(self):
        self.prerequisite.side_effect = None
        self.prerequisite.return_value = {"status": "unverified"}
        with (mock.patch.object(Path, "read_bytes", side_effect=AssertionError("unverified input read")),
              mock.patch.object(Path, "read_text", side_effect=AssertionError("unverified input read"))):
            with self.assertRaisesRegex(ValueError, "fully verified"):
                self.create_audit()
        self.assertFalse(self.audit.exists())

    def test_changed_input_bytes_after_prerequisite_are_rejected_before_risk(self):
        paths = (self.certificate_path, self.freeze_path, self.frozen / "predictions/raw.json")
        for index, path in enumerate(paths):
            with self.subTest(path=path.name):
                original = path.read_bytes()

                def change_after_verification(*args, **kwargs):
                    checked = self.verified_prerequisite(*args, **kwargs)
                    path.write_bytes(original + b" \n")
                    return checked

                self.prerequisite.side_effect = change_after_verification
                output = self.root / f"changed-input-{index}"
                try:
                    with mock.patch.object(workflow, "build_population_audit",
                                           side_effect=AssertionError("risk used changed input")):
                        with self.assertRaisesRegex(ValueError, "changed after prerequisite verification"):
                            self.create_audit(output)
                    self.assertFalse(output.exists())
                finally:
                    path.write_bytes(original)

    def test_existing_and_protected_outputs_are_rejected_without_overwrite(self):
        existing_directory = self.root / "existing-directory"
        existing_directory.mkdir()
        existing_file = self.root / "existing-file"
        existing_file.write_text("retain this evidence", encoding="utf-8")
        dangling = self.root / "dangling-output"
        dangling.symlink_to(self.root / "absent-target")
        for output in (existing_directory, existing_file, dangling):
            with self.subTest(output=output.name), self.assertRaises(FileExistsError):
                self.create_audit(output)
        link = self.root / "link-to-frozen"
        link.symlink_to(self.frozen, target_is_directory=True)
        for output in (self.data / "audit", self.frozen / "audit",
                       self.assessment / "audit", link / "audit"):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                self.create_audit(output)
            self.assertFalse(output.exists())
        self.prerequisite.assert_not_called()
        self.assertEqual(existing_file.read_text(encoding="utf-8"), "retain this evidence")
        self.assertTrue(dangling.is_symlink())

    def test_rehashed_comparison_tampering_is_rejected(self):
        self.create_audit()
        changes = (
            ("risk", lambda r: r["candidates"][0].__setitem__("exact_population_mse", ratio(0))),
            ("coverage", lambda r: r["candidates"][0].__setitem__("realized_coverage", "covered")),
            ("endpoint", lambda r: r["candidates"][0].__setitem__("issued_upper_endpoint", ratio(Fraction(1, 4)))),
            ("endpoint-display", lambda r: r["candidates"][0].__setitem__("issued_upper_endpoint_decimal_18", "0.999000000000000000")),
            ("status", lambda r: r["candidates"][0].__setitem__("certificate_status", "requirement_met")),
            ("q", lambda r: r["candidates"][0].__setitem__("calibration_empirical_mse", ratio(0))),
            ("state", lambda r: r["candidates"][0].__setitem__("frozen_state_sha256", "0" * 64)),
            ("table", lambda r: r["candidates"][0].__setitem__("prediction_table_sha256", "0" * 64)),
            ("slack", lambda r: r["candidates"][0].__setitem__("upper_minus_population_risk", ratio(0))),
            ("deviation", lambda r: r["candidates"][0].__setitem__("empirical_minus_population_risk", ratio(0))),
            ("margin", lambda r: r["candidates"][0].__setitem__("upper_minus_empirical_risk", ratio(0))),
            ("excess", lambda r: r["candidates"][0].__setitem__("fitted_excess_over_bayes", ratio(0))),
            ("requirement", lambda r: r["candidates"][0].__setitem__("population_requirement_met", True)),
            ("boolean-type", lambda r: r["candidates"][1].__setitem__("population_requirement_met", 1)),
            ("missing-candidate", lambda r: r["candidates"].pop()),
            ("candidate-order", lambda r: r["candidates"].reverse()),
        )
        for name, change in changes:
            with self.subTest(change=name):
                copied = self.copy_audit(f"comparison-{name}")
                record = read_json(copied / "audit.json")
                change(record)
                refresh_audit_envelope(copied, record)
                with self.assertRaises(ValueError):
                    self.verify_audit(copied)

    def test_rehashed_population_contributions_are_independently_rejected(self):
        self.create_audit()
        changes = (
            ("risk", lambda r: r.__setitem__("exact_population_mse", ratio(0))),
            ("loss", lambda r: r["triples"][0].__setitem__("weighted_squared_loss", ratio(0))),
            ("missing-triple", lambda r: r["triples"].pop()),
            ("duplicate-history", lambda r: r["histories"].__setitem__(1, deepcopy(r["histories"][0]))),
            ("prediction", lambda r: r["histories"][0].__setitem__("prediction_hex", 0.25.hex())),
            ("boolean-type", lambda r: r["checks"].__setitem__("oracle_exact_equality", 1)),
        )
        for name, change in changes:
            with self.subTest(change=name):
                copied = self.copy_audit(f"population-{name}")
                path = copied / "population/raw.json"
                record = read_json(path)
                change(record)
                write_json(path, record)
                refresh_audit_envelope(copied)
                with self.assertRaises(ValueError):
                    self.verify_audit(copied)

    def test_rehashed_summary_scope_contract_and_mutation_claims_are_rejected(self):
        self.create_audit()
        changes = (
            ("miss-count", lambda r: r["summary"].__setitem__("coverage_miss_count", 0)),
            ("outcome", lambda r: r["summary"].__setitem__("coverage_outcome", "all_three_covered")),
            ("all-cover", lambda r: r["summary"].__setitem__("all_available_endpoints_cover", True)),
            ("winner", lambda r: r["scope"].__setitem__("winner_selected", True)),
            ("bayes", lambda r: r["references"].__setitem__("bayes_mse", ratio(0))),
            ("contract", lambda r: r.__setitem__("contract_sha256", "0" * 64)),
            ("protocol", lambda r: r.__setitem__("calibration_protocol_sha256", "0" * 64)),
            ("family", lambda r: r["candidate_ids"].pop()),
            ("refit-type", lambda r: r.__setitem__("models_refitted", 0)),
            ("root-search", lambda r: r.__setitem__("root_search_repeated", True)),
        )
        for name, change in changes:
            with self.subTest(change=name):
                copied = self.copy_audit(f"summary-{name}")
                record = read_json(copied / "audit.json")
                change(record)
                refresh_audit_envelope(copied, record)
                with self.assertRaises(ValueError):
                    self.verify_audit(copied)

    def test_rehashed_source_snapshot_roster_and_runtime_cannot_replace_implementation(self):
        self.create_audit()
        for case in ("source-bytes", "missing-source", "extra-source", "runtime"):
            with self.subTest(case=case):
                copied = self.copy_audit(f"source-{case}")
                plan_path = copied / "audit_plan.json"
                plan = read_json(plan_path)
                name = workflow.AUDIT_SOURCE_FILES[0]
                if case == "source-bytes":
                    path = copied / "sources" / name
                    path.write_bytes(path.read_bytes() + b"\n# altered archived implementation\n")
                    plan["source_sha256"][name] = sha256_file(path)
                elif case == "missing-source":
                    (copied / "sources" / name).unlink()
                    plan["source_roster"].remove(name)
                    del plan["source_sha256"][name]
                elif case == "extra-source":
                    path = copied / "sources" / "unapproved.py"
                    path.write_text("# unapproved audit dependency\n", encoding="utf-8")
                    plan["source_roster"].append(path.name)
                    plan["source_sha256"][path.name] = sha256_file(path)
                else:
                    plan["runtime"]["test_unapproved_runtime"] = True
                rehash_record(plan_path, plan)
                refresh_audit_envelope(copied)
                with mock.patch.object(workflow, "verify_population_audit",
                                       side_effect=AssertionError("risk read before source validation")):
                    with self.assertRaises(ValueError):
                        self.verify_audit(copied)

    def test_unavailable_endpoints_are_all_retained_without_a_coverage_claim(self):
        certificate = deepcopy(self.certificate_record)
        for row in certificate["candidates"]:
            row["status"] = "certificate_unavailable"
            row["upper_endpoint"] = row["upper_endpoint_decimal_18"] = None
            row["failure"] = {"code": "series_work_limit", "message": "synthetic retained failure"}
        write_json(self.certificate_path, certificate)
        self.create_audit()
        record = read_json(self.audit / "audit.json")
        self.assertEqual(len(record["candidates"]), 3)
        self.assertEqual(record["summary"]["available_endpoint_count"], 0)
        self.assertEqual(record["summary"]["unavailable_endpoint_count"], 3)
        self.assertEqual(record["summary"]["coverage_outcome"], "no_endpoints_available")
        self.assertIsNone(record["summary"]["all_available_endpoints_cover"])
        self.assertEqual(record["summary"]["population_requirement_met_count"], 1)
        for row in record["candidates"]:
            self.assertEqual(row["certificate_status"], "certificate_unavailable")
            self.assertEqual(row["realized_coverage"], "endpoint_unavailable")
            self.assertIsNone(row["issued_upper_endpoint"])
            self.assertEqual(fraction(row["exact_population_mse"]), self.expected_risks[row["candidate_id"]])

    def test_covered_endpoint_does_not_mean_population_tolerance_is_met(self):
        certificate = deepcopy(self.certificate_record)
        certificate["candidates"][0].update({
            "upper_endpoint": ratio(Fraction(1, 5)), "upper_endpoint_decimal_18": "0.200000000000000000"})
        certificate["candidates"][2].update({
            "status": "unresolved", "upper_endpoint": ratio(Fraction(1, 2)),
            "upper_endpoint_decimal_18": "0.500000000000000000", "failure": None})
        write_json(self.certificate_path, certificate)
        self.create_audit()
        record = read_json(self.audit / "audit.json")
        self.assertEqual(record["summary"]["coverage_outcome"], "all_three_covered")
        self.assertTrue(record["summary"]["all_three_covered"])
        self.assertFalse(record["summary"]["all_three_population_requirements_met"])
        self.assertEqual(record["summary"]["population_requirement_met_count"], 1)
        self.assertEqual(record["candidates"][0]["certificate_status"], "unresolved")
        self.assertFalse(record["candidates"][0]["population_requirement_met"])

    def test_partial_audit_failure_preserves_prior_evidence_and_completed_work(self):
        before = {name: file_identities(path) for name, path in (
            ("data", self.data), ("frozen", self.frozen), ("certificate", self.assessment))}
        real_build = workflow.build_population_audit
        call_count = 0

        def fail_second(predictions):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("test exact-audit interruption")
            return real_build(predictions)

        with mock.patch.object(workflow, "build_population_audit", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "test exact-audit interruption"):
                self.create_audit()
        self.assertEqual(call_count, 2)
        self.assertTrue((self.audit / "audit_plan.json").is_file())
        self.assertTrue((self.audit / "population/raw.json").is_file())
        self.assertFalse((self.audit / "audit.json").exists())
        failure = read_json(self.audit / "audit_failure.json")
        self.assertEqual(failure["status"], "audit_failed")
        self.assertEqual(failure["phase"], "exact_population_risk_calculation")
        with self.assertRaises(FileExistsError):
            self.create_audit()
        self.assertEqual(before, {name: file_identities(path) for name, path in (
            ("data", self.data), ("frozen", self.frozen), ("certificate", self.assessment))})

    def test_existing_audit_is_bound_to_specific_prerequisite_bytes(self):
        self.create_audit()
        original = self.certificate_path.read_bytes()
        self.certificate_path.write_bytes(original + b" \n")
        with mock.patch.object(workflow, "verify_population_audit",
                               side_effect=AssertionError("risk checked after prerequisite drift")):
            with self.assertRaisesRegex(ValueError, "audit plan differs"):
                self.verify_audit()

    def test_extra_or_symlink_artifacts_are_rejected_despite_refreshed_inventory(self):
        self.create_audit()
        for name in ("extra", "symlink"):
            with self.subTest(name=name):
                copied = self.copy_audit(f"inventory-{name}")
                if name == "extra":
                    write_json(copied / "unexpected.json", {"extra": True})
                else:
                    (copied / "alias.json").symlink_to(copied / "population/raw.json")
                refresh_audit_envelope(copied)
                with self.assertRaises(ValueError):
                    self.verify_audit(copied)

    def test_missing_family_or_wrong_table_identity_fails_before_risk(self):
        original_certificate = self.certificate_path.read_bytes()
        original_freeze = self.freeze_path.read_bytes()
        table_path = self.frozen / "predictions/raw.json"
        original_table = table_path.read_bytes()
        for case in ("missing-certificate-candidate", "frozen-order", "table-candidate"):
            with self.subTest(case=case):
                if case == "missing-certificate-candidate":
                    record = deepcopy(self.certificate_record)
                    record["candidates"].pop()
                    write_json(self.certificate_path, record)
                elif case == "frozen-order":
                    record = deepcopy(self.freeze_record)
                    record["models"].reverse()
                    write_json(self.freeze_path, record)
                else:
                    table = read_json(table_path)
                    table["candidate_id"] = "kahm_mem"
                    write_json(table_path, table)
                    record = deepcopy(self.freeze_record)
                    record["models"][0]["prediction_table_sha256"] = sha256_file(table_path)
                    write_json(self.freeze_path, record)
                try:
                    with mock.patch.object(workflow, "build_population_audit",
                                           side_effect=AssertionError("risk calculated for wrong family")):
                        with self.assertRaises(ValueError):
                            self.create_audit(self.root / case)
                    self.assertFalse((self.root / case).exists())
                finally:
                    self.certificate_path.write_bytes(original_certificate)
                    self.freeze_path.write_bytes(original_freeze)
                    table_path.write_bytes(original_table)

    def test_population_decimal_display_uses_exact_nearest_even_not_float_rounding(self):
        scale = 10 ** 18
        for value, expected in (
            (Fraction(1, 3), "0.333333333333333333"),
            (Fraction(1, 2 * scale), "0.000000000000000000"),
            (Fraction(3, 2 * scale), "0.000000000000000002"),
            (Fraction(5, 2 * scale), "0.000000000000000002"),
            (Fraction(-3, 2 * scale), "-0.000000000000000002"),
            (Fraction(-1, 2 * scale), "0.000000000000000000"),
            (Fraction(1), "1.000000000000000000"),
        ):
            with self.subTest(value=value):
                self.assertEqual(workflow.decimal_18(value), expected)
        for invalid in (0.1, 1, True):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                workflow.decimal_18(invalid)


if __name__ == "__main__":
    unittest.main()

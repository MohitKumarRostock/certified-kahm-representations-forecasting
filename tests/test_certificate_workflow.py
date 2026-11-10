"""Full certificate workflow on one portable, entirely separate test stream.

The fixture fits all 225 development configurations on 128/64 episodes, freezes
them, and materializes 4096 calibration episodes with its own seed and run ID.
It never copies or opens the primary data, seed, run identity, or frozen bundle.
"""

from collections import Counter
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kahm_demo import certificate_workflow as workflow
from kahm_demo import datasets, risk_bounds
from kahm_demo._util import canonical_json_bytes, read_json, sha256_file, write_json
from kahm_demo.calibration_protocol import CALIBRATION_PROTOCOL_SHA256
from kahm_demo.contract import CANDIDATE_IDS
from kahm_demo.encoder_workflow import fit_development_encoders
from kahm_demo.forecast_workflow import fit_development_forecasters
from kahm_demo.freeze import freeze_forecasters
from kahm_demo.law import OBSERVATIONS
from kahm_demo.oracle import parse_prediction_table


TEST_SEED = 61420637
TEST_RUN_ID = "test-step6-certificate-integration"


@contextmanager
def prohibit_refitting_and_oracle():
    targets = (
        "kahm_demo.oracle.exact_risk", "kahm_demo.oracle.risk_of_predictor",
        "kahm_demo.encoder_workflow.fit_development_encoders",
        "kahm_demo.encoder_workflow.fit_encoder_candidates",
        "kahm_demo.encoder.fit_encoder_candidates",
        "kahm_demo.forecast_workflow.fit_development_forecasters",
        "kahm_demo.forecast_workflow.fit_readout_features",
        "kahm_demo.forecast_workflow.fit_ridge_grid",
        "kahm_demo.readout_features.fit_readout_features",
        "kahm_demo.ridge.fit_ridge_grid", "kahm_demo.ridge.fit_ridge",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(mock.patch(target, side_effect=AssertionError(
                f"assessment called a prohibited fitting/oracle entry point: {target}")))
        yield


@contextmanager
def prohibit_resampling_and_materialization(*, root_search=False):
    targets = [
        "kahm_demo.datasets.sample_episodes", "kahm_demo.sampling.sample_episodes",
        "kahm_demo.datasets.materialize_calibration",
        "kahm_demo.certificate_workflow.materialize_calibration",
    ]
    if root_search:
        targets.extend(("kahm_demo.risk_bounds.certify_upper_bound",
                        "kahm_demo.certificate_workflow.certify_upper_bound"))
    with prohibit_refitting_and_oracle(), ExitStack() as stack:
        for target in targets:
            stack.enter_context(mock.patch(target, side_effect=AssertionError(
                f"read-only/archive operation called prohibited entry point: {target}")))
        yield


def fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def rehash_payload(path, record):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json(path, record)


def refresh_certificate_envelope(root, record=None):
    """Refresh outer file identities so negative tests reach semantic checks."""
    path = root / "certificate.json"
    record = read_json(path) if record is None else record
    record["files"] = [
        {"path": item.relative_to(root).as_posix(), "bytes": item.stat().st_size,
         "sha256": sha256_file(item)}
        for item in sorted(root.rglob("*")) if item.is_file() and item != path
    ]
    rehash_payload(path, record)


def file_identities(root):
    return {path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, sha256_file(path))
            for path in root.rglob("*") if path.is_file()}


class CertificateWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kahm-test-step6-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.data, cls.encoders, cls.models, cls.frozen, cls.assessment = [
            cls.root / name for name in ("data", "encoders", "models", "frozen", "assessment")
        ]
        real_sampler = datasets.sample_episodes

        def development_only(count, master_seed, role, run_id):
            if (master_seed != TEST_SEED or run_id != TEST_RUN_ID
                    or (role, count) not in (("training", 128), ("validation", 64))):
                raise AssertionError("fixture attempted an undeclared development stream")
            return real_sampler(count, master_seed, role, run_id)

        with mock.patch.object(datasets, "sample_episodes", side_effect=development_only):
            datasets.generate_bundle(cls.data, master_seed=TEST_SEED, run_id=TEST_RUN_ID,
                                     training_count=128, validation_count=64)
        with (mock.patch("kahm_demo.oracle.exact_risk", side_effect=AssertionError("oracle used in development")),
              mock.patch("kahm_demo.oracle.risk_of_predictor", side_effect=AssertionError("oracle used in development")),
              mock.patch.object(datasets, "sample_episodes", side_effect=AssertionError("unexpected development sampling"))):
            fit_development_encoders(cls.data, cls.encoders)
            fit_development_forecasters(cls.data, cls.encoders, cls.models)
        with prohibit_resampling_and_materialization(root_search=True):
            cls.freeze_creation = freeze_forecasters(cls.data, cls.encoders, cls.models, cls.frozen)
        cls.freeze_path = cls.frozen / "freeze.json"
        cls.freeze_record = read_json(cls.freeze_path)
        cls.access_events = []
        cls.sample_calls = []
        cls.calibration_dataset = None
        real_materialize, real_load = workflow.materialize_calibration, workflow.load_calibration

        def committed_plan_exists(phase):
            plan_path = cls.assessment / "assessment_plan.json"
            if not plan_path.is_file():
                raise AssertionError("outcome access occurred before the assessment plan was committed")
            plan = read_json(plan_path)
            if (plan["candidate_ids"] != list(CANDIDATE_IDS)
                    or plan["calibration_count"] != 4096
                    or plan["freeze_record_sha256"] != sha256_file(cls.freeze_path)
                    or plan["calibration_protocol_sha256"] != CALIBRATION_PROTOCOL_SHA256):
                raise AssertionError("committed assessment plan has incorrect identities")
            if set(plan["source_roster"]) != set(workflow.CERTIFIER_SOURCE_FILES):
                raise AssertionError("certifier source roster was not committed")
            for name, digest in plan["source_sha256"].items():
                source = cls.assessment / "sources" / name
                if not source.is_file() or sha256_file(source) != digest:
                    raise AssertionError("certifier source bytes were not committed before outcome access")
            cls.access_events.append(phase)

        def calibration_only(count, master_seed, role, run_id):
            if (count, master_seed, role, run_id) != (4096, TEST_SEED, "calibration", TEST_RUN_ID):
                raise AssertionError("fixture attempted an undeclared calibration stream")
            committed_plan_exists("sample")
            cls.sample_calls.append((count, master_seed, role, run_id))
            return real_sampler(count, master_seed, role, run_id)

        def materialize(bundle, freeze_path):
            committed_plan_exists("materialize")
            if (cls.data / "calibration").exists():
                raise AssertionError("first assessment found unexpected calibration outcomes")
            return real_materialize(bundle, freeze_path)

        def load(bundle, freeze_path):
            committed_plan_exists("load")
            result = real_load(bundle, freeze_path)
            if cls.calibration_dataset is None:
                cls.calibration_dataset = result
            return result

        with (prohibit_refitting_and_oracle(),
              mock.patch.object(datasets, "sample_episodes", side_effect=calibration_only),
              mock.patch.object(workflow, "materialize_calibration", side_effect=materialize),
              mock.patch.object(workflow, "load_calibration", side_effect=load)):
            cls.creation = workflow.certify_frozen_family(cls.data, cls.freeze_path, cls.assessment)
        cls.record = read_json(cls.assessment / "certificate.json")
        cls.predictions = {
            model["candidate_id"]: parse_prediction_table(read_json(
                cls.frozen / model["prediction_table_path"]))
            for model in cls.freeze_record["models"]
        }
        counts = Counter((episode.previous_state, episode.current_state, episode.next_state)
                         for episode in cls.calibration_dataset.episodes)
        cls.independent_means = {}
        for candidate in CANDIDATE_IDS:
            total = Fraction(0)
            for (a, b, c), occurrences in counts.items():
                predicted = Fraction(*cls.predictions[candidate][(a, b)].as_integer_ratio())
                error = OBSERVATIONS[c] - predicted
                total += occurrences * error * error
            cls.independent_means[candidate] = total / 4096

    def temporary_directory(self):
        temporary = tempfile.TemporaryDirectory(prefix="kahm-test-certificate-case-")
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def copy_assessment(self):
        destination = self.temporary_directory() / "assessment"
        shutil.copytree(self.assessment, destination)
        return destination

    def verify(self, root):
        with prohibit_resampling_and_materialization(root_search=True):
            return workflow.verify_certificate_bundle(root / "certificate.json",
                                                      dataset_bundle=self.data,
                                                      freeze_path=self.freeze_path)

    def test_real_assessment_retains_all_candidates_and_independent_exact_means(self):
        self.assertEqual(self.creation["status"], "assessment_complete")
        self.assertEqual(self.creation["calibration_action"], "materialize_reserved_stream")
        self.assertEqual(self.record["schema"], workflow.CERTIFICATE_SCHEMA)
        self.assertEqual([row["candidate_id"] for row in self.record["candidates"]], list(CANDIDATE_IDS))
        self.assertEqual(self.record["calibration_identity"]["episode_count"], 4096)
        self.assertEqual(self.record["calibration_identity"]["run_id"], TEST_RUN_ID)
        self.assertEqual(self.record["freeze_record_sha256"], sha256_file(self.freeze_path))
        self.assertFalse(self.record["model_population_risk_evaluated"])
        self.assertFalse(self.record["models_refitted"])
        self.assertEqual(self.record["guarantee"]["per_candidate_failure_probability"],
                         {"numerator": 1, "denominator": 60})
        self.assertEqual(self.record["guarantee"]["total_failure_probability"],
                         {"numerator": 1, "denominator": 20})
        self.assertEqual(self.record["guarantee"]["primary_mse_tolerance"],
                         {"numerator": 1, "denominator": 10})
        for row in self.record["candidates"]:
            candidate = row["candidate_id"]
            self.assertEqual(fraction(row["empirical_mse"]), self.independent_means[candidate])
            self.assertEqual(fraction(self.record["ledger_summary"]["empirical_risks"][candidate]),
                             self.independent_means[candidate])
            self.assertIn(row["status"], ("requirement_met", "unresolved"))
            self.assertGreaterEqual(fraction(row["upper_endpoint"]), self.independent_means[candidate])

    def test_plan_and_sources_were_committed_before_materialization_and_loading(self):
        self.assertEqual(self.access_events[:2], ["materialize", "sample"])
        self.assertIn("load", self.access_events[2:])
        self.assertEqual(self.sample_calls, [(4096, TEST_SEED, "calibration", TEST_RUN_ID)])
        plan = read_json(self.assessment / "assessment_plan.json")
        self.assertEqual(plan["status"], "fixed_before_this_assessment_calibration_access")
        self.assertEqual(plan["calibration_action"], "materialize_reserved_stream")
        self.assertFalse(plan["protocol_changes_after_outcomes_permitted"])
        self.assertFalse(plan["model_selection_or_refitting_permitted"])

    def test_verification_is_read_only_without_sampling_refitting_or_root_search(self):
        before = {name: file_identities(root) for name, root in
                  (("assessment", self.assessment), ("data", self.data), ("frozen", self.frozen))}
        with (mock.patch.object(Path, "write_bytes", side_effect=AssertionError("verification wrote bytes")),
              mock.patch.object(Path, "write_text", side_effect=AssertionError("verification wrote text"))):
            result = self.verify(self.assessment)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["all_episode_losses_recomputed_exactly"])
        self.assertEqual(result["candidate_ids"], list(CANDIDATE_IDS))
        for field in ("root_search_repeated", "calibration_resampled", "models_refitted",
                      "model_population_risk_evaluated"):
            self.assertIs(result[field], False)
        after = {name: file_identities(root) for name, root in
                 (("assessment", self.assessment), ("data", self.data), ("frozen", self.frozen))}
        self.assertEqual(before, after)

    def test_failed_frozen_preflight_precedes_outcome_derived_artifact_reads(self):
        original_open = Path.open
        assessment = self.assessment.resolve()

        def guard(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if (resolved in {assessment / "certificate.json", assessment / "loss_ledger.jsonl"}
                    or resolved.is_relative_to(assessment / "bounds")):
                raise AssertionError("outcome-derived certificate bytes were read before frozen preflight")
            return original_open(path, *args, **kwargs)

        with (mock.patch.object(Path, "open", guard),
              mock.patch.object(workflow, "_frozen_inputs",
                                side_effect=ValueError("test-only invalid frozen preflight")) as preflight):
            with self.assertRaisesRegex(ValueError, "invalid frozen preflight"):
                workflow.verify_certificate_bundle(self.assessment / "certificate.json",
                                                   dataset_bundle=self.data, freeze_path=self.freeze_path)
        preflight.assert_called_once_with(self.data, self.freeze_path)

    def test_archived_calibration_reuse_preserves_endpoints_without_sampling(self):
        output = self.temporary_directory() / "reused-assessment"
        calibration_before = file_identities(self.data / "calibration")
        with prohibit_resampling_and_materialization():
            result = workflow.certify_frozen_family(self.data, self.freeze_path, output)
        self.assertEqual(result["calibration_action"], "reuse_matching_archived_calibration")
        for old, new in zip(self.record["candidates"], result["candidates"], strict=True):
            for field in ("candidate_id", "frozen_state_sha256", "empirical_mse", "status",
                          "upper_endpoint", "upper_endpoint_decimal_18"):
                self.assertEqual(old[field], new[field])
        self.assertEqual(calibration_before, file_identities(self.data / "calibration"))

    def test_existing_or_internal_outputs_are_rejected_before_model_access(self):
        before = sha256_file(self.assessment / "certificate.json")
        with mock.patch.object(workflow, "_frozen_inputs", side_effect=AssertionError("unexpected preflight")):
            with self.assertRaises(FileExistsError):
                workflow.certify_frozen_family(self.data, self.freeze_path, self.assessment)
            for output in (self.data / "forbidden-assessment", self.frozen / "forbidden-assessment"):
                with self.subTest(output=output):
                    with self.assertRaises(ValueError):
                        workflow.certify_frozen_family(self.data, self.freeze_path, output)
                    self.assertFalse(output.exists())
            alias = self.temporary_directory() / "frozen-alias"
            alias.symlink_to(self.frozen, target_is_directory=True)
            with self.assertRaises(ValueError):
                workflow.certify_frozen_family(self.data, self.freeze_path, alias / "forbidden-assessment")
        self.assertEqual(before, sha256_file(self.assessment / "certificate.json"))

    def test_missing_candidate_or_changed_frozen_binding_is_rejected_after_rehash(self):
        for mutate in (
            lambda record: record["candidates"].pop(),
            lambda record: record["candidates"][0].update(frozen_state_sha256="0" * 64),
            lambda record: record.update(freeze_record_sha256="0" * 64),
        ):
            root = self.copy_assessment()
            record = read_json(root / "certificate.json")
            mutate(record)
            refresh_certificate_envelope(root, record)
            with self.assertRaises(ValueError):
                self.verify(root)

    def test_changed_confidence_allocation_is_rejected_after_rehash(self):
        for location in ("plan", "guarantee"):
            with self.subTest(location=location):
                root = self.copy_assessment()
                record = read_json(root / "certificate.json")
                if location == "plan":
                    path = root / "assessment_plan.json"
                    plan = read_json(path)
                    plan["per_candidate_failure_probability"] = {"numerator": 1, "denominator": 20}
                    rehash_payload(path, plan)
                    record["assessment_plan_sha256"] = sha256_file(path)
                else:
                    record["guarantee"]["per_candidate_failure_probability"] = {"numerator": 1, "denominator": 20}
                refresh_certificate_envelope(root, record)
                with self.assertRaises(ValueError):
                    self.verify(root)

    def test_source_roster_and_rehashed_source_tampering_are_rejected(self):
        for change_bytes in (False, True):
            with self.subTest(change_bytes=change_bytes):
                root = self.copy_assessment()
                path = root / "assessment_plan.json"
                plan = read_json(path)
                name = plan["source_roster"][0]
                if change_bytes:
                    source = root / "sources" / name
                    source.write_bytes(source.read_bytes() + b"\n# test-only altered source archive\n")
                    plan["source_sha256"][name] = sha256_file(source)
                else:
                    plan["source_roster"].remove(name)
                    del plan["source_sha256"][name]
                    (root / "sources" / name).unlink()
                rehash_payload(path, plan)
                record = read_json(root / "certificate.json")
                record["assessment_plan_sha256"] = sha256_file(path)
                refresh_certificate_envelope(root, record)
                with self.assertRaises(ValueError):
                    self.verify(root)

    def test_rehashed_ledger_tampering_is_rejected_by_occurrence_recomputation(self):
        root = self.copy_assessment()
        path = root / "loss_ledger.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        import json
        row = json.loads(lines[0])
        loss = row["squared_losses"]["raw"]
        loss["numerator"] = str(int(loss["numerator"]) + 1)
        lines[0] = json.dumps(row, separators=(",", ":")) + "\n"
        path.write_text("".join(lines))
        record = read_json(root / "certificate.json")
        record["ledger_sha256"] = sha256_file(path)
        refresh_certificate_envelope(root, record)
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_rehashed_endpoint_tampering_is_rejected_by_direct_numerical_check(self):
        root = self.copy_assessment()
        record = read_json(root / "certificate.json")
        candidate = record["candidates"][0]
        path = root / candidate["bound_path"]
        bound = read_json(path)
        bound["endpoint"] = {"numerator": "0", "denominator": "1"}
        bound["endpoint_decimal_18"] = "0.000000000000000000"
        write_json(path, bound)
        candidate["bound_sha256"] = sha256_file(path)
        candidate["upper_endpoint"] = bound["endpoint"]
        candidate["upper_endpoint_decimal_18"] = bound["endpoint_decimal_18"]
        refresh_certificate_envelope(root, record)
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_rehashed_family_work_must_match_the_recorded_candidate_chain(self):
        root = self.copy_assessment()
        record = read_json(root / "certificate.json")
        record["family_work"]["series_terms"] += 1
        refresh_certificate_envelope(root, record)
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_integer_budget_failure_retains_all_three_unavailable_candidates(self):
        output = self.temporary_directory() / "unavailable-assessment"

        def exceed_integer_budget(q, p, precision_bits, budget):
            # A real fixed-cap violation records the observed 32769-bit size and
            # makes the shared budget sticky. No protocol/source file is changed.
            budget.check_int(1 << budget.maximum_integer_bits)
            raise AssertionError("the declared integer budget did not fail")

        with (prohibit_resampling_and_materialization(),
              mock.patch.object(risk_bounds, "kl_interval", side_effect=exceed_integer_budget)):
            result = workflow.certify_frozen_family(self.data, self.freeze_path, output)
        self.assertEqual([row["candidate_id"] for row in result["candidates"]], list(CANDIDATE_IDS))
        self.assertFalse(result["all_three_requirements_met"])
        self.assertFalse(result["any_kahm_requirement_met"])
        for row in result["candidates"]:
            self.assertEqual(row["status"], "certificate_unavailable")
            self.assertIsNone(row["upper_endpoint"])
            self.assertIsNone(row["upper_endpoint_decimal_18"])
            self.assertEqual(row["failure"]["code"], "integer_bit_limit")
            self.assertEqual(fraction(row["empirical_mse"]), self.independent_means[row["candidate_id"]])
        record = read_json(output / "certificate.json")
        self.assertEqual(record["family_work"]["maximum_integer_bits"], 32768)
        self.assertEqual(record["family_work"]["maximum_integer_bits_seen"], 32769)
        checked = self.verify(output)
        self.assertEqual(checked["status"], "passed")
        self.assertTrue(all(not row["endpoint_verified"] for row in checked["candidates"]))


if __name__ == "__main__":
    unittest.main()

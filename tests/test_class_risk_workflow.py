"""Real finite-support integration and adversarial evidence-binding checks.

The shared fixture actually samples calibration, solves the small class
problems, certifies both statistical routes, and computes population audits.
Only the interruption and numerical-unavailability tests inject failures.
Historical development records are read without replaying their old runtime.
"""

from fractions import Fraction as F
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kahm_demo import class_bounds, class_oracle, class_risk_workflow as workflow, lipschitz_class
from kahm_demo._util import read_json, sha256_file, write_json
from kahm_demo.verified_arithmetic import WorkBudget


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE = PROJECT / "frozen/primary/freeze.json"


def fraction(record):
    return F(int(record["numerator"]), int(record["denominator"]))


class ClassRiskWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        cls.fixture = cls.base / "complete"
        cls.result = workflow.run_class_risk(
            SOURCE_FREEZE, cls.fixture, sample_sizes=(32,), block_sizes=(2, 8),
            kappas=(F(2),), repetitions=1, sweeps=16,
            master_seed=20261119, run_id="class-risk-integration-tests")

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_case.cleanup)
        self.case = Path(self.temporary_case.name)

    def copy_run(self, name="modified"):
        destination = self.case / name
        shutil.copytree(self.fixture, destination)
        return destination

    def certificates(self, root):
        return [(root / row["path"], read_json(root / row["path"]))
                for row in read_json(root / "certificate_index.json")]

    def rehash_certificates(self, root, *, update_plan=False, update_calibration=False):
        index = read_json(root / "certificate_index.json")
        for row in index:
            path = root / row["path"]
            record = read_json(path)
            if update_plan:
                record["plan_sha256"] = sha256_file(root / "class_risk_plan.json")
            if update_calibration:
                record["calibration_sha256"] = sha256_file(root / f"calibration/rep_{record['repetition']}.json")
            write_json(path, record)
            row["sha256"] = sha256_file(path)
        write_json(root / "certificate_index.json", index)

    def rebind_changed_plan(self, root, plan):
        write_json(root / "class_risk_plan.json", plan)
        self.rehash_certificates(root, update_plan=True)

    def rebind_changed_source_freeze(self, root, freeze, plan):
        freeze["record_sha256"] = workflow._digest({k: v for k, v in freeze.items() if k != "record_sha256"})
        write_json(root / "inputs/source_freeze.json", freeze)
        digest = sha256_file(root / "inputs/source_freeze.json")
        plan["inputs"]["source_freeze.json"] = digest
        plan["source_freeze_sha256"] = digest
        self.rebind_changed_plan(root, plan)

    def test_complete_real_run_retains_every_endpoint_and_exact_joint_allocation(self):
        plan = read_json(self.fixture / "class_risk_plan.json")
        records = [record for _, record in self.certificates(self.fixture)]
        self.assertEqual(len(records), 5)
        endpoints = []
        for record in records:
            endpoints.extend([record["fitted_upper_endpoint"], record["full_empirical_endpoint"]])
            for block in record["blocks"]:
                endpoints.extend([block["block_endpoint"], block["empirical_endpoint"]])
                self.assertEqual(block["block_endpoint"]["n"], 32 // block["block_size"])
                self.assertEqual(block["empirical_endpoint"]["n"], 32)
        self.assertEqual(len(endpoints), 30)
        self.assertEqual(plan["endpoint_count"], len(endpoints))
        self.assertTrue(all(fraction(endpoint["alpha"]) == F(1, 600) for endpoint in endpoints))
        self.assertEqual(sum((fraction(endpoint["alpha"]) for endpoint in endpoints), F(0)), F(1, 20))
        self.assertEqual(self.result["status"], "verified")
        self.assertEqual(self.result["unavailable_endpoint_count"], 0)
        self.assertTrue((self.fixture / "report.md").is_file())
        self.assertTrue((self.fixture / "summary.json").is_file())
        self.assertFalse((self.fixture / "failure.json").exists())

    def test_mem_upper_remains_a_fitted_bound_when_class_membership_fails(self):
        rows = {row["representation"]: row for row in self.result["assessments"]}
        mem = rows["kahm_mem"]
        self.assertFalse(mem["admissible"])
        self.assertIsNotNone(mem["fitted_upper"])
        self.assertIsNone(mem["class_upper"])
        self.assertNotEqual(mem["status"], "requirement_met")
        self.assertGreaterEqual(fraction(mem["optimal_risk_lower"]), F(29513, 250000))
        for name in ("raw", "kahm_local"):
            self.assertTrue(rows[name]["admissible"])
            self.assertEqual(rows[name]["class_upper"], rows[name]["fitted_upper"])
            self.assertEqual(fraction(rows[name]["optimal_risk_lower"]), F(9, 160))
            self.assertEqual(fraction(rows[name]["optimal_risk_upper"]), F(9, 160))

    def test_short_history_hierarchy_has_its_own_population_reference(self):
        short = next(row for row in self.result["assessments"] if row["representation"] == "short_history")
        self.assertEqual(fraction(short["optimal_risk_lower"]), F(5, 32))
        self.assertEqual(fraction(short["optimal_risk_upper"]), F(5, 32))
        blocks = {row["block_size"]: row for row in short["blocks"]}
        self.assertEqual(fraction(blocks[2]["expected_block_score"]), F(5, 128))
        self.assertLess(fraction(blocks[8]["expected_block_score"]), F(1, 10))
        for row in blocks.values():
            self.assertIn(row["hierarchy_coverage"], ("covered", "miss", "unavailable"))

    def test_repeated_histories_keep_the_independent_episode_count(self):
        data = read_json(self.fixture / "calibration/rep_0.json")
        histories = [(row["previous_state"], row["current_state"]) for row in data]
        self.assertLess(len(set(histories)), len(data))
        self.assertEqual(len(data), 32)
        for path, record in self.certificates(self.fixture):
            full = read_json(path.parent / record["full_witness_path"])
            self.assertEqual(full["sample_count"], 32)
            self.assertEqual(fraction(full["weight_total"]), 32)
            self.assertEqual(sum(fraction(group["weight"]) for group in full["groups"]), 32)
            self.assertEqual(record["fitted_upper_endpoint"]["n"], 32)
            self.assertEqual(record["full_empirical_endpoint"]["n"], 32)

    def test_verification_uses_saved_witnesses_without_optimization_or_endpoint_search(self):
        fail = AssertionError("verification attempted a numerical search")
        with mock.patch.object(workflow, "certify_empirical_minimum", side_effect=fail), \
             mock.patch.object(workflow, "bounded_mean_endpoint", side_effect=fail), \
             mock.patch.object(lipschitz_class, "_propose", side_effect=fail), \
             mock.patch.object(lipschitz_class, "certify_weighted_minimum", side_effect=fail), \
             mock.patch.object(class_bounds, "bounded_mean_endpoint", side_effect=fail), \
             mock.patch.object(class_oracle, "audit_class", side_effect=fail):
            result = workflow.verify_class_risk(self.fixture)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["calibration_stream_replayed_in_memory"])
        self.assertTrue(result["population_audit_checked"])
        self.assertFalse(result["models_replayed"])

    def test_changed_input_cannot_be_rebound_only_by_rehashing_new_plan(self):
        for name in ("features.json", "predictions/raw.json", "training.jsonl"):
            with self.subTest(name=name):
                root = self.copy_run(name.replace("/", "_"))
                path = root / "inputs" / name
                # Appending harmless whitespace preserves mathematical values
                # but still changes the original retained source identity.
                path.write_bytes(path.read_bytes() + b"\n")
                plan = read_json(root / "class_risk_plan.json")
                plan["inputs"][name] = sha256_file(path)
                self.rebind_changed_plan(root, plan)
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch|copied input size"):
                    workflow.verify_class_risk(root)

    def test_source_model_prediction_hash_must_agree_with_its_manifest(self):
        root = self.copy_run()
        freeze = read_json(root / "inputs/source_freeze.json")
        plan = read_json(root / "class_risk_plan.json")
        freeze["models"][0]["prediction_table_sha256"] = "0" * 64
        self.rebind_changed_source_freeze(root, freeze, plan)
        with self.assertRaises(ValueError):
            workflow.verify_class_risk(root)

    def test_training_manifest_hash_and_count_remain_bound_to_copied_observations(self):
        for field, replacement in (("sha256", "0" * 64), ("episode_count", 1)):
            with self.subTest(field=field):
                root = self.copy_run(field)
                dataset_path = root / "inputs/dataset_manifest.json"
                dataset = read_json(dataset_path)
                dataset["development"]["training"][field] = replacement
                write_json(dataset_path, dataset)
                freeze = read_json(root / "inputs/source_freeze.json")
                plan = read_json(root / "class_risk_plan.json")
                original_path = freeze["development"]["paths"]["dataset"] + "/manifest.json"
                source_entry = next(entry for entry in freeze["files"] if entry["path"] == original_path)
                source_entry.update(sha256=sha256_file(dataset_path), bytes=dataset_path.stat().st_size)
                plan["inputs"]["dataset_manifest.json"] = sha256_file(dataset_path)
                self.rebind_changed_source_freeze(root, freeze, plan)
                with self.assertRaises(ValueError):
                    workflow.verify_class_risk(root)

    def test_calibration_edit_is_detected_even_after_all_hashes_are_updated(self):
        root = self.copy_run()
        path = root / "calibration/rep_0.json"
        rows = read_json(path)
        rows[0]["next_state"] = (rows[0]["next_state"] + 1) % 4
        write_json(path, rows)
        self.rehash_certificates(root, update_calibration=True)
        with self.assertRaisesRegex(ValueError, "declared deterministic stream"):
            workflow.verify_class_risk(root)

    def test_endpoint_edit_is_rejected_beyond_the_file_hash_gate(self):
        for mutation in ("endpoint", "sample_count", "allocation"):
            with self.subTest(mutation=mutation):
                root = self.copy_run(mutation)
                path, record = self.certificates(root)[0]
                endpoint = record["fitted_upper_endpoint"]
                if mutation == "endpoint":
                    endpoint["endpoint"] = class_bounds.fraction_record(F(0))
                elif mutation == "sample_count":
                    endpoint["n"] += 1
                else:
                    endpoint["alpha"] = class_bounds.fraction_record(F(1, 2))
                write_json(path, record)
                self.rehash_certificates(root)
                with self.assertRaisesRegex(ValueError, "confidence endpoint"):
                    workflow.verify_class_risk(root)

    def test_audit_menu_must_match_the_frozen_block_menu(self):
        root = self.copy_run()
        index = read_json(root / "audit_index.json")
        path = root / index[0]["path"]
        audit = read_json(path)
        audit["block_sizes"] = [2, 4, 8]
        write_json(path, audit)
        index[0]["sha256"] = sha256_file(path)
        write_json(root / "audit_index.json", index)
        with self.assertRaisesRegex(ValueError, "audit block menu"):
            workflow.verify_class_risk(root)

    def test_existing_and_nested_evidence_directories_cannot_be_overwritten(self):
        before = sha256_file(self.fixture / "class_risk_plan.json")
        for output in (self.fixture, self.fixture / "nested", SOURCE_FREEZE.parent / "test-forbidden-output"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                workflow.run_class_risk(SOURCE_FREEZE, output, sample_sizes=(32,), block_sizes=(2, 8))
        self.assertEqual(sha256_file(self.fixture / "class_risk_plan.json"), before)
        self.assertFalse((self.fixture / "nested").exists())
        self.assertFalse((SOURCE_FREEZE.parent / "test-forbidden-output").exists())

    def test_oracle_failure_retains_complete_calibration_certificates(self):
        output = self.case / "oracle-interrupted"
        seen = []

        def fail_after_certification(*args, **kwargs):
            index = read_json(output / "certificate_index.json")
            self.assertEqual(len(index), 5)
            self.assertTrue(all((output / row["path"]).is_file() for row in index))
            self.assertFalse((output / "audit_index.json").exists())
            seen.append(sha256_file(output / "calibration/rep_0.json"))
            raise RuntimeError("synthetic population audit interruption")

        with mock.patch.object(class_oracle, "audit_class", side_effect=fail_after_certification):
            with self.assertRaisesRegex(RuntimeError, "synthetic population audit interruption"):
                workflow.run_class_risk(SOURCE_FREEZE, output, sample_sizes=(8,), block_sizes=(2,),
                                         kappas=(F(2),), sweeps=0, run_id="oracle-interruption-test")
        self.assertEqual(len(seen), 1)
        self.assertEqual(sha256_file(output / "calibration/rep_0.json"), seen[0])
        failure = read_json(output / "failure.json")
        self.assertTrue(failure["partial_evidence_retained"])
        self.assertFalse(failure["sample_replaced"])
        self.assertFalse(failure["output_resumed"])
        self.assertFalse((output / "summary.json").exists())
        checked = workflow.verify_class_risk(output, include_audit=False)
        self.assertEqual(checked["status"], "verified")
        self.assertEqual(len(checked["assessments"]), 5)

    def test_one_unavailable_endpoint_is_retained_without_reallocating_confidence(self):
        output = self.case / "one-unavailable"
        original = class_bounds.bounded_mean_endpoint
        calls = []

        def exhaust_first(q, n, alpha, side):
            calls.append((q, n, alpha, side))
            if len(calls) == 1:
                return original(q, n, alpha, side, budget=WorkBudget(maximum_series_terms=0))
            return original(q, n, alpha, side)

        with mock.patch.object(workflow, "bounded_mean_endpoint", side_effect=exhaust_first):
            result = workflow.run_class_risk(SOURCE_FREEZE, output, sample_sizes=(8,), block_sizes=(2,),
                                             kappas=(F(2),), sweeps=0, run_id="unavailable-endpoint-test")
        self.assertEqual(len(calls), 20)
        self.assertTrue(all(alpha == F(1, 400) for _, _, alpha, _ in calls))
        self.assertEqual(result["unavailable_endpoint_count"], 1)
        raw = next(row for row in result["assessments"] if row["representation"] == "raw")
        self.assertIsNone(raw["fitted_upper"])
        self.assertIsNone(raw["class_upper"])
        self.assertNotEqual(raw["status"], "requirement_met")
        records = [record for _, record in self.certificates(output)]
        self.assertEqual(len(records), 5)
        endpoint = records[0]["fitted_upper_endpoint"]
        self.assertEqual(endpoint["status"], "certificate_unavailable")
        self.assertEqual(endpoint["failure"]["code"], "series_work_limit")
        self.assertEqual(workflow.verify_class_risk(output)["unavailable_endpoint_count"], 1)


if __name__ == "__main__":
    unittest.main()

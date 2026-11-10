"""Static protocol and contract checks; no calibration observations are drawn."""

import copy
from fractions import Fraction
import hashlib
from importlib.resources import files
import unittest
from unittest.mock import Mock, patch

from kahm_demo.calibration_protocol import (
    CALIBRATION_PROTOCOL_SCHEMA, CALIBRATION_PROTOCOL_SHA256,
    _validate_protocol, calibration_protocol_sha256, load_calibration_protocol,
)
from kahm_demo.contract import CONTRACT_SHA256, validate_contract


def _rational(value):
    return Fraction(value["numerator"], value["denominator"])


class CalibrationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_calibration_protocol()
        self.contract = validate_contract()

    def test_exact_resource_identity_and_statistical_allocation(self):
        contents = files("kahm_demo").joinpath("calibration_protocol.json").read_bytes()
        self.assertEqual(hashlib.sha256(contents).hexdigest(), CALIBRATION_PROTOCOL_SHA256)
        self.assertEqual(calibration_protocol_sha256(), CALIBRATION_PROTOCOL_SHA256)
        self.assertEqual(self.protocol["schema"], CALIBRATION_PROTOCOL_SCHEMA)
        self.assertEqual(self.protocol["contract_sha256"], CONTRACT_SHA256)
        self.assertEqual(self.protocol["scope"]["candidate_ids"],
                         ["raw", "kahm_mem", "kahm_local"])
        self.assertEqual(self.protocol["sampling"]["episode_count"], 4096)
        self.assertTrue(self.protocol["sampling"]["shared_calibration_across_candidates"])
        confidence = self.protocol["confidence"]
        alpha = _rational(confidence["per_candidate_failure_probability"])
        total = _rational(confidence["total_failure_probability"])
        self.assertEqual(alpha, Fraction(1, 60))
        self.assertEqual(3 * alpha, total)
        self.assertEqual(1 - total, _rational(confidence["simultaneous_coverage"]))
        self.assertEqual(_rational(self.protocol["reporting"]["primary_tolerance"]),
                         Fraction(1, 10))

    def test_every_resource_byte_is_bound_before_parsing(self):
        contents = files("kahm_demo").joinpath("calibration_protocol.json").read_bytes()
        resource = Mock()
        resource.joinpath.return_value.read_bytes.return_value = contents + b"\n"
        with patch("kahm_demo.calibration_protocol.files", return_value=resource):
            with self.assertRaisesRegex(ValueError, "resource has changed"):
                load_calibration_protocol()

    def test_loader_returns_an_independent_resource_copy(self):
        self.protocol["scope"]["candidate_ids"].append("unplanned")
        self.protocol["verified_logarithm"]["precision_bits_schedule"][0] = 64
        fresh = load_calibration_protocol()
        self.assertEqual(fresh["scope"]["candidate_ids"], ["raw", "kahm_mem", "kahm_local"])
        self.assertEqual(fresh["verified_logarithm"]["precision_bits_schedule"], [128, 256, 512])

    def test_contract_statistical_mismatches_are_rejected_by_public_loader(self):
        changes = (
            ("candidate family", lambda c: c["candidates"].pop()),
            ("calibration count", lambda c: c["data_protocol"].__setitem__("primary_calibration_episodes", 4095)),
            ("independent units", lambda c: c["certificate"].__setitem__("independent_unit_count", 4095)),
            ("per-candidate alpha", lambda c: c["certificate"].__setitem__(
                "per_candidate_failure_probability", {"numerator": 1, "denominator": 20})),
            ("family alpha", lambda c: c["certificate"].__setitem__(
                "total_failure_probability", {"numerator": 1, "denominator": 10})),
            ("shared data", lambda c: c["data_protocol"].__setitem__("calibration_is_shared_across_candidates", False)),
            ("dependent trajectory", lambda c: c["data_protocol"].__setitem__(
                "single_dependent_trajectory_used_for_primary_certificate", True)),
            ("drop duplicates", lambda c: c["data_protocol"].__setitem__("retain_repeated_histories_and_triples", False)),
            ("loss scaling", lambda c: c["claim_scope"].__setitem__("empirical_variance_normalization", True)),
            ("loss range", lambda c: c["claim_scope"].__setitem__("loss_range", [0, 2])),
            ("tolerance", lambda c: c["claim_scope"].__setitem__(
                "primary_mse_tolerance", {"numerator": 1, "denominator": 5})),
        )
        for label, alter in changes:
            changed = copy.deepcopy(self.contract)
            alter(changed)
            with self.subTest(label=label), \
                 patch("kahm_demo.calibration_protocol.validate_contract", return_value=changed):
                with self.assertRaises(ValueError):
                    load_calibration_protocol()

    def test_critical_protocol_changes_fail_structural_validation(self):
        changes = (
            ("contract identity", lambda p: p.__setitem__("contract_sha256", "0" * 64)),
            ("candidate roster", lambda p: p["scope"]["candidate_ids"].append("fourth")),
            ("wrong statistical divisor", lambda p: p["endpoint"].__setitem__("confidence_divisor", 1024)),
            ("wrong logarithm", lambda p: p["endpoint"].__setitem__("confidence_log_argument", 20)),
            ("rounded empirical mean", lambda p: p["loss"].__setitem__("empirical_mean_rounding", "nearest float")),
            ("different precision", lambda p: p["verified_logarithm"].__setitem__("precision_bits_schedule", [64, 128, 256])),
            ("unbounded trials", lambda p: p["work_limits"].__setitem__("maximum_root_trials_per_candidate", 1000000)),
            ("silent fallback", lambda p: p["work_limits"].__setitem__("automatic_looser_endpoint_or_replacement_sample", True)),
            ("different root width", lambda p: p["upper_root_algorithm"].__setitem__(
                "target_bracket_width", {"numerator": 1, "denominator": 2 ** 40})),
            ("wrong failure policy", lambda p: p["work_limits"].__setitem__("fallback", "requirement_met")),
            ("non-rational tolerance", lambda p: p["reporting"].__setitem__("primary_tolerance", .1)),
            ("boolean count", lambda p: p["sampling"].__setitem__("observations_per_episode", True)),
        )
        for label, alter in changes:
            changed = copy.deepcopy(self.protocol)
            alter(changed)
            with self.subTest(label=label), self.assertRaises(ValueError):
                _validate_protocol(changed, self.contract)

    def test_work_limits_cover_the_declared_bounded_search_and_checker(self):
        logarithm = self.protocol["verified_logarithm"]
        root = self.protocol["upper_root_algorithm"]
        limits = self.protocol["work_limits"]
        precisions = logarithm["precision_bits_schedule"]
        self.assertEqual(root["maximum_successful_bisections"], 80)
        self.assertEqual(_rational(root["target_bracket_width"]), Fraction(1, 2 ** 80))
        self.assertEqual(limits["maximum_root_interval_evaluations_per_candidate"],
                         80 * len(precisions))
        # At most two variable logarithms per interval comparison; log(2) and
        # log(60) may be cached once per precision. The direct checker has at
        # most two endpoint comparisons per candidate, and fresh constant caches.
        search_terms = 3 * 80 * 2 * sum(precisions) + 2 * sum(precisions)
        checker_terms = 3 * 2 * 2 * max(precisions) + 2 * sum(precisions)
        self.assertLessEqual(search_terms + checker_terms,
                             limits["maximum_series_term_iterations_for_family_and_checker"])
        self.assertEqual(limits["exact_empirical_risk_denominator_exponent_bound"],
                         2 * 1074 + 12)

    def test_boundary_ambiguity_and_reporting_policies_are_explicit(self):
        endpoint, root = self.protocol["endpoint"], self.protocol["upper_root_algorithm"]
        self.assertIn("U=1", endpoint["q_equals_one"])
        self.assertIn("without evaluating a logarithm", endpoint["q_equals_one"])
        self.assertIn("Do not evaluate log(0)", endpoint["q_equals_zero"])
        self.assertIn("reverse its endpoints", endpoint["q_equals_zero"])
        self.assertIn("not additionally conditional on numerical availability",
                      self.protocol["confidence"]["availability_interpretation"])
        self.assertIn("H=1", root["trivial_upper_boundary"])
        self.assertIn("certificate_unavailable", root["ambiguous_comparison"])
        self.assertTrue(root["maximum_precision_does_not_imply_success"])
        self.assertIn("never guesses a sign", root["ambiguous_comparison"])
        self.assertIn("do not call the root-search routine",
                      self.protocol["independent_numerical_check"]["scope"])
        reporting = self.protocol["reporting"]
        self.assertEqual(reporting["decimal_places"], 18)
        self.assertIn("ceil", reporting["outward_decimal_rule"])
        self.assertIn("10*A<=B", reporting["decision_rule"])
        self.assertIn("not used for the decision", reporting["decision_rule"])
        self.assertEqual(set(reporting["statuses"]),
                         {"requirement_met", "unresolved", "certificate_unavailable"})

    def test_malformed_contract_or_protocol_has_a_clear_validation_failure(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            _validate_protocol(self.protocol, {})
        with self.assertRaisesRegex(ValueError, "malformed"):
            _validate_protocol({}, self.contract)


if __name__ == "__main__":
    unittest.main()

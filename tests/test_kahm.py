"""Independent formula and exceptional-input checks for Gaussian KAHM."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from kahm_demo.kahm import (
    NormalizationError,
    fit_local_kahm,
    signed_normalized_reconstruction,
)


class KAHMConstructionTests(unittest.TestCase):
    def setUp(self):
        self.Q = np.array([[-1.0, -0.3], [0.5, -0.8], [0.7, 1.1], [-0.4, 0.9]])

    def test_gaussian_gram_and_reconstruction_match_direct_unscaled_formula(self):
        model = fit_local_kahm(self.Q, 2)
        P, S, r = model.pca.P, model.pca.S, model.pca.rank
        S_inverse = np.linalg.inv(S)

        def direct_kernel(x, y):
            v = P.T @ (x - y)
            return np.exp(-(v @ S_inverse @ v) / (2 * r))

        direct_gram = np.array([[direct_kernel(x, y) for y in self.Q] for x in self.Q])
        np.testing.assert_allclose(model.gram, direct_gram, rtol=3e-15, atol=3e-15)
        np.testing.assert_array_equal(np.diag(model.gram), np.ones(len(self.Q)))
        z = np.array([0.37, -0.21])
        direct_query = np.array([direct_kernel(z, q) for q in self.Q])
        weights = np.linalg.solve(direct_gram + model.fitted_lambda * np.eye(4), direct_query)
        expected = self.Q.T @ weights / np.sum(weights)
        result = model.reconstruct_one(z)
        self.assertEqual(result.diagnostics["mode"], "kahm")
        np.testing.assert_allclose(result.reconstruction, expected, rtol=3e-14, atol=3e-14)

    def test_reference_energy_is_not_locally_centered(self):
        Q = np.array([[2.0, 3.0], [2.5, 3.0], [2.0, 3.5], [2.5, 3.5]])
        model = fit_local_kahm(Q, 2)
        expected = float(np.sum(Q * Q) / Q.size)
        local_centered = float(np.sum((Q - Q.mean(axis=0)) ** 2) / Q.size)
        self.assertNotEqual(expected, local_centered)
        self.assertEqual(model.diagnostics["reference_energy"], expected)
        self.assertEqual(model.diagnostics["iteration_records"][0]["lambda"], 2.5 * expected)

    def test_last_completed_inverse_and_lambda_are_retained(self):
        Q = np.array([[-0.01], [0.01]])
        model = fit_local_kahm(Q, 1)
        nu = float(np.sum(Q * Q) / Q.size)
        initial_lambda = 2.5 * nu
        expected_inverse = np.linalg.solve(model.gram + initial_lambda * np.eye(2), np.eye(2))
        proposal = 2 * nu + float(np.sum((Q - model.gram @ (expected_inverse @ Q)) ** 2) / Q.size)
        self.assertLessEqual(abs(proposal - initial_lambda), 1e-3)
        self.assertEqual(model.iterations, 1)
        self.assertTrue(model.converged)
        self.assertEqual(model.fitted_lambda, initial_lambda)
        self.assertNotEqual(model.fitted_lambda, proposal)
        np.testing.assert_array_equal(model.C, expected_inverse)
        self.assertEqual(model.diagnostics["unused_next_lambda_proposal"], proposal)

    def test_iteration_limit_reports_nonconvergence_without_using_unsolved_proposal(self):
        # A short artificial work limit isolates the documented last-solve rule.
        with patch("kahm_demo.kahm.MAX_REGULARIZATION_SOLVES", 1):
            model = fit_local_kahm(self.Q, 2)
        self.assertFalse(model.converged)
        self.assertEqual(model.iterations, 1)
        initial_lambda = 2.5 * float(np.sum(self.Q ** 2) / self.Q.size)
        self.assertEqual(model.fitted_lambda, initial_lambda)
        self.assertEqual(model.diagnostics["regularization_status"], "iteration_limit_retained_last_solve")
        np.testing.assert_allclose((model.gram + initial_lambda * np.eye(4)) @ model.C, np.eye(4), atol=3e-15)

    def test_full_rank_kahm_is_not_replaced_by_identity(self):
        Q = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        model = fit_local_kahm(Q, 2)
        self.assertEqual(model.pca.rank, 2)
        z = np.array([0.0, 0.0])
        result = model.reconstruct_one(z)
        self.assertEqual(result.diagnostics["mode"], "kahm")
        self.assertGreater(np.linalg.norm(result.reconstruction - z), 1e-3)
        np.testing.assert_array_equal(model.affine_one(z), z)

    def test_projected_collisions_are_recorded_and_regularized(self):
        Q = np.array([[-2.0, -0.1], [-2.0, 0.1], [2.0, -0.1], [2.0, 0.1]])
        model = fit_local_kahm(Q, 1)
        self.assertEqual(model.pca.rank, 1)
        self.assertEqual(model.diagnostics["raw_distinct_count"], 4)
        self.assertEqual(model.diagnostics["projected_collision_count"], 2)
        self.assertFalse(model.diagnostics["projected_reference_distinctness_claimed"])
        self.assertGreater(model.fitted_lambda, 0)
        result = model.reconstruct_one([0.3, 0.7])
        self.assertTrue(np.isfinite(result.reconstruction).all())
        self.assertEqual(model.Q.shape[0], 4)

    def test_arrays_are_immutable_and_source_references_are_copied(self):
        source = self.Q.copy()
        model = fit_local_kahm(source, 2)
        source[:] = 900
        np.testing.assert_array_equal(model.Q, self.Q)
        for array in (model.Q, model.gram, model.metric_transform, model.C):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        json.dumps(model.diagnostics, allow_nan=False)


class KAHMQueryTests(unittest.TestCase):
    def test_negative_signed_sum_and_negative_weights_are_valid(self):
        Q = np.array([[0.0, 0.0], [1.0, 2.0]])
        # Denominator is -1; normalized weights are [-1, 2], not probabilities.
        value = signed_normalized_reconstruction(Q, np.array([1.0, -2.0]))
        np.testing.assert_array_equal(value, [2.0, 4.0])
        np.testing.assert_array_equal(
            signed_normalized_reconstruction(Q, np.array([-1.0, -3.0])), [0.75, 1.5]
        )

    def test_common_weight_scaling_preserves_signed_quotient(self):
        Q = np.array([[-1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])
        w = np.array([0.7, -0.2, 0.8])
        expected = Q.T @ w / w.sum()
        for scale in (1.0, 1e-200, 1e200, -7.0):
            actual = signed_normalized_reconstruction(Q, w * scale)
            np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=5e-15)

    def test_cancellation_zero_and_nonfinite_weights_fail(self):
        Q = np.array([[-1.0], [1.0]])
        for weights, reason in (
            ([1.0, -1.0 + 1e-12], "signed_cancellation"),
            ([1.0, -1.0], "signed_cancellation"),
            ([0.0, 0.0], "zero_weight_l1"),
            ([np.inf, 1.0], "nonfinite_weights"),
        ):
            with self.subTest(weights=weights), self.assertRaises(NormalizationError) as caught:
                signed_normalized_reconstruction(Q, weights)
            self.assertEqual(caught.exception.reason, reason)

    def test_query_log_rescaling_prevents_all_kernel_underflow(self):
        model = fit_local_kahm(np.array([[-1.0], [1.0]]), 1)
        z = np.array([100.0])
        logs = model.log_kernel_one(z)
        with np.errstate(under="ignore"):
            self.assertTrue(np.all(np.exp(logs) == 0))
            shifted = np.exp(logs - logs.max())
        self.assertEqual(shifted.max(), 1.0)
        result = model.reconstruct_one(z)
        self.assertEqual(result.diagnostics["mode"], "kahm")
        self.assertTrue(result.diagnostics["unscaled_kernel_would_be_all_zero"])
        expected = signed_normalized_reconstruction(model.Q, model.C @ shifted)
        np.testing.assert_array_equal(result.reconstruction, expected)

    def test_signed_cancellation_uses_affine_fallback(self):
        fitted = fit_local_kahm(np.array([[-1.0, 0.0], [1.0, 0.0]]), 1)
        # Construct weights [k0,-k0] to exercise the query guard independently
        # of whether a naturally fitted model happens to encounter cancellation.
        model = replace(fitted, C=np.array([[1.0, 0.0], [-1.0, 0.0]]))
        result = model.reconstruct_one([0.2, 7.0])
        self.assertTrue(result.diagnostics["used_fallback"])
        self.assertEqual(result.diagnostics["fallback_reason"], "signed_cancellation")
        np.testing.assert_allclose(result.reconstruction, [0.2, 0.0], atol=1e-15)
        json.dumps(result.diagnostics, allow_nan=False)

    def test_full_rank_affine_fallback_returns_input_exactly(self):
        Q = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        fitted = fit_local_kahm(Q, 2)
        model = replace(fitted, C=np.zeros((4, 4)))
        query = np.array([0.123456789, -8.7654321])
        result = model.reconstruct_one(query)
        self.assertEqual(result.diagnostics["mode"], "affine_fallback")
        np.testing.assert_array_equal(result.reconstruction, query)

    def test_rank_zero_returns_mean_without_kernel(self):
        model = fit_local_kahm(np.array([[0.25, -0.75]]), 2)
        self.assertEqual(model.pca.rank, 0)
        self.assertIsNone(model.fitted_lambda)
        self.assertEqual(model.iterations, 0)
        self.assertEqual(model.C.shape, (0, 0))
        with patch.object(type(model), "log_kernel_one", side_effect=AssertionError("kernel should not run")):
            result = model.reconstruct_one([9.0, 8.0])
        np.testing.assert_array_equal(result.reconstruction, [0.25, -0.75])
        self.assertEqual(result.diagnostics["mode"], "rank_zero_mean")
        self.assertFalse(result.diagnostics["kernel_evaluated"])

    def test_nonfinite_or_wrong_shape_queries_are_rejected_even_for_rank_zero(self):
        models = (
            fit_local_kahm(np.array([[-1.0], [1.0]]), 1),
            fit_local_kahm(np.array([[0.0]]), 1),
        )
        for model in models:
            for query in ([np.nan], [np.inf], [[0.0]], [0.0, 1.0]):
                with self.subTest(rank=model.pca.rank, query=query), self.assertRaises(ValueError):
                    model.reconstruct_one(query)

    def test_one_row_result_does_not_depend_on_other_queries(self):
        model = fit_local_kahm(np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5]]), 2)
        first = model.reconstruct_one([0.2, 0.3]).reconstruction
        model.reconstruct_one([100.0, -80.0])
        again = model.reconstruct_one([0.2, 0.3]).reconstruction
        np.testing.assert_array_equal(first, again)


if __name__ == "__main__":
    unittest.main()

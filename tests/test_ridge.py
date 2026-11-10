"""Independent objective checks and canonical inference checks for ridge."""

from dataclasses import FrozenInstanceError
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from kahm_demo.ridge import RidgeModel, fit_ridge, fit_ridge_grid


def direct_normal_equations(X, y, alpha):
    """Independent augmented system, with a zero penalty on the intercept."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    augmented = np.column_stack((np.ones(len(X)), X))
    regularizer = np.diag([0.0] + [len(X) * alpha] * X.shape[1])
    return np.linalg.solve(augmented.T @ augmented + regularizer, augmented.T @ y)


class RidgeFitTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, 0.5],
                           [2.0, -1.0, 1.0], [-1.0, 0.5, 2.0],
                           [0.5, 2.0, 0.0]])
        self.y = np.array([0.0, 0.25, 0.75, 1.0, 0.5])

    def test_matches_direct_augmented_normal_equations(self):
        model = fit_ridge(self.X, self.y, 0.3)
        direct = direct_normal_equations(self.X, self.y, 0.3)
        np.testing.assert_allclose(model.coefficients, direct[1:], rtol=2e-13, atol=2e-15)
        self.assertAlmostEqual(model.intercept, direct[0], places=14)
        self.assertEqual(model.training_count, 5)
        self.assertEqual(model.feature_count, 3)

    def test_mean_loss_requires_n_times_alpha(self):
        X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
        model = fit_ridge(X, [0.0, 1.0, 0.0, 1.0], 0.5)
        self.assertAlmostEqual(model.coefficients[0], 1.0 / 3.0, places=15)
        self.assertEqual(model.intercept, 0.5)
        self.assertEqual(model.diagnostics["normal_matrix_penalty"], 2.0)
        self.assertGreater(abs(model.coefficients[0] - 2.0 / 4.5), 0.1)

    def test_repeating_all_rows_preserves_mean_objective_fit(self):
        original = fit_ridge(self.X, self.y, 0.2)
        repeated = fit_ridge(np.repeat(self.X, 3, axis=0), np.repeat(self.y, 3), 0.2)
        np.testing.assert_allclose(repeated.coefficients, original.coefficients,
                                   rtol=2e-13, atol=2e-15)
        self.assertAlmostEqual(repeated.intercept, original.intercept, places=14)
        self.assertEqual(repeated.training_count, 15)

    def test_unpenalized_intercept_and_constant_columns(self):
        X = np.tile([2.0, -4.0, 0.0], (4, 1))
        model = fit_ridge(X, [0.0, 0.25, 0.75, 1.0], 100.0)
        np.testing.assert_array_equal(model.coefficients, [0.0, 0.0, 0.0])
        self.assertEqual(model.intercept, 0.5)
        self.assertEqual(model.feature_count, 3)
        self.assertFalse(model.diagnostics["intercept_penalized"])
        self.assertTrue(model.diagnostics["all_feature_columns_retained"])

    def test_rank_deficiency_keeps_every_coefficient(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        X = np.column_stack((x, 2.0 * x, np.ones(4)))
        y = np.array([0.0, 0.25, 0.75, 1.0])
        model = fit_ridge(X, y, 0.1)
        direct = direct_normal_equations(X, y, 0.1)
        self.assertEqual(model.coefficients.shape, (3,))
        np.testing.assert_allclose(model.coefficients, direct[1:], atol=2e-14, rtol=2e-13)
        self.assertEqual(model.diagnostics["diagnostic_numerical_rank"], 1)
        self.assertFalse(model.diagnostics["rank_threshold_used_to_drop_columns"])

    def test_more_features_than_rows_and_singleton(self):
        X = np.array([[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, 3.0, 2.0]])
        model = fit_ridge(X, [0.25, 0.75], 0.05)
        direct = direct_normal_equations(X, [0.25, 0.75], 0.05)
        np.testing.assert_allclose(model.coefficients, direct[1:], atol=2e-14, rtol=2e-13)
        singleton = fit_ridge([[3.0, -2.0]], [0.75], 1e-6)
        np.testing.assert_array_equal(singleton.coefficients, [0.0, 0.0])
        self.assertEqual(singleton.intercept, 0.75)

    def test_grid_reuses_one_svd_and_matches_independent_solutions(self):
        alphas = (1e-6, 0.1, 100.0)
        original_svd = np.linalg.svd
        with patch("kahm_demo.ridge.np.linalg.svd", wraps=original_svd) as svd:
            models = fit_ridge_grid(self.X, self.y, alphas)
            self.assertEqual(svd.call_count, 1)
        self.assertEqual(tuple(model.alpha for model in models), alphas)
        for alpha, model in zip(alphas, models, strict=True):
            with self.subTest(alpha=alpha):
                direct = direct_normal_equations(self.X, self.y, alpha)
                separate = fit_ridge(self.X, self.y, alpha)
                np.testing.assert_allclose(model.coefficients, direct[1:], rtol=1e-10, atol=2e-13)
                np.testing.assert_array_equal(model.coefficients, separate.coefficients)
                self.assertEqual(model.intercept, separate.intercept)

    def test_diagnostics_are_finite_json_and_not_a_verified_enclosure(self):
        model = fit_ridge(self.X, self.y, 0.2)
        json.dumps(model.diagnostics, allow_nan=False)
        d = model.diagnostics
        self.assertLessEqual(d["svd_reconstruction_relative_fro"], d["svd_check_tolerance"])
        self.assertLessEqual(d["svd_u_orthogonality_inf"], d["svd_check_tolerance"])
        self.assertLessEqual(d["svd_v_orthogonality_inf"], d["svd_check_tolerance"])
        self.assertLessEqual(d["normal_equation_relative_residual"], d["normal_equation_tolerance"])
        self.assertLessEqual(d["intercept_mean_residual_abs"], d["intercept_check_tolerance"])
        self.assertFalse(d["numerical_enclosure_claimed"])
        self.assertFalse(d["solver_recovery_used"])

    def test_svd_failure_does_not_silently_repair_or_skip_fit(self):
        with patch("kahm_demo.ridge.np.linalg.svd", side_effect=np.linalg.LinAlgError("test failure")):
            with self.assertRaisesRegex(ValueError, "SVD failed"):
                fit_ridge(self.X, self.y, 0.1)

    def test_svd_reconstruction_and_orthogonality_are_actually_checked(self):
        centered = self.X - self.X.mean(axis=0)
        U, singular, Vt = np.linalg.svd(centered, full_matrices=False)
        for bad_result in ((U, np.zeros_like(singular), Vt), (2.0 * U, singular / 2.0, Vt)):
            with self.subTest(kind="reconstruction" if np.all(bad_result[1] == 0) else "orthogonality"):
                with patch("kahm_demo.ridge.np.linalg.svd", return_value=bad_result):
                    with self.assertRaisesRegex(ValueError, "SVD failed reconstruction or orthogonality"):
                        fit_ridge(self.X, self.y, 0.1)

    def test_fit_validation(self):
        bad_pairs = [([], []), ([[1.0]], []), ([[1.0], [2.0]], [0.5]),
                     (np.empty((2, 0)), [0.0, 1.0]), ([[float("nan")]], [0.5]),
                     ([[float("inf")]], [0.5]), ([[1.0]], [float("nan")]),
                     ([[1.0]], [-0.01]), ([[1.0]], [1.01]), ([[1.0]], [[0.5]])]
        for X, y in bad_pairs:
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                fit_ridge(X, y, 0.1)
        for X in ([[True]], [[1 + 1j]], [["1"]]):
            with self.subTest(X=X), self.assertRaises(TypeError):
                fit_ridge(X, [0.5], 0.1)
        for y in ([True], [0.5 + 0j], ["0.5"]):
            with self.subTest(y=y), self.assertRaises(TypeError):
                fit_ridge([[1.0]], y, 0.1)
        for alpha in (0, -1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                fit_ridge(self.X, self.y, alpha)
        for alpha in (True, "1", None, 1 + 0j):
            with self.subTest(alpha=alpha), self.assertRaises(TypeError):
                fit_ridge(self.X, self.y, alpha)
        with self.assertRaises(ValueError):
            fit_ridge_grid(self.X, self.y, ())
        with self.assertRaises(TypeError):
            fit_ridge_grid(self.X, self.y, "0.1")
        with self.assertRaises(ValueError):
            fit_ridge([[1e308], [1e308]], [0.0, 1.0], 0.1)


class RidgeInferenceTests(unittest.TestCase):
    def test_clipping_follows_finite_unclipped_readout(self):
        model = RidgeModel(np.array([2.0]), 0.5, 0.1, 2, np.array([0.0]), 0.5, {})
        self.assertEqual(model.predict_unclipped_one([-1.0]), -1.5)
        self.assertEqual(model.predict_one([-1.0]), 0.0)
        self.assertEqual(model.predict_one([0.0]), 0.5)
        self.assertEqual(model.predict_unclipped_one([1.0]), 2.5)
        self.assertEqual(model.predict_one([1.0]), 1.0)
        with self.assertRaises(ValueError):
            model.predict_one([1e308])

    def test_row_and_batch_are_bit_identical_and_keep_duplicates(self):
        model = fit_ridge([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [-1.0, 4.0]],
                          [0.0, 0.25, 0.75, 1.0], 0.01)
        queries = np.array([[0.0, 1.0], [100.0, -20.0], [0.0, 1.0], [-0.0, 1.0]])
        expected = np.array([model.predict_one(row) for row in queries])
        expected_raw = np.array([model.predict_unclipped_one(row) for row in queries])
        np.testing.assert_array_equal(model.predict(queries), expected)
        np.testing.assert_array_equal(model.predict_unclipped(queries), expected_raw)
        self.assertEqual(model.predict(queries).shape, (4,))
        self.assertEqual(model.predict(np.empty((0, 2))).shape, (0,))

    def test_query_rows_do_not_refit_any_statistics(self):
        model = fit_ridge([[0.0], [2.0]], [0.0, 1.0], 0.1)
        coefficients = model.coefficients.copy()
        mean = model.feature_mean.copy()
        first = model.predict_one([0.0])
        batch = model.predict([[0.0], [1000.0], [1000.0]])
        self.assertEqual(batch[0], first)
        np.testing.assert_array_equal(model.coefficients, coefficients)
        np.testing.assert_array_equal(model.feature_mean, mean)

    def test_canonical_readout_includes_intercept_in_fsum(self):
        model = RidgeModel(np.array([1e16, 1.0, -1e16]), 0.5, 0.1, 3,
                           np.zeros(3), 0.5, {})
        self.assertEqual(model.predict_unclipped_one([1.0, 1.0, 1.0]), 1.5)
        self.assertEqual(model.predict_one([1.0, 1.0, 1.0]), 1.0)

    def test_state_is_detached_and_numerical_arrays_are_immutable(self):
        coefficients = np.array([0.25, 0.5])
        mean = np.array([0.0, 0.0])
        diagnostics = {"nested": {"flag": False}}
        model = RidgeModel(coefficients, 0.5, 0.1, 4, mean, 0.5, diagnostics)
        coefficients[:] = 99.0
        mean[:] = 99.0
        diagnostics["nested"]["flag"] = True
        np.testing.assert_array_equal(model.coefficients, [0.25, 0.5])
        np.testing.assert_array_equal(model.feature_mean, [0.0, 0.0])
        self.assertFalse(model.diagnostics["nested"]["flag"])
        for array in (model.coefficients, model.feature_mean, model.predict([[0.0, 0.0]])):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            model.intercept = 0.0

    def test_query_validation_matches_in_both_paths(self):
        model = fit_ridge([[0.0], [1.0]], [0.0, 1.0], 0.1)
        for bad in ([float("nan")], [float("inf")], [], [0.0, 1.0], [[1.0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.predict_one(bad)
        for one, many in (([True], [[True]]), (["1"], [["1"]]),
                          ([1 + 1j], [[1 + 1j]])):
            with self.subTest(one=one):
                with self.assertRaises(TypeError):
                    model.predict_one(one)
                with self.assertRaises(TypeError):
                    model.predict(many)
        with self.assertRaises(ValueError):
            model.predict([[float("inf")]])
        with self.assertRaises(ValueError):
            model.predict([0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

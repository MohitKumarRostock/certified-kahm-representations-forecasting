"""Training-only geometry checks; no calibration or response values are used."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import unittest

import numpy as np

from kahm_demo.geometry import fit_local_pca, fit_reference_partitions, fit_standardizer


class StandardizerTests(unittest.TestCase):
    def test_all_occurrences_determine_population_moments(self):
        X = np.array([[0., 0.], [0., 0.], [4., 6.]])
        fitted = fit_standardizer(X)
        np.testing.assert_allclose(fitted.mean, [float(Fraction(4, 3)), 2.], rtol=0, atol=1e-15)
        np.testing.assert_allclose(fitted.raw_std ** 2, [float(Fraction(32, 9)), 8.], rtol=0, atol=2e-15)
        self.assertEqual(fitted.sample_count, 3)
        self.assertNotEqual(fitted.mean[0], 2.)  # Giving each distinct row one vote is different.
        Z = fitted.transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0., atol=2e-16)
        np.testing.assert_allclose(np.mean(Z ** 2, axis=0), 1., atol=3e-16)

    def test_small_standard_deviations_are_replaced_including_boundary(self):
        X = [[-1e-12, -2e-12, 5.], [1e-12, 2e-12, 5.]]
        fitted = fit_standardizer(X)
        self.assertEqual(fitted.replaced_columns, (0, 2))
        np.testing.assert_array_equal(fitted.scale, [1., 2e-12, 1.])
        self.assertEqual(fitted.diagnostics["replacement_count"], 2)
        self.assertEqual(fitted.diagnostics["variance_divisor"], 2)

    def test_singleton_and_transform_one_match_matrix_transformation(self):
        fitted = fit_standardizer([[2., 3.]])
        self.assertEqual(fitted.replaced_columns, (0, 1))
        np.testing.assert_array_equal(fitted.transform_one([2., 3.]), [0., 0.])
        queries = np.array([[8., 9.], [0., -1.]])
        for index, query in enumerate(queries):
            np.testing.assert_array_equal(fitted.transform_one(query), fitted.transform(queries)[index])
        self.assertEqual(fitted.transform(np.empty((0, 2))).shape, (0, 2))

    def test_fitted_arrays_are_copied_and_cannot_be_made_writable(self):
        X = np.array([[0., 2.], [2., 4.]])
        fitted = fit_standardizer(X)
        X[:] = 100.
        np.testing.assert_array_equal(fitted.mean, [1., 3.])
        for array in (fitted.mean, fitted.scale, fitted.raw_std):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array[0] = 0.
        with self.assertRaises(FrozenInstanceError):
            fitted.sample_count = 9

    def test_invalid_inputs_and_incompatible_query_shapes(self):
        for X in ([], [[1., np.nan]], [[np.inf, 0.]], np.empty((1, 0)), [1., 2.]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                fit_standardizer(X)
        for X in ([[1 + 2j, 0]], [["1", "2"]], [[True, False]]):
            with self.subTest(X=X), self.assertRaises(TypeError):
                fit_standardizer(X)
        fitted = fit_standardizer([[0., 1.], [2., 3.]])
        with self.assertRaises(ValueError):
            fitted.transform([[1.]])
        with self.assertRaises(ValueError):
            fitted.transform_one([[1., 2.]])
        json.dumps(fitted.diagnostics, allow_nan=False)


class ReferencePartitionTests(unittest.TestCase):
    def test_duplicate_compression_preserves_weights_and_first_identities(self):
        Z = np.array([[4., 0.], [0., 0.], [4., 0.], [4., 0.]])
        partition, = fit_reference_partitions(Z, requested_groups=(1,))
        group, = partition.groups
        self.assertEqual((partition.training_count, partition.distinct_count), (4, 2))
        np.testing.assert_array_equal(group.Q, [[0., 0.], [4., 0.]])
        np.testing.assert_array_equal(group.reference_indices, [1, 0])
        np.testing.assert_array_equal(group.reference_multiplicities, [1, 3])
        np.testing.assert_array_equal(group.member_indices, [1, 0])
        np.testing.assert_array_equal(group.member_multiplicities, [1, 3])
        np.testing.assert_array_equal(group.centroid, [3., 0.])
        self.assertAlmostEqual(partition.inertia, 12.)
        # Clustering uses occurrence weights; local reference PCA uses Q once.
        local = fit_local_pca(group.Q, 1)
        np.testing.assert_array_equal(local.mu, [2., 0.])

    def test_weighted_cluster_centers_equal_expanded_training_means(self):
        distinct = np.array([[-4., 0.], [-2., 1.], [0., 0.], [3., 1.], [5., 0.]])
        multiplicities = [2, 7, 1, 4, 3]
        Z = np.repeat(distinct, multiplicities, axis=0)
        for partition in fit_reference_partitions(Z):
            observed_ids = []
            total = 0
            for group in partition.groups:
                members = {tuple(Z[i]) for i in group.member_indices}
                expanded = np.array([row for row in Z if tuple(row) in members])
                np.testing.assert_allclose(group.centroid, expanded.mean(axis=0), rtol=0, atol=1e-15)
                self.assertEqual(len(expanded), int(group.member_multiplicities.sum()))
                total += len(expanded)
                observed_ids.extend(group.member_indices.tolist())
            self.assertEqual(total, len(Z))
            self.assertEqual(len(set(observed_ids)), len(distinct))

    def test_deterministic_initialization_and_assignment_ties(self):
        Z = np.array([[2.], [0.], [4.], [1.], [3.]])
        partition, = fit_reference_partitions(Z, requested_groups=(2,))
        self.assertEqual(partition.attempts[0].initial_reference_indices, (1, 2))
        np.testing.assert_array_equal(partition.groups[0].Q[:, 0], [0., 1., 2.])
        np.testing.assert_array_equal(partition.groups[1].Q[:, 0], [3., 4.])
        np.testing.assert_array_equal(partition.groups[0].centroid, [1.])
        np.testing.assert_array_equal(partition.groups[1].centroid, [3.5])
        repeated, = fit_reference_partitions(Z, requested_groups=(2,))
        self.assertEqual(partition.diagnostics, repeated.diagnostics)

    def test_reference_budget_uses_distinct_farthest_first_rows(self):
        Z = np.array([[3.], [0.], [1.], [2.], [4.], [5.]])
        partition, = fit_reference_partitions(Z, requested_groups=(1,), reference_budget=3)
        group, = partition.groups
        np.testing.assert_array_equal(group.Q[:, 0], [0., 5., 2.])
        np.testing.assert_array_equal(group.reference_indices, [1, 5, 3])
        self.assertEqual(len(group.member_indices), 6)
        self.assertEqual(len(group.Q), 3)

    def test_unique_shortage_and_constant_reference_sets_are_recorded(self):
        partitions = fit_reference_partitions([[1., 2.], [1., 2.], [0., 0.]])
        self.assertEqual([p.effective_groups for p in partitions], [2, 2])
        self.assertFalse(partitions[0].diagnostics["unique_shortage_reduction"])
        self.assertTrue(partitions[1].diagnostics["unique_shortage_reduction"])
        for partition in fit_reference_partitions([[1., 2.]] * 5):
            self.assertEqual(partition.effective_groups, 1)
            self.assertEqual(partition.groups[0].reference_multiplicities.tolist(), [5])
            self.assertEqual(fit_local_pca(partition.groups[0].Q, 2).rank, 0)

    def test_empty_cluster_retries_halve_count_in_a_degenerate_numerical_case(self):
        # Distinct inputs whose squared distances underflow to zero expose the
        # actual empty-cluster path without mocking the optimizer.
        Z = np.array([[0.], [1e-200], [2e-200], [3e-200]])
        partition, = fit_reference_partitions(Z, requested_groups=(4,))
        self.assertEqual([a.group_count for a in partition.attempts], [4, 2, 1])
        self.assertEqual([a.status for a in partition.attempts],
                         ["empty_group_retry", "empty_group_retry", "converged"])
        self.assertEqual(partition.effective_groups, 1)
        self.assertEqual(int(partition.groups[0].member_multiplicities.sum()), 4)

    def test_iteration_limit_is_visible_and_final_centers_use_recorded_members(self):
        partition, = fit_reference_partitions([[0.], [1.], [2.], [3.], [4.]],
                                             requested_groups=(2,), max_iter=1)
        self.assertEqual(partition.attempts[-1].status, "max_iter_reached")
        self.assertEqual(partition.attempts[-1].iterations, 1)
        np.testing.assert_array_equal(partition.groups[0].centroid, [1.])
        np.testing.assert_array_equal(partition.groups[1].centroid, [3.5])

    def test_reference_identities_are_fixed_across_caps_and_state_is_immutable(self):
        Z = np.array([(a, b) for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        partitions = fit_reference_partitions(Z)
        before = [p.diagnostics for p in partitions]
        saved = [group.Q.copy() for p in partitions for group in p.groups]
        for partition in partitions:
            for group in partition.groups:
                one, two = fit_local_pca(group.Q, 1), fit_local_pca(group.Q, 2)
                np.testing.assert_array_equal(one.mu, two.mu)
                self.assertEqual(one.reference_count, two.reference_count)
                for array in (group.Q, group.centroid, group.reference_indices,
                              group.reference_multiplicities, group.member_indices,
                              group.member_multiplicities):
                    with self.assertRaises(ValueError):
                        array.setflags(write=True)
        Z[:] = 99.
        for actual, expected in zip((g.Q for p in partitions for g in p.groups), saved):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(before, [p.diagnostics for p in partitions])
        json.dumps(before, allow_nan=False)

    def test_bad_partition_parameters_are_rejected(self):
        Z = [[0., 0.], [1., 1.]]
        for keyword, value, error in (("requested_groups", (), ValueError),
                                      ("requested_groups", (0,), ValueError),
                                      ("requested_groups", (True,), TypeError),
                                      ("reference_budget", 0, ValueError),
                                      ("max_iter", 1.5, TypeError)):
            with self.subTest(keyword=keyword, value=value), self.assertRaises(error):
                fit_reference_partitions(Z, **{keyword: value})


class LocalPCATests(unittest.TestCase):
    def test_singleton_and_constant_rank_zero_rules(self):
        for Q, reason in (([[2., -3.]], "singleton"), ([[2., -3.]] * 3, "constant_references")):
            with self.subTest(reason=reason):
                local = fit_local_pca(Q, 2)
                self.assertEqual(local.rank, 0)
                self.assertEqual(local.P.shape, (2, 0))
                self.assertEqual(local.S.shape, (0, 0))
                self.assertEqual(local.reason, reason)
                np.testing.assert_array_equal(local.mu, [2., -3.])
                self.assertEqual(local.projected_unique_count, 1)
                json.dumps(local.diagnostics, allow_nan=False)

    def test_collinear_references_have_one_direction_and_sample_covariance(self):
        Q = np.array([[-2., -4.], [-1., -2.], [0., 0.], [1., 2.], [2., 4.]])
        local = fit_local_pca(Q, 2)
        self.assertEqual(local.rank, 1)
        np.testing.assert_allclose(local.P[:, 0], np.array([1., 2.]) / np.sqrt(5), atol=1e-15)
        np.testing.assert_allclose(local.S, [[12.5]], rtol=0, atol=4e-15)
        np.testing.assert_allclose(local.P.T @ local.P, [[1.]], atol=3e-16)
        self.assertEqual(local.initial_rank_request, 2)
        np.linalg.cholesky(local.S)

    def test_full_rank_orthogonality_and_projected_metric(self):
        Q = np.array([[-1., 0.], [0., 2.], [2., -1.]])
        local = fit_local_pca(Q, 2)
        self.assertEqual(local.rank, 2)
        np.testing.assert_allclose(local.P.T @ local.P, np.eye(2), atol=5e-16)
        np.testing.assert_allclose(local.S, np.cov(Q @ local.P, rowvar=False, ddof=1), atol=1e-15)
        z = np.array([4., -7.])
        np.testing.assert_allclose(local.mu + local.P @ local.P.T @ (z - local.mu), z, atol=4e-15)
        self.assertTrue(np.all(np.diag(np.linalg.cholesky(local.S)) > 0))
        one = fit_local_pca(Q, 1)
        np.testing.assert_array_equal(one.P, local.P[:, :1])

    def test_exactly_repeated_eigenvalues_get_coordinate_order_basis(self):
        Q = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        local = fit_local_pca(Q, 2)
        np.testing.assert_array_equal(local.P, np.eye(2))
        np.testing.assert_allclose(local.eigenvalues, [4 / 3, 4 / 3], atol=0, rtol=0)
        self.assertEqual(local.repeated_eigenvalue_blocks, ((0, 2),))
        np.testing.assert_array_equal(fit_local_pca(Q, 1).P[:, 0], [1., 0.])

    def test_projected_collisions_are_counted_without_discarding_references(self):
        Q = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        one, two = fit_local_pca(Q, 1), fit_local_pca(Q, 2)
        self.assertEqual((one.reference_count, two.reference_count), (4, 4))
        self.assertEqual(one.diagnostics["projected_collision_count"], 2)
        self.assertEqual(one.minimum_projected_separation, 0.)
        self.assertEqual(two.projected_unique_count, 4)
        self.assertEqual(two.minimum_projected_separation, 2.)

    def test_range_reduction_and_remaining_rank_one_violation(self):
        Q = np.array([[-1., -1e-4], [-1., 1e-4], [1., -1e-4], [1., 1e-4]])
        local = fit_local_pca(Q, 2)
        self.assertEqual(local.rank, 1)
        self.assertEqual((local.initial_rank_request, local.final_rank_request), (2, 1))
        self.assertEqual([a.range_violation for a in local.attempts], [True, False])
        self.assertEqual(local.reason, "range_reduced")
        tiny = fit_local_pca([[-1e-4, 0.], [1e-4, 0.]], 2)
        self.assertEqual(tiny.rank, 1)
        self.assertTrue(tiny.diagnostics["remaining_range_violation"])
        self.assertEqual(tiny.final_rank_request, 1)

    def test_numerically_zero_variance_is_not_given_an_invented_direction(self):
        local = fit_local_pca([[-1e-10, 0.], [1e-10, 0.]], 2)
        self.assertEqual(local.rank, 0)
        self.assertEqual(local.reason, "eigenvalue_threshold")
        self.assertGreater(local.eigenvalues[0], 0.)
        self.assertLess(local.eigenvalues[0], local.eigenvalue_threshold)

    def test_local_state_is_immutable_and_diagnostics_serializable(self):
        Q = np.array([[-1., 0.], [0., 2.], [2., -1.]])
        local = fit_local_pca(Q, 2)
        saved_mean = local.mu.copy()
        Q[:] = 100.
        np.testing.assert_array_equal(local.mu, saved_mean)
        for array in (local.mu, local.P, local.S, local.eigenvalues, local.projected_ranges):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            local.rank = 9
        json.dumps(local.diagnostics, allow_nan=False)

    def test_bad_rank_caps_and_references_are_rejected(self):
        for cap, error in ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(cap=cap), self.assertRaises(error):
                fit_local_pca([[0., 0.]], cap)
        with self.assertRaises(ValueError):
            fit_local_pca([[np.nan, 0.]], 1)


if __name__ == "__main__":
    unittest.main()

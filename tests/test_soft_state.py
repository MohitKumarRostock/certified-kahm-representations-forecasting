"""Formula and numerical-edge checks for the score/MEM/LOCAL layer."""

from dataclasses import FrozenInstanceError
import math
import unittest

import numpy as np

from kahm_demo.soft_state import (
    compose_encodings,
    partition_memberships,
    radial_angular_score,
)


class ReconstructionScoreTests(unittest.TestCase):
    def test_origin_convention(self):
        self.assertEqual(radial_angular_score([0, 0], [0, 0]),
                         1.0 / (2.0 * math.sqrt(2.0)))

    def test_identical_nonzero_vectors_have_zero_score(self):
        for vector in ([1, 0], [-3, -4], [1e308, -1e308], [1e-300, 2e-300]):
            with self.subTest(vector=vector):
                self.assertEqual(radial_angular_score(vector, vector), 0.0)

    def test_one_zero_vector_has_half_angle(self):
        expected = math.hypot(1.0 - math.exp(-1.0), 0.5) / math.sqrt(2.0)
        self.assertAlmostEqual(radial_angular_score([1, 0], [0, 0]), expected, places=15)
        self.assertAlmostEqual(radial_angular_score([0, 0], [1, 0]), expected, places=15)

    def test_orthogonal_and_opposite_vectors(self):
        orthogonal = math.hypot(1.0 - math.exp(-math.sqrt(2.0)), 0.5) / math.sqrt(2.0)
        opposite = math.hypot(1.0 - math.exp(-2.0), 1.0) / math.sqrt(2.0)
        self.assertAlmostEqual(radial_angular_score([1, 0], [0, 1]), orthogonal, places=15)
        self.assertAlmostEqual(radial_angular_score([1, 0], [-1, 0]), opposite, places=15)

    def test_score_uses_full_space_distance(self):
        # An orthogonal residual is still scored even if a local PCA/kernel
        # calculation could have ignored that coordinate.
        self.assertGreater(radial_angular_score([1, 2], [1, 0]),
                           radial_angular_score([1, 0], [1, 0]))

    def test_large_finite_vectors_do_not_overflow_angle(self):
        self.assertEqual(radial_angular_score([1e308, 0], [-1e308, 0]), 1.0)
        expected = math.hypot(1.0, 0.5) / math.sqrt(2.0)
        self.assertAlmostEqual(radial_angular_score([1e308, 0], [0, 1e308]),
                               expected, places=15)

    def test_expm1_preserves_tiny_radial_discrepancy(self):
        expected = -math.expm1(-1e-300) / math.sqrt(2.0)
        actual = radial_angular_score([1e-300, 0], [2e-300, 0])
        self.assertGreater(actual, 0.0)
        self.assertAlmostEqual(actual / expected, 1.0, places=15)

    def test_invalid_score_inputs(self):
        bad_pairs = [([], []), ([1, 2], [1]), ([[1, 2]], [[1, 2]]),
                     ([float("nan")], [0]), ([0], [float("inf")])]
        for z, reconstruction in bad_pairs:
            with self.subTest(z=z, reconstruction=reconstruction):
                with self.assertRaises(ValueError):
                    radial_angular_score(z, reconstruction)
        for bad in ([1 + 1j], [True], ["1"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    radial_angular_score(bad, [1])


class MembershipTests(unittest.TestCase):
    def test_partition_formula_and_normalizers(self):
        scores = np.array([0.1, 0.7, 0.8, 0.2, 0.9])
        result = partition_memberships(scores, (2, 3), offset=0.2, contrast=2)
        for start, stop in ((0, 2), (2, 5)):
            affinities = (1.0 - scores[start:stop] + 0.2) ** 2
            np.testing.assert_allclose(result.memberships[start:stop],
                                       affinities / affinities.sum(), rtol=2e-15, atol=0)
            self.assertAlmostEqual(float(result.memberships[start:stop].sum()), 1.0, places=15)
        self.assertAlmostEqual(float(result.memberships.sum()), 2.0, places=15)
        expected_logs = [math.log(float(((1.0 - scores[a:b] + 0.2) ** 2).sum()))
                         for a, b in ((0, 2), (2, 5))]
        np.testing.assert_allclose(result.log_normalizers, expected_logs, rtol=2e-15)

    def test_singleton_and_equal_score_partitions(self):
        result = partition_memberships([0.2, 1.0, 1.0, 1.0], (1, 3))
        self.assertEqual(result.memberships[0], 1.0)
        np.testing.assert_array_equal(result.memberships[1:], [1.0 / 3.0] * 3)

    def test_absolute_affinity_underflow_does_not_force_uniform_memberships(self):
        scores = np.array([0.99, 0.9899])
        offset, contrast = 1e-6, 200
        self.assertTrue(np.all((1.0 - scores + offset) ** contrast == 0.0))
        result = partition_memberships(scores, (2,), offset=offset, contrast=contrast)
        ratio = ((1.0 - scores[0] + offset) / (1.0 - scores[1] + offset)) ** contrast
        expected = np.array([ratio / (1.0 + ratio), 1.0 / (1.0 + ratio)])
        np.testing.assert_allclose(result.memberships, expected, rtol=2e-12, atol=0)
        self.assertLess(result.memberships[0], 0.2)
        self.assertGreater(result.memberships[1], 0.8)
        self.assertTrue(np.isfinite(result.log_normalizers).all())

    def test_underflowing_relative_weight_is_not_replaced_by_uniform(self):
        result = partition_memberships([0.1, 0.2], (2,), contrast=100000)
        np.testing.assert_array_equal(result.memberships, [1.0, 0.0])
        self.assertTrue(np.isfinite(result.log_normalizers).all())

    def test_zero_score_offset_extremes_still_normalize(self):
        result = partition_memberships([1.0, 1.0], (2,), offset=1e-300)
        np.testing.assert_array_equal(result.memberships, [0.5, 0.5])
        self.assertTrue(np.isfinite(result.log_normalizers).all())

    def test_result_arrays_are_readonly_and_detached(self):
        scores = np.array([0.1, 0.2])
        result = partition_memberships(scores, (2,))
        saved = result.memberships.copy()
        scores[:] = 1.0
        np.testing.assert_array_equal(result.memberships, saved)
        self.assertFalse(result.memberships.flags.writeable)
        self.assertFalse(result.log_normalizers.flags.writeable)
        with self.assertRaises(ValueError):
            result.memberships[0] = 1.0
        with self.assertRaises(FrozenInstanceError):
            result.memberships = np.ones(2)

    def test_invalid_scores_and_groups(self):
        for scores in ([], [[0.1]], [-0.1], [1.1], [float("nan")], [float("inf")]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    partition_memberships(scores, (1,))
        for groups in ((), (0, 2), (-1, 3), (1,), (3,)):
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    partition_memberships([0.1, 0.2], groups)
        for groups in ((True, 1), (1.0, 1), "2", 2):
            with self.subTest(groups=groups):
                with self.assertRaises(TypeError):
                    partition_memberships([0.1, 0.2], groups)

    def test_invalid_offset_and_contrast(self):
        for name in ("offset", "contrast"):
            for bad in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(ValueError):
                        partition_memberships([0.2], (1,), **{name: bad})
            for bad in (True, "1", 1 + 0j, None):
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(TypeError):
                        partition_memberships([0.2], (1,), **{name: bad})
        with self.assertRaises(ValueError):
            partition_memberships([1.0], (1,), offset=1e-300, contrast=1e308)


class EncodingTests(unittest.TestCase):
    def test_local_coordinates_use_input_and_local_means(self):
        z = np.array([3.0, 5.0])
        memberships = np.array([0.25, 0.75])
        means = np.array([[1.0, 2.0], [-1.0, 1.0]])
        bases = [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])]
        result = compose_encodings(z, memberships, means, bases)
        np.testing.assert_array_equal(result.mem, memberships)
        np.testing.assert_array_equal(result.local, [0.25, 0.75, 0.5, 3.0])
        np.testing.assert_array_equal(result.local[:2], result.mem)

    def test_signed_local_coordinates_and_basis_order(self):
        result = compose_encodings([2.0, -3.0], [1.0], [[1.0, 1.0]],
                                   [np.array([[0.0, -1.0], [1.0, 0.0]])])
        np.testing.assert_array_equal(result.local, [1.0, -4.0, -1.0])

    def test_rank_zero_models_append_nothing(self):
        bases = [np.empty((2, 0)), np.eye(2), np.empty((2, 0))]
        result = compose_encodings([2.0, 3.0], [0.25, 0.5, 0.25], np.zeros((3, 2)), bases)
        np.testing.assert_array_equal(result.local, [0.25, 0.5, 0.25, 1.0, 1.5])
        all_zero = compose_encodings([2.0, 3.0], [1.0], [[0.0, 0.0]],
                                     [np.empty((2, 0))])
        np.testing.assert_array_equal(all_zero.local, all_zero.mem)

    def test_multi_partition_membership_prefix_is_not_renormalized(self):
        memberships = partition_memberships([0.2, 0.3, 0.4], (1, 2)).memberships
        result = compose_encodings([1.0, 2.0], memberships, np.zeros((3, 2)),
                                   [np.eye(2)] * 3)
        np.testing.assert_array_equal(result.mem, memberships)
        np.testing.assert_array_equal(result.local[:3], memberships)
        self.assertAlmostEqual(float(result.mem.sum()), 2.0, places=15)

    def test_zero_membership_keeps_zero_local_block(self):
        result = compose_encodings([1.0, 2.0], [0.0, 1.0], np.zeros((2, 2)),
                                   [np.eye(2), np.eye(2)])
        np.testing.assert_array_equal(result.local, [0.0, 1.0, 0.0, 0.0, 1.0, 2.0])

    def test_output_immutability_and_input_detachment(self):
        phi = np.array([1.0])
        z = np.array([2.0, 3.0])
        result = compose_encodings(z, phi, [[0.0, 0.0]], [np.eye(2)])
        phi[:] = 0.0
        z[:] = 0.0
        np.testing.assert_array_equal(result.mem, [1.0])
        np.testing.assert_array_equal(result.local, [1.0, 2.0, 3.0])
        self.assertFalse(result.mem.flags.writeable)
        self.assertFalse(result.local.flags.writeable)
        with self.assertRaises(ValueError):
            result.local[0] = 0.0
        with self.assertRaises(FrozenInstanceError):
            result.local = np.zeros(3)

    def test_invalid_shapes_values_and_nonfinite_blocks(self):
        cases = [
            ([1, 2], [], [], []),
            ([1, 2], [-0.1], [[0, 0]], [np.eye(2)]),
            ([1, 2], [1.1], [[0, 0]], [np.eye(2)]),
            ([1, 2], [1.0], [[0]], [np.eye(2)]),
            ([1, 2], [1.0], [[0, 0]], []),
            ([1, 2], [1.0], [[0, 0]], [np.ones((3, 1))]),
            ([1, 2], [1.0], [[0, 0]], [np.ones((2, 3))]),
            ([1, 2], [1.0], [[0, float("nan")]], [np.eye(2)]),
            ([1, 2], [1.0], [[0, 0]], [np.array([[float("inf")], [0]])]),
            ([1e308, 0], [1.0], [[-1e308, 0]], [np.eye(2)]),
        ]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    compose_encodings(*args)


if __name__ == "__main__":
    unittest.main()

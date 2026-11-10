"""Scientific and numerical invariants of the matched feature construction."""

from dataclasses import fields
import json
import unittest
from unittest.mock import patch

import numpy as np

from kahm_demo.readout_features import (
    BANDWIDTH_MULTIPLIERS,
    ColumnScaler,
    ReadoutFeatures,
    _raw_nystrom_one,
    fit_column_scaler,
    fit_readout_features,
)


class ColumnScalingTests(unittest.TestCase):
    def test_ddof_zero_strict_floor_and_constant_columns_retained(self):
        X = np.array([[7.0, -0.5e-8, -1e-8, 1.0], [7.0, 0.5e-8, 1e-8, 3.0]])
        scaler = fit_column_scaler(X)
        np.testing.assert_array_equal(scaler.mean, [7.0, 0.0, 0.0, 2.0])
        np.testing.assert_array_equal(scaler.raw_std, [0.0, 0.5e-8, 1e-8, 1.0])
        np.testing.assert_array_equal(scaler.scale, [1.0, 1.0, 1e-8, 1.0])
        self.assertEqual(scaler.replaced_columns, (0, 1))
        self.assertEqual(scaler.transform(X).shape, X.shape)
        np.testing.assert_array_equal(scaler.transform(X)[:, 0], [0.0, 0.0])
        np.testing.assert_array_equal(scaler.transform(X)[:, 2], [-1.0, 1.0])

    def test_all_training_multiplicities_affect_moments(self):
        X = np.array([[0.0]] * 9 + [[1.0], [3.0], [3.0]])
        scaler = fit_column_scaler(X)
        self.assertEqual(scaler.sample_count, 12)
        self.assertEqual(scaler.mean[0], 7.0 / 12.0)
        self.assertNotEqual(scaler.mean[0], np.unique(X).mean())
        np.testing.assert_allclose(scaler.raw_std, np.std(X, axis=0, ddof=0), atol=0, rtol=0)

    def test_identical_membership_prefix_has_bitwise_identical_scaling(self):
        x = np.arange(37, dtype=np.float64)
        mem = np.column_stack((0.1 + x / 100.0, 0.9 - x / 100.0, np.full(37, 0.5)))
        local = np.column_stack((mem, x ** 2 + 0.125, np.sin(x), np.full(37, 9000.0)))
        mem_scaler = fit_column_scaler(mem)
        local_scaler = fit_column_scaler(local)
        for field in ("mean", "raw_std", "scale"):
            np.testing.assert_array_equal(getattr(mem_scaler, field), getattr(local_scaler, field)[:3])
        np.testing.assert_array_equal(mem_scaler.transform(mem), local_scaler.transform(local)[:, :3])


class NystromFeatureTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-2.0, 1.0], [-1.0, 0.25], [0.0, -0.5], [0.5, 0.7],
            [1.0, 1.2], [2.0, -1.0], [-1.0, 0.25], [0.5, 0.7],
        ])
        self.indices = (0, 2, 4, 5)

    @staticmethod
    def independent_raw_nystrom(state, row):
        u = (np.asarray(row) - state.linear_scaler.mean) / state.linear_scaler.scale
        kernel = np.array([
            np.exp(-float(np.dot(u - landmark, u - landmark)) / (2.0 * state.sigma_squared))
            for landmark in state.landmarks
        ])
        return (state.eigenvectors.T @ kernel) / np.sqrt(state.eigenvalues)

    def test_independent_spectral_formula_and_unchanged_linear_component(self):
        state = fit_readout_features(self.X, self.indices, 0.5)
        direct_gram = np.array([
            [np.exp(-np.dot(a - b, a - b) / (2 * state.sigma_squared)) for b in state.landmarks]
            for a in state.landmarks
        ])
        np.testing.assert_allclose(state.landmark_gram, direct_gram, rtol=3e-15, atol=3e-15)
        np.testing.assert_allclose(
            direct_gram @ state.eigenvectors,
            state.eigenvectors * state.eigenvalues[None, :], rtol=3e-13, atol=3e-13,
        )
        row = np.array([0.3, -0.2])
        raw_nystrom = self.independent_raw_nystrom(state, row)
        expected = np.concatenate((
            (row - state.linear_scaler.mean) / state.linear_scaler.scale,
            (raw_nystrom - state.nystrom_scaler.mean) / state.nystrom_scaler.scale,
        ))
        np.testing.assert_allclose(state.transform_one(row), expected, rtol=5e-13, atol=5e-13)
        np.testing.assert_array_equal(
            state.transform_one(row)[:state.input_width], state.linear_scaler.transform_one(row)
        )
        # Coordinate-free check against an independent truncated Gram inverse.
        vals, vecs = np.linalg.eigh(direct_gram)
        keep = vals > max(1e-12, 1e-10 * vals[-1])
        inverse = (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T
        u = state.linear_scaler.transform_one(row)
        kernel = np.exp(-np.sum((state.landmarks - u) ** 2, axis=1) / (2 * state.sigma_squared))
        self.assertAlmostEqual(float(raw_nystrom @ raw_nystrom), float(kernel @ inverse @ kernel), places=11)

    def test_full_and_reduced_landmark_ranks(self):
        X = np.arange(5, dtype=float)[:, None]
        full = fit_readout_features(X, tuple(range(5)), 0.25)
        self.assertEqual(full.nystrom_rank, 5)
        constant = fit_readout_features(np.full((8, 3), 7.0), (0, 1, 2, 3), 1.0)
        self.assertEqual(constant.landmark_count, 4)
        self.assertEqual(constant.nystrom_rank, 1)
        self.assertEqual(constant.output_width, 4)
        np.testing.assert_array_equal(constant.transform(np.full((8, 3), 7.0)), np.zeros((8, 4)))

    def test_coincident_encoded_landmarks_retain_row_identities(self):
        X = np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [2.0, 3.0]])
        state = fit_readout_features(X, (0, 1, 2, 3), 1.0)
        self.assertEqual(state.landmark_indices, (0, 1, 2, 3))
        self.assertEqual(state.landmark_count, 4)
        self.assertEqual(state.nystrom_rank, 2)
        self.assertEqual(state.diagnostics["encoded_landmark_distinct_count"], 2)

    def test_bandwidth_uses_standardized_landmarks_and_declared_grid(self):
        for multiplier in BANDWIDTH_MULTIPLIERS:
            state = fit_readout_features(self.X, self.indices, multiplier)
            positive = [
                float(np.sum((state.landmarks[i] - state.landmarks[j]) ** 2))
                for i in range(state.landmark_count) for j in range(i + 1, state.landmark_count)
                if float(np.sum((state.landmarks[i] - state.landmarks[j]) ** 2)) > 1e-14
            ]
            delta = float(np.median(positive))
            self.assertEqual(state.median_squared_distance, delta)
            self.assertEqual(state.sigma_squared, max(delta * multiplier ** 2, 1e-12))
        for invalid in (0.0, 3.0, np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_readout_features(self.X, self.indices, invalid)
        with self.assertRaises(TypeError):
            fit_readout_features(self.X, self.indices, True)

    def test_distance_cutoff_fallback_and_bandwidth_floor(self):
        near = np.array([[0.0], [1.0], [1.0 + 1e-8], [10.0]])
        state = fit_readout_features(near, (1, 2), 0.25)
        self.assertEqual(state.diagnostics["positive_distance_pair_count"], 0)
        self.assertTrue(state.diagnostics["median_distance_fallback_used"])
        self.assertEqual(state.median_squared_distance, 1.0)
        self.assertEqual(state.sigma_squared, 0.25 ** 2)
        small = np.array([[0.0], [1.0], [1.0 + 1e-5], [10.0]])
        floored = fit_readout_features(small, (1, 2), 0.25)
        self.assertGreater(floored.median_squared_distance, 1e-14)
        self.assertLess(floored.median_squared_distance * 0.25 ** 2, 1e-12)
        self.assertEqual(floored.sigma_squared, 1e-12)

    def test_nystrom_moments_include_all_training_occurrences(self):
        X = np.array([[0.0, 0.0]] * 9 + [[1.0, 1.0], [3.0, 9.0], [3.0, 9.0]])
        state = fit_readout_features(X, (0, 9, 10), 1.0)
        raw = np.stack([self.independent_raw_nystrom(state, row) for row in X])
        np.testing.assert_allclose(state.nystrom_scaler.mean, raw.mean(axis=0), rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(state.nystrom_scaler.raw_std, raw.std(axis=0, ddof=0), rtol=3e-14, atol=3e-14)
        self.assertFalse(np.allclose(state.nystrom_scaler.mean, raw[[0, 9, 10]].mean(axis=0)))
        self.assertEqual(state.nystrom_scaler.sample_count, len(X))

    def test_saved_nystrom_moments_exactly_match_frozen_inference_operands(self):
        # This regression is sensitive to C/F memory layout: an earlier fit
        # computed moments with Fortran-order retained eigenvectors, then copied
        # them to C order in the returned state. Matrix-vector roundoff changed
        # even though ordinary allclose checks passed.
        rng = np.random.default_rng(198443)
        X = rng.normal(size=(128, 7))
        X[80:] = X[:48]  # Keep every duplicated training occurrence in moments.
        for multiplier in BANDWIDTH_MULTIPLIERS:
            with self.subTest(multiplier=multiplier):
                state = fit_readout_features(X, tuple(range(16)), multiplier)
                for array in (state.landmarks, state.eigenvalues, state.eigenvectors):
                    self.assertTrue(array.flags.c_contiguous)
                    self.assertFalse(array.flags.writeable)
                standardized = state.linear_scaler.transform(X)
                raw = np.stack([
                    _raw_nystrom_one(row, state.landmarks, state.sigma_squared,
                                     state.eigenvalues, state.eigenvectors)
                    for row in standardized
                ])
                recomputed = fit_column_scaler(raw)
                for name in ("mean", "raw_std", "scale"):
                    np.testing.assert_array_equal(
                        getattr(state.nystrom_scaler, name), getattr(recomputed, name)
                    )
                self.assertEqual(state.nystrom_scaler.replaced_columns,
                                 recomputed.replaced_columns)

    def test_mem_prefix_scaling_and_values_remain_identical_in_local(self):
        a = np.arange(19, dtype=float)
        mem = np.column_stack((0.2 + a / 100, 0.8 - a / 100))
        local = np.column_stack((mem, np.sin(a), a ** 2))
        indices = (0, 5, 10, 18)
        mem_state = fit_readout_features(mem, indices, 1.0)
        local_state = fit_readout_features(local, indices, 1.0)
        for field in ("mean", "raw_std", "scale"):
            np.testing.assert_array_equal(
                getattr(mem_state.linear_scaler, field), getattr(local_state.linear_scaler, field)[:2]
            )
        np.testing.assert_array_equal(mem_state.transform(mem)[:, :2], local_state.transform(local)[:, :2])

    def test_batch_matches_canonical_rows_bit_for_bit_and_preserves_duplicates(self):
        state = fit_readout_features(self.X, self.indices, 2.0)
        rows = self.X[[5, 0, 5, 3, 1, 3, 0]]
        batch = state.transform(rows)
        canonical = np.stack([state.transform_one(row) for row in rows])
        np.testing.assert_array_equal(batch, canonical)
        self.assertEqual(len(batch), len(rows))
        np.testing.assert_array_equal(batch[0], batch[2])
        np.testing.assert_array_equal(state.transform(rows[::-1])[::-1], batch)
        self.assertEqual(state.transform(np.empty((0, state.input_width))).shape, (0, state.output_width))

    def test_query_does_not_refit_and_fitted_arrays_are_immutable(self):
        source = self.X.copy()
        state = fit_readout_features(source, self.indices, 1.0)
        expected = state.transform_one([0.3, -0.7])
        source[:] = 5000
        with patch("kahm_demo.readout_features.fit_column_scaler", side_effect=AssertionError("refit")), \
             patch("kahm_demo.readout_features.np.linalg.eigh", side_effect=AssertionError("refit")):
            np.testing.assert_array_equal(state.transform_one([0.3, -0.7]), expected)
            state.transform([[100.0, -50.0], [0.3, -0.7]])
        for array in (state.landmarks, state.landmark_gram, state.eigenvalues, state.eigenvectors,
                      state.linear_scaler.mean, state.linear_scaler.scale,
                      state.nystrom_scaler.mean, state.nystrom_scaler.raw_std):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        json.dumps(state.diagnostics, allow_nan=False)

    def test_bad_inputs_and_landmark_indices_are_rejected(self):
        for invalid in (np.array([[np.nan, 1.0]]), np.array([[np.inf, 1.0]]),
                        np.empty((0, 2)), np.empty((3, 0)), np.array([1.0, 2.0])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_readout_features(invalid, (0,), 1.0)
        for invalid in (np.array([[1j]]), np.array([["1"]]), np.array([[True]])):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                fit_readout_features(invalid, (0,), 1.0)
        for indices in ((), (0, 0), (-1, 0), (0, 100), np.array([[0, 1]])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                fit_readout_features(self.X, indices, 1.0)
        for indices in ((True,), (0.0,), "0"):
            with self.subTest(indices=indices), self.assertRaises(TypeError):
                fit_readout_features(self.X, indices, 1.0)
        state = fit_readout_features(self.X, np.array(self.indices, dtype=np.int64), 1.0)
        for query in ([np.nan, 0.0], [np.inf, 0.0], [1.0], [[1.0, 2.0]]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                state.transform_one(query)

    def test_saved_dataclass_state_reconstructs_without_refitting(self):
        state = fit_readout_features(self.X, self.indices, 1.0)
        scaler = ColumnScaler(**{field.name: getattr(state.linear_scaler, field.name) for field in fields(ColumnScaler)})
        values = {field.name: getattr(state, field.name) for field in fields(ReadoutFeatures)}
        values["linear_scaler"] = scaler
        restored = ReadoutFeatures(**values)
        np.testing.assert_array_equal(restored.transform(self.X), state.transform(self.X))


if __name__ == "__main__":
    unittest.main()

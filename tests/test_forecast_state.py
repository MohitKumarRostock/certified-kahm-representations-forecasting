"""Complete-forecaster storage checks using private deterministic fit fixtures."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kahm_demo._util import canonical_json_bytes, read_json, write_json
from kahm_demo.contract import CONTRACT_SHA256
from kahm_demo.encoder import fit_encoder
from kahm_demo import encoder_state
from kahm_demo.forecast import CANONICAL_FORECAST_METHOD, ForecastModel
from kahm_demo.forecast_state import (
    FORECAST_CANDIDATE_STATUS, FORECAST_SCHEMA, FORECAST_STATUS, forecaster_state_fingerprint,
    load_forecaster, save_forecaster, validate_forecaster,
)
from kahm_demo.geometry import fit_standardizer
from kahm_demo.readout_features import fit_readout_features
from kahm_demo.ridge import fit_ridge


def _digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _refresh_hashes(record, *, state=True):
    if state:
        record["state_sha256"] = _digest(record["state"])
    record["record_sha256"] = _digest({
        key: value for key, value in record.items() if key != "record_sha256"
    })


def _fit_models(X, y, landmarks, *, cap=1):
    standardizer = fit_standardizer(X)
    encoder = fit_encoder(X, cap=cap, requested_groups=(2, 4), reference_budget=24)
    models = {}
    for candidate in ("raw", "kahm_mem", "kahm_local"):
        representation = (standardizer.transform(X) if candidate == "raw"
                          else encoder.transform(X, candidate))
        readout = fit_readout_features(representation, landmarks, 1.0)
        ridge = fit_ridge(readout.transform(representation), y, 0.01)
        models[candidate] = ForecastModel(
            candidate, None if candidate == "raw" else f"cap_{cap}",
            standardizer if candidate == "raw" else None,
            None if candidate == "raw" else encoder, readout, ridge,
        )
    return models


class ForecastStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is an explicit testing design, not a simulator draw or population
        # oracle evaluation. The target pattern is private to these tests.
        cls.histories = np.asarray([(a, b) for a in (0., .25, .75, 1.)
                                   for b in (0., .25, .75, 1.)])
        X = np.concatenate((cls.histories, cls.histories[::-1], cls.histories))
        pattern = np.asarray([.15, .8, .45, .25, .7, .1, .65, .35,
                              .2, .6, .9, .4, .75, .3, .55, .05])
        y = np.concatenate((pattern, pattern[::-1], pattern))
        cls.models = _fit_models(X, y, tuple(range(16)))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def _record(self, candidate="raw"):
        return save_forecaster(self.models[candidate], self.directory / "original.json",
                               provenance={"fixture": "private", "count": 48})

    def _load_modified(self, record):
        path = self.directory / "modified.json"
        write_json(path, record)
        return load_forecaster(path)

    def test_save_load_all_three_actual_predictions_are_bitwise_preserved(self):
        for candidate, model in self.models.items():
            with self.subTest(candidate=candidate):
                path = self.directory / f"{candidate}.json"
                record = save_forecaster(model, path, provenance={"fixture": candidate})
                loaded = load_forecaster(path)
                self.assertEqual(record["state_sha256"], forecaster_state_fingerprint(model))
                self.assertEqual(forecaster_state_fingerprint(model),
                                 forecaster_state_fingerprint(loaded))
                self.assertEqual(record, read_json(path))
                self.assertEqual(model.predict(self.histories).tobytes(),
                                 loaded.predict(self.histories).tobytes())
                for history in self.histories:
                    before, after = model.inspect_one(history), loaded.inspect_one(history)
                    self.assertEqual(before.prediction.hex(), after.prediction.hex())
                    self.assertEqual(before.unclipped_prediction.hex(),
                                     after.unclipped_prediction.hex())
                    self.assertEqual(before.clipped, after.clipped)
                    self.assertEqual(before.encoder_fallbacks, after.encoder_fallbacks)
                with self.assertRaises(ValueError):
                    loaded.ridge.coefficients.setflags(write=True)
                with self.assertRaises(ValueError):
                    loaded.readout.landmarks.setflags(write=True)

    def test_artifact_binds_contract_runtime_status_and_provenance(self):
        record = self._record()
        self.assertEqual(record["schema"], FORECAST_SCHEMA)
        self.assertEqual(record["status"], FORECAST_STATUS)
        self.assertEqual(record["contract_sha256"], CONTRACT_SHA256)
        self.assertEqual(record["runtime"]["forecast_canonical_method"],
                         CANONICAL_FORECAST_METHOD)
        self.assertEqual(record["provenance"], {"fixture": "private", "count": 48})
        self.assertEqual(record["record_sha256"], _digest({
            key: value for key, value in record.items() if key != "record_sha256"
        }))
        record["provenance"]["fixture"] = "changed"
        with self.assertRaisesRegex(ValueError, "record hash"):
            self._load_modified(record)

    def test_both_development_statuses_roundtrip_and_unknown_status_is_rejected(self):
        for status in (FORECAST_STATUS, FORECAST_CANDIDATE_STATUS):
            with self.subTest(status=status):
                path = self.directory / f"{status}.json"
                record = save_forecaster(self.models["raw"], path, status=status)
                self.assertEqual(record["status"], status)
                loaded = load_forecaster(path)
                self.assertEqual(forecaster_state_fingerprint(loaded), record["state_sha256"])
        for status in ("frozen", "development_unknown", None):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "development status"):
                save_forecaster(self.models["raw"], self.directory / "invalid.json", status=status)
        record = read_json(self.directory / f"{FORECAST_CANDIDATE_STATUS}.json")
        record["status"] = "development_unknown"
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "development status"):
            self._load_modified(record)

    def test_wrong_required_metadata_rejected_even_with_recomputed_hash(self):
        original = self._record()
        modifications = (
            ("schema", "kahm.forecast_model.v0"), ("status", "frozen"),
            ("contract_sha256", "0" * 64), ("package_version", ""),
        )
        for key, value in modifications:
            record = copy.deepcopy(original)
            record[key] = value
            _refresh_hashes(record)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._load_modified(record)
        for key in ("canonical_method", "forecast_canonical_method"):
            record = copy.deepcopy(original)
            record["runtime"][key] = "unsupported"
            _refresh_hashes(record)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._load_modified(record)

    def test_recorded_package_and_runtime_need_not_match_installed_versions(self):
        record = self._record()
        record["package_version"] = "0.1.0.dev4"
        record["runtime"]["numpy_version"] = "recorded-runtime-differs"
        _refresh_hashes(record)
        loaded = self._load_modified(record)
        self.assertEqual(forecaster_state_fingerprint(loaded), record["state_sha256"])

    def test_state_hash_checked_independently_of_envelope_hash(self):
        record = self._record()
        record["state_sha256"] = "0" * 64
        _refresh_hashes(record, state=False)
        with self.assertRaisesRegex(ValueError, "state hash"):
            self._load_modified(record)

    def test_altered_gaussian_gram_rejected_after_both_hashes_recomputed(self):
        record = self._record()
        gram = record["state"]["fields"]["readout"]["fields"]["landmark_gram"]
        gram["values"][1] = .125.hex()
        gram["values"][16] = .125.hex()
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "Gaussian kernel"):
            self._load_modified(record)

    def test_altered_scaler_rule_rejected_after_both_hashes_recomputed(self):
        record = self._record()
        scaler = record["state"]["fields"]["readout"]["fields"]["linear_scaler"]["fields"]
        scaler["scale"]["values"][0] = 2.0.hex()
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "scales|replacement"):
            self._load_modified(record)

    def test_invalid_raw_input_scaler_is_checked(self):
        model = self.models["raw"]
        malformed = replace(model.raw_standardizer, scale=model.raw_standardizer.scale * 2)
        with self.assertRaisesRegex(ValueError, "replacement"):
            validate_forecaster(replace(model, raw_standardizer=malformed))

    def test_noncanonical_or_nonfinite_float_payloads_rejected(self):
        original = self._record()
        for value in ("0x1p-1", "nan", "inf", "0x1p+999999", .5):
            record = copy.deepcopy(original)
            record["state"]["fields"]["ridge"]["fields"]["intercept"] = {
                "_type": "float", "hex": value,
            }
            _refresh_hashes(record)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._load_modified(record)
        record = copy.deepcopy(original)
        record["state"]["fields"]["ridge"]["fields"]["intercept"] = .5
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "untagged"):
            self._load_modified(record)

    def test_landmark_ids_and_saved_distance_rule_are_checked(self):
        original = self._record()
        for indices in ([0] * 16, list(range(1, 17))[:-1] + [48]):
            record = copy.deepcopy(original)
            readout = record["state"]["fields"]["readout"]["fields"]
            readout["landmark_indices"]["items"] = indices
            _refresh_hashes(record)
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self._load_modified(record)
        record = copy.deepcopy(original)
        readout = record["state"]["fields"]["readout"]["fields"]
        readout["median_squared_distance"]["hex"] = 7.0.hex()
        readout["sigma_squared"]["hex"] = 7.0.hex()
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "distance median"):
            self._load_modified(record)

    def test_eigenpairs_and_omitted_nystrom_directions_are_checked(self):
        model = self.models["raw"]
        changed_vectors = np.array(model.readout.eigenvectors)
        changed_vectors[:, 0] *= .9
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_forecaster(replace(model, readout=replace(
                model.readout, eigenvectors=changed_vectors)))
        # Remove a genuinely retained direction while making all public widths
        # agree. Shape checking alone would accept this truncated fake state.
        readout = model.readout
        self.assertGreater(readout.nystrom_rank, 1)
        new_scaler = replace(
            readout.nystrom_scaler, mean=readout.nystrom_scaler.mean[:-1],
            scale=readout.nystrom_scaler.scale[:-1], raw_std=readout.nystrom_scaler.raw_std[:-1],
            replaced_columns=tuple(index for index in readout.nystrom_scaler.replaced_columns
                                   if index < readout.nystrom_rank - 1),
        )
        shortened = replace(readout, eigenvalues=readout.eigenvalues[:-1],
                            eigenvectors=readout.eigenvectors[:, :-1],
                            nystrom_scaler=new_scaler)
        ridge = replace(model.ridge, coefficients=model.ridge.coefficients[:-1],
                        feature_mean=model.ridge.feature_mean[:-1])
        with self.assertRaisesRegex(ValueError, "truncation"):
            validate_forecaster(replace(model, readout=shortened, ridge=ridge))

    def test_ridge_intercept_mean_relation_is_checked(self):
        model = self.models["raw"]
        changed = replace(model.ridge, intercept=model.ridge.intercept + .1)
        with self.assertRaisesRegex(ValueError, "mean relation"):
            validate_forecaster(replace(model, ridge=changed))

    def test_linked_training_counts_and_bank_id_are_checked(self):
        original = self._record("kahm_mem")
        record = copy.deepcopy(original)
        record["state"]["fields"]["bank_id"] = "cap_99"
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "bank identity"):
            self._load_modified(record)
        record = copy.deepcopy(original)
        scaler = record["state"]["fields"]["encoder"]["fields"]["standardizer"]["fields"]
        scaler["sample_count"] = 47
        _refresh_hashes(record)
        with self.assertRaisesRegex(ValueError, "training count"):
            self._load_modified(record)

    def test_rank_zero_encoder_and_constant_readout_features_roundtrip(self):
        X = np.tile([.25, .75], (8, 1))
        models = _fit_models(X, np.arange(8, dtype=float) / 7, (0,))
        for candidate, model in models.items():
            with self.subTest(candidate=candidate):
                if candidate != "raw":
                    self.assertTrue(all(local.pca.rank == 0 for local in model.encoder.models))
                self.assertTrue(np.all(model.readout.linear_scaler.raw_std == 0))
                self.assertTrue(np.all(model.readout.nystrom_scaler.raw_std == 0))
                path = self.directory / f"constant_{candidate}.json"
                save_forecaster(model, path)
                loaded = load_forecaster(path)
                self.assertEqual(model.predict(self.histories).tobytes(),
                                 loaded.predict(self.histories).tobytes())

    def test_loading_does_not_call_any_fit_function_or_change_encoder_registry(self):
        path = self.directory / "selected.json"
        save_forecaster(self.models["kahm_local"], path)
        classes_before = dict(encoder_state._CLASSES)
        with (patch("kahm_demo.encoder.fit_encoder", side_effect=AssertionError("refit")),
              patch("kahm_demo.geometry.fit_standardizer", side_effect=AssertionError("refit")),
              patch("kahm_demo.readout_features.fit_readout_features", side_effect=AssertionError("refit")),
              patch("kahm_demo.ridge.fit_ridge", side_effect=AssertionError("refit"))):
            loaded = load_forecaster(path)
        self.assertEqual(encoder_state._CLASSES, classes_before)
        self.assertEqual(forecaster_state_fingerprint(loaded),
                         forecaster_state_fingerprint(self.models["kahm_local"]))

    def test_provenance_floats_are_lossless_and_output_is_not_overwritten(self):
        path = self.directory / "model.json"
        record = save_forecaster(self.models["raw"], path, provenance={"number": .1})
        self.assertEqual(record["provenance"]["number"], {"_type": "float", "hex": .1.hex()})
        load_forecaster(path)
        with self.assertRaises(FileExistsError):
            save_forecaster(self.models["raw"], path)


if __name__ == "__main__":
    unittest.main()

"""Complete matched development search and auditable three-model selection.

Only training and validation episodes enter this stage. Exact empirical loss
uses the observed episode counts, never the benchmark population probabilities.
The output is selected development state, not the final model freeze.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import __version__
from ._util import canonical_json_bytes, checked_relative_file, read_json, require_hash, sha256_file, write_json
from .contract import CONTRACT_SHA256, validate_contract
from .datasets import load_development
from .encoder_state import encoder_state_fingerprint, load_encoder, runtime_record
from .encoder_workflow import SOURCE_FILES as ENCODER_SOURCE_FILES, verify_encoder_bundle
from .forecast import CANONICAL_FORECAST_METHOD, ForecastModel
from .forecast_state import (forecaster_state_fingerprint, load_forecaster,
                             save_forecaster, validate_forecaster, _pack as _pack_forecast)
from .law import HISTORY_KEYS, OBSERVATION_FLOATS
from .oracle import parse_prediction_table, prediction_table_from_mapping
from .readout_features import fit_column_scaler, fit_readout_features, _raw_nystrom_one
from .ridge import fit_ridge_grid
from .selection import (empirical_counts, empirical_mse_from_counts, fraction_record,
                        parse_fraction_record, select_landmarks, select_trial_records)

READOUT_SPEC_SHA256 = "1fc29f4dec462e65efc7312c20b33c89b4ec1b9ea003ce508a459711c69a71f9"
FORECAST_BUNDLE_SCHEMA = "kahm.forecast_development_bundle.v1"
SEARCH_SCHEMA = "kahm.forecast_search.v1"
SOURCE_FILES = tuple(ENCODER_SOURCE_FILES) + (
    "readout_features.py", "ridge.py", "selection.py", "forecast.py",
    "forecast_state.py", "forecast_workflow.py", "readout_specification.json")


def load_readout_specification() -> dict:
    raw = resources.files("kahm_demo").joinpath("readout_specification.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != READOUT_SPEC_SHA256:
        raise ValueError("readout specification differs from the fixed readout development menu")
    return json.loads(raw)


def forecast_source_identities() -> dict:
    base = Path(__file__).parent
    return {name: sha256_file(base / name) for name in SOURCE_FILES}


def forecast_runtime_record() -> dict:
    return {**runtime_record(), "forecast_canonical_method": CANONICAL_FORECAST_METHOD}


def _data_identity(bundle: Path) -> dict:
    manifest = read_json(bundle / "manifest.json")
    return {"run_id": manifest["run_id"], "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "training_data_sha256": manifest["development"]["training"]["sha256"],
            "validation_data_sha256": manifest["development"]["validation"]["sha256"],
            "training_episodes": manifest["development"]["training"]["episode_count"],
            "validation_episodes": manifest["development"]["validation"]["episode_count"]}


def _expected_roster(spec: dict) -> list[dict]:
    procedures = [("raw", None)] + [(candidate, bank) for bank in spec["encoder_bank_order"]
                                    for candidate in ("kahm_mem", "kahm_local")]
    roster = []
    for candidate, bank in procedures:
        for bi, bandwidth in enumerate(spec["bandwidth_multipliers"]):
            for ai, alpha in enumerate(spec["ridge_alphas"]):
                name = candidate if bank is None else f"{candidate}__{bank}"
                roster.append({"trial_id": f"{name}__b{bi:02d}_a{ai:02d}",
                               "candidate_id": candidate, "bank_id": bank,
                               "grid_index": bi * len(spec["ridge_alphas"]) + ai,
                               "bandwidth_multiplier": bandwidth, "ridge_alpha": alpha})
    return roster


def _load_banks(encoders: Path, bundle: Path, spec: dict) -> tuple[dict, dict]:
    checked = verify_encoder_bundle(encoders, dataset_bundle=bundle)
    if not checked["numerical_source_identity_matches"]:
        raise ValueError("encoder numerical sources differ; resolve development provenance before fitting")
    manifest = read_json(encoders / "manifest.json")
    if manifest["bank_order"] != spec["encoder_bank_order"]:
        raise ValueError("encoder bank menu differs from the fixed readout specification")
    banks = {}
    for record in manifest["banks"]:
        banks[record["bank_id"]] = load_encoder(checked_relative_file(encoders, record["state_path"]))
    return banks, manifest


def _support_inputs() -> np.ndarray:
    return np.asarray([(OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b]) for a, b in HISTORY_KEYS])


def _clip_record(unclipped, training_counts, validation_counts) -> dict:
    indices = [index for index, value in enumerate(unclipped) if value < 0.0 or value > 1.0]
    histories = {HISTORY_KEYS[index] for index in indices}
    return {"support_history_ids": indices,
            "training_episodes": sum(count for (a, b, _), count in training_counts.items() if (a, b) in histories),
            "validation_episodes": sum(count for (a, b, _), count in validation_counts.items() if (a, b) in histories)}


def _selected_trial_ids(selection: dict) -> dict:
    return {"raw": selection["raw_trial_id"], "kahm_mem": selection["mem_trial_id"],
            "kahm_local": selection["local_trial_id"]}


def _model_provenance(data_identity: dict, encoder_manifest_hash: str, trial_id: str) -> dict:
    return {"data_identity": data_identity, "encoder_bundle_manifest_sha256": encoder_manifest_hash,
            "readout_specification_sha256": READOUT_SPEC_SHA256, "selected_trial_id": trial_id,
            "fitting_role": "training", "selection_role": "validation",
            "refit_on_training_plus_validation": False, "calibration_used": False,
            "oracle_population_risk_used": False}


def _candidate_provenance(data_identity: dict, encoder_manifest_hash: str, trial_id: str) -> dict:
    record = _model_provenance(data_identity, encoder_manifest_hash, trial_id)
    record["trial_id"] = record.pop("selected_trial_id")
    record["artifact_role"] = "development_search_candidate"
    return record


def _complete_predictions(model: ForecastModel, identity: str) -> tuple[dict, dict]:
    values, unclipped, fallbacks = {}, [], []
    for index, (history, x) in enumerate(zip(HISTORY_KEYS, _support_inputs(), strict=True)):
        result = model.inspect_one(x)
        values[history] = result.prediction
        unclipped.append(result.unclipped_prediction.hex())
        if result.encoder_fallbacks:
            fallbacks.append({"history_id": index, "events": list(result.encoder_fallbacks)})
    return prediction_table_from_mapping(values, identity), {
        "unclipped_prediction_hex": unclipped,
        "encoder_fallback_events": fallbacks,
        "all_support_predictions_finite_and_bounded": True}


def _summary(model: ForecastModel, trial: dict) -> dict:
    return {"candidate_id": model.candidate_id, "bank_id": model.bank_id,
            "selected_trial_id": trial["trial_id"],
            "bandwidth_multiplier": model.readout.bandwidth_multiplier,
            "ridge_alpha": model.ridge.alpha,
            "representation_width": model.readout.input_width,
            "nystrom_rank": model.readout.nystrom_rank, "decoder_width": model.readout.output_width,
            "landmark_count": model.readout.landmark_count,
            "training_mse": trial["training_mse"], "validation_mse": trial["validation_mse"],
            "clipping": trial["clipping"]}


def fit_development_forecasters(bundle: Path, encoders: Path, output: Path) -> dict:
    validate_contract()
    spec = load_readout_specification()
    bundle, encoders, output = Path(bundle), Path(encoders), Path(output)
    if output.exists():
        raise FileExistsError(f"forecast output already exists: {output}; choose a fresh directory")
    if (bundle / "calibration").exists():
        raise ValueError("calibration has already been materialized; this development run requires an untouched reservation")
    banks, encoder_manifest = _load_banks(encoders, bundle, spec)
    training = load_development(bundle, "training")
    validation = load_development(bundle, "validation")
    x, y = training.xy()
    raw_training = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    training_counts, validation_counts = empirical_counts(training.episodes), empirical_counts(validation.episodes)
    identity = _data_identity(bundle)
    encoder_manifest_hash = sha256_file(encoders / "manifest.json")
    standardizer = banks[spec["encoder_bank_order"][0]].standardizer
    landmark_plan = select_landmarks(raw_training, standardizer, budget=spec["landmark_budget"])
    raw_support = _support_inputs()
    procedures = [("raw", None)] + [(candidate, bank) for bank in spec["encoder_bank_order"]
                                    for candidate in ("kahm_mem", "kahm_local")]
    representations = {}
    for candidate, bank_id in procedures:
        if candidate == "raw":
            representations[(candidate, bank_id)] = (standardizer.transform(raw_training), standardizer.transform(raw_support))
        else:
            bank = banks[bank_id]
            representations[(candidate, bank_id)] = (bank.transform(raw_training, candidate), bank.transform(raw_support, candidate))
    roster = _expected_roster(spec)
    records, fitted = [], {}
    membership_scalers = {}
    for candidate, bank_id in procedures:
        representation, support = representations[(candidate, bank_id)]
        for bi, bandwidth in enumerate(spec["bandwidth_multipliers"]):
            readout = fit_readout_features(representation, landmark_plan.row_indices, bandwidth)
            if candidate == "kahm_mem":
                membership_scalers[(bank_id, bi)] = readout.linear_scaler
            elif candidate == "kahm_local":
                shared = membership_scalers[(bank_id, bi)]
                width = shared.input_width
                if any(not np.array_equal(getattr(shared, name), getattr(readout.linear_scaler, name)[:width])
                       for name in ("mean", "scale", "raw_std")):
                    raise ValueError("MEM and LOCAL membership feature scaling differs")
            features = readout.transform(representation)
            support_features = readout.transform(support)
            ridges = fit_ridge_grid(features, y, spec["ridge_alphas"])
            for ai, ridge in enumerate(ridges):
                expected = roster[len(records)]
                if (expected["candidate_id"], expected["bank_id"], expected["grid_index"]) != (candidate, bank_id, bi * 9 + ai):
                    raise ValueError("internal search order differs from the declared roster")
                model = ForecastModel(candidate, bank_id, standardizer if candidate == "raw" else None,
                                      None if candidate == "raw" else banks[bank_id], readout, ridge)
                unclipped = ridge.predict_unclipped(support_features)
                predictions = {history: float(value) for history, value in zip(HISTORY_KEYS, ridge.predict(support_features), strict=True)}
                table = prediction_table_from_mapping(predictions, expected["trial_id"])
                record = {**expected,
                          "representation_width": readout.input_width, "decoder_width": readout.output_width,
                          "nystrom_rank": readout.nystrom_rank, "landmark_count": readout.landmark_count,
                          "sigma_squared": readout.sigma_squared,
                          "training_mse": fraction_record(empirical_mse_from_counts(training_counts, predictions)),
                          "validation_mse": fraction_record(empirical_mse_from_counts(validation_counts, predictions)),
                          "prediction_table": table,
                          "unclipped_prediction_hex": [float(value).hex() for value in unclipped],
                          "clipping": _clip_record(unclipped, training_counts, validation_counts),
                          "readout_diagnostics": readout.diagnostics,
                          "ridge_diagnostics": ridge.diagnostics}
                records.append(record)
                fitted[expected["trial_id"]] = model
    if len(records) != spec["expected_search_counts"]["total"]:
        raise ValueError("incomplete decoder search")
    selection = select_trial_records(records, spec["encoder_bank_order"])
    by_id = {record["trial_id"]: record for record in records}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kahm-forecasters-", dir=output.parent))
    try:
        spec_bytes = resources.files("kahm_demo").joinpath("readout_specification.json").read_bytes()
        (stage / "readout_specification.json").write_bytes(spec_bytes)
        write_json(stage / "landmarks.json", landmark_plan.record())
        (stage / "search_states").mkdir()
        for record in records:
            trial_id = record["trial_id"]
            relative = f"search_states/{trial_id}.json"
            model = fitted[trial_id]
            save_forecaster(model, stage / relative,
                            provenance=_candidate_provenance(identity, encoder_manifest_hash, trial_id),
                            status="development_candidate")
            record["model_state_path"] = relative
            record["model_state_file_sha256"] = sha256_file(stage / relative)
            record["model_state_sha256"] = forecaster_state_fingerprint(model)
        search = {"schema": SEARCH_SCHEMA, "readout_specification_sha256": READOUT_SPEC_SHA256,
                  "data_identity": identity, "records": records, "selection": selection,
                  "loss_basis": "exact empirical episode counts and actual clipped float predictions",
                  "population_risk_used": False, "calibration_used": False}
        write_json(stage / "search.json", search)
        selected_records = []
        for candidate, trial_id in _selected_trial_ids(selection).items():
            model, trial = fitted[trial_id], by_id[trial_id]
            model_path, table_path = f"{candidate}.json", f"{candidate}_predictions.json"
            provenance = _model_provenance(identity, encoder_manifest_hash, trial_id)
            save_forecaster(model, stage / model_path, provenance=provenance)
            table, inference = _complete_predictions(model, candidate)
            selected_values = parse_prediction_table(table)
            search_values = parse_prediction_table(trial["prediction_table"])
            if any(selected_values[history].hex() != search_values[history].hex() for history in HISTORY_KEYS):
                raise ValueError("complete forecaster inference differs from development search prediction")
            if inference["unclipped_prediction_hex"] != trial["unclipped_prediction_hex"]:
                raise ValueError("unclipped complete inference differs from development search")
            write_json(stage / table_path, table)
            reloaded = load_forecaster(stage / model_path)
            replay, replay_inference = _complete_predictions(reloaded, candidate)
            if table != replay or inference != replay_inference:
                raise ValueError("save/reload changed canonical forecasts")
            selected_records.append({**_summary(model, trial), "model_path": model_path,
                                     "model_file_sha256": sha256_file(stage / model_path),
                                     "prediction_table_path": table_path,
                                     "prediction_table_sha256": sha256_file(stage / table_path),
                                     "state_sha256": forecaster_state_fingerprint(model),
                                     "inference": inference, "save_reload_exact_support_match": True})
        manifest = {"schema": FORECAST_BUNDLE_SCHEMA, "package_version": __version__,
                    "status": "development_selected_not_frozen", "contract_sha256": CONTRACT_SHA256,
                    "readout_specification_path": "readout_specification.json",
                    "readout_specification_sha256": READOUT_SPEC_SHA256,
                    "data_identity": identity, "encoder_bundle_manifest_sha256": encoder_manifest_hash,
                    "runtime": forecast_runtime_record(), "source_sha256": forecast_source_identities(),
                    "landmarks_path": "landmarks.json", "landmarks_sha256": sha256_file(stage / "landmarks.json"),
                    "search_path": "search.json", "search_sha256": sha256_file(stage / "search.json"),
                    "search_count": len(records), "selection": selection, "selected_models": selected_records,
                    "membership_scaling_shared": True, "retained_training_fitted_parameters": True,
                    "calibration_status": "reserved_not_accessed", "final_freeze_created": False,
                    "statistical_certificate_issued": False,
                    "population_risk_evaluated": False,
                    "interpretation": "Validation-selected predictors; validation scores are development results, not confidence bounds."}
        write_json(stage / "manifest.json", manifest)
        # Bind final search, losses, selected state, and saved inference to the
        # actual development data before committing this output directory.
        verify_forecast_bundle(stage, dataset_bundle=bundle, encoder_bundle=encoders)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"status": "completed", "output": str(output), "search_count": len(records),
            "selected_shared_bank": selection["shared_bank_id"],
            "selected_models": [_summary(fitted[trial_id], by_id[trial_id])
                                for trial_id in _selected_trial_ids(selection).values()],
            "freeze_status": "development_selected_not_frozen", "calibration_status": "reserved_not_accessed"}


def _check_training_binding(model, x, y, landmark_plan, cache=None) -> None:
    """Check saved transformations and ridge equations on bound training rows.

    This checks moment and estimating equations, not the execution chronology.
    The stored spectral basis is used; no PCA, clustering, or SVD is rerun.
    """
    readout = model.readout
    input_scaler = model.raw_standardizer if model.candidate_id == "raw" else model.encoder.standardizer
    counts = (input_scaler.sample_count, readout.training_count, readout.linear_scaler.sample_count,
              readout.nystrom_scaler.sample_count, model.ridge.training_count)
    if any(type(count) is not int or count != len(x) for count in counts):
        raise ValueError("saved training counts differ from the bound training data")
    if readout.landmark_indices != landmark_plan.row_indices:
        raise ValueError("selected models use different landmark row identities")
    cache = {} if cache is None else cache
    cache_key = (model.candidate_id, model.bank_id,
                 hashlib.sha256(canonical_json_bytes(_pack_forecast(readout))).hexdigest())
    if cache_key not in cache:
        representation = (model.raw_standardizer.transform(x) if model.candidate_id == "raw"
                          else model.encoder.transform(x, model.candidate_id))
        expected_scaler = fit_column_scaler(representation)
        if any(not np.array_equal(getattr(expected_scaler, key), getattr(readout.linear_scaler, key))
               for key in ("mean", "scale", "raw_std")):
            raise ValueError("saved representation scaler differs from training moments")
        standardized = readout.linear_scaler.transform(representation)
        if not np.array_equal(standardized[list(landmark_plan.row_indices)], readout.landmarks):
            raise ValueError("saved encoded landmarks differ from shared training identities")
        row_cache, raw_nystrom = {}, []
        for row in standardized:
            key = row.tobytes()
            if key not in row_cache:
                row_cache[key] = _raw_nystrom_one(row, readout.landmarks, readout.sigma_squared,
                                                 readout.eigenvalues, readout.eigenvectors)
            raw_nystrom.append(row_cache[key])
        expected_nystrom_scaler = fit_column_scaler(np.stack(raw_nystrom))
        if any(not np.array_equal(getattr(expected_nystrom_scaler, key), getattr(readout.nystrom_scaler, key))
               for key in ("mean", "scale", "raw_std")):
            raise ValueError("saved Nystrom scaler differs from training moments")
        features = readout.transform(representation)
        means, target_mean = np.mean(features, axis=0), float(np.mean(y))
        centered, targets = features - means, y - target_mean
        cache[cache_key] = (means, target_mean, centered.T @ centered, centered.T @ targets)
    means, target_mean, gram, rhs = cache[cache_key]
    if not np.array_equal(means, model.ridge.feature_mean) or target_mean != model.ridge.target_mean:
        raise ValueError("saved ridge centering differs from the training data")
    matrix = gram + len(x) * model.ridge.alpha * np.eye(len(means))
    residual = float(np.linalg.norm(matrix @ model.ridge.coefficients - rhs, ord=np.inf))
    scale = float(np.linalg.norm(matrix, ord=np.inf)) * float(np.linalg.norm(model.ridge.coefficients, ord=np.inf)) + float(np.linalg.norm(rhs, ord=np.inf))
    ratio = residual / scale if scale else 0.0 if residual == 0 else float("inf")
    intercept_residual = abs(math.fsum([model.ridge.intercept, -target_mean] +
                                      [float(mean) * float(coefficient)
                                       for mean, coefficient in zip(means, model.ridge.coefficients, strict=True)]))
    if not np.isfinite(ratio) or ratio > 1e-10 or not np.isfinite(intercept_residual) or intercept_residual > 1e-10:
        raise ValueError("selected ridge state does not satisfy the bound training equations")


def verify_forecast_bundle(output: Path, *, dataset_bundle: Path, encoder_bundle: Path) -> dict:
    validate_contract()
    spec = load_readout_specification()
    output, dataset_bundle, encoder_bundle = Path(output), Path(dataset_bundle), Path(encoder_bundle)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != FORECAST_BUNDLE_SCHEMA or manifest.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("unsupported forecast bundle or scientific contract")
    if manifest.get("status") != "development_selected_not_frozen":
        raise ValueError("unsupported development forecasting status")
    if manifest.get("calibration_status") != "reserved_not_accessed":
        raise ValueError("unsupported calibration status for readout development")
    if any(manifest.get(key) is not False for key in ("final_freeze_created", "statistical_certificate_issued", "population_risk_evaluated")):
        raise ValueError("readout development bundle has an unsupported freeze, certificate, or population-risk claim")
    spec_path = checked_relative_file(output, manifest["readout_specification_path"])
    if manifest.get("readout_specification_sha256") != READOUT_SPEC_SHA256:
        raise ValueError("forecast bundle refers to another decoder menu")
    require_hash(spec_path, READOUT_SPEC_SHA256)
    banks, _ = _load_banks(encoder_bundle, dataset_bundle, spec)
    require_hash(encoder_bundle / "manifest.json", manifest["encoder_bundle_manifest_sha256"])
    training = load_development(dataset_bundle, "training")
    validation = load_development(dataset_bundle, "validation")
    identity = _data_identity(dataset_bundle)
    if manifest.get("data_identity") != identity:
        raise ValueError("forecast development data identity differs from supplied data")
    x, y = training.xy()
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    counts_train, counts_valid = empirical_counts(training.episodes), empirical_counts(validation.episodes)
    base_standardizer = banks[spec["encoder_bank_order"][0]].standardizer
    landmarks = select_landmarks(x, base_standardizer, spec["landmark_budget"])
    landmark_path = checked_relative_file(output, manifest["landmarks_path"])
    require_hash(landmark_path, manifest["landmarks_sha256"])
    if read_json(landmark_path) != landmarks.record():
        raise ValueError("landmark plan differs from the bound training inputs")
    search_path = checked_relative_file(output, manifest["search_path"])
    require_hash(search_path, manifest["search_sha256"])
    search = read_json(search_path)
    if search.get("schema") != SEARCH_SCHEMA or search.get("data_identity") != identity:
        raise ValueError("invalid search schema or data identity")
    if search.get("readout_specification_sha256") != READOUT_SPEC_SHA256:
        raise ValueError("search decoder menu differs from the specification")
    if search.get("population_risk_used") is not False or search.get("calibration_used") is not False:
        raise ValueError("search declares a forbidden evaluation-data use")
    records = search.get("records")
    roster = _expected_roster(spec)
    if not isinstance(records, list) or len(records) != len(roster) or manifest.get("search_count") != len(roster):
        raise ValueError("search does not contain the complete declared configuration roster")
    training_checks_cache = {}
    candidate_state_hashes = {}
    for record, expected in zip(records, roster, strict=True):
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("search configuration identity/order differs from the declared roster")
        table = record["prediction_table"]
        if table.get("candidate_id") != expected["trial_id"]:
            raise ValueError("search prediction table has the wrong trial identity")
        predictions = parse_prediction_table(table)
        if parse_fraction_record(record["training_mse"]) != empirical_mse_from_counts(counts_train, predictions):
            raise ValueError("search training MSE differs from exact observed losses")
        if parse_fraction_record(record["validation_mse"]) != empirical_mse_from_counts(counts_valid, predictions):
            raise ValueError("search validation MSE differs from exact observed losses")
        raw_values = record.get("unclipped_prediction_hex")
        if not isinstance(raw_values, list) or len(raw_values) != len(HISTORY_KEYS):
            raise ValueError("incomplete unclipped prediction record")
        unclipped = []
        for history, value in zip(HISTORY_KEYS, raw_values, strict=True):
            number = float.fromhex(value)
            if not np.isfinite(number) or number.hex() != value:
                raise ValueError("nonfinite or noncanonical unclipped prediction")
            if float(min(1.0, max(0.0, number))).hex() != predictions[history].hex():
                raise ValueError("search clipping does not match the prediction table")
            unclipped.append(number)
        if record.get("clipping") != _clip_record(unclipped, counts_train, counts_valid):
            raise ValueError("search clipping counts differ from the observed episodes")
        state_path = checked_relative_file(output, record["model_state_path"])
        require_hash(state_path, record["model_state_file_sha256"])
        model = load_forecaster(state_path)
        model_record = read_json(state_path)
        if model_record.get("status") != "development_candidate":
            raise ValueError("search model has the wrong development status")
        expected_provenance = _candidate_provenance(identity, manifest["encoder_bundle_manifest_sha256"], expected["trial_id"])
        if model_record.get("provenance") != expected_provenance:
            raise ValueError("search model provenance differs from the trial/data binding")
        state_hash = forecaster_state_fingerprint(model)
        if record.get("model_state_sha256") != state_hash:
            raise ValueError("search state fingerprint differs from its record")
        candidate_state_hashes[expected["trial_id"]] = state_hash
        if model.candidate_id != expected["candidate_id"] or model.bank_id != expected["bank_id"]:
            raise ValueError("search model candidate/bank identity differs from its trial")
        if model.candidate_id != "raw":
            if encoder_state_fingerprint(model.encoder) != encoder_state_fingerprint(banks[model.bank_id]):
                raise ValueError("search model embeds a different KAHM bank")
        elif any(not np.array_equal(getattr(model.raw_standardizer, key), getattr(base_standardizer, key))
                 for key in ("mean", "scale", "raw_std")):
            raise ValueError("search RAW preprocessing differs from the training preprocessing")
        derived = {"bandwidth_multiplier": model.readout.bandwidth_multiplier, "ridge_alpha": model.ridge.alpha,
                   "representation_width": model.readout.input_width, "decoder_width": model.readout.output_width,
                   "nystrom_rank": model.readout.nystrom_rank, "landmark_count": model.readout.landmark_count,
                   "sigma_squared": model.readout.sigma_squared,
                   "readout_diagnostics": model.readout.diagnostics, "ridge_diagnostics": model.ridge.diagnostics}
        if any(record.get(key) != value for key, value in derived.items()):
            raise ValueError("search summary/diagnostics differ from the fitted candidate state")
        actual_table, actual_inference = _complete_predictions(model, expected["trial_id"])
        if actual_table != table or actual_inference["unclipped_prediction_hex"] != record["unclipped_prediction_hex"]:
            raise ValueError("search candidate does not execute its saved predictions")
        _check_training_binding(model, x, y, landmarks, training_checks_cache)
    selection = select_trial_records(records, spec["encoder_bank_order"])
    if search.get("selection") != selection or manifest.get("selection") != selection:
        raise ValueError("recorded selection differs from exact validation selection")
    selected = manifest.get("selected_models")
    if not isinstance(selected, list) or [row.get("candidate_id") for row in selected] != spec["candidate_ids"]:
        raise ValueError("selected family must contain exactly RAW, MEM, and LOCAL in order")
    by_id, winner_ids = {row["trial_id"]: row for row in records}, _selected_trial_ids(selection)
    checked_models, loaded = [], {}
    for record in selected:
        candidate = record["candidate_id"]
        trial_id = winner_ids[candidate]
        if record.get("selected_trial_id") != trial_id:
            raise ValueError("saved model is not the declared validation winner")
        model_path = checked_relative_file(output, record["model_path"])
        table_path = checked_relative_file(output, record["prediction_table_path"])
        require_hash(model_path, record["model_file_sha256"])
        require_hash(table_path, record["prediction_table_sha256"])
        model = load_forecaster(model_path)
        loaded[candidate] = model
        if model.candidate_id != candidate or forecaster_state_fingerprint(model) != record.get("state_sha256"):
            raise ValueError("selected model identity differs from saved state")
        selected_record = read_json(model_path)
        if selected_record.get("status") != "development_selected_not_frozen":
            raise ValueError("selected model has the wrong development status")
        if record.get("state_sha256") != candidate_state_hashes[trial_id]:
            raise ValueError("selected state differs from the complete winning search state")
        if selected_record.get("provenance") != _model_provenance(identity, manifest["encoder_bundle_manifest_sha256"], trial_id):
            raise ValueError("selected model provenance differs from the search/data binding")
        if candidate != "raw":
            if model.bank_id != selection["shared_bank_id"]:
                raise ValueError("MEM and LOCAL do not use the selected shared bank")
            if encoder_state_fingerprint(model.encoder) != encoder_state_fingerprint(banks[model.bank_id]):
                raise ValueError("embedded selected encoder differs from the development bank")
        else:
            for key in ("mean", "scale", "raw_std"):
                if not np.array_equal(getattr(model.raw_standardizer, key), getattr(base_standardizer, key)):
                    raise ValueError("RAW input standardization differs from the shared training preprocessing")
        expected_summary = _summary(model, by_id[trial_id])
        if any(record.get(key) != value for key, value in expected_summary.items()):
            raise ValueError("selected model summary differs from model state or search loss")
        if model.readout.bandwidth_multiplier != by_id[trial_id]["bandwidth_multiplier"] or model.ridge.alpha != by_id[trial_id]["ridge_alpha"]:
            raise ValueError("selected model hyperparameters differ from its winning trial")
        table, inference = _complete_predictions(model, candidate)
        if read_json(table_path) != table or record.get("inference") != inference:
            raise ValueError("selected canonical predictions differ from saved inference")
        predictions = parse_prediction_table(table)
        winning_predictions = parse_prediction_table(by_id[trial_id]["prediction_table"])
        if any(predictions[history].hex() != winning_predictions[history].hex() for history in HISTORY_KEYS):
            raise ValueError("selected model does not execute its recorded winning predictions")
        if inference["unclipped_prediction_hex"] != by_id[trial_id]["unclipped_prediction_hex"]:
            raise ValueError("selected unclipped predictions differ from the winning trial")
        if record.get("save_reload_exact_support_match") is not True:
            raise ValueError("missing selected-model replay declaration")
        _check_training_binding(model, x, y, landmarks, training_checks_cache)
        checked_models.append({"candidate_id": candidate, "bank_id": model.bank_id,
                               "all_16_predictions_match_exactly": True,
                               "training_moments_and_ridge_equations_checked": True,
                               "training_mse": record["training_mse"], "validation_mse": record["validation_mse"]})
    mem_scaler, local_scaler = loaded["kahm_mem"].readout.linear_scaler, loaded["kahm_local"].readout.linear_scaler
    if any(not np.array_equal(getattr(mem_scaler, key), getattr(local_scaler, key)[:mem_scaler.input_width])
           for key in ("mean", "scale", "raw_std")):
        raise ValueError("selected MEM/LOCAL membership scaling is not shared")
    if manifest.get("membership_scaling_shared") is not True or manifest.get("retained_training_fitted_parameters") is not True:
        raise ValueError("unsupported shared-scaling or training-fit declaration")
    return {"status": "passed", "search_configurations_checked": len(records),
            "exact_training_and_validation_losses_recomputed": True,
            "shared_bank_selection_recomputed": True, "selected_shared_bank": selection["shared_bank_id"],
            "selected_models": checked_models, "common_landmark_identities_checked": True,
            "membership_scaling_shared": True,
            "source_identity_matches": manifest.get("source_sha256") == forecast_source_identities(),
            "runtime_metadata_matches": manifest.get("runtime") == forecast_runtime_record(),
            "all_225_candidate_states_replayed": True,
            "all_candidate_training_moments_and_ridge_equations_checked": True,
            "models_refitted_during_verification": False,
            "calibration_accessed": False, "freeze_status": "development_selected_not_frozen",
            "interpretation": "Search roster, empirical losses, selection, and selected computational predictors checked; no statistical risk certificate."}

"""Reproducible encoder development artifacts, without forecast fitting."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import __version__
from ._util import checked_relative_file, read_json, require_hash, sha256_file, write_json
from .contract import CONTRACT_SHA256, validate_contract
from .datasets import load_development
from .encoder import CANONICAL_ENCODER_METHOD, fit_encoder_candidates
from .encoder_state import (encoder_state_fingerprint, load_encoder, runtime_record,
                            save_encoder, shared_geometry_fingerprint)
from .law import HISTORY_KEYS, OBSERVATION_FLOATS

ENCODER_BUNDLE_SCHEMA = "kahm.encoder_development_bundle.v1"
FEATURE_TABLE_SCHEMA = "kahm.encoder_support_features.v1"
SOURCE_FILES = ("geometry.py", "kahm.py", "soft_state.py", "encoder.py",
                "encoder_state.py", "encoder_workflow.py")


def numerical_source_identities() -> dict[str, str]:
    base = Path(__file__).parent
    return {name: sha256_file(base / name) for name in SOURCE_FILES}


def _hex_vector(array) -> list[str]:
    return [float(value).hex() for value in array]


def support_feature_table(encoder, bank_id: str) -> tuple[dict, dict]:
    """Enumerate inputs only; no targets, model risks, or calibration draws."""
    rows = []
    fallback_events = []
    rank_zero_evaluations = 0
    ratios = []
    largest_sum_error = 0.0
    smallest_membership = 1.0
    largest_membership = 0.0
    for a, b in HISTORY_KEYS:
        history_id = 4 * a + b
        x = (OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b])
        encoded = encoder.encode_one(x)
        arrays = (encoded.raw, encoded.kahm_mem, encoded.kahm_local, encoded.scores,
                  encoded.log_normalizers, encoded.reconstructions)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError(f"nonfinite features for history {history_id}")
        if not np.array_equal(encoded.kahm_mem, encoded.kahm_local[:len(encoder.models)]):
            raise ValueError("MEM is not the identical prefix of LOCAL")
        if np.any(encoded.scores < 0) or np.any(encoded.scores > 1):
            raise ValueError("reconstruction scores are outside [0,1]")
        start = 0
        for count in encoder.group_sizes:
            block = encoded.kahm_mem[start:start + count]
            error = abs(float(np.sum(block)) - 1.0)
            largest_sum_error = max(largest_sum_error, error)
            if np.any(block < 0) or np.any(block > 1) or error > 1e-13:
                raise ValueError("invalid membership simplex block")
            start += count
        smallest_membership = min(smallest_membership, float(np.min(encoded.kahm_mem)))
        largest_membership = max(largest_membership, float(np.max(encoded.kahm_mem)))
        for model_index, (model, diagnostic) in enumerate(zip(encoder.models, encoded.model_diagnostics, strict=True)):
            if model.pca.rank == 0:
                rank_zero_evaluations += 1
            if diagnostic.get("used_fallback"):
                fallback_events.append({"history_id": history_id, "model_index": model_index,
                                        "diagnostics": diagnostic})
            ratio = diagnostic.get("cancellation_ratio")
            if ratio is not None:
                ratios.append(float(ratio))
        rows.append({"history_id": history_id, "previous_state": a, "current_state": b,
                     "input_hex": _hex_vector(x), "raw_hex": _hex_vector(encoded.raw),
                     "kahm_mem_hex": _hex_vector(encoded.kahm_mem),
                     "kahm_local_hex": _hex_vector(encoded.kahm_local),
                     "score_hex": _hex_vector(encoded.scores),
                     "log_normalizer_hex": _hex_vector(encoded.log_normalizers),
                     "reconstruction_hex": [_hex_vector(row) for row in encoded.reconstructions]})
    table = {"schema": FEATURE_TABLE_SCHEMA, "bank_id": bank_id,
             "state_sha256": encoder_state_fingerprint(encoder),
             "canonical_method": CANONICAL_ENCODER_METHOD,
             "feature_widths": encoder.feature_widths, "entries": rows}
    diagnostics = {"support_histories": len(rows), "local_model_evaluations": len(rows) * len(encoder.models),
                   "all_features_finite": True, "mem_is_exact_local_prefix": True,
                   "maximum_partition_sum_error": largest_sum_error,
                   "minimum_membership": smallest_membership, "maximum_membership": largest_membership,
                   "minimum_cancellation_ratio": min(ratios) if ratios else None,
                   "rank_zero_evaluations": rank_zero_evaluations,
                   "fallback_count": len(fallback_events), "fallback_events": fallback_events}
    return table, diagnostics


def fit_development_encoders(bundle: Path, output: Path, *, caps: tuple[int, ...] = (1, 2)) -> dict:
    """Fit the ordered bank menu using training X; reserve selection for readout development."""
    validate_contract()
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"encoder output already exists: {output}; choose a fresh directory")
    training = load_development(bundle, "training")
    # Do not even construct a target array: geometry accepts observed inputs only.
    histories = [episode.x for episode in training.episodes]
    banks = fit_encoder_candidates(histories, caps=caps)
    source_manifest = read_json(bundle / "manifest.json")
    training_record = source_manifest["development"]["training"]
    provenance = {"run_id": training.run_id, "role": "training",
                  "training_episode_count": len(histories),
                  "training_data_sha256": training_record["sha256"],
                  "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
                  "targets_used": False, "validation_used": False, "calibration_used": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kahm-encoders-", dir=output.parent))
    try:
        bank_records = []
        for bank in banks:
            bank_id = f"cap_{bank.config.cap}"
            state_path = f"{bank_id}.json"
            features_path = f"{bank_id}_features.json"
            save_encoder(bank, stage / state_path, provenance=provenance)
            table, diagnostics = support_feature_table(bank, bank_id)
            write_json(stage / features_path, table)
            reloaded = load_encoder(stage / state_path)
            replay, replay_diagnostics = support_feature_table(reloaded, bank_id)
            if table != replay or diagnostics != replay_diagnostics:
                raise ValueError(f"save/reload changed canonical encoder outputs for {bank_id}")
            bank_records.append({"bank_id": bank_id, "cap": bank.config.cap,
                                 "state_path": state_path, "state_file_sha256": sha256_file(stage / state_path),
                                 "feature_table_path": features_path,
                                 "feature_table_sha256": sha256_file(stage / features_path),
                                 "feature_widths": bank.feature_widths,
                                 "effective_ranks": [model.pca.rank for model in bank.models],
                                 "regularizers": [model.fitted_lambda for model in bank.models],
                                 "regularization_converged": [model.converged for model in bank.models],
                                 "model_diagnostics": [model.diagnostics for model in bank.models],
                                 "support_diagnostics": diagnostics,
                                 "save_reload_exact_support_match": True})
        geometry_path = stage / "reference_geometry.json"
        write_json(geometry_path, {"standardizer": banks[0].standardizer.diagnostics,
                                  "partitions": [partition.diagnostics for partition in banks[0].partitions],
                                  "shared_across_caps": True, "shared_between_mem_and_local": True})
        manifest = {"schema": ENCODER_BUNDLE_SCHEMA, "package_version": __version__,
                    "contract_sha256": CONTRACT_SHA256, "status": "development_banks_unselected",
                    "selected_bank_id": None, "bank_order": [record["bank_id"] for record in bank_records],
                    "selection_rule": "readout development: mean of MEM and LOCAL best validation MSEs; ties follow bank_order",
                    "runtime": runtime_record(), "source_sha256": numerical_source_identities(),
                    "training_provenance": provenance,
                    "geometry_path": "reference_geometry.json", "geometry_sha256": sha256_file(geometry_path),
                    "banks": bank_records,
                    "calibration_status": "reserved_not_accessed",
                    "forecast_models_fitted": False, "statistical_certificate_issued": False,
                    "scope": "Gaussian KAHM computational encoders; numerical diagnostics are not verified real-arithmetic bounds"}
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"status": "completed", "output": str(output), "bank_order": manifest["bank_order"],
            "banks": [{key: record[key] for key in ("bank_id", "feature_widths", "effective_ranks",
                                                    "support_diagnostics", "save_reload_exact_support_match")}
                      for record in bank_records],
            "selection_status": "unselected; awaiting fitted readouts and validation",
            "calibration_status": "reserved_not_accessed"}


def verify_encoder_bundle(output: Path, *, dataset_bundle: Path | None = None) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != ENCODER_BUNDLE_SCHEMA or manifest.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("unsupported encoder bundle or contract")
    if manifest.get("status") != "development_banks_unselected" or manifest.get("selected_bank_id") is not None:
        raise ValueError("unexpected encoder selection status")
    bank_records = manifest.get("banks")
    if not isinstance(bank_records, list) or not bank_records:
        raise ValueError("encoder bundle has no bank records")
    expected_order = [record["bank_id"] for record in bank_records]
    if manifest.get("bank_order") != expected_order or len(set(expected_order)) != len(expected_order):
        raise ValueError("encoder bank ordering or identity is inconsistent")
    source_matches = manifest.get("source_sha256") == numerical_source_identities()
    geometry_path = checked_relative_file(output, manifest["geometry_path"])
    require_hash(geometry_path, manifest["geometry_sha256"])
    provenance = manifest.get("training_provenance")
    if not isinstance(provenance, dict) or provenance.get("role") != "training":
        raise ValueError("invalid training provenance")
    if any(provenance.get(key) is not False for key in ("targets_used", "validation_used", "calibration_used")):
        raise ValueError("encoder provenance does not describe training-input-only fitting")
    if dataset_bundle is not None:
        dataset_bundle = Path(dataset_bundle)
        training = load_development(dataset_bundle, "training")
        require_hash(dataset_bundle / "manifest.json", provenance["dataset_manifest_sha256"])
        dataset_manifest = read_json(dataset_bundle / "manifest.json")
        if dataset_manifest["development"]["training"]["sha256"] != provenance.get("training_data_sha256"):
            raise ValueError("encoder training data identity differs from development data")
        if provenance.get("training_episode_count") != len(training.episodes) or provenance.get("run_id") != training.run_id:
            raise ValueError("encoder training provenance differs from development data")
    checks = []
    geometry_fingerprint = None
    for record in bank_records:
        state_path = checked_relative_file(output, record["state_path"])
        feature_path = checked_relative_file(output, record["feature_table_path"])
        require_hash(state_path, record["state_file_sha256"])
        require_hash(feature_path, record["feature_table_sha256"])
        encoder = load_encoder(state_path)
        state_record = read_json(state_path)
        if state_record.get("provenance") != provenance:
            raise ValueError("bank provenance differs from encoder bundle")
        fingerprint = shared_geometry_fingerprint(encoder)
        if geometry_fingerprint is None:
            geometry_fingerprint = fingerprint
            geometry = {"standardizer": encoder.standardizer.diagnostics,
                        "partitions": [partition.diagnostics for partition in encoder.partitions],
                        "shared_across_caps": True, "shared_between_mem_and_local": True}
            if read_json(geometry_path) != geometry:
                raise ValueError("reference geometry report differs from saved bank")
        elif fingerprint != geometry_fingerprint:
            raise ValueError("candidate banks do not share preprocessing and reference identities")
        expected = read_json(feature_path)
        replay, diagnostics = support_feature_table(encoder, record["bank_id"])
        if replay != expected:
            raise ValueError(f"canonical feature replay differs for {record['bank_id']}; use the recorded runtime or refit during development")
        if record.get("feature_widths") != encoder.feature_widths or record.get("cap") != encoder.config.cap:
            raise ValueError("bank metadata differs from saved encoder state")
        derived_summaries = {"effective_ranks": [model.pca.rank for model in encoder.models],
                             "regularizers": [model.fitted_lambda for model in encoder.models],
                             "regularization_converged": [model.converged for model in encoder.models],
                             "model_diagnostics": [model.diagnostics for model in encoder.models],
                             "save_reload_exact_support_match": True}
        for key, value in derived_summaries.items():
            if record.get(key) != value:
                raise ValueError(f"saved {key} summary differs from encoder state/replay")
        if diagnostics != record.get("support_diagnostics"):
            raise ValueError("saved support diagnostics differ from replay")
        checks.append({"bank_id": record["bank_id"], "all_16_feature_rows_match_exactly": True,
                       "fallback_count": diagnostics["fallback_count"], "feature_widths": encoder.feature_widths})
    return {"status": "passed", "banks": checks,
            "numerical_source_identity_matches": source_matches,
            "runtime_metadata_matches": manifest.get("runtime") == runtime_record(),
            "dataset_provenance_checked": dataset_bundle is not None,
            "candidate_geometry_identity_matches": True,
            "calibration_status": "not_accessed",
            "interpretation": "Saved-state consistency and exact supported-input replay; no forecasting risk certificate."}

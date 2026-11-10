"""Freeze the selected computational family before independent calibration.

The snapshot retains the original development artifacts byte for byte. Their
internal statuses remain historical; the enclosing, content-identified freeze
record establishes the frozen family. Verification executes the archived states
using the installed, identity-checked implementation, never imported snapshot code.

File identities and workflow guards detect inconsistent artifacts. They do not
authenticate execution chronology or make an independently editable directory
tamper-proof. No function in this module reads calibration outcomes or evaluates
model population risk.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from importlib import resources
from pathlib import Path
import shutil
import tempfile

from . import __version__
from ._util import canonical_json_bytes, checked_relative_file, read_json, require_hash, sha256_file, write_json
from .calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol
from .contract import CANDIDATE_IDS, CONTRACT_SHA256, PRIMARY_CALIBRATION_COUNT, validate_contract
from .datasets import FREEZE_SCHEMA, load_development, validate_freeze_record
from .encoder_state import encoder_state_fingerprint
from .forecast import CANONICAL_FORECAST_METHOD, ForecastModel
from .forecast_state import forecaster_state_fingerprint, load_forecaster
from .forecast_workflow import _complete_predictions, forecast_runtime_record, verify_forecast_bundle
from .law import HISTORY_KEYS, OBSERVATION_FLOATS
from .sampling import stream_descriptor


PREFLIGHT_SCHEMA = "kahm.frozen_forecast_bundle.v1"
FREEZE_FILENAME = "freeze.json"
# A fixed dependency roster, not a list chosen by an incoming artifact. The
# package version initializer and CLI are archived separately: neither defines
# predictions. The initializer must remain a docstring and literal version only.
EXECUTION_SOURCE_FILES = (
    "_util.py", "calibration_protocol.json", "calibration_protocol.py", "contract.py",
    "datasets.py", "encoder.py", "encoder_state.py", "encoder_workflow.py", "forecast.py",
    "forecast_state.py", "forecast_workflow.py", "freeze.py", "geometry.py", "kahm.py",
    "law.py", "oracle.py", "readout_features.py", "readout_specification.json", "ridge.py",
    "sampling.py", "selection.py", "soft_state.py", "specification.json",
)
_DEVELOPMENT_PATHS = {"dataset": "development/data", "encoders": "development/encoders",
                      "forecasters": "development/forecasters"}
_FREEZE_FIELDS = {
    "schema", "model_preflight_schema", "status", "contract_sha256", "run_id",
    "dataset_manifest_sha256", "calibration_plan_sha256", "package_version", "inference",
    "models", "development", "calibration_protocol", "calibration_reservation",
    "calibration_status_at_freeze", "statistical_certificate_issued", "model_population_risk_evaluated",
    "refit_during_freeze", "files", "record_sha256",
}


def _record_digest(record: dict) -> str:
    return hashlib.sha256(canonical_json_bytes({k: v for k, v in record.items()
                                               if k != "record_sha256"})).hexdigest()


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_version_only_initializer(path: Path) -> str:
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if (len(body) != 1 or not isinstance(body[0], ast.Assign)
            or len(body[0].targets) != 1 or not isinstance(body[0].targets[0], ast.Name)
            or body[0].targets[0].id != "__version__"
            or not isinstance(body[0].value, ast.Constant)
            or not isinstance(body[0].value.value, str) or not body[0].value.value):
        raise ValueError("package initializer is not a metadata-only version declaration")
    return body[0].value.value


def _inventory(root: Path) -> list[dict]:
    result = []
    paths = sorted(root.rglob("*"))
    # Reject outcome directories before hashing any bytes in this snapshot.
    if any("calibration" in path.relative_to(root).parts for path in paths):
        raise ValueError("frozen snapshots must not contain calibration outcomes")
    for path in paths:
        if path.is_symlink():
            raise ValueError("frozen snapshots must not contain symlinks")
        if path.is_file() and path != root / FREEZE_FILENAME:
            result.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                           "sha256": sha256_file(path)})
    return result


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _dataset_record(bundle: Path) -> tuple[dict, dict, list[str]]:
    # These loaders verify both development splits and the reserved plan, but
    # never inspect an existing calibration directory or its outcomes.
    calibration_root = (bundle / "calibration").resolve()
    manifest_path = checked_relative_file(bundle, "manifest.json")
    if manifest_path.is_relative_to(calibration_root):
        raise ValueError("dataset manifest resolves inside calibration outcomes")
    manifest = read_json(manifest_path)
    plan_path = manifest["calibration_plan"]["path"]
    paths = ["manifest.json", plan_path] + [manifest["development"][role]["path"]
                                           for role in ("training", "validation")]
    if len(set(paths)) != 4 or any(Path(path).parts[0] == "calibration" for path in paths):
        raise ValueError("development snapshot paths must exclude calibration outcomes")
    for relative in paths:
        if checked_relative_file(bundle, relative).is_relative_to(calibration_root):
            raise ValueError("development snapshot path resolves inside calibration outcomes")
    for role in ("training", "validation"):
        load_development(bundle, role)
    plan = read_json(checked_relative_file(bundle, plan_path))
    if plan.get("stream") != stream_descriptor(int(plan["master_seed_integer"]), "calibration", plan["run_id"]):
        raise ValueError("reserved calibration sampler/runtime differs from the current implementation")
    return manifest, plan, paths


def _encoder_paths(root: Path) -> list[str]:
    record = read_json(root / "manifest.json")
    return ["manifest.json", record["geometry_path"]] + [row[key] for row in record["banks"]
                                                          for key in ("state_path", "feature_table_path")]


def _forecaster_paths(root: Path) -> list[str]:
    record = read_json(root / "manifest.json")
    search = read_json(checked_relative_file(root, record["search_path"]))
    return (["manifest.json", record["landmarks_path"], record["readout_specification_path"], record["search_path"]]
            + [row[key] for row in record["selected_models"] for key in ("model_path", "prediction_table_path")]
            + [row["model_state_path"] for row in search["records"]])


def _copy_declared_files(source: Path, target: Path, paths: list[str]) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError("snapshot declares duplicate artifact paths")
    for relative in paths:
        original = checked_relative_file(source, relative)
        _copy_file(original, target / relative)


def _source_records(snapshot: Path) -> list[dict]:
    return [{"path": path.relative_to(snapshot).as_posix(), "sha256": sha256_file(path)}
            for path in sorted((snapshot / "sources/kahm_demo").iterdir()) if path.is_file()]


def _model_record(snapshot: Path, selected: dict) -> tuple[dict, ForecastModel]:
    candidate = selected["candidate_id"]
    artifact = f"models/{candidate}.json"
    predictions = f"predictions/{candidate}.json"
    artifact_path = checked_relative_file(snapshot, artifact)
    table_path = checked_relative_file(snapshot, predictions)
    model = load_forecaster(artifact_path)
    table, diagnostics = _complete_predictions(model, candidate)
    if model.candidate_id != candidate or read_json(table_path) != table:
        raise ValueError("frozen model execution differs from its canonical prediction table")
    if forecaster_state_fingerprint(model) != selected["state_sha256"]:
        raise ValueError("frozen fitted state differs from the selected development state")
    histories = [[OBSERVATION_FLOATS[a], OBSERVATION_FLOATS[b]] for a, b in HISTORY_KEYS]
    batch = model.predict(histories + histories[:1])
    scalar = [model.predict_one(row).hex() for row in histories + histories[:1]]
    if [float(value).hex() for value in batch] != scalar:
        raise ValueError("frozen scalar and repeated-row batch inference differ")
    record = {"candidate_id": candidate, "status": "frozen", "bank_id": model.bank_id,
              "selected_trial_id": selected["selected_trial_id"],
              "artifact_path": artifact, "artifact_sha256": sha256_file(artifact_path),
              "state_sha256": forecaster_state_fingerprint(model),
              "prediction_table_path": predictions, "prediction_table_sha256": sha256_file(table_path),
              "canonical_method": CANONICAL_FORECAST_METHOD,
              "support_history_count": len(HISTORY_KEYS), "execution_diagnostics": diagnostics,
              "scalar_batch_exact_match": True,
              "original_artifact_status": read_json(artifact_path)["status"]}
    return record, model


def freeze_forecasters(bundle: Path, encoders: Path, models: Path, output: Path) -> dict:
    """Commit a verified, self-contained snapshot without fitting or calibration."""
    validate_contract()
    load_calibration_protocol()
    bundle, encoders, models, output = map(Path, (bundle, encoders, models, output))
    if _path_present(output):
        raise FileExistsError(f"freeze output already exists: {output}; existing freezes are not replaced")
    if _path_present(bundle / "calibration"):
        raise ValueError("cannot create a model freeze after calibration has been materialized")
    if any(output.resolve().is_relative_to(parent.resolve()) for parent in (bundle, encoders, models)):
        raise ValueError("freeze output must not be inside a source data or model bundle")
    data_manifest, plan, data_paths = _dataset_record(bundle)
    development_check = verify_forecast_bundle(models, dataset_bundle=bundle, encoder_bundle=encoders)
    if not development_check["source_identity_matches"] or not development_check["runtime_metadata_matches"]:
        raise ValueError("development source/runtime identity must match before final freeze")
    model_manifest = read_json(models / "manifest.json")
    package_root = Path(__file__).parent
    _require_version_only_initializer(package_root / "__init__.py")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kahm-freeze-", dir=output.parent))
    try:
        _copy_declared_files(bundle, stage / _DEVELOPMENT_PATHS["dataset"], data_paths)
        _copy_declared_files(encoders, stage / _DEVELOPMENT_PATHS["encoders"], _encoder_paths(encoders))
        _copy_declared_files(models, stage / _DEVELOPMENT_PATHS["forecasters"], _forecaster_paths(models))
        for path in sorted(package_root.iterdir()):
            if path.is_file() and path.suffix in (".py", ".json"):
                _copy_file(path, stage / "sources/kahm_demo" / path.name)
        protocol_path = stage / "calibration_protocol.json"
        protocol_path.write_bytes(resources.files("kahm_demo").joinpath("calibration_protocol.json").read_bytes())
        frozen_models = []
        for selected in model_manifest["selected_models"]:
            candidate = selected["candidate_id"]
            _copy_file(checked_relative_file(models, selected["model_path"]), stage / f"models/{candidate}.json")
            _copy_file(checked_relative_file(models, selected["prediction_table_path"]), stage / f"predictions/{candidate}.json")
            frozen_models.append(_model_record(stage, selected)[0])
        record = {
            "schema": FREEZE_SCHEMA, "model_preflight_schema": PREFLIGHT_SCHEMA, "status": "frozen",
            "contract_sha256": CONTRACT_SHA256, "run_id": data_manifest["run_id"],
            "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "calibration_plan_sha256": data_manifest["calibration_plan"]["sha256"],
            "package_version": __version__,
            "inference": {"canonical_method": CANONICAL_FORECAST_METHOD, "runtime": forecast_runtime_record(),
                          "source_files": _source_records(stage),
                          "execution_source_roster": list(EXECUTION_SOURCE_FILES),
                          "version_initializer_policy": "docstring_and_literal_version_only"},
            "models": frozen_models,
            "development": {"paths": _DEVELOPMENT_PATHS.copy(),
                            "manifest_sha256": {kind: sha256_file(stage / path / "manifest.json")
                                                for kind, path in _DEVELOPMENT_PATHS.items()},
                            "search_count": model_manifest["search_count"],
                            "selection": model_manifest["selection"],
                            "retained_training_fitted_parameters": True},
            "calibration_protocol": {"path": "calibration_protocol.json", "sha256": CALIBRATION_PROTOCOL_SHA256},
            "calibration_reservation": {"episode_count": PRIMARY_CALIBRATION_COUNT,
                                        "plan_sha256": data_manifest["calibration_plan"]["sha256"],
                                        "stream": plan["stream"]},
            "calibration_status_at_freeze": "reserved_not_materialized",
            "statistical_certificate_issued": False, "model_population_risk_evaluated": False,
            "refit_during_freeze": False,
            "files": _inventory(stage),
        }
        record["record_sha256"] = _record_digest(record)
        write_json(stage / FREEZE_FILENAME, record)
        verification, _ = _verify_frozen_bundle(stage / FREEZE_FILENAME, dataset_bundle=bundle)
        if _path_present(bundle / "calibration"):
            raise ValueError("calibration appeared during freeze creation; freeze was not committed")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"status": "frozen", "freeze_path": str(output / FREEZE_FILENAME),
            "freeze_record_sha256": sha256_file(output / FREEZE_FILENAME),
            "record_payload_sha256": record["record_sha256"],
            "candidate_ids": list(CANDIDATE_IDS), "selected_shared_bank": model_manifest["selection"]["shared_bank_id"],
            "calibration_status": "reserved_not_materialized", "statistical_certificate_issued": False,
            "verification": verification}


def _verify_frozen_bundle(freeze_path: Path, *, dataset_bundle: Path | None) -> tuple[dict, dict[str, ForecastModel]]:
    """Return the actual verified objects; callers must not reload unchecked state."""
    validate_contract()
    load_calibration_protocol()
    freeze_path = Path(freeze_path)
    if freeze_path.name != FREEZE_FILENAME:
        raise ValueError(f"frozen bundle record must be named {FREEZE_FILENAME}")
    root = freeze_path.parent
    record = read_json(freeze_path)
    if not isinstance(record, dict) or set(record) != _FREEZE_FIELDS:
        raise ValueError("unsupported frozen model record fields")
    if record.get("record_sha256") != _record_digest(record):
        raise ValueError("freeze record payload identity mismatch")
    if record.get("model_preflight_schema") != PREFLIGHT_SCHEMA or record.get("status") != "frozen":
        raise ValueError("unsupported model-specific frozen bundle")
    if record.get("files") != _inventory(root):
        raise ValueError("frozen snapshot inventory or file identities differ")
    if any(record.get(key) is not False for key in
           ("refit_during_freeze", "statistical_certificate_issued", "model_population_risk_evaluated")):
        raise ValueError("freeze has an unsupported refit, certificate, or population-risk claim")
    if record.get("calibration_status_at_freeze") != "reserved_not_materialized":
        raise ValueError("freeze does not declare an untouched calibration reservation")
    if not isinstance(record.get("package_version"), str) or not record["package_version"]:
        raise ValueError("freeze has no recorded package version")
    inference = record.get("inference")
    expected_inference_fields = {"canonical_method", "runtime", "source_files", "execution_source_roster", "version_initializer_policy"}
    if not isinstance(inference, dict) or set(inference) != expected_inference_fields:
        raise ValueError("invalid frozen inference record")
    if inference["execution_source_roster"] != list(EXECUTION_SOURCE_FILES):
        raise ValueError("frozen execution dependency roster differs")
    if inference["canonical_method"] != CANONICAL_FORECAST_METHOD or inference["runtime"] != forecast_runtime_record():
        raise ValueError("frozen canonical inference method or runtime differs")
    if inference["source_files"] != _source_records(root):
        raise ValueError("frozen source archive identity differs")
    if inference["version_initializer_policy"] != "docstring_and_literal_version_only":
        raise ValueError("unsupported package initializer policy")
    package_root = Path(__file__).parent
    _require_version_only_initializer(package_root / "__init__.py")
    if _require_version_only_initializer(root / "sources/kahm_demo/__init__.py") != record["package_version"]:
        raise ValueError("archived package version differs from the freeze provenance")
    for name in EXECUTION_SOURCE_FILES:
        require_hash(package_root / name, sha256_file(checked_relative_file(root, f"sources/kahm_demo/{name}")))
    if record["calibration_protocol"] != {"path": "calibration_protocol.json", "sha256": CALIBRATION_PROTOCOL_SHA256}:
        raise ValueError("frozen calibration protocol differs from the declared protocol")
    require_hash(checked_relative_file(root, "calibration_protocol.json"), CALIBRATION_PROTOCOL_SHA256)
    development = record.get("development")
    if (not isinstance(development, dict)
            or set(development) != {"paths", "manifest_sha256", "search_count", "selection", "retained_training_fitted_parameters"}
            or development["paths"] != _DEVELOPMENT_PATHS
            or development["retained_training_fitted_parameters"] is not True):
        raise ValueError("invalid frozen development snapshot metadata")
    archived_data = root / _DEVELOPMENT_PATHS["dataset"]
    if _path_present(archived_data / "calibration"):
        raise ValueError("frozen development snapshot must not contain calibration outcomes")
    if development["manifest_sha256"] != {kind: sha256_file(root / path / "manifest.json")
                                           for kind, path in _DEVELOPMENT_PATHS.items()}:
        raise ValueError("archived development manifest identity differs")
    archived_manifest, archived_plan, archived_paths = _dataset_record(archived_data)
    # Exactly the declared development files are retained, with no outcome copy.
    if {p.relative_to(archived_data).as_posix() for p in archived_data.rglob("*") if p.is_file()} != set(archived_paths):
        raise ValueError("frozen data snapshot contains undeclared files")
    bound_data = archived_data if dataset_bundle is None else Path(dataset_bundle)
    bound_manifest, bound_plan, _ = _dataset_record(bound_data)
    if (sha256_file(bound_data / "manifest.json") != sha256_file(archived_data / "manifest.json")
            or bound_manifest != archived_manifest or bound_plan != archived_plan):
        raise ValueError("external data bundle differs from the frozen development identity")
    validate_freeze_record(bound_data, freeze_path)
    expected_reservation = {"episode_count": PRIMARY_CALIBRATION_COUNT,
                            "plan_sha256": archived_manifest["calibration_plan"]["sha256"],
                            "stream": archived_plan["stream"]}
    if record["calibration_reservation"] != expected_reservation:
        raise ValueError("frozen calibration stream/count differs from its reservation")
    checked = verify_forecast_bundle(root / _DEVELOPMENT_PATHS["forecasters"],
                                    dataset_bundle=archived_data,
                                    encoder_bundle=root / _DEVELOPMENT_PATHS["encoders"])
    if not checked["source_identity_matches"] or not checked["runtime_metadata_matches"]:
        raise ValueError("archived development source/runtime identity differs")
    model_manifest = read_json(root / _DEVELOPMENT_PATHS["forecasters"] / "manifest.json")
    if development["search_count"] != model_manifest["search_count"] or development["selection"] != model_manifest["selection"]:
        raise ValueError("frozen search or selection differs from the verified development history")
    records = record["models"]
    if not isinstance(records, list) or [row.get("candidate_id") for row in records] != list(CANDIDATE_IDS):
        raise ValueError("freeze must retain exactly the ordered three-member family")
    loaded, checks = {}, []
    for frozen, selected in zip(records, model_manifest["selected_models"], strict=True):
        derived, model = _model_record(root, selected)
        if frozen != derived:
            raise ValueError("frozen model metadata or canonical replay differs")
        source_root = root / _DEVELOPMENT_PATHS["forecasters"]
        if (derived["artifact_sha256"] != sha256_file(checked_relative_file(source_root, selected["model_path"]))
                or derived["prediction_table_sha256"] != sha256_file(checked_relative_file(source_root, selected["prediction_table_path"]))):
            raise ValueError("frozen selected artifacts differ from the development selection")
        loaded[model.candidate_id] = model
        checks.append({"candidate_id": model.candidate_id, "bank_id": model.bank_id,
                       "state_sha256": derived["state_sha256"], "all_16_predictions_match_exactly": True,
                       "scalar_batch_exact_match": True,
                       "clipping_count_on_support": sum(float.fromhex(x) < 0 or float.fromhex(x) > 1
                                                         for x in derived["execution_diagnostics"]["unclipped_prediction_hex"]),
                       "encoder_fallback_events": derived["execution_diagnostics"]["encoder_fallback_events"]})
    if encoder_state_fingerprint(loaded["kahm_mem"].encoder) != encoder_state_fingerprint(loaded["kahm_local"].encoder):
        raise ValueError("frozen MEM and LOCAL do not share one identical encoder bank")
    result = {"schema": "kahm.freeze_verification.v1", "status": "passed", "freeze_status": "frozen",
              "freeze_record_sha256": sha256_file(freeze_path), "record_payload_sha256": record["record_sha256"],
              "candidate_ids": list(CANDIDATE_IDS), "selected_models": checks,
              "selected_shared_bank": model_manifest["selection"]["shared_bank_id"],
              "source_identity_matches": True, "runtime_metadata_matches": True,
              "model_artifacts_byte_identical_to_selected_development": True,
              "all_225_development_candidate_states_replayed": True,
              "exact_development_losses_and_selection_checked": True,
              "self_contained_development_snapshot_checked": True,
              "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
              "calibration_count_reserved": PRIMARY_CALIBRATION_COUNT,
              "external_dataset_identity_checked": dataset_bundle is not None,
              "calibration_outcomes_accessed": False, "model_population_risk_evaluated": False,
              "models_refitted": False, "statistical_certificate_issued": False,
              "interpretation": "The selected computational family and calibration protocol are frozen and checked; no statistical risk certificate has been issued."}
    return result, loaded


def verify_frozen_bundle(freeze_path: Path, *, dataset_bundle: Path | None = None) -> dict:
    """Check the frozen family; never sample or load any calibration outcomes."""
    return _verify_frozen_bundle(Path(freeze_path), dataset_bundle=dataset_bundle)[0]


@dataclass(frozen=True, slots=True)
class FrozenForecaster:
    """An already verified model instance bound to one frozen family identity."""

    model: ForecastModel
    freeze_record_sha256: str
    calibration_protocol_sha256: str

    @property
    def candidate_id(self) -> str:
        return self.model.candidate_id

    @property
    def bank_id(self) -> str | None:
        return self.model.bank_id

    def predict_one(self, history) -> float:
        return self.model.predict_one(history)

    def inspect_one(self, history):
        return self.model.inspect_one(history)

    def predict(self, histories):
        return self.model.predict(histories)


def load_frozen_forecaster(freeze_path: Path, candidate_id: str, *, dataset_bundle: Path | None = None) -> FrozenForecaster:
    """Return the same in-memory object whose fingerprint and outputs were checked."""
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError("unknown candidate in the frozen three-member family")
    result, loaded = _verify_frozen_bundle(Path(freeze_path), dataset_bundle=dataset_bundle)
    return FrozenForecaster(loaded[candidate_id], result["freeze_record_sha256"], CALIBRATION_PROTOCOL_SHA256)

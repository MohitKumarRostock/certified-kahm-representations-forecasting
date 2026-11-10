"""Separate development data and a reserved, freeze-bound calibration plan.

The primary calibration stream is not sampled by ``generate_bundle``. Public
materialization and loading require both a structurally valid freeze record and
the full model-specific inference preflight. This is a workflow guard, not an
authentication service or a proof of execution chronology.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import tempfile

from . import __version__
from ._util import checked_relative_file, read_json, require_hash, sha256_file, write_json
from .contract import CANDIDATE_IDS, CONTRACT_SHA256, PRIMARY_CALIBRATION_COUNT, validate_contract
from .law import BENCHMARK_ID, Episode
from .oracle import parse_prediction_table
from .sampling import SAMPLER_VERSION, sample_episodes, stream_descriptor

BUNDLE_SCHEMA = "kahm.dataset_bundle.v1"
PLAN_SCHEMA = "kahm.calibration_plan.v1"
FREEZE_SCHEMA = "kahm.model_freeze.v1"
EPISODE_KEYS = {"episode_id", "previous_state", "current_state", "next_state"}


@dataclass(frozen=True, slots=True)
class Dataset:
    role: str
    run_id: str
    episodes: tuple[Episode, ...]

    def xy(self) -> tuple[list[tuple[float, float]], list[float]]:
        """Return inputs/targets without state metadata or episode IDs as features."""
        return ([e.x for e in self.episodes], [e.y for e in self.episodes])

    def episode_key(self, index: int) -> tuple[str, str, int]:
        return (self.run_id, self.role, self.episodes[index].episode_id)


def _require_positive_count(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _check_stream_record(record: dict, master_seed: int, role: str, run_id: str) -> None:
    """Validate stream identity while preserving its historical runtime record."""
    expected = stream_descriptor(master_seed, role, run_id)
    if not isinstance(record, dict):
        raise ValueError("stream record must be an object")
    # Reading archived integer rows does not require the current interpreter to
    # have the same build string as the interpreter that generated those rows.
    identity = {k: v for k, v in record.items() if k != "rng"}
    expected_identity = {k: v for k, v in expected.items() if k != "rng"}
    if identity != expected_identity:
        raise ValueError("stream identity/seed derivation is inconsistent")
    rng = record.get("rng")
    if not isinstance(rng, dict):
        raise ValueError("stream has no RNG provenance")
    for key in ("implementation", "algorithm", "state_format_version"):
        if rng.get(key) != expected["rng"][key]:
            raise ValueError("unsupported recorded RNG")
    for key in ("python_implementation", "python_version", "python_full_version"):
        if not isinstance(rng.get(key), str) or not rng[key]:
            raise ValueError("stream has incomplete historical runtime provenance")


def _write_episodes(path: Path, episodes: tuple[Episode, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for e in episodes:
            row = {"episode_id": e.episode_id, "previous_state": e.previous_state,
                   "current_state": e.current_state, "next_state": e.next_state}
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")


def read_episode_file(path: Path, expected_count: int) -> tuple[Episode, ...]:
    _require_positive_count(expected_count, "expected_count")
    episodes = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            def unique(pairs):
                row = {}
                for key, value in pairs:
                    if key in row:
                        raise ValueError(f"duplicate episode field: {key}")
                    row[key] = value
                return row
            row = json.loads(line, object_pairs_hook=unique)
            if not isinstance(row, dict) or set(row) != EPISODE_KEYS:
                raise ValueError(f"invalid episode fields at row {index}")
            e = Episode(**row)
            if e.episode_id != index:
                raise ValueError(f"episode IDs must be sequential at row {index}")
            episodes.append(e)
    if len(episodes) != expected_count:
        raise ValueError(f"expected {expected_count} episodes, got {len(episodes)}")
    return tuple(episodes)


def generate_bundle(output: Path, *, master_seed: int = 20261110,
                    run_id: str = "primary-v0.1", training_count: int = 4096,
                    validation_count: int = 1024) -> dict:
    """Create a new deterministic development bundle and an unused calibration plan."""
    validate_contract()
    _require_positive_count(training_count, "training_count")
    _require_positive_count(validation_count, "validation_count")
    # Validate all seed metadata before creating an output directory.
    descriptors = {role: stream_descriptor(master_seed, role, run_id)
                   for role in ("training", "validation", "calibration")}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; choose a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kahm-data-", dir=output.parent))
    try:
        (stage / "development").mkdir()
        records = {}
        for role, count in (("training", training_count), ("validation", validation_count)):
            relative = f"development/{role}.jsonl"
            episodes = sample_episodes(count, master_seed, role, run_id)
            _write_episodes(stage / relative, episodes)
            records[role] = {"path": relative, "role": role, "episode_count": count,
                             "sha256": sha256_file(stage / relative), "stream": descriptors[role]}
        plan = {
            "schema": PLAN_SCHEMA, "status": "reserved_not_materialized",
            "benchmark_id": BENCHMARK_ID, "contract_sha256": CONTRACT_SHA256,
            "run_id": run_id, "master_seed_integer": str(master_seed),
            "role": "calibration", "episode_count": PRIMARY_CALIBRATION_COUNT,
            "stream": descriptors["calibration"],
            "materialization_requirement": "matching three-model freeze record with artifact identities; model-specific execution/reload checks are completed during the model freeze",
            "note": "This immutable reservation records the primary stream before outcomes. A separate receipt records later materialization.",
        }
        write_json(stage / "calibration_plan.json", plan)
        manifest = {
            "schema": BUNDLE_SCHEMA, "benchmark_id": BENCHMARK_ID,
            "contract_sha256": CONTRACT_SHA256, "run_id": run_id,
            "master_seed_integer": str(master_seed), "sampler_version": SAMPLER_VERSION,
            "package_version": __version__,
            "environment": {"python_implementation": platform.python_implementation(),
                            "python_version": platform.python_version()},
            "development": records,
            "calibration_plan": {"path": "calibration_plan.json",
                                 "sha256": sha256_file(stage / "calibration_plan.json")},
            "sampling_assumption": "iid episodes from the declared law; PRNG stream metadata provides reproducibility, not a proof of independence",
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_bundle(output)


def _load_bundle_records(bundle: Path) -> tuple[dict, dict]:
    validate_contract()
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("data-bundle manifest must be an object")
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("unsupported data-bundle schema or benchmark")
    if manifest.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("bundle does not match the fixed scientific contract")
    if manifest.get("sampler_version") != SAMPLER_VERSION:
        raise ValueError("bundle sampler version differs from this implementation")
    if set(manifest.get("development", {})) != {"training", "validation"}:
        raise ValueError("bundle must contain exactly the two development roles")
    plan_record = manifest["calibration_plan"]
    plan_path = checked_relative_file(bundle, plan_record["path"])
    require_hash(plan_path, plan_record["sha256"])
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("calibration plan must be an object")
    seed = int(manifest["master_seed_integer"])
    if str(seed) != manifest["master_seed_integer"]:
        raise ValueError("noncanonical master seed")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "reserved_not_materialized"
        or plan.get("benchmark_id") != BENCHMARK_ID or plan.get("contract_sha256") != CONTRACT_SHA256
        or plan.get("run_id") != manifest["run_id"] or plan.get("master_seed_integer") != str(seed)
        or plan.get("role") != "calibration" or type(plan.get("episode_count")) is not int
        or plan["episode_count"] != PRIMARY_CALIBRATION_COUNT):
        raise ValueError("calibration plan differs from the declared contract/bundle")
    _check_stream_record(plan.get("stream"), seed, "calibration", manifest["run_id"])
    return manifest, plan


def load_development(bundle: Path, role: str) -> Dataset:
    if role not in ("training", "validation"):
        raise ValueError("development loader accepts training or validation only")
    bundle = Path(bundle)
    manifest, _ = _load_bundle_records(bundle)
    record = manifest["development"][role]
    if record.get("role") != role:
        raise ValueError("development role metadata mismatch")
    _check_stream_record(record.get("stream"), int(manifest["master_seed_integer"]), role, manifest["run_id"])
    path = checked_relative_file(bundle, record["path"])
    require_hash(path, record["sha256"])
    episodes = read_episode_file(path, record["episode_count"])
    return Dataset(role, manifest["run_id"], episodes)


def verify_bundle(bundle: Path) -> dict:
    """Check identity/structure; never sample the reserved calibration stream."""
    bundle = Path(bundle)
    manifest, plan = _load_bundle_records(bundle)
    counts = {role: len(load_development(bundle, role).episodes)
              for role in ("training", "validation")}
    calibration_dir = bundle / "calibration"
    state = "reserved_not_materialized"
    if calibration_dir.exists():
        receipt_path = calibration_dir / "receipt.json"
        if not receipt_path.is_file():
            raise ValueError("calibration files exist without a materialization receipt")
        receipt = read_json(receipt_path)
        if not isinstance(receipt, dict):
            raise ValueError("calibration receipt must be an object")
        freeze_hash = receipt.get("freeze_record_sha256")
        if (receipt.get("schema") != "kahm.calibration_receipt.v1"
            or receipt.get("status") != "materialized_after_recorded_freeze"
            or not isinstance(freeze_hash, str) or len(freeze_hash) != 64
            or any(c not in "0123456789abcdef" for c in freeze_hash)
            or receipt.get("candidate_ids") != sorted(CANDIDATE_IDS)
            or receipt.get("dataset_manifest_sha256") != sha256_file(bundle / "manifest.json")
            or receipt.get("calibration_plan_sha256") != manifest["calibration_plan"]["sha256"]
            or type(receipt.get("episode_count")) is not int
            or receipt.get("episode_count") != PRIMARY_CALIBRATION_COUNT):
            raise ValueError("calibration receipt is inconsistent")
        path = checked_relative_file(calibration_dir, receipt["data_path"])
        require_hash(path, receipt["data_sha256"])
        # Inspect structure only after materialization; no risk or label summaries.
        read_episode_file(path, PRIMARY_CALIBRATION_COUNT)
        state = "materialized_after_recorded_freeze"
    return {"schema": "kahm.bundle_verification.v1", "status": "passed",
            "contract_sha256": CONTRACT_SHA256, "run_id": manifest["run_id"],
            "development_counts": counts, "calibration_count_reserved": plan["episode_count"],
            "calibration_state": state,
            "freeze_artifacts_rechecked": False,
            "note": "Integrity checks establish file consistency, not execution chronology or a risk certificate."}


def validate_freeze_record(bundle: Path, freeze_path: Path) -> dict:
    """Check structure and artifact identities of a later freeze record.

    The model freeze supplies the model-specific preflight: run the saved models, compare
    their canonical tables, verify save/reload, and record development selection.
    This data-layer checker does not execute model artifacts or prove chronology.
    """
    bundle, freeze_path = Path(bundle), Path(freeze_path)
    manifest, _ = _load_bundle_records(bundle)
    freeze = read_json(freeze_path)
    if not isinstance(freeze, dict):
        raise ValueError("model-freeze record must be an object")
    expected = {"schema": FREEZE_SCHEMA, "status": "frozen",
                "contract_sha256": CONTRACT_SHA256, "run_id": manifest["run_id"],
                "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
                "calibration_plan_sha256": manifest["calibration_plan"]["sha256"]}
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise ValueError(f"freeze record mismatch: {key}")
    models = freeze.get("models")
    if (not isinstance(models, list) or len(models) != 3
        or any(not isinstance(m, dict) for m in models)
        or {m.get("candidate_id") for m in models} != set(CANDIDATE_IDS)):
        raise ValueError("freeze must identify exactly the three declared candidates")
    inference = freeze.get("inference")
    if (not isinstance(inference, dict) or not isinstance(inference.get("runtime"), dict)
        or not inference["runtime"] or not isinstance(inference.get("canonical_method"), str)
        or not inference["canonical_method"]
        or not isinstance(inference.get("source_files"), list) or not inference["source_files"]):
        raise ValueError("freeze needs runtime, canonical method and inference-source identities")
    for record in inference["source_files"]:
        path = checked_relative_file(freeze_path.parent, record["path"])
        require_hash(path, record["sha256"])
    for model in models:
        artifact = checked_relative_file(freeze_path.parent, model["artifact_path"])
        require_hash(artifact, model["artifact_sha256"])
        predictions = checked_relative_file(freeze_path.parent, model["prediction_table_path"])
        require_hash(predictions, model["prediction_table_sha256"])
        table = read_json(predictions)
        if table.get("candidate_id") != model["candidate_id"]:
            raise ValueError("prediction table is assigned to the wrong candidate")
        parse_prediction_table(table)
    return freeze


def _model_freeze_preflight(freeze_path: Path, *, dataset_bundle: Path) -> dict:
    """Run the model-specific gate without introducing an import-time cycle.

    The verifier checks actual saved models, canonical prediction replay, and
    source/runtime identities. It must not inspect calibration outcomes. Keeping
    this import local also leaves the independent structural/data utilities
    usable without importing the numerical model implementation.
    """
    from .freeze import verify_frozen_bundle

    return verify_frozen_bundle(Path(freeze_path), dataset_bundle=Path(dataset_bundle))


def _calibration_directory_outside_freeze(bundle: Path, freeze_path: Path) -> Path:
    """Keep mutable outcomes outside the sealed, outcome-free snapshot tree."""
    directory = bundle / "calibration"
    if directory.resolve().is_relative_to(freeze_path.parent.resolve()):
        raise ValueError("calibration outcomes must remain outside the frozen snapshot")
    return directory


def materialize_calibration(bundle: Path, freeze_path: Path) -> dict:
    """Generate an external reserved stream after full and structural preflights."""
    bundle, freeze_path = Path(bundle), Path(freeze_path)
    output = bundle / "calibration"
    # Reject existing data before verify_bundle can inspect their episode rows.
    # A dangling symlink also reserves the output path and must not be replaced.
    if output.exists() or output.is_symlink():
        raise FileExistsError("calibration has already been materialized; archived data are not replaced")
    output = _calibration_directory_outside_freeze(bundle, freeze_path)
    # The full preflight validates declared data paths before loading any plan;
    # it must precede the independently usable structural check below.
    _model_freeze_preflight(freeze_path, dataset_bundle=bundle)
    freeze = validate_freeze_record(bundle, freeze_path)
    verify_bundle(bundle)
    manifest, plan = _load_bundle_records(bundle)
    stage = Path(tempfile.mkdtemp(prefix=".calibration-", dir=bundle))
    try:
        episodes = sample_episodes(PRIMARY_CALIBRATION_COUNT,
                                   int(plan["master_seed_integer"]), "calibration", plan["run_id"])
        _write_episodes(stage / "episodes.jsonl", episodes)
        receipt = {"schema": "kahm.calibration_receipt.v1",
                   "status": "materialized_after_recorded_freeze",
                   "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
                   "calibration_plan_sha256": manifest["calibration_plan"]["sha256"],
                   "freeze_record_sha256": sha256_file(freeze_path),
                   "candidate_ids": sorted(m["candidate_id"] for m in freeze["models"]),
                   "episode_count": PRIMARY_CALIBRATION_COUNT, "data_path": "episodes.jsonl",
                   "data_sha256": sha256_file(stage / "episodes.jsonl")}
        write_json(stage / "receipt.json", receipt)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt


def load_calibration(bundle: Path, freeze_path: Path) -> Dataset:
    """Read outcomes only after model preflight and matching freeze binding."""
    bundle, freeze_path = Path(bundle), Path(freeze_path)
    directory = _calibration_directory_outside_freeze(bundle, freeze_path)
    _model_freeze_preflight(freeze_path, dataset_bundle=bundle)
    validate_freeze_record(bundle, freeze_path)
    if not directory.exists():
        raise ValueError("primary calibration remains reserved")
    # A different otherwise-valid freeze must not authorize reading these rows.
    # Inspect the outcome-free receipt before verify_bundle's episode checks.
    receipt = read_json(directory / "receipt.json")
    if not isinstance(receipt, dict):
        raise ValueError("calibration receipt must be an object")
    if receipt.get("freeze_record_sha256") != sha256_file(freeze_path):
        raise ValueError("calibration was materialized for a different freeze record")
    verification = verify_bundle(bundle)
    if verification["calibration_state"] != "materialized_after_recorded_freeze":
        raise ValueError("primary calibration remains reserved")
    manifest, _ = _load_bundle_records(bundle)
    data_path = checked_relative_file(directory, receipt["data_path"])
    return Dataset("calibration", manifest["run_id"],
                   read_episode_file(data_path, PRIMARY_CALIBRATION_COUNT))


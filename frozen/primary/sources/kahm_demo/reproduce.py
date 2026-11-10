"""Run one fresh, retained demonstration through the existing verified stages.

This onboarding wrapper does not implement fitting, sampling, certification, or
population arithmetic. It calls the established stages once, in order, and only
starts the population audit after the certificate workflow has completed and
verified its artifact. Pipeline completion is independent of certificate
acceptance and realized coverage. Replaying a seed/run identity is not new
independent evidence.

The plan is committed before the first stage. Existing output is never resumed
or replaced, and failures retain the output created so far. Progress callbacks
receive stage names; a callback exception stops the run without a retry.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
from typing import Callable

from ._util import canonical_json_bytes, read_json, sha256_file
from .audit_workflow import audit_certificate_family
from .calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol
from .certificate_workflow import certify_frozen_family
from .contract import CANDIDATE_IDS, CONTRACT_SHA256, PRIMARY_CALIBRATION_COUNT, validate_contract
from .datasets import generate_bundle
from .encoder_workflow import fit_development_encoders
from .forecast_workflow import fit_development_forecasters, forecast_runtime_record, load_readout_specification
from .freeze import freeze_forecasters
from .sampling import stream_descriptor


REPRODUCTION_PLAN_SCHEMA = "kahm.demo_reproduction_plan.v1"
REPRODUCTION_SCHEMA = "kahm.demo_reproduction.v1"
REPRODUCTION_FAILURE_SCHEMA = "kahm.demo_reproduction_failure.v1"
STAGES = ("generate_data", "fit_encoders", "fit_forecasters", "freeze_models", "certify", "audit")
PROGRESS_STAGES = STAGES + ("complete",)
ENCODER_CAPS = (1, 2)
FORECAST_SEARCH_COUNT = 225

_PATHS = {
    "data": "data", "encoders": "encoders", "models": "models",
    "frozen": "frozen", "freeze": "frozen/freeze.json",
    "certificates": "certificates", "certificate": "certificates/certificate.json",
    "audits": "audits", "audit": "audits/audit.json",
}
_ARTIFACT_PATHS = {
    "dataset_manifest": "data/manifest.json",
    "encoder_manifest": "encoders/manifest.json",
    "model_manifest": "models/manifest.json",
    "freeze": "frozen/freeze.json",
    "certificate": "certificates/certificate.json",
    "audit": "audits/audit.json",
}
_BUNDLE_MARKERS = {
    "manifest.json": frozenset({"kahm.dataset_bundle.v1", "kahm.encoder_development_bundle.v1",
                                "kahm.forecast_development_bundle.v1"}),
    "receipt.json": frozenset({"kahm.calibration_receipt.v1"}),
    "calibration_plan.json": frozenset({"kahm.calibration_plan.v1"}),
    "freeze.json": frozenset({"kahm.model_freeze.v1"}),
    "assessment_plan.json": frozenset({"kahm.calibration_assessment_plan.v1"}),
    "certificate.json": frozenset({"kahm.fitted_risk_certificate_family.v1"}),
    "assessment_failure.json": frozenset({"kahm.assessment_failure.v1"}),
    "audit_plan.json": frozenset({"kahm.population_audit_plan.v1"}),
    "audit.json": frozenset({"kahm.certified_family_population_audit.v1"}),
    "audit_failure.json": frozenset({"kahm.population_audit_failure.v1"}),
    "reproduction_plan.json": frozenset({REPRODUCTION_PLAN_SCHEMA}),
    "reproduction.json": frozenset({REPRODUCTION_SCHEMA}),
    "reproduction_failure.json": frozenset({REPRODUCTION_FAILURE_SCHEMA}),
}
_GENERIC_MARKERS = frozenset({"manifest.json", "receipt.json"})
_CERTIFICATE_STATUSES = ("requirement_met", "unresolved", "certificate_unavailable")


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _new_output(output: Path) -> Path:
    """Reject protected ancestors on both lexical and resolved symlink routes."""
    requested = Path(output)
    if _present(requested):
        raise FileExistsError(f"reproduction output already exists: {requested}; no overwrite or resume is supported")
    lexical = Path(os.path.abspath(requested))
    resolved = requested.resolve()
    inspected = set()
    for route in (lexical, resolved):
        for ancestor in route.parents:
            if ancestor in inspected:
                continue
            inspected.add(ancestor)
            if ancestor.is_symlink() and not ancestor.exists():
                raise ValueError(f"output passes through a dangling symlink: {ancestor}")
            if not ancestor.exists():
                continue
            if not ancestor.is_dir():
                raise ValueError(f"output ancestor is not a directory: {ancestor}")
            for name, schemas in _BUNDLE_MARKERS.items():
                marker = ancestor / name
                if not _present(marker):
                    continue
                if not marker.is_file():
                    raise ValueError(f"cannot inspect possible bundle marker: {marker}")
                try:
                    record = read_json(marker)
                except (OSError, UnicodeError, ValueError) as error:
                    raise ValueError(f"cannot inspect possible bundle marker: {marker}") from error
                schema = record.get("schema") if isinstance(record, dict) else None
                if schema in schemas or name not in _GENERIC_MARKERS:
                    raise ValueError(f"reproduction output must be outside existing bundles: {marker}")
    return resolved


def _digest(record: dict) -> str:
    return hashlib.sha256(canonical_json_bytes({key: value for key, value in record.items()
                                               if key != "record_sha256"})).hexdigest()


def _write_new(path: Path, record: dict) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(record))


def _artifact(root: Path, relative: str) -> dict:
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"missing or redirected reproduction artifact: {relative}")
    return {"path": relative, "sha256": sha256_file(path)}


def _check_plan(root: Path, plan_hash: str, source_hash: str) -> None:
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("reproduction output was redirected during execution")
    if _artifact(root, "reproduction_plan.json")["sha256"] != plan_hash:
        raise ValueError("committed reproduction plan changed during execution")
    if sha256_file(Path(__file__)) != source_hash:
        raise ValueError("reproduction implementation changed during execution")


def _require_status(result: dict, status: str, *, verified: bool = False) -> None:
    if not isinstance(result, dict) or result.get("status") != status:
        raise ValueError(f"stage did not report the required completion status: {status}")
    if verified and (not isinstance(result.get("verification"), dict)
                     or result["verification"].get("status") != "passed"):
        raise ValueError("stage did not complete its existing artifact verification")


def _ordered_rows(rows: object, status_key: str) -> list[dict]:
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or [row.get("candidate_id") for row in rows] != list(CANDIDATE_IDS)
            or any(row.get(status_key) not in _CERTIFICATE_STATUSES for row in rows)):
        raise ValueError("completed artifact does not retain the exact ordered three-model family")
    return rows


def reproduce_demo(output: Path, *, master_seed: int = 20261110,
                   run_id: str = "tutorial-v0.1", training_count: int = 4096,
                   validation_count: int = 1024,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Create one complete fresh run, retaining every assessment/audit outcome.

    ``paths`` inside the saved record are relative to the reproduction directory.
    The returned dictionary adds absolute ``output``/``reproduction_path`` and
    the completed file's SHA-256. No stage is retried. Smaller development counts
    do not alter the full search, fixed 4096 calibration size, or confidence menu.
    """
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    for name, value in (("training_count", training_count), ("validation_count", validation_count)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # These operations validate metadata and fixed resources, never draw rows.
    streams = {role: stream_descriptor(master_seed, role, run_id)
               for role in ("training", "validation", "calibration")}
    validate_contract()
    load_calibration_protocol()
    readout = load_readout_specification()
    if (readout["expected_search_counts"]["total"] != FORECAST_SEARCH_COUNT
            or readout["encoder_bank_order"] != ["cap_1", "cap_2"]):
        raise ValueError("the fixed full-search menu differs from this reproduction entry point")
    root = _new_output(output)
    from . import __version__
    source_hash = sha256_file(Path(__file__))
    plan = {
        "schema": REPRODUCTION_PLAN_SCHEMA, "status": "planned", "package_version": __version__,
        "contract_sha256": CONTRACT_SHA256, "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
        "run_id": run_id, "master_seed_integer": str(master_seed),
        "training_count": training_count, "validation_count": validation_count,
        "calibration_count": PRIMARY_CALIBRATION_COUNT, "streams": streams,
        "encoder_caps": list(ENCODER_CAPS), "forecast_search_count": FORECAST_SEARCH_COUNT,
        "candidate_ids": list(CANDIDATE_IDS), "stage_order": list(STAGES), "paths": _PATHS.copy(),
        "source_sha256": {"reproduce.py": source_hash}, "runtime": forecast_runtime_record(),
        "policy": {
            "automatic_retries": False, "resume_existing_output": False, "overwrite_existing_output": False,
            "discard_unfavorable_outcomes": False, "calibration_after_model_freeze_only": True,
            "population_audit_after_verified_certificate_only": True,
            "completion_is_independent_of_certificate_acceptance": True,
            "replay_is_new_independent_evidence": False,
        },
    }
    plan["record_sha256"] = _digest(plan)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after creating missing parents; do not follow a redirected route.
    if _new_output(output) != root:
        raise ValueError("output route changed before reproduction plan commitment")
    root.mkdir()
    completed: list[str] = []
    artifacts: dict[str, dict] = {}
    phase, plan_hash = "commit_plan", None
    try:
        _write_new(root / "reproduction_plan.json", plan)
        plan_hash = sha256_file(root / "reproduction_plan.json")

        def begin(stage: str, destination: str) -> None:
            nonlocal phase
            phase = stage
            if progress is not None:
                progress(stage)
            _check_plan(root, plan_hash, source_hash)
            path = root / destination
            if _present(path) or path.resolve().parent != root:
                raise FileExistsError(f"stage output is not fresh: {path}; no overwrite or resume is supported")

        begin("generate_data", "data")
        generated = generate_bundle(root / "data", master_seed=master_seed, run_id=run_id,
                                    training_count=training_count, validation_count=validation_count)
        _require_status(generated, "passed")
        artifacts["dataset_manifest"] = _artifact(root, _ARTIFACT_PATHS["dataset_manifest"])
        completed.append("generate_data")

        begin("fit_encoders", "encoders")
        encoded = fit_development_encoders(root / "data", root / "encoders", caps=ENCODER_CAPS)
        _require_status(encoded, "completed")
        if encoded.get("bank_order") != ["cap_1", "cap_2"]:
            raise ValueError("encoder stage did not retain both declared banks")
        artifacts["encoder_manifest"] = _artifact(root, _ARTIFACT_PATHS["encoder_manifest"])
        completed.append("fit_encoders")

        begin("fit_forecasters", "models")
        fitted = fit_development_forecasters(root / "data", root / "encoders", root / "models")
        _require_status(fitted, "completed")
        if type(fitted.get("search_count")) is not int or fitted["search_count"] != FORECAST_SEARCH_COUNT:
            raise ValueError("forecast stage did not complete the full 225-fit search")
        artifacts["model_manifest"] = _artifact(root, _ARTIFACT_PATHS["model_manifest"])
        completed.append("fit_forecasters")

        begin("freeze_models", "frozen")
        frozen = freeze_forecasters(root / "data", root / "encoders", root / "models", root / "frozen")
        _require_status(frozen, "frozen", verified=True)
        artifacts["freeze"] = _artifact(root, _ARTIFACT_PATHS["freeze"])
        if frozen.get("freeze_record_sha256") != artifacts["freeze"]["sha256"]:
            raise ValueError("freeze result does not bind the saved freeze")
        completed.append("freeze_models")

        begin("certify", "certificates")
        certified = certify_frozen_family(root / "data", root / "frozen/freeze.json", root / "certificates")
        _require_status(certified, "assessment_complete", verified=True)
        _ordered_rows(certified.get("candidates"), "status")
        if certified.get("calibration_action") != "materialize_reserved_stream":
            raise ValueError("fresh reproduction did not materialize its unused reserved calibration stream")
        if (type(certified.get("calibration_count")) is not int
                or certified["calibration_count"] != PRIMARY_CALIBRATION_COUNT):
            raise ValueError("certificate used a different calibration count")
        artifacts["certificate"] = _artifact(root, _ARTIFACT_PATHS["certificate"])
        if certified.get("certificate_sha256") != artifacts["certificate"]["sha256"]:
            raise ValueError("certificate result does not bind the saved assessment")
        completed.append("certify")

        begin("audit", "audits")
        audited = audit_certificate_family(root / "certificates/certificate.json", dataset_bundle=root / "data",
                                            freeze_path=root / "frozen/freeze.json", output=root / "audits")
        _require_status(audited, "audit_complete", verified=True)
        artifacts["audit"] = _artifact(root, _ARTIFACT_PATHS["audit"])
        if audited.get("audit_sha256") != artifacts["audit"]["sha256"]:
            raise ValueError("audit result does not bind the saved audit")
        completed.append("audit")

        phase = "complete"
        if progress is not None:
            progress("complete")
        _check_plan(root, plan_hash, source_hash)
        for identity in artifacts.values():
            if _artifact(root, identity["path"]) != identity:
                raise ValueError(f"completed artifact changed during reproduction: {identity['path']}")
        certificate = read_json(root / "certificates/certificate.json")
        audit = read_json(root / "audits/audit.json")
        cert_rows = _ordered_rows(certificate.get("candidates"), "status")
        audit_rows = _ordered_rows(audit.get("candidates"), "certificate_status")
        if (canonical_json_bytes(cert_rows) != canonical_json_bytes(certified["candidates"])
                or canonical_json_bytes(audit_rows) != canonical_json_bytes(audited.get("candidates"))
                or canonical_json_bytes(audit.get("summary")) != canonical_json_bytes(audited.get("summary"))
                or any(before["status"] != after["certificate_status"]
                       for before, after in zip(cert_rows, audit_rows, strict=True))):
            raise ValueError("reproduction results differ from the completed certificate or audit")
        counts = Counter(row["status"] for row in cert_rows)
        record = {
            "schema": REPRODUCTION_SCHEMA, "status": "pipeline_complete", "pipeline_complete": True,
            "package_version": __version__, "contract_sha256": CONTRACT_SHA256,
            "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
            "run_id": run_id, "master_seed_integer": str(master_seed),
            "training_count": training_count, "validation_count": validation_count,
            "calibration_count": PRIMARY_CALIBRATION_COUNT,
            "encoder_caps": list(ENCODER_CAPS), "forecast_search_count": FORECAST_SEARCH_COUNT,
            "reproduction_plan": {"path": "reproduction_plan.json", "sha256": plan_hash},
            "completed_stages": completed.copy(), "paths": _PATHS.copy(), "artifacts": artifacts,
            "candidate_ids": list(CANDIDATE_IDS), "candidates": audit_rows,
            "certificate_summary": {
                "all_three_requirements_met": counts["requirement_met"] == len(CANDIDATE_IDS),
                "any_kahm_requirement_met": any(row["status"] == "requirement_met"
                                                for row in cert_rows if row["candidate_id"] != "raw"),
                "status_counts": {status: counts[status] for status in _CERTIFICATE_STATUSES},
            },
            "audit_summary": audit["summary"], "automatic_retries_performed": False,
            "unfavorable_outcomes_filtered": False,
            "interpretation": "All workflow stages completed. Certificate acceptance and realized coverage remain separate retained results; the original conditional iid premises are unchanged. Reusing a seed/run identity supplies no new independent evidence.",
        }
        record["record_sha256"] = _digest(record)
        _write_new(root / "reproduction.json", record)
    except BaseException as error:
        failure = {
            "schema": REPRODUCTION_FAILURE_SCHEMA, "status": "pipeline_failed", "pipeline_complete": False,
            "phase": phase, "completed_stages": completed.copy(), "run_id": run_id,
            "master_seed_integer": str(master_seed), "reproduction_plan_sha256": plan_hash,
            "error_type": type(error).__name__, "message": str(error),
            "completed_artifacts": artifacts, "automatic_retries_performed": False,
            "instruction": "Retain this output and all completed or partial stage evidence. This entry point never resumes, overwrites, resamples, or retries an existing run; do not filter runs in response to certificate or audit outcomes.",
        }
        failure["record_sha256"] = _digest(failure)
        try:
            _write_new(root / "reproduction_failure.json", failure)
        except Exception as recording_error:
            error.add_note(f"Could not record reproduction failure without overwriting evidence: {recording_error}")
        raise
    return {**record, "output": str(root), "reproduction_path": str(root / "reproduction.json"),
            "reproduction_sha256": sha256_file(root / "reproduction.json")}

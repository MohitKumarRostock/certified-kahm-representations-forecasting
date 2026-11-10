"""Assess one frozen family under the exact, predeclared calibration protocol.

The plan and implementing source snapshot are committed before any calibration
outcome is generated or loaded. A reserved stream is materialized once; an
existing matching receipt is reused without resampling. Every frozen candidate
is retained, including numerical failures. Verification recomputes the exact
loss ledger and directly checks endpoints; it does not run the root search.

The probability statement is conditional on the stipulated iid evaluation law
and independence from development. PRNG metadata provides reproducibility, not
an independent proof of that statistical premise. This module never calculates
the fitted models' population oracle risks or refits a predictor.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from . import __version__
from ._util import canonical_json_bytes, checked_relative_file, read_json, sha256_file, write_json
from .calibration_ledger import write_calibration_ledger, verify_calibration_ledger
from .calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol
from .contract import CANDIDATE_IDS, CONTRACT_SHA256, PRIMARY_CALIBRATION_COUNT
from .datasets import load_calibration, materialize_calibration
from .forecast_workflow import forecast_runtime_record
from .freeze import verify_frozen_bundle
from .oracle import parse_prediction_table
from .risk_bounds import certify_upper_bound, verify_upper_bound
from .verified_arithmetic import WorkBudget


ASSESSMENT_PLAN_SCHEMA = "kahm.calibration_assessment_plan.v1"
CERTIFICATE_SCHEMA = "kahm.fitted_risk_certificate_family.v1"
CERTIFIER_SOURCE_FILES = (
    "calibration_ledger.py", "certificate_workflow.py", "risk_bounds.py", "verified_arithmetic.py",
)
_CERTIFICATE_FIELDS = {
    "schema", "status", "contract_sha256", "calibration_protocol_sha256", "freeze_record_sha256",
    "assessment_plan_path", "assessment_plan_sha256", "calibration_identity", "ledger_path",
    "ledger_sha256", "ledger_summary", "candidates", "guarantee", "family_work",
    "all_three_requirements_met", "any_kahm_requirement_met", "model_population_risk_evaluated",
    "models_refitted", "files", "record_sha256",
}


def _digest(record: dict) -> str:
    return hashlib.sha256(canonical_json_bytes({k: v for k, v in record.items()
                                               if k != "record_sha256"})).hexdigest()


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _fraction(record: dict) -> Fraction:
    if not isinstance(record, dict) or set(record) != {"numerator", "denominator"}:
        raise ValueError("invalid exact rational record")
    a, b = record["numerator"], record["denominator"]
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("rational numerator and denominator must be decimal integer strings")
    try:
        numerator, denominator = int(a), int(b)
    except ValueError as error:
        raise ValueError("invalid rational integer encoding") from error
    if str(numerator) != a or str(denominator) != b or denominator <= 0:
        raise ValueError("noncanonical rational integer encoding")
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise ValueError("rational record is not reduced")
    return value


def _source_identities() -> dict:
    package = Path(__file__).parent
    return {name: sha256_file(package / name) for name in CERTIFIER_SOURCE_FILES}


def _verified_json(path: Path, expected_hash: str) -> dict:
    """Hash and parse the same bytes, including for post-preflight table use."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"artifact changed after preflight: {path.name}")
    return json.loads(data)


def _frozen_inputs(bundle: Path, freeze_path: Path) -> tuple[dict, dict, dict]:
    checked = verify_frozen_bundle(freeze_path, dataset_bundle=bundle)
    freeze = _verified_json(freeze_path, checked["freeze_record_sha256"])
    predictions = {}
    for row in freeze["models"]:
        path = checked_relative_file(freeze_path.parent, row["prediction_table_path"])
        table = _verified_json(path, row["prediction_table_sha256"])
        if table.get("candidate_id") != row["candidate_id"]:
            raise ValueError("frozen prediction table candidate mismatch")
        predictions[row["candidate_id"]] = parse_prediction_table(table)
    if list(predictions) != list(CANDIDATE_IDS):
        raise ValueError("assessment requires the exact ordered frozen family")
    return checked, freeze, predictions


def _plan_record(bundle: Path, freeze: dict, freeze_hash: str, action: str) -> dict:
    protocol = load_calibration_protocol()
    if action not in ("materialize_reserved_stream", "reuse_matching_archived_calibration"):
        raise ValueError("unsupported calibration action")
    plan = {
        "schema": ASSESSMENT_PLAN_SCHEMA, "status": "fixed_before_this_assessment_calibration_access",
        "contract_sha256": CONTRACT_SHA256, "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
        "freeze_record_sha256": freeze_hash, "candidate_ids": list(CANDIDATE_IDS),
        "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "calibration_plan_sha256": freeze["calibration_plan_sha256"],
        "run_id": freeze["run_id"], "calibration_count": PRIMARY_CALIBRATION_COUNT,
        "calibration_action": action, "package_version": __version__,
        "source_sha256": _source_identities(), "source_roster": list(CERTIFIER_SOURCE_FILES),
        "runtime": forecast_runtime_record(),
        "total_failure_probability": protocol["confidence"]["total_failure_probability"],
        "per_candidate_failure_probability": protocol["confidence"]["per_candidate_failure_probability"],
        "primary_mse_tolerance": protocol["reporting"]["primary_tolerance"],
        "protocol_changes_after_outcomes_permitted": False,
        "model_selection_or_refitting_permitted": False,
        "population_oracle_risk_use_permitted": False,
    }
    plan["record_sha256"] = _digest(plan)
    return plan


def _verify_plan(root: Path, bundle: Path, freeze: dict, freeze_hash: str) -> dict:
    plan = read_json(root / "assessment_plan.json")
    if not isinstance(plan, dict) or plan.get("record_sha256") != _digest(plan):
        raise ValueError("assessment plan record identity differs")
    # Package version is historical metadata. The fixed numerical module roster
    # and current source hashes must still match exactly.
    expected = _plan_record(bundle, freeze, freeze_hash, plan.get("calibration_action"))
    expected["package_version"] = plan.get("package_version")
    expected["record_sha256"] = _digest(expected)
    if not isinstance(plan.get("package_version"), str) or not plan["package_version"] or plan != expected:
        raise ValueError("assessment plan differs from the fixed protocol, family, data, or implementation")
    for name, digest in plan["source_sha256"].items():
        if sha256_file(checked_relative_file(root, f"sources/{name}")) != digest:
            raise ValueError("archived certifier source identity differs")
    if {p.name for p in (root / "sources").iterdir() if p.is_file()} != set(CERTIFIER_SOURCE_FILES):
        raise ValueError("archived certifier dependency roster differs")
    return plan


def _inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("certificate artifacts must not contain symlinks")
        if path.is_file() and path != root / "certificate.json":
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)})
    return rows


def _work_record(budget: WorkBudget) -> dict:
    return {"series_terms": budget.series_terms,
            "maximum_integer_bits_seen": budget.maximum_integer_bits_seen,
            "maximum_series_terms": budget.maximum_series_terms,
            "maximum_integer_bits": budget.maximum_integer_bits}


def _guarantee() -> dict:
    protocol = load_calibration_protocol()
    return {
        "type": "simultaneous_one_sided_fitted_mse_upper_bounds",
        "family_size": 3,
        "simultaneous_confidence": protocol["confidence"]["simultaneous_coverage"],
        "total_failure_probability": protocol["confidence"]["total_failure_probability"],
        "per_candidate_failure_probability": protocol["confidence"]["per_candidate_failure_probability"],
        "primary_mse_tolerance": protocol["reporting"]["primary_tolerance"],
        "risk": "population scalar MSE of the actual frozen predictor under the declared benchmark law",
        "statistical_premises": "iid calibration episodes from the stipulated law, independent of all development; bounded squared losses",
        "sampling_premise_status": "declared statistical model; not independently proved by deterministic PRNG metadata",
        "coverage_statement": "Conditional on development and under the stipulated premises, the probability that any reported valid upper endpoint fails to cover its fitted risk is at most 1/20. No additional conditioning on availability or acceptance is used.",
        "unavailable_candidates": "retained without an endpoint; their probability allocation is not reassigned",
        "evaluation_family": "the three selected predictors frozen before calibration",
        "between_candidate_independence_required": False,
        "per_prediction_interval_coverage_claimed": False,
        "class_optimality_or_representation_superiority_claimed": False,
    }


def _candidate_summary(candidate: str, freeze_row: dict, mean: dict, bound: dict, root: Path) -> dict:
    relative = f"bounds/{candidate}.json"
    return {"candidate_id": candidate, "bank_id": freeze_row["bank_id"],
            "frozen_state_sha256": freeze_row["state_sha256"],
            "bound_path": relative, "bound_sha256": sha256_file(root / relative),
            "empirical_mse": mean, "status": bound["status"],
            "upper_endpoint": bound["endpoint"], "upper_endpoint_decimal_18": bound["endpoint_decimal_18"],
            "failure": bound["failure"]}


def certify_frozen_family(bundle: Path, freeze_path: Path, output: Path) -> dict:
    """Write a complete assessment, materializing only an unused reserved stream."""
    bundle, freeze_path, output = map(Path, (bundle, freeze_path, output))
    if _present(output):
        raise FileExistsError(f"assessment output already exists: {output}; existing evidence is not replaced")
    if output.resolve().is_relative_to(bundle.resolve()) or output.resolve().is_relative_to(freeze_path.parent.resolve()):
        raise ValueError("assessment output must be outside the data bundle and frozen snapshot")
    checked, freeze, predictions = _frozen_inputs(bundle, freeze_path)
    action = ("reuse_matching_archived_calibration" if _present(bundle / "calibration")
              else "materialize_reserved_stream")
    plan = _plan_record(bundle, freeze, checked["freeze_record_sha256"], action)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kahm-assessment-", dir=output.parent))
    try:
        (stage / "sources").mkdir()
        for name in CERTIFIER_SOURCE_FILES:
            shutil.copyfile(Path(__file__).parent / name, stage / "sources" / name)
        write_json(stage / "assessment_plan.json", plan)
        _verify_plan(stage, bundle, freeze, checked["freeze_record_sha256"])
        # This committed plan is retained even if later outcome access or a
        # numerical calculation fails. The reserved data are never discarded.
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    phase = "calibration_materialization_or_load"
    try:
        if action == "materialize_reserved_stream":
            materialize_calibration(bundle, freeze_path)
        dataset = load_calibration(bundle, freeze_path)
        receipt = read_json(bundle / "calibration/receipt.json")
        receipt_path = output / "calibration_receipt.json"
        shutil.copyfile(bundle / "calibration/receipt.json", receipt_path)
        phase = "exact_loss_ledger"
        ledger_path = output / "loss_ledger.jsonl"
        ledger = write_calibration_ledger(dataset, predictions, ledger_path)
        phase = "verified_upper_endpoints"
        (output / "bounds").mkdir()
        budget, candidates = WorkBudget(), []
        for candidate, freeze_row in zip(CANDIDATE_IDS, freeze["models"], strict=True):
            mean = ledger["empirical_risks"][candidate]
            bound = certify_upper_bound(_fraction(mean), budget)
            if bound["status"] not in ("requirement_met", "unresolved", "certificate_unavailable"):
                raise ValueError("numerical certifier returned an unsupported status")
            write_json(output / "bounds" / f"{candidate}.json", bound)
            candidates.append(_candidate_summary(candidate, freeze_row, mean, bound, output))
        phase = "assessment_record"
        # Check the committed implementation and unchanged frozen identity again
        # before publishing a numerical claim; never silently switch algorithms.
        _verify_plan(output, bundle, freeze, checked["freeze_record_sha256"])
        if sha256_file(freeze_path) != checked["freeze_record_sha256"]:
            raise ValueError("frozen family changed during assessment")
        record = {
            "schema": CERTIFICATE_SCHEMA, "status": "assessment_complete",
            "contract_sha256": CONTRACT_SHA256, "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
            "freeze_record_sha256": checked["freeze_record_sha256"],
            "assessment_plan_path": "assessment_plan.json", "assessment_plan_sha256": sha256_file(output / "assessment_plan.json"),
            "calibration_identity": {"run_id": dataset.run_id, "episode_count": len(dataset.episodes),
                                     "receipt_path": "calibration_receipt.json", "receipt_sha256": sha256_file(receipt_path),
                                     "data_sha256": receipt["data_sha256"],
                                     "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
                                     "calibration_plan_sha256": receipt["calibration_plan_sha256"]},
            "ledger_path": "loss_ledger.jsonl", "ledger_sha256": sha256_file(ledger_path), "ledger_summary": ledger,
            "candidates": candidates, "guarantee": _guarantee(), "family_work": _work_record(budget),
            "all_three_requirements_met": all(row["status"] == "requirement_met" for row in candidates),
            "any_kahm_requirement_met": any(row["status"] == "requirement_met" for row in candidates if row["candidate_id"] != "raw"),
            "model_population_risk_evaluated": False, "models_refitted": False, "files": _inventory(output),
        }
        record["record_sha256"] = _digest(record)
        write_json(output / "certificate.json", record)
        phase = "independent_artifact_verification"
        verification = verify_certificate_bundle(output / "certificate.json", dataset_bundle=bundle, freeze_path=freeze_path)
    except BaseException as error:
        write_json(output / "assessment_failure.json", {
            "schema": "kahm.assessment_failure.v1", "status": "assessment_failed", "phase": phase,
            "error_type": type(error).__name__, "message": str(error),
            "candidate_ids": list(CANDIDATE_IDS), "calibration_data_retained": _present(bundle / "calibration"),
            "instruction": "Retain the existing reservation, outcomes, and failure evidence. Do not replace calibration or change the fixed protocol in response to outcomes.",
        })
        raise
    return {"status": "assessment_complete", "certificate_path": str(output / "certificate.json"),
            "certificate_sha256": sha256_file(output / "certificate.json"),
            "freeze_record_sha256": record["freeze_record_sha256"], "calibration_action": action,
            "calibration_count": len(dataset.episodes), "candidates": candidates,
            "all_three_requirements_met": record["all_three_requirements_met"],
            "any_kahm_requirement_met": record["any_kahm_requirement_met"], "verification": verification}


def verify_certificate_bundle(certificate_path: Path, *, dataset_bundle: Path, freeze_path: Path) -> dict:
    """Recompute exact observed losses and verify endpoints without root search."""
    certificate_path, bundle, freeze_path = map(Path, (certificate_path, dataset_bundle, freeze_path))
    if certificate_path.name != "certificate.json":
        raise ValueError("certificate bundle record must be named certificate.json")
    root = certificate_path.parent
    if root.resolve().is_relative_to(bundle.resolve()) or root.resolve().is_relative_to(freeze_path.parent.resolve()):
        raise ValueError("certificate evidence must remain outside data and frozen model bundles")
    # The certificate and its ledger contain outcome-derived quantities. Check
    # the frozen family and outcome-free implementing plan before reading them.
    checked, freeze, predictions = _frozen_inputs(bundle, freeze_path)
    plan = _verify_plan(root, bundle, freeze, checked["freeze_record_sha256"])
    record = read_json(certificate_path)
    if not isinstance(record, dict) or set(record) != _CERTIFICATE_FIELDS or record.get("record_sha256") != _digest(record):
        raise ValueError("certificate record fields or payload identity differ")
    if record["schema"] != CERTIFICATE_SCHEMA or record["status"] != "assessment_complete":
        raise ValueError("unsupported certificate family record")
    if record["contract_sha256"] != CONTRACT_SHA256 or record["calibration_protocol_sha256"] != CALIBRATION_PROTOCOL_SHA256:
        raise ValueError("certificate scientific contract or calibration protocol differs")
    if record["files"] != _inventory(root):
        raise ValueError("certificate artifact inventory or hashes differ")
    if record["freeze_record_sha256"] != checked["freeze_record_sha256"]:
        raise ValueError("certificate belongs to a different frozen family")
    if (record["assessment_plan_path"] != "assessment_plan.json"
            or record["assessment_plan_sha256"] != sha256_file(root / "assessment_plan.json")):
        raise ValueError("certificate assessment-plan identity differs")
    dataset = load_calibration(bundle, freeze_path)
    receipt = read_json(bundle / "calibration/receipt.json")
    copied = checked_relative_file(root, "calibration_receipt.json")
    if copied.read_bytes() != (bundle / "calibration/receipt.json").read_bytes():
        raise ValueError("archived calibration receipt differs from the verified data receipt")
    expected_identity = {"run_id": dataset.run_id, "episode_count": len(dataset.episodes),
                         "receipt_path": "calibration_receipt.json", "receipt_sha256": sha256_file(copied),
                         "data_sha256": receipt["data_sha256"],
                         "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
                         "calibration_plan_sha256": receipt["calibration_plan_sha256"]}
    if record["calibration_identity"] != expected_identity:
        raise ValueError("certificate calibration identity differs")
    if record["ledger_path"] != "loss_ledger.jsonl":
        raise ValueError("unsupported calibration ledger path")
    ledger_path = checked_relative_file(root, record["ledger_path"])
    if record["ledger_sha256"] != sha256_file(ledger_path):
        raise ValueError("calibration loss ledger hash differs")
    ledger = verify_calibration_ledger(dataset, predictions, ledger_path)
    if ledger != record["ledger_summary"]:
        raise ValueError("certificate exact empirical losses differ from the observed episodes")
    candidates = record["candidates"]
    if not isinstance(candidates, list) or [row.get("candidate_id") for row in candidates] != list(CANDIDATE_IDS):
        raise ValueError("certificate does not retain the exact ordered frozen family")
    budget, checks = WorkBudget(), []
    generation_work = _work_record(WorkBudget())
    for row, freeze_row in zip(candidates, freeze["models"], strict=True):
        candidate = row["candidate_id"]
        path = checked_relative_file(root, f"bounds/{candidate}.json")
        bound = read_json(path)
        expected = _candidate_summary(candidate, freeze_row, ledger["empirical_risks"][candidate], bound, root)
        if row != expected:
            raise ValueError("candidate summary differs from its frozen identity or numerical record")
        check = verify_upper_bound(_fraction(ledger["empirical_risks"][candidate]), bound, budget)
        if check["status"] != row["status"]:
            raise ValueError("reported candidate status differs from direct endpoint verification")
        if row["status"] != "certificate_unavailable" and check.get("verified") is not True:
            raise ValueError("reported endpoint is not verified")
        if row["status"] == "certificate_unavailable" and check.get("unavailable_record_consistent") is not True:
            raise ValueError("unavailable certificate record is internally inconsistent")
        if bound["work"]["before"] != generation_work:
            raise ValueError("candidate generation work does not continue the shared family budget")
        generation_work = bound["work"]["after"]
        if (generation_work["maximum_integer_bits_seen"] > generation_work["maximum_integer_bits"]
                and (row["status"] != "certificate_unavailable"
                     or row["failure"]["code"] != "integer_bit_limit")):
            raise ValueError("observed integer-limit violation lacks consistent unavailable evidence")
        checks.append({"candidate_id": candidate, "status": row["status"],
                       "endpoint_verified": check.get("verified") is True,
                       "empirical_mse": row["empirical_mse"], "upper_endpoint": row["upper_endpoint"],
                       "upper_endpoint_decimal_18": row["upper_endpoint_decimal_18"]})
    if record["guarantee"] != _guarantee():
        raise ValueError("certificate confidence allocation or guarantee scope differs")
    if any(record[key] is not False for key in ("model_population_risk_evaluated", "models_refitted")):
        raise ValueError("certificate declares unsupported oracle use or refitting")
    if record["all_three_requirements_met"] is not all(row["status"] == "requirement_met" for row in candidates):
        raise ValueError("family acceptance summary differs")
    if record["any_kahm_requirement_met"] is not any(row["status"] == "requirement_met" for row in candidates if row["candidate_id"] != "raw"):
        raise ValueError("KAHM acceptance summary differs")
    work = record["family_work"]
    protocol = load_calibration_protocol()
    limits = protocol["work_limits"]
    if (not isinstance(work, dict)
            or set(work) != {"series_terms", "maximum_integer_bits_seen", "maximum_series_terms", "maximum_integer_bits"}
            or any(type(value) is not int for value in work.values())
            or not 0 <= work["series_terms"] <= work["maximum_series_terms"] == limits["maximum_series_term_iterations_for_family_and_checker"]
            or work["maximum_integer_bits"] != limits["maximum_integer_bit_length"]
            or work["maximum_integer_bits_seen"] < 0
            or work != generation_work):
        raise ValueError("certificate generation work record exceeds or differs from the fixed limits")
    return {"schema": "kahm.certificate_verification.v1", "status": "passed",
            "certificate_sha256": sha256_file(certificate_path),
            "freeze_record_sha256": checked["freeze_record_sha256"],
            "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
            "calibration_count": len(dataset.episodes), "all_episode_losses_recomputed_exactly": True,
            "candidate_ids": list(CANDIDATE_IDS), "candidates": checks,
            "root_search_repeated": False, "calibration_resampled": False, "models_refitted": False,
            "model_population_risk_evaluated": False, "certifier_source_identity_matches": True,
            "certifier_runtime_matches": True, "direct_verification_work": _work_record(budget),
            "sampling_premise_status": record["guarantee"]["sampling_premise_status"],
            "interpretation": "Exact observed losses and direct numerical endpoint checks pass. Statistical coverage is conditional on the declared iid evaluation premises."}

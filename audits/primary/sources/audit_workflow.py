"""Audit already issued certificates against the exact stipulated benchmark.

This is a post-certificate reference calculation. It retains every frozen
candidate and every coverage outcome, and never changes a forecast, endpoint,
calibration observation, tolerance, or selection. The existing certificate is
verified before any model-specific population-risk computation. An outcome-free
audit plan and the two new source modules are committed before that computation.
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
from .calibration_protocol import CALIBRATION_PROTOCOL_SHA256
from .certificate_workflow import verify_certificate_bundle
from .contract import CANDIDATE_IDS, CONTRACT_SHA256, validate_contract
from .forecast_workflow import forecast_runtime_record
from .oracle import parse_prediction_table
from .population_audit import build_population_audit, verify_population_audit


AUDIT_SCHEMA = "kahm.certified_family_population_audit.v1"
PLAN_SCHEMA = "kahm.population_audit_plan.v1"
AUDIT_SOURCE_FILES = ("audit_workflow.py", "population_audit.py")
TOLERANCE = Fraction(1, 10)
_AUDIT_FIELDS = {
    "schema", "status", "contract_sha256", "calibration_protocol_sha256",
    "freeze_record_sha256", "certificate_sha256", "audit_plan_sha256",
    "run_id", "candidate_ids", "candidates", "summary", "references",
    "scope", "models_refitted", "calibration_resampled", "root_search_repeated",
    "issued_certificate_modified", "model_selection_performed", "files", "record_sha256",
}


def _digest(record: dict) -> str:
    return hashlib.sha256(canonical_json_bytes({key: value for key, value in record.items()
                                               if key != "record_sha256"})).hexdigest()


def _same(actual, expected) -> bool:
    # JSON-byte equality also distinguishes booleans, integers, and floats.
    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


def _fraction(record: dict) -> Fraction:
    if type(record) is not dict or set(record) != {"numerator", "denominator"}:
        raise ValueError("an audit fraction requires two canonical integer strings")
    a, b = record["numerator"], record["denominator"]
    if type(a) is not str or type(b) is not str:
        raise ValueError("audit fraction fields must be strings")
    numerator, denominator = int(a), int(b)
    if str(numerator) != a or str(denominator) != b or denominator <= 0:
        raise ValueError("noncanonical audit fraction integers")
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise ValueError("audit fraction must be reduced")
    return value


def _ratio(value: Fraction) -> dict:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def decimal_18(value: Fraction) -> str:
    """Approximate display, rounded to nearest with ties to even using integers."""
    if type(value) is not Fraction:
        raise TypeError("the audit decimal display requires an exact Fraction")
    scale = 10 ** 18
    quotient, remainder = divmod(abs(value.numerator) * scale, value.denominator)
    if 2 * remainder > value.denominator or (2 * remainder == value.denominator and quotient % 2):
        quotient += 1
    sign = "-" if value < 0 and quotient else ""
    return f"{sign}{quotient // scale}.{quotient % scale:018d}"


def _verified_json(path: Path, expected_hash: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"audit input changed after prerequisite verification: {path.name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def _certified_inputs(certificate_path: Path, bundle: Path, freeze_path: Path) -> tuple[dict, dict, dict, dict]:
    checked = verify_certificate_bundle(certificate_path, dataset_bundle=bundle, freeze_path=freeze_path)
    if checked.get("status") != "passed":
        raise ValueError("a fully verified existing certificate is required before population auditing")
    validate_contract()
    certificate = _verified_json(certificate_path, checked["certificate_sha256"])
    freeze = _verified_json(freeze_path, checked["freeze_record_sha256"])
    if ([row["candidate_id"] for row in certificate["candidates"]] != list(CANDIDATE_IDS)
            or [row["candidate_id"] for row in freeze["models"]] != list(CANDIDATE_IDS)):
        raise ValueError("the audit requires all three ordered frozen candidates")
    predictions = {}
    for row in freeze["models"]:
        path = checked_relative_file(freeze_path.parent, row["prediction_table_path"])
        table = _verified_json(path, row["prediction_table_sha256"])
        if table.get("candidate_id") != row["candidate_id"]:
            raise ValueError("frozen prediction-table candidate identity differs")
        predictions[row["candidate_id"]] = parse_prediction_table(table)
    return checked, certificate, freeze, predictions


def _outside_inputs(output: Path, certificate_path: Path, bundle: Path, freeze_path: Path) -> None:
    resolved = output.resolve()
    if any(resolved.is_relative_to(tree.resolve()) for tree in
           (certificate_path.parent, bundle, freeze_path.parent)):
        raise ValueError("audit evidence must remain outside data, certificate, and frozen-model bundles")


def _source_identities() -> dict:
    return {name: sha256_file(Path(__file__).parent / name) for name in AUDIT_SOURCE_FILES}


def _plan(checked: dict, freeze: dict) -> dict:
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "committed_after_certificate_verification_before_this_population_audit",
        "contract_sha256": CONTRACT_SHA256,
        "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
        "freeze_record_sha256": checked["freeze_record_sha256"],
        "certificate_sha256": checked["certificate_sha256"],
        "run_id": freeze["run_id"], "candidate_ids": list(CANDIDATE_IDS),
        "source_roster": list(AUDIT_SOURCE_FILES), "source_sha256": _source_identities(),
        "package_version": __version__, "runtime": forecast_runtime_record(),
        "history_count": 16, "triple_count_per_candidate": 64,
        "primary_mse_tolerance": _ratio(TOLERANCE),
        "reference_risk_use": "post_certificate_reference_audit_only",
        "coverage_miss_rule": "retain every issued endpoint and report every realized miss without repair or run filtering",
        "model_or_seed_selection_permitted": False,
        "refitting_or_resampling_permitted": False,
        "endpoint_recomputation_or_replacement_permitted": False,
    }
    plan["record_sha256"] = _digest(plan)
    return plan


def _verify_plan(root: Path, checked: dict, freeze: dict) -> dict:
    plan = read_json(root / "audit_plan.json")
    if type(plan) is not dict or plan.get("record_sha256") != _digest(plan):
        raise ValueError("audit plan identity differs")
    expected = _plan(checked, freeze)
    expected["package_version"] = plan.get("package_version")
    expected["record_sha256"] = _digest(expected)
    if type(plan.get("package_version")) is not str or not plan["package_version"] or not _same(plan, expected):
        raise ValueError("audit plan differs from its source, runtime, frozen family, or issued certificate")
    if {path.name for path in (root / "sources").iterdir()} != set(AUDIT_SOURCE_FILES):
        raise ValueError("audit source snapshot roster differs")
    for name, digest in plan["source_sha256"].items():
        if sha256_file(checked_relative_file(root, f"sources/{name}")) != digest:
            raise ValueError("archived audit source identity differs")
    return plan


def _inventory(root: Path) -> list[dict]:
    expected = {"audit_plan.json", *{f"sources/{name}" for name in AUDIT_SOURCE_FILES},
                *{f"population/{candidate}.json" for candidate in CANDIDATE_IDS}}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("audit artifacts must not contain symlinks")
        if path.is_file() and path != root / "audit.json":
            relative = path.relative_to(root).as_posix()
            if relative not in expected:
                raise ValueError("unexpected audit artifact")
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if {row["path"] for row in rows} != expected:
        raise ValueError("incomplete audit artifact inventory")
    return rows


def _comparison(candidate: str, certificate_row: dict, frozen_row: dict, exact: dict, root: Path) -> dict:
    risk = _fraction(exact["exact_population_mse"])
    q = _fraction(certificate_row["empirical_mse"])
    upper_record = certificate_row["upper_endpoint"]
    upper = None if upper_record is None else _fraction(upper_record)
    status = certificate_row["status"]
    if status not in ("requirement_met", "unresolved", "certificate_unavailable"):
        raise ValueError("unsupported issued certificate status")
    if (status == "certificate_unavailable") != (upper is None):
        raise ValueError("issued status and endpoint availability disagree")
    if not Fraction(0) <= q <= 1 or not Fraction(0) <= risk <= 1:
        raise ValueError("bounded MSE must lie in [0,1]")
    if upper is not None and (not q <= upper <= 1 or (status == "requirement_met") != (upper <= TOLERANCE)):
        raise ValueError("issued endpoint and certificate decision disagree")
    relative = f"population/{candidate}.json"
    return {
        "candidate_id": candidate, "bank_id": frozen_row["bank_id"],
        "frozen_state_sha256": frozen_row["state_sha256"],
        "prediction_table_sha256": frozen_row["prediction_table_sha256"],
        "population_audit_path": relative, "population_audit_sha256": sha256_file(root / relative),
        "calibration_empirical_mse": certificate_row["empirical_mse"],
        "exact_population_mse": exact["exact_population_mse"],
        "population_mse_decimal_18_approximate": decimal_18(risk),
        "bayes_reference_mse": exact["bayes_mse"],
        "fitted_excess_over_bayes": exact["excess_over_bayes"],
        "excess_over_bayes_decimal_18_approximate": decimal_18(_fraction(exact["excess_over_bayes"])),
        "certificate_status": status,
        "issued_upper_endpoint": upper_record,
        "issued_upper_endpoint_decimal_18": certificate_row["upper_endpoint_decimal_18"],
        "realized_coverage": ("endpoint_unavailable" if upper is None else "covered" if risk <= upper else "coverage_miss"),
        "upper_minus_population_risk": None if upper is None else _ratio(upper - risk),
        "empirical_minus_population_risk": _ratio(q - risk),
        "upper_minus_empirical_risk": None if upper is None else _ratio(upper - q),
        "primary_mse_tolerance": _ratio(TOLERANCE),
        "population_requirement_met": risk <= TOLERANCE,
        "independent_exact_decomposition_checked": True,
    }


def _summary(candidates: list[dict]) -> dict:
    covered = sum(row["realized_coverage"] == "covered" for row in candidates)
    misses = sum(row["realized_coverage"] == "coverage_miss" for row in candidates)
    unavailable = sum(row["realized_coverage"] == "endpoint_unavailable" for row in candidates)
    available = covered + misses
    return {
        "candidate_count": 3, "available_endpoint_count": available,
        "covered_endpoint_count": covered, "coverage_miss_count": misses,
        "unavailable_endpoint_count": unavailable,
        "all_three_endpoints_available": available == 3,
        "all_available_endpoints_cover": None if available == 0 else misses == 0,
        "all_three_covered": covered == 3,
        "population_requirement_met_count": sum(row["population_requirement_met"] for row in candidates),
        "all_three_population_requirements_met": all(row["population_requirement_met"] for row in candidates),
        "coverage_outcome": ("coverage_miss_retained" if misses else "all_three_covered" if covered == 3
                             else "no_endpoints_available" if available == 0 else "partially_available_no_observed_miss"),
    }


def _scope() -> dict:
    return {
        "object": "the three already certified frozen computational predictors under the stipulated finite benchmark law",
        "risk_arithmetic": "exact rational weighted enumeration of 64 triples, checked against an independent conditional-risk decomposition",
        "excess_interpretation": "fitted-predictor excess over the full-history Bayes risk; not representation information loss or decoder-class optimality",
        "empirical_minus_population_interpretation": "one realized calibration sampling deviation, not estimated systematic bias",
        "upper_minus_population_interpretation": "realized certificate slack, not a bound on numerical rounding error",
        "display": "population and excess decimals rounded to nearest, ties to even, at 18 places; issued upper decimal preserved unchanged",
        "coverage_interpretation": "a realized check of fixed endpoints; not a measurement of 95% coverage frequency or a new statistical guarantee",
        "statistical_premises": "the original certificate retains its conditional iid evaluation premises; this finite-law calculation does not establish those premises",
        "all_outcomes_retained": True, "general_representation_superiority_claimed": False,
        "confidence_allocation_changed": False, "winner_selected": False,
    }


def audit_certificate_family(certificate_path: Path, *, dataset_bundle: Path, freeze_path: Path, output: Path) -> dict:
    certificate_path, bundle, freeze_path, output = map(Path, (certificate_path, dataset_bundle, freeze_path, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"audit output already exists: {output}; existing evidence is retained")
    _outside_inputs(output, certificate_path, bundle, freeze_path)
    checked, certificate, freeze, predictions = _certified_inputs(certificate_path, bundle, freeze_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kahm-audit-", dir=output.parent))
    try:
        (stage / "sources").mkdir()
        for name in AUDIT_SOURCE_FILES:
            shutil.copyfile(Path(__file__).parent / name, stage / "sources" / name)
        write_json(stage / "audit_plan.json", _plan(checked, freeze))
        _verify_plan(stage, checked, freeze)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    phase = "exact_population_risk_calculation"
    try:
        (output / "population").mkdir()
        candidates, references = [], None
        for candidate, cert_row, frozen_row in zip(CANDIDATE_IDS, certificate["candidates"], freeze["models"], strict=True):
            exact = build_population_audit(predictions[candidate])
            if references is not None and not _same(references, exact["references"]):
                raise ValueError("benchmark reference calculations disagree across candidates")
            references = exact["references"]
            write_json(output / "population" / f"{candidate}.json", exact)
            candidates.append(_comparison(candidate, cert_row, frozen_row, exact, output))
        phase = "audit_record"
        _verify_plan(output, checked, freeze)
        if (sha256_file(certificate_path) != checked["certificate_sha256"]
                or sha256_file(freeze_path) != checked["freeze_record_sha256"]):
            raise ValueError("issued certificate or frozen family changed during auditing")
        record = {
            "schema": AUDIT_SCHEMA, "status": "audit_complete",
            "contract_sha256": CONTRACT_SHA256, "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
            "freeze_record_sha256": checked["freeze_record_sha256"], "certificate_sha256": checked["certificate_sha256"],
            "audit_plan_sha256": sha256_file(output / "audit_plan.json"),
            "run_id": freeze["run_id"], "candidate_ids": list(CANDIDATE_IDS),
            "candidates": candidates, "summary": _summary(candidates), "references": references, "scope": _scope(),
            "models_refitted": False, "calibration_resampled": False, "root_search_repeated": False,
            "issued_certificate_modified": False, "model_selection_performed": False, "files": _inventory(output),
        }
        record["record_sha256"] = _digest(record)
        write_json(output / "audit.json", record)
        phase = "independent_artifact_verification"
        verification = verify_audit_bundle(output / "audit.json", certificate_path=certificate_path,
                                          dataset_bundle=bundle, freeze_path=freeze_path)
    except BaseException as error:
        write_json(output / "audit_failure.json", {
            "schema": "kahm.population_audit_failure.v1", "status": "audit_failed", "phase": phase,
            "error_type": type(error).__name__, "message": str(error),
            "instruction": "Retain prior certificate, calibration, models, and all audit evidence. A coverage miss alone is a retained audit result, not a reason to rerun or repair an endpoint.",
        })
        raise
    return {"status": "audit_complete", "audit_path": str(output / "audit.json"),
            "audit_sha256": sha256_file(output / "audit.json"), "candidates": candidates,
            "summary": record["summary"], "verification": verification}


def verify_audit_bundle(audit_path: Path, *, certificate_path: Path, dataset_bundle: Path, freeze_path: Path) -> dict:
    audit_path, certificate_path, bundle, freeze_path = map(Path, (audit_path, certificate_path, dataset_bundle, freeze_path))
    if audit_path.name != "audit.json":
        raise ValueError("audit bundle record must be named audit.json")
    root = audit_path.parent
    _outside_inputs(root, certificate_path, bundle, freeze_path)
    # Validate existing evidence and the outcome-free audit plan before reading
    # the model-specific exact population-risk records.
    checked, certificate, freeze, predictions = _certified_inputs(certificate_path, bundle, freeze_path)
    _verify_plan(root, checked, freeze)
    record = read_json(audit_path)
    if type(record) is not dict or set(record) != _AUDIT_FIELDS or record.get("record_sha256") != _digest(record):
        raise ValueError("audit record fields or identity differ")
    if (record["schema"] != AUDIT_SCHEMA or record["status"] != "audit_complete"
            or record["contract_sha256"] != CONTRACT_SHA256
            or record["calibration_protocol_sha256"] != CALIBRATION_PROTOCOL_SHA256
            or record["freeze_record_sha256"] != checked["freeze_record_sha256"]
            or record["certificate_sha256"] != checked["certificate_sha256"]
            or record["audit_plan_sha256"] != sha256_file(root / "audit_plan.json")
            or record["run_id"] != freeze["run_id"] or record["candidate_ids"] != list(CANDIDATE_IDS)):
        raise ValueError("audit schema, contract, plan, or prerequisite identity differs")
    if not _same(record["files"], _inventory(root)):
        raise ValueError("audit file inventory or hashes differ")
    expected_candidates, references = [], None
    for candidate, cert_row, frozen_row in zip(CANDIDATE_IDS, certificate["candidates"], freeze["models"], strict=True):
        exact = read_json(checked_relative_file(root, f"population/{candidate}.json"))
        verify_population_audit(predictions[candidate], exact)
        if references is not None and not _same(references, exact["references"]):
            raise ValueError("audit benchmark references disagree")
        references = exact["references"]
        expected_candidates.append(_comparison(candidate, cert_row, frozen_row, exact, root))
    if not _same(record["candidates"], expected_candidates):
        raise ValueError("audit comparison differs from exact risks or issued endpoints")
    if not _same(record["summary"], _summary(expected_candidates)):
        raise ValueError("audit family summary does not retain the exact coverage outcomes")
    if not _same(record["references"], references) or not _same(record["scope"], _scope()):
        raise ValueError("audit reference values or interpretation differ")
    if any(record[key] is not False for key in ("models_refitted", "calibration_resampled", "root_search_repeated",
                                               "issued_certificate_modified", "model_selection_performed")):
        raise ValueError("audit declares an unsupported mutation or selection")
    return {
        "schema": "kahm.population_audit_verification.v1", "status": "passed",
        "audit_sha256": sha256_file(audit_path), "certificate_sha256": checked["certificate_sha256"],
        "freeze_record_sha256": checked["freeze_record_sha256"],
        "candidate_ids": list(CANDIDATE_IDS), "candidates": expected_candidates, "summary": record["summary"],
        "exact_population_risks_recomputed": True, "all_64_triples_per_candidate_checked": True,
        "independent_conditional_decomposition_checked": True, "preserved_issued_endpoints": True,
        "source_identity_matches": True, "runtime_metadata_matches": True,
        "calibration_resampled": False, "models_refitted": False, "root_search_repeated": False,
        "model_selection_performed": False, "issued_certificate_modified": False,
        "interpretation": "Exact benchmark risks and all realized coverage outcomes reproduce. This is a post-certificate audit, not a new confidence claim or an empirical coverage-frequency estimate.",
    }

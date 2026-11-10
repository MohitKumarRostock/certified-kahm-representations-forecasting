"""Read the released primary example, or explicitly verify and replay it.

``inspect_primary`` uses only the Python standard library. It checks fixed
primary artifact identities and their payload bytes, then displays recorded
values. Hash agreement is not a fresh model execution, certificate check, or
population-risk calculation. ``runtime_compatibility`` is a separate, lazy
runtime/source precheck. Only ``verify_primary`` invokes the strict numerical
verification chain and replays the selected-history forecasts.

These helpers neither create evidence nor provide a way to alter the primary
family. They deliberately support the released primary artifacts, rather than
accepting arbitrary replacement anchors or silently choosing another run.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath


# Presentation anchors are assigned only after the public evidence is issued.
# The frozen source snapshot retains these unassigned values; this presentation
# module is outside the scientific execution-source roster.
PRIMARY_FREEZE_SHA256 = "UNASSIGNED_PUBLIC_PRIMARY_FREEZE"
PRIMARY_CERTIFICATE_SHA256 = "UNASSIGNED_PUBLIC_PRIMARY_CERTIFICATE"
PRIMARY_AUDIT_SHA256 = "UNASSIGNED_PUBLIC_PRIMARY_AUDIT"
_CANDIDATES = ("raw", "kahm_mem", "kahm_local")
_OBSERVATIONS = (0.0, 0.25, 0.75, 1.0)
_RECORDS = (
    ("freeze", "frozen/primary/freeze.json", PRIMARY_FREEZE_SHA256,
     "kahm.model_freeze.v1", "frozen"),
    ("certificate", "certificates/primary/certificate.json", PRIMARY_CERTIFICATE_SHA256,
     "kahm.fitted_risk_certificate_family.v1", "assessment_complete"),
    ("audit", "audits/primary/audit.json", PRIMARY_AUDIT_SHA256,
     "kahm.certified_family_population_audit.v1", "audit_complete"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def history_from_observations(values) -> tuple[int, int]:
    """Convert two observations on the exact finite support to state labels.

    Compare the supplied real values before float conversion. In particular,
    an off-support rational that rounds to a support float remains invalid.
    """
    try:
        observations = tuple(values)
    except TypeError as error:
        raise TypeError("history observations must be an iterable of two real values") from error
    if len(observations) != 2:
        raise ValueError("history requires exactly two observations")
    labels = []
    for value in observations:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("history observations must be real numbers, not booleans")
        matches = [index for index, supported in enumerate(_OBSERVATIONS) if value == supported]
        if len(matches) != 1:
            raise ValueError("history observations must lie exactly in the finite support {0, 0.25, 0.75, 1}")
        labels.append(matches[0])
    return labels[0], labels[1]


def _history(history: tuple[int, int]) -> tuple[int, int]:
    if type(history) is not tuple or len(history) != 2 or any(type(value) is not int for value in history):
        raise TypeError("history must be a tuple of two integer state labels, not observation values or booleans")
    if any(value < 0 or value > 3 for value in history):
        raise ValueError("history state labels must lie in {0, 1, 2, 3}")
    return history


def _file(root: Path, relative: str) -> Path:
    if type(relative) is not str:
        raise ValueError("an archive path must be a relative POSIX string")
    value = PurePosixPath(relative)
    if (not value.parts or value.is_absolute() or ".." in value.parts
            or value.as_posix() != relative or "\\" in relative):
        raise ValueError("an archive path is unsafe or noncanonical")
    path = root
    for part in value.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"archive identity inspection refuses symlinks: {relative}")
    if not path.is_file():
        raise ValueError(f"required archive file is absent: {relative}")
    return path


def _loads(data: bytes) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in primary evidence: {key}")
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant in primary evidence: {value}")

    record = json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)
    if type(record) is not dict:
        raise ValueError("a primary evidence record must be a JSON object")
    return record


def _checked_bytes(root: Path, relative: str, expected: str, size: int | None = None) -> bytes:
    data = _file(root, relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected or (size is not None and len(data) != size):
        raise ValueError(f"primary archive identity mismatch: {relative}")
    return data


def _bundle(project: Path, relative: str, digest: str, schema: str, status: str) -> tuple[dict, dict, dict]:
    path = _file(project, relative)
    record = _loads(_checked_bytes(project, relative, digest))
    _require(record.get("schema") == schema and record.get("status") == status,
             "the primary artifact schema or status differs")
    entries = record.get("files")
    _require(type(entries) is list, "the anchored record lacks its payload inventory")
    expected = {}
    for row in entries:
        _require(type(row) is dict and set(row) == {"path", "sha256", "bytes"},
                 "the anchored payload inventory has unexpected fields")
        _require(type(row["path"]) is str and row["path"] not in expected and row["path"] != path.name,
                 "the anchored payload inventory repeats a path")
        _require(type(row["bytes"]) is int and row["bytes"] >= 0,
                 "the anchored payload size is invalid")
        expected[row["path"]] = row
    actual = set()
    for child in path.parent.rglob("*"):
        if child.is_symlink():
            raise ValueError("primary evidence inventories must not contain symlinks")
        if child.is_file() and child != path:
            actual.add(child.relative_to(path.parent).as_posix())
    _require(actual == set(expected), f"primary archive payload inventory differs: {relative}")
    payloads = {name: _checked_bytes(path.parent, name, row["sha256"], row["bytes"])
                for name, row in expected.items()}
    return record, payloads, {"payload_count": len(payloads),
                              "payload_bytes": sum(len(data) for data in payloads.values())}


def _payload(payloads: dict, relative: str) -> dict:
    if relative not in payloads:
        raise ValueError(f"a displayed record is outside its anchored payload inventory: {relative}")
    return _loads(payloads[relative])


def _check_dataset(project: Path, freeze: dict, certificate: dict, payloads: dict) -> int:
    """Hash the six linked dataset files; do not parse or recompute any losses."""
    identity = certificate["calibration_identity"]
    _require(identity["dataset_manifest_sha256"] == freeze["dataset_manifest_sha256"],
             "certificate and frozen dataset identities disagree")
    prefix = "data/primary/"
    manifest = _loads(_checked_bytes(project, prefix + "manifest.json", identity["dataset_manifest_sha256"]))
    _require(manifest["run_id"] == identity["run_id"] == freeze["run_id"],
             "the recorded primary run identity differs")
    plan = manifest["calibration_plan"]
    _require(plan["sha256"] == identity["calibration_plan_sha256"] == freeze["calibration_plan_sha256"],
             "the linked calibration plan identity differs")
    _checked_bytes(project, prefix + plan["path"], plan["sha256"])
    for role in ("training", "validation"):
        row = manifest["development"][role]
        _checked_bytes(project, prefix + row["path"], row["sha256"])
    archived = payloads[identity["receipt_path"]]
    receipt_bytes = _checked_bytes(project, prefix + "calibration/receipt.json", identity["receipt_sha256"])
    _require(receipt_bytes == archived, "live and archived calibration receipt bytes differ")
    receipt = _loads(receipt_bytes)
    _require(receipt["data_sha256"] == identity["data_sha256"], "the calibration data identity differs")
    _checked_bytes(project, prefix + "calibration/" + receipt["data_path"], identity["data_sha256"])
    return 6


def _read_primary(project: Path) -> tuple[Path, dict, dict, dict]:
    project = Path(project).resolve()
    _require(project.is_dir(), "the primary project directory is absent")
    records, payloads, inventory = {}, {}, {}
    for name, relative, digest, schema, status in _RECORDS:
        records[name], payloads[name], inventory[name] = _bundle(project, relative, digest, schema, status)
    freeze, certificate, audit = (records[name] for name in ("freeze", "certificate", "audit"))
    _require(certificate["freeze_record_sha256"] == audit["freeze_record_sha256"] == PRIMARY_FREEZE_SHA256
             and audit["certificate_sha256"] == PRIMARY_CERTIFICATE_SHA256,
             "the anchored records refer to another primary family")
    _require([row["candidate_id"] for row in freeze["models"]] == list(_CANDIDATES)
             and [row["candidate_id"] for row in certificate["candidates"]] == list(_CANDIDATES)
             and [row["candidate_id"] for row in audit["candidates"]] == list(_CANDIDATES)
             and audit["candidate_ids"] == list(_CANDIDATES),
             "the anchored records do not retain the three ordered primary candidates")
    inventory["external_dataset_file_count"] = _check_dataset(project, freeze, certificate, payloads["certificate"])
    return project, records, payloads, inventory


def _fraction(record: dict) -> Fraction:
    _require(type(record) is dict and set(record) == {"numerator", "denominator"},
             "a displayed exact value requires two canonical fraction fields")
    a, b = record["numerator"], record["denominator"]
    _require(type(a) is str and type(b) is str, "displayed exact fraction fields must be strings")
    numerator, denominator = int(a), int(b)
    _require(str(numerator) == a and str(denominator) == b and denominator > 0,
             "displayed exact fraction integers are not canonical")
    value = Fraction(numerator, denominator)
    _require(value.numerator == numerator and value.denominator == denominator,
             "displayed exact fraction is not reduced")
    return value


def _decimal_18(record: dict) -> str:
    # Display conversion only; neither a loss calculation nor a bound decision.
    value = _fraction(record)
    quotient, remainder = divmod(abs(value.numerator) * 10 ** 18, value.denominator)
    if 2 * remainder > value.denominator or (2 * remainder == value.denominator and quotient % 2):
        quotient += 1
    sign = "-" if value < 0 and quotient else ""
    return f"{sign}{quotient // 10 ** 18}.{quotient % 10 ** 18:018d}"


def _hex_float(value: str) -> float:
    _require(type(value) is str, "a recorded computational value must be hexadecimal text")
    number = float.fromhex(value)
    _require(math.isfinite(number) and number.hex() == value,
             "a recorded computational value is not canonical finite binary64")
    return number


def _support_entry(entries: list, history: tuple[int, int]) -> dict:
    _require(type(entries) is list and len(entries) == 16, "a recorded support table must have 16 histories")
    for index, row in enumerate(entries):
        a, b = divmod(index, 4)
        _require(type(row) is dict and type(row.get("history_id")) is int and row["history_id"] == index
                 and type(row.get("previous_state")) is int and row["previous_state"] == a
                 and type(row.get("current_state")) is int and row["current_state"] == b,
                 "a recorded support table does not preserve ordered history identities")
    return entries[4 * history[0] + history[1]]


def _display(records: dict, payloads: dict, inventory: dict, history: tuple[int, int]) -> dict:
    freeze, certificate, audit = (records[name] for name in ("freeze", "certificate", "audit"))
    table, predictions = [], []
    for model, issued, audited in zip(freeze["models"], certificate["candidates"], audit["candidates"], strict=True):
        candidate = model["candidate_id"]
        _require(audited["calibration_empirical_mse"] == issued["empirical_mse"]
                 and audited["issued_upper_endpoint"] == issued["upper_endpoint"]
                 and audited["certificate_status"] == issued["status"],
                 "recorded audit and certificate values disagree")
        table.append({
            "candidate_id": candidate, "bank_id": model["bank_id"],
            "empirical_mse": issued["empirical_mse"],
            "empirical_mse_decimal_18_approximate": _decimal_18(issued["empirical_mse"]),
            "exact_population_mse": audited["exact_population_mse"],
            "population_mse_decimal_18_approximate": audited["population_mse_decimal_18_approximate"],
            "issued_upper_endpoint": issued["upper_endpoint"],
            "issued_upper_endpoint_decimal_18": issued["upper_endpoint_decimal_18"],
            "certificate_status": issued["status"], "realized_coverage": audited["realized_coverage"],
            "population_requirement_met": audited["population_requirement_met"],
            "fitted_excess_over_bayes": audited["fitted_excess_over_bayes"],
        })
        support = _payload(payloads["freeze"], model["prediction_table_path"])
        _require(support["candidate_id"] == candidate, "recorded prediction-table identity differs")
        row = _support_entry(support["predictions"], history)
        prediction = _hex_float(row["prediction_hex"])
        _require(0 <= prediction <= 1, "recorded clipped prediction is outside [0,1]")
        predictions.append({"candidate_id": candidate, "prediction": prediction,
                            "prediction_hex": row["prediction_hex"]})
    selected_bank = freeze["models"][1]["bank_id"]
    _require(selected_bank in {"cap_1", "cap_2"}
             and freeze["models"][2]["bank_id"] == selected_bank,
             "the recorded KAHM predictors do not share a supported bank")
    features = _payload(payloads["freeze"], f"development/encoders/{selected_bank}_features.json")
    geometry = _payload(payloads["freeze"], "development/encoders/reference_geometry.json")
    entry = _support_entry(features["entries"], history)
    widths = features["feature_widths"]
    _require(features["bank_id"] == selected_bank and widths["raw"] == 2 and widths["kahm_mem"] == 6,
             "recorded features differ from the selected primary bank")
    raw, memberships, enriched = ([ _hex_float(value) for value in entry[name] ]
                                  for name in ("raw_hex", "kahm_mem_hex", "kahm_local_hex"))
    _require((len(raw), len(memberships), len(enriched)) == (2, 6, widths["kahm_local"])
             and entry["kahm_local_hex"][:6] == entry["kahm_mem_hex"],
             "recorded local enrichment does not retain the membership prefix")
    partitions, start = [], 0
    for partition in geometry["partitions"]:
        count = partition["effective_groups"]
        partitions.append({"partition_id": partition["partition_id"], "group_count": count,
                           "membership_start": start, "membership_stop": start + count,
                           "memberships": memberships[start:start + count],
                           "reference_counts": [group["reference_count"] for group in partition["groups"]]})
        start += count
    _require([row["group_count"] for row in partitions] == [2, 4] and start == 6,
             "recorded primary partitions differ from the 2-group/4-group construction")
    observations = [_OBSERVATIONS[label] for label in history]
    _require(entry["input_hex"] == [value.hex() for value in observations],
             "recorded feature history has different observation values")
    return {
        "schema": "kahm.primary_tutorial_inspection.v1", "status": "archive_identity_checked",
        "mode": "archive_identity_only", "archive_identity_checked": True,
        "model_replayed": False, "certificate_reverified": False, "population_risk_recomputed": False,
        "calibration_losses_recomputed": False, "calibration_resampled": False,
        "models_refitted": False, "root_search_repeated": False,
        "anchors": {"freeze_sha256": PRIMARY_FREEZE_SHA256, "certificate_sha256": PRIMARY_CERTIFICATE_SHA256,
                    "audit_sha256": PRIMARY_AUDIT_SHA256},
        "inventory": inventory, "candidate_ids": list(_CANDIDATES), "table": table,
        "history": {
            "state_labels": list(history), "history_id": 4 * history[0] + history[1],
            "observations": observations, "observation_hex": [value.hex() for value in observations],
            "recorded_predictions": predictions,
            "recorded_encoder": {
                "bank_id": selected_bank, "recorded_values_only": True, "feature_widths": widths,
                "raw": raw, "raw_hex": entry["raw_hex"],
                "memberships": memberships, "memberships_hex": entry["kahm_mem_hex"],
                "enriched": enriched, "enriched_hex": entry["kahm_local_hex"],
                "partition_groups": partitions,
                "raw_meaning": "training-standardized two-observation history",
                "enriched_order": "six memberships, then phi_j * P_j.T @ (z - mu_j) in partition/group order using each stored local basis",
                "membership_meaning": "each partition normalizes separately; the concatenated six values are not one probability distribution",
            },
        },
        "recorded_audit_summary": audit["summary"], "recorded_guarantee": certificate["guarantee"],
        "scope": "Recorded values match the released primary anchors and payload bytes. No model, loss, endpoint, or population-risk computation has been rerun. Runtime/source compatibility is a separate precheck; strict numerical verification is explicit.",
    }


def inspect_primary(project: Path, history: tuple[int, int] = (1, 2)) -> dict:
    """Display identity-checked recorded primary values, using stdlib only."""
    selected = _history(history)
    _, records, payloads, inventory = _read_primary(project)
    return _display(records, payloads, inventory, selected)


def runtime_compatibility(project: Path) -> dict:
    """Compare runtime/source metadata without replay or numerical verification.

    Archive tampering raises. Missing NumPy or unavailable runtime helpers are
    returned as explicit precheck failures, never as an implicit verification.
    Matching metadata permits an attempt at strict verification; it does not
    guarantee its success on the current machine.
    """
    _, records, payloads, _ = _read_primary(project)
    plans = {"certificate": _payload(payloads["certificate"], "assessment_plan.json"),
             "audit": _payload(payloads["audit"], "audit_plan.json")}
    expected = {"freeze": records["freeze"]["inference"]["runtime"],
                **{name: plan["runtime"] for name, plan in plans.items()}}
    result = {
        "schema": "kahm.primary_runtime_compatibility.v1", "mode": "runtime_precheck_only",
        "status": "unavailable", "compatible": False, "archive_identity_checked": True,
        "model_replayed": False, "certificate_reverified": False, "population_risk_recomputed": False,
        "full_verification_performed": False, "runtime_metadata_matches": None, "source_identity_matches": None,
        "dependency_status": {"numpy_available": False, "numpy_version": None,
                              "runtime_helpers_loaded": False, "error": None},
        "expected_runtime": expected, "current_runtime": None,
        "runtime_differences": [], "source_differences": [], "source_roster_differences": [],
        "scope": "This compares metadata and current source hashes only. A compatible result is permission to attempt strict verification, not a replay or a verified certificate.",
    }
    try:
        numpy = importlib.import_module("numpy")
        result["dependency_status"].update(numpy_available=True, numpy_version=numpy.__version__)
        forecast = importlib.import_module("kahm_demo.forecast_workflow")
        freeze = importlib.import_module("kahm_demo.freeze")
        certificate = importlib.import_module("kahm_demo.certificate_workflow")
        audit = importlib.import_module("kahm_demo.audit_workflow")
        current = forecast.forecast_runtime_record()
        result["dependency_status"]["runtime_helpers_loaded"] = True
        result["current_runtime"] = current
        for name, recorded in expected.items():
            for field in sorted(set(recorded) | set(current)):
                if recorded.get(field) != current.get(field) or type(recorded.get(field)) is not type(current.get(field)):
                    result["runtime_differences"].append({"record": name, "field": field,
                                                          "expected": recorded.get(field), "current": current.get(field)})
        inference = records["freeze"]["inference"]
        expected_sources = {"freeze": {
            name: next(row["sha256"] for row in inference["source_files"]
                       if row["path"] == "sources/kahm_demo/" + name)
            for name in inference["execution_source_roster"]},
            **{name: plan["source_sha256"] for name, plan in plans.items()}}
        package = Path(freeze.__file__).parent
        freeze._require_version_only_initializer(package / "__init__.py")
        current_sources = {"freeze": {name: hashlib.sha256(_file(package, name).read_bytes()).hexdigest()
                                       for name in freeze.EXECUTION_SOURCE_FILES},
                           "certificate": certificate._source_identities(), "audit": audit._source_identities()}
        rosters = {"freeze": (inference["execution_source_roster"], list(freeze.EXECUTION_SOURCE_FILES)),
                   "certificate": (plans["certificate"]["source_roster"], list(certificate.CERTIFIER_SOURCE_FILES)),
                   "audit": (plans["audit"]["source_roster"], list(audit.AUDIT_SOURCE_FILES))}
        for name, (recorded, installed) in rosters.items():
            if recorded != installed:
                result["source_roster_differences"].append({"record": name, "expected": recorded, "current": installed})
        for name, recorded in expected_sources.items():
            installed = current_sources[name]
            for source in sorted(set(recorded) | set(installed)):
                if recorded.get(source) != installed.get(source):
                    result["source_differences"].append({"record": name, "path": source,
                                                         "expected_sha256": recorded.get(source),
                                                         "current_sha256": installed.get(source)})
        result["runtime_metadata_matches"] = not result["runtime_differences"]
        result["source_identity_matches"] = not result["source_differences"] and not result["source_roster_differences"]
        result["compatible"] = result["runtime_metadata_matches"] and result["source_identity_matches"]
        result["status"] = "compatible" if result["compatible"] else "incompatible"
    except Exception as error:
        result["dependency_status"]["error"] = {"type": type(error).__name__, "message": str(error)}
        result["compatible"] = False
        result["status"] = "unavailable"
    return result


def verify_primary(project: Path, history: tuple[int, int] = (1, 2)) -> dict:
    """Explicitly reverify the primary audit/certificate and replay three models.

    The strict verifier's exceptions propagate. No alternative runtime, archive
    fallback, replacement endpoint, model fit, sample, or root search is tried.
    """
    selected = _history(history)
    project, records, payloads, inventory = _read_primary(project)
    result = _display(records, payloads, inventory, selected)
    audit_module = importlib.import_module("kahm_demo.audit_workflow")
    freeze_module = importlib.import_module("kahm_demo.freeze")
    freeze_path, bundle = project / "frozen/primary/freeze.json", project / "data/primary"
    audit = audit_module.verify_audit_bundle(project / "audits/primary/audit.json",
                                           certificate_path=project / "certificates/primary/certificate.json",
                                           dataset_bundle=bundle, freeze_path=freeze_path)
    _require(audit.get("status") == "passed" and audit.get("freeze_record_sha256") == PRIMARY_FREEZE_SHA256
             and audit.get("certificate_sha256") == PRIMARY_CERTIFICATE_SHA256
             and audit.get("audit_sha256") == PRIMARY_AUDIT_SHA256
             and audit.get("candidate_ids") == list(_CANDIDATES),
             "strict primary verification did not return the anchored three-candidate audit")
    for field in ("exact_population_risks_recomputed", "preserved_issued_endpoints", "source_identity_matches",
                  "runtime_metadata_matches", "all_64_triples_per_candidate_checked",
                  "independent_conditional_decomposition_checked"):
        _require(audit.get(field) is True, f"strict audit verification did not establish: {field}")
    _require(audit.get("candidates") == records["audit"]["candidates"],
             "strict audit results differ from the identity-checked archive")
    for field in ("calibration_resampled", "models_refitted", "root_search_repeated",
                  "issued_certificate_modified", "model_selection_performed"):
        _require(audit.get(field) is False, f"strict audit reports an unsupported action: {field}")
    # This existing internal preflight returns the same model objects whose
    # sources, fitted states, and all support predictions it has checked. One
    # call avoids repeating the 225-state preflight separately for each model.
    checked, models = freeze_module._verify_frozen_bundle(freeze_path, dataset_bundle=bundle)
    _require(checked.get("status") == "passed" and checked.get("freeze_record_sha256") == PRIMARY_FREEZE_SHA256
             and set(models) == set(_CANDIDATES), "frozen preflight returned a different family")
    replayed = []
    for recorded in result["history"]["recorded_predictions"]:
        candidate = recorded["candidate_id"]
        model = models[candidate]
        _require(model.candidate_id == candidate, "the replayed model candidate identity differs")
        prediction = model.predict_one(result["history"]["observations"])
        _require(type(prediction) is float and math.isfinite(prediction)
                 and prediction.hex() == recorded["prediction_hex"],
                 "selected-history replay differs from the anchored prediction")
        replayed.append({"candidate_id": candidate, "prediction": prediction,
                         "prediction_hex": prediction.hex(), "matches_recorded": True})
    _read_primary(project)  # Refuse evidence changes during explicit verification.
    result.update(schema="kahm.primary_tutorial_verification.v1", status="passed",
                  mode="full_read_only_verification", model_replayed=True,
                  certificate_reverified=True, population_risk_recomputed=True,
                  calibration_losses_recomputed=True, audit_verification=audit,
                  replayed_predictions=replayed,
                  scope="The original certificate and population audit passed strict read-only verification; all three selected-history forecasts were replayed exactly. The original conditional statistical premises and claim limits are unchanged.")
    return result

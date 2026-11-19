"""Retained class-risk assessment on the complete finite benchmark support.

This versioned workflow assesses exact archived computational feature and
prediction tables. It does not replay a historical numerical model or claim
global regularity of its executable readout. Class definitions, independent
calibration streams and a joint confidence budget are written before sampling.
Population oracles are imported only after all sample certificates are saved.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from ._util import canonical_json_bytes, checked_relative_file, read_json, require_hash, sha256_file
from .class_bounds import bounded_mean_endpoint, fraction_record, parse_fraction, verify_mean_endpoint
from .law import BENCHMARK_ID, Episode, OBSERVATIONS
from .lipschitz_class import make_geometry, membership, certify_empirical_minimum, verify_empirical_minimum
from .sampling import sample_episodes, stream_descriptor


PROTOCOL = "kahm.class_risk_protocol.v1"
PLAN_SCHEMA = "kahm.class_risk_plan.v1"
CERTIFICATE_SCHEMA = "kahm.class_risk_certificate.v1"
RUN_SCHEMA = "kahm.class_risk_run.v1"
SOURCE_FILES = (
    "__init__.py", "_util.py", "cli.py", "law.py", "oracle.py", "sampling.py", "verified_arithmetic.py",
    "contract.py", "specification.json",
    "calibration_protocol.py", "calibration_protocol.json", "class_bounds.py",
    "lipschitz_class.py", "class_oracle.py", "class_risk_workflow.py",
)
_CANDIDATES = ("raw", "kahm_mem", "kahm_local")
_SCOPE = {"representation": "exact finite-support computational feature table",
          "historical_model_replayed": False, "ideal_real_arithmetic_encoder_claimed": False,
          "global_executable_readout_lipschitz_claimed": False,
          "iid_calibration_independent_of_development_assumed": True,
          "population_oracle_used_for_certification": False,
          "confidence_scope": "all declared repetitions, sample sizes, classes, block and empirical lower bounds, and fitted upper bounds jointly",
          "prefixes_share_observations": True, "seed_replay_is_new_independent_evidence": False}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def _fraction(value):
    return parse_fraction(value)


def _hex_fraction(value):
    _require(type(value) is str, "computational coordinates must be hexadecimal strings")
    converted = float.fromhex(value)
    _require(math.isfinite(converted) and converted.hex() == value,
             "noncanonical or nonfinite computational coordinate")
    return Fraction(converted)


def _ordered_rows(rows, label):
    _require(type(rows) is list and len(rows) == 16, f"{label} must cover all 16 histories")
    result = {}
    for row in rows:
        _require(type(row) is dict and type(row.get("history_id")) is int,
                 f"invalid {label} row")
        h = row["history_id"]
        _require(0 <= h < 16 and h not in result, f"duplicate or invalid {label} history")
        _require(type(row.get("previous_state")) is int and type(row.get("current_state")) is int
                 and (row["previous_state"], row["current_state"]) == divmod(h, 4),
                 f"{label} history fields disagree")
        result[h] = row
    return tuple(result[h] for h in range(16))


def _read_training(path):
    episodes = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        _require(type(row) is dict and set(row) == {"episode_id", "previous_state", "current_state", "next_state"},
                 "unsupported training row")
        episode = Episode(**row)
        _require(episode.episode_id == i, "training episode order differs")
        episodes.append(episode)
    _require(bool(episodes), "training observations are required for the control predictors")
    return episodes


def _source_inputs(freeze_path):
    """Read hash-bound development tables only; never open calibration records."""
    freeze_path = Path(freeze_path).resolve()
    freeze = read_json(freeze_path)
    _require(freeze.get("schema") == "kahm.model_freeze.v1", "unsupported source model freeze")
    _require(freeze.get("record_sha256") == _digest({k: v for k, v in freeze.items() if k != "record_sha256"}),
             "source freeze payload digest differs")
    files = freeze.get("files")
    _require(type(files) is list, "missing source file manifest")
    declared = {}
    for item in files:
        _require(type(item) is dict and type(item.get("path")) is str and item["path"] not in declared,
                 "invalid or repeated source manifest path")
        declared[item["path"]] = item

    def get(relative):
        _require(relative in declared, "required development input is not bound by the source freeze")
        path = checked_relative_file(freeze_path.parent, relative)
        require_hash(path, declared[relative]["sha256"])
        _require(path.stat().st_size == declared[relative]["bytes"], "source byte count differs")
        return path

    models = freeze.get("models")
    _require(type(models) is list and [m.get("candidate_id") for m in models] == list(_CANDIDATES),
             "the source freeze must contain RAW, MEM and LOCAL in their declared order")
    bank = models[1].get("bank_id")
    _require(type(bank) is str and bank == models[2].get("bank_id") and "/" not in bank and ".." not in bank,
             "MEM and LOCAL must share one declared bank")
    paths = freeze["development"]["paths"]
    features_path = get(f"{paths['encoders']}/{bank}_features.json")
    feature_record = read_json(features_path)
    _require(feature_record.get("bank_id") == bank, "feature bank identity differs")
    features = _ordered_rows(feature_record.get("entries"), "features")
    predictions = {}
    input_paths = {"source_freeze.json": freeze_path, "features.json": features_path}
    for model in models:
        candidate = model["candidate_id"]
        path = get(model["prediction_table_path"])
        require_hash(path, model["prediction_table_sha256"])
        record = read_json(path)
        _require(record.get("candidate_id") == candidate and record.get("schema") == "kahm.prediction_table.v1",
                 "prediction identity differs")
        rows = _ordered_rows(record.get("predictions"), "predictions")
        predictions[candidate] = tuple(_hex_fraction(row["prediction_hex"]) for row in rows)
        _require(all(0 <= z <= 1 for z in predictions[candidate]), "predictions must lie in [0,1]")
        input_paths[f"predictions/{candidate}.json"] = path
    dataset_path = get(f"{paths['dataset']}/manifest.json")
    dataset = read_json(dataset_path)
    input_paths["dataset_manifest.json"] = dataset_path
    training = dataset["development"]["training"]
    training_path = get(f"{paths['dataset']}/{training['path']}")
    require_hash(training_path, training["sha256"])
    episodes = _read_training(training_path)
    _require(len(episodes) == training["episode_count"], "training count differs")
    input_paths["training.jsonl"] = training_path
    return freeze, features, predictions, episodes, input_paths


def _control_predictions(training, short):
    totals = [Fraction(0) for _ in range(4 if short else 16)]
    counts = [0 for _ in totals]
    for row in training:
        key = row.current_state if short else row.history_id
        totals[key] += OBSERVATIONS[row.next_state]
        counts[key] += 1
    fallback = sum(totals) / sum(counts)
    means = [total / n if n else fallback for total, n in zip(totals, counts, strict=True)]
    return tuple(means[h % 4 if short else h] for h in range(16))


def _classes(features, predictions, training, kappas):
    result = []
    for kappa in kappas:
        suffix = f"k{kappa.numerator}_{kappa.denominator}"
        for candidate in _CANDIDATES:
            coords = tuple(tuple(_hex_fraction(x) for x in row[candidate + "_hex"]) for row in features)
            _require(all(len(row) > 0 for row in coords), "features cannot be empty")
            result.append({"class_id": f"{candidate}_{suffix}", "representation": candidate,
                           "kappa": fraction_record(kappa), "metric": "euclidean",
                           "coordinates": "archived representation before readout column scaling",
                           "features": [[fraction_record(x) for x in row] for row in coords],
                           "predictions": [fraction_record(x) for x in predictions[candidate]],
                           "predictor_source": "archived computational table"})
    for short in (True, False):
        name = "short_history" if short else "full_history"
        coords = [(OBSERVATIONS[h % 4],) if short else (OBSERVATIONS[h // 4], OBSERVATIONS[h % 4])
                  for h in range(16)]
        result.append({"class_id": name + "_k4_1", "representation": name,
                       "kappa": fraction_record(Fraction(4)), "metric": "euclidean",
                       "coordinates": "original unstandardized observations",
                       "features": [[fraction_record(x) for x in row] for row in coords],
                       "predictions": [fraction_record(x) for x in _control_predictions(training, short)],
                       "predictor_source": "training-only conditional sample means; training grand mean fallback"})
    return result


def _geometry(candidate):
    return make_geometry(tuple(tuple(_fraction(x) for x in row) for row in candidate["features"]),
                         _fraction(candidate["kappa"]))


def _predictions(candidate):
    return tuple(_fraction(x) for x in candidate["predictions"])


def _parameters(sample_sizes, block_sizes, kappas, repetitions, delta, tolerance, sweeps, master_seed, run_id):
    for values, name, minimum in ((sample_sizes, "sample sizes", 2), (block_sizes, "block sizes", 2)):
        _require(bool(values) and all(type(x) is int and x >= minimum for x in values), f"invalid {name}")
        _require(tuple(sorted(set(values))) == tuple(values), f"{name} must be unique and increasing")
    _require(max(block_sizes) <= min(sample_sizes), "every block size must fit every sample size")
    _require(max(sample_sizes) <= 1000000 and max(block_sizes) <= 2048,
             "sample sizes must not exceed 1000000 and block sizes must not exceed 2048")
    _require(bool(kappas) and all(type(x) is Fraction and x >= 0 for x in kappas), "invalid Lipschitz bounds")
    _require(tuple(sorted(set(kappas))) == tuple(kappas), "Lipschitz bounds must be unique and increasing")
    _require(type(repetitions) is int and 1 <= repetitions <= 1000, "repetitions must be 1..1000")
    _require(type(delta) is Fraction and 0 < delta < 1, "delta must be an exact fraction in (0,1)")
    _require(type(tolerance) is Fraction and 0 <= tolerance <= 1, "tolerance must be an exact fraction in [0,1]")
    _require(type(sweeps) is int and 0 <= sweeps <= 10000, "solver sweeps must be 0..10000")
    _require(type(master_seed) is int and master_seed >= 0, "master seed must be a nonnegative integer")
    _require(type(run_id) is str and run_id.strip(), "run ID must be nonempty")


def _new_output(output, source_freeze):
    requested = Path(output)
    _require(not requested.exists() and not requested.is_symlink(), "output already exists; no overwrite or resume")
    resolved = requested.resolve()
    _require(not resolved.is_relative_to(Path(source_freeze).resolve().parent), "output must be outside the source freeze")
    markers = ("class_risk_plan.json", "freeze.json", "certificate.json", "reproduction_plan.json", "assessment_plan.json",
               "calibration_plan.json", "audit_plan.json", "manifest.json", "receipt.json")
    for route in (Path(os.path.abspath(requested)), resolved):
        for ancestor in (route, *route.parents):
            _require(not ancestor.is_symlink() or ancestor.exists(), "output passes through a dangling symlink")
            for marker in markers:
                path = ancestor / marker
                if path.exists() or path.is_symlink():
                    _require(path.is_file() and not path.is_symlink(), "invalid scientific bundle marker")
                    if marker in {"manifest.json", "receipt.json"}:
                        record = read_json(path)
                        schema = record.get("schema") if type(record) is dict else None
                        _require(type(schema) is not str or not schema.startswith("kahm."),
                                 "output must be outside retained scientific bundles")
                    else:
                        raise ValueError("output must be outside retained scientific bundles")
    return resolved


def _endpoint(mean, n, alpha, side):
    record = bounded_mean_endpoint(mean, n, alpha, side)
    check = verify_mean_endpoint(record)
    _require(check.get("verified") is True or check.get("unavailable_record_consistent") is True,
             "new endpoint failed its independent numerical check")
    return record


def _endpoint_value(record):
    return _fraction(record["endpoint"]) if record["status"] == "available" else None


def _decision(lower, upper, admissible, tolerance):
    excluded = lower > tolerance
    accepted = upper is not None and upper <= tolerance and admissible
    if excluded and accepted:
        return "conflicting_certificates"
    if excluded:
        return "class_excluded"
    if accepted:
        return "requirement_met"
    return "unresolved"


def _certificate(candidate, geometry, histories, targets, block_sizes, alpha, tolerance, sweeps, folder, progress):
    predictions = _predictions(candidate)
    admissibility = membership(geometry, predictions)
    # Membership is a deterministic geometric statement; no calibration labels
    # enter the frozen predictor or its class definition.
    n = len(histories)
    q = sum((y - predictions[h]) ** 2 for h, y in zip(histories, targets, strict=True)) / n
    upper = _endpoint(q, n, alpha, "upper")
    entries = []
    lowers = [Fraction(0)]
    for b in block_sizes:
        start = time.perf_counter()
        count = n // b
        witnesses = []
        lower_sse = Fraction(0)
        for k in range(count):
            left, right = k * b, (k + 1) * b
            witness = certify_empirical_minimum(geometry, histories[left:right], targets[left:right], sweeps=sweeps)
            value = verify_empirical_minimum(geometry, histories[left:right], targets[left:right], witness)
            _require(value >= 0, "negative certified lower SSE")
            witnesses.append(witness)
            lower_sse += value
        cb = Fraction((b * b) // 4, b * (b - 1))
        mean_t = lower_sse / (count * (b - 1))
        _require(0 <= mean_t <= cb, "certified score exceeds its proved range")
        block_endpoint = _endpoint(mean_t / cb, count, alpha, "lower")
        empirical_endpoint = _endpoint(lower_sse / n, n, alpha, "lower")
        lb = _endpoint_value(block_endpoint)
        le = _endpoint_value(empirical_endpoint)
        if lb is not None:
            lowers.append(cb * lb)
        if le is not None:
            lowers.append(le)
        witness_path = f"blocks_{b}.json"
        _write(folder / witness_path, {"block_size": b, "block_count": count, "partition": "consecutive disjoint prefix blocks",
                                      "witnesses": witnesses})
        entries.append({"block_size": b, "block_count": count, "unused_observations": n - b * count,
                        "score_range": fraction_record(cb), "mean_lower_score": fraction_record(mean_t),
                        "empirical_minimum_lower": fraction_record(lower_sse / n),
                        "block_endpoint": block_endpoint, "empirical_endpoint": empirical_endpoint,
                        "block_lower": fraction_record(cb * lb) if lb is not None else None,
                        "empirical_lower": fraction_record(le) if le is not None else None,
                        "witness_path": witness_path, "witness_sha256": sha256_file(folder / witness_path),
                        "elapsed_seconds": time.perf_counter() - start})
        if progress:
            progress(f"{candidate['class_id']}: n={n}, b={b}")
    start = time.perf_counter()
    full = certify_empirical_minimum(geometry, histories, targets, sweeps=sweeps)
    full_lower = verify_empirical_minimum(geometry, histories, targets, full) / n
    full_endpoint = _endpoint(full_lower, n, alpha, "lower")
    lf = _endpoint_value(full_endpoint)
    if lf is not None:
        lowers.append(lf)
    _write(folder / "full_empirical.json", full)
    lower = max(lowers)
    upper_value = _endpoint_value(upper)
    admissible = admissibility["admissible"]
    return {"schema": CERTIFICATE_SCHEMA, "class_id": candidate["class_id"], "geometry_sha256": geometry.fingerprint,
            "n": n, "alpha_per_endpoint": fraction_record(alpha), "tolerance": fraction_record(tolerance),
            "calibration_mse": fraction_record(q), "fitted_upper_endpoint": upper, "fitted_admissibility": admissibility,
            "blocks": entries, "full_empirical_minimum_lower": fraction_record(full_lower),
            "full_empirical_endpoint": full_endpoint,
            "full_witness_path": "full_empirical.json", "full_witness_sha256": sha256_file(folder / "full_empirical.json"),
            "full_elapsed_seconds": time.perf_counter() - start, "combined_lower": fraction_record(lower),
            "fitted_upper": fraction_record(upper_value) if upper_value is not None else None,
            "class_upper": fraction_record(upper_value) if upper_value is not None and admissible else None,
            "status": _decision(lower, upper_value, admissible, tolerance)}


def run_class_risk(freeze_path, output, *, sample_sizes=(4096,), block_sizes=(2, 8, 16, 32, 64),
                   kappas=(Fraction(2),), repetitions=1, delta=Fraction(1, 20), tolerance=Fraction(1, 10),
                   sweeps=100, master_seed=20261119, run_id="class-risk-v1", progress=None):
    """Create one new, complete study; preserve partial evidence on failure."""
    sample_sizes, block_sizes, kappas = tuple(sample_sizes), tuple(block_sizes), tuple(kappas)
    _parameters(sample_sizes, block_sizes, kappas, repetitions, delta, tolerance, sweeps, master_seed, run_id)
    output = _new_output(output, freeze_path)
    freeze, features, predictions, training, inputs = _source_inputs(freeze_path)
    classes = _classes(features, predictions, training, kappas)
    endpoint_count = repetitions * len(sample_sizes) * len(classes) * (2 * len(block_sizes) + 2)
    alpha = delta / endpoint_count
    streams = [stream_descriptor(master_seed, "calibration", f"class-risk-v1:{run_id}:rep-{r}")
               for r in range(repetitions)]
    source_dir = Path(__file__).parent
    plan = {"schema": PLAN_SCHEMA, "protocol": PROTOCOL, "benchmark_id": BENCHMARK_ID,
            "run_id": run_id, "master_seed": str(master_seed), "sample_sizes": list(sample_sizes),
            "block_sizes": list(block_sizes), "kappas": [fraction_record(x) for x in kappas],
            "repetitions": repetitions, "delta": fraction_record(delta), "tolerance": fraction_record(tolerance),
            "endpoint_count": endpoint_count, "alpha_per_endpoint": fraction_record(alpha),
            "solver_sweeps": sweeps, "class_ids": [c["class_id"] for c in classes],
            "classes_sha256": _digest(classes), "source_freeze_sha256": sha256_file(Path(freeze_path)),
            "source_run_id": freeze["run_id"], "streams": streams,
            "inputs": {name: sha256_file(path) for name, path in inputs.items()},
            "sources": {name: sha256_file(source_dir / name) for name in SOURCE_FILES},
            "runtime": {"python": sys.version, "platform": platform.platform()},
            "scope": dict(_SCOPE)}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "class_risk_plan.json", plan)
    try:
        for name, path in inputs.items():
            destination = output / "inputs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            require_hash(destination, plan["inputs"][name])
        for name in SOURCE_FILES:
            destination = output / "sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / name, destination)
            require_hash(destination, plan["sources"][name])
        _write(output / "classes.json", classes)
        geometries = {c["class_id"]: _geometry(c) for c in classes}
        all_certificates = []
        for rep, stream in enumerate(streams):
            if progress:
                progress(f"sampling fresh calibration repetition {rep}")
            episodes = sample_episodes(max(sample_sizes), master_seed, "calibration", stream["run_id"])
            rows = [{"episode_id": e.episode_id, "previous_state": e.previous_state,
                     "current_state": e.current_state, "next_state": e.next_state} for e in episodes]
            _write(output / f"calibration/rep_{rep}.json", rows)
            histories = tuple(e.history_id for e in episodes)
            targets = tuple(OBSERVATIONS[e.next_state] for e in episodes)
            for n in sample_sizes:
                for candidate in classes:
                    relative = f"certificates/rep_{rep}/n_{n}/{candidate['class_id']}"
                    folder = output / relative
                    folder.mkdir(parents=True)
                    record = _certificate(candidate, geometries[candidate["class_id"]], histories[:n], targets[:n],
                                          block_sizes, alpha, tolerance, sweeps, folder, progress)
                    record["repetition"] = rep
                    record["plan_sha256"] = sha256_file(output / "class_risk_plan.json")
                    record["calibration_sha256"] = sha256_file(output / f"calibration/rep_{rep}.json")
                    _write(folder / "certificate.json", record)
                    all_certificates.append({"path": relative + "/certificate.json", "sha256": sha256_file(folder / "certificate.json")})
        _write(output / "certificate_index.json", all_certificates)
        if progress:
            progress("verifying all sample certificates before population audit")
        verify_class_risk(output, include_audit=False)
        # Importing this module is deliberately delayed until all issued sample
        # certificates have been persisted and checked. No oracle enters the
        # class/predictor construction, solver proposals, or confidence inputs.
        from .class_oracle import audit_class
        audits = []
        for candidate in classes:
            if progress:
                progress(f"population audit: {candidate['class_id']}")
            record = audit_class(geometries[candidate["class_id"]], _predictions(candidate),
                                 block_sizes=block_sizes, sweeps=max(sweeps, 200))
            relative = f"audits/{candidate['class_id']}.json"
            _write(output / relative, record)
            audits.append({"class_id": candidate["class_id"], "path": relative, "sha256": sha256_file(output / relative)})
        _write(output / "audit_index.json", audits)
        result = verify_class_risk(output, include_audit=True)
        result["schema"] = RUN_SCHEMA
        result["protocol"] = PROTOCOL
        _write(output / "summary.json", result)
        _write_report(output, result)
        if progress:
            progress("complete")
        return result
    except Exception as error:
        _write(output / "failure.json", {"schema": "kahm.class_risk_failure.v1", "error_type": type(error).__name__,
                                         "message": str(error), "partial_evidence_retained": True,
                                         "sample_replaced": False, "output_resumed": False})
        raise


def _check_endpoint(record, q, n, alpha, side):
    _require(record.get("q") == fraction_record(q) and record.get("n") == n
             and record.get("alpha") == fraction_record(alpha) and record.get("side") == side,
             "confidence endpoint is not bound to its recomputed input")
    verified = verify_mean_endpoint(record)
    _require(verified.get("verified") is True or verified.get("unavailable_record_consistent") is True,
             "confidence endpoint verification failed")
    return _endpoint_value(record)


def _verify_candidate(root, candidate, geometry, histories, targets, record, plan):
    n, alpha = len(histories), _fraction(plan["alpha_per_endpoint"])
    _require(record.get("schema") == CERTIFICATE_SCHEMA and record.get("class_id") == candidate["class_id"]
             and record.get("n") == n and record.get("geometry_sha256") == geometry.fingerprint,
             "certificate identity differs")
    _require(record.get("alpha_per_endpoint") == plan["alpha_per_endpoint"]
             and record.get("tolerance") == plan["tolerance"], "certificate confidence or tolerance differs")
    predictions = _predictions(candidate)
    admissibility = membership(geometry, predictions)
    _require(record.get("fitted_admissibility") == admissibility, "fitted admissibility record differs")
    q = sum((y - predictions[h]) ** 2 for h, y in zip(histories, targets, strict=True)) / n
    _require(record.get("calibration_mse") == fraction_record(q), "calibration MSE differs")
    upper = _check_endpoint(record["fitted_upper_endpoint"], q, n, alpha, "upper")
    _require(record.get("fitted_upper") == (fraction_record(upper) if upper is not None else None), "fitted upper summary differs")
    admissible = admissibility["admissible"]
    _require(record.get("class_upper") == (fraction_record(upper) if upper is not None and admissible else None),
             "class upper requires a verified admissible fitted table")
    lowers = [Fraction(0)]
    _require([r.get("block_size") for r in record["blocks"]] == plan["block_sizes"], "block menu differs")
    for entry in record["blocks"]:
        b = entry["block_size"]
        count = n // b
        cb = Fraction((b * b) // 4, b * (b - 1))
        _require(entry["block_count"] == count and entry["unused_observations"] == n - b * count
                 and entry["score_range"] == fraction_record(cb), "block normalization differs")
        path = checked_relative_file(root, entry["witness_path"])
        require_hash(path, entry["witness_sha256"])
        witnesses = read_json(path)
        _require(witnesses["block_size"] == b and witnesses["block_count"] == count
                 and witnesses["partition"] == "consecutive disjoint prefix blocks"
                 and len(witnesses["witnesses"]) == count, "block witness layout differs")
        total = Fraction(0)
        for k, witness in enumerate(witnesses["witnesses"]):
            total += verify_empirical_minimum(geometry, histories[k*b:(k+1)*b], targets[k*b:(k+1)*b], witness)
        mean_t = total / (count * (b - 1))
        _require(entry["mean_lower_score"] == fraction_record(mean_t)
                 and entry["empirical_minimum_lower"] == fraction_record(total / n), "block aggregate differs")
        lb = _check_endpoint(entry["block_endpoint"], mean_t / cb, count, alpha, "lower")
        le = _check_endpoint(entry["empirical_endpoint"], total / n, n, alpha, "lower")
        _require(entry["block_lower"] == (fraction_record(cb * lb) if lb is not None else None)
                 and entry["empirical_lower"] == (fraction_record(le) if le is not None else None), "lower summary differs")
        if lb is not None:
            lowers.append(cb * lb)
        if le is not None:
            lowers.append(le)
    path = checked_relative_file(root, record["full_witness_path"])
    require_hash(path, record["full_witness_sha256"])
    full_lower = verify_empirical_minimum(geometry, histories, targets, read_json(path)) / n
    _require(record["full_empirical_minimum_lower"] == fraction_record(full_lower), "full empirical input differs")
    lf = _check_endpoint(record["full_empirical_endpoint"], full_lower, n, alpha, "lower")
    if lf is not None:
        lowers.append(lf)
    lower = max(lowers)
    _require(record["combined_lower"] == fraction_record(lower), "combined lower bound differs")
    status = _decision(lower, upper, admissible, _fraction(plan["tolerance"]))
    _require(record["status"] == status, "class decision differs")
    return {"class_id": candidate["class_id"], "representation": candidate["representation"],
            "kappa": candidate["kappa"], "repetition": record["repetition"], "n": n,
            "lower": fraction_record(lower), "fitted_upper": record["fitted_upper"],
            "class_upper": record["class_upper"], "admissible": admissible, "status": status,
            "unavailable_endpoints": sum(endpoint["status"] != "available" for endpoint in
                [record["fitted_upper_endpoint"], record["full_empirical_endpoint"]]
                + [endpoint for entry in record["blocks"] for endpoint in (entry["block_endpoint"], entry["empirical_endpoint"])]),
            "blocks": [{k: entry[k] for k in ("block_size", "block_lower", "empirical_lower", "mean_lower_score", "elapsed_seconds")}
                       for entry in record["blocks"]]}


def verify_class_risk(output, *, include_audit=True):
    """Recompute numerical witnesses and decisions, without fitting or searching."""
    root = Path(output).resolve()
    plan = read_json(root / "class_risk_plan.json")
    _require(plan.get("schema") == PLAN_SCHEMA and plan.get("protocol") == PROTOCOL
             and plan.get("benchmark_id") == BENCHMARK_ID, "unsupported class-risk protocol")
    _require(plan.get("scope") == _SCOPE, "assessment scope or statistical premises differ")
    kappas = tuple(_fraction(x) for x in plan["kappas"])
    _parameters(plan["sample_sizes"], plan["block_sizes"], kappas, plan["repetitions"], _fraction(plan["delta"]),
                _fraction(plan["tolerance"]), plan["solver_sweeps"], int(plan["master_seed"]), plan["run_id"])
    _require(set(plan["sources"]) == set(SOURCE_FILES), "source roster differs")
    for name, digest in plan["sources"].items():
        require_hash(checked_relative_file(root, "sources/" + name), digest)
        require_hash(Path(__file__).parent / name, digest)
    for name, digest in plan["inputs"].items():
        require_hash(checked_relative_file(root, "inputs/" + name), digest)
    source_freeze = read_json(root / "inputs/source_freeze.json")
    _require(sha256_file(root / "inputs/source_freeze.json") == plan["source_freeze_sha256"], "source freeze differs")
    _require(source_freeze.get("record_sha256") == _digest({k: v for k, v in source_freeze.items() if k != "record_sha256"}),
             "copied source freeze payload differs")
    _require(source_freeze.get("schema") == "kahm.model_freeze.v1"
             and [m.get("candidate_id") for m in source_freeze.get("models", [])] == list(_CANDIDATES),
             "copied freeze family differs")
    _require(source_freeze.get("run_id") == plan["source_run_id"], "source run identity differs")
    models = source_freeze["models"]
    bank = models[1]["bank_id"]
    _require(type(bank) is str and bank == models[2]["bank_id"] and "/" not in bank and ".." not in bank,
             "copied feature banks differ")
    declared = {item["path"]: item for item in source_freeze["files"]}
    _require(len(declared) == len(source_freeze["files"]), "duplicate source manifest entry")
    development_paths = source_freeze["development"]["paths"]
    dataset = read_json(root / "inputs/dataset_manifest.json")
    bindings = {"dataset_manifest.json": development_paths["dataset"] + "/manifest.json",
                "features.json": development_paths["encoders"] + f"/{bank}_features.json",
                "training.jsonl": development_paths["dataset"] + "/" + dataset["development"]["training"]["path"]}
    bindings.update({f"predictions/{m['candidate_id']}.json": m["prediction_table_path"] for m in models})
    _require(set(plan["inputs"]) == {"source_freeze.json", *bindings}, "copied input roster differs")
    for name, original in bindings.items():
        _require(original in declared, "copied input is not bound by source freeze")
        copied = checked_relative_file(root, "inputs/" + name)
        require_hash(copied, declared[original]["sha256"])
        _require(copied.stat().st_size == declared[original]["bytes"], "copied input size differs")
    for model in models:
        prediction_path = root / f"inputs/predictions/{model['candidate_id']}.json"
        require_hash(prediction_path, model["prediction_table_sha256"])
        prediction_record = read_json(prediction_path)
        _require(prediction_record.get("schema") == "kahm.prediction_table.v1"
                 and prediction_record.get("candidate_id") == model["candidate_id"], "copied predictor identity differs")
    require_hash(root / "inputs/training.jsonl", dataset["development"]["training"]["sha256"])
    _require(len(_read_training(root / "inputs/training.jsonl")) == dataset["development"]["training"]["episode_count"],
             "copied training count differs")
    _require(read_json(root / "inputs/features.json").get("bank_id") == bank, "copied feature bank differs")
    features = _ordered_rows(read_json(root / "inputs/features.json")["entries"], "features")
    predictions = {candidate: tuple(_hex_fraction(r["prediction_hex"]) for r in
                    _ordered_rows(read_json(root / f"inputs/predictions/{candidate}.json")["predictions"], "predictions"))
                   for candidate in _CANDIDATES}
    classes = _classes(features, predictions, _read_training(root / "inputs/training.jsonl"), kappas)
    _require(read_json(root / "classes.json") == classes and _digest(classes) == plan["classes_sha256"], "frozen classes differ")
    _require(plan["class_ids"] == [c["class_id"] for c in classes], "class menu differs")
    expected_endpoints = plan["repetitions"] * len(plan["sample_sizes"]) * len(classes) * (2 * len(plan["block_sizes"]) + 2)
    _require(plan["endpoint_count"] == expected_endpoints
             and _fraction(plan["alpha_per_endpoint"]) * expected_endpoints == _fraction(plan["delta"]),
             "joint confidence allocation differs")
    expected_streams = [stream_descriptor(int(plan["master_seed"]), "calibration", f"class-risk-v1:{plan['run_id']}:rep-{r}")
                        for r in range(plan["repetitions"])]
    _require(type(plan["streams"]) is list and len(plan["streams"]) == len(expected_streams), "calibration stream count differs")
    for saved, expected in zip(plan["streams"], expected_streams, strict=True):
        _require(type(saved) is dict and type(saved.get("rng")) is dict, "invalid sampler runtime record")
        portable = dict(saved)
        rng = dict(saved["rng"])
        for key in ("python_implementation", "python_version", "python_full_version"):
            _require(type(rng.get(key)) is str and bool(rng[key]), "missing recorded Python provenance")
            rng[key] = expected["rng"][key]
        portable["rng"] = rng
        _require(portable == expected, "calibration stream identity differs")
    geometries = {c["class_id"]: _geometry(c) for c in classes}
    index = read_json(root / "certificate_index.json")
    expected_paths = [f"certificates/rep_{r}/n_{n}/{c['class_id']}/certificate.json"
                      for r in range(plan["repetitions"]) for n in plan["sample_sizes"] for c in classes]
    _require([x.get("path") for x in index] == expected_paths, "certificate family is incomplete or reordered")
    bound_paths = {x["path"]: x["sha256"] for x in index}
    summary = []
    for rep, stream in enumerate(expected_streams):
        data_path = root / f"calibration/rep_{rep}.json"
        data = read_json(data_path)
        _require(type(data) is list and len(data) == max(plan["sample_sizes"]), "calibration count differs")
        episodes = tuple(Episode(**row) for row in data)
        _require(episodes == sample_episodes(len(data), int(plan["master_seed"]), "calibration", stream["run_id"]),
                 "saved observations differ from the declared deterministic stream")
        histories = tuple(e.history_id for e in episodes)
        targets = tuple(OBSERVATIONS[e.next_state] for e in episodes)
        for n in plan["sample_sizes"]:
            for candidate in classes:
                relative = f"certificates/rep_{rep}/n_{n}/{candidate['class_id']}/certificate.json"
                path = checked_relative_file(root, relative)
                require_hash(path, bound_paths[relative])
                record = read_json(path)
                _require(record.get("repetition") == rep and record.get("plan_sha256") == sha256_file(root / "class_risk_plan.json")
                         and record.get("calibration_sha256") == sha256_file(data_path), "assessment provenance differs")
                summary.append(_verify_candidate(path.parent, candidate, geometries[candidate["class_id"]],
                                                 histories[:n], targets[:n], record, plan))
    if include_audit:
        from .class_oracle import verify_class_audit
        audits = read_json(root / "audit_index.json")
        _require([x.get("class_id") for x in audits] == plan["class_ids"], "audit family differs")
        by_id = {}
        for candidate, reference in zip(classes, audits, strict=True):
            path = checked_relative_file(root, reference["path"])
            require_hash(path, reference["sha256"])
            audit = read_json(path)
            _require(audit.get("block_sizes") == plan["block_sizes"], "audit block menu differs from the frozen plan")
            verified = verify_class_audit(geometries[candidate["class_id"]], audit, _predictions(candidate))
            _require(type(verified) is dict and verified.get("verified") is True, "class oracle verification failed")
            by_id[candidate["class_id"]] = audit
        for entry in summary:
            audit = by_id[entry["class_id"]]
            entry["optimal_risk_lower"] = audit["optimal_risk_lower"]
            entry["optimal_risk_upper"] = audit["optimal_risk_upper"]
            entry["fitted_risk"] = audit["fitted_risk"]
            lower = _fraction(entry["lower"])
            lo, hi = _fraction(audit["optimal_risk_lower"]), _fraction(audit["optimal_risk_upper"])
            entry["lower_coverage"] = "covered" if lower <= lo else "miss" if lower > hi else "oracle_interval_inconclusive"
            upper = _fraction(entry["fitted_upper"]) if entry["fitted_upper"] is not None else None
            entry["upper_coverage"] = "unavailable" if upper is None else "covered" if _fraction(audit["fitted_risk"]) <= upper else "miss"
            hierarchy = {row["block_size"]: row["expected_block_score"]
                         for row in audit.get("analytic_partition", {}).get("hierarchy", [])}
            for block in entry["blocks"]:
                reference = hierarchy.get(block["block_size"])
                block["expected_block_score"] = reference
                endpoint = block["block_lower"]
                block["hierarchy_coverage"] = ("unavailable" if endpoint is None else
                    "reference_unavailable" if reference is None else
                    "covered" if _fraction(endpoint) <= _fraction(reference) else "miss")
    return {"schema": "kahm.class_risk_verification.v1", "status": "verified", "plan_sha256": sha256_file(root / "class_risk_plan.json"),
            "run_id": plan["run_id"], "source_identity_matches": True, "numerical_witnesses_checked": True,
            "models_replayed": False, "population_audit_checked": include_audit,
            "calibration_stream_replayed_in_memory": True,
            "delta": plan["delta"], "endpoint_count": expected_endpoints, "alpha_per_endpoint": plan["alpha_per_endpoint"],
            "unavailable_endpoint_count": sum(row["unavailable_endpoints"] for row in summary),
            "status_counts": dict(Counter(row["status"] for row in summary)), "assessments": summary,
            "interpretation": "Numerical/provenance verification is distinct from favorable decisions and realized coverage. Statistical validity assumes iid calibration independent of development."}


def _write_report(output, result):
    lines = ["# Verified lower and upper class-risk assessment", "", f"Run: `{result['run_id']}`.", "",
             "The numerical witnesses and provenance were checked. Statistical conclusions assume independent identically distributed calibration episodes, independent of development.", "",
             "The predictors and features are exact archived finite-support computational tables; historical model replay and global executable-readout regularity are not claimed.", "",
             "| Class | n | Repetition | Lower | Class upper | Class-risk audit interval | Decision |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in result["assessments"]:
        lower = float(_fraction(row["lower"]))
        upper = "unavailable" if row["class_upper"] is None else f"{float(_fraction(row['class_upper'])):.8f}"
        lo, hi = float(_fraction(row["optimal_risk_lower"])), float(_fraction(row["optimal_risk_upper"]))
        lines.append(f"| {row['class_id']} | {row['n']} | {row['repetition']} | {lower:.8f} | {upper} | [{lo:.8f}, {hi:.8f}] | {row['status']} |")
    lines.extend(["", "Decimal values are descriptive approximations. Exact fractions in the JSON records govern every decision.", "",
                  "Each block record reports both the normalized-block lower confidence bound and the empirical-optimum bound obtained from the same optimization witnesses. The full empirical bound is reported separately.", "",
                  "Use `python -m kahm_demo verify-class-risk --run PATH` to recompute witness checks without refitting or solving new optimization problems.", ""])
    with (Path(output) / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))

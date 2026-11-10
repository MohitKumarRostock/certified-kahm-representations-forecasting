"""Portable CI and explicit matching-runtime gates for the released demonstrator.

The hosted gate never relaxes or claims the recorded primary runtime identity.
It selects every test except three named primary-runtime integration cases,
then builds an installed wheel and creates a separate full run on that runner.
The strict-primary command is an explicit, failing gate on incompatible hosts.
No command retries a failed experiment or treats favorable outcomes as success.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest


PRIMARY_RUNTIME_ONLY_TESTS = {
    "test_tutorial.RuntimeAndExplicitVerificationTests.test_primary_runtime_precheck_matches_without_running_verification":
        "Requires the executing interpreter/build, NumPy/BLAS metadata, and sources to match the archived primary runtime exactly.",
    "test_tutorial.RuntimeAndExplicitVerificationTests.test_runtime_and_source_differences_are_explicit":
        "Starts from the archived primary runtime and asserts that an injected NumPy version is the only runtime difference; a hosted Python build adds legitimate differences.",
    "test_tutorial.RuntimeAndExplicitVerificationTests.test_actual_full_verification_replays_all_three_without_creation_or_refitting":
        "Reexecutes the immutable primary snapshot under its mandatory exact runtime/source identity checks.",
}
CANDIDATES = ["raw", "kahm_mem", "kahm_local"]
CERTIFICATE_STATUSES = ("requirement_met", "unresolved", "certificate_unavailable")
PIPELINE_STAGES = ["generate_data", "fit_encoders", "fit_forecasters", "freeze_models", "certify", "audit"]
PRINCIPAL_ARTIFACTS = {
    "dataset_manifest": "data/manifest.json", "encoder_manifest": "encoders/manifest.json",
    "model_manifest": "models/manifest.json", "freeze": "frozen/freeze.json",
    "certificate": "certificates/certificate.json", "audit": "audits/audit.json",
}
NUMPY_VERSION = "2.3.5"
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
    "reproduction_plan.json": frozenset({"kahm.demo_reproduction_plan.v1"}),
    "reproduction.json": frozenset({"kahm.demo_reproduction.v1"}),
    "reproduction_failure.json": frozenset({"kahm.demo_reproduction_failure.v1"}),
    "ci_plan.json": frozenset({"kahm.ci_fresh_wheel_plan.v1"}),
    "ci_report.json": frozenset({"kahm.ci_fresh_wheel.v1"}),
}
_GENERIC_MARKER_NAMES = frozenset({"manifest.json", "receipt.json"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(record):
    return hashlib.sha256(_json_bytes({key: value for key, value in record.items() if key != "record_sha256"})).hexdigest()


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def _write_new(path, record):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(record))


def _present(path):
    return path.exists() or path.is_symlink()


def _project(path):
    root = Path(path).resolve()
    _require(root.is_dir() and (root / "pyproject.toml").is_file() and (root / "kahm_demo").is_dir(),
             "--project must identify the demonstrator source/evidence directory")
    return root


def _outside_scientific_bundles(path):
    """Inspect both spelling and resolved routes without creating any files.

    Generic manifests/receipts are recognized by schema. Dedicated scientific
    marker names fail closed even when incomplete or from an unsupported schema;
    interrupted evidence is not a suitable destination for a new report/run.
    """
    requested = Path(path)
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
            _require(ancestor.is_dir(), f"output ancestor is not a directory: {ancestor}")
            for name, schemas in _BUNDLE_MARKERS.items():
                marker = ancestor / name
                if not _present(marker):
                    continue
                _require(marker.is_file() and not marker.is_symlink(),
                         f"cannot inspect possible scientific bundle marker: {marker}")
                try:
                    record = _read(marker)
                except (OSError, UnicodeError, ValueError) as error:
                    raise ValueError(f"cannot inspect possible scientific bundle marker: {marker}") from error
                schema = record.get("schema") if isinstance(record, dict) else None
                recognized = type(schema) is str and schema in schemas
                _require(not recognized and name in _GENERIC_MARKER_NAMES,
                         f"output must remain outside existing scientific/experiment bundles: {marker}")
    return resolved


def _report_path(path, project):
    requested = Path(path)
    if _present(requested):
        raise FileExistsError(f"CI report already exists: {requested}; evidence is not overwritten")
    result = _outside_scientific_bundles(requested)
    protected = ("kahm_demo", "data", "encoders", "models", "frozen", "certificates", "audits")
    _require(not any(result.is_relative_to(project / name) for name in protected),
             "CI reports must remain outside code and scientific evidence bundles")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


@contextmanager
def _source_import_path(project):
    """Source tests deliberately use the checkout; wheel subprocesses do not."""
    before = sys.path.copy()
    sys.path.insert(0, str(project))
    try:
        yield
    finally:
        sys.path[:] = before


def flatten_tests(suite):
    result = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            result.extend(flatten_tests(item))
        else:
            result.append(item)
    return result


def select_tests(suite, mode="portable"):
    """Select exact IDs; missing exclusions or duplicate discovery fail closed."""
    _require(mode in {"portable", "strict-primary"}, "unsupported test selection mode")
    tests = flatten_tests(suite)
    ids = [test.id() for test in tests]
    _require(len(ids) == len(set(ids)), "test discovery returned duplicate IDs")
    missing = set(PRIMARY_RUNTIME_ONLY_TESTS) - set(ids)
    _require(not missing, "explicit primary-runtime test IDs missing from discovery: " + ", ".join(sorted(missing)))
    chosen = [test for test in tests if (test.id() not in PRIMARY_RUNTIME_ONLY_TESTS) == (mode == "portable")]
    _require(bool(chosen), "test selection is empty")
    chosen_ids = {test.id() for test in chosen}
    selection = {
        "schema": "kahm.ci_test_selection.v1", "mode": mode, "discovered_count": len(tests),
        "selected_count": len(chosen), "selected_ids": [test.id() for test in chosen],
        "primary_runtime_tests": [{"test_id": name, "reason": reason}
                                   for name, reason in PRIMARY_RUNTIME_ONLY_TESTS.items()],
        "excluded_ids": [test_id for test_id in ids if test_id not in chosen_ids],
        "exclusions_are_explicit_not_runtime_skips": True,
    }
    return unittest.TestSuite(chosen), selection


def discover_tests(project):
    loader = unittest.TestLoader()
    suite = loader.discover(str(project / "tests"), pattern="test_*.py")
    _require(not loader.errors, "test discovery failed:\n" + "\n".join(loader.errors))
    return suite


def _runtime_precheck(project):
    tutorial = importlib.import_module("kahm_demo.tutorial")
    return tutorial.runtime_compatibility(project)


def run_test_gate(project, *, mode, report):
    project = _project(project)
    path = _report_path(report, project)
    log_path = path.with_suffix(".log")
    if _present(log_path):
        raise FileExistsError(f"CI test log already exists: {log_path}")
    record = {
        "schema": "kahm.ci_test_gate.v1", "mode": mode, "status": "failed",
        "python_version": platform.python_version(), "python_full_version": sys.version,
        "original_primary_verification_performed": False,
        "runtime_identity_checks_relaxed": False, "automatic_retries_performed": False,
        "log_path": str(log_path),
    }
    started = time.monotonic()
    try:
        with _source_import_path(project):
            suite, selection = select_tests(discover_tests(project), mode)
            record["selection"] = selection
            if mode == "strict-primary":
                precheck = _runtime_precheck(project)
                record["runtime_precheck"] = precheck
                _require(precheck.get("compatible") is True,
                         "strict primary runtime/source precheck is incompatible or unavailable; no skip, fallback, or identity relaxation is permitted")
            with log_path.open("x", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            record.update(
                tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                skipped=[{"test_id": test.id(), "reason": reason} for test, reason in result.skipped],
                expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
            )
            passed = (result.wasSuccessful() and result.testsRun == selection["selected_count"]
                      and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses)
            record["status"] = "passed" if passed else "failed"
            record["original_primary_verification_performed"] = mode == "strict-primary" and passed
            if not passed:
                record["error"] = {"type": "TestGateFailure", "message": "Selected tests did not all execute and pass without unexpected skips."}
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["scope"] = (
        "Portable source tests include archive identity inspection and synthetic integration fixtures created on this runtime. The three named primary-runtime tests are explicitly excluded; the immutable primary numerical certificate is not reverified."
        if mode == "portable" else
        "Explicit matching-runtime gate for the three recorded-primary integration tests. Incompatible metadata, native verification errors, and test failures fail this gate without fallback."
    )
    _write_new(path, record)
    return record


INSPECTION_RUNNER = r'''
import builtins
import hashlib
import json
from pathlib import Path
import sys

configuration = json.loads(sys.argv[1])
project = Path(configuration['project']).resolve()
if configuration['source_import']:
    sys.path.insert(0, str(project))
forbidden = {'numpy', 'IPython', 'jupyter', 'nbformat', 'ipywidgets', 'matplotlib'}
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('archive inspection imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
import kahm_demo
from kahm_demo.tutorial import inspect_primary
package_file = Path(kahm_demo.__file__).resolve()
if configuration['source_import']:
    assert package_file.is_relative_to(project / 'kahm_demo')
else:
    assert package_file.is_relative_to(Path(configuration['installed_prefix']).resolve())
    assert not package_file.is_relative_to(project)
result = inspect_primary(project)
assert result['mode'] == 'archive_identity_only'
assert result['archive_identity_checked'] is True
for key in ('model_replayed', 'certificate_reverified', 'population_risk_recomputed',
            'calibration_losses_recomputed', 'calibration_resampled', 'models_refitted', 'root_search_repeated'):
    assert result[key] is False, key
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
print(json.dumps({'schema': 'kahm.ci_archive_inspection.v1', 'status': 'passed',
                  'mode': result['mode'], 'archive_identity_checked': True,
                  'package_file': str(package_file), 'package_version': kahm_demo.__version__,
                  'stdlib_only': True, 'candidate_ids': result['candidate_ids'], 'anchors': result['anchors'],
                  'model_replayed': False, 'certificate_reverified': False,
                  'population_risk_recomputed': False, 'calibration_losses_recomputed': False}))
'''

PROBE_RUNNER = r'''
import hashlib
import json
from pathlib import Path
import platform
import sys
import kahm_demo
import numpy
configuration = json.loads(sys.argv[1])
package = Path(kahm_demo.__file__).resolve().parent
prefix, project = Path(configuration['installed_prefix']).resolve(), Path(configuration['project']).resolve()
assert package.is_relative_to(prefix), 'package was not imported from the isolated installation'
assert not package.is_relative_to(project), 'checkout shadowed the installed wheel'
assert kahm_demo.__version__ == configuration['package_version']
assert numpy.__version__ == configuration['numpy_version']
for relative, digest in configuration['source_sha256'].items():
    assert hashlib.sha256((package / relative).read_bytes()).hexdigest() == digest, relative
assert sys.version_info >= (3, 12)
print(json.dumps({'schema': 'kahm.ci_installed_probe.v1', 'status': 'passed',
                  'package_file': str(package / '__init__.py'), 'package_version': kahm_demo.__version__,
                  'python_version': platform.python_version(), 'python_full_version': sys.version,
                  'numpy_version': numpy.__version__, 'installed_prefix': str(prefix),
                  'installed_package_source_hashes_match': True, 'checkout_shadowed_import': False}))
'''


def _environment():
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _command(arguments, *, cwd, logs, label):
    """One subprocess invocation, with retained output and no retry."""
    result = subprocess.run([str(value) for value in arguments], cwd=cwd, env=_environment(),
                            capture_output=True, text=True)
    stdout, stderr = logs / f"{label}.stdout.log", logs / f"{label}.stderr.log"
    with stdout.open("x", encoding="utf-8") as stream:
        stream.write(result.stdout)
    with stderr.open("x", encoding="utf-8") as stream:
        stream.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited {result.returncode}; retained logs: {stdout}, {stderr}")
    return result.stdout


def archive_inspect(project, *, report):
    project = _project(project)
    path = _report_path(report, project)
    record = {"schema": "kahm.ci_archive_gate.v1", "status": "failed",
              "original_primary_numerical_verification_performed": False}
    try:
        with tempfile.TemporaryDirectory(prefix="kahm_archive_inspection_") as directory:
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", INSPECTION_RUNNER,
                                     json.dumps({"project": str(project), "source_import": True})],
                                    cwd=directory, env=_environment(), capture_output=True, text=True)
            record["exit_code"] = result.returncode
            record["stderr"] = result.stderr
            _require(result.returncode == 0, "stdlib-only archive inspection failed: " + result.stderr)
            record["inspection"] = json.loads(result.stdout)
            _require(record["inspection"]["status"] == "passed", "archive inspector did not pass")
            record["status"] = "passed"
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    _write_new(path, record)
    return record


def summarize_reproduction(run_root, returned):
    """Check saved identities/outcome retention; never judge the outcome's favorability."""
    run_root = Path(run_root).resolve()
    record_path = run_root / "reproduction.json"
    record = _read(record_path)
    _require(record["schema"] == "kahm.demo_reproduction.v1" and record["status"] == "pipeline_complete"
             and record["pipeline_complete"] is True, "fresh pipeline did not complete")
    _require(record["record_sha256"] == _digest(record), "reproduction payload hash differs")
    _require(returned["reproduction_sha256"] == _sha(record_path), "CLI return does not bind the completed reproduction")
    _require(record["completed_stages"] == PIPELINE_STAGES and record["candidate_ids"] == CANDIDATES,
             "fresh pipeline did not retain the complete fixed sequence/family")
    _require(record["training_count"] == 4096 and record["validation_count"] == 1024
             and record["calibration_count"] == 4096 and record["forecast_search_count"] == 225
             and record["encoder_caps"] == [1, 2], "fresh CI run changed the full demonstrator menu")
    _require(record["automatic_retries_performed"] is False and record["unfavorable_outcomes_filtered"] is False,
             "fresh run declares retries or outcome filtering")
    _require(not any(path.name in {"reproduction_failure.json", "assessment_failure.json", "audit_failure.json"}
                     for path in run_root.rglob("*")), "completed run contains a retained failure record")
    _require(set(record["artifacts"]) == set(PRINCIPAL_ARTIFACTS), "principal artifact roster differs")
    for key, relative in PRINCIPAL_ARTIFACTS.items():
        path = run_root / relative
        _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_root)
                 and record["artifacts"][key] == {"path": relative, "sha256": _sha(path)},
                 "principal reproduction artifact identity differs: " + key)
    plan_path = run_root / "reproduction_plan.json"
    _require(record["reproduction_plan"] == {"path": "reproduction_plan.json", "sha256": _sha(plan_path)},
             "reproduction plan identity differs")
    certificate, audit = _read(run_root / "certificates/certificate.json"), _read(run_root / "audits/audit.json")
    _require([row["candidate_id"] for row in certificate["candidates"]] == CANDIDATES
             and [row["candidate_id"] for row in audit["candidates"]] == CANDIDATES,
             "fresh results omit or reorder a candidate")
    _require(record["candidates"] == audit["candidates"] == returned["candidates"]
             and record["audit_summary"] == audit["summary"] == returned["audit_summary"],
             "saved, returned, and audited outcomes differ")
    counts = Counter(row["status"] for row in certificate["candidates"])
    _require(set(counts) <= set(CERTIFICATE_STATUSES), "unknown certificate status")
    expected = {
        "all_three_requirements_met": counts["requirement_met"] == 3,
        "any_kahm_requirement_met": any(row["status"] == "requirement_met" for row in certificate["candidates"]
                                       if row["candidate_id"] != "raw"),
        "status_counts": {name: counts[name] for name in CERTIFICATE_STATUSES},
    }
    _require(record["certificate_summary"] == expected == returned["certificate_summary"],
             "certificate acceptance summary differs from the retained outcomes")
    for issued, audited in zip(certificate["candidates"], audit["candidates"], strict=True):
        _require(audited["certificate_status"] == issued["status"]
                 and audited["calibration_empirical_mse"] == issued["empirical_mse"]
                 and audited["issued_upper_endpoint"] == issued["upper_endpoint"],
                 "audit changed a certificate result")
        allowed = {"endpoint_unavailable"} if issued["status"] == "certificate_unavailable" else {"covered", "coverage_miss"}
        _require(audited["realized_coverage"] in allowed, "audit changed an unavailable/available outcome")
    return {"pipeline_complete": True, "reproduction_sha256": _sha(record_path),
            "run_id": record["run_id"], "master_seed_integer": record["master_seed_integer"],
            "candidate_ids": CANDIDATES, "candidates": record["candidates"],
            "certificate_summary": expected, "audit_summary": record["audit_summary"],
            "all_outcomes_retained": True, "automatic_retries_performed": False,
            "principal_artifacts": record["artifacts"]}


def _fresh_output(output, project):
    requested = Path(output)
    if _present(requested):
        raise FileExistsError(f"fresh-wheel output already exists: {requested}; no resume or replacement")
    root = _outside_scientific_bundles(requested)
    _require(not root.is_relative_to(project), "fresh-wheel output and working directories must be outside the checkout")
    return root


def fresh_wheel(project, *, output, run_id="ci-fresh-v0.1", master_seed=20261110):
    project = _project(project)
    _require(type(run_id) is str and bool(run_id.strip()), "run_id must be a nonempty string")
    _require(type(master_seed) is int and master_seed >= 0, "master_seed must be a nonnegative integer")
    root = _fresh_output(output, project)
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    source_hashes = {path.name: _sha(path) for path in sorted((project / "kahm_demo").iterdir())
                     if path.is_file() and path.suffix in {".py", ".json"}}
    root.mkdir(parents=True)
    logs, work = root / "logs", root / "work"
    logs.mkdir()
    work.mkdir()
    plan = {
        "schema": "kahm.ci_fresh_wheel_plan.v1", "status": "planned", "package_version": version,
        "run_id": run_id, "master_seed_integer": str(master_seed), "numpy_version": NUMPY_VERSION,
        "source_sha256": source_hashes, "ci_runner_sha256": _sha(Path(__file__)),
        "fresh_pipeline_invocation_limit": 1, "automatic_retries": False,
        "original_primary_strict_verification_requested": False,
        "success_is_independent_of_certificate_acceptance_and_coverage": True,
    }
    _write_new(root / "ci_plan.json", plan)
    record = {
        "schema": "kahm.ci_fresh_wheel.v1", "status": "failed", "package_version": version,
        "plan_sha256": _sha(root / "ci_plan.json"), "fresh_pipeline_invocation_count": 0,
        "original_primary_numerical_verification_performed": False,
        "runtime_identity_checks_relaxed": False, "automatic_retries_performed": False,
        "output": str(root), "report_path": str(root / "ci_report.json"),
    }
    started, phase = time.monotonic(), "build_wheel"
    try:
        source, wheels, environment = root / "build-source", root / "wheelhouse", root / "venv"
        source.mkdir()
        wheels.mkdir()
        for path in project.iterdir():
            if path.is_file() and (path.name in {"pyproject.toml", "README.md"}
                                   or path.name.startswith(("LICENSE", "COPYING", "NOTICE"))):
                shutil.copyfile(path, source / path.name)
        shutil.copytree(project / "kahm_demo", source / "kahm_demo",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        _command([sys.executable, "-I", "-B", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                  "--no-index", str(source), "--wheel-dir", str(wheels)], cwd=work, logs=logs, label=phase)
        candidates = sorted(wheels.glob("*.whl"))
        _require(len(candidates) == 1 and candidates[0].name.startswith(f"kahm_demonstrator-{version}-"),
                 "build did not produce exactly the expected project wheel")
        wheel = candidates[0]
        record["wheel"] = {"path": str(wheel), "sha256": _sha(wheel)}
        phase = "create_environment"
        _command([sys.executable, "-I", "-B", "-m", "venv", "--system-site-packages", str(environment)],
                 cwd=work, logs=logs, label=phase)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        phase = "install_wheel"
        _command([python, "-I", "-B", "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile",
                  "--force-reinstall", str(wheel)], cwd=work, logs=logs, label=phase)
        phase = "installed_origin_probe"
        configuration = {"project": str(project), "installed_prefix": str(environment), "package_version": version,
                         "numpy_version": NUMPY_VERSION, "source_sha256": source_hashes}
        probe = json.loads(_command([python, "-I", "-B", "-c", PROBE_RUNNER, json.dumps(configuration)],
                                    cwd=work, logs=logs, label=phase))
        _require(probe["status"] == "passed" and probe["checkout_shadowed_import"] is False,
                 "installed-wheel origin/source probe did not pass")
        record["installed_probe"] = probe
        phase = "installed_archive_inspection"
        inspection = json.loads(_command([python, "-I", "-B", "-c", INSPECTION_RUNNER,
                                           json.dumps({"project": str(project), "source_import": False,
                                                       "installed_prefix": str(environment)})],
                                          cwd=work, logs=logs, label=phase))
        _require(inspection["status"] == "passed" and inspection["mode"] == "archive_identity_only",
                 "installed archive inspection did not pass")
        record["archive_inspection"] = inspection
        phase = "fresh_pipeline"
        record["fresh_pipeline_invocation_count"] = 1
        returned = json.loads(_command([python, "-I", "-B", "-m", "kahm_demo", "reproduce",
                                         "--output", str(root / "run"), "--master-seed", str(master_seed),
                                         "--run-id", run_id], cwd=work, logs=logs, label=phase))
        phase = "retain_outcomes"
        record["fresh_reproduction"] = summarize_reproduction(root / "run", returned)
        _require(record["fresh_reproduction"]["run_id"] == run_id
                 and record["fresh_reproduction"]["master_seed_integer"] == str(master_seed),
                 "fresh run identity differs from the committed CI plan")
        _require(source_hashes == {name: _sha(project / "kahm_demo" / name) for name in source_hashes},
                 "source package changed during installed-wheel CI")
        record["status"] = "passed"
    except BaseException as error:
        record["failure_phase"] = phase
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["scope"] = (
        "CI pass means the installed package and the complete fresh pipeline executed successfully on this runner. "
        "All three scientific outcomes are retained, including unresolved or unavailable certificates and coverage misses. "
        "The fresh certificate keeps its conditional iid premises. The archived primary example received identity-only "
        "inspection; its original numerical runtime identity was neither relaxed nor claimed to match."
    )
    _write_new(root / "ci_report.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("portable", "strict-primary", "archive-inspect"):
        command = commands.add_parser(name)
        command.add_argument("--project", type=Path, default=Path("."))
        command.add_argument("--report", type=Path, required=True)
    command = commands.add_parser("fresh-wheel")
    command.add_argument("--project", type=Path, default=Path("."))
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--run-id", default="ci-fresh-v0.1")
    command.add_argument("--master-seed", type=int, default=20261110)
    args = parser.parse_args(argv)
    try:
        if args.command in {"portable", "strict-primary"}:
            result = run_test_gate(args.project, mode=args.command, report=args.report)
        elif args.command == "archive-inspect":
            result = archive_inspect(args.project, report=args.report)
        else:
            result = fresh_wheel(args.project, output=args.output, run_id=args.run_id, master_seed=args.master_seed)
    except BaseException as error:
        result = {"schema": "kahm.ci_command_failure.v1", "status": "failed", "command": args.command,
                  "error": {"type": type(error).__name__, "message": str(error)},
                  "automatic_retries_performed": False}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

# CI: portable checks and strict recorded-runtime verification

The prepared GitHub workflow checks the package on its current runner and runs
one new, complete demonstration there. The original primary snapshot retains its
exact runtime and source checks. A normal GitHub-hosted Python installation is
not claimed to match the recorded CPython build, NumPy/BLAS configuration, or
other numerical runtime metadata.

The workflow is prepared locally. Its presence is not evidence that it has run
on GitHub. Execution evidence must identify the actual run and its retained
reports.

## Four explicit commands

Run these commands from the project directory. Use fresh report paths and a
fresh external directory for the installed-wheel demonstration.

| Command | What it establishes |
| --- | --- |
| `python scripts/ci_checks.py archive-inspect --project . --report /tmp/kahm-archive-inspection.json` | The released archive identities and recorded values can be inspected with the standard library. It does not replay models or reverify a certificate. |
| `python scripts/ci_checks.py portable --project . --report /tmp/kahm-portable-ci.json` | Every discovered test other than the three named recorded-runtime cases executes and passes. |
| `python scripts/ci_checks.py fresh-wheel --project . --output /tmp/kahm-fresh-wheel --run-id local-ci-v0.1` | An installed wheel outside the checkout runs a new complete development, freeze, calibration, certification, and population-audit sequence. |
| `python scripts/ci_checks.py strict-primary --project . --report /tmp/kahm-strict-primary.json` | The three recorded-primary integration tests pass after an explicit matching-runtime/source precheck. |

Each command exits nonzero on failure. Test reports are never overwritten.
The fresh-wheel command refuses existing output, retains its plan and any partial
evidence on failure, and never resumes or retries an experiment automatically.
Successful CI execution does not require an accepted certificate or successful
realized coverage.

New reports and fresh-run directories must also remain outside existing data,
encoder, model, freeze, certificate, audit, reproduction, and CI experiment
bundles. The shared guard checks both the requested path and its resolved
symlink route. A fresh sibling directory is allowed; a nested destination cannot
add files to preserved evidence such as `frozen/primary/`.

## Portable test selection

The portable runner discovers `tests/test_*.py`, then excludes exactly these
three test IDs from `test_tutorial.RuntimeAndExplicitVerificationTests`:

| Test method | Reason it belongs to the separate strict gate |
| --- | --- |
| `test_primary_runtime_precheck_matches_without_running_verification` | Its positive assertion requires an exact match to the archived primary runtime and source identities. |
| `test_runtime_and_source_differences_are_explicit` | It injects one NumPy version mismatch and expects that to be the only runtime difference. That assertion also assumes the remaining host metadata already matches the primary snapshot. |
| `test_actual_full_verification_replays_all_three_without_creation_or_refitting` | It executes the original frozen primary family through the full strict certificate/audit chain. |

These are an explicit selection policy, not conditional skips added after a
test fails. Missing names or duplicate discovered IDs fail selection. The JSON
report lists every selected ID and the excluded IDs with their reasons. An
unexpected `unittest` skip, expected failure, incomplete execution, test failure,
or import error fails the gate.

All `PrimaryInspectionTests` remain included, together with the history-input
tests and the portable negative tests for missing runtime dependencies, native
verification failures, invalid verification flags, and mismatched replayed
predictions. The complete scientific unit tests and synthetic integration tests
also remain included. Their temporary models and certificates are created and
checked on the same executing runtime; they do not relax the original snapshot.

The ordinary command `python -m unittest discover -s tests -v` still runs the
complete suite. Use it when the recorded primary runtime is available. The
portable command makes its narrower scope visible in its report.

## Installed-wheel and fresh-run gate

The initial hosted configuration is Ubuntu 24.04, Python 3.12, and NumPy 2.3.5.
This is one explicit CI configuration, not a claim of an already tested platform
matrix. Install the local gate's dependencies with:

```bash
python -m pip install 'numpy==2.3.5' 'setuptools>=77.0.3' wheel
```

`fresh-wheel` copies the package and build metadata into a new external staging
directory. It builds one wheel without resolving dependencies, creates a fresh
virtual environment, and installs that wheel. The virtual environment inherits
the host's already installed numerical dependencies; its package installation
is separate. Python runs in isolated mode (`-I`), outside the checkout, with
`PYTHONPATH` removed. The probe requires the imported package to reside inside
the new environment and compares its package-source bytes with the committed
CI plan. A checkout-shadowed import fails the gate.

The installed package first performs archive-only inspection with NumPy and
notebook imports prohibited. It then calls the existing `reproduce` entry point
exactly once, with a distinct CI run identity and fresh output. The full menu
remains fixed: 4,096 training episodes, 1,024 validation episodes, both encoder
banks, all 225 development fits, three selected predictors, and 4,096 shared
calibration episodes. Calibration is materialized after the model freeze, and
model-specific population risks are evaluated only after the existing
certificate workflow completes.

The original primary numerical verifier is not invoked by this gate. The fresh
run creates and verifies its own snapshot on the executing runtime. Its
statistical statement retains the declared conditional iid assumptions and the
same fixed confidence allocation and tolerance.

The gate checks the completed reproduction's principal hashes and preserves
all three results. `unresolved`, `certificate_unavailable`, and `coverage_miss`
remain reported outcomes; none triggers a replacement sample, alternative seed,
refit, or retry. A completed pipeline with such outcomes can pass CI because
the workflow and evidence are functioning correctly. A failed calculation or
verification cannot pass.

## Strict primary gate

`strict-primary` performs the recorded runtime/source compatibility check first.
If it is incompatible or unavailable, the command writes a failed report and
exits nonzero. It does not silently skip the tests, fall back to archive display,
rewrite archived identities, or modify numerical checks.

When the precheck matches, the command runs the three explicitly listed tests,
including the real primary forecast/certificate/audit verification. This gate
is opt-in on a matching environment. The hosted workflow does not pretend that
selecting Python 3.12 and NumPy 2.3.5 alone reproduces the original interpreter
build and numerical runtime.

## Retained CI evidence and workflow permissions

`.github/workflows/ci.yml` runs on pushes, pull requests, and explicit workflow
dispatch. It requests only `contents: read`, does not persist checkout
credentials, and contains no publication, deployment, package-upload, or secret
inputs.

The artifact-upload step uses `if: always()`. It retains the JSON reports, test
logs, CI plan, wheel, subprocess logs, and the complete fresh run directory,
including partial outputs after failure. It excludes the duplicate build tree
and virtual environment. A canceled job or infrastructure failure can prevent
completion or artifact delivery; the workflow does not convert either into a
success claim.

The CI run identity uses the GitHub run ID. Re-running the same GitHub run uses
the same stream identity; it supplies no new independent evidence. The artifact
name also includes the attempt number so earlier retained attempts are not
replaced. Separate CI runs are software checks, not an outcome-filtered study
or an empirical estimate of 95% coverage.

The workflow pins the following official action releases to complete commit
identities, checked on 2026-11-10:

| Action | Release | Official commit |
| --- | --- | --- |
| `actions/checkout` | v7.0.1 | [3d3c42e5aac5ba805825da76410c181273ba90b1](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1) |
| `actions/setup-python` | v7.0.0 | [5fda3b95a4ea91299a34e894583c3862153e4b97](https://github.com/actions/setup-python/commit/5fda3b95a4ea91299a34e894583c3862153e4b97) |
| `actions/upload-artifact` | v7.0.2 | [cf430e030ddbb5b0abf93d22962f4752f3646cd9](https://github.com/actions/upload-artifact/commit/cf430e030ddbb5b0abf93d22962f4752f3646cd9) |

Update these pins through an ordinary reviewed change after checking the
official release and commit. The scientific runtime identity gates remain
unchanged when workflow dependencies are updated.

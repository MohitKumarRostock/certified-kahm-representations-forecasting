# Contributing

The repository for the KAHM forecasting demonstrator is
`MohitKumarRostock/certified-kahm-representations-forecasting`. It is licensed
under [Apache-2.0](LICENSE), with **Copyright 2026 Mohit Kumar** and software
author **Mohit Kumar**.

Contributions intentionally submitted for inclusion are governed by Section 5
of the licence unless explicitly stated otherwise or covered by a separate
agreement. No separate contributor agreement is introduced here. Preserve
applicable notices and identify changes when redistributing modified files.

## Start with the example and its scope

Use the [quick start](docs/QUICKSTART.md) and the guided notebook before changing
the numerical workflow. Default inspection displays identity-checked recorded
values and does not execute the predictors. Explicit verification checks the
saved computational family, calibration losses, endpoints, and population audit.
A fresh run fits new models and uses its own data and evidence directories.

The [scientific contract](kahm_demo/specification.json) defines the finite
benchmark, evaluation family, loss, confidence allocation, and claim limits.
The [method overview](docs/METHOD_OVERVIEW.md) links the complete local
definitions and the fitted-risk proof. Improvements should keep
the distinction between a fitted predictor, an entire decoder class, and a
representation's information content explicit.

## Local development

Use Python 3.12 or newer. From the project root, install the package into your
chosen development environment:

```bash
python -m pip install 'numpy==2.3.5' 'setuptools>=77.0.3' wheel
python -m pip install -e .
python -m kahm_demo tutorial --project . --mode inspect --history 0.25 0.75
```

The pinned NumPy and host build tools above are prerequisites for the
fresh-wheel check, which builds without dependency downloads or build
isolation. They do not establish the primary run's full runtime identity.

The notebook and plotting dependencies are optional:

```bash
python -m pip install -e '.[tutorial]'
```

Installing the package does not reproduce the supplied primary runtime.
Strict replay checks the recorded Python build, NumPy/BLAS configuration,
machine, and numerical source identities. Version constraints or a successful
installation alone do not establish that match.

## Checks to run

Use the repository's public check runner. Its modes and result schemas are
documented in [CI.md](docs/CI.md).

| Purpose | Command from the project root |
| --- | --- |
| Check the supplied archive without replaying it | `python scripts/ci_checks.py archive-inspect --project . --report ../kahm-archive-inspection.json` |
| Run the portable regression checks | `python scripts/ci_checks.py portable --project . --report ../kahm-portable-checks.json` |
| Build an isolated wheel and complete a fresh integration run | `python scripts/ci_checks.py fresh-wheel --project . --output ../kahm-contribution-run --run-id contribution-check` |
| Explicitly verify the original primary run in its matching runtime | `python scripts/ci_checks.py strict-primary --project . --report ../kahm-strict-primary.json` |

The fresh-wheel output must be a new directory outside the checkout. For a
later run, choose another unused directory and retain the earlier result.
Report files also must be new; the example commands place them outside the
checkout so they do not become source changes.
This mode performs a fresh experiment; archive inspection and strict-primary
verification operate on existing evidence.

Portable checks distinguish tests that work on supported runtimes from the
small, explicit set requiring the primary run's exact environment. The
exclusion list is maintained in the check runner and documented in `CI.md`.
A portable pass is not a strict-primary pass. Strict verification fails on an
execution identity mismatch and does not substitute archive inspection.

The ordinary full suite is also available in a matching environment:

```bash
python -m unittest discover -s tests -v
```

For a focused change, first run the affected tests. Add or update tests where
they establish the changed behavior, a numerical boundary, or a provenance
guard. A change in text alone normally needs link and rendering checks, not a
new numerical experiment. A change to fitting or verification needs the
relevant regression coverage and a fresh integration result.

## Preserve the original evidence

The primary artifacts are retained scientific records. Do not rewrite their
contents to make a new implementation pass an old check. This includes
`data/primary`, `encoders/primary`, `models/primary`, `frozen/primary`,
`certificates/primary`, and `audits/primary`, as well as historical validation
reports. Their saved model statuses and recorded source/runtime identities
remain historical facts.

Changing a numerical source file included in a freeze, certificate, or audit
identity can make strict verification of that old record fail. Explain that
effect in the contribution and use a separate fresh run to assess the changed
implementation. An old record still requires its identified implementation and
runtime; copying new source bytes into its archived source directory does not
repair its provenance.

Changes to the scientific contract, confidence protocol, target law, tolerance,
selection rule, or numerical work limits require an explicitly versioned
scientific decision and a new assessment. Do not edit the primary artifacts,
reuse their untouched status, or retain their conclusions for a changed task.
The [adaptation guide](docs/ADAPTING_TO_NEW_DATA.md) explains the wider decisions
needed for another dataset or evaluation law.

## Numerical and statistical contribution rules

- Fit transformations and predictors on the declared development data; freeze
  them before accessing the relevant calibration outcomes.
- Retain repeated observations in fitting and empirical-loss denominators.
  Reference and landmark identity selection has its own documented rules.
- Use exact stored fractions for endpoint, tolerance, and coverage comparisons.
  Rounded display values do not determine scientific decisions.
- Keep unavailable certificates, unresolved decisions, failed checks, and
  coverage misses in the evidence. Fixing a defect does not justify silently
  replacing a sample or selecting a more favorable seed.
- Keep oracle population-risk calculations in their declared audit role. They
  must not become a model-selection criterion or a way to repair issued bounds.
- Preserve the distinction between archive inspection, source/runtime
  prechecking, strict verification, and a fresh experiment in APIs and reports.
- Describe the conditional sampling assumptions and the actual computation
  being assessed. A new example does not inherit the primary benchmark's
  guarantee merely by using the same functions.

The detailed numerical contracts are in [Encoder design](docs/ENCODER_DESIGN.md),
[Readout design](docs/READOUT_DESIGN.md), and
[Certification design](docs/CERTIFICATION_DESIGN.md).

## Reporting a problem or proposing a change

Use the repository's issue and pull request interfaces for problems and
proposed changes. No separate maintainer email address is declared here.

A useful report gives the package version or commit, Python/NumPy versions,
exact command, selected mode, run identifier, and observed error or JSON status.
For an artifact disagreement, include the affected relative path and expected
versus observed identity. Distinguish an expected runtime mismatch from a
failure in an environment that actually matches the recorded metadata.

A proposed change should explain the problem, the resulting behavior, and the
checks that support it. Identify any change in scientific scope or compatibility
with existing evidence. Report actual local check results and, once available,
the corresponding remote CI run for the proposed commit. Do not describe a
workflow as having passed on GitHub merely because its configuration or a local
equivalent exists.

For changes to a method, explain the relevant mathematical definition and
assumptions, and cite external work where appropriate. Preserve third-party
notices. Include only material whose redistribution terms permit its inclusion.

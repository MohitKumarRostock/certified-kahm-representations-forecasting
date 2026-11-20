# Software versions

## 0.2.0 — class-risk assessment

Version 0.2.0 provides:

- The separately versioned `class-risk` assessment protocol and
  `verify-class-risk` command.
- Exact finite-support Lipschitz class definitions and membership checks,
  weighted dual optimization witnesses, verified primal bounds, and lower
  and upper binary-KL confidence endpoints.
- Block-score and empirical-risk lower-confidence constructions, including
  reuse of the same block witnesses for an empirical-optimum bound.
- Short- and full-history analytical controls and population class-optimum
  audits performed after sample certificates are issued.
- A 180-endpoint scientific study, complete numerical verification archive,
  scientific figures, mathematical documentation, and 51 focused numerical
  and provenance tests.
- Installed-wheel class-risk integration in GitHub Actions.
- Self-contained definitions of optimal class risk, fitted-predictor risk,
  Lipschitz bounds, MSE thresholds, admissibility, block scores, and
  simultaneous confidence bounds.
- Installation instructions and citation metadata for the software, with
  separate identities for the complete source project and numerical evidence ZIP.

The [validation report](reports/standalone_0.2.0/report.md) records completed
checks and their scope. The class-risk protocol has its own calibration
observations, class definitions, confidence allocation, source snapshots,
and results. Its allocation is separate from the fitted-predictor assessment.

## 0.1.1 — fitted-predictor workflow

Version 0.1.1 provides the fitted-predictor workflow and the `public-v0.1.0`
experiment, with these interface and validation rules:

- Complete-ZIP download, extraction, project-directory, and local HTML viewing
  instructions; the tutorial requires `data/primary/`.
- Exact validation of command-line observations against the finite benchmark
  support in both the package CLI and the guided tutorial.
- Controlled rejection of malformed encoder runtime metadata.
- Separate JSON and test-log outputs for a CI report whose name ends in `.log`.
- Explicit iid calibration assumptions.

The v0.1.1 execution-source bytes differ from those bound by the
`public-v0.1.0` freeze. Archive inspection is supported. Strict replay of that
experiment requires the complete unmodified `v0.1.0` sources and exact recorded
runtime. Each separately executed experiment records its own source/runtime
identities, models, calibration losses, endpoints, and audit.

See the [0.1.1 validation report](reports/PATCH_0.1.1_VALIDATION.md) for the
checks performed and the [release guide](docs/RELEASE_GUIDE.md) for artifact
identity requirements.

## 0.1.0 — fitted-risk demonstration

Version 0.1.0 provides:

- A complete finite-benchmark demonstration of RAW, KAHM membership, and
  KAHM local-coordinate forecasting representations.
- Definitions of the benchmark, encoder, readout, development choices, and
  verified fitted-risk guarantee.
- An elementary bounded-loss concentration proof, simultaneous confidence
  allocation, and conservative numerical endpoint checks.
- The `public-v0.1.0` experiment: a recorded plan, both encoder banks, all
  225 development fits, the selected family, and every calibration and
  population-audit outcome.
- Separate archive-inspection, strict-verification, and experiment-execution
  routes.
- A guided notebook, static HTML reading view, command-line tutorial,
  contributor guide, automated checks, and pinned GitHub Actions workflow.
- Apache-2.0 licensing, Copyright 2026 Mohit Kumar, software author Mohit
  Kumar, and a repository-based software citation.

The [0.1.0 validation report](reports/RELEASE_VALIDATION.md) records the checks
performed for this source/runtime snapshot. `PUBLICATION_STATUS.json` records
software preparation and validation scope; repository and GitHub Actions
records identify remote publication and CI results.

## Tutorial materials

The interactive notebook executes cells and responds to changes in `HISTORY`.
The static HTML displays saved output. A worked LOCAL example for
`[0.25, 0.75]` relates the six MEM memberships to local PCA coordinates,
membership-weighted coordinates, and all 18 LOCAL entries. Its displayed
values belong to the `public-v0.1.0` experiment.

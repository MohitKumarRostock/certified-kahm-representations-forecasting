# Changelog

## Unreleased — documentation refinements

- Identify cell execution and changes to `HISTORY` as interactive-notebook
  actions; explain that the static HTML displays saved output.
- Add a fixed worked LOCAL example for the archived history `[0.25, 0.75]`,
  connecting the six MEM memberships, local PCA coordinates, weighted
  coordinates, and all 18 vector positions.

These documentation changes retain the notebook code and saved outputs.
The published v0.1.1 tag, package assets, and release manifest remain the
original release snapshot.

## 0.1.1

This patch improves first use and error handling while retaining the complete
`public-v0.1.0` experiment and its recorded outcomes.

- Add explicit complete-ZIP download, extraction, project-directory, and local
  HTML viewing instructions. Include `data/primary/` in the tutorial prerequisites.
- Check tutorial command-line observations as exact numbers before accepting
  values from the finite benchmark support. Apply the same rule to the package
  CLI and the human-readable guided tutorial.
- Reject malformed encoder runtime metadata with a controlled validation error.
- Keep a CI report whose name ends in `.log` separate from its test log, retaining
  both the JSON report and test output.
- Clarify the stipulated iid calibration premise in the introductory explanation.

The corrected encoder loader changes a source file bound by the historical
primary freeze. Archive inspection remains supported, but strict replay of the
original primary evidence requires the complete unmodified `v0.1.0` source and
its exact recorded runtime. New experiments use `0.1.1` and retain their own
source/runtime identities. No primary model, data, loss, endpoint, or audit is
rewritten by this patch.

See [patch validation](reports/PATCH_0.1.1_VALIDATION.md) for the checks performed
and [the release guide](docs/RELEASE_GUIDE.md) for preservation requirements.

## 0.1.0 — self-contained public edition

- Provides a complete finite-benchmark demonstration of RAW, KAHM membership,
  and KAHM local-coordinate forecasting representations.
- Documents the benchmark, encoder, readout, development choices, and verified
  fitted-risk guarantee within this repository.
- Supplies the elementary bounded-loss concentration proof, the simultaneous
  confidence allocation, and the conservative numerical endpoint checks.
- Includes one newly generated public primary experiment with a recorded plan,
  both encoder banks, all 225 development fits, the selected family, and every
  calibration and population-audit outcome.
- Keeps archive inspection, strict verification, and fresh experiments as
  explicit separate operations.
- Includes a guided notebook, static HTML reading view, command-line tutorial,
  contributor guide, automated checks, and a pinned GitHub Actions workflow.
- Records Apache-2.0 licensing, Copyright 2026 Mohit Kumar, software author
  Mohit Kumar, and the repository-based software citation.

Actual local validation is recorded in `reports/RELEASE_VALIDATION.md`.
`PUBLICATION_STATUS.json` distinguishes local preparation from an observed
remote repository publication or GitHub Actions run.

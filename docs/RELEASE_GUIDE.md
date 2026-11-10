# Release guide

The public edition is self-contained. Its method definitions, benchmark,
implementation decisions, guarantee proof, and numerical verification rules
are supplied in this repository. The release's introductory research statement
is an origin statement; the local mathematics and recorded evidence support
its specific technical claims.

## Software identity

| Field | Value |
| --- | --- |
| Repository | `MohitKumarRostock/certified-kahm-representations-forecasting` |
| Package | `kahm-demonstrator` |
| Version | `0.1.1` |
| Software author | Mohit Kumar |
| Copyright | Copyright 2026 Mohit Kumar |
| License | Apache-2.0; complete terms in [LICENSE](../LICENSE) |
| Software citation | [CITATION.cff](../CITATION.cff) and [CITATION.md](../CITATION.md) |

No software DOI or author identifier is invented. `PUBLICATION_STATUS.json`
records release preparation. GitHub's commit, release, and workflow records
establish actual publication and remote CI state; a prepared archive or
workflow configuration alone does not establish either.

## What the release contains

The [README](../README.md) and [quick start](QUICKSTART.md) lead to an
inspection-only walkthrough, the guided notebook, and the static reading view.
[Method overview](METHOD_OVERVIEW.md) links the complete local definitions.
[Fitted-risk guarantee](FITTED_RISK_GUARANTEE.md) gives the proof and numerical
conditions behind the certificate, and [Certificates explained](CERTIFICATES_EXPLAINED.md)
interprets its practical scope.

The supplied evidence includes the public primary development data, both
encoder banks, all 225 search states, selected model freeze, reserved calibration
outcomes, exact loss ledger, verified endpoints, and exact finite-law population
audit. Every candidate and outcome is retained. A wheel provides the software;
the complete project also provides these experiment artifacts.

Version `0.1.1` is an interface and documentation patch. The primary experiment
remains `public-v0.1.0`, and its evidence and original validation reports are
retained unchanged. See [the changelog](../CHANGELOG.md) for the fixes and
[the patch validation report](../reports/PATCH_0.1.1_VALIDATION.md) for checks
performed on this software version.

The public primary run was defined by a local plan before its execution:
master seed `20261110`, run ID `public-v0.1.0`, 4,096 training episodes,
1,024 validation episodes, and 4,096 reserved calibration episodes. A local
plan and editable hashes do not constitute independent public preregistration.
The random-stream construction is a reproducibility mechanism and does not
prove the iid statistical premise.

## Evidence identities and preservation

The exact primary freeze, certificate, and audit identities are listed in
`reports/RELEASE_VALIDATION.md` and checked by `kahm_demo.tutorial`.
The retained `RELEASE_MANIFEST.json` lists the size and SHA-256 digest of each
payload in the published [v0.1.1 release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.1).
It describes that release snapshot and excludes itself to avoid a circular
digest. Later documentation commits on `main` can differ from its entries;
their Git commit identities record those changes. Check the manifest against
the v0.1.1 tag or complete release ZIP.

The tutorial's presentation anchors are assigned after the primary evidence
has been issued. The archived source snapshot keeps their pre-issuance
unassigned values; `tutorial.py` is outside the scientific execution-source
roster. This avoids placing a freeze's own digest inside the bytes it hashes.
The issued scientific artifacts are never rewritten to change those anchors.

Do not edit retained evidence to make a changed implementation pass an old
verification. A change to an execution source, specification, statistical
protocol, or numerical rule requires the appropriate versioned workflow and
new evidence. Preserve all outcomes of that new run.

## Validation commands

From a complete checkout, install the build dependencies and use unused output
paths:

```bash
python -m pip install 'numpy==2.3.5' 'setuptools>=77.0.3' wheel
python scripts/ci_checks.py archive-inspect --project . --report ../kahm-release-archive.json
python scripts/ci_checks.py portable --project . --report ../kahm-release-portable.json
python scripts/ci_checks.py fresh-wheel --project . --output ../kahm-release-check --run-id release-check
```

The fresh-wheel command runs one separate complete experiment and retains all
results. Do not repeatedly change seeds or run identities to obtain acceptance.
For strict verification of the historical primary experiment, use the complete
unmodified [`v0.1.0` release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.0)
in the primary runtime. Run this command from that original source directory:

```bash
python scripts/ci_checks.py strict-primary --project . --report ../kahm-release-strict.json
```

The recorded runtime includes the full Python build, architecture, and complete
NumPy/BLAS configuration. Matching Python and NumPy version numbers alone is
insufficient. A strict mismatch fails; it is never replaced by an inspection
result under a verification label. See [CI.md](CI.md).

The encoder-loader correction in `0.1.1` changes a source file bound by the
primary freeze. Consequently, even a matching runtime cannot strictly replay
the historical primary evidence using `0.1.1` source. Inspection remains
available. Fresh experiments created with `0.1.1` record their own source and
runtime identities and are verified against those identities.

Before distributing a package, inspect its wheel metadata and check the archive
manifest after extracting it into a clean directory. Confirm that imports come
from that extraction or an isolated installation. The validation report must
name only checks actually performed on this version. Remote CI needs its own
observed run URL and result.

## Publishing and later versions

Publish the identified source and assets at the confirmed repository. Record
the actual commit, tag, release URL, asset digests, and observed CI results when
available. Use the complete Apache-2.0 license and notice, retain applicable
third-party notices, and keep the software citation consistent with the actual
version and publication date.

A later public research reference can be added as further reading without
becoming a runtime dependency. Adding a bibliography alone does not change
an existing certificate. Changing a hash-bound resource or execution source
does change its identity and must be handled through the documented workflow.

The software release does not extend the statistical claim beyond the stated
conditional iid family of three frozen predictors, bounded squared loss, and
one calibration experiment. The exact population audit is available because
this benchmark has a known finite law.

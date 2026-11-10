# Patch validation — version 0.1.1

Version `0.1.1` corrects the issues identified in an internal AI-assisted
technical review of `0.1.0`. It improves first-use instructions, exact tutorial
input validation, malformed encoder-metadata errors, and CI report/log naming.
The review and these checks are not independent human peer review or a formal
proof of the software. The scientific statement remains the conditional
guarantee documented in [the local proof](../docs/FITTED_RISK_GUARANTEE.md).

## Preserved historical evidence

All **552 original scientific-evidence files** and all **22 original report
files** were compared with the validated `0.1.0` candidate. Every size and
SHA-256 digest matches: **574 files, 76,777,587 bytes, no changes or omissions**.

| Directory | Original files preserved |
| --- | ---: |
| `data/` | 6 |
| `encoders/` | 6 |
| `models/` | 235 |
| `frozen/` | 287 |
| `certificates/` | 11 |
| `audits/` | 7 |
| Historical `reports/` files | 22 |

The primary run remains `public-v0.1.0`. Its freeze, certificate, audit, and
reported numerical outcomes are unchanged. The original
[release-validation report](RELEASE_VALIDATION.md) continues to describe the
checks performed on `0.1.0`; its counts are not presented as new `0.1.1` tests.
The original public tag and release remain the reference for those source bytes.

The encoder-loader correction changes `encoder_state.py`, which is bound by
the historical freeze. Therefore strict replay of that historical experiment
requires the complete unmodified
[`v0.1.0` release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.0)
and its exact recorded Python/NumPy/BLAS runtime. Matching runtime version
numbers alone is insufficient. Version `0.1.1` supports archive inspection and
fresh experiments with their own source and runtime identities. No historical
identity was relaxed or rewritten to make changed sources pass verification.

## Focused regression checks

| Test module | Result | Elapsed |
| --- | --- | ---: |
| `test_cli_boundaries.py` | 4 passed | 2.626 s |
| `test_encoder.py` | 19 passed | 1.261 s |
| `test_ci_checks.py` | 28 passed | 0.032 s |

These **51 focused tests** cover both tutorial entry points, supported exact
decimal/exponent forms, defaults, near-support tokens that floating-point
parsing previously rounded into the support, underflowing tokens, nonfinite or
malformed inputs, controlled encoder-loader errors, unchanged encoder replay,
and separate retention of JSON reports and test logs for `.log` report names.
The log-name regressions exercise both passing and failing synthetic suites
and preserve a pre-existing log before test discovery.

Three additional command checks used the actual installed release wheel in
isolated Python outside the checkout. A supported exact input succeeded; the
near-support decimal `0.25000000000000001` and underflowing token `1e-400`
returned the intended error status without traceback or support rounding.

## Installed wheel and one complete fresh experiment

The exact wheel selected for distribution passed the existing `fresh-wheel`
gate with run ID `release-v0.1.1-check` in **104.192 seconds**. The gate invoked
the fresh pipeline **once**, retained every outcome, and performed no automatic
retry. It used Python **3.12.14** and NumPy **2.3.5**; the complete local runtime
identity is recorded with that new experiment.

The gate built one wheel, installed it into a fresh virtual environment, and
required isolated imports from that installation. All **34 package source and
resource files** match the planned candidate bytes; imports were not shadowed
by the checkout. The environment inherits the host's numerical dependencies,
as described in [the CI guide](../docs/CI.md).

The fresh experiment executed the complete fixed menu: 4,096 training episodes,
1,024 validation episodes, both encoder banks, all 225 development fits, three
selected predictors, and 4,096 calibration episodes. It completed development,
freezing, certification, and population auditing in the declared order. All
three certificate results were `requirement_met` and all three realized audit
comparisons were `covered`. These outcomes are retained observations, not
requirements for passing CI and not an empirical estimate of coverage.

Archive inspection passed both from the project and the installed wheel. It
checked the original evidence identities without claiming numerical replay of
the historical primary experiment. The fresh run verified its own identities;
no runtime or source check was relaxed.

| Release wheel property | Verified value |
| --- | --- |
| Filename | `kahm_demonstrator-0.1.1-py3-none-any.whl` |
| Size | 157,625 bytes |
| SHA-256 | `5ccc0916bad192858f954ce3075537d01d1634132eca98ca44b51dd7b596fad4` |
| Package source/resource matches | 34 |
| Hashed wheel RECORD entries checked | 40 |
| Metadata | Mohit Kumar; version 0.1.1; Apache-2.0 |
| License and NOTICE | Exact repository bytes |

The distributed wheel is the exact successfully tested file; it was not
replaced by a later rebuild. The complete fresh plan, reports, subprocess logs,
wheel, and all fresh experiment files are retained in the separate release
asset `KAHM_v0.1.1_Validation_Evidence.zip`.

## Documentation and notebook checks

Download, extraction, project-directory, and local HTML instructions identify
the complete release ZIP and its `certified-kahm-representations-forecasting/`
root directory. The tutorial prerequisites include `data/primary/`. Introductory
wording distinguishes pseudorandom reproducibility from the stipulated iid
calibration premise. Verification instructions distinguish historical `0.1.0`
replay from fresh `0.1.1` experiments.

Only six notebook Markdown cells changed. Code cells, saved outputs, execution
counts, and metadata are identical to the historical notebook. Those code cells
were not re-executed for this documentation patch. The HTML reading view was
regenerated with Pandoc, parsed, and checked for its one embedded PNG, retained
baseline results, absence of external scripts/stylesheets, and valid relative
file targets. No browser-rendering or live Jupyter-kernel test is claimed.

## Records and publication gate

[Machine-readable package validation](patch_0.1.1_validation.json) records the
checks above and the digests of the local archive-inspection and fresh-wheel
reports. The validation evidence ZIP contains those underlying reports and
complete fresh outcomes. `RELEASE_MANIFEST.json` records the exact release
payloads; the release's `SHA256SUMS.txt` records downloadable asset digests.

This document is a local preparation record. Publication is gated on an observed
successful run of the repository's complete portable suite and installed-wheel
fresh-run workflow for the candidate commit. Actual remote results have their
own [GitHub Actions records](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/actions)
and are linked from the published release notes. No remote result is inferred
from a workflow file or from the local checks recorded here.

# Public release validation — version 0.1.0

The self-contained public edition passed **415 tests**: 412 portable checks and
3 strict checks in the recorded primary runtime. The installed wheel passed
archive inspection and full read-only verification. Both notebook modes executed
all seven code cells successfully in fresh IPython processes.

## One public primary experiment

The local plan fixed seed `20261110`, run ID `public-v0.1.0`, counts
4,096/1,024/4,096, both encoder banks, and all 225 development fits before work.
The driver ran once and completed generation, encoder fitting, readout selection,
model/protocol freezing, certification, and population auditing in that order.
It retained every candidate outcome and performed no retry or outcome filtering.
The elapsed execution time was 103.322 seconds.

| Predictor | Calibration q (approx.) | Population R (approx.) | Issued U (rounded up) | Assessment | Audit |
| --- | ---: | ---: | ---: | --- | --- |
| `raw` | 0.064505720727162463 | 0.056437536124516396 | 0.076071470660035393 | `requirement_met` | `covered` |
| `kahm_mem` | 0.064505721671099938 | 0.056437535407142705 | 0.076071471677418599 | `requirement_met` | `covered` |
| `kahm_local` | 0.064505724551054440 | 0.056437536250430780 | 0.076071474781456121 | `requirement_met` | `covered` |

All three exact endpoints are below the fixed tolerance of `1/10`. Each covers
its predictor's exact finite-benchmark risk in this realized audit. The fitted
risks are nearly equal; these results do not establish general representation
superiority. Exact fractions in the records govern all decisions.

The simultaneous statistical statement is conditional on development and iid
calibration independent of development. Its total failure allocation is `1/20`,
with `1/60` allocated to each fixed predictor. There is no additional
conditioning on numerical availability or acceptance. The local
[guarantee and proof](../docs/FITTED_RISK_GUARANTEE.md) give the precise statement.

## Evidence identities

| Record | SHA-256 |
| --- | --- |
| Scientific contract | `83a9fa471044e967d2055e7b507132749b7e64ab1a2788eac71024d83c52b4f7` |
| Calibration protocol | `26d62b5c737e67f4cdb08250f10eddc8bf1c06c996b034c6bc6e1356f4e19c04` |
| Public run plan | `9c19c91a1f2f78e31c2544d2e331c4337364c3bb9a92cee2f0942c4d985b18e4` |
| Model freeze | `d9de3f2511dd70a2f1a9eaf7fb7fbb772cfa192a4d141ff472d7e032efeed5e5` |
| Certificate | `b7586d0afa78c0f832feb482132499bd923d5bf35cdd53bd5e84f52f6d9c9446` |
| Population audit | `65e1bd54dcbc87994f197b15d6fd0f7e56a0f8eb2ef1cb4b699022d2e02c52eb` |
| Validated wheel | `d34201a61cbbd5de90ae9980c2af85a18ae4a954f1525e83e9ab928787955fc0` |

The primary scientific artifacts remain unchanged after issuance. The current
tutorial presentation module received the completed evidence hashes afterward;
its frozen source snapshot retains the pre-issuance unassigned values. That
module is outside the scientific execution-source roster. Strict verification
checks the required source/runtime identities without relaxing them.

## Verification performed

| Check | Result |
| --- | --- |
| Complete portable gate | 412 passed; 385.108 seconds |
| Strict primary gate | 3 passed; 26.166 seconds |
| Notebook inspection | 7 code cells passed; 0.509 seconds |
| Notebook full verification | 7 code cells passed; 26.923 seconds |
| Installed wheel | 34 source/resource files match; isolated import, metadata, license, RECORD, inspection, and verification passed |
| Citation metadata | Valid CFF 1.2.0; Mohit Kumar, version 0.1.0, Apache-2.0 |
| Standalone-content review | Local method definitions, proof, resources, snapshots, and wheel checked |
| Static HTML | Pandoc export, embedded image, result text, and HTML parsing checked; generated plot visually inspected |

The initial portable run found one stale expected payload count in a tutorial
test: 278 instead of the new freeze's 286. The eight additional payloads are
current package modules included in the new source snapshot. The assertion
was corrected and the **entire** 412-test portable gate was rerun successfully.
The initial report and log remain beside the final report. This correction
changed no model, data, issued endpoint, or scientific execution source.

The notebook cells were executed in fresh IPython processes with guards against
fitting, sampling, creating certificates, and repeating endpoint searches. This
is not a claim that JupyterLab, a browser kernel, or a browser-rendered HTML page
was exercised. Opening the static HTML performs no verification.

The runtime is CPython 3.12.14, NumPy
2.3.5, and x86_64, with the full
Python build and NumPy/BLAS configuration retained in the evidence. Matching
version numbers alone is insufficient for strict replay.

## Records and scope

- [Public plan](public_primary_plan.json) and [execution](public_primary_execution.json).
- [Final portable tests](portable_tests_final.json) and [strict tests](strict_primary_tests.json).
- [Initial portable report](portable_tests.json), retained with its corrected assertion explained above.
- [Installed-wheel validation](installed_wheel_validation.json).
- [Notebook inspection](notebook_inspect.json), [verification](notebook_verify.json), and [HTML checks](notebook_html_validation.json).
- [Machine-readable release validation](release_validation.json).

The package includes the complete Apache-2.0 license and Copyright 2026 Mohit
Kumar notice. Mohit Kumar is the software author. These local checks do not
assert an unobserved remote GitHub CI result; actual remote execution has its
own workflow record. The exact population oracle is specific to this known
finite benchmark and is not required or generally available for a new dataset.

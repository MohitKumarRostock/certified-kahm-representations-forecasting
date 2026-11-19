# Citing the software

**KAHM forecasting demonstrator**, version **0.2.0**, is a standalone
implementation of representation-based forecasting and risk certification.
It supplies lower confidence bounds on optimal class risk, upper confidence
bounds on fitted-predictor risk, admissibility checks, and independent
population auditing under a controlled finite benchmark law.

| Field | Value |
| --- | --- |
| Software author | Mohit Kumar |
| Software title | KAHM forecasting demonstrator |
| Version | `0.2.0` |
| Software licence | Apache-2.0 |
| Repository | <https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting> |
| Software DOI | No DOI is assigned in this repository |

Use [CITATION.cff](CITATION.cff) for machine-readable metadata. A software
reference can be written as:

> Kumar, Mohit. KAHM forecasting demonstrator. Version 0.2.0. GitHub.
> https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting

Add the **exact Git commit** of the copy used, obtained with `git rev-parse HEAD`,
and the access or execution date appropriate to your record. A commit-specific
GitHub URL or a versioned source archive identifies an immutable source snapshot;
the default branch can change. Use an observed tag or release date only when
that tag or release actually exists. Do not invent a software DOI.

The documentation is independent of an external manuscript. A later research
article can reference this repository and its identified software version;
using the implementation does not require that article to be available.

## Identify the scientific evidence

For the retained lower/upper study, cite run
`lower-certificates-20261119-v1` and identify the study plan and bundle digests
from [the manifest](reports/lower_certificates_v1/manifest.json). The
[study report](reports/lower_certificates_v1/report.md) and
[summary](reports/lower_certificates_v1/summary.json) retain all 15 assessments
and the 180-endpoint confidence allocation.

For a new assessment, retain its plan, class definitions, calibration
observations, certificate records, verification summary, and population audit.
State the representation, metric, Lipschitz bound, output interval, target law,
MSE threshold, calibration size, and joint confidence budget. Distinguish the
optimal class risk, fitted-predictor risk, and the issued lower and upper
confidence bounds. Displayed decimals do not replace the exact fractions that
determine the decisions.

State which activity was performed:

- **Archive inspection:** checked archive identities and displayed recorded values.
- **Class-risk verification:** checked retained source identities, numerical
  witnesses, confidence endpoints, admissibility, decisions, and audit records.
- **Strict historical verification:** replayed the original fitted models in
  their required source/runtime environment and checked their original evidence.
- **Fresh experiment:** created separately identified development and/or
  calibration records and retained every outcome.

The original fitted-predictor experiment remains `public-v0.1.0`; version 0.2.0
does not rewrite its models, calibration losses, endpoints, or audit. Strict
replay of that historical experiment requires its original v0.1.0 sources and
recorded runtime. The retained class-risk study evaluates the frozen
finite-support computational tables and has its own independent calibration
protocol and numerical witnesses.

## Research and licence scope

Software authorship does not assert sole authorship of every underlying
mathematical idea. The [scientific reference](docs/SCIENTIFIC_REFERENCE.md),
[self-contained class-risk proof](docs/LOWER_CERTIFICATE_PROOF.md), and
[fitted-risk guarantee](docs/FITTED_RISK_GUARANTEE.md) define the implemented
claims and acknowledge relevant prior work.

The request to cite the software is voluntary. It adds no research-only,
noncommercial, or scholarly-citation condition to [Apache-2.0](LICENSE).
Required licence and attribution notices remain governed by the licence.

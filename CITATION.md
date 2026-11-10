# Citing the software

This project is the **KAHM forecasting demonstrator**, an executable example of
geometry-induced representations, frozen forecasting models, fitted-risk upper
certificates, and exact population auditing under a controlled benchmark law.
Its methods and assumptions are documented within the repository.

| Field | Value |
| --- | --- |
| Software author | Mohit Kumar |
| Copyright holder | Mohit Kumar |
| Software title | KAHM forecasting demonstrator |
| Version | `0.1.1` |
| Software licence | Apache-2.0 |
| Repository | `https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting` |
| Software DOI and release date | Not assigned here |

Use [CITATION.cff](CITATION.cff) for machine-readable metadata. A plain-text
citation can identify the software as:

> Mohit Kumar. KAHM forecasting demonstrator. Version 0.1.1.
> https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting

Add the Git commit or archive SHA-256 of the actual copy used and the access or
execution date appropriate to your record. Do not invent a software DOI or use
a publication date that has not been established.

## Identify the evidence you used

State which activity you performed:

- **Archive inspection:** checked identities and displayed retained results.
- **Strict verification:** replayed the supplied computational models and checked
  their certificates and audit in the required source/runtime environment.
- **Fresh experiment:** ran development, freezing, calibration, certification,
  and auditing under a separately recorded run identity.

For numerical results, identify the run, model freeze, certificate, and audit
records. Report all relevant candidate outcomes and distinguish empirical
calibration MSE `q`, statistical upper endpoint `U`, and exact benchmark risk `R`.
The bundled primary results belong to the unchanged `public-v0.1.0` experiment.
Version `0.1.1` improves the software interface and documentation; it does not
replace those results with a new primary experiment. Strict replay of that
historical experiment requires the original `v0.1.0` source and recorded runtime.
The meaning of those quantities is in
[Certificates explained](docs/CERTIFICATES_EXPLAINED.md).

## Research and authorship scope

This implementation is based on our research. The authorship metadata here
identify the software author; they do not assert sole authorship of every
underlying research contribution. The local [method overview](docs/METHOD_OVERVIEW.md)
and [risk guarantee](docs/FITTED_RISK_GUARANTEE.md) define the construction and
claim made by this software.

The request to cite the software is voluntary. It adds no scholarly-citation,
research-only, or noncommercial condition to [Apache-2.0](LICENSE). Required
licence and attribution notices remain governed by the licence itself.

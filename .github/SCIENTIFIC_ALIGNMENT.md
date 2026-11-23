# Scientific documentation alignment (maintainer checklist)

This is a **maintenance checklist**, not part of the mathematical contract.
The [scientific reference](../docs/SCIENTIFIC_REFERENCE.md), [proof note](../docs/LOWER_CERTIFICATE_PROOF.md), and [user guide](../README.md) remain self-contained; readers do not need the associated manuscript.

## Scope and naming

The manuscript used for the JMLR submission is *A Hierarchy of Risk Certificates for Representation-Based Prediction*. This repository implements a finite-state **forecasting demonstrator** of the more general representation-based prediction framework. Forecasting-specific language is appropriate for the benchmark and APIs; general statements should use the following scientific terms:

| Concept | Preferred wording / notation |
| --- | --- |
| Prespecified candidate | Representation–decoder class, specified by a representation, feature pseudometric, output interval, and admissible decoder class |
| Evaluation target | Optimal population class risk, `R_\kappa^*` |
| Individual predictor | Fitted decoder `\widehat g` and its population risk `R(\widehat g)` |
| Calibration | Simultaneous one-sided population-risk confidence bounds, conditional on development |
| Optimization on fixed observations | Deterministic lower/upper bounds on minimum empirical risk |
| Blocks | Normalized block score `T_b` and its expectation `\mathcal W_b` |

Do not describe a fitted-risk upper endpoint as an upper bound on *optimal class risk* unless class admissibility is established. The MEM fitted predictor in the recorded study has a valid individual-risk upper bound but is inadmissible for its declared Lipschitz class.

## What must be reviewed when a claim changes

1. **Definitions and claims:** synchronize `docs/SCIENTIFIC_REFERENCE.md`, `docs/LOWER_CERTIFICATE_PROOF.md`, and any shorter statements in `README.md` or `docs/CERTIFICATES_EXPLAINED.md`. Preserve the representation, pseudometric, Lipschitz bound, evaluation law, bounded-target premise, and independence assumptions.
2. **Threshold decisions:** keep the strict exclusion rule `L > tau` distinct from class feasibility established by an admissible fitted decoder with `U_fit <= tau`. Preserve `unresolved`, numerical unavailability, and conflict reporting.
3. **Numerical results:** check `README.md` against `reports/lower_certificates_v1/summary.json`. Stored exact fractions determine decisions; rounded display values are descriptive.
4. **Reproducibility:** verify commands, file paths, supported Python versions, and whether each command only rechecks saved results, regenerates predictions, or actually refits models.
5. **Evidence provenance:** keep retained evidence and its manifests immutable. In particular, `RELEASE_MANIFEST.json` is a dated release-preparation snapshot, and `reports/lower_certificates_v1/manifest.json` identifies a separate verified numerical evidence bundle. Neither should be silently rewritten when current narrative documentation changes.
6. **Submission permalink:** after merging a reviewed documentation PR, identify the exact resulting source commit. Update any manuscript source URL that is intended to point to that revision. Do not use a moving branch URL as the scientific archival reference.

## Mechanical checks and limitations

Run `python scripts/check_scientific_docs.py` before committing. The standard-library check validates GitHub-compatible dollar-delimited Markdown mathematics, local link targets, and the principal numerical results table against the recorded exact-fraction summary. GitHub Actions runs the same check, along with the existing software and verification checks.

Mechanical checks cannot validate the proofs, the sampling assumptions, or rendered math visually. Review mathematical statements and any changed GitHub rendering manually. A passing replay does not convert a fixed-dataset empirical-risk statement into a population-risk certificate.

## Separate experimental archive

The experimental records for the other study groups are held in the separate `Kernel_Affine_Hull_Koopman_Machine` repository. Its historical submission source and `MANIFEST.json` are linked to a specific commit. Changes to *any* file covered by that manifest require a **new source revision** and a matching regenerated file hash entry; existing validation reports remain evidence for the original manifest and must not be re-labelled as validation of a new one. Verify the new revision before updating the manuscript's experiment-repository permalink.

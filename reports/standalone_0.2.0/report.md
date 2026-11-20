# Software version 0.2.0: validation

This report records validation of the lower/upper risk-certificate demonstrator,
version 0.2.0. Its numerical package sources match commit
`820c41cda14e5ac574eeac1cc6b19e5bdf11ab7b`. The
[manifest](manifest.json) identifies the distribution wheel and validation
records.

## Scientific documentation and package

- A [standalone scientific reference](../../docs/SCIENTIFIC_REFERENCE.md)
  defining the prediction problem, optimal class risk, fitted-predictor risk,
  Lipschitz decoder class, block hierarchy, confidence constructions, numerical
  verification, and independent population audit.
- [Self-contained proofs](../../docs/LOWER_CERTIFICATE_PROOF.md) with general
  bounded-interval notation and an explicit finite-benchmark specialization.
- Quickstart and operational guides, software citation, source/runtime
  requirements for each verification route, and packaging guidance.
- A [validated 0.2.0 wheel](../../dist/kahm_demonstrator-0.2.0-py3-none-any.whl).
- A GitHub CI gate checking the committed 180-endpoint numerical
  evidence with the installed wheel.

## Local checks performed on 19 November 2026

| Check | Result | Recorded evidence |
| --- | --- | --- |
| Archive inspection | Passed; identity inspection only | [Report](archive-inspection.json) |
| Portable regression suite | 470 tests passed; zero failures/errors/skips among selected tests | [Report](portable-tests.json), [log](portable-tests.log) |
| Fresh isolated-wheel demonstration | Passed; all 225 development fits, freeze, fitted-risk calibration, verification, and population audit completed | [Report](fresh-wheel-check.json) |
| Recorded class-risk study | Verified all 180 endpoints and all 15 assessments; 9 requirement-met, 5 class-excluded, 1 unresolved | [Verification](retained-class-risk-verification.json) |
| Class-risk assessment of freshly fitted models | Verified 120 endpoints at 1,024 and 4,096 episodes; 6 requirement-met, 3 class-excluded, 1 unresolved | [Verification](fresh-class-risk-verification.json) |
| Documented small quickstart | Verified 30 endpoints at 128 episodes; all five class decisions remained unresolved | [Verification](documented-quickstart-verification.json) |
| Documentation and package consistency | Relative links and local anchors resolved; mathematical delimiters balanced; citation version matched; all 38 package Python/JSON files matched source | [Report](documentation-and-package.json) |
| Distribution installation | Imported from the isolated installation; all 38 numerical package files matched the checkout | [Report](final-installed-package.json) |
| Distribution verification command | Rechecked the quickstart evidence using the final installed wheel | [Verification](final-wheel-quickstart-verification.json) |

The full fresh-wheel pipeline and the distribution checks used identical
numerical sources. Distribution checks covered isolated installation,
byte-for-byte equality of all 38 numerical package files, and independent
witness verification. The [manifest](manifest.json) identifies the distribution
and each validation record.

The three tests requiring the unmodified v0.1.0 sources and exact archived
numerical runtime are outside the portable gate. These checks do not establish
replay in that runtime. See [CI.md](../../docs/CI.md)
for the exact exclusions and the separate strict-primary gate.

The fresh runs are software integration checks. They are not additional
repetitions of the prespecified scientific study, and their outcomes
are not pooled to estimate coverage or decision frequencies. All recorded
outcomes, including unresolved decisions, are preserved.

## Scientific evidence and interpretation

The [class-risk study](../lower_certificates_v1/report.md) supplies the scientific
example. Its numerical bundle, development wheel, figures, plan, summary, and
manifest identify the experiment. The scientific documentation states the
numerical and sampling assumptions.

The study demonstrates class-risk decisions, the admissibility distinction,
and a larger-block certificate that resolves a threshold missed by the
pairwise certificate. It does not establish unrestricted information loss,
general forecasting superiority, or a computational advantage over full
empirical optimization. The separate known-law audit checks realized coverage;
a verifier pass does not establish the iid premise or coverage on every sample.

## Remote publication

This report records local checks and their exact scope. The repository's
[commit history](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/commits/main)
and [GitHub Actions](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/actions)
provide the authoritative published source and remote checks. Cite the actual
version and commit used, following [CITATION.md](../../CITATION.md).

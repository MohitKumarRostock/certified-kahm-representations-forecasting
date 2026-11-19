# Standalone version 0.2.0: completion and validation

This edition completes the scientific documentation and publication metadata
for the lower/upper risk-certificate demonstrator. It uses the same numerical
package sources as commit `820c41cda14e5ac574eeac1cc6b19e5bdf11ab7b`; no
numerical calculation, retained calibration observation, class specification,
endpoint, or historical evidence record was changed during documentation
completion.

## What is complete

- A [standalone scientific reference](../../docs/SCIENTIFIC_REFERENCE.md)
  defining the prediction problem, optimal class risk, fitted-predictor risk,
  Lipschitz decoder class, block hierarchy, confidence constructions, numerical
  verification, and independent population audit.
- [Self-contained proofs](../../docs/LOWER_CERTIFICATE_PROOF.md) with general
  bounded-interval notation and an explicit finite-benchmark specialization.
- Updated quickstart, operational guides, terminology, software citation,
  current-versus-historical verification instructions, and packaging guidance.
- A [validated 0.2.0 wheel](../../dist/kahm_demonstrator-0.2.0-py3-none-any.whl).
- An additional GitHub CI gate checking the committed 180-endpoint numerical
  evidence with the installed wheel.

## Local checks performed on 19 November 2026

| Check | Result | Retained evidence |
| --- | --- | --- |
| Historical archive inspection | Passed; identity inspection only | [Report](archive-inspection.json) |
| Portable regression suite | 470 tests passed; zero failures/errors/skips among selected tests | [Report](portable-tests.json), [log](portable-tests.log) |
| Fresh isolated-wheel demonstration | Passed; all 225 development fits, freeze, fitted-risk calibration, verification, and population audit completed | [Report](fresh-wheel-check.json) |
| Retained class-risk study | Verified all 180 endpoints and all 15 assessments; 9 requirement-met, 5 class-excluded, 1 unresolved | [Verification](retained-class-risk-verification.json) |
| Class-risk assessment of freshly fitted models | Verified 120 endpoints at 1,024 and 4,096 episodes; 6 requirement-met, 3 class-excluded, 1 unresolved | [Verification](fresh-class-risk-verification.json) |
| Documented small quickstart | Verified 30 endpoints at 128 episodes; all five class decisions remained unresolved | [Verification](documented-quickstart-verification.json) |
| Documentation and package consistency | Relative links and local anchors resolved; mathematical delimiters balanced; citation version matched; all 38 package Python/JSON files matched source | [Report](documentation-and-package.json) |
| Final distribution installation | Imported from the isolated installation; all 38 numerical package files matched the checkout | [Report](final-installed-package.json) |
| Final distribution verification command | Rechecked the quickstart evidence using the final installed wheel | [Verification](final-wheel-quickstart-verification.json) |

The full fresh-wheel pipeline preceded three final wording corrections. The
final wheel was rebuilt to include the corrected README metadata, reinstalled
in the isolated environment, checked byte-for-byte against all 38 numerical
package files, and used for independent witness verification. The numerical
sources were identical throughout. The [manifest](manifest.json) identifies
the final distribution and each retained validation record.

The three tests requiring the original v0.1.0 sources and exact archived
numerical runtime remain explicitly outside the portable gate. This edition
does not claim to replay that historical runtime. See [CI.md](../../docs/CI.md)
for the exact exclusions and the separate strict-primary gate.

The fresh runs are software integration checks. They are not additional
repetitions of the prespecified retained scientific study, and their outcomes
are not pooled to estimate coverage or decision frequencies. All recorded
outcomes, including unresolved decisions, are preserved.

## Scientific evidence and interpretation

The [retained study](../lower_certificates_v1/report.md) remains the scientific
example. Its original numerical bundle, development wheel, figures, plan,
summary, and manifest are unchanged. The completed documentation uses the
same concepts and explicitly states the numerical and sampling assumptions.

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

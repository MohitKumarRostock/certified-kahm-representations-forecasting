# Release and reproducibility guide

Version **0.2.0** is the standalone class-risk edition of the KAHM forecasting
demonstrator. Its scientific definitions, proofs, implementation, benchmark,
recorded experiments, and reproduction instructions are contained in this
repository. The [scientific reference](SCIENTIFIC_REFERENCE.md) is the main
entry point; [CITATION.md](../CITATION.md) explains how to identify the software
and the particular experiment used.

## Software identity

| Field | Value |
| --- | --- |
| Repository | `MohitKumarRostock/certified-kahm-representations-forecasting` |
| Package | `kahm-demonstrator` |
| Version | `0.2.0` |
| Software author | Mohit Kumar |
| Copyright | Copyright 2026 Mohit Kumar |
| License | [Apache-2.0](../LICENSE) |
| Citation metadata | [CITATION.cff](../CITATION.cff) |

GitHub commit, tag, release, and workflow records establish the actual remote
publication and CI state. `PUBLICATION_STATUS.json` describes the prepared
software and validation scope; it is not a substitute for those remote records.
No software DOI or author identifier is invented.

## Source, package, and numerical evidence

| Artifact | Contents and use |
| --- | --- |
| Complete Git checkout or GitHub source archive | Package source, tests, documentation, notebooks, figures, and both retained experiments |
| [Version 0.2.0 wheel](../dist/kahm_demonstrator-0.2.0-py3-none-any.whl) | Validated installable package; the complete project is still needed for the supplied experiment files |
| `reports/lower_certificates_v1/full_verification_bundle.zip` | Retained class-risk run, calibration data, class definitions, witnesses, source snapshot, and population audits; not a full source release |
| `reports/lower_certificates_v1/manifest.json` | SHA-256 identities of the retained study plan, summary, verification ZIP, and original development wheel |
| Historical `RELEASE_MANIFEST.json` | Inventory of the published v0.1.1 snapshot; apply it only to that historical release |

The package already identified itself as 0.2.0 when the retained class-risk
study was created. Completing the documentation changes the wheel's descriptive
metadata when it is rebuilt, but does not change the numerical package sources.
The original development wheel remains in the study directory with its
original manifest digest. A newly built distribution must retain its own
digest; do not overwrite the historical wheel or its manifest to make them
agree with a later build.

## Two retained scientific records

The original fitted-predictor experiment is `public-v0.1.0`. It retains all 225
development fits, selected models, calibration losses, upper certificates, and
population audits. Later versions preserve this evidence. Archive inspection
remains available, while strict replay requires the complete unmodified
[v0.1.0 source](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.0)
and its exact recorded Python/NumPy/BLAS runtime.

The separate class-risk study is `lower-certificates-20261119-v1`. It treats the
archived finite-support feature and prediction tables as its exact computational
objects, declares its own calibration stream, and retains five classes, three
sample sizes, five block sizes, and all 180 confidence endpoints. Its numerical
witnesses can be checked using the current matching numerical sources without
replaying the historical encoder. The [study report](../reports/lower_certificates_v1/report.md)
and [class-risk guide](LOWER_CERTIFICATES.md) explain the scope.

A saved local plan and editable hashes do not constitute independent public
preregistration. Pseudorandom streams provide reproducibility; the statistical
premise remains iid calibration episodes independent of development.

## Preserve source and evidence identities

Do not edit retained data, predictions, witnesses, endpoints, or source
snapshots to make a changed implementation pass an old verification. A change
to a source or resource included in a run's identity requires a separately
identified assessment under the changed implementation. Even a cosmetic edit
to a source file can change its hash. Documentation outside that roster can be
updated while the numerical study remains unchanged.

The tutorial presentation anchors are assigned after the primary evidence is
issued. `tutorial.py` is outside that original execution-source roster, so a
freeze need not contain its own digest. Keep this distinction explicit; never
rewrite an issued scientific artifact to adjust a presentation anchor.

## Validation before publication

The [standalone 0.2.0 validation report](../reports/standalone_0.2.0/report.md)
records the checks actually completed for this edition.

From a complete checkout with Python 3.12+, install the declared gate
dependencies and use fresh output paths outside the checkout:

```bash
python -m pip install 'numpy==2.3.5' 'setuptools>=77.0.3' wheel
python scripts/ci_checks.py archive-inspect --project . --report ../kahm-release-checks/archive-inspection.json
python scripts/ci_checks.py portable --project . --report ../kahm-release-checks/portable-tests.json
python scripts/ci_checks.py fresh-wheel --project . --output ../kahm-release-checks/fresh-wheel --run-id release-check-v0.2
```

Then verify the retained study with the installed wheel:

```bash
python -m zipfile -e reports/lower_certificates_v1/full_verification_bundle.zip ../kahm-release-checks/retained-class-risk
../kahm-release-checks/fresh-wheel/venv/bin/python -I -B -m kahm_demo verify-class-risk --run ../kahm-release-checks/retained-class-risk/run
```

Finally test the path from newly fitted models to class-risk assessment:

```bash
../kahm-release-checks/fresh-wheel/venv/bin/python -I -B -m kahm_demo class-risk --freeze ../kahm-release-checks/fresh-wheel/run/frozen/freeze.json --output ../kahm-release-checks/new-class-risk --sample-sizes 1024 4096 --block-sizes 2 8 16 32 64 --solver-sweeps 64 --run-id release-class-check-v0.2
../kahm-release-checks/fresh-wheel/venv/bin/python -I -B -m kahm_demo verify-class-risk --run ../kahm-release-checks/new-class-risk
```

These interpreter paths are for POSIX environments; on Windows the virtual
environment interpreter is `venv/Scripts/python.exe`. The fresh-wheel gate
builds an isolated package installation and executes one full fitted-predictor
pipeline. It does not itself invoke `class-risk`; the final commands above and
the GitHub workflow perform that separate integration check.

Retain all outcomes. Requirement satisfaction, class exclusion, unresolved
results, and realized coverage misses are scientific results, not criteria for
selecting which software checks to report. A failed calculation or numerical
verification fails the gate. [CI.md](CI.md) documents the exact portable and
historical-runtime boundaries.

For documentation changes, also check relative links, mathematical notation,
citation metadata, and that every stated command matches the implemented CLI.
Do not add numerical experiments solely to validate prose.

## Publish and cite an identified snapshot

Publish the completed source to the confirmed GitHub repository after the
applicable gates pass. Identify the resulting commit and, if created, its tag
or release. Check the remote CI result for the exact published source. A
prepared archive, local test result, or configured workflow does not establish
a successful remote run.

Build distributable archives from that identified source, exclude temporary
runs and local environments, and retain their SHA-256 digests separately.
Preserve the Apache-2.0 licence, notice, and applicable third-party attribution.
Use the actual version and commit in scientific references. The implementation
and its documentation remain independently usable when cited by another work.

# KAHM forecasting demonstrator

**Version `0.1.1` · Apache-2.0 · Standalone methods and reproducible assessment**

Based on our research, this project demonstrates forecasting with
**geometry-induced soft states** and checkable upper bounds on the fitted
models' population mean squared error. It connects an observed history to
Gaussian Kernel Affine Hull Machine (KAHM) features, a fitted nonlinear readout,
and a retained record of development, calibration, certification, and auditing.
The method, assumptions, and numerical rules are documented in this repository.

[Quick start](docs/QUICKSTART.md) ·
[Guided notebook](notebooks/01_kahm_certified_forecasting.ipynb) ·
[HTML walkthrough](notebooks/01_kahm_certified_forecasting.html) ·
[Certificates explained](docs/CERTIFICATES_EXPLAINED.md)

## What the example does

Fit the geometry and readout on training data, select using validation data,
freeze the complete computational predictors, and assess reserved calibration
observations. A separate exact population-risk audit then checks their issued
bounds under a known finite probability law.

| Predictor | Representation supplied to the readout |
| --- | --- |
| RAW | Training-standardized observed history |
| KAHM MEM | Relative local reconstruction affinities, normalized within each partition |
| KAHM LOCAL | The identical memberships plus membership-weighted local coordinates of the input |

MEM and LOCAL use the **same fitted Gaussian KAHM bank**. Memberships describe
local geometry; they are not calibrated probabilities of latent regimes. LOCAL
preserves MEM and appends `phi[j] * P[j].T @ (z-mu[j])`. Each branch uses the same
Gaussian Nyström and ridge readout procedure, followed by clipping to `[0,1]`.
Feature widths and total search effort differ; no equal-capacity or general
KAHM-superiority claim is made.

The [benchmark](docs/BENCHMARK.md) predicts the next scalar observation from two
past observations. Its four values are `0`, `0.25`, `0.75`, and `1`. Statistical
calibration assumes a stipulated iid episode model, independent of development;
the pseudorandom generator provides reproducible samples, not proof of that
premise. The known finite law also permits exact population-risk auditing.

## Start with the supplied experiment

Use Python **3.12 or newer**. Download
[Certified_KAHM_Representations_Forecasting_v0.1.1.zip](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/download/v0.1.1/Certified_KAHM_Representations_Forecasting_v0.1.1.zip)
from the [v0.1.1 release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.1).
Place the ZIP in a fresh directory and open a terminal there, then extract and
enter the project:

```bash
python -m zipfile -e Certified_KAHM_Representations_Forecasting_v0.1.1.zip .
cd certified-kahm-representations-forecasting
```

The complete ZIP contains both code and experiment evidence; the wheel contains code only.
From the extracted project directory:

```bash
python -m pip install .
python -m kahm_demo tutorial --project . --mode inspect --history 0.25 0.75
```

The history order is **previous observation, current observation**. Inspection
checks fixed archive hashes and bindings, then displays recorded features,
forecasts, and all three assessment outcomes. Its scope is explicit:

```json
{
  "mode": "archive_identity_only",
  "archive_identity_checked": true,
  "model_replayed": false,
  "certificate_reverified": false,
  "population_risk_recomputed": false
}
```

This is an excerpt of the expected output. Archive identity alone does not
establish current-runtime model replay or mathematical certificate verification.

For the notebook:

```bash
python -m pip install '.[tutorial]'
python -m jupyterlab notebooks/01_kahm_certified_forecasting.ipynb
```

The notebook defaults to inspection. **Run All does not fit models or sample
calibration data.** For the static reading copy, open the extracted
`notebooks/01_kahm_certified_forecasting.html` file in your browser. Keep it in
the complete project so its relative documentation links work. Opening the HTML
performs no verification.

## Three ways to use the project

| Path | Work performed | When to use it |
| --- | --- | --- |
| **Inspect** | Check archive identities and display recorded values | Learn the example without matching its original numerical runtime |
| **Verify historical baseline** | Replay saved models, recompute exact losses, check issued endpoints, and reproduce the population audit | Use unmodified v0.1.0 sources and the exact recorded runtime |
| **Fresh experiment** | Generate data, fit all 225 configurations, select, freeze, calibrate, certify, and audit | Use v0.1.1 in your own supported runtime, with its own records |

### Verify the supplied experiment

The supplied baseline remains `public-v0.1.0`. Version 0.1.1 fixes encoder
artifact input validation, so its execution-source bytes differ from that
historical freeze. Inspection remains available in v0.1.1; strict verification
of the baseline requires the unmodified
[v0.1.0 release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.0)
**and the exact recorded runtime**. From that v0.1.0 project directory:

```bash
python -m kahm_demo tutorial --project . --mode verify --history 0.25 0.75
```

Strict verification refuses source or runtime mismatches. The identity includes
the full Python build string, machine architecture, NumPy/BLAS configuration,
and execution-source bytes. **Version pins alone are insufficient.** Verification
neither refits models nor samples replacement observations or searches for new
endpoints. An incompatible environment needs its own fresh experiment.

### Run a fresh experiment

Use the installed v0.1.1 package from the v0.1.1 project directory. This creates
new evidence for the current sources and runtime:

```bash
python -m kahm_demo reproduce --output runs/tutorial --master-seed 20261110 --run-id tutorial-v0.1
```

The output directory must be new. Defaults are 4,096 training, 1,024 validation,
and 4,096 calibration episodes. A completed run writes `reproduction.json` and
the complete data, models, freeze, certificate, and audit. Pipeline completion
is distinct from meeting the risk tolerance. All candidate outcomes and partial
failures remain recorded, without automatic overwrite, resume, or retry.
Reusing a stream identity provides reproducibility, not new independent evidence.

## Public baseline

The public primary experiment ran once with master seed `20261110`, run ID
`public-v0.1.0`, and 4,096/1,024/4,096 training, validation, and calibration
episodes. It retained all 225 development fits and all three final outcomes.

| Predictor | Calibration MSE q (approx.) | Population MSE R (approx.) | Issued upper U (rounded up) | Assessment | Audit |
| --- | ---: | ---: | ---: | --- | --- |
| RAW | 0.064505721 | 0.056437536 | 0.076071471 | `requirement_met` | `covered` |
| KAHM MEM | 0.064505722 | 0.056437535 | 0.076071472 | `requirement_met` | `covered` |
| KAHM LOCAL | 0.064505725 | 0.056437536 | 0.076071475 | `requirement_met` | `covered` |

This one experiment meets the fixed MSE tolerance of `0.1` for all three
predictors under the stated assumptions, and the exact benchmark risks lie below all three
issued endpoints. The nearly equal fitted risks do not establish a general
advantage for either KAHM representation. See
[the validation report](reports/RELEASE_VALIDATION.md) for identities and checks.

For each predictor, `q` denotes its calibration sample mean, `U` its verified
statistical upper endpoint, and `R` its exact benchmark population MSE. Exact
fractions govern comparisons; displayed decimals have stated rounding rules.

Conditional on development, under the **declared iid calibration law independent
of development**, the probability that any reported valid upper endpoint misses
its fitted risk is at most `1/20`. Each of the three predictors receives allocation
`1/60`. There is no extra conditioning on acceptance or numerical availability,
and an unavailable candidate's allocation is not reassigned. The fixed MSE
tolerance is `1/10`. This is a population mean-error bound, not an interval for
an individual forecast. The [local proof](docs/FITTED_RISK_GUARANTEE.md) states
and derives the guarantee; stream labels alone do not prove its iid premise.

## What a complete run retains

| Location | Contents |
| --- | --- |
| `models/primary/`, `encoders/primary/` | Both geometry banks, all 225 fitted development states, and selection records |
| `frozen/primary/` | Selected states, canonical predictions, full development snapshot, and source/runtime identities |
| `certificates/primary/`, `data/primary/calibration/` | Every calibration episode, exact loss ledger, endpoints, statuses, and verification evidence |
| `audits/primary/` | All 64 weighted cases per predictor and every coverage comparison |
| `kahm_demo/`, `tests/`, `examples/` | Implementation, regression tests, and executable walkthroughs |

These are the release baseline paths. Fresh runs place the same workflow layers
under their requested output directory. Inspection, strict verification, and
new execution are separate activities and are labeled accordingly.

## Scope and documentation

The known law permits exact `R` here. General applications usually lack that
oracle. The package has **no generic CSV or time-series adapter**; a new task
needs its own data interface, evaluation design, and justified certificate.
Overlapping trajectory windows are not automatically iid. The example makes
no claim of representation optimality, predictive sufficiency, Koopman closure,
certified latent rollout, or equivalence to an ideal real-arithmetic KAHM map.

| Read next | Purpose |
| --- | --- |
| [Benchmark](docs/BENCHMARK.md) and [method overview](docs/METHOD_OVERVIEW.md) | Complete task, data roles, and implementation map |
| [Encoder](docs/ENCODER_DESIGN.md) and [readout](docs/READOUT_DESIGN.md) | Formulas, numerical conventions, and development selection |
| [Risk guarantee](docs/FITTED_RISK_GUARANTEE.md) and [certification](docs/CERTIFICATION_DESIGN.md) | Statistical proof, exact loss arithmetic, and verified endpoints |
| [Freeze interface](docs/FREEZE_INTERFACE.md) and [population audit](docs/POPULATION_AUDIT.md) | Provenance gates and exact reference checks |
| [Full tutorial](docs/TUTORIAL.md) and [technical overview](docs/TECHNICAL_OVERVIEW.md) | Walkthrough, APIs, and artifact lifecycle |
| [Adapting to new data](docs/ADAPTING_TO_NEW_DATA.md) | Application design and required module changes |
| [CI guide](docs/CI.md) | Portable checks and the separate strict runtime gate |

## Software and release information

Software author: **Mohit Kumar**. Copyright **2026 Mohit Kumar**.
Licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE),
[CITATION.md](CITATION.md), and [CITATION.cff](CITATION.cff).

Repository:
[MohitKumarRostock/certified-kahm-representations-forecasting](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting).
See the [release guide](docs/RELEASE_GUIDE.md) for artifact preservation and
validation, and the [contribution guide](CONTRIBUTING.md) to propose improvements.

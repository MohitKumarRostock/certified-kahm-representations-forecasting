# Technical overview: workflow and artifact interfaces

Version `0.1.0` supplies an executable controlled benchmark from training data
to frozen forecasts, calibration losses, checked risk bounds, and an exact
population audit. The [method overview](METHOD_OVERVIEW.md) gives the conceptual
map; this guide describes the lifecycle, interfaces, and retained artifacts.
Baseline availability and realized results are recorded in the
[README](../README.md#public-baseline).

## 1. Read or recompute an existing experiment

From the complete project directory, using Python 3.12 or newer:

```bash
python -m pip install .
python -m kahm_demo tutorial --project . --mode inspect --history 0.25 0.75
python -m examples.guided_tutorial --project . --mode inspect --history 0.25 0.75
```

The tutorial's primary paths are `data/primary`, `frozen/primary`,
`certificates/primary`, and `audits/primary`. Inspection checks fixed archive
identities and displays recorded results; it does not replay the numerical
models or reverify their endpoint inequalities.

For explicit numerical verification:

```bash
python -m kahm_demo tutorial --project . --mode verify --history 0.25 0.75
```

This invokes the strict verification chain, including model replay, exact loss
recomputation, direct endpoint checks, and population-risk recomputation. It does
not refit, resample, or search for replacement endpoints. Source/runtime mismatch
stops the operation rather than silently reducing its scope.

Runtime identity includes the full Python build string, machine information,
NumPy version, and NumPy/BLAS configuration, as well as numerical source bytes.
Package-version pins alone do not establish that identity. A different runtime
can create its own complete development run and assessment.

The [notebook](../notebooks/01_kahm_certified_forecasting.ipynb) defaults to
inspection; its Run All does not fit models or generate calibration data. Its
[HTML copy](../notebooks/01_kahm_certified_forecasting.html) is a static reading
artifact. See [TUTORIAL.md](TUTORIAL.md) for the guided interpretation.

## 2. Development produces three selected predictors

The [benchmark](BENCHMARK.md) supplies histories of width two and scalar targets
in `[0,1]`. Training inputs fit history standardization, reference partitions,
local PCA, and Gaussian KAHMs. MEM contains partition-normalized memberships;
LOCAL appends `phi[j] * P[j].T @ (z-mu[j])` to the identical membership prefix.
The two candidates share one selected bank. RAW uses the standardized history.

Readout construction uses training-only column scaling, common raw landmark
identities, a Gaussian Nyström block, another training-fitted scaler for that
block, and a retained linear component. Ridge minimizes mean squared training
error plus `alpha * ||w||²` with an unpenalized intercept. Its centered SVD
solution uses `n_train * alpha`. Forecasts are finite-checked, then clipped.

The search contains five bandwidths and nine penalties: 45 RAW fits and 90
fits for each KAHM representation across two banks. Within each bank, MEM and
LOCAL decoders are selected separately; their common bank minimizes the mean
of those two best validation losses. RAW is selected independently. Exact
validation arithmetic and fixed tie ordering determine selection. There is
no training-plus-validation refit and no oracle-risk input to selection.

All 225 complete candidate states are retained. Their status is
`development_candidate`; the three selected state files carry
`development_selected_not_frozen`. Those labels describe their development
origin. A later verified outer freeze record is the authority for frozen status,
without rewriting the selected state bytes.

The [encoder](ENCODER_DESIGN.md) and [readout](READOUT_DESIGN.md) guides define
all formulas, thresholds, exceptional-input behavior, and selection rules.

## 3. Freeze the actual computational predictor

A freeze binds the selected models, complete development snapshot, all 16
canonical predictions per model, source/runtime identities, calibration
reservation, and numerical protocol. Its preflight replays the saved candidate
states, checks fitted-equation consistency and development losses, repeats the
selection calculation, and verifies the selected models' identities.

Canonical inference evaluates one history at a time. The final affine readout
uses `math.fsum` of the individually rounded binary64 products and intercept;
every product and result must be finite before clipping. Batch inference reuses
the same row calculation and may cache identical histories while retaining order
and multiplicity. Loading stored arrays does not rerun model fitting.

Use the frozen interface once the baseline artifacts are present:

```python
from pathlib import Path
from kahm_demo.freeze import load_frozen_forecaster

model = load_frozen_forecaster(Path("frozen/primary/freeze.json"), "kahm_local")
history = [0.25, 0.75]
print(model.predict_one(history))
print(model.inspect_one(history))
print(model.predict([history, history]))
print(model.freeze_record_sha256)
```

Inference-only commands check the predictor's freeze, not the separate risk
certificate. The data APIs require full frozen-model verification before
calibration materialization or loading; they also reject calibration output
inside the sealed snapshot. [FREEZE_INTERFACE.md](FREEZE_INTERFACE.md) defines
these gates and their order.

## 4. Assess reserved calibration observations

The protocol fixes 4,096 calibration episodes, three candidates, allocation
`1/60` each, total failure probability `1/20`, and exact MSE tolerance `1/10`.
The assessment plan and certifier-source snapshot are committed before outcome
access. A missing reserved calibration stream is materialized once; an existing
matching stream is reused as the same evidence.

Every actual binary64 forecast is interpreted as its exact represented rational
value. The ledger retains all 4,096 episode occurrences and all three squared
losses per episode. It uses exact targets and rational subtraction, squaring,
summation, and division. No extra clipping or variance normalization enters
assessment.

Conditional on development, iid calibration independent of development gives
probability at most `1/20` that any reported valid endpoint misses its fitted
risk. Numerical availability and acceptance are not additional conditioning
events, and an unavailable candidate's allocation is not reassigned. The
[local proof](FITTED_RISK_GUARANTEE.md) derives the bounded-loss argument and
simultaneous statement.

The endpoint is an upper enclosure of the upper binary-KL inverse. Exact
integer/rational arithmetic and directed logarithm enclosures retain its
numerical direction. A direct checker verifies the final endpoint without
running the root search. Exact fractions determine status; displayed `U`
decimals round upward. The [certification guide](CERTIFICATION_DESIGN.md)
records the complete procedure and work limits.

| Status | Meaning |
| --- | --- |
| `requirement_met` | A valid exact upper endpoint is at most `1/10` |
| `unresolved` | A valid endpoint exceeds `1/10`; this does not establish excessive actual risk |
| `certificate_unavailable` | The required endpoint calculation or verification was not established |

The certificate-verification interface is:

```python
from pathlib import Path
from kahm_demo.certificate_workflow import verify_certificate_bundle

check = verify_certificate_bundle(
    Path("certificates/primary/certificate.json"),
    dataset_bundle=Path("data/primary"),
    freeze_path=Path("frozen/primary/freeze.json"),
)
print(check["candidates"])
```

## 5. Audit the exact benchmark risk afterwards

The audit first verifies the existing certificate. It binds the same frozen
prediction tables, commits an audit plan/source snapshot, and calculates exact
risk over all 64 weighted history-target cases. The independent enumerator and
separate law-helper path must agree. The conditional-mean decomposition provides
an additional exact check.

The report retains the original `q`, issued `U`, and certificate status. It adds
exact population risk `R`, fitted excess over the Bayes risk, exact compliance
with `1/10`, and `covered`, `coverage_miss`, or `endpoint_unavailable`. A coverage
miss is retained without endpoint repair, resampling, or selection of a new winner.

This checks one realized family. It does not estimate long-run coverage
frequency or improve the original confidence level. A calibration sample mean
may be below the Bayes population risk. Model-specific oracle risks never enter
development or certificate construction. See [POPULATION_AUDIT.md](POPULATION_AUDIT.md).

## 6. Run the complete procedure in a fresh directory

```bash
python -m kahm_demo reproduce --output runs/tutorial --master-seed 20261110 --run-id tutorial-v0.1
```

The runner commits a reproduction plan before work and writes
`reproduction.json` after all stages complete. Completion, certificate acceptance,
and realized coverage are separate outcomes. Unresolved/unavailable endpoints,
coverage misses, and partial failure evidence remain retained. There is no
automatic overwrite, resume, or retry. Reusing a seed/run identity replays a
stream; it is not additional independent evidence.

For explicit stage control, use fresh output directories:

```bash
python -m kahm_demo generate --output data/replay --master-seed 20261110 --run-id tutorial-v0.1
python -m kahm_demo verify --bundle data/replay
python -m kahm_demo fit-encoders --bundle data/replay --output encoders/replay
python -m kahm_demo verify-encoders --encoders encoders/replay --bundle data/replay
python -m kahm_demo fit-forecasters --bundle data/replay --encoders encoders/replay --output models/replay
python -m kahm_demo verify-forecasters --models models/replay --bundle data/replay --encoders encoders/replay
python -m kahm_demo freeze-forecasters --bundle data/replay --encoders encoders/replay --models models/replay --output frozen/replay
python -m kahm_demo certify --bundle data/replay --freeze frozen/replay/freeze.json --output certificates/replay
python -m kahm_demo verify-certificate --certificate certificates/replay/certificate.json --bundle data/replay --freeze frozen/replay/freeze.json
python -m kahm_demo audit-certificate --certificate certificates/replay/certificate.json --bundle data/replay --freeze frozen/replay/freeze.json --output audits/replay
python -m kahm_demo verify-audit --audit audits/replay/audit.json --certificate certificates/replay/certificate.json --bundle data/replay --freeze frozen/replay/freeze.json
```

An already materialized calibration bundle cannot supply development for a new
freeze. Use the existing verified freeze, or develop in a fresh bundle before
its calibration is materialized. Do not edit archived source/runtime identities
to make an incompatible snapshot pass.

## 7. Artifact map and verification limits

A complete release baseline uses the following paths; fresh runs place the same
layers under their chosen output directory.

| Path | Retained evidence |
| --- | --- |
| `encoders/primary/`, `models/primary/` | Both banks, all 225 fitted candidate states, and complete selection records |
| `frozen/primary/freeze.json` | Authoritative model/protocol freeze and inventory |
| `frozen/primary/models/`, `frozen/primary/predictions/` | Selected state bytes and all 16 canonical predictions per candidate |
| `frozen/primary/development/`, `frozen/primary/sources/` | Full development snapshot and prediction-source identities |
| `data/primary/calibration/` | Reserved observations and freeze-bound materialization receipt |
| `certificates/primary/assessment_plan.json`, `sources/` | Assessment identities and committed certifier implementation |
| `certificates/primary/loss_ledger.jsonl`, `bounds/` | Every calibration occurrence, exact losses, endpoints, traces, and work counters |
| `certificates/primary/certificate.json` | All family statuses, exact endpoints, scope, and inventory |
| `audits/primary/audit_plan.json`, `sources/`, `population/` | Audit identities, implementation, weighted cases, and decompositions |
| `audits/primary/audit.json` | Exact risks, unchanged certificate statuses, and every coverage comparison |

Verification recomputes numerical and interpretation fields rather than merely
trusting editable hashes. Hashes, replay, and fitted-equation checks establish
consistency; they do not authenticate historical chronology or make writable
directories tamper-proof. Repeating verification supplies no new independent
statistical sample.

The certificate concerns the actual frozen computational predictor. It provides
no per-forecast interval, class optimum, input-recovery theorem, predictive
sufficiency, Koopman closure, or rollout guarantee. Broader applications need
new data and justified evaluation assumptions, as described in
[ADAPTING_TO_NEW_DATA.md](ADAPTING_TO_NEW_DATA.md).

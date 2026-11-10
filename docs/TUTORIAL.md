# Guided tutorial: KAHM forecasting and fitted-risk certificates

Start with the supplied results, inspect what each representation contains, and
then decide whether to verify the historical experiment in its original
source/runtime environment or build a fresh v0.1.1 experiment. The notebook
defaults to **archive inspection**. Its default cells read existing evidence;
they do not fit models, sample data, or issue new certificates.

Use the complete project directory, containing `kahm_demo/`, `data/primary/`,
`frozen/primary/`, `certificates/primary/`, and `audits/primary/`. Installing the
package alone does not supply those archived experiment directories. The
scientific definitions remain in [the fixed contract](../kahm_demo/specification.json).

The `tutorial` helper recognizes the supplied primary archive using fixed
artifact identities. Its `--project` argument points to this complete project,
not to an arbitrary replacement experiment.

## 1. Choose a route

| Route | What it does | What its success establishes |
| --- | --- | --- |
| `inspect`, the default | Checks archive hashes and record bindings; displays saved features, forecasts, and assessment values | Archive identity and consistency of the displayed records. It makes no current-runtime model replay or mathematical certificate-verification claim. |
| `verify` | With unmodified v0.1.0 sources and the exact recorded runtime, replays the historical models, checks exact calibration losses and issued endpoints, and recomputes the population audit | The verification checks pass for the historical baseline, under the certificate's declared statistical assumptions. |
| `reproduce`, a separate command | Using v0.1.1, generates a fresh development experiment, fits all 225 settings, freezes the selected family, calibrates it, and audits it | A new complete experiment with its own source/runtime identities and outcomes. Its acceptance results need not match the supplied run. |

Archive inspection is useful on a different machine. A hash match establishes
which recorded bytes were read; it does not execute the archived predictor or
independently establish the mathematics of an endpoint. Editable hashes also do
not authenticate historical chronology. The stricter route preserves the
existing source and numerical-runtime requirements rather than weakening them.

## 2. Open the notebook

Use Python 3.12 or newer. Download the complete
[Certified_KAHM_Representations_Forecasting_v0.1.1.zip](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/download/v0.1.1/Certified_KAHM_Representations_Forecasting_v0.1.1.zip)
from the [v0.1.1 release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.1).
Place the ZIP in a fresh directory and open a terminal there, then extract and
enter the project:

```bash
python -m zipfile -e Certified_KAHM_Representations_Forecasting_v0.1.1.zip .
cd certified-kahm-representations-forecasting
```

From this root, create an isolated environment and install the tutorial dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[tutorial]'
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. In JupyterLab,
open [01_kahm_certified_forecasting.ipynb](../notebooks/01_kahm_certified_forecasting.ipynb)
and choose the Python kernel from this environment. Run the cells in order with
`MODE = "inspect"`. Start with the history `[0.25, 0.75]`; its entries mean
**previous observed value, current observed value**. The archive covers the 16
ordered histories whose entries belong to `{0, 0.25, 0.75, 1}`.

The notebook presents the benchmark, representation comparison, one archived
forecast, the three risk quantities, and the retained assessment outcomes. Its
fresh-run command is deliberately separate from cell execution. Switching
`MODE` to `"verify"` requests the strict checks described below. For the supplied
baseline, those checks require the unmodified v0.1.0 release and its exact
recorded runtime; v0.1.1 has different execution-source bytes. Verification
does not retrain or generate replacement calibration data.

The [JupyterLab installation guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) and
[JupyterLab startup guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html)
cover the notebook application. A supplied notebook or HTML export does not, by
itself, demonstrate that a browser kernel executed the notebook on your machine.

### Without Jupyter

Install the ordinary package and run the same inspection from a terminal:

```bash
python -m pip install .
python -m kahm_demo tutorial --project . --mode inspect --history 0.25 0.75
```

The corresponding script is:

```bash
python -m examples.guided_tutorial --project . --mode inspect --history 0.25 0.75
```

For [the HTML reading view](../notebooks/01_kahm_certified_forecasting.html), open
the extracted `notebooks/01_kahm_certified_forecasting.html` file in your browser.
Keep it in the complete project so its relative documentation links work.
This supplies the guided explanation and saved output without a notebook
server. Opening it performs no model or certificate verification. The terminal
commands perform the selected checks when invoked.

## 3. Understand the forecasting problem

The benchmark has four observed values: `0`, `1/4`, `3/4`, and `1`. Each episode
starts with a fresh uniformly drawn ordered pair of states `(a,b)`. The next
state equals `(a+b) mod 4` with probability `17/20`; each other state has
probability `1/20`. The model receives the two observed values, not their hidden
state labels, and forecasts the next observed value. This gives 16 input
histories and 64 history-target cases with known probabilities.

The experiment has three separate data roles. All repeated episodes retain
their original weight.

| Role | Default count | Use |
| --- | ---: | --- |
| Training | 4,096 | Fit history scaling, Gaussian KAHM geometry, readout features, and ridge coefficients |
| Validation | 1,024 | Select decoder settings and the shared KAHM bank |
| Calibration | 4,096 | Assess the three selected predictors after the model/protocol freeze |

Encoder fitting uses training histories without future-response targets.
Targets enter the training decoder fit and validation selection. The calibration
stream is reserved during development and materialized only after the full
freeze preflight. See [the freeze interface](FREEZE_INTERFACE.md) for the
implemented access gates.

## 4. Follow RAW, MEM, and LOCAL

Let `z` be the history after stored training-only standardization. The three
branches represent the same history in different ways:

| Candidate | Representation before readout scaling | Width in the selected primary bank |
| --- | --- | ---: |
| RAW | The two coordinates of `z` | 2 |
| KAHM MEM | Six local reconstruction-affinity memberships `phi` | 6 |
| KAHM LOCAL | The identical MEM prefix, followed by `phi[j] * P[j].T @ (z-mu[j])` for each local model | 18 |

The KAHM models use Gaussian kernels and a signed regularized reconstruction.
Reconstruction scores become memberships normalized **within each partition**.
The selected bank has one partition with two local models and another with
four. Each partition's memberships sum to one; the concatenation has two such
blocks. These numbers describe relative reconstruction affinities. They are not
calibrated probabilities of latent states or future outcomes.

MEM and LOCAL share exactly the same selected bank, preprocessing, references,
PCA, reconstruction rules, and memberships. LOCAL adds coordinates of the input
relative to local means and bases. It preserves MEM exactly as its prefix.
The primary selected bank is `cap_2`; both `cap_1` and `cap_2` remain in the
development archive. [ENCODER_DESIGN.md](ENCODER_DESIGN.md) gives the equations
and numerical fallback rules.

Each branch then receives training-only column scaling, Gaussian Nyström
features using common raw-history landmark identities, and ridge regression
with an unpenalized intercept. Finite predictions are clipped to `[0,1]`.
The search contains 45 RAW settings and 90 settings for each KAHM representation,
225 in total. Within each bank, MEM and LOCAL select their best decoders; their
shared bank is selected by the mean of those two validation losses. Matching
this readout procedure does not imply equal feature capacity or total search
effort. See [READOUT_DESIGN.md](READOUT_DESIGN.md).

In inspection mode, the displayed forecast is a lookup from the hash-bound
canonical prediction table. Strict verification additionally establishes its
agreement with the saved computational model in the current runtime.

## 5. Read q, U, and R separately

| Quantity | Meaning | How it is obtained |
| --- | --- | --- |
| Calibration MSE `q` | Average loss on the reserved 4,096 calibration episodes | Exact rational arithmetic on the represented binary64 forecasts and exact targets |
| Issued upper endpoint `U` | A statistical upper bound on the frozen predictor's population MSE, under the declared premises | Verified binary-KL inversion and independent checks of the saved endpoint inequalities |
| Population MSE `R` | Expected loss under this benchmark's known probability law | Exact weighted enumeration of all 64 cases, after certificate issuance and verification |

The target tolerance is exactly `1/10`. Acceptance compares the exact fraction
`U` with this tolerance. Endpoint decimals are rounded upward; displayed `q`
and `R` values are approximate presentations of exact fractions. A calibration
mean can be smaller than the Bayes population risk. A single sample mean is
not a population lower bound.

The three possible certificate statuses are retained:

| Status | Meaning |
| --- | --- |
| `requirement_met` | A verified endpoint satisfies `U <= 1/10` |
| `unresolved` | A verified endpoint exceeds `1/10`; the actual risk may still meet the tolerance |
| `certificate_unavailable` | The required numerical check was not established; no endpoint is supplied |

The audit reports a separate coverage outcome: `covered` when `R <= U`,
`coverage_miss` when `R > U`, or `endpoint_unavailable`. It also reports the
independent comparison `R <= 1/10`. Every candidate and every miss remains in
the report. An exact risk calculation does not repair an unavailable or missed
certificate. The supplied primary records report all three requirements met
and all three endpoints covered; those are outcomes of this recorded experiment.

Conditional on development, and assuming iid calibration episodes from the
declared law independent of development, the probability that **any reported
valid endpoint** misses its corresponding fitted risk is at most `1/20`.
Each candidate receives failure allocation `1/60`; all three share the same
calibration episodes. There is no extra conditioning on acceptance or numerical
availability, and an unavailable candidate's allocation is not reassigned.
Recorded pseudorandom stream identities support reproducibility; they do not
independently prove these statistical assumptions.

This statement concerns population MSE of the actual frozen computational
predictors. The demonstrated comparisons do not establish per-forecast coverage,
decoder-class optimality, information preservation, Koopman closure, recursive
rollout accuracy, or general KAHM superiority. See
[CERTIFICATION_DESIGN.md](CERTIFICATION_DESIGN.md) and
[POPULATION_AUDIT.md](POPULATION_AUDIT.md) for the exact claim and arithmetic.

## 6. Request strict verification

The retained baseline is `public-v0.1.0`. Version 0.1.1 fixes encoder artifact
input validation, changing source bytes that the historical freeze binds.
Inspection remains available in v0.1.1. For strict verification, obtain the
unmodified [v0.1.0 release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.0)
and use the exact recorded runtime. Run both commands below from that v0.1.0
project directory:

```bash
python -m kahm_demo tutorial --project . --mode verify --history 0.25 0.75
```

The underlying audit check can also be called directly:

```bash
python -m kahm_demo verify-audit --audit audits/primary/audit.json --certificate certificates/primary/certificate.json --bundle data/primary --freeze frozen/primary/freeze.json
```

This checks the frozen development snapshot, all 225 saved candidate states,
the exact calibration loss ledger and issued endpoint inequalities, and the
population audit. Verification replays saved parameters; it does not fit new
parameters, resample, or repeat the endpoint root search.

**A runtime mismatch is a refusal to verify.** The supplied snapshot records
CPython 3.12.14 and NumPy 2.3.5, but matching those version numbers is insufficient.
The comparison also includes the full Python build string (`sys.version`),
machine architecture, and complete NumPy build configuration, including BLAS
metadata. Installed numerical source identities must match too. The
version pins in `requirements-reproduction.txt` do not recreate all of that environment.

Keep the supplied archive intact. Use inspection for portable reading, or build
a fresh experiment in your current runtime. Do not edit archived hashes or
remove a failed runtime check to obtain a passing verification result.

## 7. Build a fresh full experiment

Return to the v0.1.1 project and its installed package. Run this separate
command from its root to create evidence for the current sources and runtime:

```bash
python -m kahm_demo reproduce --output runs/tutorial --master-seed 20261110 --run-id tutorial-v0.1
```

The output must be a fresh directory. The default run uses 4,096 training
episodes, 1,024 validation episodes, the complete 225-setting search, and 4,096
reserved calibration episodes. It records the current runtime and completes
development, freezing, certification, and the exact post-certificate audit.
The run ID differs from `public-v0.1.0`, so its documented stream identity is a
different experiment even with the same master seed.

The new directory contains `data/`, `encoders/`, `models/`, `frozen/`,
`certificates/`, and `audits/`. Its `reproduction_plan.json` records the intended
run before data generation; `reproduction.json` records completed stages and
all outcomes. `pipeline_complete` describes completion, independently of whether
any certificate meets the tolerance. An interrupted run retains its partial
artifacts and a failure record; the command does not resume or retry it.

The anchored tutorial continues to use `data/primary`, `frozen/primary`,
`certificates/primary`, and `audits/primary`. Fresh runs remain separate and
retain all their outcomes. Repeating an existing seed and run identity is a
reproducibility exercise, not a new independent sample.

Use the ordinary strict API to verify this fresh run:

```bash
python -m kahm_demo verify-audit --audit runs/tutorial/audits/audit.json --certificate runs/tutorial/certificates/certificate.json --bundle runs/tutorial/data --freeze runs/tutorial/frozen/freeze.json
```

Existing output is refused. Keep every completed or failed attempt and its
outcomes. Rerunning the same stream supplies no new
independent evidence; repeatedly changing streams until an attractive result
appears requires an appropriate repeated-attempt statistical design. The fixed
95% statement is for one final calibrated family.

For the supplied experiment, the authoritative records are
[freeze.json](../frozen/primary/freeze.json),
[certificate.json](../certificates/primary/certificate.json), and
[audit.json](../audits/primary/audit.json). Model files preserve their historical
development statuses; the enclosing freeze and the separate certificate supply
their later authorities. The public run plan and stage outputs are retained in `reports/`.

For a real dataset, continue with
[ADAPTING_TO_NEW_DATA.md](ADAPTING_TO_NEW_DATA.md). That is an engineering and
evaluation design task: the current command line implements this controlled
four-state benchmark and has no generic CSV adapter.

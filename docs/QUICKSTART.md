# Quick start

Start with the recorded class-risk study or inspect the fitted-risk experiment
`public-v0.1.0`. The package can also run separate experiments with their own
evidence. Strict replay of `public-v0.1.0` requires the unmodified v0.1.0 sources
and exact recorded runtime. Class-risk verification checks archived numerical
witnesses and does not replay the executable encoder.
The [scientific reference](SCIENTIFIC_REFERENCE.md) explains the guarantees,
and the [method overview](METHOD_OVERVIEW.md) maps them to the implementation.

## 1. Prepare the complete project

Use Python **3.12 or newer** and obtain the complete published repository:

```bash
git clone https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting.git
cd certified-kahm-representations-forecasting
```

For the supplied walkthrough, the project directory must contain `kahm_demo/`,
`data/primary/`, `frozen/primary/`, `certificates/primary/`, and `audits/primary/`.
A wheel installs the code but does not include these experiment directories.
The class-risk study additionally uses `reports/lower_certificates_v1/`.
Use a documented release or commit when reproducing or citing a study.

An optional isolated environment keeps installation separate from other work:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows PowerShell, replace the activation command with
`.venv\Scripts\Activate.ps1`. The package depends on NumPy `>=2,<3`.
The source-directory inspection command below uses only the standard library;
it can also run directly from the complete checkout without installing NumPy.
Fitting and strict numerical verification require the numerical dependencies.

## 2. Verify the recorded class-risk study

From the matching source checkout, extract the evidence into a fresh directory
and run its verifier:

```bash
python -m zipfile -e reports/lower_certificates_v1/full_verification_bundle.zip runs/retained-lower-study
python -m kahm_demo verify-class-risk --run runs/retained-lower-study/run
```

The archive contains the recorded observations, class definitions, numerical
witnesses, source snapshots, and population audit. It is an evidence bundle,
not the complete source distribution; the checkout or supplied wheel provides
the executable package. Verification recomputes the witness checks, source
identities, and calibration-stream identity. It neither fits models nor solves
new optimization problems, and it does not replay the executable encoder.

Read [the study report](../reports/lower_certificates_v1/report.md) for the
five-class comparison, block hierarchy, sample-size tradeoff, and
unresolved result. A lower confidence bound $L\gt \tau$ excludes the prescribed
class; an admissible fitted predictor with upper bound $U\leq\tau$ meets the
MSE requirement. See [Certificates explained](CERTIFICATES_EXPLAINED.md).

To create a separate, single-size assessment on the same archived features:

```bash
python -m kahm_demo class-risk --freeze frozen/primary/freeze.json --output runs/class-risk
python -m kahm_demo verify-class-risk --run runs/class-risk
```

This command creates a new class specification and calibration plan with its
own confidence allocation. It uses exact archived feature/prediction tables
on the benchmark support. The [class-risk guide](LOWER_CERTIFICATES.md)
documents parameters, assumptions, and the complete study design. Reusing
the same seed and run ID replays observations rather than creating another
independent replication.

## 3. Inspect the fitted-risk example

```bash
python -m kahm_demo tutorial --project . --mode inspect --history 0.25 0.75
```

The two input numbers are the **previous and current observed values**, in that
order. Each must belong exactly to `{0, 0.25, 0.75, 1}`. The helper rejects
unsupported values instead of rounding them to a nearby supported history.

The command checks the fixed primary archive identities, complete recorded
inventories, and linked data bytes. It displays recorded encoder features,
predictions, and all three risk results. The following is an excerpt of the
expected JSON fields:

```json
{
  "status": "archive_identity_checked",
  "mode": "archive_identity_only",
  "archive_identity_checked": true,
  "model_replayed": false,
  "certificate_reverified": false,
  "population_risk_recomputed": false,
  "calibration_resampled": false,
  "models_refitted": false
}
```

These false flags are intentional. Inspection establishes the identity of the
displayed archive, not current-runtime execution of its models or independent
mathematical verification of its risk bounds. See
[Certificates explained](CERTIFICATES_EXPLAINED.md) for the distinction.

For a shorter, human-readable terminal walkthrough of the same evidence:

```bash
python -m examples.guided_tutorial --project . --mode inspect --history 0.25 0.75
```

The `tutorial` helper is anchored to the supplied primary experiment. It does
not search for another run or accept arbitrary replacement artifact identities.

## 4. Use the notebook or static reading view

```bash
python -m pip install '.[tutorial]'
python -m jupyterlab notebooks/01_kahm_certified_forecasting.ipynb
```

Select the Python kernel from the environment in which you installed the
package. The [notebook](../notebooks/01_kahm_certified_forecasting.ipynb) defaults
to `MODE = "inspect"`. In the interactive notebook, run its cells in order.
It shows the representation layout, one history's forecasts, and the separate
meanings of calibration mean `q`, upper endpoint `U`, and benchmark population
risk `R`.

**Run All does not start training or create calibration data.** The fresh-run
command is displayed in Markdown, outside executable cells. In the interactive
notebook, changing `MODE` to `"verify"` requests the strict checks in the next
section; the `public-v0.1.0` example requires the v0.1.0 release and its exact
recorded runtime.

For the [HTML walkthrough](../notebooks/01_kahm_certified_forecasting.html), open
the extracted `notebooks/01_kahm_certified_forecasting.html` file in your browser.
Keep it in the complete project so its relative documentation links work.
It is a static reading copy with recorded inspection output; opening it does
not execute a notebook kernel or verify a certificate. The
[v0.1.0 validation report](../reports/RELEASE_VALIDATION.md) records the
checks performed for the v0.1.0 baseline.
Official application setup is documented by
[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

## 5. Verify the fitted-risk numerical evidence

The supplied fitted-risk evidence belongs to `public-v0.1.0`. Its freeze binds
the v0.1.0 execution sources and recorded runtime. Archive inspection does not
require a matching runtime; strict verification does. Obtain the unmodified
[v0.1.0 release](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/releases/tag/v0.1.0)
and use the exact recorded runtime. Run this command from that v0.1.0 project:

```bash
python -m kahm_demo tutorial --project . --mode verify --history 0.25 0.75
```

A successful result reports `mode: full_read_only_verification` and true values
for `model_replayed`, `certificate_reverified`,
`calibration_losses_recomputed`, and `population_risk_recomputed`. The existing
strict verifier checks the frozen states, calibration receipt and exact loss
ledger, issued endpoint inequalities, and exact population audit. The helper
then replays the selected history's three forecasts and compares their exact
binary64 encodings with the archived predictions.

Verification reads existing data and states. It does not sample, refit, repeat
the endpoint root search, or create a replacement assessment. Failure stops the
operation; inspection is never substituted for a failed verification.

### If your sources or runtime do not match

The primary records name CPython 3.12.14 and NumPy 2.3.5. They also bind the
full Python build string, machine architecture, complete NumPy/BLAS build
configuration, and numerical execution sources. Installing the same version
numbers, including through `requirements-reproduction.txt`, does not guarantee
that these identities match.

Keep the primary evidence unchanged. Use `inspect` to read it portably, or use
the fresh-run route below. A supported environment may be perfectly suitable
for a new experiment while being incompatible with strict replay of this
particular snapshot. Do not edit archived identities to override a mismatch.

### Lower-level read-only commands

| Command | Scope |
| --- | --- |
| `verify-freeze` | Frozen model/protocol snapshot and development replay |
| `forecast-frozen` | One history's forecast from an identity-checked frozen model |
| `verify-certificate` | Frozen preflight, exact calibration losses, and direct checks of issued endpoints |
| `verify-audit` | Certificate verification plus recomputation of exact benchmark risks and comparisons |

These commands have the same source/runtime requirement. For
`public-v0.1.0`, run them from the unmodified v0.1.0 project in its recorded runtime:

```bash
python -m kahm_demo forecast-frozen --freeze frozen/primary/freeze.json --candidate kahm_local --history 0.25 0.75
python -m kahm_demo verify-certificate --certificate certificates/primary/certificate.json --bundle data/primary --freeze frozen/primary/freeze.json
```

`forecast-frozen` reports `risk_certificate_checked: false`: that particular
command verifies prediction provenance but does not inspect the separate risk
assessment. It does not mean that no certificate exists. The installed
`kahm-demo` executable exposes the same command interface as
`python -m kahm_demo`.

## 6. Run a fresh fitted-predictor experiment

Use the complete project and its installed package for a fresh experiment.
This operation fits models and generates artifacts bound to the executing
sources and runtime:

```bash
python -m kahm_demo reproduce --output runs/tutorial --master-seed 20261110 --run-id tutorial-v0.1
```

`runs/tutorial` must not already exist. The defaults use 4,096 training episodes,
1,024 validation episodes, both encoder banks, all 225 decoder configurations,
and 4,096 calibration episodes. The orchestrator commits a plan before work and
then runs these stages once:

1. Generate development data and reserve the calibration stream.
2. Fit training-only Gaussian KAHM encoders.
3. Fit and validation-select the three complete forecasting procedures.
4. Freeze the selected models and fixed calibration protocol.
5. Materialize reserved calibration data and issue the verified assessment.
6. Verify that assessment and perform the exact post-certificate population audit.

Progress messages name the stages. Successful completion returns
`status: pipeline_complete` and writes `runs/tutorial/reproduction.json`.
That status means the workflow completed; candidate certificate statuses and
coverage outcomes remain separate fields. The output layout is:

| Location under `runs/tutorial/` | Contents |
| --- | --- |
| `reproduction_plan.json` | Configuration, stream identities, intended stages, and runtime recorded before work |
| `data/` | Training, validation, reserved plan, and later calibration receipt/outcomes |
| `encoders/`, `models/` | Both banks, all 225 fitted states, and selection evidence |
| `frozen/` | Model/protocol freeze and complete archived development snapshot |
| `certificates/` | Loss ledger, issued endpoints, checks, and all three statuses |
| `audits/` | Exact benchmark risks and every realized coverage outcome |
| `reproduction.json` | Completed stages, artifact identities, and recorded results |

An interrupted run keeps its partial evidence and failure record. The command
does not overwrite, resume, retry, or discard an unfavorable result.

Verify the new run through the ordinary strict interface:

```bash
python -m kahm_demo verify-audit --audit runs/tutorial/audits/audit.json --certificate runs/tutorial/certificates/certificate.json --bundle runs/tutorial/data --freeze runs/tutorial/frozen/freeze.json
```

The public primary experiment uses run ID `public-v0.1.0`. The fresh-run
example uses a different identity. Reusing an existing seed and run identity
is a reproducibility exercise, not another independent sample. Repeatedly
changing identities until a favorable assessment appears requires an appropriate
statistical design; the fixed confidence statement applies to one final
calibrated family.

The resulting freeze can also supply the class-risk workflow:

```bash
python -m kahm_demo class-risk --freeze runs/tutorial/frozen/freeze.json --output runs/tutorial-class-risk
python -m kahm_demo verify-class-risk --run runs/tutorial-class-risk
```

This uses that run's newly fitted feature tables. Its outcomes can differ from
the recorded class-risk study, which assesses the `public-v0.1.0` tables.

## 7. Common questions and failures

| Situation | Meaning and next step |
| --- | --- |
| Missing primary archive or identity mismatch | Use the complete unmodified project and the correct `--project` path. The wheel alone is insufficient for archive inspection. |
| Unsupported history | Use exactly two supported observation values in temporal order. The tutorial is a finite-benchmark interface. |
| Strict source/runtime mismatch | Inspect the archive, verify `public-v0.1.0` with unmodified v0.1.0 sources and its exact runtime, or create a fresh experiment. Do not change archived identities to bypass the check. |
| Existing reproduction output | Preserve it. A new execution needs a genuinely fresh destination; the command cannot resume or overwrite a run. |
| `unresolved` | Available bounds establish neither the relevant adequacy decision nor class exclusion. An upper endpoint above the threshold does not prove excessive risk. |
| `class_excluded` | The class-risk lower confidence bound exceeds the MSE threshold for the prescribed representation, metric, and Lipschitz bound. |
| `certificate_unavailable` | The required endpoint was not established; retain the diagnostic and all candidates. |
| `coverage_miss` | The known benchmark risk exceeds an issued endpoint; the audit retains this outcome without changing the bound. |
| `pipeline_complete` with a disappointing result | The engineering workflow completed. Acceptance and coverage are deliberately independent of that completion status. |

For portable automated checks and the separate strict-runtime gate, see
[CI.md](CI.md). For the complete workflow and Python APIs, see
[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md).

For your own dataset, continue with
[ADAPTING_TO_NEW_DATA.md](ADAPTING_TO_NEW_DATA.md). The command line has
no generic data importer, and its iid assumptions do not automatically apply
to overlapping time-series windows.

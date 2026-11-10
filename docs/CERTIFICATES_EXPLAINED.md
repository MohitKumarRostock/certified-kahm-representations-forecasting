# What the forecasting certificates mean

The certificate gives an upper bound on the **population mean squared error of
a particular frozen forecasting model**, under a declared evaluation law and
sampling assumptions. It is a statement about expected loss over that
population. It does not provide an interval for each next observation or
guarantee that every individual forecast has small error.

The demonstrator reports three such models: RAW, KAHM MEM, and KAHM LOCAL.
They are selected during development and fixed before final calibration.
Their geometry, preprocessing, decoder, numerical behavior, and clipping rule
are all part of the assessed predictor. The governing definitions are the
[scientific contract](../kahm_demo/specification.json) and
[calibration protocol](../kahm_demo/calibration_protocol.json). Version 0.1.0
includes the complete local proof in
[FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md).

## Three numbers with different jobs

| Quantity | Meaning | Where it comes from |
| --- | --- | --- |
| **q: calibration MSE** | Average squared error on the reserved sample | All 4,096 calibration episodes, with every occurrence retained |
| **U: issued upper endpoint** | A statistical upper bound on the fitted predictor's population MSE, under the stated premises | The bounded-loss confidence method and verified numerical calculation |
| **R: population MSE** | Expected squared error under the declared benchmark law | Exact enumeration of all 64 history-target cases, after certificate issuance and verification |

For development outcome `D` and selected predictor `f_m,D`, the certified target
is

$$
R_m(D)=\mathbb E_{(X,Y)\sim P}\bigl[(Y-f_{m,D}(X))^2\bigr].
$$

Here `P` is the specific [four-state benchmark distribution](BENCHMARK.md). The model predicts
one scalar observation from a two-observation history. Targets and clipped
forecasts lie in `[0,1]`, so the squared loss lies in `[0,1]`.

The sample mean `q` can differ from `R` in either direction. Even a calibration
mean below the Bayes population risk is possible: a sample mean is not a lower
bound on population error. The difference `q-R` is one realized sampling
deviation. The difference `U-R` is realized certificate slack, not simply a
floating-point rounding error.

The known finite law makes exact `R` available in this example. Most real
datasets do not supply such an oracle. A justified fitted-risk certificate can
still be possible without an exact population audit; a test-set average must
remain labeled as an empirical estimate rather than exact `R`.

## What “95% simultaneous” means

The final family contains exactly three selected predictors. They share 4,096
calibration episodes. Each receives failure-probability allocation `1/60`,
which sums to `1/20 = 0.05` across the family.

The statistical assumptions are:

1. Calibration episodes are iid draws from the declared evaluation law.
2. Calibration is independent of all development that determined the predictors.
3. The assessed loss is bounded in `[0,1]`, as specified before calibration.
4. The reported numerical endpoints satisfy the required verified inequalities.

Conditional on development, the statement is

$$
\Pr_{\mathrm{Cal}}\!\left[
\exists m\text{ with a reported valid endpoint and }R_m(D)>U_m
\mid D\right]\le\frac1{20}.
$$

Thus, over repetitions of calibration under these premises with development
held fixed, the probability of any reported bound missing its corresponding
fitted risk is at most 5%. There is no additional conditioning on numerical
availability or acceptance. Shared calibration does not require the three
endpoint events to be independent. An unavailable candidate remains in the
report, and its allocation is not reassigned.

The 225 fitted development configurations are selection history. The calibrated
family consists of the three selected models frozen before calibration, so the
allocation is across those three. Additional outcome-driven attempts are not
automatically covered by that allocation.

Recorded pseudorandom stream identities make the numerical experiment
reproducible. They do not independently establish iid sampling or independence
from development. Arbitrary overlapping windows from a single trajectory do
not inherit this guarantee by being called “episodes.” A new application needs
an evaluation design and theorem appropriate to its sampling dependence.

## Acceptance and realized coverage are different decisions

The primary requirement is population MSE at most the exact rational `1/10`.
The **certificate status** uses the issued endpoint:

| Status | Meaning |
| --- | --- |
| `requirement_met` | A verified exact endpoint satisfies `U <= 1/10`, under the declared premises |
| `unresolved` | A verified endpoint exceeds `1/10`; actual risk could still satisfy the requirement |
| `certificate_unavailable` | The required numerical calculation or verification was not established; no endpoint is reported |

Because the benchmark additionally provides exact `R`, the **population audit**
can make a separate realized comparison:

| Audit outcome | Meaning |
| --- | --- |
| `covered` | `R <= U` |
| `coverage_miss` | `R > U`; the issued bound and this miss are both retained |
| `endpoint_unavailable` | There is no endpoint to compare with the known benchmark risk |

The audit also reports whether `R <= 1/10`. That exact-law decision does not
change the original certificate status. For example, a model can satisfy the
population tolerance while its certificate remains unresolved. An exact audit
also does not turn an unavailable endpoint into an issued certificate.

Read the actual statuses from `certificates/primary/certificate.json` and
`audits/primary/audit.json` when those records are present. No successful
outcome is assumed by this guide. Each completed population audit is one
realized check, not an estimate of long-run coverage frequency or evidence of
general representation superiority.

A 95% guarantee permits some realized misses. A miss is retained and
investigated, rather than repaired by selecting another seed or changing the
endpoint after seeing the audit. The software preserves all candidates,
unavailable results, unresolved requirements, and partial failure evidence.

## How the numerical evidence is constructed

The workflow has five distinct stages:

1. **Develop.** Fit geometry from training inputs, fit decoders from training
   inputs and targets, and choose settings using validation outcomes. MEM and
   LOCAL share the selected KAHM bank.
2. **Freeze.** Save the complete computational predictors, inference rules,
   source/runtime identities, and calibration protocol before final outcomes
   are accessed.
3. **Measure.** Evaluate every reserved calibration occurrence using the frozen
   predictions. Interpret finite binary64 outputs as their exact represented
   rational numbers, then compute the squared losses and means exactly.
4. **Certify.** Construct a conservative upper inverse of binary KL at
   `log(60)/4096`. Use verified logarithm enclosures and directed root bracketing;
   recheck the final endpoint inequalities with fresh arithmetic enclosures.
5. **Audit.** After verifying the issued certificate, enumerate the benchmark's
   known population law. Compare exact `R` with the preserved endpoint and
   tolerance while retaining every outcome.

Binary KL is the function used for the bounded-mean confidence calculation;
the target observations need not be binary. The implemented root calculation
uses exact integer/rational arithmetic with bounded work. If a required
calculation cannot be established, its result is unavailable rather than a
silently substituted endpoint.

Exact fraction fields govern decisions. The displayed upper endpoint is rounded
**upward** to 18 decimal places; displayed `q` and `R` values are approximations
of their exact fractions. Neither a displayed decimal nor a plotting conversion
is used for acceptance.

The assessed object is the actual saved computational predictor. Numerical
fitting diagnostics do not prove equivalence to an ideal real-arithmetic KAHM
map. This fitted-risk upper certificate does not need a full-history
injectivity, predictive-sufficiency, or Koopman-closure theorem. Those would be
separate claims with separate evidence.

## Inspection and verification establish different things

| Operation | What it establishes |
| --- | --- |
| Archive inspection | Fixed primary record and payload identities match; the displayed values are the archived ones |
| Runtime compatibility precheck | Recorded and current source/runtime metadata match sufficiently to attempt strict verification |
| Strict certificate verification | Frozen preflight passes; all calibration losses reproduce exactly; saved endpoint inequalities, allocation, and protocol checks pass |
| Strict population-audit verification | Certificate verification passes; exact population contributions and saved comparisons independently reproduce |

Archive inspection does not execute the model or recheck the endpoint
mathematics. A compatibility precheck is also not full verification. Strict
verification checks exact execution-source identities and the recorded runtime,
including the full Python build and NumPy/BLAS configuration. Version pins
alone are insufficient. A mismatch stops verification without a fallback to
inspection or an adjustment to the archived evidence.

The read-only certificate verifier rechecks issued endpoint inequalities; it
does not repeat the root search. It neither refits nor resamples. The population
audit deliberately recomputes exact model risks, while certificate-only
verification does not use them. Instructions are in
[QUICKSTART.md](QUICKSTART.md).

## Scope when moving beyond this benchmark

The current certificate concerns one final frozen family, one scalar horizon,
one bounded squared loss, and the declared evaluation population. It does not
establish pointwise forecast accuracy, decoder-class optimality, representation
information retention, global Lipschitz admissibility, certified recursive
rollout, or performance under a changed law.

A new dataset, changed loss or clipping rule, multiple horizons, a modified
predictor, or repeated attempts require a newly justified assessment design.
The current package has no generic CSV adapter. The
[adaptation guide](ADAPTING_TO_NEW_DATA.md) maps those decisions to the modules
that would need to change.

For the complete mathematical proof, read
[FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md). For implementation details, read
[CERTIFICATION_DESIGN.md](CERTIFICATION_DESIGN.md),
[FREEZE_INTERFACE.md](FREEZE_INTERFACE.md), and
[POPULATION_AUDIT.md](POPULATION_AUDIT.md). The
[method overview](METHOD_OVERVIEW.md) connects these calculations to the
forecasting representations and readout.

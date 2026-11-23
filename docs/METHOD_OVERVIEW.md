# Method and implementation overview

This project combines a geometric representation, fitted forecasting models,
and risk certificates for both fitted predictors and prescribed
representation–decoder classes. This guide maps the mathematical objects to their definitions and
executable modules. The [scientific reference](SCIENTIFIC_REFERENCE.md) gives
the self-contained statement of the problem, assumptions, hierarchy, and
decision rules.

## 1. Separate the objects being constructed

| Layer | Inputs and output | Local definition | Main modules |
| --- | --- | --- | --- |
| Benchmark | Independently sampled two-observation histories and scalar next targets | [Benchmark](BENCHMARK.md) | `law.py`, `sampling.py`, `datasets.py` |
| Geometry | Training inputs to shared reference partitions, local PCA, and Gaussian KAHMs | [Encoder design](ENCODER_DESIGN.md), sections 2–4 | `geometry.py`, `kahm.py`, `encoder_workflow.py` |
| Soft-state representation | Reconstruction scores to MEM or membership-weighted input coordinates in LOCAL | [Encoder design](ENCODER_DESIGN.md), section 5 | `soft_state.py`, `encoder.py` |
| Readout and selection | Training targets and validation losses to three selected clipped predictors | [Readout design](READOUT_DESIGN.md) | `readout_features.py`, `ridge.py`, `selection.py`, `forecast_workflow.py` |
| Frozen predictor | Complete saved state to canonical predictions and bound source/runtime identities | [Freeze interface](FREEZE_INTERFACE.md) | `forecast_state.py`, `freeze.py` |
| Fitted-predictor risk assessment | Reserved calibration losses to checked upper confidence bounds | [Fitted-risk guarantee](FITTED_RISK_GUARANTEE.md), [certification](CERTIFICATION_DESIGN.md) | `calibration_ledger.py`, `verified_arithmetic.py`, `risk_bounds.py`, `certificate_workflow.py` |
| Optimal class-risk assessment | Fixed representation, metric, and Lipschitz bound to lower confidence bounds and admissibility checks | [Class-risk guide](LOWER_CERTIFICATES.md), [proof](LOWER_CERTIFICATE_PROOF.md) | `lipschitz_class.py`, `class_bounds.py`, `class_risk_workflow.py` |
| Population audit | Known finite law to fitted risks, bounds on optimal class risk, and recorded coverage comparisons | [Population audit](POPULATION_AUDIT.md), [class-risk guide](LOWER_CERTIFICATES.md) | `population_audit.py`, `audit_workflow.py`, `class_oracle.py` |

The encoder is fitted without forecasting targets. A fitted decoder, also
called the readout in the implementation, completes the forecasting model.
An upper confidence bound assesses that fitted predictor. A lower confidence
bound assesses the best risk attainable by any decoder satisfying the
prescribed class constraints. These statements have different targets.

## 2. Gaussian KAHM features

A local model reconstructs a standardized history using a regularized Gaussian
kernel over reference histories. Its normalized affine weights can be signed.
The implementation defines cancellation checks and an affine fallback; these
choices are part of the computational model.

Each reconstruction receives a radial/angular discrepancy score. Affinities
are normalized separately within each partition. Their concatenation is MEM.
LOCAL retains that exact membership prefix and appends

```math
\phi_j(z)P_j^T(z-\mu_j)
```

for each local model. These are coordinates of the input relative to its local
geometry, not coordinates of the reconstruction. Rank-zero models append no
coordinates. Memberships express relative reconstruction affinities, not
calibrated probabilities of latent states.

The MEM and LOCAL candidates share one fitted bank, including preprocessing,
partitions, references, PCA, Gaussian metrics, regularizers, scores, and fallback
rules. RAW provides the standardized observed history as a control. The encoder
guide defines every formula, threshold, and zero-norm convention.

## 3. Comparable readouts and explicit selection

All candidates use training-only column scaling, a Gaussian Nyström block
appended to the standardized linear representation, and ridge with an
unpenalized intercept. The readout keeps every supplied feature column. The
final scalar forecast is finite-checked and clipped to `[0,1]`.

The menu uses five bandwidths and nine ridge penalties. RAW has 45 fits; MEM
and LOCAL each have 90 across two encoder banks. Within each bank the best MEM
and LOCAL decoders are selected separately, then their common bank is chosen by
the mean of those two validation losses. RAW is selected separately. Exact
ties follow the recorded ordering. No population risk enters selection.

The matched family and common raw-history landmark identities improve the
interpretability of this comparison. They do not make feature widths, retained
kernel ranks, effective capacities, or total search effort equal. The assessed
objects are the complete selected procedures.

## 4. Fitted-predictor risk and its upper confidence bound

| Quantity | Definition | What establishes it |
| --- | --- | --- |
| `q` | Mean loss on the reserved calibration sample | Every observed occurrence and exact represented prediction |
| `U` | Conservative statistical upper endpoint for the fitted population risk | The stated sampling assumptions, bounded-loss argument, and verified endpoint arithmetic |
| `R` | Exact expectation under this known benchmark law | All 64 weighted history-target cases, evaluated after certification |

The `certify` workflow conditions on development: the selected predictors are fixed
before calibration. With 4,096 iid calibration episodes independent of development,
allocation `1/60` per candidate bounds the probability that any reported valid
endpoint misses its risk by `1/20`. The local proof derives this from a
bounded-loss Chernoff inequality and a union bound. It does not assume Bernoulli
losses or independence between the three endpoints.

An upper endpoint is checked numerically with exact rational arithmetic and
directed logarithm enclosures. The decimal display of `U` rounds upward; the
requirement comparison uses its exact fraction. An unavailable endpoint stays
unavailable, with diagnostics and its unchanged allocation.

The later exact audit checks a realized coverage inequality. It does not
measure long-run coverage frequency, improve the confidence level, or repair a
miss. A calibration mean can lie below the Bayes population risk without a
contradiction because `q` is a finite-sample observation.

## 5. Optimal class risk and the block hierarchy

Fix a representation $\Gamma$, a feature pseudometric $d$, a Lipschitz bound
$\kappa\geq0$, and a common target and prediction interval $I=[0,1]$.
An admissible decoder satisfies

```math
g:\mathcal U\to I,\qquad
|g(u)-g(v)|\leq\kappa d(u,v).
```

Its population MSE is $R(g)=\mathbb E[(Y-g(\Gamma(X)))^2]$, and the
**optimal class risk** is $`R_\kappa^*=\inf_{g\in\mathcal G_\kappa(d)}R(g)`$,
where $\mathcal G_\kappa(d)$ is the measurable $I$-valued Lipschitz decoder class.
The `class-risk` workflow freezes the class specification before drawing its
own iid calibration episodes, independently of development.

For $b\geq2$ observations, the block score $T_b$ is the minimum total squared
error subject to all pairwise Lipschitz constraints, divided by $b-1$.
The expected scores satisfy

```math
\mathcal W_2\leq\mathcal W_3\leq\cdots\leq R_\kappa^*,
\qquad \mathcal W_b=\mathbb E[T_b].
```

The [proof](LOWER_CERTIFICATE_PROOF.md) derives this hierarchy and its
convergence on the implemented finite feature support. Numerically verified
optimization lower bounds support two statistical constructions: a bound
using independent block scores and a bound on minimum empirical risk using
the same optimization witnesses. The study also reports full empirical optimization.
Taking the maximum of the prespecified, simultaneously valid lower endpoints
gives $`L\leq R_\kappa^*`$ on the joint confidence event.

An upper bound $U$ on a fitted predictor's risk also bounds optimal class risk
only when that predictor is admissible. On the benchmark support, the verifier
checks all required pairwise constraints on its prediction table. For the MSE
threshold $\tau$:

| Decision | Required evidence |
| --- | --- |
| `class_excluded` | $L>\tau$: every decoder in the prescribed class exceeds the threshold |
| `requirement_met` | An admissible fitted prediction table has $U\leq\tau$ |
| `unresolved` | Neither conclusion has been established |

The workflow retains conflicting conclusions and unavailable individual
endpoints for inspection. It does not replace calibration observations to
obtain a favorable decision. Its confidence allocation is separate from the
three-endpoint `certify` protocol.

The [recorded study](../reports/lower_certificates_v1/report.md) compares RAW,
MEM, LOCAL, and short/full-history controls at three calibration sizes. It
demonstrates class exclusion missed by pairwise certification, admissible
predictors meeting the threshold, and an unresolved small-sample result.

## 6. Computational and scientific scope

Saved numerical arrays and canonical inference define the predictor. Residual
checks, state replay, and source/runtime matching are engineering checks. They
are distinct from a verified error enclosure around an ideal real-arithmetic
Gaussian function. Certification concerns the represented predictions and the
stated evaluation law.

The class-risk workflow treats archived feature coordinates and prediction
tables as exact computational functions on the 16 supported histories. Its
admissibility check establishes the existence of a matching Lipschitz decoder;
it does not establish global regularity of the executable readout.
The known-law audit supplies independent reference bounds on optimal class
risk after sample certificates are issued.

Class exclusion is specific to $(\Gamma,d,\kappa)$ and the evaluation law.
Changing the coordinates, metric, or Lipschitz bound changes the assessed
class. The results establish no general representation superiority,
predictive sufficiency, Koopman closure, or recursive-rollout guarantee.
This small finite benchmark also does not establish a computational advantage
of blocks over full empirical optimization. A new application needs its own
task, data design, and justified assessment; see
[ADAPTING_TO_NEW_DATA.md](ADAPTING_TO_NEW_DATA.md).

Software authorship is recorded in [CITATION.md](../CITATION.md). It does not
assign sole authorship of all underlying research contributions. The method
scope, software licence, and statistical assumptions are separate statements.

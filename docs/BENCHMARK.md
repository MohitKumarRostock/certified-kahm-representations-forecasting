# Benchmark, forecast target, and data protocol

This document defines the controlled task used by both fitted-predictor and
class-risk assessments. The
machine-readable [scientific contract](../kahm_demo/specification.json),
[readout specification](../kahm_demo/readout_specification.json), and
[calibration protocol](../kahm_demo/calibration_protocol.json) bind its concrete
fitted-predictor implementation. The class-risk assessment has a
separate plan on the same task, described in Section 7. Changing the task
requires a separately identified contract and assessment. The
[scientific reference](SCIENTIFIC_REFERENCE.md) defines the common notation.

## 1. One-step forecasting from two observations

There are four state labels, `0,1,2,3`, with observation values

$$
v=(0,\tfrac14,\tfrac34,1).
$$

For state labels $a,b,c$, the model input and target are

$$
X=(v_a,v_b),\qquad Y=v_c.
$$

The history is ordered **previous observation, current observation**; the target
is the next observation. The task has one scalar output and horizon one.
Integer state labels and episode IDs identify records but are not model features.
The input alphabet contains 16 histories, and the joint history-target support
contains 64 triples.

The conditional law is

$$
P(c\mid a,b)=\frac45\,\mathbf1\{c=(a+b)\bmod4\}+\frac1{20}.
$$

The nominal successor has probability `17/20`; each other successor has
probability `1/20`. Each episode starts with a freshly drawn uniform ordered
pair, so

$$
P(a,b)=\frac1{16},\qquad
P(a,b,c)=
\begin{cases}
17/320,&c=(a+b)\bmod4,\\
1/320,&\text{otherwise}.
\end{cases}
$$

The pair distribution is stationary for the corresponding second-order chain:
for fixed $b,c$, summing $P(c\mid a,b)/16$ over $a$ gives `1/16`.
The implemented evaluation design nevertheless draws **independent episodes**;
it does not extract overlapping calibration windows from a single chain. There
is one history-target pair per episode and no burn-in step.

## 2. Sampling and exact observations

The sampler creates a private `random.Random` stream for each role. It draws a
uniform integer pair ID in `0..15`, decoded by `divmod(pair_id, 4)`, and a fresh
uniform ticket in `0..19`. Tickets `0..15` choose the nominal successor;
tickets `16..19` choose states `0..3`, respectively. The nominal successor thus
receives 17 of the 20 tickets.

Stream seeds are derived from a canonical record containing the sampler
namespace, benchmark identity, master seed, run ID, and role. Saved metadata
records the derivation, derived integer seed, RNG, and Python version. These
records make the generated observations reproducible. A fixed pseudorandom
sequence and different stream labels do not themselves prove the statistical
iid or independence assumptions; those remain premises of the evaluation model.

The four observation values are exact dyadic rationals and are also exactly
representable in binary64. Loss arithmetic uses their exact represented values.
Repeated histories and repeated triples are expected. Every episode retains
its own ID and full weight in fitting or evaluation. Deduplicating reference
and landmark identities or caching inference does not delete sample occurrences.

## 3. Data roles and lifecycle

| Role | Default episodes | Permitted use |
| --- | ---: | --- |
| Training | 4,096 | Fit input scaling, reference geometry, feature transformations, and ridge parameters |
| Validation | 1,024 | Select the readout configuration and common MEM/LOCAL bank by empirical MSE |
| Calibration | 4,096 | Assess the three already frozen computational predictors |

Encoder fitting accepts histories without targets. Readout feature fitting also
uses training inputs only; targets enter ridge fitting and validation selection.
The selected training-fit models are retained, without a pooled training-plus-
validation refit. The complete 225-configuration development search remains
available for replay.

A calibration reservation identifies the stream and fixed assessment protocol
before outcomes are accessed. The complete prediction rule, source/runtime
identity, sample count, loss, confidence allocation, and reporting rules are
frozen first. Calibration materialization and loading require full model
preflight. Existing calibration observations cannot be replaced, and their output
must remain outside the sealed model snapshot. See
[FREEZE_INTERFACE.md](FREEZE_INTERFACE.md) for the public gates.

The three final candidates share the same calibration episodes. Their endpoints
can be statistically dependent; the simultaneous guarantee does not require
independence across candidates. Additional model or seed attempts are not covered
automatically by this one-family allocation.

## 4. Actual fitted predictor and loss

The candidates are RAW, KAHM MEM, and KAHM LOCAL. MEM and LOCAL use the same
selected Gaussian KAHM bank. The readout supplies a finite scalar that is clipped
to `[0,1]` after required finite checks. All preprocessing, fallback behavior,
numerical types, and accumulation conventions belong to the frozen predictor.

For fixed development outcome $D$, the assessed risk is

$$
R_m(D)=\mathbb E_{(X,Y)\sim P}
       [(Y-f_{m,D}(X))^2].
$$

Both $Y$ and $f_{m,D}(X)$ lie in `[0,1]`, so the squared loss lies in
`[0,1]`. Risk is measured in squared units of the stated observations. No
empirical target-variance normalization is used. The certificate concerns the
actual computational prediction values, not an ideal real-arithmetic substitute.

## 5. Fitted-predictor assessment protocol

| Setting | Value |
| --- | --- |
| Final family | RAW, KAHM MEM, KAHM LOCAL |
| Calibration independent units | 4,096 episodes |
| Failure probability per predictor | `1/60` |
| Total failure probability | `1/20` |
| Simultaneous confidence | At least `19/20`, under the declared premises |
| Population-MSE threshold $\tau$ | Exact `1/10` |
| Numerical mean | Exact rational mean of all represented squared losses |
| Endpoint | Verified upper enclosure of the upper binary-KL inverse |

Conditional on development, iid calibration episodes from the stated law,
independent of development, give

$$
\Pr\!\left(\exists m\text{ with a reported valid endpoint and }
R_m(D)>U_m\mid D\right)\le\frac1{20}.
$$

There is no additional conditioning on acceptance or numerical availability.
Unavailable endpoints remain visible and do not release probability budget for
other candidates. The [local proof](FITTED_RISK_GUARANTEE.md) derives the
bounded-loss inequality, inversion, and simultaneous claim. The
[certification guide](CERTIFICATION_DESIGN.md) defines the numerical procedure.

An available endpoint at most `1/10` yields `requirement_met`; one above that
tolerance yields `unresolved`; a required numerical check that cannot be
established yields `certificate_unavailable`. None of these statuses is a
per-forecast interval or a statement about optimal risk over a model class.

## 6. Exact reference audit

Because the law and support are known, a frozen predictor has exact risk

$$
R(f)=\sum_{a,b,c}P(a,b,c)\,[v_c-f(v_a,v_b)]^2.
$$

The represented binary64 forecasts are treated as exact rationals. The model's
conditional-mean reference and decomposition are

$$
\mu(a,b)=\frac1{10}+\frac45v_{(a+b)\bmod4},\qquad
R(f)=\frac9{160}+\frac1{16}\sum_{a,b}[f(v_a,v_b)-\mu(a,b)]^2.
$$

The full-history Bayes risk is `9/160`; the best constant prediction `1/2` has
risk `5/32`; the nominal successor's observation has risk `1/16`. These are
analytical references, not additional learned or calibrated candidates.

Model-specific population risks are calculated only after the issued
certificate has been verified. They do not enter development selection,
endpoint construction, seed selection, or run filtering. The audit retains
`covered`, `coverage_miss`, and `endpoint_unavailable` outcomes without changing
the existing endpoint. Details are in [POPULATION_AUDIT.md](POPULATION_AUDIT.md).

Exact risk is a special benefit of this controlled law. A general dataset need
not supply an oracle, and a fitted-risk upper certificate does not require one.
The [adaptation guide](ADAPTING_TO_NEW_DATA.md) explains which task, dependence,
loss, and implementation assumptions must be reconsidered for an application.

## 7. Class-risk assessment on the same evaluation law

The `class-risk` workflow assesses optimal population MSE over prescribed
Lipschitz decoder classes. It uses the same 16 histories and 64 history–target
triples, with independently drawn calibration episodes. It commits its own
class definitions, calibration-size menu, block partitions, numerical sources,
and confidence allocation before generating that calibration sample.

The recorded study uses the following five classes:

| Representation | Coordinates and metric | Lipschitz bound |
| --- | --- | ---: |
| RAW | Archived history features before the readout's internal column scaling; Euclidean distance | $\kappa=2$ |
| KAHM MEM | Archived membership features before readout scaling; Euclidean distance | $\kappa=2$ |
| KAHM LOCAL | Archived membership and local input coordinates before readout scaling; Euclidean distance | $\kappa=2$ |
| Short history | Current observation in its original units; Euclidean distance | $\kappa=4$ |
| Full history | Previous and current observations in their original units; Euclidean distance | $\kappa=4$ |

Targets and decoder outputs lie in $I=[0,1]$, and the MSE threshold is
$\tau=1/10$. The stored feature and fitted-prediction values are
interpreted as exact computational tables on the benchmark support. A fitted
table is checked for class membership separately; a successful check ensures
that some Lipschitz decoder agrees with the table on that support.

The recorded study declares sample sizes $1{,}024$, $4{,}096$, and $16{,}384$,
block sizes $2,8,16,32,64$, and one repetition. Each class/sample-size
assessment includes one fitted-risk upper endpoint, five block lower
endpoints, five empirical-risk lower endpoints using the same block witnesses,
and one full empirical-risk lower endpoint. The resulting 180 allocations are
each $1/3600$, summing to $\delta=0.05$. These allocations are separate from
the three-predictor protocol in Section 5. The default command uses
one calibration size, giving 60 endpoints with allocation $1/1200$ each.

The short-history optimum is $5/32$, whereas the full-history optimum is
$9/160$. The known-law audit also checks RAW/MEM/LOCAL reference quantities
after the sample certificates have been saved and verified. It does not
determine the lower confidence bound from calibration or change an unresolved
decision. The [class-risk guide](LOWER_CERTIFICATES.md) documents the exact
study, commands, and preserved results; the
[proof](LOWER_CERTIFICATE_PROOF.md) derives the hierarchy and reference risks.

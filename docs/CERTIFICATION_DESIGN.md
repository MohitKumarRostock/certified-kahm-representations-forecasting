# Exact calibration losses and verified fitted-risk upper bounds

In version 0.1.0, this stage evaluates the three already frozen computational
predictors under the [calibration protocol](../kahm_demo/calibration_protocol.json)
that was fixed with their freeze.
It does not choose or refit predictors. Its numerical routines are implemented
in `verified_arithmetic.py` and `risk_bounds.py`; the loss ledger is implemented
in `calibration_ledger.py`, and `certificate_workflow.py` binds those calculations
to the actual frozen family and reserved data. The complete statistical proof
and numerical enclosure argument are in
[Fitted-risk guarantee](FITTED_RISK_GUARANTEE.md).

## Statistical object and assumptions

For a fixed development outcome D and each frozen predictor f_m,D, the target is
the population scalar mean squared error

$$
R_m(D)=\mathbb{E}_{(X,Y)\sim P}\big[(Y-f_{m,D}(X))^2\big].
$$

The evaluation law P is the [four-state benchmark](BENCHMARK.md). The
statistical statement assumes 4,096 iid calibration episodes from that law,
independent of all development. There is one history/target pair per episode.
Both targets and the frozen clipped forecasts lie in [0,1], so squared losses
also lie in [0,1]. The same episodes may be shared across all three predictors;
independence between their resulting endpoints is not required.

The implemented pseudorandom sampler and its recorded stream identities make
the numerical experiment reproducible. They do not independently prove the iid
statistical premise. Certificate reports retain this distinction explicitly.

The three-member family is RAW, KAHM MEM, and KAHM LOCAL. The 225 development
fits are selection history. Conditional on development, the three selected
predictors are fixed before calibration; their individual failure allocations
are each 1/60, summing to 1/20.

For every reported valid endpoint, the intended simultaneous statement is

$$
\Pr_{\mathrm{Cal}}\!\left[
  \exists m\text{ with a reported valid endpoint and }R_m(D)>U_m
  \mid D
\right]\le \frac1{20},
$$

under the stated premises. There is no additional conditioning on numerical
availability or acceptance. Every unavailable candidate remains visible in the
report, and its probability allocation is not reassigned.

This is a fitted population-MSE statement. It supplies neither a per-forecast
prediction interval nor an assertion of representation superiority or optimal
risk over a decoder class.

Sections 2–3 of [Fitted-risk guarantee](FITTED_RISK_GUARANTEE.md) derive the
bounded-loss Chernoff inequality and its one-sided KL inverse. Its Section 4
proves the simultaneous statement by a union bound over the fixed family,
including data-dependent numerical availability and acceptance.

## Exact observed losses

The freeze verifier executes the saved models on all 16 supported histories
and establishes their canonical prediction tables. The assessment uses those
verified tables, reading and hashing the same bytes before parsing them.
Prediction caching does not remove repeated episode occurrences.

Every finite binary64 forecast is interpreted as its exact represented rational
number. The targets are the exact observation values 0, 1/4, 3/4, and 1.
Subtraction, squaring, summation, and division by 4,096 are exact rational
operations:

$$
q_m=\frac{1}{4096}\sum_{i=1}^{4096}(y_i-f_m(x_i))^2.
$$

There is no target-variance normalization, decimal approximation, or additional
prediction clipping during evaluation. The canonical clipped predictor was
already fixed before calibration.

The JSONL ledger retains all 4,096 rows. Each row binds its run and episode ID,
three integer state labels, history ID, exact target, three hexadecimal
predictions, and three exact squared losses. Fractions are reduced and stored
as canonical decimal numerator/denominator strings. The verifier independently
recomputes every row and all three means from the verified calibration dataset
and frozen predictions.

## Verified logarithms

The endpoint is the upper inverse of binary KL:

$$
U_m\ge\sup\left\{p\in[q_m,1]:
  \operatorname{kl}(q_m\Vert p)\le\frac{\log60}{4096}\right\},
$$

where zero-weight logarithm terms use their continuous conventions. The
implementation encloses each logarithm with exact integer and rational
arithmetic. For positive rational x, integer range reduction gives

$$
x=2^a u,\qquad 1\le u<2,\qquad z=\frac{u-1}{u+1}.
$$

The atanh series satisfies, for 0<=z<=1/3,

$$
S_M(z)=2\sum_{j=0}^{M-1}\frac{z^{2j+1}}{2j+1},\qquad
0\le\log\frac{1+z}{1-z}-S_M(z)
\le\frac{3}{4(2M+1)9^M}.
$$

At precision b in {128,256,512}, M=b and D=2^b. Exact integer floor and ceiling
operations enclose each power and each term on the D grid. The rigorous
nonnegative tail is added upward. The same series at z=1/3 encloses log(2);
negative a reverses the endpoints of the multiplied log(2) interval.

Only log(2) and log(60) are cached, separately by precision and work-budget
instance. The code checks intermediate integer sizes as well as reduced
fractions. Production logarithms do not use floating-point or Decimal arithmetic.
The tests use separate exact rational and high-precision Decimal references.

## Root search and independent endpoint check

The search retains exact q and brackets the root of

$$
F(p)=4096\operatorname{kl}(q\Vert p)-\log60.
$$

For q<1, it starts at [q,1], whose lower and upper directions are known
analytically. It chooses the exact midpoint, first evaluates at 128 bits, and
tries 256 and 512 bits only if the comparison remains undecided. A nonnegative
lower enclosure for F moves the upper endpoint; a nonpositive upper enclosure
moves the lower endpoint. No bracket endpoint moves on an ambiguous comparison.

At most 80 successful halvings are permitted. A reported endpoint requires
width at most 2^-80, and the endpoint is the exact upper bracket value. The q=1
case has the exact bracket [1,1]. At q=0, the calculation uses -log(1-p), with
the interval endpoints reversed when negating the logarithm enclosure.

The direct checker does not call the search routine. It recomputes the final
endpoint logarithm enclosures with fresh constant caches and verifies the
saved lower/upper inequalities, exact mean, bracket width, and reporting rules.
It also checks the retained trace's structural arithmetic and bookkeeping.
The logarithm backend is shared, so its enclosure correctness is additionally
tested against independent references; this is not a second wholly separate
implementation of every mathematical operation.

The complete family and initial direct checks share the fixed budget of one
million series-term iterations and a 32,768-bit integer limit. Counters are
carried between candidates. A resource failure remains persistent. An observed
integer-size violation is retained diagnostically, even though it exceeds the
allowed cap; it cannot support an available endpoint.

## Reporting and statuses

The exact upper fraction U=A/B is preserved. Its display uses
ceil(10^18 A/B)/10^18, formatted to 18 decimal places. This decimal is rounded
upward. The requirement decision uses the exact integer comparison 10A<=B.

| Status | Meaning |
| --- | --- |
| `requirement_met` | A valid upper endpoint is at most the exact tolerance 1/10, under the declared premises |
| `unresolved` | A valid upper endpoint exceeds 1/10; this does not establish that the actual risk exceeds 1/10 |
| `certificate_unavailable` | The required numerical calculation or verification could not be established; no endpoint is reported |

Unavailable records retain the exact empirical mean when supported, the failure
code and message, available bracket/trace diagnostics, and work counters. The
workflow retains every candidate and does not replace calibration observations,
substitute a looser endpoint, or revise the protocol in response to results.

## Assessment lifecycle and artifact verification

The workflow first verifies the frozen family without reading calibration
outcomes. It commits `assessment_plan.json` and a snapshot of the four certifier
source modules before generating or loading any outcome for that assessment.
The plan binds the freeze, original reservation, exact protocol, implementation,
runtime, ordered candidates, confidence allocation, and tolerance.

If calibration is absent, the existing frozen-model gate materializes the
reserved stream once. If it already exists, the assessment reuses the same
freeze-bound archived data; replay is not additional independent evidence.
Existing assessment output is never overwritten. Unexpected operational failures
retain the committed plan and an `assessment_failure.json` record, and existing
calibration observations remain in place.

The original reservation remains unchanged. Its separate materialization receipt
records the calibration data identity and the exact freeze file identity. This
preserves the reservation and frozen development hashes after calibration.

The completed assessment contains the receipt copy, exact loss ledger, all three
numerical bound records, source snapshot, and `certificate.json`. The latter
binds a complete artifact inventory. Verification checks the current and archived
certifier source identities, the unchanged frozen family and data receipt,
every exact loss, direct endpoint proofs, confidence scope, all statuses, and the
shared generation-work chain. It performs no sampling, model fitting, root search,
or fitted-model population-risk calculation.

Hashes and execution checks detect inconsistent artifacts; they do not
authenticate historical chronology. The later reference-risk audit remains a
separate stage and does not determine the certificate endpoint.

That separate audit stage is described in
[POPULATION_AUDIT.md](POPULATION_AUDIT.md). Its `verify-audit` command deliberately
recomputes model-specific exact benchmark risks and compares them with the
unchanged issued endpoints. This document's `verify-certificate` workflow keeps
its original scope and does not compute those population risks.

## Commands

To assess a previously frozen family in a fresh result directory:

```bash
python -m kahm_demo certify --bundle data/primary --freeze frozen/primary/freeze.json --output certificates/primary
```

To verify an existing assessment without resampling or repeating the root search:

```bash
python -m kahm_demo verify-certificate --certificate certificates/primary/certificate.json --bundle data/primary --freeze frozen/primary/freeze.json
```

The paths above are the primary assessment layout. When those artifacts are
present, use the verification command to inspect them; the creation command
requires a fresh output directory and does not overwrite an assessment.
Repeating numerical evaluation of identical archived calibration data is a
replay of the same evidence, not a new independent calibration experiment.

## Local proof and specification

The mathematical justification is self-contained in
[FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md): Sections 2–4 establish
the fitted-risk confidence statement, Sections 5–6 prove the directed logarithm
and endpoint enclosures, and Sections 7–8 explain unavailable outcomes and the
remaining sampling premises. The concrete precision schedule, work limits,
exact empirical arithmetic, and acceptance rules are fixed in the
[protocol resource](../kahm_demo/calibration_protocol.json) before calibration.
The [method overview](METHOD_OVERVIEW.md) describes the broader demonstrator.

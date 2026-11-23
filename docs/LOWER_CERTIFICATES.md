# Using lower class-risk and upper fitted-risk certificates

The `class-risk` command assesses the optimal population mean squared error
(MSE) over a prescribed Lipschitz decoder class. It computes verified
empirical-risk lower bounds, calibrates them into lower confidence bounds
on optimal class risk, and reports upper confidence bounds on the risk of
fixed fitted predictors. A separate population audit supplies references
after those certificates are issued.

Start with the [scientific reference](SCIENTIFIC_REFERENCE.md) for the
definitions, assumptions, and interpretation. The
[proof note](LOWER_CERTIFICATE_PROOF.md) gives the full mathematical
arguments. This guide covers execution and recorded evidence.

## Run the example

From a complete source checkout, use Python 3.12+ with NumPy 2.x.
The following small run exercises all five classes with fewer blocks and
optimization sweeps than the default study:

```bash
python -m pip install .
python -m kahm_demo class-risk \
  --freeze frozen/primary/freeze.json \
  --output runs/class-risk \
  --sample-sizes 128 \
  --block-sizes 2 8 \
  --solver-sweeps 10 \
  --run-id class-risk-quickstart-v1

python -m kahm_demo verify-class-risk --run runs/class-risk
```

This is an execution example; small samples and a small optimization budget
can leave threshold decisions unresolved. Its five classes and two block
sizes give 30 one-sided endpoints, each allocated $0.05/30=1/600$.
The separate population audit still uses at least 200 proposal sweeps on
the small support.

Use a new output directory for each execution. The command refuses to
overwrite or resume an existing run. It also refuses to write inside an input
scientific bundle. A failure leaves the partial evidence and a failure record;
it does not replace calibration observations or retry with a different seed.

To run the default assessment instead, use a different output directory:

```bash
python -m kahm_demo class-risk \
  --freeze frozen/primary/freeze.json \
  --output runs/class-risk-default
python -m kahm_demo verify-class-risk --run runs/class-risk-default
```

Its defaults are:

| Choice | Default |
| --- | --- |
| Archived KAHM candidates | RAW, MEM, LOCAL |
| Metric | Euclidean distance before the readout's internal column scaling |
| Lipschitz bound for those candidates | $\kappa=2$ |
| Analytical controls | Short history and full history, original observation coordinates, $\kappa=4$ |
| Calibration episodes | $n=4096$ |
| Block sizes | $2,8,16,32,64$ |
| Repetitions | 1 |
| MSE threshold | $\tau=0.1$ |
| Total failure probability | $\delta=0.05$ |
| Calibration optimization proposal budget | 100 coordinate sweeps per problem |
| Master seed / run ID | `20261119` / `class-risk-v1` |

There are five classes. Each class receives one fitted-risk upper endpoint,
one full empirical-optimum lower endpoint, and two lower endpoints for each
of the five block sizes. The default family therefore contains **60
endpoints**, each allocated failure probability **$1/1200$**. The simultaneous
guarantee covers the entire declared family, conditional on development and
under the stated iid calibration assumptions.

The class-risk assessment uses its own calibration sample and confidence
allocation, separate from the three fitted-risk upper endpoints issued by
`certify`.

## What is assessed

The source freeze identifies archived computational feature and prediction
tables on all 16 possible histories. The command verifies the hashes that bind
those development tables, copies the required inputs, and freezes the new
class specification before drawing calibration episodes. It does **not**
replay the executable model in the assessment runtime.

For the prescribed representation $\Gamma$, feature pseudometric $d$, and
Lipschitz bound $\kappa$, let $\mathcal G_\kappa(d)$ contain all measurable
$\kappa$-Lipschitz decoders into $I=[0,1]$. Their optimal class risk is

$$
R_\kappa^*
=\inf_{g\in\mathcal G_\kappa(d)}
\mathbb E[(Y-g(\Gamma(X)))^2].
$$

The workflow constructs a lower confidence bound $L$ on this infimum and an
upper confidence bound $U_{\mathrm{fit}}$ on the archived fitted decoder's
risk $R(\widehat g)=\mathbb E[(Y-\widehat g(\Gamma(X)))^2]$. The combined statement

$$
L\le R_\kappa^*\le R(\widehat g)\le U_{\mathrm{fit}}
$$

requires the fitted prediction table to satisfy the declared class constraints.
All 120 pairs of supported histories are checked using exact squared-distance
inequalities. A passing table has a matching Lipschitz extension; this check
does not establish the global Lipschitz bound of the executable nonlinear readout on
arbitrary feature inputs.

| Reported decision | Meaning |
| --- | --- |
| `class_excluded` | $L>\tau$: every decoder in the specified class fails the requirement. |
| `requirement_met` | An admissible fitted table has $U_{\rm fit}\le\tau$. |
| `unresolved` | Neither conclusion has been established. |
| `conflicting_certificates` | The issued lower and admissible upper conclusions conflict; the result is retained for inspection. |

Individual numerical endpoints can be `certificate_unavailable`. An
unavailable lower endpoint contributes no information beyond zero, and an
unavailable upper endpoint cannot establish acceptance. Allocations are not
reassigned after numerical failure. Numerical verification and a completed run
do not imply a favorable decision or realized statistical coverage.

## Two lower bounds from the same block optimizations

Each block minimizes total squared error under joint Lipschitz constraints.
A floating-point procedure proposes signed dual multipliers. A separate exact
checker evaluates a conservative lower bound on the objective. The optimizer's
termination status is not the mathematical certificate.

For block size $b$, let $\underline Q_{b,k}$ denote its verified lower SSE,
$K_b=\lfloor n/b\rfloor$, and

$$
\underline T_{b,k}=\frac{\underline Q_{b,k}}{b-1},
\qquad
C_b=\frac{\lfloor b^2/4\rfloor}{b(b-1)}.
$$

The **block lower confidence bound** calibrates the average normalized
score using the $K_b$ independent true block scores in $[0,C_b]$. The
**empirical-risk lower confidence bound** calibrates
$d_{n,b}=n^{-1}\sum_k\underline Q_{b,k}\le a_n$, where $a_n$ is the
minimum empirical MSE on all $n$ observations. It uses the full episode
count $n$ by comparison with each fixed decoder's iid losses; it does not
assume that the optimized value $d_{n,b}$ is an iid average.
Unused observations remain in its denominator. Both calculations use
exactly the same optimization witnesses.

A separate optimization of all calibration observations gives a full
empirical-optimum lower bound. Repeated histories are aggregated with their
counts and label sums, so this benchmark requires at most 16 prediction
variables even when $n$ is large. Every sampled occurrence retains its weight.
The reported combined lower endpoint is the maximum of the declared valid
lower endpoints and zero.

Larger blocks can increase the expected score $\mathcal W_b$ while reducing
the number of independent block scores at fixed $n$. A reported lower
confidence bound need not increase with block size, and neither calibration
route always dominates the other. This small benchmark also makes full
empirical optimization inexpensive; it does not by itself establish a
computational advantage for blocking on large feature spaces.

The complete mathematical argument is in
[Lower-certificate proof](LOWER_CERTIFICATE_PROOF.md).

## Diagnostic experiments

### Short and full histories

The short control keeps only the current observation. The full control keeps
the previous and current observations. Their predictors use conditional sample
means from the source training data, with the training grand mean as fallback;
they do not use population conditional means during certificate construction.

At $\kappa=4$ in the original coordinates, the optimal class risks are

$$
R^*_{\rm short}=5/32=0.15625,
\qquad
R^*_{\rm full}=9/160=0.05625.
$$

For the short control the exact population block score is

$$
\mathcal W_b=\frac5{32}\frac{b-4+4(3/4)^b}{b-1}.
$$

Thus $\mathcal W_2=0.0390625$ and $\mathcal W_8\approx0.0982244$ lie below
the threshold, whereas $\mathcal W_{16}\approx0.1254176$ lies above it. This is
an analytical reference, not a promise that a finite-sample endpoint at $b=16$
will exceed the threshold. The experiment measures how calibration size and
block size affect that decision.

### RAW, MEM, and LOCAL geometry

For the archived `cap_2` features, exact pair checks establish that RAW and
LOCAL can attain risk $0.05625$ at $\kappa=2$. A disjoint-pair calculation gives
the MEM class a population lower bound of $0.118052$. The population optimizer
can tighten this reference through a verified interval.

The MEM fitted predictor in the recorded study has low risk but fails the $\kappa=2$
admissibility check in these coordinates. Its fitted-risk upper endpoint
remains meaningful for that predictor; it is not used as an upper bound on the
constrained class optimum. All 16 archived MEM vectors are distinct, so this
result concerns the prescribed Lipschitz decoder class and feature geometry.
It does not establish unrestricted information loss.

These reference facts are checked by the population audit after sample
certificates have been saved and verified. The audit does not choose a class,
readout, calibration seed, or endpoint.

## Validation study

The [recorded study](../reports/lower_certificates_v1/report.md) uses sample
sizes 1,024, 4,096, and 16,384, the five default block sizes,
$\kappa=2$ for RAW/MEM/LOCAL, $\kappa=4$ for the history controls, and
one repetition. Its family contains 180 endpoints, each with failure
allocation $1/3600$. All endpoints were numerically available.

At $n=4096$, the MEM class has combined lower endpoint approximately
**0.11470296**, exceeding the 0.1 MSE threshold. RAW and LOCAL have admissible
upper endpoints approximately **0.07227278**. At $n=1024$, MEM is
unresolved; the result is included in the study record.

For the short-history control at $n=16384$, the block lower endpoint rises
from approximately **0.03595112** at $b=2$ to **0.10959758** at $b=16$.
The latter establishes class exclusion using the block construction alone.
Its values at $b=32$ and $b=64$ are approximately **0.11917561** and
**0.11762887**, respectively: increasing the population hierarchy does not
force the finite-sample lower endpoint to increase.

These are observations from one declared stream and its shared prefixes, not
estimates of decision probabilities across independent replications. The
saved JSON fractions and numerical witnesses are authoritative. The
[figures](../reports/lower_certificates_v1/figures/) show the class comparison,
hierarchy, and calibration-size tradeoff.

## Change the declared study

For a larger study of calibration size and the Lipschitz bound, declare the
entire menu in one command. This example deliberately requests many more
assessments than the quickstart:

```bash
python -m kahm_demo class-risk \
  --freeze frozen/primary/freeze.json \
  --output runs/class-risk-grid \
  --sample-sizes 256 1024 4096 \
  --block-sizes 2 8 16 32 64 \
  --kappas 1 2 4 \
  --repetitions 5 \
  --solver-sweeps 100 \
  --run-id class-risk-grid-v1
```

Sample sizes, block sizes, and Lipschitz bounds must be unique and increasing;
every block must fit the smallest sample size. The input limits are
`n <= 1000000` and `b <= 2048`; the block limit keeps exact analytical
hierarchy references within the declared serialization limits. Fractions such as `1/2` are
accepted for Lipschitz bounds, `--delta`, and `--tolerance`.

Within a repetition, sample sizes use prefixes of the same reserved stream.
The joint confidence allocation includes every requested sample size,
repetition, class, and bound. With $J$ Lipschitz bounds, $S$ sample sizes, $R$
repetitions, and $H$ block sizes, its endpoint count is

$$
N_{\rm endpoints}=(3J+2)SR(2H+2).
$$

More comparisons therefore reduce the allocation per endpoint. Different
stream labels and pseudorandom seeds make execution reproducible; they do not
prove the iid or development-independence premises. Reusing a seed/run ID is
not a new independent replication.

## Recorded evidence and verification

| File or directory | Contents |
| --- | --- |
| `class_risk_plan.json` | Frozen menu, sample streams, confidence allocation, source and input identities |
| `classes.json` | Exact coordinates, metrics, Lipschitz bounds, and prediction tables |
| `inputs/`, `sources/` | Copied development inputs and assessment sources |
| `calibration/` | All sampled episodes, including repeats |
| `certificates/` | Lower/upper endpoints, admissibility records, and block/full optimization witnesses |
| `audits/` | Known-law class-optimum intervals, fitted risks, and analytical hierarchy references |
| `summary.json`, `report.md` | Recomputed assessment results and a readable summary |
| `failure.json`, if present | A recorded execution failure without replacement sampling |

`verify-class-risk` checks source identities, reconstructs classes and exact
losses, verifies saved dual/primal and KL witnesses, and checks the population
audits. It replays each declared deterministic calibration stream **in memory**
to compare it with the saved observations. It does not fit models, optimize new
witnesses, write replacement observations, or search for new endpoints.

Verification requires assessment sources matching those saved with the run.
An assessment must use sources matching its recorded identity. Reading the
archived computational tables does not require the runtime used to fit them;
this operation does not establish executable model replay.

## Export the figures

Install the existing plotting extra and verify the run before rendering:

```bash
python -m pip install -e '.[tutorial]'
python -m kahm_demo verify-class-risk --run runs/class-risk
python examples/plot_class_risk.py \
  --run runs/class-risk \
  --output runs/class-risk-figures
```

The script writes PNG and vector PDF figures into a new directory outside the
run. It plots class-risk reference intervals, lower and upper certificates,
fitted risks and admissibility, and the short/full block hierarchy. When the
run includes several sample sizes, it also writes a sample-size comparison.
By default it selects the largest sample size and repetition zero for the
first two figures; use `--sample-size` and `--repetition` to select another
declared assessment. Exported metadata identifies that selection and the input
hashes.

Plotting reads recorded values and does not verify them or perform another
experiment. Decimal coordinates in figures are descriptive approximations;
exact JSON fractions govern all decisions.

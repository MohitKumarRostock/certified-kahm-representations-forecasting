# Exact benchmark population-risk audit

In version 0.1.0, the population audit evaluates the three already frozen computational predictors
against the stipulated finite benchmark law, after the fitted-risk certificate has
been issued and verified. The scientific contract explicitly requires
this independent reference audit and retention of every realized coverage
outcome. Its implementation is `population_audit.py` plus `audit_workflow.py`.

## 1. Three different quantities

For each frozen predictor, the report keeps three quantities separate:

| Quantity | Meaning | Evidence |
| --- | --- | --- |
| Calibration MSE q | Average loss on the one archived 4,096-episode calibration sample | Existing exact loss ledger |
| Population MSE R | Expected loss under the stipulated benchmark law | Exact weighted sum over all 64 history-target combinations |
| Issued upper bound U | The conservative endpoint produced by the fixed certification procedure | Existing verified numerical certificate |

The audit calculates R and compares it with the already issued U. The original
q, U, certificate status, tolerance, and probability allocation remain unchanged.
All three selected predictors remain in their original order. No new winner is
selected and no observation or endpoint is replaced.

## 2. Exact finite-support expectation

The [benchmark specification and derivation](BENCHMARK.md) define the law.
Let z=(0,1/4,3/4,1), with history labels a,b and successor label c in {0,1,2,3}.
Each history has probability 1/16. The joint probabilities are

$$
p(a,b,c)=\begin{cases}
17/320,&c=(a+b)\bmod4,\\
1/320,&\text{otherwise}.
\end{cases}
$$

For the actual frozen binary64 prediction f(a,b), the exact population MSE is

$$
R(f)=\sum_{a=0}^3\sum_{b=0}^3\sum_{c=0}^3
 p(a,b,c)\,[z_c-f(a,b)]^2.
$$

Every stored float is converted to its exact represented rational value. The
audit performs no decimal rationalization, extra clipping, variance normalization,
or replacement of a rounded prediction by an ideal mathematical value. Exact
fractions are canonical numerator/denominator strings. Approximate population
and excess-risk displays round to nearest, ties to even, at 18 decimal places.
The issued U display is copied unchanged from its upward-rounded certificate record.
All comparisons use exact fractions.

Each candidate's population record retains all 64 weighted loss contributions
and all 16 history records. The weights sum exactly to one; each history has
exact mass 1/16. The records are indexed canonically by `history_id=4*a+b` and
`triple_id=4*history_id+c`.

## 3. Independent calculation and decomposition

The independent enumerator writes the declared probabilities explicitly and checks them
against the fixed scientific contract. It does not call the law module's
probability or conditional-mean helpers. Its exact result must agree with the
unchanged `oracle.exact_risk` implementation, which follows that separate helper
path. Both implementations use Python's exact `Fraction` arithmetic.

The conditional mean for a history is

$$
\mu(a,b)=\frac1{10}+\frac45z_{(a+b)\bmod4}.
$$

The audit also checks the per-history conditional variance and the identity

$$
R(f)=\underbrace{\frac9{160}}_{R_{\mathrm{Bayes}}}
 +\frac1{16}\sum_{a,b}[f(a,b)-\mu(a,b)]^2.
$$

The second term is the **fitted predictor's excess risk over the full-history
Bayes risk**. It is not a measurement of information discarded by a representation,
and it does not establish the best risk achievable within a decoder class.
Estimating either of those different objects would require additional analysis.

The independent enumeration also verifies the analytical references:

| Reference | Exact population MSE |
| --- | --- |
| Full-history conditional mean | 9/160 = 0.05625 |
| Best constant prediction 1/2 | 5/32 = 0.15625 |
| Nominal successor's observation | 1/16 = 0.0625 |

These calculations are checks on the declared benchmark law. They are not
learned candidates or additional calibrated models.

## 4. Retaining all outcomes

Each candidate retains its original `certificate_status` and receives separate
audit fields:

| Field | Interpretation |
| --- | --- |
| `population_requirement_met` | Exact R <= 1/10 under the stipulated finite law |
| `realized_coverage = covered` | An issued valid endpoint exists and R <= U |
| `realized_coverage = coverage_miss` | An issued valid endpoint exists and R > U; the original endpoint and miss are retained |
| `realized_coverage = endpoint_unavailable` | No issued endpoint exists; exact R is still reported |
| `upper_minus_population_risk` | Exact realized certificate slack U-R, or null when U is unavailable |
| `empirical_minus_population_risk` | Exact realized sampling deviation q-R |
| `upper_minus_empirical_risk` | Exact certificate margin U-q, or null when U is unavailable |

A coverage miss is a completed audit outcome. It is not a reason to change U,
resample calibration, refit a predictor, select another seed, or filter the run.
Operational or arithmetic inconsistencies instead stop the audit. Failures after
the audit plan is committed leave an `audit_failure.json` record; prerequisite
failures stop before creating an audit bundle. The original data and certificates
remain intact.

The family summary separately reports available, covered, missed, and unavailable
endpoint counts. `all_available_endpoints_cover` is null if there are no available
endpoints. `all_three_covered` is true only when three endpoints are available
and each covers its exact risk. This avoids treating an empty set of bounds as
successful three-model coverage.

The sample mean q may be below the Bayes population risk. One finite-sample
deviation is not a contradiction or an estimate of systematic bias. Similarly,
U-R includes both realized sampling deviation and confidence padding; it is not
merely numerical rounding error.

## 5. What the audit does and does not establish

Exact risk under the stipulated finite law is an analytical property of the
specified frozen predictor. It does not require a new calibration sample. The
original statistical certificate still has its conditional iid premises.
Neither exact arithmetic nor PRNG stream identities establish those premises
for observations from an application.

This audit checks the coverage inequalities for one realized family and its
already issued endpoints. It does not estimate a long-run 95% coverage frequency,
upgrade the original confidence level, or issue a new confidence guarantee.
Any ordering of these three exact benchmark risks is specific to these fitted
predictors and this stipulated law, and establishes no general representation
superiority.

The computational predictor remains the object being assessed. The audit does
not establish equivalence to an ideal real-arithmetic KAHM function, global
injectivity, Markov sufficiency, Koopman closure, or recursive rollout accuracy.

## 6. Artifact lifecycle and verification

The workflow first invokes the complete certificate verifier. It then
reads and hashes the same bytes of the verified certificate, freeze, and frozen
prediction tables before using them. An audit plan and a snapshot of the two
audit modules are committed before any model-specific exact-risk calculation
in that audit. The plan binds the existing certificate and freeze hashes,
unchanged contract and protocol, runtime, all three candidates, and scope.

The two audit modules form a third source-provenance layer. They do not alter
the frozen prediction dependencies or certifier source archive. Existing
audit output directories cannot be overwritten, and audit evidence must remain
outside the data, frozen-model, and certificate bundles.

`verify-audit` rechecks the existing certificate and audit source/runtime plan
before reading the saved population-risk records. It then recomputes each exact
expectation, independent decomposition, and comparison. Rehashed semantic
tampering is rejected because numerical and interpretation fields must reproduce,
not merely match an editable file hash. These checks do not authenticate execution
chronology or make editable directories tamper-proof.

| Path under an audit directory | Contents |
| --- | --- |
| `audit_plan.json` | Fixed audit scope, prerequisites, implementation identities, and runtime |
| `sources/` | Two audit source modules |
| `population/{raw,kahm_mem,kahm_local}.json` | Exact 64-triple expectations, 16-history decompositions, and cross-checks |
| `audit.json` | Complete family comparison, retained outcomes, references, and artifact inventory |

## 7. Commands

When a primary audit exists at the standard paths, verify it with:

```bash
python -m kahm_demo verify-audit --audit audits/primary/audit.json --certificate certificates/primary/certificate.json --bundle data/primary --freeze frozen/primary/freeze.json
python -m examples.population_audit_walkthrough
```

For a different already assessed family, create an audit in a fresh directory:

```bash
python -m kahm_demo audit-certificate --certificate certificates/replay/certificate.json --bundle data/replay --freeze frozen/replay/freeze.json --output audits/replay
```

Repeating an exact calculation supplies no new independent statistical evidence.
The `verify-certificate` command checks the issued certificate without
evaluating model-specific population risks.

## Local mathematical basis

The [scientific contract](../kahm_demo/specification.json), especially
`reference_audit`, specifies the exact finite-law evaluation, excluded uses, and
coverage-miss retention rule. [BENCHMARK.md](BENCHMARK.md) derives the law and
analytical reference risks; the audit explicitly checks the conditional
squared-loss decomposition. The separate statistical guarantee is proved in
[FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md) and implemented as described
in [CERTIFICATION_DESIGN.md](CERTIFICATION_DESIGN.md). The audit adds an exact
reference comparison to an already issued certificate and leaves its
mathematical scope unchanged.

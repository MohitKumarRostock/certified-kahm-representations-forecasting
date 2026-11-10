# Adapting the demonstrator to a new forecasting problem

This package is a complete **controlled four-state benchmark**, with a fixed
two-observation input, scalar one-step target, three selected predictors, and
a known finite evaluation law. It does not provide a generic CSV, pandas,
time-series, or dataset adapter. `reproduce` reruns this benchmark with fresh
artifacts; it does not import a new problem.

The Gaussian KAHM and readout components are useful starting points for another
application. Reusing them requires a new task specification, data interface,
evaluation design, numerical provenance, and justified certificate. Preserve
the supplied archive as an example and create a separately identified
experiment. The fixed hashes and strict checks deliberately prevent silently
relabeling the existing certificate as a certificate for new data.

## 1. Specify the object before adapting the code

Write a new scientific contract that answers these questions:

| Decision | What to record |
| --- | --- |
| Forecast input | Available variables, history length, timestamps, missing-value treatment, and what is known at forecast time |
| Target | Observation definition, forecast horizon, units, measurement representation, and any transformation |
| Evaluation population | Which future subjects, machines, episodes, locations, or time periods the expected loss concerns |
| Evaluation unit | The independent unit, or an explicit dependence model with a suitable theorem |
| Loss and tolerance | A defined loss for every admissible outcome, its justified range or other assumptions, and a tolerance in declared units |
| Development protocol | Training and validation roles, geometry/readout menus, baselines, selection rules, and tie handling |
| Final family | Exactly which frozen predictors and reported horizons receive coverage, with a fixed failure-probability allocation |
| Numerical object | The implemented prediction rule, precision, clipping, failures/abstentions, and source/runtime identity |

A fitted-risk certificate can target the actual implemented predictor. It does
not require proving full-history injectivity, predictive sufficiency, or exact
Koopman closure of the encoding. Those are separate mathematical claims. Ridge
regularization is also a fitting penalty, not a verified hard Lipschitz bound
on the rounded decoder. The current example certifies fitted scalar MSE; it
does not certify an optimum over a representation or decoder class.

Use [BENCHMARK.md](BENCHMARK.md) and the
[scientific contract](../kahm_demo/specification.json) as a worked specification,
not as a configuration file whose constants can be changed without revising
the associated implementation and evidence.

## 2. Preserve the data-role boundaries

Fit preprocessing and KAHM geometry using training histories. Fit the decoder
with training targets. Use validation outcomes to choose geometry, features,
hyperparameters, and the final family. Complete these choices before accessing
the reserved final calibration outcomes. Freeze the complete prediction rule,
loss, confidence allocation, numerical protocol, and reporting rules first.

An application data adapter should preserve stable sample identities and the
information needed to audit splitting. Useful fields include an episode or
subject identifier, forecast origin, ordered history values, target values,
and the declared role. This is a proposed application schema, not a currently
implemented `Dataset` interface. The existing `Episode` contains three integer
states and is specific to the four-state generator.

Do not deduplicate repeated observations out of training or calibration losses.
The current code deduplicates only selected reference/landmark identities and
caches repeated inference; every episode still contributes its full weight.
For real data, establish whether repeated rows are independent observations,
repeated measurements of one unit, or duplicate records before assigning that
weight. Preserve exclusion rules and missingness decisions made during design.

### Trajectories require a justified evaluation design

The current 4,096 calibration examples are stipulated independent episodes,
each starting from a newly drawn history pair. They are not 4,096 overlapping
windows cut from one trajectory. A chronological split alone does not make
windows within the calibration segment iid; overlap and temporal dependence
remain.

One possible design uses independently sampled episodes or subjects and a
predeclared bounded loss per independent unit. If each unit contributes the
average of several forecast losses, the certified risk is the expectation of
that unit-level loss under the declared sampling law. The number of independent
units, rather than the total number of windows, governs an iid certificate.
Independence from development and the intended population still need to be
justified.

Another design may use a theorem for dependent observations with explicit
mixing, blocking, martingale, or other applicable assumptions. Such a design
requires its own bound and implementation. Adding gaps, shuffling windows, or
changing random seeds does not automatically establish those assumptions. The
current iid KL protocol must not be applied to arbitrary trajectories by
simply relabeling their windows as episodes.

## 3. Adapt the numerical components and workflow together

Some lower-level components accept general dense arrays, but the end-to-end
workflow includes deliberate finite-support assumptions. The following map
identifies the places an application implementation must address.

| Concern | Current modules/resources | Adaptation work |
| --- | --- | --- |
| Task and protocol identity | `specification.json`, `contract.py`, `calibration_protocol.json`, `calibration_protocol.py` | Define a new versioned contract and protocol. Validate the new facts explicitly and retain the original archive. |
| Observations and splits | `law.py`, `sampling.py`, `datasets.py` | Replace the four-state episode schema/generator with an audited source and split interface. Bind data, role, sampling design, target representation, and counts. |
| Training-only geometry | `geometry.py`, `kahm.py`, `soft_state.py`, `encoder.py`, `encoder_workflow.py` | Fit the new input dimensions and training-only geometry; specify partitions, reference budgets, PCA choices, score parameters, and every numerical fallback. |
| Matched decoders and selection | `readout_features.py`, `ridge.py`, `selection.py`, `forecast.py`, `forecast_workflow.py`, `readout_specification.json` | Choose development-only landmark and hyperparameter budgets, generalize support-specific evaluation, and retain complete selection provenance. |
| Saved prediction rule | `encoder_state.py`, `forecast_state.py`, `freeze.py` | Bind preprocessing, state, inference semantics, source/runtime, and evaluation policy. Replace the assumption that 16 history outputs fully enumerate the input support. |
| Calibration loss evidence | `calibration_ledger.py`, `certificate_workflow.py` | Generalize the fixed 4,096-episode/16-history interface; evaluate actual frozen predictions for every calibration occurrence and retain exact or verified loss arithmetic. |
| Statistical endpoint | `risk_bounds.py`, `verified_arithmetic.py` | Use the new justified theorem, sample size, family allocation, loss scale, and numerical work policy; independently verify reported endpoints. |
| Population audit and reports | `oracle.py`, `population_audit.py`, `audit_workflow.py` | Use a population oracle only when the new probability law and support genuinely permit it. Otherwise report what is observable without inventing exact population risk. |

The current endpoint protocol fixes `n=4096`, three candidates, per-candidate
failure probability `1/60`, and tolerance `1/10`. These are not generic command
line certificate parameters. Changing a number in a report cannot adapt the
mathematical or numerical procedure. A generalized implementation needs a
matching validated protocol and tests for its new parameter domain.

### Keep the MEM/LOCAL comparison identifiable

For a direct membership-versus-enrichment comparison, fit one common KAHM bank
for the two candidates. Share training preprocessing, partitions, reference
identities, PCA, Gaussian models, reconstruction scores, and membership rules.
Use

$$
\Gamma_{\mathrm{MEM}}(x)=\phi(z),\qquad
\Gamma_{\mathrm{LOCAL}}(x)=\operatorname{col}\left(
\phi(z),\{\phi_j(z)P_j^T(z-\mu_j)\}_j\right).
$$

The appended coordinates describe the input, and the MEM prefix is identical.
Each partition has its own normalization; membership weights are reconstruction
affinities, not calibrated regime probabilities. Include the RAW-history
baseline, preserve comparable data roles and readout procedures, and document
feature widths, kernel ranks, and search budgets. Shared readouts do not imply
equal capacity, and the current 45/90/90 search does not imply equal total
effort.

Changing the number of references, PCA caps, landmark budget, bandwidth grid,
decoder family, or bank-selection rule is a development decision. Choose and
validate it before final calibration. The current 16-landmark choice exploits
a tiny benchmark support; it is not a general recommendation for application
scale. [ENCODER_DESIGN.md](ENCODER_DESIGN.md) and
[READOUT_DESIGN.md](READOUT_DESIGN.md) document the implemented construction and
its fixed numerical and development choices.

## 4. Re-derive the loss scale and certified target

The example relies on targets and clipped predictions in `[0,1]`, so squared
loss is bounded by one. If a new problem has a justified target range `[A,B]`
with `A < B`, a possible normalized loss is

$$
\ell(x,y)=\left(\frac{y-\operatorname{clip}(f(x),A,B)}{B-A}\right)^2
\in[0,1].
$$

A bound on this normalized risk converts to a bound on squared-error risk in
the original units by multiplication by `(B-A)^2`. Record both the clipping
policy and tolerance units before calibration. A range estimated only from
training extrema is not automatically a bound on future targets. Clipping the
prediction alone does not bound squared loss when targets can be unbounded.

For unbounded outcomes, choose and justify an appropriate different assumption
or a predeclared bounded loss. A capped loss defines a different risk from
ordinary MSE. Additional horizons, vector outputs, weighting, target-variance
normalization, or other loss changes likewise require an explicit new target
and bound. Exact arithmetic on stored observations concerns those represented
measurements; it does not resolve uncertainty about an unobserved physical
quantity.

For multiple horizons, either predeclare one bounded aggregate loss or allocate
the failure budget across every separately reported predictor/horizon claim.
Three predictors at each of `H` horizons can create `3H` reported risks. A
recursive latent rollout is another prediction rule with its own errors and
loss; the current one-step certificate does not certify such a rollout.

## 5. Freeze the actual executable predictor

Specify the entire path from raw input to final output: preprocessing, encoder,
readout feature map, coefficients, accumulation order, numeric types, clipping,
and handling of invalid inputs or numerical failures. A fallback or abstention
policy must be part of the frozen rule and loss definition. Silently dropping
hard calibration examples changes the evaluated object.

The current package stores arrays and outputs in lossless hexadecimal form and
checks canonical row inference, including batch equivalence. Its strict runtime
identity includes the full Python build, NumPy version, NumPy/BLAS configuration,
machine information, and source bytes. Version pins alone do not recreate that
identity. A different implementation or runtime should create its own
development artifacts and freeze, followed by its reserved assessment.

On continuous or large input support, exhaustive comparison of 16 predictions
is no longer sufficient. Preserve an executable state and provenance, verify
the actual calibration predictions, and test the new inference interface and
failure policy. Do not infer global equivalence of two implementations from a
small agreement sample. A fitted-risk certificate for an explicitly defined
computational predictor remains conceptually separate from a verified error
bound relative to an ideal real-arithmetic Gaussian KAHM map.

## 6. Protect the final assessment from repeated development

Changing the fitted rule after viewing calibration outcomes turns those
outcomes into development information. A new assessment then needs untouched
evaluation data and an appropriate policy for repeated attempts. The three-way
allocation in this example does not automatically cover an open-ended sequence
of new models, seeds, datasets, or tolerances.

Plan the final family and confidence budget in advance, or use a justified
sequential/multiple-attempt procedure. Keep unavailable numerical results,
unresolved requirements, and coverage misses visible. Do not replace difficult
calibration observations, reassign an unavailable candidate's allocation, or
report only runs with favorable bounds. Rerunning the same observations and
saved model supplies a reproducibility check, not another independent sample.

Equivalent code or a conservative recalculation can sometimes reuse archived
calibration evidence, but equivalence or conservativeness must be established
for the actual assessed object. This is a mathematical and numerical argument,
not permission to bypass source/runtime checks or rewrite the historical freeze.

## 7. Report the evidence available for the new problem

The benchmark has known rational probabilities, so its exact population risk
can be calculated after certification. Ordinary application datasets usually
do not supply that oracle. Keep the quantities distinct:

| Evidence | What can be reported |
| --- | --- |
| Reserved sample with a valid evaluation design | Observed loss `q`, with sample identity and every included occurrence |
| Applicable theorem and verified numerical endpoint | Fitted-risk upper endpoint `U`, its confidence scope, assumptions, and status |
| Fully known finite law and evaluated support | Exact population risk `R` and the realized comparison with `U` |
| Additional held-out sample | Another empirical estimate under its stated sampling design, rather than exact `R` |

A fitted-risk upper certificate does not require an exact population oracle.
When that oracle is unavailable, leave the exact-audit field unavailable and
explain why. A real-data test mean must not be labeled exact population risk,
and one experiment must not be presented as an empirical measurement of 95%
coverage frequency.

An application deliverable should include the new contract, data-role and
sampling records, complete implementation, development settings and selection
history, frozen executable state, runtime/source identities, calibration loss
evidence, endpoint checks, all candidate statuses, and a reproduction guide.
Report the performance of those fitted procedures. Claims about general
representation superiority, sufficiency, or decoder-class limits require
separate evidence.

## Local methods and guarantees

The current task and sampling law are defined in [BENCHMARK.md](BENCHMARK.md).
The construction is documented in [ENCODER_DESIGN.md](ENCODER_DESIGN.md) and
[READOUT_DESIGN.md](READOUT_DESIGN.md). The bounded iid loss argument and its
numerical direction are derived in
[FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md); the implemented assessment
protocol is detailed in [CERTIFICATION_DESIGN.md](CERTIFICATION_DESIGN.md).
The known-law audit and its scope are in [POPULATION_AUDIT.md](POPULATION_AUDIT.md).
[METHOD_OVERVIEW.md](METHOD_OVERVIEW.md) connects these layers to the code.

The adaptation decisions above are a proposed engineering and evaluation path.
The current package does not implement a new-data adapter, and its statistical
argument does not apply to a new dataset without its assumptions being
established for that dataset and prediction rule.

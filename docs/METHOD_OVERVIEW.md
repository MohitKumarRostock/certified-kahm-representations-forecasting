# Method and implementation overview

Based on our research, this project combines a geometric representation with a
complete forecasting and assessment workflow. This guide maps its mathematical
objects to local definitions and executable modules. The repository defines
the method and its guarantee through the local guides and executable code.

## 1. Separate the objects being constructed

| Layer | Inputs and output | Local definition | Main modules |
| --- | --- | --- | --- |
| Benchmark | Independently sampled two-observation histories and scalar next targets | [Benchmark](BENCHMARK.md) | `law.py`, `sampling.py`, `datasets.py` |
| Geometry | Training inputs to shared reference partitions, local PCA, and Gaussian KAHMs | [Encoder design](ENCODER_DESIGN.md), sections 2–4 | `geometry.py`, `kahm.py`, `encoder_workflow.py` |
| Soft-state representation | Reconstruction scores to MEM or membership-weighted input coordinates in LOCAL | [Encoder design](ENCODER_DESIGN.md), section 5 | `soft_state.py`, `encoder.py` |
| Readout and selection | Training targets and validation losses to three selected clipped predictors | [Readout design](READOUT_DESIGN.md) | `readout_features.py`, `ridge.py`, `selection.py`, `forecast_workflow.py` |
| Frozen predictor | Complete saved state to canonical predictions and bound source/runtime identities | [Freeze interface](FREEZE_INTERFACE.md) | `forecast_state.py`, `freeze.py` |
| Risk assessment | Reserved calibration losses to checked upper endpoints | [Risk guarantee](FITTED_RISK_GUARANTEE.md), [certification](CERTIFICATION_DESIGN.md) | `calibration_ledger.py`, `verified_arithmetic.py`, `risk_bounds.py`, `certificate_workflow.py` |
| Population audit | Known finite law and frozen predictions to exact risks and retained comparisons | [Population audit](POPULATION_AUDIT.md) | `population_audit.py`, `audit_workflow.py` |

The encoder is fitted without forecasting targets. A readout makes the complete
forecasting model. The certificate then assesses that actual fitted model under
the stated law. A useful geometric description alone does not establish its
prediction error.

## 2. Gaussian KAHM features

A local model reconstructs a standardized history using a regularized Gaussian
kernel over reference histories. Its normalized affine weights can be signed.
The implementation defines cancellation checks and an affine fallback; these
choices are part of the computational model.

Each reconstruction receives a radial/angular discrepancy score. Affinities
are normalized separately within each partition. Their concatenation is MEM.
LOCAL retains that exact membership prefix and appends

$$
\phi_j(z)P_j^T(z-\mu_j)
$$

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

## 4. Empirical error, upper endpoint, and population risk

| Quantity | Definition | What establishes it |
| --- | --- | --- |
| `q` | Mean loss on the reserved calibration sample | Every observed occurrence and exact represented prediction |
| `U` | Conservative statistical upper endpoint for the fitted population risk | The stated sampling assumptions, bounded-loss argument, and verified endpoint arithmetic |
| `R` | Exact expectation under this known benchmark law | All 64 weighted history-target cases, evaluated after certification |

The certificate conditions on development: the selected predictors are fixed
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

## 5. Computational and scientific scope

Saved numerical arrays and canonical inference define the predictor. Residual
checks, state replay, and source/runtime matching are engineering checks. They
are distinct from a verified error enclosure around an ideal real-arithmetic
Gaussian function. Certification concerns the represented predictions and the
stated evaluation law.

The workflow establishes no representation-class optimum, full-history
injectivity, predictive sufficiency, Markov property, Koopman closure, certified
latent rollout, or general superiority. A fitted-risk upper bound does not
require those claims. A new application needs its own task, loss, data design,
and justified assessment; see [ADAPTING_TO_NEW_DATA.md](ADAPTING_TO_NEW_DATA.md).

Software authorship is recorded in [CITATION.md](../CITATION.md). It does not
assign sole authorship of all underlying research contributions. The method
scope, software licence, and statistical assumptions are separate statements.

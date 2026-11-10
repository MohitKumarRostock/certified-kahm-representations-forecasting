# Forecasting readouts and development selection

The development procedure fits and selects three complete computational
forecasters: `raw`, `kahm_mem`, and `kahm_local`. The selected MEM and LOCAL
forecasters use one identical Gaussian KAHM encoder bank. All three use the
same readout family and hyperparameter grids. They need not have the same
feature width, retained kernel rank, effective capacity, or total search effort.

The task is defined in [BENCHMARK.md](BENCHMARK.md). The machine-readable
[scientific contract](../kahm_demo/specification.json) and
[readout specification](../kahm_demo/readout_specification.json) record the
fixed procedure. This guide supplies its mathematical and numerical definitions.

Development uses training and validation data. Selection retains a training-fit
model; a separate [freeze](FREEZE_INTERFACE.md) binds its executable state and
assessment protocol before calibration outcomes are accessed. Development
selection itself issues no risk certificate and uses no population-risk oracle.

## 1. Implementation map

| Component | Local definition | Implementation |
| --- | --- | --- |
| Gaussian KAHM geometry, reconstruction, scores, and memberships | [Encoder guide](ENCODER_DESIGN.md) | `kahm.py`, `soft_state.py`, `encoder.py` |
| RAW, MEM, and LOCAL; training-only scaling | Section 2 | `readout_features.py` |
| Shared raw-history landmark identities | Section 3 | `readout_features.py`, `forecast_workflow.py` |
| Gaussian Nyström block with a linear component | Section 4 | `readout_features.py` |
| Mean-error ridge and clipped canonical prediction | Section 5 | `ridge.py`, `forecast.py` |
| Exact validation and shared-bank selection | Section 6 | `selection.py`, `forecast_workflow.py` |
| Saved state and complete replay | Section 7 | `forecast_state.py`, `forecast_workflow.py` |

The definitions below are sufficient to identify the implemented family and
selection procedure. The bounded-loss certificate for the resulting frozen
predictors is justified separately in
[FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md).

## 2. Representations and training-only transformations

An input is the ordered pair of observed values
`(previous_observation, current_observation)`. The scalar response is the next
observed value. Hidden state labels and episode identifiers are not features.
Let $z$ denote the history after the stored training-only history
standardization. Before readout scaling, the three representations are

$$
\Gamma_{\rm RAW}(x)=z,\qquad
\Gamma_{\rm MEM}(x)=\phi(z),\qquad
\Gamma_{\rm LOCAL}(x)=\operatorname{col}\!\left(
  \phi(z),\{\phi_j(z)P_j^T(z-\mu_j)\}_{j=1}^{J}\right).
$$

The LOCAL coordinates describe the input relative to each local reference
geometry. They are not coordinates of the KAHM reconstruction. Rank-zero local
models append no coordinates. MEM is the exact prefix of LOCAL within a bank.
Neither scores nor log normalizers are additional candidate coordinates.

For each candidate representation, form all $n$ training rows
$\gamma_i=\Gamma(x_i)$. For coordinate $a$, fit

$$
\bar\gamma_a=\frac1n\sum_i\gamma_{ia},\qquad
s_a=\sqrt{\frac1n\sum_i(\gamma_{ia}-\bar\gamma_a)^2},\qquad
d_a=\begin{cases}1,&s_a<10^{-8},\\s_a,&s_a\ge10^{-8},\end{cases}
\qquad
u_a(x)=\frac{\Gamma_a(x)-\bar\gamma_a}{d_a}.
$$

The threshold comparison is strict. Constant and dependent feature columns
remain present. All training occurrences contribute to the moments, including
repeated histories. The implementation computes each column's moments on a
contiguous column array; appending LOCAL columns therefore preserves the
identical MEM prefix and its fitted scaling. The RAW branch uses its stored
history standardizer, followed by this same representation-scaling procedure.

These fitted transformations are reused on validation and future queries.
`readout_features.py` accepts no response targets. Targets enter only the
training ridge fit and the development validation loss.

## 3. Common raw-history landmark identities

The landmark budget is 16. Select the identities once from distinct **raw
training histories**, using the first training row at which each distinct
history occurs. If all distinct histories fit within the budget, use them in
lexicographic order. If fewer than 16 occur, retain that smaller set and record
the shortage. The generic procedure for more than 16 distinct histories uses
deterministic farthest-first selection in training-standardized raw coordinates;
its first choice and distance ties follow lexicographic order.

Exactly these training row identities are reused for RAW and for both encodings
under both candidate encoder banks. Landmark choice uses neither targets nor
encoded distances. Only the landmark identities are deduplicated: no training
or validation episode is removed from its fit or loss.

Write the resulting standardized encoded landmarks as
$\ell_j=u(x_{i_j})$, $j=1,\ldots,m$, with $m\le16$.
Distinct raw histories can have identical encodings. Such encoded collisions are
retained; the following common spectral truncation handles the Gram matrix.
Shared identities do not imply shared landmark coordinates, bandwidths, kernel
ranks, or feature widths.

## 4. Gaussian Nyström block with a linear component

Compute $\delta$ as the median of $\|\ell_i-\ell_j\|^2$ over
$i<j$, excluding distances at most $10^{-14}$. Set $\delta=1$
when no distances remain. For each bandwidth multiplier

$$
b\in\{0.25,0.5,1,2,4\},\qquad
\sigma_b^2=\max(\delta b^2,10^{-12}),\qquad
k_b(u,v)=\exp\!\left(-\frac{\|u-v\|^2}{2\sigma_b^2}\right).
$$

Form the landmark Gram matrix $K_{ij}=k_b(\ell_i,\ell_j)$, symmetrize
it, and order its eigenvalues from largest to smallest. Retain exactly those
eigenvalues satisfying

$$
\lambda_i>\max(10^{-12},10^{-10}\lambda_{\max}).
$$

Let $V$ contain the retained eigenvectors and $\Lambda$ the retained
positive eigenvalues. For the column vector of landmark similarities
$k_L(u)$, the unscaled Nyström block is

$$
\nu_b(u)=\Lambda^{-1/2}V^Tk_L(u).
$$

Fit a second column scaler to $\nu_b(u(x_i))$ over **all** training
episodes, with the same divisor $n$ and strict $10^{-8}$ standard
deviation rule. Denote the standardized block by $\widetilde\nu_b$. The
decoder input is

$$
f_b(x)=\operatorname{col}\bigl(u(x),\widetilde\nu_b(u(x))\bigr).
$$

Thus the standardized linear representation is preserved unchanged at the
front of the decoder input. The new block has its own fitted scaling. If the
representation has width $p$ and the retained kernel rank is $r$, the
ridge design has width $p+r$. Eigenvector conventions and residual checks
are recorded by the implementation. They do not constitute verified error
bounds relative to exact real arithmetic.

## 5. Scalar ridge fitting and clipped prediction

For a fixed feature map, let $F$ be the $n$-row training design and
let $y\in[0,1]^n$ contain the scalar next-observation targets. Fit

$$
(\widehat\beta,\widehat w)
=\arg\min_{\beta,w}\left\{
\frac1n\sum_{i=1}^{n}(y_i-\beta-F_iw)^2+\alpha\|w\|_2^2
\right\},
\qquad
\alpha\in\{10^{-6},10^{-5},\ldots,10^2\}.
$$

The intercept is unpenalized. Center $F$ and $y$, and compute the
reduced SVD $F_c=U\operatorname{diag}(s_i)V^T$. The stored coefficients
and intercept are computed from

$$
\widehat w
=V\operatorname{diag}\!\left(\frac{s_i}{s_i^2+n\alpha}\right)U^Ty_c,
\qquad
\widehat\beta=\bar y-\bar F\widehat w.
$$

The factor $n\alpha$ is required by the **mean** squared-error objective.
Ridge retains all supplied feature columns; it does not remove columns using a
diagnostic numerical rank. `fit_ridge_grid` shares one centered SVD across the
nine penalties for a feature map, preserving the same rows and objective.

Training solves the displayed quadratic objective before clipping. The complete
predictor subsequently evaluates

$$
g(x)=\widehat\beta+f_b(x)^T\widehat w,\qquad
\widehat y(x)=\min(1,\max(0,g(x))).
$$

The computational definition evaluates the affine sum with `math.fsum`, including
the intercept and the individually rounded binary64 products in feature order.
Every product and the final sum must be finite before clipping. Invalid or
nonfinite calculations raise an error; clipping does not repair them. Validation
uses the resulting clipped predictions, not the unbounded training fits.

Each query follows one canonical row calculation through the saved encoder,
feature map, and ridge model. Batch methods reuse that same row calculation and
may memoize identical rows; they retain every output occurrence and its order.
No query triggers a new fit. `inspect_one` additionally returns the unclipped
value, whether clipping changed it, and any KAHM fallback diagnostics.

## 6. Exact empirical validation and shared-bank selection

For a candidate computational predictor $h$, let

$$
L_B(h)=\frac1{n_B}\sum_{i\in B}(y_i-h(x_i))^2.
$$

Selection treats each actual clipped binary64 output as its exact rational
value, and computes this empirical mean exactly with rational arithmetic.
The observation values are also exact dyadic rationals. Aggregating repeated
observed triples by their sample counts is an arithmetic shortcut that preserves
every episode's weight. Population probabilities do not enter this calculation.
There is no target-variance normalization and no population-risk oracle call.
Exact empirical arithmetic removes ambiguity in comparisons; it does not turn
validation loss into a population risk certificate.

The procedure is:

1. For each bank in the fixed order `cap_1`, `cap_2`, select the MEM decoder
   minimizing $L_B$ over its 45 settings. Separately select the best LOCAL
   decoder in that same bank. Decoder ties use ascending bandwidth multiplier,
   then ascending ridge penalty.
2. Assign that bank the arithmetic mean of its two best validation losses:
   $S_c=(\min L_{B,\mathrm{MEM},c}+\min L_{B,\mathrm{LOCAL},c})/2$.
   Select the bank with the smallest exact $S_c$; bank ties use the fixed
   order above. Retain the MEM and LOCAL winners from this one shared bank.
3. Select RAW independently over its 45 decoder settings using the same decoder
   tie order. RAW has no encoder-bank search.

MEM and LOCAL may select different bandwidths and ridge penalties. Their shared
bank fixes the history preprocessing, partitions, references, PCA, KAHM models,
scores, memberships, and fallback rules. Selection does not splice together
encodings from different banks. The stored training-fitted parameters are
retained; there is no refit on pooled training and validation data.

| Candidate | Encoder banks searched | Bandwidths | Ridge penalties | Candidate fits |
| --- | ---: | ---: | ---: | ---: |
| RAW | 1 representation, no bank search | 5 | 9 | 45 |
| KAHM MEM | 2 | 5 | 9 | 90 |
| KAHM LOCAL | 2 | 5 | 9 | 90 |
| Total | | | | 225 |

The grid contains 25 feature constructions and 225 ridge fits. Matching the
readout family, landmark identities, and grids does not establish equal
effective capacity or equal total search effort. Comparisons concern these
complete selected procedures, including their development choices.

## 7. Numerical checks, saved state, and replay

Feature fitting requires finite values and a nonempty retained positive Gram
spectrum. Ridge fitting records SVD reconstruction and orthogonality checks,
normal-equation residuals, and an intercept residual, with tolerances of
$10^{-10}$. A failure of fitting or a required finite-value check stops the
development run. No failed grid setting is silently omitted, and the procedure
does not silently switch to another solver after a required check fails.

The arrays and stored scalar parameters define the implemented predictor.
`forecast_state.load_forecaster` reloads saved forecasting state without
refitting; `forecast.ForecastModel` supplies `inspect_one`, `predict_one`, and
`predict`. The canonical inference identifier is
`single_history_numpy_float64_forecast_v1`. Replaying the saved state on the
benchmark's finite input support checks its actual prediction behavior. It does
not promise identical retraining across different numerical runtimes.

The development bundle saves **all 225 complete fitted search states**, including
the configurations that were not selected. Their files are
`models/primary/search_states/<trial_id>.json`, with status
`development_candidate`. Each contains the fitted preprocessing, encoder when
applicable, feature transformation, and ridge state needed to execute that
candidate. The search record binds each state to its grid identity, source data,
canonical predictions, empirical losses, and diagnostics.

The three selected files, `models/primary/raw.json`, `kahm_mem.json`, and
`kahm_local.json`, have status `development_selected_not_frozen`. Their numerical
states are the winning search states, retained without refitting. The different
status records the outcome of development selection; it is not a final freeze
record or a risk certificate.

`verify-forecasters` performs complete state replay: it loads every one of the
225 candidate files, executes its predictions on all 16 supported histories,
and compares the results to the stored canonical predictions. It checks the
states' training moments and ridge equations against the development data,
shared landmark identities, KAHM bank identities, and recorded diagnostics. It
recomputes the exact training and validation losses and the shared-bank selection
from the complete declared grid. It then checks each selected state's identity
against its winning candidate, replays the three selected prediction tables,
and checks their shared membership scaling. These checks do not refit models,
access primary calibration, or evaluate oracle population risk. Artifact hashes
and fitted-equation checks establish consistency, not execution chronology.

Residual diagnostics and replay checks are engineering checks. They are not a
verified numerical enclosure around an ideal Gaussian map. The separate certificate
concerns the **actual frozen clipped predictor** under the contract's law.
Its freeze precedes independent calibration. Development selection establishes
neither that risk guarantee nor a lower bound on the best risk attainable by
an entire representation class.

## 8. Scope of this readout family

| Choice | Consequence |
| --- | --- |
| One next-observation scalar | The model is a direct one-step predictor, with no latent rollout |
| Ordinary squared loss | No empirical target-variance normalization enters selection or certification |
| Finite check followed by clipping to `[0,1]` | The complete predictor has a declared range; invalid arithmetic is not repaired by clipping |
| At most 16 raw-history landmark identities | The small benchmark support is represented by a compact fixed menu |
| One nonlinear family with a retained linear component | Comparisons use the same fitting procedure while retaining different representation widths |
| Complete candidate retention | All 225 fits remain inspectable, including configurations not selected |
| Required fitting check fails | Development stops instead of omitting the failed configuration |

These choices define the evaluated procedures. Their performance depends on the
fitted data and benchmark law. They imply no general representation superiority,
input injectivity, predictive sufficiency, Markov property, Koopman closure, or
optimality over a decoder class. A fitted-risk upper bound requires none of
those additional representation claims.

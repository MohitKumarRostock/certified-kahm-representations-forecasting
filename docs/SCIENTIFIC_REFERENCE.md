# Scientific reference: risk certificates for fixed representations

This demonstrator assesses whether a prescribed representation and decoder
class can meet a population mean squared error (MSE) threshold. It provides
lower confidence bounds on the **optimal class risk** and upper confidence
bounds on **fitted-predictor risk**. These are different quantities: a fitted
predictor can fail even when another decoder in its class could succeed.

This reference gives self-contained definitions, assumptions, and
interpretations. The
[implementation guide](LOWER_CERTIFICATES.md) gives commands and artifact
formats; the [proof note](LOWER_CERTIFICATE_PROOF.md) proves the results and
explains the numerical witnesses.

## 1. The prescribed prediction problem

Let $X$ denote an observed history and $Y$ its scalar future target. A fixed
measurable representation maps the history to a feature
$U=\Gamma(X)\in\mathcal U$. Equip $\mathcal U$ with a prescribed measurable,
finite-valued **pseudometric** $d$: it is nonnegative, symmetric, and obeys the
triangle inequality, and satisfies $d(u,u)=0$, but can assign zero distance
to different points. An
ordinary metric additionally separates different points.

Fix a **Lipschitz bound** $\kappa\ge0$ and a known bounded interval
$I=[y_-,y_+]$ containing the target almost surely. Predictions also lie in
$I$. Write $D=y_+-y_->0$ and $B=D^2$, so squared loss lies in $[0,B]$. The
class of admissible decoders is

$$
\mathcal G_\kappa(d)=
\{g:\mathcal U\to I\text{ measurable}:
|g(u)-g(v)|\le\kappa d(u,v)\text{ for all }u,v\}.
$$

Under the specified evaluation law of $(X,Y)$, define

$$
R(g)=\mathbb E[(Y-g(\Gamma(X)))^2],
\qquad
R_\kappa^*=\inf_{g\in\mathcal G_\kappa(d)}R(g).
$$

The risk $R(g)$ belongs to one decoder; $R_\kappa^*$ belongs to the entire
prescribed class. Both depend on the evaluation law. The representation,
coordinates, metric, and Lipschitz bound must be reported together. For
example, replacing $d$ by $c\,d$ and $\kappa$ by $\kappa/c$, for $c>0$,
leaves the class unchanged. Changing feature scale while holding $\kappa$
fixed can change the class.

| Symbol | Meaning |
| --- | --- |
| $\Gamma,d,\kappa,I$ | Prescribed representation, feature pseudometric, Lipschitz bound, and output interval |
| $R_\kappa^*$ | Infimum population MSE over this decoder class |
| $\widehat f$ | A predictor fixed using development data, with outputs in $I$ |
| $R_{\rm fit}$ | $\mathbb E[(Y-\widehat f(X))^2]$ |
| $a_n$ | Minimum empirical MSE over the class on the $n$ calibration observations |
| $T_b,\mathcal W_b$ | Normalized minimum block error and its expectation |
| $L,U_{\rm fit}$ | Lower confidence bound on $R_\kappa^*$ and upper confidence bound on $R_{\rm fit}$ |
| $\tau,\delta$ | Required MSE threshold and total allocated failure probability |

If $\widehat f=\widehat g\circ\Gamma$ for an admissible decoder
$\widehat g\in\mathcal G_\kappa(d)$, then
$R_\kappa^*\le R_{\rm fit}$. That inequality requires an admissibility
check; a small fitted prediction error does not establish it.

The implementation uses $I=[0,1]$, hence $D=B=1$, Euclidean distance on
archived feature coordinates, and a finite benchmark with 16 possible
histories. The general notation above does not mean that the command accepts
arbitrary intervals, pseudometrics, or data sources.

## 2. Population risk and fixed-dataset risk

For a fixed calibration sample, the empirical optimum is

$$
a_n=\inf_{g\in\mathcal G_\kappa(d)}
\frac1n\sum_{i=1}^n(Y_i-g(U_i))^2.
$$

A verified optimization bound $d_n\le a_n$ is a deterministic statement
about those observations. It becomes a lower confidence bound on population
risk only after a valid statistical calibration. A feasible prediction
assignment constructed from the calibration targets gives an upper bound
on $a_n$; it does not by itself give an upper bound on future risk.

Population statements here assume iid calibration episodes from the
specified evaluation law, independent of development. The representation,
decoder class, fitted predictor, sample-size menu, block partitions, and
confidence allocations are fixed before calibration values are examined.
The guarantees are conditional on development and also hold after averaging
over it.

Numerical optimization may use the calibration observations, provided its
reported lower bound is valid for every dataset. This permission does not
extend to choosing new representations or a new iid block partition after
inspecting results.

## 3. A hierarchy from joint Lipschitz constraints

For $b\ge2$ iid feature-target observations, let

$$
Q_b=\min_v\left\{
\sum_{i=1}^b(Y_i-v_i)^2:
|v_i-v_j|\le\kappa d(U_i,U_j)\text{ for all }i<j
\right\},
\qquad
T_b=\frac{Q_b}{b-1},
\qquad
\mathcal W_b=\mathbb E[T_b].
$$

Clipping predictions to $I$ leaves the minimum unchanged. A feasible
finite prediction table has a Lipschitz extension, so $Q_b/b$ equals the
minimum empirical MSE over $\mathcal G_\kappa(d)$ on that block.

The normalization uses **$b-1$**, rather than $b$, because a common shift of
a feasible prediction vector preserves its pairwise constraints. Centering
the residuals of any fixed admissible decoder gives a feasible block
assignment whose normalized error is its residual sample variance.
That sample variance is unbiased for the residual variance. Consequently,

$$
0\le\mathcal W_2\le\mathcal W_3\le\cdots\le R_\kappa^*,
\qquad
0\le T_b\le
C_b:=\frac{B\lfloor b^2/4\rfloor}{b(b-1)}.
$$

The monotonicity follows from a deterministic subset-average inequality,
proved in the [proof note](LOWER_CERTIFICATE_PROOF.md#2-a-block-lower-bound).
For $b=2$, the score has the explicit form

$$
T_2=\frac12\bigl(|Y_1-Y_2|-\kappa d(U_1,U_2)\bigr)_+^2.
$$

On the implemented finite support, $\mathcal W_b\uparrow R_\kappa^*$.
The same convergence holds for a compact metric feature space under the
measurability and boundedness assumptions stated in the proof note.
Thus, for a fixed such prediction problem and any $\tau<R_\kappa^*$,
some population hierarchy level exceeds $\tau$. This statement does not
specify the sample size needed to certify the comparison, and no fixed
block size works uniformly well across all prediction problems.

## 4. Two statistical uses of the same optimization results

Prespecify $K_b=\lfloor n/b\rfloor$ disjoint iid blocks. For each block,
the numerical checker supplies $\underline Q_{b,k}\le Q_{b,k}$.
The following routes use the same optimization witnesses:

| Route | Quantity supplied by the witnesses | Statistical calibration |
| --- | --- | --- |
| Block lower confidence bound | $\bar t_b=K_b^{-1}\sum_k\underline Q_{b,k}/(b-1)$ | $K_b$ independent true block scores in $[0,C_b]$ |
| Empirical-risk lower confidence bound | $d_{n,b}=n^{-1}\sum_k\underline Q_{b,k}\le a_n$ | Comparison with each fixed decoder's $n$ iid losses in $[0,B]$ |
| Full empirical-risk lower confidence bound | A verified lower bound $d_{n,\rm full}\le a_n$ from joint optimization | The same empirical-risk calibration |

The empirical-risk route does **not** treat the optimized value $d_n$ as
an average of iid variables. Its validity follows from the pointwise
inequality $d_n\le a_n\le\widehat R_n(g)$ for every fixed admissible $g$.
The proof passes to the population infimum after applying a bounded-loss
inequality to fixed decoders.

Larger blocks can improve $\mathcal W_b$ while reducing $K_b$. Their lower
confidence endpoints need not increase with $b$. Neither statistical route
is guaranteed to dominate the other. Every reported endpoint uses its
declared allocation; taking their maximum is covered by the simultaneous
confidence statement.

In this benchmark, repeated histories are aggregated with their exact
weights. Full empirical optimization therefore needs at most 16 prediction
variables even when $n$ is large. The experiment does not establish a
computational advantage for blocking on a large or continuous feature space.

## 5. Simultaneous confidence and threshold decisions

Allocate one-sided failure probabilities over the entire prespecified
family, with total at most $\delta$. Under the sampling and numerical
premises, the union bound gives simultaneous coverage with probability
at least $1-\delta$. Shared observations and nested sample prefixes are
allowed; independence between confidence statements is unnecessary.

For each class, let $L$ be the maximum of zero and its available lower
confidence endpoints. Every fixed fitted predictor has its own valid
upper endpoint $U_{\rm fit}$. When the predictor is admissible, the
simultaneous event gives

$$
L\le R_\kappa^*\le R_{\rm fit}\le U_{\rm fit}.
$$

| Outcome | Mathematical condition | Interpretation |
| --- | --- | --- |
| `class_excluded` | $L>\tau$ | Every decoder in the prescribed class exceeds the MSE threshold |
| `requirement_met` | Admissible fitted predictor and $U_{\rm fit}\le\tau$ | That predictor, and therefore its class, meets the requirement |
| `unresolved` | Neither condition is established | The certificates do not decide class adequacy |
| `certificate_unavailable` | A required numerical check cannot certify an individual endpoint | That endpoint supplies no favorable decision; its allocation is retained |

A bound above $\tau$ on one fitted predictor's error does not exclude its
class. Conversely, a low-risk predictor outside the class does not
contradict class exclusion. A conflicting lower and admissible upper
conclusion is retained as `conflicting_certificates` for inspection; such
a conflict cannot occur on the simultaneous coverage event.

These are statements about expected squared loss. They are not prediction
intervals, guarantees for each individual forecast, or confidence that the
sampling assumptions themselves hold.

## 6. Numerical evidence and the separate population audit

Floating-point optimization proposes dual multipliers and primal prediction
tables. Exact rational checks and conservative distance enclosures establish
the optimization inequalities. KL calculations use verified enclosures of
logarithms and outward-rounded endpoints. A solver's convergence flag is not
used as proof of a lower bound.

The assessed features are the exact represented values in archived tables.
All support-pair constraints are checked for an admissible fitted table, and
the Lipschitz extension argument supplies a decoder agreeing with that table.
This establishes the stated finite-support risk comparison. It does not
establish that the executable nonlinear readout obeys the same
Lipschitz bound on arbitrary inputs, or that binary64 features equal an ideal
real-arithmetic encoder.

The benchmark law is known, so a separate **population audit** can bound
optimal class risk and evaluate fitted risk after the sample certificates
have been saved and checked. It supplies reference quantities, not inputs
to class selection, confidence-endpoint construction, or seed selection.
An audit can record a coverage miss without changing the issued
certificate.

The [recorded study](../reports/lower_certificates_v1/report.md) demonstrates
class exclusion, admissibility, unresolved outcomes, and a larger-block
certificate that resolves a threshold missed by the pairwise certificate.
Its single declared stream does not estimate decision probabilities or
coverage frequency across repeated studies. MEM exclusion at the prescribed
Lipschitz bound does not establish unrestricted information loss, and the
study does not establish general forecasting or runtime superiority.

## 7. Relation to established methods

Lipschitz regression and extension in metric spaces provide the underlying
class formulation [1]. Lower bounds from sample-average optimization and
confidence bounds on an optimal population objective have a substantial
literature [2,3]. The bounded-mean concentration step used here follows
Hoeffding's inequality [4]; model-class risk and distribution-free
falsification are also studied directly in [5].

The proof note states the exact construction used by this software rather
than attributing the whole framework to one source. In particular,
monotonicity of expected empirical minima $a_b$ alone does not imply
monotonicity of $\mathbb E[b\,a_b/(b-1)]$, because $b/(b-1)$ decreases.
The subset-average residual-variance argument establishes the latter result.
No claim of universal statistical or computational superiority is needed
for these guarantees.

### Published references

1. Lee-Ad Gottlieb, Aryeh Kontorovich, and Robert Krauthgamer (2017).
   *Efficient Regression in Metric Spaces via Approximate Lipschitz
   Extension*. IEEE Transactions on Information Theory 63(8), 4838–4849.
   [DOI](https://doi.org/10.1109/TIT.2017.2713820);
   [preprint](https://arxiv.org/abs/1111.4470).
2. Wai-Kei Mak, David P. Morton, and R. Kevin Wood (1999).
   *Monte Carlo Bounding Techniques for Determining Solution Quality in
   Stochastic Programs*. Operations Research Letters 24(1–2), 47–56.
   [DOI](https://doi.org/10.1016/S0167-6377(98)00054-6).
3. Vincent Guigues, Anatoli Juditsky, and Arkadi Nemirovski (2017).
   *Non-asymptotic Confidence Bounds for the Optimal Value of a Stochastic
   Program*. Optimization Methods and Software 32(5), 1033–1058.
   [DOI](https://doi.org/10.1080/10556788.2017.1350177);
   [author-hosted article](https://www2.isye.gatech.edu/~nemirovs/GuiJudNem2017.pdf).
4. Wassily Hoeffding (1963). *Probability Inequalities for Sums of Bounded
   Random Variables*. Journal of the American Statistical Association
   58(301), 13–30.
   [DOI](https://doi.org/10.1080/01621459.1963.10500830).
5. Manuel M. Müller, Yuetian Luo, and Rina Foygel Barber (2025).
   *Are All Models Wrong? Fundamental Limits in Distribution-Free Empirical
   Model Falsification*. Proceedings of Machine Learning Research 291,
   4271–4308.
   [Published paper](https://proceedings.mlr.press/v291/muller25a.html).

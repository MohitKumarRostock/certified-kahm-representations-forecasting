# Mathematical guarantee for the class-risk extension

This note states the guarantee implemented by `class-risk`. The assessment
uses the finite benchmark in [BENCHMARK.md](BENCHMARK.md), outputs and targets
in $[0,1]$, and a declared Euclidean geometry on exact archived feature
coordinates. The block hierarchy and the two lower-confidence constructions
are those developed in *A Hierarchy of Risk Certificates for
Representation-Based Forecasting*.

The guarantee is conditional on the development information. All classes,
predictors, sample sizes, block index sets, and failure allocations are fixed
before calibration. Calibration consists of iid episodes from the declared
law, independent of development. The finite probability law is used in a
separate audit after certificates are issued.

## 1. The population quantity

For $U=\Gamma(X)\in\mathcal U$, let

$$
\mathcal G_\kappa(d)
=\{g:\mathcal U\to[0,1]: |g(u)-g(v)|\le\kappa d(u,v)\},
\qquad
R^*=\inf_{g\in\mathcal G_\kappa(d)}\mathbb E[(Y-g(U))^2].
$$

Every claim is specific to $(\Gamma,d,\kappa)$ and the evaluation law.
The archived coordinates are exact represented binary64 values. Euclidean
distances between them can be irrational; distance enclosures are verified
separately. Nothing here assumes that floating-point features equal an ideal
real-arithmetic encoder.

On a finite set of observed feature points, any bounded feasible prediction
table has a Lipschitz extension with the same constant. One construction is

$$
g(u)=\operatorname{clip}_{[0,1]}
\min_i\{z_i+\kappa d(u,u_i)\}.
$$

It agrees with every feasible assigned value $z_i$. This also justifies
omitting unobserved histories from an empirical optimization. The resulting
extension need not be the executable nonlinear readout used originally.

## 2. A block lower bound

For $b\ge2$ iid observations, write

$$
Q_b=\min_v\sum_{i=1}^b(Y_i-v_i)^2,
\quad |v_i-v_j|\le\kappa d(U_i,U_j),
\qquad T_b=\frac{Q_b}{b-1}.
$$

The minimum is unchanged by requiring $v_i\in[0,1]$: clipping preserves the
constraints and cannot increase squared error to a target in that interval.

For any fixed admissible $g$, put $r_i=Y_i-g(U_i)$. Shifting all predictions by
$\bar r$ preserves pairwise constraints, and clipping can only reduce loss.
Consequently,

$$
T_b\le\frac1{b-1}\sum_i(r_i-\bar r)^2.
$$

The expectation on the right is $\operatorname{Var}(Y-g(U))$, which is at
most $R(g)$. Taking the infimum over $g$ gives

$$
\mathcal W_b:=\mathbb E T_b\le R^*.
$$

A constant prediction at the sample mean is feasible. The largest sample
squared deviation for $b$ values in $[0,1]$ occurs at endpoints, giving

$$
0\le T_b\le C_b,
\qquad C_b=\frac{\lfloor b^2/4\rfloor}{b(b-1)}.
$$

### Why the expected hierarchy is monotone

At an optimal unconstrained-range block table, residuals have zero sum:
otherwise a common shift reduces their SSE. On any $m$-element subset $S$,
recenter that table's residuals within $S$. The recentered assignment is
feasible, so

$$
T_m(S)\le
\frac1{m(m-1)}\sum_{i<j,\ i,j\in S}(r_i-r_j)^2.
$$

Averaging over all such subsets counts every pair equally. The resulting
right-hand side is

$$
\frac1{b(b-1)}\sum_{i<j}(r_i-r_j)^2=T_b.
$$

Taking expectations yields $\mathcal W_m\le\mathcal W_b$. On this finite
feature support the feasible prediction set is compact, and the empirical
squared-error objective converges uniformly to its population counterpart.
Thus $T_b\to R^*$ and $\mathcal W_b\to R^*$ as $b\to\infty$.
This is a population statement; lower confidence endpoints need not be
monotone at fixed total calibration size.

## 3. Verified weighted optimization

Repeated observations at a feature value are compressed without losing
weight. A weighted problem takes the form

$$
Q=A+\min_z\sum_i w_i(z_i-m_i)^2,
\qquad |z_i-z_j|\le\kappa d_{ij},
$$

where $w_i>0$, $m_i\in[0,1]$, and $A\ge0$ is the within-group SSE. Coincident
feature rows are merged by adding weights, taking their weighted mean, and
adding their between-mean SSE to $A$. When $\kappa=0$, all rows are merged.
These transformations are exact identities.

Choose any subset of edges and arbitrary signed multipliers $\lambda_e$.
Let $s=M\lambda$, where each incidence column has a $+1$ and a $-1$.
For every feasible $z$,

$$
s^\top z\le\kappa\sum_e d_e|\lambda_e|.
$$

Completing squares therefore gives the weak-duality bound

$$
D(\lambda)
=A+s^\top m-\sum_i\frac{s_i^2}{4w_i}
-\kappa\sum_e d_e|\lambda_e|
\le Q.
$$

No convergence claim or multiplier optimality is needed. The floating-point
coordinate method only proposes multipliers; every issued objective is
recomputed using exact rational arithmetic. Upper distance enclosures are
used in the subtracted penalty, making the result conservative. Taking
$\max(0,D)$ is valid because squared-error objectives are nonnegative.

For an exact rational squared distance $q$, integer square-root arithmetic
constructs a dyadic enclosure $d^-\le\sqrt q\le d^+$. A proposed primal table
is repaired and then checked against every exact squared constraint. Its
objective is an upper bound on the optimum. A primal success cannot replace
the dual check needed for a lower bound.

The numerical verifier rechecks the input statistics, distances, multipliers,
both objective values, and primal feasibility. It does not rerun the optimizer.

## 4. One-sided bounded-mean endpoints

For $q,p\in[0,1]$, define binary KL divergence

$$
\operatorname{kl}(q\Vert p)
=q\log(q/p)+(1-q)\log((1-q)/(1-p)),
$$

with the usual endpoint conventions. Define

$$
k_-(q,c)=\inf\{p\le q:\operatorname{kl}(q\Vert p)\le c\},
\qquad
k_+(q,c)=\sup\{p\ge q:\operatorname{kl}(q\Vert p)\le c\}.
$$

For iid $Z_i\in[0,1]$ with mean $p$ and sample mean $\bar Z$,

$$
\Pr\{k_-(\bar Z,\log(1/\alpha)/N)>p\}\le\alpha,
\qquad
\Pr\{k_+(\bar Z,\log(1/\alpha)/N)<p\}\le\alpha.
$$

The Chernoff argument applies to bounded, not necessarily Bernoulli, variables:
convexity gives $\mathbb E e^{tZ}\le1-p+pe^t$. Optimizing the exponential
tail bound yields binary KL. Both endpoint functions are nondecreasing in
$q$.

The implementation encloses logarithms and the final KL comparisons with
exact rational arithmetic. Lower endpoints are rounded down and upper
endpoints up. General rational means are accepted; block normalization need
not produce a dyadic fraction. An inconclusive numerical check produces an
unavailable endpoint, not an optimistic replacement.

## 5. The two lower-confidence constructions

Fix $K_b=\lfloor n/b\rfloor$ consecutive disjoint blocks before seeing
calibration values. Suppose the numerical witnesses satisfy, for every
dataset, $0\le\underline Q_{b,k}\le Q_{b,k}$. Set

$$
\bar t_b=\frac1{K_b}\sum_k\frac{\underline Q_{b,k}}{b-1}.
$$

The block lower endpoint is

$$
L_b^{\rm blk}
=C_b k_-\left(\frac{\bar t_b}{C_b},
\frac{\log(1/\alpha_b)}{K_b}\right).
$$

The true block scores are iid in $[0,C_b]$. Numerical scores lie below them
pointwise, and endpoint monotonicity gives
$\Pr\{L_b^{\rm blk}>\mathcal W_b\}\le\alpha_b$.
Because $\mathcal W_b\le R^*$, this is also a lower bound on class risk.

Let $a_n$ be the minimum average squared error over the full calibration
sample. Any full feasible prediction restricts to a feasible prediction on
each block, so

$$
d_n=\frac1n\sum_k\underline Q_{b,k}\le a_n.
$$

Observations outside complete blocks contribute zero to the numerator and
remain in the denominator. The empirical-optimum endpoint is

$$
L^{\rm emp}=k_-\left(d_n,\frac{\log(1/\alpha_{\rm emp})}{n}\right).
$$

To prove its validity without assuming that the population infimum is attained,
choose a fixed admissible $g_j$ with $R(g_j)\le R^*+1/j$. Pointwise,
$d_n\le\widehat R_n(g_j)$. Endpoint monotonicity and the bounded-mean result
imply

$$
\Pr\{L^{\rm emp}>R^*+1/j\}
\le\Pr\{L^{\rm emp}>R(g_j)\}\le\alpha_{\rm emp}.
$$

The events on the left increase to $\{L^{\rm emp}>R^*\}$ as $j\to\infty$,
so the same bound holds for $R^*$. No uniform-convergence penalty over the
decoder class is required for this lower direction. The full-sample dual
certificate supplies another valid $d_n$ through the same argument.

Optimization choices may depend on calibration values because their verified
inequalities hold for every dataset. This does not authorize choosing new
classes or rearranging the iid block partition after inspecting outcomes.

## 6. Upper bounds, simultaneous coverage, and decisions

The fixed fitted predictor's exact squared losses lie in $[0,1]$. Its upper
endpoint is

$$
U=k_+\left(\frac1n\sum_i(Y_i-\widehat g(U_i))^2,
\frac{\log(1/\alpha_U)}n\right).
$$

It bounds that predictor's risk whether or not the predictor is admissible.
The additional inequality $R^*\le R(\widehat g)$ is used only when its
prediction table passes the class-membership check.

The protocol allocates failure probabilities across all requested classes,
sample sizes, repetitions, and lower/upper endpoints. Their sum is $\delta$.
A union bound gives simultaneous coverage with probability at least
$1-\delta$, conditional on development. Shared observations and prefixes are
permitted; independence between endpoint events is unnecessary.

On this event the maximum $L$ of the valid lower endpoints and zero satisfies
$L\le R^*$. An admissible fitted upper witness gives $R^*\le R(\widehat g)\le U$.
Thus $L>\tau$ excludes the class and $U\le\tau$ establishes acceptance.
Numerically unavailable endpoints keep their allocation and supply no
favorable conclusion. A verifier pass establishes consistency of retained
mathematical witnesses; it does not establish the iid premise or guarantee
realized coverage on every sample.

## 7. Independent population references

The benchmark gives

$$
\mu_{ab}=\frac1{10}+\frac45v_{(a+b)\bmod4},
\qquad
R(z)=\frac9{160}+\frac1{16}\sum_h(z_h-\mu_h)^2.
$$

The class audit solves this 16-variable constrained problem with weights
$1/16$ and constant $9/160$, verifies both optimization witnesses, and
independently enumerates the primal's risk over all 64 outcomes. The result is
an interval unless verified bounds establish equality. A large number of
decimal places in an ordinary solver output is not an exact optimum.

For a uniform $M$-point representation where distinct-point constraints are
inactive, let $\sigma^2=\mathbb E\operatorname{Var}(Y\mid\Gamma(X))$.
Conditional on feature counts $N_j$, the optimized within-group SSE has
expectation $\sum_j\sigma_j^2(N_j-1)_+$. Since

$$
\mathbb E[(N_j-1)_+]=\frac bM-1+(1-1/M)^b,
$$

the exact hierarchy is

$$
\mathcal W_b
=\sigma^2\frac{b-M+M(1-1/M)^b}{b-1}.
$$

For the short-history control, $M=4$ and $\sigma^2=5/32$. For the full-history
control, $M=16$ and $\sigma^2=9/160$. These references explain why the pairwise
bound can remain below a risk threshold even when a larger-block bound can
resolve the class's failure.

The MEM reference also admits a direct disjoint-pair proof. At $\kappa=2$,
the archived feature pairs with history IDs $(0,5),(10,15),(9,13),(6,7)$ each
have distance at most $1/500$. Their conditional means differ by
$3/5,3/5,4/5,4/5$. Every admissible prediction table therefore satisfies

$$
R(z)\ge\frac9{160}
+\frac1{32}\left[
2\left(\frac35-\frac1{250}\right)^2
+2\left(\frac45-\frac1{250}\right)^2\right]
=\frac{29513}{250000}.
$$

The implemented audit checks these distance premises exactly. This known-law
calculation is a reference for the sample certificate, not its input.

## Scope

The implementation establishes the finite-support numerical and statistical
statements above. It does not treat overlapping windows of a dependent time
series as iid, extrapolate the decoder's checked regularity outside the
declared support, or identify statistical information loss from a failure at
one prescribed Lipschitz bound. Source hashes bind stored bytes; they do not
prove historical chronology or sampling independence.

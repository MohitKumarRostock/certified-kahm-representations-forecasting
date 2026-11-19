# Proofs for lower class-risk and upper fitted-risk certificates

This note gives self-contained proofs for the block hierarchy, optimization
witnesses, and confidence bounds used by `class-risk`. The
[scientific reference](SCIENTIFIC_REFERENCE.md) explains their interpretation
and relation to published work; the [implementation guide](LOWER_CERTIFICATES.md)
gives commands and record formats.

The mathematics uses a prescribed interval $I=[y_-,y_+]$, width
$D=y_+-y_->0$, and squared-loss bound $B=D^2$. The implementation specializes
to $I=[0,1]$, hence $D=B=1$, Euclidean distance on exact archived feature
coordinates, and the finite benchmark defined in Section 7. Broader
statements about metric spaces are mathematical results, not additional
supported data interfaces.

All population guarantees are conditional on development. The classes,
fitted predictors, sample sizes, block partitions, and confidence allocations
are fixed before calibration. Calibration episodes are iid from the
prescribed evaluation law and independent of development. The known law is
used in a separate population audit after sample certificates are issued.

## 1. Optimal class risk and finite prediction tables

Let $X$ be the observed history, $Y\in I$ almost surely its scalar future
target, and $U=\Gamma(X)\in\mathcal U$ a prescribed measurable representation.
Let $d$ be a measurable, finite-valued pseudometric and fix $\kappa\ge0$.
Assume the infima and statistics below are measurable. Define

$$
\mathcal G_\kappa(d)=
\{g:\mathcal U\to I\text{ measurable}:
|g(u)-g(v)|\le\kappa d(u,v)\text{ for all }u,v\},
$$

$$
R(g)=\mathbb E[(Y-g(U))^2],
\qquad
R_\kappa^*=\inf_{g\in\mathcal G_\kappa(d)}R(g).
$$

Every claim is specific to $(\Gamma,d,\kappa,I)$ and the evaluation law.
Clipping a real-valued Lipschitz decoder to $I$ preserves its Lipschitz
bound and cannot increase squared loss to a target in $I$. Constants are
admissible, so
$0\le R_\kappa^*\le\operatorname{Var}(Y)\le B/4$.
If $D=0$, all risks are zero and no statistical inversion is needed.

On finitely many feature points $u_i$, an $I$-valued prediction table $z_i$
satisfying all pairwise constraints has the extension

$$
g_z(u)=\operatorname{clip}_I
\left(\min_i\{z_i+\kappa d(u,u_i)\}\right).
$$

Each function inside the minimum is $\kappa$-Lipschitz by the reverse
triangle inequality; a finite minimum and clipping preserve that bound.
At $u_i$, feasibility gives $z_i\le z_j+\kappa d(u_i,u_j)$ for every $j$,
with equality for $j=i$. Thus $g_z(u_i)=z_i$. The extension is measurable
under the stated assumptions. Conversely, every admissible decoder
restricts to a feasible table. Finite empirical optimization over tables
therefore equals optimization over $\mathcal G_\kappa(d)$.

The implementation treats archived binary64 coordinates as exact represented
values. Their Euclidean distances can be irrational, so numerical enclosures
are checked separately. The extension need not equal the original executable
nonlinear readout away from the support.

## 2. A block lower bound

For $b\ge2$ iid feature-target observations, write

$$
Q_b=\min_{v\in\mathbb R^b}
\left\{\sum_{i=1}^b(Y_i-v_i)^2:
|v_i-v_j|\le\kappa d(U_i,U_j)\text{ for all }i<j\right\},
\qquad
T_b=\frac{Q_b}{b-1},
\qquad
\mathcal W_b=\mathbb E[T_b].
$$

Clipping predictions to $I$ leaves the minimum unchanged. For $b=2$,
separating the prediction sum and difference gives

$$
T_2=\frac12\bigl(|Y_1-Y_2|-\kappa d(U_1,U_2)\bigr)_+^2.
$$

The optimal sum is $Y_1+Y_2$, and the optimal difference is $Y_1-Y_2$
clipped to $[-\kappa d(U_1,U_2),\kappa d(U_1,U_2)]$.

### Why the denominator is $b-1$

For any fixed admissible $g$, put $r_i=Y_i-g(U_i)$. Adding the sample
residual mean $\bar r$ to every prediction preserves all pairwise
constraints. Clipping those shifted predictions to $I$ can only reduce
loss. Consequently,

$$
T_b\le\frac1{b-1}\sum_i(r_i-\bar r)^2.
$$

The expectation on the right is
$\operatorname{Var}(Y-g(U))\le R(g)$. Taking the infimum over admissible $g$
gives $\mathcal W_b\le R_\kappa^*$.

A constant prediction at the sample target mean is feasible. The sample
variance of $b$ targets in $I$ is maximal at interval endpoints.
If $k$ targets are at the upper endpoint, the pairwise squared-difference
sum is $B\,k(b-k)$. Maximizing over integer $k$ gives

$$
0\le T_b\le C_b,
\qquad
C_b=\frac{B\lfloor b^2/4\rfloor}{b(b-1)}\le B/2.
$$

### Subset comparison and monotonicity

At an optimal unconstrained-range block table, residuals have zero sum;
otherwise a common shift would reduce their sum of squared errors. For
any subset $S$ of size $m\ge2$, recenter that table's residuals within $S$.
The shifted subset prediction table is feasible, so

$$
T_m(S)\le
\frac1{m-1}\sum_{i\in S}(r_i-\bar r_S)^2
=\frac1{m(m-1)}
\sum_{\substack{i<j\\i,j\in S}}(r_i-r_j)^2.
$$

Each pair belongs to $\binom{b-2}{m-2}$ of the $\binom bm$ subsets.
Since their ratio is $m(m-1)/(b(b-1))$,

$$
\binom bm^{-1}\sum_{|S|=m}T_m(S)
\le\frac1{b(b-1)}\sum_{i<j}(r_i-r_j)^2=T_b.
$$

Every $m$-subset has the same iid distribution as a block of size $m$.
Taking expectations proves

$$
0\le\mathcal W_2\le\mathcal W_3\le\cdots\le R_\kappa^*.
$$

This is a subset-average inequality. An individual block score need not
exceed every smaller subset's score.

### Convergence and its scope

On a finite feature support $u_1,\ldots,u_M$, feasible prediction vectors
form a closed subset of $I^M$, hence a compact set. Empirical risk is a
quadratic polynomial in those predictions. Its finitely many coefficients
are empirical frequencies and bounded first and second target moments.
The strong law makes those coefficients converge almost surely, uniformly
over the feasible set. Section 1 identifies its minimum with

$$
a_b=\inf_{g\in\mathcal G_\kappa(d)}
\frac1b\sum_{i=1}^b(Y_i-g(U_i))^2.
$$

Thus $a_b\to R_\kappa^*$ almost surely. Since
$T_b=b\,a_b/(b-1)$ and $0\le T_b\le B/2$, dominated convergence gives
$\mathcal W_b\uparrow R_\kappa^*$. This is the convergence statement directly
instantiated by the implementation.

The same conclusion holds on a compact metric feature space. The bounded
Lipschitz class is closed and equicontinuous, hence compact in uniform norm
by Arzelà–Ascoli. A finite uniform $\varepsilon$-net, together with

$$
|(y-g(u))^2-(y-h(u))^2|\le2D\|g-h\|_\infty,
$$

reduces uniform convergence of empirical risks to the strong law for
finitely many decoders. Let $b$ increase, then let $\varepsilon$ decrease
along a countable sequence. This yields
$\sup_g|\widehat R_b(g)-R(g)|\to0$ almost surely, and hence convergence of
the infima and of $\mathcal W_b$. No convergence claim for arbitrary
noncompact feature spaces follows from this argument.

For any fixed such law and $\tau<R_\kappa^*$, some finite $b$ satisfies
$\mathcal W_b>\tau$. This does not specify a sufficient calibration size.
At fixed $n$, increasing $b$ leaves fewer independent blocks, so lower
confidence endpoints need not increase. Section 7 also gives examples where
a fixed $b$ captures an arbitrarily small fraction of positive class risk.

## 3. Verified weighted optimization

Repeated observations at a feature value can be aggregated exactly:

$$
Q=A+\min_z\sum_i w_i(z_i-m_i)^2,
\qquad |z_i-z_j|\le\kappa d_{ij},
$$

where $w_i>0$, $m_i\in I$, and $A\ge0$ is the within-group sum of squared
errors (SSE). Merging coincident feature rows adds their weights, takes the
weighted mean, and adds between-mean SSE to $A$. When $\kappa=0$, every
prediction must coincide and all rows can be merged. Sample occurrences
retain their full weight.

Choose any subset of constraint edges and arbitrary signed multipliers
$\lambda_e$. Let $s=M\lambda$, with one $+1$ and one $-1$ in each incidence
column. Every feasible $z$ satisfies

$$
s^\top z\le\kappa\sum_e d_e|\lambda_e|.
$$

Adding this valid penalty to the objective and minimizing the resulting
unconstrained quadratic gives

$$
D(\lambda)=A+s^\top m-\sum_i\frac{s_i^2}{4w_i}
-\kappa\sum_e d_e|\lambda_e|\le Q.
$$

This is weak duality: no convergence or multiplier-optimality claim is
needed. The numerical optimizer proposes multipliers; the checker
recomputes their objective using exact rational arithmetic. Upper
enclosures of distances are used in the subtracted penalty, which preserves
the lower-bound direction. Taking $\max(0,D(\lambda))$ is valid because
squared-error objectives are nonnegative.

For an exact rational squared distance $q$, integer square-root arithmetic
constructs dyadic bounds $d^-\le\sqrt q\le d^+$. A proposed primal table is
repaired and then checked against every exact squared constraint and the
output interval. Its objective gives an upper bound on the empirical
optimum. A feasible primal does not replace the dual lower-bound witness.

The verifier rechecks sufficient statistics, distances, multipliers,
objective values, and primal feasibility. It does not rerun the optimizer.

## 4. One-sided confidence bounds for a bounded mean

For $q,p\in[0,1]$, let

$$
\operatorname{kl}(q\Vert p)
=q\log(q/p)+(1-q)\log((1-q)/(1-p))
$$

with the usual endpoint conventions. Define

$$
k_-(q,c)=\inf\{p\in[0,q]:\operatorname{kl}(q\Vert p)\le c\},
\qquad
k_+(q,c)=\sup\{p\in[q,1]:\operatorname{kl}(q\Vert p)\le c\}.
$$

For iid $Z_i\in[0,1]$ with mean $p$ and sample mean $\bar Z$,

$$
\Pr\{k_-(\bar Z,\log(1/\alpha)/N)>p\}\le\alpha,
\qquad
\Pr\{k_+(\bar Z,\log(1/\alpha)/N)<p\}\le\alpha.
$$

The Chernoff argument applies to bounded, not necessarily Bernoulli,
variables: convexity gives $\mathbb E[e^{tZ}]\le1-p+pe^t$.
Optimizing the exponential tail bound and inverting it gives the displayed
bounds. Both endpoint functions are nondecreasing in their first argument.
This is the bounded-mean consequence of Hoeffding's inequality; see the
[published references](SCIENTIFIC_REFERENCE.md#published-references).

For observations in $[0,C]$, $C>0$, divide their mean by $C$ before inversion
and multiply the resulting endpoint by $C$. If $C=0$, both endpoints are zero.

The implementation encloses logarithms and KL comparisons using exact
rational bounds. It rounds lower endpoints downward and upper endpoints
upward. General rational means are accepted. An inconclusive numerical
comparison produces an unavailable endpoint.

## 5. Two lower confidence constructions

Fix $K_b=\lfloor n/b\rfloor$ disjoint consecutive blocks before examining
calibration values. Assume the numerical witnesses satisfy
$0\le\underline Q_{b,k}\le Q_{b,k}$ for every dataset. Put

$$
\bar t_b=\frac1{K_b}\sum_k\frac{\underline Q_{b,k}}{b-1}.
$$

The **block lower confidence bound** is

$$
L_b^{\rm blk}=C_b k_-\left(
\frac{\bar t_b}{C_b},
\frac{\log(1/\alpha_b^{\rm blk})}{K_b}\right).
$$

The true block scores are iid in $[0,C_b]$. The numerical scores lie below
them pointwise, even if their numerical computation uses shared information.
Monotonicity of $k_-$ and Section 4 give
$\Pr\{L_b^{\rm blk}>\mathcal W_b\}\le\alpha_b^{\rm blk}$.
Because $\mathcal W_b\le R_\kappa^*$, this is also a lower confidence bound
on optimal class risk.

For the **empirical-risk lower confidence bound**, let $a_n$ be the minimum
empirical MSE over the full calibration sample. A full feasible prediction
restricts to a feasible table on each block, so

$$
d_{n,b}=\frac1n\sum_k\underline Q_{b,k}\le a_n.
$$

Observations outside complete blocks contribute zero to the numerator and
remain in the denominator. More generally, any measurable numerical bound
$0\le d_n\le a_n$ valid for every dataset gives

$$
L^{\rm emp}=B\,k_-\left(
\frac{d_n}{B},\frac{\log(1/\alpha_{\rm emp})}{n}\right).
$$

To prove validity without assuming attainment of the population infimum,
choose a fixed admissible $g_j$ with
$R(g_j)\le R_\kappa^*+1/j$. Pointwise,
$d_n\le a_n\le\widehat R_n(g_j)$. Monotonicity and Section 4 imply

$$
\Pr\{L^{\rm emp}>R_\kappa^*+1/j\}
\le\Pr\{L^{\rm emp}>R(g_j)\}\le\alpha_{\rm emp}.
$$

The events on the left increase to $\{L^{\rm emp}>R_\kappa^*\}$, so the
same bound holds in the limit. This argument does not treat $d_n$ as an
average of iid variables and does not need a class-complexity penalty in
this lower direction.

A full-sample optimization supplies another valid $d_n$ by the same proof.
The maximum of several bounds on the same $a_n$ is itself a valid $d_n$.
The software nevertheless records and allocates its prespecified
individual endpoints separately, so their individual claims are covered.

Optimization choices may depend on calibration values when their checked
inequalities hold for every dataset. That does not authorize choosing new
classes, sample sizes, or iid block partitions after inspecting results.
Neither $L_b^{\rm blk}$ nor $L^{\rm emp}$ universally dominates the other:
their normalizations, loss ranges, and effective calibration sample sizes
differ.

## 6. Fitted-risk upper bounds, simultaneous coverage, and decisions

Let $\widehat f:\mathcal X\to I$ be fixed using development data and put
$R_{\rm fit}=\mathbb E[(Y-\widehat f(X))^2]$. Suppose each calibration loss
has a valid numerical upper bound
$(Y_i-\widehat f(X_i))^2\le\overline\ell_i\le B$.
Then

$$
U_{\rm fit}=B\,k_+\left(
\frac{n^{-1}\sum_i\overline\ell_i}{B},
\frac{\log(1/\alpha_U)}n\right)
$$

satisfies $\Pr\{R_{\rm fit}>U_{\rm fit}\}\le\alpha_U$.
In the implemented finite benchmark these represented losses are computed
exactly. Admissibility is unnecessary for this fitted-predictor statement.

If the fitted prediction table passes every support-pair Lipschitz check,
Section 1 supplies an admissible decoder $\widehat g$ agreeing with
$\widehat f$ on all supported histories. Consequently
$R_\kappa^*\le R(\widehat g)=R_{\rm fit}$. An inadmissible table gives no
such inequality; its accurate fitted risk can coexist with a higher
optimal risk in the prescribed class.

Allocate failure probabilities across all requested classes, sample sizes,
repetitions, and lower/upper endpoints with total at most $\delta$.
The union bound gives simultaneous coverage with probability at least
$1-\delta$, conditional on development. Shared observations and nested
sample prefixes are permitted; independence between endpoint events is
unnecessary.

On that event, the maximum $L$ of valid lower endpoints and zero satisfies
$L\le R_\kappa^*$. For an admissible fitted predictor,

$$
L\le R_\kappa^*\le R_{\rm fit}\le U_{\rm fit}.
$$

Thus $L>\tau$ excludes the class, while admissibility and
$U_{\rm fit}\le\tau$ establish that the requirement is met. Otherwise the
decision is unresolved. Numerically unavailable endpoints retain their
allocation and supply no favorable conclusion. A conflict between an
available class lower and an admissible fitted upper cannot occur on this
simultaneous event and is retained for inspection.

A verifier pass establishes the retained numerical inequalities and
provenance checks. It does not prove iid sampling, development independence,
or coverage on every realized calibration sample.

## 7. Separate known-law population references

The implemented benchmark has observation values

$$
v=(0,\tfrac14,\tfrac34,1),\qquad X=(v_a,v_b),\qquad Y=v_c,
$$

with $a,b$ uniform over the 16 ordered pairs and

$$
\Pr(c\mid a,b)=\frac45\,\mathbf1\{c=(a+b)\bmod4\}+\frac1{20}.
$$

Its 64 history-target outcomes are specified in [BENCHMARK.md](BENCHMARK.md).
Direct summation gives

$$
\mu_{ab}=\mathbb E[Y\mid a,b]
=\frac1{10}+\frac45v_{(a+b)\bmod4},
\qquad
R(z)=\frac9{160}+\frac1{16}\sum_{a,b}(z_{ab}-\mu_{ab})^2.
$$

The first term is the full-history Bayes risk. The class audit uses weights
$1/16$ and constant $9/160$, verifies lower and upper optimization witnesses,
and independently enumerates a feasible table's risk over all 64 outcomes.
Its optimum is reported as an interval unless the witnesses prove equality.
Ordinary solver precision is not a proof of an exact optimum.

### Exact block hierarchy for the history controls

Suppose $U$ is uniform on $M$ points and constraints between distinct points
are inactive for all predictions in $I$; a sufficient condition is
$\kappa d(u_i,u_j)\ge D$ for $i\ne j$. Let
$\sigma_j^2=\operatorname{Var}(Y\mid U=u_j)$ and
$\sigma^2=M^{-1}\sum_j\sigma_j^2$.
Conditional on counts $N_j$, each group's optimal prediction is its sample
target mean. Its expected SSE is $\sigma_j^2(N_j-1)_+$. Since

$$
\mathbb E[(N_j-1)_+]=\frac bM-1+(1-1/M)^b,
$$

the exact hierarchy is

$$
\mathcal W_b=\sigma^2
\frac{b-M+M(1-1/M)^b}{b-1}.
$$

For the short-history control at $\kappa=4$, $M=4$ and
$\sigma^2=5/32$. For the full-history control at $\kappa=4$,
$M=16$ and $\sigma^2=9/160$. The distinct-point constraints are inactive
in their declared original observation coordinates.

This formula also illustrates the weakness of any fixed block size.
Take $Y$ independent of $U$, with $\Pr(Y=y_-)=\Pr(Y=y_+)=1/2$,
a uniform $M$-point feature distribution, and inactive distinct-point
constraints. Then
$R_\kappa^*=B/4$, while the ratio $\mathcal W_b/R_\kappa^*$ tends to zero
as $M\to\infty$ for each fixed $b$. For each fixed finite $M$, it instead
tends to one as $b\to\infty$.

### MEM geometry reference

At $\kappa=2$, the archived MEM feature pairs with history IDs
$(0,5),(10,15),(9,13),(6,7)$ have distances at most $1/500$.
Their conditional-mean differences are $3/5,3/5,4/5,4/5$.
Applying the two-point squared-error bound to these disjoint pairs gives

$$
R(z)\ge\frac9{160}
+\frac1{32}\left[
2\left(\frac35-\frac1{250}\right)^2+
2\left(\frac45-\frac1{250}\right)^2\right]
=\frac{29513}{250000}.
$$

The audit checks those distance premises exactly. This known-law argument
is a reference for the sample certificate, not its input. Its conclusion
concerns the prescribed metric and Lipschitz bound. It does not establish
unrestricted information loss; all archived MEM feature vectors are distinct.

## 8. Scope of the implemented evidence

The implementation establishes the finite-support numerical statements and
the statistical guarantees under the premises above. It does not treat
overlapping windows of a dependent time series as iid. It does not prove
that archived features equal an ideal real-arithmetic encoder or that the
original executable readout has the required global Lipschitz constant.
It does not establish general forecasting superiority or a runtime benefit
from blocking.

The audit follows issuance and verification and does not select classes,
predictors, calibration streams, or confidence endpoints. Source hashes bind
stored bytes; they do not prove historical chronology or sampling
independence. Selection of a new study after inspecting earlier outcomes
requires its own declared inference scope.

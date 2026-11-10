# Fitted-risk guarantee and its numerical verification

This version 0.1.0 demonstrator is based on our research. This note gives a self-contained
proof of its fitted-risk guarantee and describes the arithmetic conditions
that make a reported numerical endpoint conservative. All assumptions and
constants needed for the argument appear below.

The guarantee concerns the population mean squared error of each of three
already selected, frozen computational predictors. It does not assert that a
representation is sufficient, that a decoder is optimal, or that any individual
forecast lies in a prediction interval. The confidence statement is conditional
on development and on the stated sampling premises.

## 1. Predictors, risk, and the calibration experiment

Let $D$ denote all development information: training and validation data,
hyperparameter searches, selection decisions, fitted parameters, clipping
rules, and the computational conventions that determine predictions. After
development, the family consists of exactly three deterministic predictors,

$$
f_{1,D},\quad f_{2,D},\quad f_{3,D},
$$

called RAW, KAHM MEM, and KAHM LOCAL. Each returns a value in $[0,1]$.
Conditional on $D$, these functions and the calibration protocol are fixed.

Let $P$ be the stipulated evaluation distribution of a history–target pair
$Z=(X,Y)$, with $Y\in[0,1]$. The risk of predictor $m$ is

$$
R_m(D)=\mathbb E_{Z\sim P}\left[(Y-f_{m,D}(X))^2\right].
$$

The calibration premise is that

$$
Z_1,\ldots,Z_n\mid D\ \overset{\mathrm{iid}}{\sim}\ P,
\qquad n=4096.
$$

In particular, calibration is independent of the information used to develop
and select the predictors. There is one loss observation per independently
sampled episode. Observations within a history may have temporal dependence;
the premise here concerns independence of the complete calibration episodes.
A single dependent trajectory is not covered by this proof.

Define

$$
L_{i,m}=(Y_i-f_{m,D}(X_i))^2\in[0,1],
\qquad
q_m=\frac1n\sum_{i=1}^n L_{i,m}.
$$

The same $4096$ episodes may be evaluated by all three predictors. For each
fixed $m$, its losses are iid over $i$; losses for different predictors
on the same episode need not be independent. The effective sample count for
each risk bound is $4096$, not three times that count.

Repeated histories and repeated complete episodes retain their multiplicity.
Caching a prediction for a repeated history does not delete a loss observation.

### Exact computational loss

The predictor being assessed is the actual frozen computational function.
Each finite binary64 output is interpreted as its exact represented rational
number, not as a rounded decimal transcription or as the output of an ideal
real-arithmetic model. Clipping to $[0,1]$ is part of that frozen function;
evaluation does not introduce a new clipping rule.

For the finite benchmark, targets are exactly
$0,1/4,3/4,1$. Subtraction, squaring, summation, and division by $4096$
are performed with exact rational arithmetic. Thus the value passed to the
bound calculation is the exact empirical mean $q_m$ for those computational
predictions. It is not the result of a floating-point MSE reduction.

A binary64 value in $[0,1]$ has dyadic denominator at most $2^{1074}$.
Subtracting one of the benchmark targets does not increase that maximum;
squaring gives denominator at most $2^{2148}$, and division by
$4096=2^{12}$ gives at most $2^{2160}$. The numerical API consequently
requires an exact dyadic $q\in[0,1]$ with denominator exponent at most
$2160$. This is an implementation domain restriction, not an assumption
that a general bounded-loss concentration inequality needs dyadic data.

## 2. An elementary lower-tail inequality for bounded losses

Fix a development outcome $D$ and one predictor. Suppress their subscripts,
write $R=\mathbb E[L_i\mid D]$, and work throughout under the conditional
calibration distribution.

For every real $t$, convexity of $x\mapsto e^{tx}$ implies that its graph
on $[0,1]$ lies below the chord joining its endpoint values:

$$
e^{tx}\le (1-x)+xe^t,\qquad 0\le x\le1.
$$

Taking expectations gives

$$
\mathbb E[e^{tL_i}\mid D]\le 1-R+Re^t.
$$

Independence over calibration episodes therefore yields

$$
\mathbb E\!\left[e^{t\sum_{i=1}^n L_i}\mid D\right]
\le (1-R+Re^t)^n.
$$

For $t<0$ and $0\le s<R$, the event $q\le s$ implies
$e^{t\sum_i L_i}\ge e^{tns}$. Markov's inequality gives

$$
\Pr(q\le s\mid D)
\le \exp\!\left(n\left[\log(1-R+Re^t)-ts\right]\right).
$$

If $0<s<R<1$, the minimizing value is

$$
t_* = \log\frac{s(1-R)}{R(1-s)}<0.
$$

Substituting it proves

$$
\boxed{\Pr(q\le s\mid D)\le e^{-n\operatorname{kl}(s\Vert R)}}
\tag{1}
$$

where the binary relative entropy is

$$
\operatorname{kl}(a\Vert p)
=a\log\frac ap+(1-a)\log\frac{1-a}{1-p}.
$$

The convention is $0\log(0/u)=0$, including the continuous boundary
cases. A positive numerator divided by zero contributes infinite divergence.
At $s=0<R<1$, taking $t\to-\infty$ in the preceding bound gives
$(1-R)^n=e^{-n\operatorname{kl}(0\Vert R)}$, so (1) also holds there.
At $s=R$, its right side is one. If $R=0$ or $R=1$, boundedness
implies $L_i=R$ almost surely and the relevant boundary statements are
immediate.

No step assumes that the loss is a Bernoulli random variable. The Bernoulli
expression arises because the chord bound controls the moment generating
function of every $[0,1]$-valued loss with the same mean.

## 3. Inverting the inequality in the upper direction

Set

$$
\delta=\frac1{60},\qquad
c=\frac{\log(1/\delta)}n=\frac{\log60}{4096}.
$$

For an empirical mean $q$, define the ideal upper endpoint

$$
u(q)=\sup\left\{p\in[q,1]:
\operatorname{kl}(q\Vert p)\le c\right\}.
\tag{2}
$$

If $q=1$, then $u(q)=1$. For $q<1$, the divergence is continuous
in $p\in[q,1)$, starts at zero, and tends to infinity as $p\to1$.
For $q<p<1$,

$$
\frac{\partial}{\partial p}\operatorname{kl}(q\Vert p)
=\frac{p-q}{p(1-p)}>0.
$$

Consequently, there is a unique $u(q)\in(q,1)$ satisfying
$\operatorname{kl}(q\Vert u(q))=c$. At $q=0$, the equivalent
identity is $u(0)=1-e^{-c}$; the implementation does not need to evaluate
this exponential.

We now prove, for every fixed predictor,

$$
\Pr\bigl(R>u(q)\mid D\bigr)\le\delta.
\tag{3}
$$

If $R\in\{0,1\}$, the almost-sure statements above make (3) immediate.
Suppose $0<R<1$. A failure $R>u(q)$ requires $q<R$, since
$u(q)\ge q$. For $q<R$, strict monotonicity in the second argument
shows that

$$
R>u(q)\quad\Longleftrightarrow\quad
\operatorname{kl}(q\Vert R)>c.
$$

For fixed $R$, the function
$h(s)=\operatorname{kl}(s\Vert R)$ is continuous and strictly decreasing
on $[0,R]$, with $h(R)=0$ and $h(0)=-\log(1-R)$. In the
interior this follows from

$$
h'(s)=\log\frac{s(1-R)}{R(1-s)}<0.
$$

If $c\ge h(0)$, failure is impossible. Otherwise, there is a unique
$s_*\in(0,R)$ with $h(s_*)=c$, and the failure event is exactly
$q<s_*$. Applying (1) at this fixed threshold gives

$$
\Pr(R>u(q)\mid D)
=\Pr(q<s_*\mid D)
\le\Pr(q\le s_*\mid D)
\le e^{-nc}=\frac1{60}.
$$

This argument uses a threshold determined by the fixed true risk. It does
not substitute a random empirical threshold into a fixed-threshold tail bound.

## 4. Three predictors, partial availability, and acceptance

Let $A_m$ mean that the implementation reports a valid endpoint for
predictor $m$. On $A_m$, the numerical checks described below establish
the pointwise deterministic inequality

$$
U_m\ge u(q_m).
\tag{4}
$$

Numerical availability may depend on calibration data, and the availability
events for different predictors may be coupled by a shared work budget. Neither
kind of dependence affects the following event inclusion:

$$
\{A_m\text{ and }R_m(D)>U_m\}
\subseteq \{R_m(D)>u(q_m)\}.
$$

**Theorem (simultaneous fitted-risk upper bounds).** Under the assumptions in
Section 1, if every reported endpoint satisfies (4), the following bound holds.
Combining (3), the preceding event inclusion, and the union bound proves it:

$$
\boxed{
\Pr_{\mathrm{Cal}}\!\left[
  \exists m\in\{1,2,3\}: A_m\text{ and }R_m(D)>U_m
  \mid D
\right]
\le\frac3{60}=\frac1{20}.
}
\tag{5}
$$

Equivalently, with conditional probability at least $19/20$, every
reported endpoint simultaneously covers its own fitted risk. If all three
endpoints are reported, that event covers all three risks. If some are
unavailable, (5) covers the reported endpoints, supplies no endpoint for the
unavailable candidates, and retains the original allocation of $1/60$ per
predictor. If all are unavailable, the statement about reported endpoints is
vacuous; it is not a successful risk assessment.

**There is no additional conditioning on availability or acceptance.** In
particular, (5) is not a claim that
$\Pr(R_m\le U_m\mid D,A_m)\ge19/20$, or that coverage conditional on
all three calculations succeeding is at least $19/20$. Data-dependent
conditioning can change those probabilities.

For the fixed tolerance $\tau=1/10$, define acceptance by
$A_m$ and $U_m\le\tau$. Then

$$
\Pr_{\mathrm{Cal}}\!\left[
  \exists m: A_m,\ U_m\le\tau,\ R_m(D)>\tau
  \mid D
\right]\le\frac1{20},
\tag{6}
$$

because that event is a subset of the failure event in (5). A user may select
any accepted member of this fixed family without adding a new allocation.
This remains a bound on the probability of a false acceptance under the full
calibration experiment, not a posterior probability or a conditional-on-pass
guarantee.

Development may have considered many fits before freezing these three. Their
selection is already included in $D$, so this argument does not apply a
union bound over every development fit. It would not justify adding or revising
predictors after looking at calibration, or choosing a favorable calibration
run from repeated attempts without accounting for those attempts.

## 5. Exact directed enclosures for logarithms

The numerical calculation must prove (4) without relying on an approximate
root whose rounding direction is unknown. It uses integer arithmetic and
exact rational intervals to enclose the logarithms in the KL divergence.

### 5.1 Range reduction and a series remainder

For a positive rational $x$, integer bit lengths and exact comparisons
give an integer $a$ and rational $u$ such that

$$
x=2^a u,\qquad 1\le u<2.
$$

Set $z=(u-1)/(u+1)$, so $0\le z<1/3$. Integration of the geometric
series for $1/(1-t^2)$, or direct differentiation and evaluation at zero,
gives

$$
\log\frac{1+z}{1-z}
=2\sum_{j=0}^{\infty}\frac{z^{2j+1}}{2j+1}.
$$

For
$S_M(z)=2\sum_{j=0}^{M-1}z^{2j+1}/(2j+1)$, the omitted terms are
nonnegative and satisfy

$$
\begin{aligned}
0\le \log\frac{1+z}{1-z}-S_M(z)
&\le \frac{2z^{2M+1}}{(2M+1)(1-z^2)}\\
&\le \frac{3}{4(2M+1)9^M},
\qquad 0\le z\le\frac13.
\end{aligned}
\tag{7}
$$

The first inequality replaces each denominator $2j+1$ in the tail by
$2M+1$ and sums a geometric series. The second uses $z\le1/3$.
The same formula at $z=1/3$ encloses $\log2$.

### 5.2 Directed dyadic recurrence

Choose a permitted precision $b\in\{128,256,512\}$, set $M=b$, and
use the integer grid denominator $G=2^b$. Initialize

$$
\ell_0=\lfloor Gz\rfloor,\qquad h_0=\lceil Gz\rceil.
$$

For each next required power set

$$
\ell_{j+1}=\lfloor\ell_j z^2\rfloor,\qquad
h_{j+1}=\lceil h_j z^2\rceil.
$$

Induction, using $z^2\ge0$, proves

$$
\ell_j/G\le z^{2j+1}\le h_j/G.
$$

Accumulate the integer bounds

$$
A=\sum_{j=0}^{M-1}\left\lfloor\frac{2\ell_j}{2j+1}\right\rfloor,
\qquad
B=\sum_{j=0}^{M-1}\left\lceil\frac{2h_j}{2j+1}\right\rceil.
$$

Each term is enclosed in the correct direction. By (7), the exact interval

$$
\left[
\frac A G,
\frac{B+\left\lceil G\,\frac{3}{4(2M+1)9^M}\right\rceil}{G}
\right]
\tag{8}
$$

encloses $\log u$. At $z=0$, its exact value zero is returned
directly. Every floor and ceiling above is computed by exact integer division,
including the divisions implicit in rational $z^2$.

Finally, use $\log x=\log u+a\log2$. When $a<0$, multiply the
upper endpoint of the $\log2$ interval to obtain the lower product
endpoint, and the lower endpoint to obtain the upper product endpoint.
Forgetting this reversal would break the enclosure. The $\log2$ and
$\log60$ intervals may be cached by precision within a work-budget
instance; their rational values do not change on cache reuse.

The number $b$ fixes the grid and series length. It is not a promise that
every final expression has error at most $2^{-b}$, or that every sign can
be resolved at that precision. Correctness follows from interval inclusion,
not from an assumed number of accurate digits.

### 5.3 KL and root residual intervals

For $0<q<1$ and $0<p<1$, enclose
$\log(q/p)$ and $\log((1-q)/(1-p))$. Their weights $q$ and
$1-q$ are nonnegative, so multiplying and adding their lower endpoints
gives a lower KL endpoint; doing the same with their upper endpoints gives an
upper KL endpoint. Rational arithmetic performs these operations exactly.

At $q=0$, use $\operatorname{kl}(0\Vert p)=-\log(1-p)$, reversing
the logarithm interval's endpoints on negation. At $q=1$, the finite
formula is $-\log p$. At $p=q$, divergence is exactly zero. No
zero-weight term causes an evaluation of $\log0$. Infinite KL boundary
values are handled analytically, not encoded as finite rational intervals.

Write

$$
F_q(p)=4096\operatorname{kl}(q\Vert p)-\log60.
$$

If the KL interval is $[K_-,K_+]$ and the confidence-log interval is
$[C_-,C_+]$, then

$$
F_q(p)\in
[4096K_- - C_+,\ 4096K_+ - C_-].
\tag{9}
$$

Thus a nonnegative lower endpoint in (9) proves $F_q(p)\ge0$, while a
nonpositive upper endpoint proves $F_q(p)\le0$. An interval containing
both negative and positive values proves neither sign.

## 6. Bracketing and a direct proof of the reported endpoint

For $q<1$, initialize the exact bracket $[L,H]=[q,1]$. Its lower
direction follows from $F_q(q)=-\log60<0$; the upper boundary is
analytic, since $u(q)<1$. The invariant is

$$
q\le L\le u(q)\le H\le1.
\tag{10}
$$

When $H-L>2^{-80}$, form the exact midpoint $p=(L+H)/2$. Evaluate
(9) first at $128$ bits, then at $256$, and then at $512$ only
if needed, always keeping the same midpoint during those precision attempts.

- If the lower residual endpoint is nonnegative, set $H=p$.
- Otherwise, if the upper residual endpoint is nonpositive, set $L=p$.
- If neither sign is proved, leave the bracket unchanged and try the next
  permitted precision. If all three attempts are ambiguous, report
  `certificate_unavailable`.

Monotonicity of $F_q$ proves that either justified move preserves (10).
The upper move takes precedence if both comparisons hold exactly. Each
successful move halves the width, so at most $80$ successful bisections
suffice to reach width $2^{-80}$ from an initial width at most one.
There are at most $240$ root interval evaluations per candidate.
This is a bound on successful progress, not a theorem that ambiguous midpoint
comparisons will always become decidable.

Before an endpoint is issued, a direct checker verifies the saved exact
empirical mean and bracket and freshly checks the final endpoint witnesses:

1. It verifies $0\le q\le L\le H\le1$, the exact width identity, and
   $H-L\le2^{-80}$.
2. It accepts $L=q$ analytically. Otherwise it requires $q<L<1$
   and a fresh interval evaluation proving $F_q(L)\le0$.
3. It accepts $H=1$ analytically. Otherwise it requires $q<H<1$
   and a fresh interval evaluation proving $F_q(H)\ge0$.
4. It recomputes the nonanalytic endpoint enclosures at their recorded
   precisions with fresh constant caches, compares them with the saved
   witnesses, and checks the record's exact reporting rules and consistency.

At the workflow level, the empirical mean supplied to this checker is also
recomputed from all calibration occurrences and the verified frozen
predictions. A plausible-looking bound record alone does not establish that
its $q$ came from the required data and predictor.

The direct checker does not call the root search. Its final endpoint
inequalities are enough to establish (10) without trusting successful search
decisions as numerical evidence. In particular, $F_q(H)\ge0$ and strict
monotonicity imply $H\ge u(q)$. The reported endpoint is

$$
U=H,
$$

so (4) holds deterministically whenever the checks succeed. Its conservatism
also satisfies $0\le U-u(q)\le H-L\le2^{-80}$.

At $q=1$, the exact bracket is $[1,1]$ and the endpoint is $U=1$,
subject to the remaining input, identity, and work checks; no logarithm is
required. If $q<1$ but the initial bracket already has width
at most $2^{-80}$, the same rules can establish the analytic upper
endpoint $H=1$ without bisection. An upper endpoint of one is not used
as a fallback when the required width has not been established.

The checker is separate from the search procedure, but both use the same
logarithm enclosure backend. This separation is not a claim of two wholly
independent implementations or a machine-checked proof of the source code.
The mathematical justification of that backend is (7)–(9), together with
correct implementation of exact integer and rational operations.

## 7. Work limits, unavailability, and display

The fixed calculation permits precision values $128,256,512$, at most
$512$ terms in each atanh-series evaluation, integers of at most $32768$ bits,
and a shared budget of $1{,}000{,}000$ series-term iterations for the
three searches and their initial direct checks. A range-reduced logarithm may
also need the $\log2$ series when its constant is not already cached; all
such iterations count against the shared budget. Resource counters continue
across candidates; a resource failure stays in effect. A later verification
uses its own bounded verification budget and does not repeat the root search.

An unproved sign, a work-limit violation, an unsupported numerical input, an
inadequate bracket width, or a failed direct check yields no reported endpoint.
The numerical bound routine retains that candidate as `certificate_unavailable`,
with a reason and available diagnostics. A model, data, source, or record
identity inconsistency can instead stop the surrounding workflow before it
produces a completed certificate bundle. Such an operational failure is not
an issued endpoint or a successful assessment.
The fixed protocol does not automatically replace the sample, increase the
budget, or report a wider endpoint in response to a failure. Even a valid
coarse bracket is diagnostic only if the required reporting checks fail.

These rules can reduce availability. They do not use an extra statistical
failure allowance: on success, enclosure and endpoint direction are
deterministic implications of exact arithmetic. This statement assumes the
specified arithmetic and checks are correctly implemented; it does not assign
zero probability to all possible software or hardware faults.

The endpoint is stored as a reduced fraction $U=A/B$, with $B>0$,
using canonical integer strings. For display only, compute

$$
J=\left\lceil\frac{10^{18}A}{B}\right\rceil,
\qquad U_{\mathrm{display}}=J/10^{18}.
$$

Then $U_{\mathrm{display}}\ge U$, with upward rounding error less than
$10^{-18}$. The displayed string has exactly $18$ fractional digits.
Acceptance uses the exact comparison $10A\le B$, not the displayed
decimal and not a binary floating-point conversion.

| Status | Meaning under the stated premises |
| --- | --- |
| `requirement_met` | All required checks support an endpoint and its exact value satisfies $U\le1/10$ |
| `unresolved` | A valid endpoint satisfies $U>1/10$; this does not prove $R>1/10$ |
| `certificate_unavailable` | No endpoint is issued because a required condition or check was not established |

An unavailable record can be internally consistent without being a verified
risk bound. Consistency of its failure diagnostics must not be displayed as
successful endpoint verification.

## 8. Interpretation and limits of the guarantee

Equation (5) is a statement about repeated iid calibration samples from the
declared law, conditional on development. Once a particular calibration file
and its endpoints have been observed, the numerical inequalities can be
checked exactly, but this does not turn a frequentist confidence statement
into a probability distribution over the fixed unknown risks.

A reproducible pseudorandom sampler, fixed seeds, and separate named streams
do not by themselves establish the iid premise. If all randomness used to
generate a sequence is fixed and included in the conditioning information,
that sequence is deterministic; iid repeated-sampling coverage does not follow
merely from replaying it. The theorem's conditioning fixes development while
leaving the stipulated independent calibration draw random. The
computational experiment records the sampler and states this sampling premise
explicitly rather than claiming to prove it from hashes or seed names.

Hashes and replay checks bind the saved predictors, data, and arithmetic
records and can detect inconsistencies. They do not authenticate historical
chronology or independently prove that calibration was never inspected during
development. Independence, freezing before calibration, and the evaluation law
remain substantive premises of the statistical claim.

The guarantee concerns one fixed family and one prescribed calibration
experiment. It does not establish an optimal decoder risk, a lower bound on
representation quality, sufficiency of an encoding, generalization to another
data distribution, or superiority over other representation families. A later
finite-law calculation of the actual fitted risks can audit the retained
endpoints, but it neither determines these endpoints nor adds a new confidence
guarantee or a repeated-sampling coverage estimate.

## 9. Implementation responsibilities

The mathematical argument is complete above. The following source components
connect it to the concrete artifact checks:

| Responsibility | Source component |
| --- | --- |
| Fixed family size, count, allocation, numerical rules, and tolerance | `kahm_demo/calibration_protocol.json` |
| Exact losses and all-occurrence empirical means | `kahm_demo/calibration_ledger.py` |
| Directed logarithm and binary-KL enclosures; bounded arithmetic | `kahm_demo/verified_arithmetic.py` |
| Root bracketing, final endpoint checks, exact decisions, and unavailable records | `kahm_demo/risk_bounds.py` |
| Binding the freeze, calibration receipt, ledger, sources, and all three bound records | `kahm_demo/certificate_workflow.py` |

The numerical routines alone neither inspect calibration observations nor
establish the iid and frozen-predictor premises. The workflow must supply the
verified exact means and model/data identities, and the user must assess the
declared sampling and development assumptions before interpreting the result
as a fitted-risk certificate.

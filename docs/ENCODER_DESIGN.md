# Gaussian KAHM encoder and soft-state representations

This guide defines the encoder used by version `0.1.0`. It fits geometry from
training inputs, produces membership-only and locally enriched representations,
and supplies two banks for development selection. Readout fitting and shared-bank
selection are defined in [READOUT_DESIGN.md](READOUT_DESIGN.md). The statistical
claim concerns the complete frozen predictor described in
[CERTIFICATION_DESIGN.md](CERTIFICATION_DESIGN.md), rather than this encoder alone.

## 1. Definitions and implementation map

Let the training-standardized input be a column vector `z` of dimension `d`.
Local model `j` stores `n_j` reference rows in `Q_j`, of shape `n_j × d`; a local
mean `mu_j`; an orthonormal basis `P_j`, of shape `d × r_j`; and, at positive
rank, a projected covariance `S_j`, of shape `r_j × r_j`. Formulas below omit
the local index when discussing one model. In code, input histories and
reference matrices use row-array conventions consistent with these formulas.

| Definition | Location in this guide | Implementation |
| --- | --- | --- |
| Training standardization, partitions, and references | Section 2 | `geometry.py`, `encoder_workflow.py` |
| Local PCA and projected Gaussian metric | Section 3 | `geometry.py`, `kahm.py` |
| Reference regularizer and signed reconstruction | Section 4 | `kahm.py` |
| Radial/angular score and partition normalization | Section 5 | `soft_state.py` |
| MEM and LOCAL feature ordering | Section 5 | `soft_state.py`, `encoder.py` |
| Stored state and canonical replay | Section 7 | `encoder_state.py`, `encoder_workflow.py` |

These are the implemented definitions, including numerical exceptions and
fallbacks. No external document is needed to identify them. The task and data
roles are specified in [BENCHMARK.md](BENCHMARK.md), and the machine-readable
contract is [specification.json](../kahm_demo/specification.json).

## 2. Training observations and reference geometry

Raw inputs have order `(previous_observation, current_observation)`. No hidden
state label, episode ID, future observation, or future-state label is an input
feature. The encoder fitting API accepts an input matrix, with no target argument.
The bundle workflow constructs that matrix directly from training episode inputs.

For each coordinate, fit the mean and standard deviation using all training
rows, including duplicates. Variance has divisor `n_train`. A standard deviation
at most `1e-12` is replaced by one. Store and reuse

$$
z=(x-\mu_{\rm train})/s_{\rm train}.
$$

Two partitions request 2 and 4 groups. For deterministic clustering, distinct
standardized rows are sorted lexicographically and carry their occurrence counts.
The first center is the first row; subsequent centers are farthest from the
selected centers. Distance ties use the first row; assignment ties use the first
center. Lloyd updates use occurrence-weighted means, preserving the empirical
training distribution. At most 150 iterations are allowed. A stable assignment
or unchanged center finishes an attempt. An iteration-limit fit is retained and
explicitly marked. Too few distinct rows reduce the group count. An empty group
halves the attempted count and restarts, with every attempt recorded.

Each group contributes at most 24 distinct reference histories. If all fit, they
are retained in lexicographic order; otherwise the same deterministic
farthest-first rule selects them. Reference identities record the first original
training row at which each distinct history occurred. The occurrence counts are
recorded, but selected reference rows receive equal weight in local PCA.
All original episodes remain available for later decoder fitting and loss
calculation.

Partitions and reference identities are fitted once and shared across the
ordered PCA-cap menu `(1, 2)`. Within a bank, MEM and LOCAL share every fitted
component. References may be reused across the two different partitions; they
still belong to distinct local models. Actual counts and ranks are recorded in
each fitted bank rather than assumed from the requested group counts.

## 3. Local PCA and projected Gaussian metric

Model j stores reference rows in Q, their mean mu, and basis columns in P.
Start with rank request `min(cap, input_dimension, reference_count-1)`.
Sample covariance uses divisor `reference_count-1`.

Let `eps64=2^-52`, the binary64 machine epsilon. Eigenvalues are sorted
descending. Retain values strictly greater than
`max(eps64, eps64*largest_eigenvalue)`. If a retained projected coordinate has
range below `1e-3`, reduce the request and repeat. Stop at request one and record
any remaining range violation. Singleton, constant, or entirely thresholded
geometry uses rank zero. It returns the local mean and contributes no LOCAL
coordinates.

PCA signs make the largest-absolute component positive, with first-index ties.
For exactly equal returned eigenvalues, a coordinate-projection basis of their
eigenspace supplies a fixed ordering. Nearly equal eigenvalues are not merged.
Positive-rank projected covariance S must pass Cholesky; the fit adds no jitter.
Unexpected fitting failures stop execution.

For positive rank r,

$$
S=\operatorname{Cov}_{n-1}(P^Tq_i),\qquad
\kappa(z,q)=\exp\!\left[-\frac{(P^T(z-q))^TS^{-1}(P^T(z-q))}{2r}\right].
$$

If `S=L L.T`, the stored metric is `M=P L^(-T)/sqrt(r)`, evaluated by a
linear-system solve. Thus the kernel log is `-0.5*||(z-q)M||^2`. Explicit
differences are used. The loader checks that M matches P and S, that its Gram
matrix matches the Gaussian kernel, and that the saved regularized inverse is
consistent with its saved regularizer.

### Projected-reference collisions

Distinct raw references can coincide after a rank-reducing projection. These
collisions are counted and retained. Positive lambda makes `G+lambda*I`
invertible in the mathematical Gaussian construction even when G has duplicate
rows; the implemented solve must also pass its finite-value and residual checks.
The shared reference set is not silently altered after projection. No identity
requiring distinct projected references, exact input recovery, or equivalence
to an ideal real-arithmetic function is asserted for the fitted bank.

## 4. Reference regularization and signed reconstruction

Let $G_{ij}=\kappa(q_i,q_j)$ be the unrescaled reference Gaussian Gram matrix,
with $n$ reference rows and input dimension $d$. Reference energy is

$$
\nu=\frac{\|Q\|_F^2}{nd},\qquad \lambda^{(0)}=2.5\nu.
$$

Q contains globally standardized references; the energy calculation does not
subtract the local mean. At each completed solve,

$$
C^{(a)}=(G+\lambda^{(a)}I)^{-1},\qquad
\lambda^{(a+1)}=2\nu+\frac{\|Q-GC^{(a)}Q\|_F^2}{nd}.
$$

Stop when absolute regularizer change is at most `1e-3`, or after 100 solves.
Store the last completed matching pair `(C,lambda)`, not the subsequent proposal.
The inverse is obtained by solving against the identity matrix. Finite-value
and scaled residual checks are recorded. Rank-zero models have no inverse or
regularizer; a positive-rank fit requires positive finite lambda.

At query z, form its log-kernel vector and subtract its maximum before
exponentiation. This common scaling prevents every kernel entry from underflowing
to zero. It is applied only to the query vector, not to rows of the fitting Gram.

The KAHM reconstruction is

$$
w=Ck(z),\qquad A(z)=\frac{Q^Tw}{\mathbf1^Tw}.
$$

Weights can be negative. A negative denominator alone is valid. Common weight
rescaling by `max(abs(w))` avoids overflow in sums and preserves the exact
quotient. Require nonzero weight mass and

$$
\frac{|\sum_iw_i|}{\sum_i|w_i|}\ge 10^{-10}.
$$

Failed finite-value or cancellation checks use affine reconstruction
`mu + P P.T (z-mu)`. Full-rank affine fallback returns z directly. This identity
shortcut applies to the fallback; full-rank KAHM is still evaluated using its
Gaussian quotient. The complete score is computed from whichever reconstruction
was actually used. Every fallback has a recorded reason.

## 5. Scores and soft-state features

Given reconstruction a, compute the score in the full standardized input space:

$$
e=1-e^{-\|z-a\|_2},\qquad
\theta=\pi^{-1}\arccos(\operatorname{clip}(c,-1,1)),\qquad
T=\sqrt{(e^2+\theta^2)/2}.
$$

Here c is the cosine when both norms are positive, and zero otherwise. In
particular, `T(0,0)=1/(2*sqrt(2))`. Identical nonzero vectors receive zero score.
Scaled angular arithmetic, `expm1`, and `hypot` reduce avoidable overflow and
underflow. Cosine and final score are clipped only against their specified bounds.

For offset eta=`1e-6` and contrast omega=`16`, normalize within each partition:

$$
\ell_j=\omega\log(1-T_j+\eta),\qquad
h_b=\max_{i\in\mathcal J_b}\ell_i,
$$

$$
\zeta_b=h_b+\log\sum_{i\in\mathcal J_b}e^{\ell_i-h_b},\qquad
\phi_j=\frac{e^{\ell_j-h_b}}{\sum_{i\in\mathcal J_b}e^{\ell_i-h_b}}
\quad(j\in\mathcal J_b).
$$

There is no small-sum uniform branch or artificial membership floor. Extreme
relative affinities may round to zero for other parameter choices; parameters
producing nonfinite log normalizers are rejected. Stored feature diagnostics
and canonical replay check the memberships on every supported history.

The feature vectors are

$$
\Gamma_{\rm MEM}=\phi,\qquad
\Gamma_{\rm LOCAL}=\operatorname{col}\!\left(\phi,
\{\phi_j P_j^T(z-\mu_j)\}_{j=1}^{J}\right).
$$

The local coordinates belong to the input, not its reconstruction. The MEM block
is copied identically into LOCAL. Scores and log normalizers are retained as
diagnostics but are not appended to either of the declared candidate encodings.
Decoder feature standardization belongs to the readout fit; these are the unscaled
representation coordinates.

## 6. Fixed choices and their consequences

| Choice | Implemented setting and consequence |
| --- | --- |
| Geometry arithmetic | Float64 arrays and recorded NumPy/BLAS runtime; retraining across runtimes need not be bitwise identical |
| Partitions | Requested group counts 2 and 4; insufficient or degenerate geometry follows the recorded reduction rules |
| Clustering | Deterministic farthest-first initialization and occurrence-weighted Lloyd updates |
| References | At most 24 distinct histories per group; equal local-PCA weights after selection |
| PCA menu | Caps 1 and 2, sharing preprocessing, partitions, and reference identities |
| Rank cutoff | Absolute and relative machine-epsilon safeguards, plus the projected-range rule |
| Query cancellation | Relative signed mass below `1e-10` triggers the declared affine fallback |
| Affinities | Offset `1e-6`, contrast `16`, separate partition normalization, no small-sum uniform replacement |
| Shared-bank selection | Mean of the best MEM and LOCAL validation decoder MSEs; see the readout guide |

The bank menu and numerical behavior are development choices fixed before
calibration. Changing them requires a newly identified fitted predictor and
assessment, not a relabeling of an existing certificate.

## 7. State, replay, and limits of the present checks

Fitted arrays are saved as hexadecimal floating values in typed JSON; no pickle
or code execution is needed to load them. Loading reconstructs stored state and
checks mathematical consistency without refitting. The archive also contains
every canonical feature row for the 16 supported histories. Verification checks
file identities, shared geometry, state provenance, derived diagnostic summaries,
and exact feature replay. Hashes establish consistency, not execution chronology.

Canonical inference evaluates one history at a time. Batch APIs reuse that same
calculation and may cache identical rows while retaining all output occurrences.
Arrays are float64. PCA and linear algebra use NumPy; software versions are
recorded. Bitwise retraining across different NumPy/BLAS/hardware configurations
is not promised. The loader preserves values, and the full-support replay check
detects changes in actual features caused by another runtime.

Finite real inputs of the fitted width are required. Nonfinite inputs, invalid
shapes, failed fitting factorizations, or nonfinite standardization/local feature
results raise errors. Query-level KAHM normalization failures use the declared
affine fallback when it is finite. There is no silent output clipping or arbitrary
uniform-state replacement for an encoder failure. A valid freeze must check all
16 benchmark histories, including their actual fallback behavior.

Residuals, PCA diagnostics, and successful replay are engineering evidence. They
are not verified error enclosures around an ideal real-arithmetic Gaussian map.
The certificate concerns the actual frozen computational predictor, with bounded
forecast loss on the declared evaluation law. Encoder construction alone
establishes neither predictive sufficiency nor a performance advantage for LOCAL.

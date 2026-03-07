# Fast finite-precision polar and inverse roots: condition-number-driven, log-centered plan

## 1. Goal

Compute an ML-useful approximation to the polar factor
$$
Q = \operatorname{polar}(G) = G(G^T G)^{-1/2}
$$
as fast as possible in finite precision (bf16 compute, small-side fp32 certs).

Primary objective:
$$
\text{minimize wall time to reach a target conditioning of the applied certificate.}
$$

The same framework extends to SPD inverse square root (whitening), inverse $r$-th roots, and applied transforms $G P^{-s/r}$.

---

## 2. What quality means (make condition number the spec)

Assume $m \ge n$ (otherwise swap sides and certify on the smaller side). Define
$$
B := G^T G,\qquad \widehat Q := GZ,\qquad S := \widehat Q^T \widehat Q = Z^T B Z \succ 0.
$$

### 2.1 Primary spec: residual anisotropy after apply

The preconditioning objective is the condition number of the whitened certificate:
$$
\kappa(S) := \frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}.
$$

Use log-width:
$$
\eta(S) := \frac12 \log\!\left(\frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}\right)
= \frac12\log \kappa(S).
$$

Targets are specified as $\kappa(S)\le \kappa_\star$ (equivalently $\eta(S)\le \eta_\star$ with $\eta_\star=\tfrac12\log\kappa_\star$).

Example: $\kappa_\star=1.5$ implies $\eta_\star=\tfrac12\log(1.5)$.

### 2.2 Secondary reporting (compatibility / debugging only)

Define $E := S-I$ and report
$$
\rho_2 := \|E\|_2,\qquad \delta_F := \|E\|_F,
$$
but these are not the primary design objective.

If an additive band is needed for interpretation,
$$
\kappa(S) \le \kappa_\star \iff \rho_2 \le \frac{\kappa_\star-1}{\kappa_\star+1}.
$$
So $\kappa_\star=1.5$ corresponds exactly to $\rho_2 \le 0.2$.

---

## 3. Log-centering is mandatory (remove scale drift)

Scalar rescaling of $Z$ does not change $\kappa(S)$, but it improves numerical stability and makes log-objectives symmetric.

We track and remove multiplicative drift via the mean-log drift:
$$
c_{\det}(S) := \frac{1}{n}\log\det(S)
= \frac{1}{n}\sum_{i=1}^n \log \lambda_i(S).
$$

Recenter by
$$
Z \leftarrow e^{-c_{\det}(S)/2}\,Z
\quad\Longrightarrow\quad
S \leftarrow e^{-c_{\det}(S)}\,S,
$$
so $\det(S)$ is driven to 1 (mean log-eigenvalue becomes 0).

Practical computation: if $S = LL^T$ (Cholesky),
$$
\log\det(S) = 2\sum_{i=1}^n \log L_{ii}.
$$

Optional stronger recenter (if eigs are cheap): use endpoint geometric mean
$$
c_{\mathrm{end}}(S)=\log\sqrt{\lambda_{\min}(S)\lambda_{\max}(S)},
$$
but this is not required if we use $c_{\det}$.

---

## 4. Core exact dynamics (design in log space, not additive space)

A right-update has the form
$$
Z_+ = Z\,q(S),\qquad S=Z^T B Z,
$$
so
$$
S_+ = q(S)\,S\,q(S).
$$
Eigenvalues evolve by the scalar map
$$
x \mapsto \phi(x) := x\,q(x)^2.
$$

Define log-coordinate $z=\log x$ and
$$
\psi(z) := \log\!\big(\phi(e^z)\big).
$$

If we allow recentering after each step (using $c_{\det}$), the compression objective is to shrink $\eta(S)$:
$$
\eta(S_+) \le \eta_\phi(\eta(S)),
$$
where $\eta_\phi$ is the induced log-width map (measured after recenter).

Reciprocal symmetry is ideal:
$$
\phi(1/x)=1/\phi(x) \iff \psi(-z)=-\psi(z),
$$
which makes log-centering "automatic" in exact arithmetic (we still recenter using $c_{\det}$ for robustness in fp).

---

## 5. Default algorithm family: 1-2 step rational Gram-side policy

We adopt a fixed small number of steps because solves are fast and we want predictable wall time.

### 5.1 Rational step family (1 Cholesky + 1 solve)

Use the Mobius/Padé family
$$
q_c(x) = \frac{x+c}{cx+1},\qquad
\phi_c(x)=x\left(\frac{x+c}{cx+1}\right)^2.
$$

Matrix update (one factorization + one solve with many RHS):
$$
Z_+ = Z\,(S+cI)\,(cS+I)^{-1}.
$$

### 5.2 Choose targets by condition number (log-symmetric)

Choose a condition target $\kappa_\star$ (default: $1.5$).
The terminal step is designed (offline, once) by the discrete predecessor problem in a log-symmetric way:
find $c_2$ and a predecessor band around 1 such that
$$
\phi_{c_2}(x) \in \left[\frac{1}{\sqrt{\kappa_\star}},\ \sqrt{\kappa_\star}\right]
\quad\text{for all } x \text{ in the predecessor band.}
$$

### 5.3 Penultimate step is optimized to feed the terminal step (band-to-band)

Once the terminal step and its predecessor band $X_2$ are fixed, the penultimate step is designed to map the widest possible band $X_1$ into $X_2$:
$$
\phi_{c_1}(x)\in X_2 \quad \forall x\in X_1,
$$
maximizing $\eta(X_1)=\tfrac12\log(x_{\max}/x_{\min})$.

This is the correct "final step fixed, final-1 step optimal" design.

### 5.4 Practical runtime policy (no power iteration)

We do not estimate $(\lambda_{\min},\lambda_{\max})$ up front.

Algorithm sketch (default path uses 2 solves total):

0) Form $B=G^T G$ (or certify on the other side if $m<n$).

1) Initialize with trace centering:
$$
\mu := \frac{1}{n}\operatorname{tr}(B),\qquad Z \leftarrow \mu^{-1/2} I.
$$

2) Step 1 (penultimate compressor): apply rational step with $c_1$.

3) Recenter using $c_{\det}(S)$ (Cholesky logdet):
$$
S=Z^T B Z,\quad Z \leftarrow e^{-c_{\det}(S)/2} Z.
$$

4) Certify cheaply (Section 6). If already $\kappa(S)\le \kappa_\star$, stop (1-step success).

5) Step 2 (terminal): apply rational step with $c_2$.

6) Recenter again using $c_{\det}(S)$, certify again, stop.

Important: 2 steps are extremely powerful, but not a universal deployment guarantee without any guard. Keep one fallback for rare pathologies.

---

## 6. Certification without eigendecomposition (Cholesky + traces)

We want a cheap, robust certificate of $\kappa(S)$, avoiding eigvalsh and avoiding power iterations.

### 6.1 Log-center (always do)
Compute $c_{\det}(S)$ via Cholesky and recenter $Z$.

### 6.2 Conservative $\kappa$ upper bound using $(\operatorname{tr}S,\log\det S)$

Let
$$
a := \frac{1}{n}\operatorname{tr}(S),\qquad
g := \exp\!\left(\frac{1}{n}\log\det(S)\right),\qquad
r := \frac{a}{g}\ge 1.
$$
Define
$$
F_n(\kappa) := \frac{(n-1)+\kappa}{n\,\kappa^{1/n}}.
$$
A rigorous worst-case bound is obtained by solving
$$
F_n(\kappa_{\text{bound}})=r,\qquad \kappa_{\text{bound}}\ge 1
$$
by 1D monotone bisection. If $\kappa_{\text{bound}} \le \kappa_\star$, stop.

### 6.3 Optional sharper moment bounds (GPU-friendly)
If needed, also compute
$$
\operatorname{tr}(S^2)\ \text{(equivalently } \|S\|_F^2\text{)},
$$
and combine with trace-based inequalities to tighten decisions. Use only if it reduces total wall time.

---

## 7. Guards and fallbacks

Always:
- symmetrize $B$ and $S$ before factorization/certification: $X \leftarrow \tfrac12(X+X^T)$,
- check NaN/Inf,
- check Cholesky failure; allow tiny ridge $\epsilon I$ (log it),
- check non-monotone spikes in $\kappa_{\text{bound}}$; if spike, recenter and retry once.

Fallbacks:
- if 2-step rational fails to reduce $\kappa_{\text{bound}}$, run one safer step (either a more conservative rational parameter or a GEMM-only odd polynomial step), then re-enter the 2-step pipeline.

### Do we still need Frobenius scaling?
Not as a primary design component. Keep it only as:
- a cheap magnitude safety cap (avoid overflow; crude $\lambda_{\max}$ upper bound),
- a restart tool if Cholesky becomes ill-behaved,
- a baseline for comparisons.

---

## 8. Policy comparison (direct vs Gram-side vs hybrid)

Compare policies, not single steps:

A) Direct: update $G$ with odd polynomials/rationals $X_+=Xq(X^T X)$.

B) Gram-side: form $B=G^T G$, refine $Z\approx B^{-1/2}$ on the small side, apply once.

C) Hybrid: a few direct global-compression steps, then small-side refinement and one final apply.

Winner is the policy with best time-to-target conditioning and acceptable tail risk.

---

## 9. Benchmark plan (simplified)

We no longer split many cases up front; we benchmark the policy "1 step if possible, else 2 steps".

Record:
- wall time (median, p95),
- $\kappa_{\text{bound}}$ after each step and final,
- optional $\delta_F$ for compatibility,
- number of solves, number of small GEMMs, tall passes,
- failure/guard triggers.

Compare:
- 1-step vs 2-step (same final $\kappa_\star$),
- rational vs direct odd polynomial baseline,
- effect of $c_{\det}$ recentering.

---

## 10. Optional: using spectrum-spread bounds to predict 1-step vs 2-step

The goal is to decide whether the terminal step alone will succeed (skip penultimate), without eigenvalues.

Preferred:
- use $\kappa_{\text{bound}}$ from (6.2) and compare to the terminal predecessor allowance.

Optional:
- use normal-matrix spread / Kantorovich-ratio style bounds as additional cheap surrogates for spread (relevant since $S$ is Hermitian/normal), if they improve decisions in practice.

---

## 11. Immediate tasks

1) Implement the condition-number certificate:
   - Cholesky logdet for $c_{\det}(S)$,
   - $(\operatorname{tr}S,\log\det S)\to \kappa_{\text{bound}}$ bisection.

2) Extend the scalar predecessor solver to:
   - log-symmetric terminal target for chosen $\kappa_\star$,
   - band-to-band design for $c_1$ feeding the terminal predecessor set.

3) Implement the 2-step rational policy with recenter+certify between steps.

4) Benchmark: how often 1 step suffices vs needing 2 steps, and tail failures.

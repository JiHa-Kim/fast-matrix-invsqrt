# Fast finite-precision polar and inverse roots: lean execution plan (condition-number + log-centered)

## 0. One-line objective

Minimize wall time to reach a target *residual anisotropy* after apply:
$$
S := Z^T B Z \approx I,\quad B := G^T G,\quad \widehat Q := GZ,\quad S=\widehat Q^T\widehat Q.
$$

---

## 1. Primary spec: condition number (not additive radius)

For whitening / preconditioning, the most meaningful global quality spec is
$$
\kappa(S) := \frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}.
$$

Use the log-width coordinate
$$
\eta(S) := \frac12 \log\kappa(S)
= \frac12\log\!\left(\frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}\right).
$$

Target by tier (example):
- medium: $\kappa(S)\le 1.5$ (equivalently $\eta(S)\le \tfrac12\log(1.5)$),
- strong: choose smaller $\kappa_\star$ as needed.

Connection to additive band near 1:
if $\lambda(S)\subset[1-\rho,1+\rho]$ and $\rho<1$, then
$$
\kappa(S)\le \frac{1+\rho}{1-\rho},\quad
\rho = \frac{\kappa-1}{\kappa+1}.
$$
So $\kappa_\star=1.5$ corresponds to $\rho=0.2$.

---

## 2. Log-centering is mandatory (remove scale drift)

Scalar rescaling $Z\leftarrow \alpha Z$ induces $S\leftarrow \alpha^2 S$.
We exploit this to remove multiplicative drift cheaply.

### 2.1 Cholesky-only log-center (recommended default)

Define the mean-log drift
$$
c_{\det}(S) := \frac{1}{n}\log\det(S).
$$
Recenter by
$$
Z \leftarrow \exp\!\left(-\frac{c_{\det}(S)}{2}\right) Z,
\quad\Longrightarrow\quad
S \leftarrow \exp(-c_{\det}(S))\,S,
$$
so $\det(S)$ is normalized to 1 (mean log-eigenvalue is 0).

Compute $c_{\det}(S)$ via Cholesky: if $S=LL^T$ then
$$
\log\det(S)=2\sum_{i=1}^n \log L_{ii}.
$$

### 2.2 Optional endpoint log-center (when eigs are cheap)

If you already computed $\lambda_{\min},\lambda_{\max}$, you can also use
$$
c_{\mathrm{end}}(S) := \tfrac12(\log\lambda_{\max}+\log\lambda_{\min})
$$
to set the geometric mean of endpoints to 1.

---

## 3. Local update dynamics (exact arithmetic)

For a small-side step
$$
Z_+ = Z\,q(S),\quad S=Z^T B Z,
$$
we get
$$
S_+ = q(S)S q(S),
$$
so eigenvalues evolve by the scalar map
$$
x \mapsto \phi(x) := x\,q(x)^2.
$$

We design *policies* (sequences of steps + recenter + guards), not isolated polynomials.

---

## 4. Rational one-solve family (fast if Cholesky is fast)

Use the 1-parameter Mobius family
$$
q_c(x)=\frac{x+c}{cx+1},
\qquad
\phi_c(x)=x\left(\frac{x+c}{cx+1}\right)^2,
\qquad c>0.
$$

Matrix implementation (one factorization + one solve with many RHS):
$$
Z_+ = Z\,(S+cI)\,(cS+I)^{-1}.
$$

This family is reciprocal-symmetric:
$$
q_c(1/x)=\frac{1}{q_c(x)},\quad \phi_c(1/x)=\frac{1}{\phi_c(x)},
$$
so it naturally fits log-width objectives.

---

## 5. Certification without power iteration

Power iteration is not required (and can be fragile in bf16).
We use one of:

### 5.1 Exact small-side eigs (when $n$ is modest)
Compute $\lambda_{\min},\lambda_{\max}$ of $S_{\mathrm{sym}}=\tfrac12(S+S^T)$ in fp32,
then $\kappa(S)$ and $\eta(S)$ exactly.

### 5.2 Cholesky-only conservative $\kappa$ bound (no eigs)
Let
$$
a := \frac{1}{n}\operatorname{tr}(S),\quad
g := \exp\!\left(\frac{1}{n}\log\det(S)\right),\quad r:=\frac{a}{g}\ge 1.
$$
Then solve for $\kappa_{\mathrm{bound}}\ge 1$ from
$$
r = \frac{(n-1)+\kappa}{n\,\kappa^{1/n}}
$$
(by 1D bisection). This gives a rigorous upper bound $\kappa(S)\le \kappa_{\mathrm{bound}}$.

### 5.3 Optional trace-based spread surrogates (very cheap)
If you need ultra-cheap early-stage screening, use trace/Frobenius/trace-of-powers style bounds
as surrogates for spread/condition; validate periodically with 5.1 or 5.2.

---

## 6. Default policy for $\kappa_\star=1.5$ (1 or 2 steps, log-centered)

We treat "1 vs 2 steps" as a policy choice, decided by cheap certification.

### Constants (from scalar predecessor design; fp32 scalar model)
- Final spec: $\kappa_\star=1.5$  (equivalently additive $\rho=0.2$ near 1).
- Final rational parameter: $c_2 = 4.24603987205059$.

### Policy (Gram-side)
Input: SPD $B$ (or $B=G^T G$), start with $Z=I$.

Repeat for up to 2 steps:
1) Form certificate $S=Z^T B Z$ (small side, fp32), symmetrize.
2) Log-center using $c_{\det}(S)$ (Cholesky-only): $Z\leftarrow e^{-c_{\det}(S)/2}Z$.
3) Apply one rational step with chosen $c$:
   - Try $c=c_2$ first (fast path).
   - If certification says still too wide (or if you want a safer wide-basin first step),
     do one "compression" step (choose a larger-basin $c_1$), then finish with $c_2$.
4) Certify $\kappa(S)$ (exact eigs if cheap, else Cholesky-only bound).
5) Stop if $\kappa(S)\le \kappa_\star$ (or if the tier metric $\delta_F$ is satisfied).

Return:
- Gram-side: $Z$ for inverse sqrt use.
- Polar: $\widehat Q = GZ$ (one final apply).

Guards:
- NaN/Inf detection
- Cholesky failure: add tiny ridge $\epsilon I$ and log it
- Non-monotone spikes in $\kappa$ or $\delta_F$: restart with Frobenius safety scaling

---

## 7. Do we still need Frobenius scaling?

Not as the main design axis.

Keep it as:
- a cheap magnitude safety cap (avoid overflow; crude $\lambda_{\max}$ upper bound),
- a fallback restart when something goes unstable,
- an optional comparison baseline.

The primary path is condition-number + log-centering + 1-2 rational steps.

---

## 8. Benchmark plan (minimal)

For each snapshot/policy:
- time (median, p95)
- final $\kappa(S)$ and $\eta(S)$
- final $\delta_F$ (optional, for compatibility with prior reporting)
- # small Cholesky factorizations, # small GEMMs, # tall passes (if polar)
- guard triggers, failures, restarts

Compare:
- Gram-side 1-step vs 2-step (condition-driven stopping),
- Direct odd (GEMM-only) vs Gram-side (solve-heavy),
- Hybrid schedules.

---
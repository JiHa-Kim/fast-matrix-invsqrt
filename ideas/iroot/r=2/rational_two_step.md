# Robust 2-step damped inverse square root for PSD matrices (Gram-side rational)

This document specifies a fixed-cost, fail-closed method to compute a numerically safe approximation to the inverse square root of a symmetric positive semidefinite matrix
$A \succeq 0$, $A\in\mathbb{R}^{d\times d}$.

The design goal is ML-style robustness: **avoid catastrophic failures** (NaNs, Cholesky breakdowns) with predictable cost, while tolerating mild bias and finite-precision error.

---

## 1. What we compute

For a SPD matrix $A$, the exact $A^{-1/2}$ may not exist if $A$ is singular or has issues from finite precision. We therefore compute a **damped** inverse square root:
$$
(A+\lambda I)^{-1/2},
$$
with $\lambda>0$ chosen automatically from a scale-free damping factor $\tau$.

We output a matrix $Z$ such that the "whitened certificate"
$$
S := Z^T A Z
$$
is close to a scaled identity (in practice we keep it near $I$ via centering). The main interpretation in the general PSD case is: $Z$ is a stable preconditioner/whitener for the metric induced by $A$.

Special case: if $A=G^T G$, then $GZ$ is an approximate polar-like factor with near-orthonormal columns.

---

## 2. Damping and normalization (the main safety lever)

Compute the scale:
$$
\mu := \frac{1}{d}\operatorname{tr}(A).
$$

Choose $\lambda$ via a dimensionless damping factor $\tau$:
$$
\lambda := \tau \mu.
$$

Define the normalized damped matrix:
$$
\tilde A := \frac{A+\lambda I}{\mu+\lambda}
= \frac{A+\tau\mu I}{(1+\tau)\mu}.
$$

Then:
- $\tilde A \succ 0$ for any $\tau>0$ (even if $A$ is singular or slightly indefinite from roundoff).
- $\frac{1}{d}\operatorname{tr}(\tilde A)=1$, so the small-side magnitudes are well-scaled.

Conservative worst-case coverage cap:
$$
\kappa(\tilde A)\le \frac{d+\tau}{\tau}\approx \frac{d}{\tau}.
$$

We will design the 2-step contraction coefficients to cover $\kappa_0 := d/\tau$ (the conservative cap).

---

## 3. The 2-step rational update

Maintain $Z\in\mathbb{R}^{d\times d}$ and repeatedly form
$$
S := Z^T \tilde A Z.
$$

For a scalar parameter $c>0$ define
$$
q_c(x)=\frac{x+c}{cx+1}.
$$

(This is inspired from Halley's third-order rational iteration. You can try adding another scalar parameter but this is already very good and easier to solve). Update rule:
$$
Z \leftarrow Z\,q_c(S)
= Z\,(S+cI)\,(cS+I)^{-1}.
$$

We run exactly two steps with coefficients $(c_1,c_2)$.

Each step costs:
- form $S$ (small GEMM),
- Cholesky factorization of $M := cS + I$,
- solve many RHS,
- update $Z$ (small GEMM).

---

## 4. Centering (scale control) and the "snake eating its tail" issue

Scaling $Z$ by a scalar does not change conditioning of $S$ in an essential way, but improves numerical stability.

### 4.1 Default: trace-centering (cannot fail)

Given $S$:
$$
\alpha := \sqrt{\frac{d}{\operatorname{tr}(S)}},\qquad
Z \leftarrow \alpha Z.
$$

This is additive centering of eigenvalues, it keeps the arithmetic mean eigenvalue $\frac{1}{d}\operatorname{tr}(S)$ of $S$ near 1 and prevents scale drift without requiring any additional factorizations.

### 4.2 Optional: logdet-centering (more elegant)

If you want log-symmetric centering:
$$
c_{\det}(S)=\frac{1}{d}\log\det(S),\qquad
Z \leftarrow e^{-c_{\det}(S)/2}Z.
$$

This is multiplicative centering of eigenvalues (the "proper" approach), it keeps the geometric mean eigenvalue $\det(S)^{1/d}$ of $S$ near 1. To avoid "need to scale before Cholesky, but need Cholesky to get logdet", pre-scale using a scalar you can compute without Cholesky, e.g. $s=\operatorname{tr}(S)/d$:
$$
\tilde S = \frac{S}{s},\qquad
\log\det(S)=\log\det(\tilde S) + d\log s.
$$
So you can always trace-scale first, then do a stable Cholesky on $\tilde S$ to compute logdet.

In the robust fixed-cost pipeline below, trace-centering is typically sufficient.

---

## 5. Fail-closed restart policy (eliminate catastrophes)

Never limp forward after PD failure. Restart with more damping.

If any Cholesky fails (typically on $M=cS+I$):
- increase $\tau \leftarrow 4\tau$ (power-of-two jump),
- rebuild $\tilde A$,
- restart both steps from $Z=I$.

Suggested ladder for default $\tau=2^{-8}$:
$$
2^{-8}\to 2^{-6}\to 2^{-4}.
$$

If all levels fail, fail closed to a safe fallback (e.g. diagonal scaling or $Z=\mu^{-1/2}I$) so downstream never crashes.

---

## 6. Undo normalization (returning a preconditioner for the original scale)

We iterate on $\tilde A$, so $Z$ approximates $\tilde A^{-1/2}$.

Since
$$
\tilde A = \frac{A+\lambda I}{\mu+\lambda},
$$
we have
$$
(A+\lambda I)^{-1/2} \approx \frac{1}{\sqrt{\mu+\lambda}}\,Z.
$$

So define
$$
Z_{\text{full}} := \frac{Z}{\sqrt{\mu+\lambda}}.
$$

This $Z_{\text{full}}$ is the damped inverse square root preconditioner for $A$.

---

## 7. Coefficient design model and reproducibility

We design $(c_1,c_2)$ using a scalar log-width model. For a given coverage cap $\kappa_0 = d/\tau$, let
$$
\eta_0^{\max}=\tfrac12\log\kappa_0.
$$

Define the scalar map
$$
\phi_c(x)=x\left(\frac{x+c}{cx+1}\right)^2,
$$
and the induced worst-case log-width (for a log-centered band) as
$$
\eta_+(c,\eta)=\left|\log\phi_c(e^\eta)\right|.
$$

Two-step width:
$$
\eta_2(\eta;c_1,c_2)=\eta_+(c_2,\eta_+(c_1,\eta)).
$$

Design objective (min-max):
$$
(c_1,c_2)\in\arg\min_{c_1,c_2>0}\ \max_{0\le \eta\le \eta_0^{\max}} \eta_2(\eta;c_1,c_2).
$$

We report the model-predicted worst-case output conditioning as
$$
\kappa_{\text{out,model}}=\exp\big(2\max_{\eta\in[0,\eta_0^{\max}]}\eta_2(\eta)\big).
$$

---

## 8. Tau and dimension

We use $\tau=2^{-8}$ by default:
$$
\tau=2^{-8}=0.00390625.
$$

The conservative coverage cap is
$$
\kappa_0 := \kappa(\tilde A) = \frac{d+\tau}{\tau} = \frac{d}{\tau}+1.
$$

Coefficients drift with $d$ because the design covers $\kappa_0$, but the drift is slow since the iteration behaves in log space:
$$
\eta_0^{\max}=\tfrac12\log\kappa_0=\tfrac12\log\left(\frac{d}{\tau}+1\right),
$$
so increasing $d$ only adds $\tfrac12\log d$ to the width.

**Example ($d=7168$):** $\kappa_0=256d=1{,}835{,}008$. Interpolate in $\log_2(d)$ between lookup table entries at $4096$ and $8192$.
With $(c_1,c_2)_{4096}=(16.1549,3.37417)$ and $(c_1,c_2)_{8192}=(18.0646,3.43539)$,
$$
t=\log_2(7168/4096)\approx 0.807,
$$
giving
$$
(c_1,c_2)\approx(17.70,\ 3.42).
$$

---

## 9. Coefficients for tau = 2^-8 (generated offline)

Generated with:

`python ./coeffs/inv_sqrt/rational.py --tau 2**-8 --dim-powers 8 14 --out coeffs_tau2m8.json`

Results:

|     d | kappa0 = d/tau |      c1 |      c2 | kappa_out_model |
| ----: | -------------: | ------: | ------: | --------------: |
|   256 |          65536 |    10.3 | 3.17578 |         1.02697 |
|   512 |         131072 | 11.4995 | 3.21621 |         1.03669 |
|  1024 |         262144 | 12.8589 | 3.26368 |         1.04856 |
|  2048 |         524288 | 14.3903 | 3.31515 |         1.06298 |
|  4096 |        1048576 | 16.1549 | 3.37417 |         1.08029 |
|  8192 |        2097152 | 18.0646 | 3.43539 |         1.10020 |
| 16384 |        4194304 | 20.2638 | 3.50471 |         1.12296 |

Notes:
- Use these as a lookup table keyed by $d$ with log2 interpolation for intermediate dims (e.g. 7168).
- If you implement the restart ladder that increases $\tau$ on failure, coefficients designed for the smallest $\tau$ remain safe when $\tau$ increases (the problem only gets easier).

---

## 10. Full algorithm (pseudocode, SPD $A$)

Inputs: symmetric $A\succeq 0$, damping $\tau$, coeffs $(c_1,c_2)$.

1. Symmetrize: $A\leftarrow \tfrac12(A+A^T)$.
2. For $\tau$ in a bounded ladder (e.g. $\tau,4\tau,16\tau$):
   1. $\mu=\operatorname{tr}(A)/d$. If invalid: return fallback.
   2. $\lambda=\tau\mu$.
   3. $\tilde A=(A+\lambda I)/(\mu+\lambda)$.
   4. Set $Z=I$.
   5. For $c$ in $\{c_1,c_2\}$:
      1. $S=\mathrm{sym}(Z^T\tilde A Z)$.
      2. Trace-center: $Z\leftarrow \sqrt{d/\operatorname{tr}(S)}\,Z$.
      3. $M=\mathrm{sym}(I+cS)$.
      4. Cholesky $M=LL^T$. If fail: restart with larger $\tau$.
      5. Solve $X=(cS+I)^{-1}(S+cI)$ via $L$.
      6. Update $Z\leftarrow ZX$.
   6. Return $Z_{\text{full}}=Z/\sqrt{\mu+\lambda}$.
3. If all $\tau$ fail: return safe fallback.

---

## 11. Implementation checklist (robustness)

- Accumulate $A$ (if formed from data) in fp32, keep small-side ops in fp32.
- Always symmetrize $A$, $S$, $M$: $X\leftarrow \tfrac12(X+X^T)$.
- Use trace-centering each step (cannot fail).
- Fail closed: on any Cholesky failure, increase $\tau$ and restart.
- Clamp $\mu$ away from zero if needed for extremely small matrices.
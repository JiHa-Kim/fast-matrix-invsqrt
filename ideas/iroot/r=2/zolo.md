# Fast finite-precision Gram-side inverse square root and polar: 1-step Zolotarev + 2-step fail-closed fallback

This document specifies a fixed-cost, fail-closed method to compute a numerically safe approximation to the damped inverse square root of a symmetric positive semidefinite matrix $A \succeq 0$, $A\in\mathbb{R}^{d\times d}$, optimized for ML workloads on GPUs.

Design goals:
- **Minimize wall time to reach a target applied conditioning**, not raw inverse-root error.
- Use **bf16 GEMMs for throughput**, but keep **small-side SPD math in fp32** to avoid divergence.
- Prefer **1 solve** (Cholesky factorization + many RHS) in the common case.
- Provide a **deterministic 2-solve fallback** for extreme or numerically unstable cases.

---

## 1. What we compute

We compute a **damped** inverse square root
$$
(A+\lambda I)^{-1/2},
$$
and return a matrix $Z$ such that the whitened certificate
$$
S := Z^T A Z
$$
has condition number near a target $\kappa_\star$ (e.g. $\kappa_\star=1.5$).

Special case (polar-like): if $A=G^T G$ and $m \ge d$, then $Q := GZ$ has near-orthonormal columns and approximates the polar factor $G(G^T G)^{-1/2}$.

---

## 2. Damping and normalization (safety lever)

Compute the scale
$$
\mu := \frac{1}{d}\operatorname{tr}(A).
$$

Choose a dimensionless damping factor $\tau>0$ and set
$$
\lambda := \tau \mu.
$$

Define the normalized damped matrix
$$
\tilde A := \frac{A+\lambda I}{\mu+\lambda}
= \frac{A+\tau\mu I}{(1+\tau)\mu}.
$$

Then:
- $\tilde A \succ 0$ for any $\tau>0$ (even if $A$ is singular or slightly indefinite from roundoff).
- $\frac{1}{d}\operatorname{tr}(\tilde A)=1$, so small-side magnitudes are well-scaled.

Conservative coverage cap (rarely tight, but always safe):
$$
\kappa(\tilde A)\le \frac{d+\tau}{\tau}\approx \frac{d}{\tau}.
$$

We will use a fail-closed restart ladder on $\tau$ if any factorization becomes numerically unsafe.

---

## 3. Quality metric and target band

The primary objective is the applied certificate conditioning
$$
\kappa(S) := \frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}.
$$

Equivalently, singular values of the whitened factor lie in the band
$$
\sigma(Z^T A^{1/2}) \subset \left[\frac{1}{\sqrt{\kappa(S)}},\sqrt{\kappa(S)}\right].
$$

We target $\kappa(S)\le \kappa_\star$ (example: $\kappa_\star=1.5$).

---

## 4. Centering (scale control, mandatory)

Scalar scaling of $Z$ does not change $\kappa(S)$, but prevents drift and improves numerical behavior.

We use **trace-centering** each time we form $S$:
$$
\alpha := \sqrt{\frac{d}{\operatorname{tr}(S)}},\qquad Z \leftarrow \alpha Z.
$$

This cannot fail and costs only a trace reduction.

Optional (debug / validation): logdet-centering
$$
c_{\det}(S)=\frac{1}{d}\log\det(S),\qquad Z \leftarrow e^{-c_{\det}(S)/2}Z,
$$
but this requires a Cholesky of $S$ and is typically too expensive for the inner loop when solves dominate.

---

## 5. Primary step: 1-solve scaled Zolotarev map (type $(7,6)$)

### 5.1 Scalar model

After scaling so that $\sigma_{\max}(X)=1$, suppose $\sigma(X)\subset[\ell,1]$ where $\ell=1/\kappa(G)$ in the Gram-side polar interpretation.

The scaled Zolotarev map $\hat Z_{2r+1}(x;\ell)$ is minimax optimal for shrinking $[\ell,1]$ toward 1, and has the odd rational structure
$$
\hat Z_{2r+1}(x;\ell) = x\,q(x^2),\qquad q(t)=\alpha\frac{P(t)}{Q(t)},\quad \deg P=\deg Q=r.
$$

For the primary path we use $r=3$ (type $(7,6)$). In the scalar worst-case band model, designing $q$ for $\kappa_\star=1.5$ yields coverage around
$$
\kappa(G) \approx 3000,\qquad \kappa(S)\approx \kappa(G)^2 \approx 9\times 10^6,
$$
while landing in the target band in one step.

Interpretation: this covers the typical regime we care about, with only one solve.

### 5.2 Matrix update (one solve)

Maintain $Z\in\mathbb{R}^{d\times d}$ and form
$$
S := Z^T \tilde A Z.
$$

Define
$$
q(t)=\alpha\frac{P(t)}{Q(t)},\qquad \deg P=\deg Q=3,
$$
with coefficients generated offline.

Update:
$$
Z \leftarrow Z\,q(S) = Z\,P(S)\,Q(S)^{-1}.
$$

Implementation shape:
- Build $S$ in fp32 (bf16 inputs, fp32 accumulate).
- Build $P(S)$ and $Q(S)$ in fp32 (use $S^2,S^3$ or Horner).
- Symmetrize $Q(S)\leftarrow \tfrac12(Q(S)+Q(S)^T)$.
- Cholesky factorize $Q(S)=LL^T$ (fp32).
- Right-solve $ZP(S)$ against $Q(S)$ using $L$ (many RHS).
- Cast $Z$ to bf16 only if needed for downstream GEMMs.

This is exactly "1 inverse + spam GEMMs".

---

## 6. Fallback: minimax 2-step Zolotarev (type $(5,4)$ then $(5,4)$)

If the primary step diverges or fails (Section 7), run a deterministic 2-step minimax-optimal fallback:

- Step 1: $r_1=2$ (type $(5,4)$) designed for the initial endpoint $\ell_0$.
- Step 2: $r_2=2$ (type $(5,4)$) re-parameterized by the new endpoint $\ell_1$ produced by step 1.

This "re-parameterize at the midpoint" structure is crucial: each step is minimax optimal on the interval it actually sees.

In the scalar worst-case band model, for $\kappa_\star=1.5$ this 2-step covers extremely large spreads (orders of magnitude beyond anything meaningful without damping), while keeping per-step degree small, which is important for stability in finite precision.

Fallback update:
$$
Z \leftarrow Z\,q_1(S)\quad\text{then}\quad Z \leftarrow Z\,q_2(S),
$$
where each $q_i(t)=\alpha_i P_i(t)/Q_i(t)$ has $\deg P_i=\deg Q_i=2$ and coefficients are generated offline as a matched pair.

Cost: two solves, but much smaller per-step degree and much more robust.

---

## 7. Divergence detection and fail-closed policy

We prioritize "never crash downstream" over squeezing the last bit of accuracy.

### 7.1 Divergence triggers (any is a failure)

Treat as failure and switch to fallback (or restart with larger $\tau$):
- Cholesky factorization fails for $Q(S)$ (not PD) or produces NaN/Inf.
- Large ridge is required: if $Q(S)+\epsilon I$ needs $\epsilon$ above a tiny threshold (choose conservatively).
- A cheap spread proxy gets worse (choose one):
  - $\|S-I\|_F$ increases after trace-centering, or
  - the max absolute diagonal deviation $\max_i |S_{ii}-1|$ spikes, or
  - a few power-iteration steps suggest $\kappa(S)$ increased.

### 7.2 Restart ladder on damping

Never limp forward after PD failure. Restart with more damping.

If primary or fallback fails at any solve:
- increase $\tau \leftarrow 4\tau$,
- rebuild $\tilde A$,
- restart from $Z=I$.

Suggested ladder for default $\tau=2^{-8}$:
$$
2^{-8}\to 2^{-6}\to 2^{-4}.
$$

If all levels fail: return a safe fallback, e.g.
$$
Z = (\mu+\lambda)^{-1/2}I
$$
(or diagonal scaling), so downstream never crashes.

---

## 8. Undo normalization (return preconditioner for original scale)

We iterate on $\tilde A$, so $Z \approx \tilde A^{-1/2}$.

Since
$$
\tilde A = \frac{A+\lambda I}{\mu+\lambda},
$$
we return
$$
Z_{\text{full}} := \frac{Z}{\sqrt{\mu+\lambda}}
\approx (A+\lambda I)^{-1/2}.
$$

---

## 9. Coefficient generation and reproducibility

We store coefficients offline in JSON.

### 9.1 Primary (1-step, type $(7,6)$)

For target $\kappa_\star$ and degree $r=3$, solve for the largest endpoint $\ell$ such that
$$
\hat Z_{7}(\ell;\ell)=\frac{1}{\sqrt{\kappa_\star}},
$$
then export $q(t)=\alpha P(t)/Q(t)$.

Model coverage:
$$
\kappa(G)_{\max} \approx \frac{1}{\ell},\qquad \kappa(S)_{\max}\approx \frac{1}{\ell^2}.
$$

### 9.2 Fallback (2-step, type $(5,4)+(5,4)$)

Solve for $\ell_0$ such that
$$
\ell_2=\hat Z_{5}(\ell_1;\ell_1),\qquad \ell_1=\hat Z_{5}(\ell_0;\ell_0),\qquad
\ell_2=\frac{1}{\sqrt{\kappa_\star}}.
$$

Export step 1 coefficients from $\ell_0$, and step 2 coefficients from $\ell_1$.

---

## 10. Full algorithm (pseudocode)

Inputs: symmetric $A\succeq 0$, target $\kappa_\star$, damping ladder $\tau$, coefficient sets.

1. Symmetrize: $A\leftarrow \tfrac12(A+A^T)$.
2. For $\tau$ in a bounded ladder (e.g. $\tau,4\tau,16\tau$):
   1. $\mu=\operatorname{tr}(A)/d$. If invalid: return safe fallback.
   2. $\lambda=\tau\mu$.
   3. $\tilde A=(A+\lambda I)/(\mu+\lambda)$.
   4. Set $Z=I$.
   5. Try primary 1-step:
      1. $S=\mathrm{sym}(Z^T\tilde A Z)$.
      2. Trace-center: $Z\leftarrow \sqrt{d/\operatorname{tr}(S)}\,Z$.
      3. Build $P(S),Q(S)$ for primary $(7,6)$ in fp32.
      4. Cholesky $Q(S)=LL^T$. If fail: go to fallback.
      5. Update $Z\leftarrow Z\,P(S)\,Q(S)^{-1}$.
      6. If post-check passes: return $Z_{\text{full}}=Z/\sqrt{\mu+\lambda}$.
   6. Fallback 2-step:
      1. For step in {fallback1, fallback2}:
         1. $S=\mathrm{sym}(Z^T\tilde A Z)$.
         2. Trace-center: $Z\leftarrow \sqrt{d/\operatorname{tr}(S)}\,Z$.
         3. Build $P(S),Q(S)$ for this step in fp32.
         4. Cholesky $Q(S)=LL^T$. If fail: restart with larger $\tau$.
         5. Update $Z\leftarrow Z\,P(S)\,Q(S)^{-1}$.
      2. If post-check passes: return $Z_{\text{full}}=Z/\sqrt{\mu+\lambda}$.
3. If all $\tau$ fail: return safe fallback.

---

## 11. Implementation checklist (robustness and performance)

- Keep $A$, $\tilde A$, $S$, $P(S)$, $Q(S)$ in fp32 (bf16 inputs, fp32 accumulate is fine).
- Always symmetrize $A$ and any SPD matrix before Cholesky: $X\leftarrow \tfrac12(X+X^T)$.
- Use trace-centering before each solve (cheap and cannot fail).
- Ridge policy: if Cholesky fails, try $Q\leftarrow Q+\epsilon I$ with tiny $\epsilon$ once; if still fails, treat as failure.
- Fail closed: on any failure, increase $\tau$ and restart from $Z=I$.
- Prefer 1-step primary to minimize solves; use fallback only on divergence or extreme spread.
- Keep degrees small in bf16 regimes: primary $r=3$, fallback $r=2$ is intentionally conservative.
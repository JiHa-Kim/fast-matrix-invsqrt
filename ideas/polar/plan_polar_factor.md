# Fast finite-precision polar and inverse roots: lean execution plan

## 1. Goal

Compute an ML-useful approximation to the polar factor
$$
Q = \operatorname{polar}(G) = G(G^T G)^{-1/2}
$$
as fast as possible in finite precision.

Primary objective:
$$
\text{minimize wall time to target applied quality, not raw approximation error.}
$$

The same framework should later extend to SPD inverse square roots, inverse $r$-th roots, and applied transforms such as
$$
G P^{-s/r}.
$$

---

## 2. What quality means

For $m \ge n$, define
$$
B := G^T G,
\qquad
\widehat Q := GZ,
\qquad
S := \widehat Q^T \widehat Q = Z^T B Z.
$$

For $m < n$, swap sides and certify on the smaller side.

The error certificate is
$$
E := S - I.
$$

Main metrics:
$$
\rho_2 := \|E\|_2,
\qquad
\delta_F := \|E\|_F.
$$

Always,
$$
\rho_2 \le \delta_F \le \sqrt{r}\,\rho_2,
\qquad
r := \min(m,n).
$$

If $\rho_2 < 1$, then
$$
\kappa(S) \le \frac{1+\rho_2}{1-\rho_2}.
$$

Useful log-conditioning coordinate:
$$
\eta := \frac12 \log\!\left(\frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}\right).
$$
For a symmetric band $[1-\rho,1+\rho]$,
$$
\eta = \frac12 \log\!\left(\frac{1+\rho}{1-\rho}\right) = \operatorname{atanh}(\rho).
$$

Target tiers:
$$
\text{light: } \delta_F \le 0.35 \quad (\text{or } 0.5 \text{ if very cheap}),
$$
$$
\text{medium: } \delta_F \le 0.20,
\qquad
\text{strong: } \delta_F \le 0.10.
$$

---

## 3. Compare policies, not isolated polynomials

We compare three policy families.

### A. Direct
Update $G$ directly with odd matrix polynomials, e.g.
$$
X_+ = X q(X^T X).
$$
For the cubic family,
$$
X_+ = aX + bX(X^T X) + cX(X^T X)^2.
$$

### B. Gram-side
Form
$$
B = G^T G,
$$
refine only
$$
Z \approx B^{-1/2},
$$
then apply once:
$$
\widehat Q = GZ.
$$

### C. Hybrid
Do a small number of direct global-compression steps, then switch to small-side refinement and apply once at the end.

The winner is the policy with the best time-to-target and acceptable tail risk.

---

## 4. Core exact fact

For a local step
$$
Z_+ = Z q(S),
\qquad
S = Z^T B Z,
$$
we get
$$
S_+ = q(S) S q(S).
$$

So certificate eigenvalues evolve by the scalar map
$$
x \mapsto x q(x)^2.
$$

The same map appears for direct odd updates on $G$. Therefore direct and Gram-side methods share the same exact local scalar dynamics. The difference is cost, state location, certification, and finite-precision behavior.

For a candidate $q$, define the local contraction function
$$
m_q(\rho) := \sup_{x \in [1-\rho,1+\rho]} |x q(x)^2 - 1|.
$$

Exact arithmetic guarantee:
if
$$
\rho_2(S) \le \rho,
$$
then after one step
$$
\rho_2(S_+) \le m_q(\rho).
$$

---

## 5. Default design: two phases

### Phase 1: global compression
Use a cheap, wide-basin, bf16-safe policy to move the spectrum into a profitable local basin.

Default candidates:
- direct odd degree-$3$ or degree-$5$ schedules,
- simple adaptive direct schedules,
- Frobenius normalization with a safety factor,
- Gram-side scaling or preprocessing when clearly helpful.

### Phase 2: local finish
Use one or two aggressive local steps designed from the certificate map.

Default local model:
- center at $1$,
- represent $q$ in shifted Chebyshev form,
- evaluate with Clenshaw,
- optimize the deployed scalar step directly in bf16.

Rationale: centering at $1$ avoids interval remapping noise and keeps the local coordinate naturally in the stable Chebyshev region.

---

## 6. Exact bf16 local design problem

For the local scalar step, fix the deployed model
$$
t = \operatorname{rn}_{bf16}(x-1),
$$
$$
q(x) = \sum_{j=0}^d c_j T_j(t),
$$
with bf16 Clenshaw evaluation, and deployed certificate map
$$
\Phi(x) := \operatorname{rn}_{bf16}\!\left(x \cdot \operatorname{rn}_{bf16}(q(x)^2)\right).
$$

Let the terminal target set be
$$
\mathcal T_\tau := \{ y \in \mathrm{BF16} : |y-1| \le \tau \}.
$$

Then for each degree $d$, solve the discrete predecessor problem:
find the largest contiguous bf16 input band around $1$ such that
$$
\forall x \in \mathcal X_d,
\qquad
\Phi(x) \in \mathcal T_\tau.
$$

This is the correct finite-precision local object.

Use the log-width score
$$
\eta(\mathcal X_d) := \frac12 \log\!\left(\frac{x_{\max}}{x_{\min}}\right)
$$
for the band $\mathcal X_d = [x_{\min},x_{\max}] \cap \mathrm{BF16}$.

Why this score:
- it matches conditioning,
- it reduces to $\operatorname{atanh}(\rho)$ for a symmetric continuous band,
- it is more meaningful than plain additive radius once the band is not tiny.

Practical rule:
- optimize $\eta$ for raw basin width,
- optimize $\eta / \text{cost}(d)$ for step efficiency.

Default degree search for the first step:
$$
d \in \{2,3,4\}.
$$

---

## 7. Finite-precision stance

Separate three levels.

### A. Exact arithmetic theory
This gives the rigorous scalar map
$$
x \mapsto x q(x)^2
$$
and guarantees through $m_q(\rho)$.

### B. Exact scalar bf16 deployment model
This is the centered-at-$1$ Chebyshev + bf16 Clenshaw predecessor solve above.

### C. Real matrix-kernel deployment
This includes GEMM accumulation, reduction order, casts, and backend details. It must be calibrated empirically.

Do not claim:
- a universal bf16 floor,
- a universal GEMM perturbation constant,
- theorem-level deployed guarantees for full matrix kernels without calibration.

---

## 8. Certification and guards

Always certify on the small side.

Symmetrize before spectral measurement:
$$
S_{\mathrm{sym}} := \tfrac12(S + S^T).
$$

Then measure in fp32 or fp64 when cheap enough:
$$
\rho_2 = \|S_{\mathrm{sym}} - I\|_2,
\qquad
\delta_F = \|S_{\mathrm{sym}} - I\|_F.
$$

Every deployed policy should include:
- NaN or Inf detection,
- non-monotone spike detection,
- overshoot detection,
- one restart or rescale path,
- one safer fallback.

---

## 9. Cost model and regime split

Assume $m \ge n$.

A direct odd step costs roughly:
- one tall Gram-like build,
- small-side products,
- one tall apply back to $G$.

A Gram-side policy costs:
- one-time formation of $B = G^T G$,
- only small-side refinement after that,
- one final apply $GZ$.

So the regime split is empirical.

Key break-even comparison:
- one more direct step,
versus
- one more small-side step plus the final apply.

Expectation:
- direct may win near square or for light targets,
- Gram-side may win for very rectangular matrices,
- hybrid may win in the middle.

But this is a benchmark question, not a theorem.

---

## 10. Minimal benchmark plan

### Synthetic coverage
Sweep:
$$
\frac{m}{n} \in \{1,2,4,8,16,32\},
$$
with several $n$ values and spectra such as:
- flat,
- moderate decay,
- severe ill-conditioning,
- clustered endpoints,
- two-mass adversarial mixtures.

### Real snapshots
Use saved training matrices or Gram snapshots whenever possible.

### Record per run
- wall time,
- median and $p95$ over repeats,
- final $\rho_2$ and $\delta_F$,
- number of tall passes,
- number of small-side GEMMs,
- switch point for hybrid,
- scaling used,
- guard triggers,
- failures and fallback use,
- monotonicity of the certificate.

---

## 11. Default execution order

### Step 1
Lock the local evaluator to centered-at-$1$ shifted Chebyshev with Clenshaw.

### Step 2
Solve the exact scalar bf16 predecessor problem for degrees $2,3,4$ and pick:
- the widest local basin,
- the best $\eta / \text{cost}$ tradeoff.

### Step 3
Build a minimal Phase 1 baseline using direct odd updates with simple safe scaling.

### Step 4
Benchmark three policy families:
- direct,
- Gram-side,
- hybrid.

### Step 5
For each target tier, ship:
- one default fast policy,
- one safer fallback.

---

## 12. What not to optimize directly

Do not optimize
$$
\|Z - B^{-1/r}\|
$$
unless it clearly improves the applied objective.

For polar and whitening-style use, the primary object is
$$
S = Z^T B Z \approx I,
$$
or equivalently
$$
\widehat Q^T \widehat Q \approx I.
$$

So optimize applied whitening quality and time, not abstract root approximation in isolation.

---

## 13. Immediate tasks

1. Finish the exact scalar bf16 predecessor solve for degrees $2,3,4$.
2. Choose the best first local step by $\eta$ and by $\eta / \text{cost}$.
3. Implement a minimal direct Phase 1 baseline with certification.
4. Add a Gram-side baseline with final apply.
5. Benchmark direct vs Gram-side vs hybrid on aspect-ratio sweeps and real snapshots.
6. Ship one default fast policy and one fallback per target tier.

---

## 14. One-sentence project statement

Find the fastest finite-precision policy for producing an ML-useful polar factor approximation by combining bf16-safe global compression, exact local certificate design, small-side certification, and aspect-ratio-aware switching between direct, Gram-side, and hybrid policies.

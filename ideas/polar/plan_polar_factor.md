# Fastest path to ML-useful polar factor:
# direct odd polynomial vs preconditioner-first vs hybrid

## 0. Core question

Given $G \in \mathbb{R}^{m \times n}$, what is the fastest way to produce an update direction that is "polar enough" for ML?

Current baseline:
$$
Q \approx \mathrm{polar}(G) = G(G^T G)^{-1/2}
$$
via direct odd matrix polynomials on $G$:
$$
X_{k+1}
= a_k X_k + b_k X_k(X_k^T X_k) + c_k X_k(X_k^T X_k)^2.
$$

Alternative:
treat
$$
B := G^T G
$$
as the primary object, compute or apply
$$
Z \approx B^{-1/2},
$$
and return
$$
Q = GZ.
$$

The goal is not best asymptotic numerical analysis in the classical sense. The goal is:

- minimum wall-clock to reach an ML-relevant whitening target,
- under bf16-safe rules,
- with runtime certificates on the small side.

## 1. Competing hypotheses

### H1. Direct odd polynomial on $G$ is fastest for loose targets and moderate aspect ratios
If the target is only "singular values close enough to 1" and $m$ is not much larger than $n$, directly updating $G$ may still win because it avoids:
- explicitly forming/storing a separate preconditioner,
- a final apply $GZ$,
- extra small-side refinement logic.

### H2. Preconditioner-first wins for very rectangular matrices
If $m \gg n$, then moving iteration to the $n \times n$ side may be cheaper:
- form $B = G^T G$ once,
- refine $Z$ using only $n \times n$ GEMMs,
- apply $Q = GZ$ once at the end.

The expected advantage grows with aspect ratio because direct odd iteration pays tall-side cost every iteration, while preconditioner-first pays most of that cost only at the beginning and end.

### H3. The true optimum may be hybrid
A likely best policy is:
- do 1-2 direct odd steps on $G$ to compress the singular spectrum quickly,
- switch to a preconditioner-side local refinement on $B = G^T G$,
- finish with 1 final apply.

This hybrid should be tested explicitly rather than assumed away.

## 2. Exact maps to compare

### 2.1 Direct odd polynomial family on $G$
Let
$$
A_k := X_k^T X_k,
\qquad
p_k(t) := a_k + b_k t + c_k t^2.
$$
Then
$$
X_{k+1} = X_k p_k(A_k).
$$

If
$$
X_k = U \Sigma_k V^T,
$$
then
$$
X_{k+1} = U \big(\Sigma_k p_k(\Sigma_k^2)\big) V^T.
$$
So singular values evolve by the scalar map
$$
\sigma \mapsto \sigma\, p_k(\sigma^2).
$$

This is the natural design object for direct polar iterations.

### 2.2 Preconditioner-side family on $B$
Let
$$
B = G^T G,
\qquad
S_k := Z_k^T B Z_k.
$$
Update
$$
Z_{k+1} = Z_k q_k(S_k).
$$
Then
$$
S_{k+1} = q_k(S_k)\, S_k\, q_k(S_k),
$$
so eigenvalues evolve by
$$
\lambda \mapsto \lambda\, q_k(\lambda)^2.
$$

This gives a clean small-side certificate:
$$
\delta_F := \|S_k - I\|_F,
\qquad
\rho_2 := \|S_k - I\|_2.
$$

The main attraction of the preconditioner route is that certificate, switching logic, and local refinement all live on the small side.

## 3. Cost model to test, not assume

### 3.1 Direct odd step on $G$
For $m \ge n$:
- $A = X^T X$: cost scale $m n^2$
- $A^2$: cost scale $n^3$
- $X \leftarrow aX + X(bA + cA^2)$: another $m n^2$

So each direct step is roughly:
$$
2\,m n^2 + n^3
$$
up to fusion/overhead constants.

### 3.2 Preconditioner-first step on $B$
One-time:
- form $B = G^T G$: cost scale $m n^2$

Per refinement step:
- work only with $n \times n$ matrices:
  certificate formation, local polynomial evaluation, update of $Z$

Final:
- apply $Q = GZ$: cost scale $m n^2$

So preconditioner-first is attractive when:
- many refinement steps are needed,
- or $m/n$ is large,
- or local finishing on the small side is much more aggressive than direct singular-value compression.

### 3.3 Hybrid
Hybrid cost model:
- 1-2 direct steps on $G$
- switch to small-side refinement
- final apply once

This is likely the highest-payoff unexplored regime.

## 4. What "better" means

We are not optimizing raw approximation error to $B^{-1/2}$.
We optimize the applied objective:
$$
Q = GZ,
\qquad
Q^T Q = Z^T G^T G Z = Z^T B Z = S.
$$

Primary metric:
$$
\delta_F = \|S - I\|_F.
$$

Secondary metrics:
$$
\rho_2 = \|S-I\|_2,
\qquad
\kappa(S) \le \frac{1+\rho_2}{1-\rho_2}.
$$

Recommended target tiers:
- light: $\delta_F \le 0.35$
- medium: $\delta_F \le 0.2$
- strong: $\delta_F \le 0.1$

Primary benchmark:
minimum wall time to hit target.

## 5. Policy families to benchmark

### Family D: direct-on-$G$
Policies:
- fixed odd degree-5 with constant coefficients
- Polar-Express-like adaptive odd degree-5
- direct local/global two-phase variants if available

### Family P: preconditioner-first on $B$
Policies:
- global safe scaling on $B$
- small-side local inverse-square-root refinement
- final apply $Q = GZ$

### Family H: hybrid
Policies:
- 1 direct step, then preconditioner-side finish
- 2 direct steps, then preconditioner-side finish
- switch when measured small-side certificate enters a chosen band

## 6. Matrix regimes to sweep

### Synthetic sweeps
Aspect ratios:
$$
m/n \in \{1, 2, 4, 8, 16, 32\}
$$
with fixed $n$ grid.

Spectral shapes:
- flat
- moderately decaying
- highly ill-conditioned
- clustered with a few tiny singular values

### Real ML sweeps
Use saved gradient / momentum snapshots from training.
For each snapshot, log:
- shape
- Frobenius norm
- estimated spectral spread
- runtime to target for each policy

## 7. Design tasks

### Task A. Build a unified benchmark harness
One runner that supports all three families:
- same normalization entry point
- same measurement code
- same stop criteria
- same wall-clock accounting

### Task B. Implement direct odd baseline exactly
Match current best-known direct method:
- same normalization rule
- same coefficient schedule
- same transpose-to-small-side trick
- same bf16 path

### Task C. Implement preconditioner-first baseline
Minimal version:
- form $B = G^T G$
- initialize $Z_0$
- refine using small-side policy
- apply once at the end

### Task D. Implement hybrid switch
Switch rules:
- fixed step count switch
- certificate-based switch
- aspect-ratio-dependent switch

## 8. Decision rules

We should not ask "which method is best in theory?"
We should ask:

1. For each target tier, which policy has the lowest median wall time?
2. For each target tier, which policy has the best p95 wall time and lowest failure rate?
3. At what aspect ratio does preconditioner-first overtake direct?
4. Does hybrid dominate both in the middle regime?

## 9. Deliverables

### Code
- `bench_polar_policies.py`
- `policy_direct_odd.py`
- `policy_precond_inverse_modulus.py`
- `policy_hybrid_switch.py`

### Reports
- `report_polar_cost_model.md`
- `report_polar_sweep_aspect_ratio.md`
- `report_polar_real_snapshots.md`

### Final ship decision
One default policy per target tier:
- fastest-light
- fastest-medium
- fastest-strong
# Applied inverse-root roadmap:
# from inverse modulus to general G P^{-s/r}

## 0. Goal

Generalize the preconditioner-side logic from inverse square root to the applied transform
$$
G P^{-s/r},
$$
with $P \succeq 0$, $r > 0$, and $s > 0$.

This is motivated by optimizer-style uses:
- Shampoo-like preconditioning
- quarter-root / inverse-root style updates
- applied transforms where forming the full matrix function explicitly may be unnecessary

Immediate focus:
- inverse square root ($r=2$, $s=1$)

Next:
- small integer $r$ such as $4$

Later:
- broader real $r > 0$ if still useful

## 1. Core principle

The object of interest is usually not the explicit matrix function itself.
It is the applied transform on a matrix $G$.

So the main question is:

Can we compute
$$
G P^{-s/r}
$$
faster and more stably than:
1. computing $P^{-1/r}$ explicitly, then
2. multiplying by $G$ afterward?

The working hypothesis is yes, especially when:
- the small-side certificate can be maintained directly,
- the update can be written as a coupled polynomial iteration,
- explicit unstable amplification is avoided until the end or avoided entirely.

## 2. Start with the inverse-square-root special case

For $r=2$, the nearest-term target is:
$$
G P^{-1/2}.
$$

This should be developed first because:
- it directly matches the inverse-modulus / whitening problem,
- the local certificate is clean,
- it is the most relevant special case for the polar-factor comparison.

Deliverable:
- a fully benchmarked applied inverse-square-root path before any general-$r$ work.

## 3. Coupled-iteration family to explore

Use the applied coupled template:
$$
G_{t+1} = G_t \big(a_{t+1}I + b_{t+1}P_t + c_{t+1}P_t^2\big)^s,
$$
$$
P_{t+1} = \big(a_{t+1}I + b_{t+1}P_t + c_{t+1}P_t^2\big)^r P_t.
$$

If $P_t$ remains invertible and tends to $I$, then the limit gives the desired applied root.

This suggests the roadmap:
- first solve the "make $P_t \to I$ quickly and safely" problem,
- then lift that design to the coupled applied iteration.

## 4. Initialization / scaling to test

Initialization quality is critical.

Candidates:
- Frobenius scaling
- scaling by $\sqrt{\mathrm{tr}(P^2)}$
- diagonal/Jacobi scaling
- ridge-regularized scaling:
  $$
  P_\lambda = P + \lambda I
  $$

The purpose is to compress eigenvalues into a safe interval for the polynomial map while preserving enough useful gain.

## 5. Policy structure

### Phase 1: global contraction
Use bf16-safe global steps that move the spectrum of the certificate toward $1$ without overshoot.

### Phase 2: local aggressive finish
Once the certificate is close enough to identity, use local minimax-style steps designed on
$$
[1-\rho, 1+\rho].
$$

This is the same structural logic as the inverse-square-root case, but now applied to the coupled iteration.

## 6. What to verify

### Exact-arithmetic verification
For each candidate policy:
- scalar contraction of the relevant map
- interval of validity
- composition behavior across multiple steps

### Applied verification
For the returned transform:
$$
U := G P^{-s/r}_{\text{approx}}
$$
or its coupled-iteration analogue, measure the induced certificate on the appropriate small side.

For inverse square root:
$$
S = Z^T P Z \approx I.
$$

For more general applied roots, define the downstream metric based on the optimizer/preconditioner objective rather than on raw matrix-function error alone.

## 7. Research sequence

### Stage 1. Lock inverse-square-root
- finalize preconditioner-side local design
- compare explicit $Z$ vs coupled applied route
- benchmark against direct polar-style odd iteration where applicable

### Stage 2. Extend to integer $r$
- start with $r=4$
- reuse the two-phase structure
- redesign local maps for the new scalar objective

### Stage 3. Decide whether real $r$ is worth it
General real powers are algebraically possible, but the design space and finite-precision behavior become less clean.
Only pursue this if integer-$r$ results show clear practical value.

## 8. Deliverables

### Code
- `policy_apply_inv_sqrt.py`
- `policy_apply_inv_rth_root.py`
- `bench_applied_roots.py`

### Reports
- `report_inv_sqrt_vs_explicit_root.md`
- `report_integer_r_roots.md`
- `report_scaling_and_ridge.md`

## 9. Final decision criterion

Ship the family that minimizes wall-clock to a downstream-useful certificate under bf16-safe guards.

For this track, "best" means:
- fastest applied preconditioning effect,
- not best raw approximation of $P^{-1/r}$ in isolation.
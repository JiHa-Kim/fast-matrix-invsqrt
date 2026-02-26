# Shared Math + Implementation Tricks

This page captures shared ideas used by all methods.

## 1) Core State and Objective

Given SPD-ish matrix `A`, the goal is to compute `X ≈ A^{-1/p}` for arbitrary `p ∈ {1, 2, 3, 4, ...}`.

Coupled methods maintain:

- `X_t`: current inverse p-th root estimate
- `Y_t`: residual state, ideally `Y_t → I`

Target residual:

$$
X^p A \approx I
$$

## 2) Coupled Update Form

All coupled kernels use a quadratic multiplier `B_t = a_t I + b_t Y + c_t Y²` and update:

$$
X_{t+1} = X_t B_t,\qquad Y_{t+1} = B_t^p Y_t \quad (\text{via binary exponentiation for } p \geq 3)
$$

For `p=2`, the symmetric form `B Y B` is used to preserve symmetry in finite precision.

### Why This Update Matches the Scalar PE Model

In exact arithmetic, coupled PE relies on a commutativity invariant: starting from `X_0 = I` and `Y_0 = A`,
each update uses `B_t = q(Y_t)` where `q` is a polynomial, so `B_t` is a polynomial in `A`. By induction, `X_t`
and `Y_t` remain polynomials in `A`, hence commute with `A` and with each other.

Under the invariant `Y_t = X_t^p A`, the `X` update implies:

$$
Y_{t+1} = X_{t+1}^p A = (X_t B_t)^p A = X_t^p B_t^p A = B_t^p X_t^p A = B_t^p Y_t.
$$

Finite precision caveat: commutativity is only approximate. Symmetrizing `Y` helps, and a pragmatic extra
stabilization option is to periodically refresh `Y := X^p A` (uncoupled recomputation) if drift is observed.

## 3) Terminal Last-Step Optimization

For the last iteration step, only `X <- X B` is required for output.
Updating `Y` in that final step does not change returned `X`, so it is skipped.

Implemented in all core kernels via `terminal_last_step=True`.

## 4) Preconditioning Pipeline

Implemented in `precond_spd(...)`:

1. Optional scaling mode:
- `none`
- `frob`
- `aol` (default, diagonal similarity scaling)

2. Optional ridge:
- adds `ridge_rel * mean(diag)` to diagonal

3. Upper normalization:
row-sum upper bound proxy:

$$
u \;=\; \max_i \sum_j |A_{ij}|
$$

and normalization:

$$
A_{\text{norm}} = \frac{A_{\text{pre}}}{u}
$$

4. Optional floor enforcement:
- Gershgorin-style lower proxy
- diagonal shift if needed to enforce `l_target`
- properly scaling renormalizer by `r + shift` to lock the explicit lower bounds
- **Strict SPD requirement:** `precond_spd` verifies that input matrices have strictly positive diagonals, as required for SPD logic and Gershgorin proxies.

Returned:
- normalized `A_norm`
- `PrecondStats` (`rho_proxy`, `gersh_lo`, `kappa_proxy`)

## 5) Precision and Buffering Tricks

- GEMMs use preallocated workspace tensors to avoid per-iteration allocations.
- `torch.matmul(..., out=...)` and fused `torch.addmm`/`torch.baddbmm` for zero-copy operations.
- Arbitrary batch dimensions (e.g., `(..., n, n)`) are fully supported by internal routines like `_addmm_into` which reshape and fallback to `baddbmm` dynamically.
- Binary exponentiation (`_bpow_times_y`) for O(log p) coupled Y-updates.
- Coefficients are extracted once to CPU scalars/lists to avoid GPU scalar sync overhead.
- Optional symmetrization is applied on `Y` each full step.

## 6) Schedule / Coefficient Tricks

- Precomputed schedules for default target (`l_target=0.05`) are embedded.
- Optional tuned schedules come from `scripts/coeff_tuner.py` (`coeff-mode=tuned`).
- Coefficient safety scaling supported:
  - global safety factor
  - optional no-safety on final step

## 7) Quality Metrics Used by Benchmark Harness

`scripts/matrix_iroot.py` reports:

- wall-clock split: precond + iteration
- residual median / p95 / max
- relative error vs eigendecomp (`metrics-mode=full`)
- optional spectral residual proxy (`power_iters`)
- symmetry diagnostics (`symX`, `symW`)
- optional apply-to-vector metric (`mv_samples`)
- bad count (NaN/Inf)

Method selection is based on practical preconditioning quality and stability, not only theoretical contraction.

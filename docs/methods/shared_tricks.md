# Shared Math + Implementation Tricks

This page captures shared ideas used by all methods.

## 1) Core State and Objective

Given SPD-ish matrix `A`, all methods maintain:

- `X_t`: current inverse-square-root estimate
- `Y_t`: whitened state, ideally `Y_t = X_t A X_t`

Target:

$$
X A X \approx I
$$

Equivalent residual-state goal:

$$
Y \to I
$$

## 2) Coupled Update Form

All kernels use a multiplier `B_t = q_t(Y_t)` and update:

$$
X_{t+1} = X_t B_t,\qquad Y_{t+1} = B_t Y_t B_t
$$

`Y <- B Y B` is used (instead of `Y <- Y B^2`) to preserve symmetry better in finite precision.

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
- renormalize upper bound after shift

Returned:
- normalized `A_norm`
- `PrecondStats` (`rho_proxy`, `gersh_lo`, `kappa_proxy`)

## 5) Precision and Buffering Tricks

- GEMMs use preallocated workspace tensors (`IsqrtWorkspace`) to avoid per-iteration allocations.
- `torch.matmul(..., out=...)` is used for buffer reuse.
- Coefficients are extracted once to CPU scalars/lists to avoid GPU scalar sync overhead.
- Optional symmetrization is applied on `Y` each full step.

## 6) Schedule / Coefficient Tricks

- Precomputed schedules for default target (`l_target=0.05`) are embedded.
- Optional tuned schedules come from `coeff_tuner.py` (`coeff-mode=tuned`).
- Coefficient safety scaling supported:
  - global safety factor
  - optional no-safety on final step

## 7) Quality Metrics Used by Benchmark Harness

`matrix_isqrt.py` reports:

- wall-clock split: precond + iteration
- residual median / p95 / max
- optional spectral residual proxy (`power_iters`)
- symmetry diagnostics (`symX`, `symW`)
- optional apply-to-vector metric (`mv_samples`)
- optional relative error vs eigendecomp (`metrics-mode=full`)

This is deliberate: method selection is based on practical preconditioning quality and stability, not only theoretical contraction.

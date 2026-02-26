# Shared Math + Implementation Notes

This page documents invariants and implementation choices shared by the active methods.

## 1) Target and State

For SPD `A`, we seek `X ≈ A^{-1/p}`.

- Uncoupled: track only `X`.
- Coupled: track `X` and `Y`, with ideal invariant `Y = X^p A`.

Residual target:

$$
X^p A \approx I.
$$

## 2) Coupled Quadratic Form

All coupled PE-Quad kernels use

$$
B_t = a_t I + b_t Y_t + c_t Y_t^2,\quad X_{t+1} = X_t B_t.
$$

`Y` update:

- `p=1`: `Y <- B Y`
- `p=2`: symmetric form `Y <- B Y B`
- general `p`: `Y <- B^p Y` in the commuting polynomial model.

The implementation uses:

- `_bpow` in coupled kernels for `B^h` building.
- `_bpow_times_y` in uncoupled kernels for `X^p A`.

## 3) Terminal Last-Step Optimization

When `terminal_last_step=True`, the final `Y` update is skipped in coupled paths because output uses only final `X`/`Z`.

## 4) Symmetry and Stability Controls

- `symmetrize_Y` toggles `Y <- 0.5*(Y + Y^T)` in coupled methods.
- `symmetrize_every` applies this every `k` non-terminal steps (`k >= 1`).
- `symmetrize_X` is available in uncoupled iteration.

## 5) Preconditioning Pipeline (`precond_spd`)

1. Optional scaling mode: `none`, `frob`, `aol`.
2. Optional ridge (`ridge_rel * mean(diag)`).
3. Upper normalization via row-sum bound.
4. Optional floor enforcement with Gershgorin-style lower proxy + diagonal shift.

Returned stats: `rho_proxy`, `gersh_lo`, `kappa_proxy`.

## 6) Low-Level Performance Choices

- Workspace reuse (`ws`) to avoid repeated allocations.
- `torch.matmul(..., out=...)` and fused `_addmm_into` (`addmm`/`baddbmm`).
- Fast paths for 2D/3D mm/bmm in `_matmul_into`.
- Coefficients unpacked into CPU scalar tuples for low overhead in loops.
- Affine PE step fast path (`c=0`) avoids `Y@Y` GEMM when building `B = aI + bY`.

## 7) Optional Online Coefficient Scheduling (Solve Harness)

`scripts/matrix_solve.py` supports online coefficient scheduling for the
coupled PE apply method:

- `greedy-newton` (default): choose between baseline quadratic and inverse-Newton affine.
- `off`: disable online coefficient adaptation.
- `greedy-minimax`: also evaluates a local-basis minimax-alpha candidate
  `q(y)=1-(1/p)(y-1)+alpha(y-1)^2`, with dominance gating vs inverse-Newton in
  mapped interval log-width.
- Selection uses interval contraction per estimated GEMM cost and keeps baseline
  as fallback unless predicted improvement clears threshold gates.

## 8) Benchmark Metrics

`scripts/matrix_iroot.py` reports:

- total/precond/iter latency medians,
- residual median/p95/max,
- optional relative error (`metrics-mode=full`),
- optional probes (`power_iters`, `mv_samples`, `hard_probe_iters`),
- symmetry diagnostics (`symX`, `symW`),
- `bad` non-finite count.

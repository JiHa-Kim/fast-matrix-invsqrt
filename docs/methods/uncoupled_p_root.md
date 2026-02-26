# Uncoupled Inverse p-th Root

## Idea

Do not persist `Y`. Recompute it each step from the current `X`:

$$
Y_k = X_k^p A,\qquad X_{k+1}=X_k P_k(Y_k),
$$

with quadratic `P_k(y)=a_k + b_k y + c_k y^2`.

## Workspace

`IrootWorkspaceUncoupled` uses 4 dense `(..., n, n)` tensors:

- `X`, `Xbuf`, `T1`, `T2`

Compared with 6 tensors in coupled inverse-root workspace.

## Performance Details

- Fused `_addmm_into` for polynomial evaluation (`aI + bY + cY^2` path).
- Fast exponentiation helpers:
  - specialized chains for `p=1`, `p=2`, `p=4`,
  - `_bpow_times_y` for generic `p`.
- Optional `symmetrize_X` after each step.

## When It Tends to Win

- Accuracy-sensitive runs for higher exponents (`p=8` in current benchmark tables).
- Memory-constrained scenarios where fewer `n x n` buffers help.

See `results/benchmark_report.md` for current data.

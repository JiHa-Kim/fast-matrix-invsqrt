# Chebyshev Direct Apply (`Z = A^{-1/p} B`)

`apply_inverse_proot_chebyshev` approximates `f(A)B` with `f(x)=x^{-1/p}` without explicitly building dense `A^{-1/p}`.

## Why It Exists

For `B in R^{n x k}` with `k << n`:

- Explicit inverse-root route is dominated by `O(n^3)`.
- Direct apply route is dominated by `O(n^2 k)`.

## Approximation Model

- Approximate `x^{-1/p}` on `[l_min, l_max]` with Chebyshev polynomial coefficients.
- Coefficients are computed with discrete Chebyshev orthogonality and cached (`lru_cache`).
- Runtime evaluation uses Clenshaw recurrence on matrices.

## Clenshaw Form Used

With affine map `t(A)` from `[l_min, l_max]` to `[-1,1]`, recurrence is evaluated backward:

$$
y_k = c_k B + 2 t(A)y_{k+1} - y_{k+2}.
$$

Final output:

$$
Z = c_0 B + t(A)y_1 - y_2.
$$

## Implementation Notes

- Workspace: 5 tensors shaped like `B` (`T_curr`, `T_prev`, `T_next`, `Z`, `tmp`).
- Strict validation for `l_min > 0`, shape compatibility, and dtype/device match.
- No dense `n x n` inverse-root buffer.

## Practical Behavior (Latest Solve Benchmarks)

- At `n=2048`, Chebyshev is clearly fastest in the current `p=2` benchmarks.
- At `n=1024`, performance is mixed and can be case-dependent.
- Chebyshev consistently uses less memory than PE-Quad apply variants.

See `reports/chebyshev_solve_benchmark.md` and raw logs in `artifacts/benchmarks/`.

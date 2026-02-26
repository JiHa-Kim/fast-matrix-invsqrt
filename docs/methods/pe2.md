# PE-Quad (Quadratic Polynomial-Express)

Primary inverse-root family used in this repo.

## Form

At step `t`:

$$
B_t = a_t I + b_t Y_t + c_t Y_t^2.
$$

Then:

- Uncoupled: `Y_t = X_t^p A`, `X_{t+1} = X_t B_t`.
- Coupled: `X_{t+1} = X_t B_t`, `Y_{t+1} = B_t^p Y_t` (commuting-model update).

## Implementations

- `inverse_proot_pe_quadratic_uncoupled`
- `inverse_proot_pe_quadratic_coupled`
- `inverse_solve_pe_quadratic_coupled` (apply variant for `Z = A^{-1/p}B`)

## Current Optimized Paths

- `p=2` coupled specialization (`inverse_sqrt_pe_quadratic`).
- `p=3` coupled odd-`p` specialization avoiding extra copy overhead.
- terminal last-step skip in all coupled variants.
- configurable `symmetrize_every` cadence.

## Practical Guidance (from latest benchmarks)

- `p=1`: `Inverse-Newton` often dominates both speed and residual.
- `p=2`: performance split is workload-dependent; coupled often fastest, uncoupled often best residual.
- `p=3`: coupled commonly fastest, uncoupled commonly best residual/relerr.
- `p>=4`: coupled usually fastest; uncoupled can still win accuracy for harder exponents (notably `p=8`).

See `benchmark_results/` and `reports/latest/` for current solver benchmark outputs.

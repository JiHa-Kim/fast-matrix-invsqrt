# PE-Quad Method (Quadratic Polynomial Express)

The primary iteration method for computing `A^{-1/p}`.

## Variants

- **Coupled** (`inverse_proot_pe_quadratic_coupled`): Tracks both `X` and `Y`, skips Y-update on terminal step
- **Uncoupled** (`inverse_proot_pe_quadratic_uncoupled`): Tracks only `X`, recomputes `Y = X^p · A` each step

## Multiplier

Uses tuned quadratic coefficients per step:

$$
B_t = a_t I + b_t Y + c_t Y^2
$$

## Coupled Update Equations

Per full step:

$$
Y_2 \leftarrow Y Y, \quad
B_t = a_t I + b_t Y + c_t Y_2
$$

$$
X \leftarrow X B_t, \quad
Y \leftarrow B_t^p Y  \quad (\text{via binary exponentiation for } p \geq 3)
$$

Optional symmetrization: $Y \leftarrow \frac{1}{2}(Y + Y^\top)$

Terminal step (default): compute only $X \leftarrow X B_t$, skip Y-update.

## Uncoupled Update Equations

Per step:

$$
Y = X^p A, \quad B_t = a_t I + b_t Y + c_t Y^2, \quad X \leftarrow X B_t
$$

## Performance Optimizations

- Fused `addmm`/`baddbmm` for polynomial evaluations in both coupled and uncoupled paths (saves multiple kernel launches/iter)
- Binary exponentiation for coupled Y-update (`_bpow_times_y`): O(log p) matmuls
- `matmul(out=...)` throughout for zero-allocation iterations
- Pre-extracted CPU coefficient triples
- Terminal last-step skip (coupled only)

## When to Use Which Variant

| Scenario | Recommended |
|----------|-------------|
| p ≥ 2, n ≥ 512 | **PE-Quad-Coupled** (10-14% faster via terminal savings) |
| p = 1, n ≤ 512 | **PE-Quad** (lower workspace overhead) |
| Memory constrained | **PE-Quad** (4 vs 6 workspace tensors) |

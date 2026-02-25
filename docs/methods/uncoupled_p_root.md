# Uncoupled p-th Root Iterations

## Motivation

The coupled iteration tracks both `X ≈ A^{-1/p}` and `Y ≈ A·X^p`, requiring 8 workspace tensors.
The uncoupled formulation drops `Y` persistence, recomputing it each step from `X`:

$$ Y_k = X_k^p A $$

Then:

$$ X_{k+1} = X_k \cdot P_k(Y_k) $$

where `P_k` is a quadratic polynomial with tuned coefficients.

## Advantages Over Coupled

1. **Lower memory**: 4 workspace tensors (`X, Xbuf, T1, T2`) vs 6
2. **Simpler logic**: No Y-tracking or buffer management
3. **Natural p-generalization**: `X^p` computed via specialized fast paths (p=1,2,3,4) or generic loop

## Performance Optimizations

- **Fused `addmm`**: The final `X_new = a·X + b·(X·Y)` uses a single `_addmm_into` BLAS call
  instead of separate matmul + copy + mul + add (saves 2 kernel launches per iteration)
- **p-specific matmul chains**: p=4 uses `X^2 → (X^2)^2` (2 matmuls) instead of generic loop (3 matmuls)
- **Binary exponentiation**: General $p \geq 5$ delegates to `_bpow_times_y` for $O(\log p)$ matmuls instead of an $O(p)$ naive loop.

## API

```python
from fast_iroot import inverse_proot_pe_quadratic_uncoupled

X, ws = inverse_proot_pe_quadratic_uncoupled(
    A_norm,
    abc_t=quad_coeffs,
    p_val=4,
    symmetrize_X=True,
)
```

## When to Use

- Memory-constrained settings (large n, limited GPU memory)
- p=1 at small sizes (coupled overhead not worth it)
- When workspace reuse across calls is not needed

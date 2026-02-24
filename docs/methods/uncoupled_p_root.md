# Uncoupled $p$-th Root Iterations

## Motivation
Standard Newton-Schulz and standard Polar-Express formulations (like PE-NS3, PE2) operate as **Coupled Iterations**: they retain dense states $X_k$ and a residual approximation matrix $Y_k$. While iteratively efficient, carrying internal caches (like `Y, Ybuf`) requires holding nearly $\approx 3 \rightarrow 4$ copies of $n \times n$ matrices sequentially inflating peak operations artificially.

Generalizing across $p$-th roots intensifies temporal logic complexities scaling exponentially worse memory thresholds if dense dependencies aren't avoided.

## Uncoupled Mathematics
The `uncoupled` formulation completely drops $Y_k$ persistence across steps. Instead of mathematically iterating two formulas, we reconstruct $Y$:

$$ Y_k = X_k^p A $$

Then update:

$$ X_{k+1} = X_k P_k(Y_k) $$

### Advantages
1. **Memory Ceiling Drop**: Shrinks required variables to exclusively $X, Xbuf, T1, T2$.
2. **Simplified $p$-roots**: Decoupled algorithms inherently generalize to arbitrary $p \in \mathbb{N}$ without structural refactoring.
3. **Optimized Kernel Scaling**: $X_k^2 \rightarrow X_k^4 \rightarrow Y_k$ bounds are natively executed eliminating generic intermediate looping behaviors.

## API Usage
Coupled counterparts strictly evaluate `p=2`. 
Uncoupled functions within `isqrt_core.py` gracefully handle flexible $p$.

```python
from isqrt_core import inverse_proot_pe_affine_uncoupled, inverse_proot_pe_quadratic_uncoupled

# For arbitrary roots, e.g., p=4
X, ws_uncoupled = inverse_proot_pe_quadratic_uncoupled(
    A,
    abc_t=pe_quad_coeffs,
    p_val=4,
    ws=None,
    symmetrize_X=True
)
```

## When to use
Refer to evaluation metrics determining thresholding behavior in benchmarking:
1. $N \le 1024$: Retain the coupled versions natively.
2. $N \ge 2048$ or heavily constrained memory states: The memory reductions explicitly overpower slight throughput variations prioritizing Uncoupled execution cleanly.

# Chebyshev Direct Apply Method

The `apply_inverse_proot_chebyshev` method is designed for directly evaluating the application of an inverse p-th root to a block of right-hand sides, i.e., $Z \approx A^{-1/p} B$, without ever explicitly forming the dense inverse root matrix $A^{-1/p}$.

## Multiplier

This method relies on finding the minimax polynomial approximation of the function $f(x) = x^{-1/p}$ over the preconditioned eigenvalue interval $x \in [\ell_{min}, 1.0]$. The Chebyshev polynomial expansion minimizes the maximum error bound over this domain.

We use discrete orthogonality (via `scipy.integrate`-style calculation) mapped to the $[ -1, 1 ]$ interval to compute the highly precise coefficients $c_k$.

## Clenshaw Recurrence

Instead of naively evaluating $c_0 + c_1 T_1(A) B + c_2 T_2(A) B + \dots$, we use the Clenshaw recurrence. This avoids accumulating floating-point sum errors and minimizes tensor storage allocations.

Given $t(A)$ as the linearly mapped operator $A$ scaled to $[-1, 1]$:
$$ t(A) = \frac{2}{1 - \ell_{min}} A - \frac{1 + \ell_{min}}{1 - \ell_{min}} I $$

The recurrence steps backward from the polynomial degree $d$:
$$ y_{d+1} = 0 $$
$$ y_d = c_d B $$
$$ y_k = c_k B + 2 t(A) y_{k+1} - y_{k+2} $$
$$ \text{Result} = c_0 B + t(A) y_1 - y_2 $$

## Performance Optimizations

- **$O(N^2 K)$ Complexity:** Because the recurrence natively groups multiplication by $B$ (which is $N \times K$), all operations are bounded by block-matrix multiplication sizes shape $N \times K$. For $K \ll N$, this is fundamentally faster than iterating intermediate $N \times N$ matrices ($O(N^3)$).
- **Static Coefficient Caching:** Calculates Scipy matrices exactly once via `functools.lru_cache` and seamlessly passes `Tuple[float]` pointers into PyTorch execution memory. 
- **Workspace Reuse:** Pre-allocates precisely 5 working buffers of size $N \times K$ (`T_curr`, `T_prev`, `T_next`, `Z`, `tmp`). Uses mapped, fused PyTorch BLAS steps (`_matmul_into`, `mul_`, `add_`) natively avoiding allocation overhead entirely. 

## When to Use Which Variant

| Scenario | Recommended |
|----------|-------------|
| $A^{-1/p} B$ where $B$ is sparse feature vectors ($K \le 128$) | **Chebyshev-Apply** ( > 10x Speedup ) |
| Needing to output explicitly dense $A^{-1/p}$ ($N \times N$) | **PE-Quad-Coupled** or **PE-Quad (Uncoupled)** |

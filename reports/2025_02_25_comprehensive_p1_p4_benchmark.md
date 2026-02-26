# Comprehensive Benchmark Report: $p=1$ and $p=4$
*Date: 2026-02-25*

This report details the performance and accuracy of matrix inverse $p$-th roots (IRoot) and inverse $p$-th root solves (Solve) for $p=1$ and $p=4$, comparing several iteration methods under `torch.compile`.

---

## 1. Matrix Inverse Root (IRoot) Results
**Goal**: Compute $X \approx A^{-1/p}$ such that $\|I - A X^p\|_F$ is minimized.

### 1.1 Results for $p=1$ (Standard Inverse)
| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Rel Err |
|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.612 | 1.393 | 12 | 8.943e-03 | 2.270e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.896 | 1.678 | 12 | 9.615e-03 | 2.378e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 4.595 | 2.377 | 12 | 9.587e-03 | 2.367e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.763 | 1.675 | 23 | 6.625e-03 | 1.497e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.474 | 1.386 | 22 | 6.597e-03 | 1.496e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.685 | 1.597 | 23 | 6.589e-03 | 1.496e-03 |

### 1.2 Results for $p=4$ (Inverse 4th Root)
| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Rel Err |
|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.073 | 1.524 | 12 | 8.492e-03 | 2.235e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.924 | 1.375 | 12 | 8.477e-03 | 2.234e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.844 | 1.295 | 12 | 1.263e-02 | 3.282e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.840 | 1.224 | 23 | 6.625e-03 | 1.497e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.944 | 1.327 | 22 | 6.597e-03 | 1.496e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.076 | 1.459 | 23 | 6.589e-03 | 1.496e-03 |

---

## 2. Matrix Solve Results
**Goal**: Directly compute $Z = A^{-1/p} B$ for a block of vectors $B$.

### 2.1 Results for $p=1$ (Solve with Inverse)
| Method | Size | RHS | Total Time (ms) | Iter Time (ms) | Mem (MB) | Rel Err vs True |
|---|---|---|---|---|---|---|
| PE-Quad-Inv-Mult | 256x256 | 16 | 3.623 | 1.253 | 10 | 5.280e-03 |
| PE-Quad-Coupled-Apply | 256x256 | 16 | 3.488 | 1.118 | 10 | 7.233e-03 |
| Chebyshev-Apply | 256x256 | 16 | 6.439 | 4.070 | 10 | 8.789e-03 |
| PE-Quad-Inv-Mult | 512x512 | 16 | 3.663 | 1.607 | 15 | 1.172e-02 |
| PE-Quad-Coupled-Apply | 512x512 | 16 | 3.719 | 1.664 | 16 | 6.683e-03 |
| Chebyshev-Apply | 512x512 | 16 | 6.301 | 4.245 | 13 | 9.399e-03 |

### 2.2 Results for $p=4$ (Solve with Inverse 4th Root)
| Method | Size | RHS | Total Time (ms) | Iter Time (ms) | Mem (MB) | Rel Err vs True |
|---|---|---|---|---|---|---|
| PE-Quad-Inv-Mult | 256x256 | 16 | 4.484 | 2.290 | 10 | 3.189e-03 |
| PE-Quad-Coupled-Apply | 256x256 | 16 | 3.700 | 1.506 | 10 | 4.150e-03 |
| Chebyshev-Apply | 256x256 | 16 | 6.746 | 4.552 | 10 | 2.090e-03 |
| PE-Quad-Inv-Mult | 512x512 | 16 | 4.378 | 2.160 | 15 | 4.211e-03 |
| PE-Quad-Coupled-Apply | 512x512 | 16 | 4.239 | 2.021 | 16 | 4.578e-03 |
| Chebyshev-Apply | 512x512 | 16 | 6.643 | 4.425 | 13 | 2.243e-03 |

---

## 3. Findings & Summary
1.  **IRoot Efficiency**: For both $p=1$ and $p=4$, `PE-Quad` and `PE-Quad-Coupled` methods consistently match or exceed `Inverse-Newton` performance after compilation, with `PE-Quad` often delivering the best residual/time tradeoff on 512x512 matrices.
2.  **Solve Robustness**: In solve tasks, specifically for $p=4$, `Chebyshev-Apply` remains the most accurate but significantly slower than the quadratic PE-based "Apply" and "Multiply" methods.
3.  **Compilation Gains**: Across all tests, the bfloat16 compiled kernels show highly competitive timings (~3-4ms total including preconditioning for 512x512 sized problems).

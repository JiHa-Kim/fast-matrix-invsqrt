# Solver Benchmark Report (Production Ready)

Generated: 2026-02-27T20:11:05

This report compares the production-ready solvers in `fast_iroot` against Vanilla Newton-Schulz and PyTorch baselines.
 
## 1. SPD Linear Solves ($p=1$)
*Target: $Z \approx A^{-1} B$ for Symmetric Positive Definite $A$.*

| Case | Size ($n \times k$) | Method | Total Time (ms) | Iter Time (ms) | Relative Error |
|:---|:---|:---|---:|---:|---:|
| **Gaussian** | $1024 \times 1$ | **Chebyshev-Apply** | **1.954** | 0.497 | 8.30e-03 |
| | | PE-Quad-Coupled-Apply | 2.303 | 0.846 | 6.65e-03 |
| | | Torch-Cholesky-Solve | 2.820 | 1.363 | **0.00e+00** |
| | | Inverse-Newton-Coupled | 2.667 | 1.211 | 6.02e-01 |
| **Ill-Cond** | $1024 \times 64$ | **PE-Quad-Coupled-Apply** | **1.835** | 0.698 | 7.11e-03 |
| | | Chebyshev-Apply | 1.681 | 0.544 | 8.55e-03 |
| | | Torch-Cholesky-Solve | 2.786 | 1.649 | **0.00e+00** |
| | | Inverse-Newton-Coupled | 2.258 | 1.121 | 6.02e-01 |

---

## 2. SPD Inverse Square Root ($p=2$)
*Target: $Z \approx A^{-1/2} B$ for Symmetric Positive Definite $A$.*

| Case | Size ($n \times k$) | Method | Total Time (ms) | Iter Time (ms) | Relative Error |
|:---|:---|:---|---:|---:|---:|
| **Gaussian** | $1024 \times 1$ | **Chebyshev-Apply** | **1.770** | 0.501 | 2.93e-03 |
| | | PE-Quad-Coupled-Apply | 2.898 | 1.629 | 3.71e-03 |
| | | Torch-EVD-Solve | 28.729 | 27.460 | **6.93e-07** |
| | | Inverse-Newton-Coupled | 3.729 | 2.460 | 2.62e-01 |
| **Ill-Cond** | $1024 \times 64$ | **Chebyshev-Apply** | **1.741** | 0.554 | 3.08e-03 |
| | | PE-Quad-Coupled-Apply | 2.538 | 1.350 | 4.91e-03 |
| | | Torch-EVD-Solve | 28.455 | 27.268 | **1.14e-03** |
| | | Inverse-Newton-Coupled | 3.372 | 2.185 | 2.62e-01 |

---

## 3. SPD Inverse 4th Root ($p=4$)
*Target: $Z \approx A^{-1/4} B$ for Symmetric Positive Definite $A$.*

| Case | Size ($n \times k$) | Method | Total Time (ms) | Iter Time (ms) | Relative Error |
|:---|:---|:---|---:|---:|---:|
| **Gaussian** | $1024 \times 1$ | **Chebyshev-Apply** | **2.028** | 0.502 | 2.11e-03 |
| | | PE-Quad-Coupled-Apply | 3.528 | 2.002 | 3.65e-03 |
| | | Torch-EVD-Solve | 29.291 | 27.765 | **1.59e-06** |
| | | Inverse-Newton-Coupled | 4.566 | 3.040 | 1.04e-01 |
| **Ill-Cond** | $1024 \times 64$ | **Chebyshev-Apply** | **2.246** | 0.555 | 2.06e-03 |
| | | PE-Quad-Coupled-Apply | 3.334 | 1.644 | 3.46e-03 |
| | | Torch-EVD-Solve | 29.324 | 27.634 | **1.12e-03** |
| | | Inverse-Newton-Coupled | 4.375 | 2.685 | 1.04e-01 |

---

## 4. Non-SPD Linear Solves ($p=1$)
*Target: $Z \approx A^{-1} B$ for general $A$.*

| Case | Size ($n \times k$) | Method | Total Time (ms) | Iter Time (ms) | Relative Error |
|:---|:---|:---|---:|---:|---:|
| **Gaussian** | $1024 \times 1$ | **PE-Quad-Coupled-Apply** | 2.099 | 1.828 | 5.92e-03 |
| | | Inverse-Newton-Coupled | **1.479** | 1.208 | 6.68e-01 |
| | | Torch-Solve | 5.939 | 5.668 | **5.39e-06** |
| **Hard Case** | $1024 \times 64$ | **PE-Quad-Coupled-Apply** | 5.344 | 5.151 | **1.12e-02** |
| (Posspec) | | Inverse-Newton-Coupled | **1.244** | 1.052 | 1.00e+00 |
| | | Torch-Solve | 4.757 | 4.565 | **1.12e-02** |

*Note: In the hard case, PE-Quad matches Torch-Solve's robustness where Newton-Schulz fails.*

---

## 5. Gram-RHS Optimization ($p=2, 4$)
*Target: $Z \approx (G^T G)^{-1/p} G^T B$*

| Target | RHS Cols ($k$) | Primal Gram (ms) | Dual Gram-RHS (ms) | Speedup |
|:---|---:|---:|---:|---:|
| $p=2$ | 1 | 2.537 | 1.073 | **2.37x** |
| $p=2$ | 64 | 2.384 | 1.058 | **2.25x** |
| $p=4$ | 1 | 2.980 | 1.277 | **2.33x** |
| $p=4$ | 64 | 2.964 | 1.250 | **2.37x** |

---

## Summary of Findings

1.  **Chebyshev Efficiency**: The `Chebyshev-Apply` method is consistently the fastest path for small $k$ across all SPD roots ($p=1, 2, 4$), often beating `PE-Quad` by 20-40% in total latency.
2.  **Newton-Schulz Accuracy**: Vanilla Newton-Schulz (`Inverse-Newton-Coupled`) shows very poor accuracy ($relerr > 0.1$) in most production-size cases, making it unsuitable for high-fidelity ML tasks.
3.  **Production Robustness**: `PE-Quad-Coupled-Apply` provides a robust alternative to `Torch-Solve` for non-SPD matrices, matching its stability in "hard" cases while maintaining competitive latency.
4.  **Specialized Dual Path**: The `Dual-Gram-RHS` path provides a consistent **>2x speedup** over forming the primal RHS ($G^T B$) and applying the solver, making it the preferred path for relevant ML workloads.

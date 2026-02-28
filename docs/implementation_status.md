# Implementation Status

This document tracks the current state of implemented features versus experimental ideas.

## Implemented

### Kernels & Apply Paths
- **Coupled PE**: Quadratic PE inverse-root kernels (`coupled.py`).
- **SPD Fast Path ($p=2, 4$)**: Specialized coupled updates for common inverse square root and 4th root cases.
- **Chebyshev Direct-Apply**: Clenshaw recurrence for $A^{-1/p} B$ without dense inversions (`chebyshev.py`).
- **Matrix-Free Gram Chebyshev**: Massively faster path for wide Gram matrices.
- **Dual Gram-RHS**: Specialized dual-space path for $Z = (G^T G)^{-1/p} G^T B$.

### Scheduling & Preconditioning
- **Online Scheduling**: Greedy Newton and Greedy Affine-Optimal coefficient selection.
- **SPD Preconditioning**: Jacobi (diagonal), Ruiz balancing, AOL, and Frobenius scaling.
- **Non-SPD Safeguards**: Adaptive inverse-Newton fallback and growth-tolerant damping for $p=1$.
- **Workspace Management**: Reusable workspaces to minimize allocation overhead.

---

## Partially Implemented

- **Minimax Adaptation**: Currently limited to local quadratic/affine slope adaptation; full degree-$d$ interval minimax contraction tables are planned.
- **Turbo-like Normalization**: Scaling exists, but robust $\lambda_{min}$-aware optimal scalar initialization is missing.
- **Mixed-Precision Solves**: Basic support via PyTorch dtypes, but no specialized iterative refinement for mixed-precision factorization.

---

## Archived / Legacy

- **Uncoupled PE**: Legacy iterative kernels that track only $X$ (`archive/uncoupled.py`).
- **NSRC (Neumann-Series Residual Correction)**: An additive refinement method for $p=1$. Underperformed compared to PE-Quad in benchmarks and has been moved to `archive/nsrc.py`.
- **Quality Metrics**: Historical validation metrics (`archive/metrics.py`). Production validation now uses direct error checks.

---

## Roadmap (Planned / High Value)

- **Staged Policies**: Interval-focused minimax early, residual-binomial late for $p=2, 4$.
- **Runtime Coefficient Lookup**: Lookup tables by $(p, 	ext{degree}, \kappa)$ for optimal contraction.
- **$p=4$ Two-Stage Reduction**: Tuned $p=2$ kernels used as a base for $p=4$.
- **Block-Diagonal Preconditioning**: More robust spectral clustering for matrices with block structure.
- **Spectral-Clustering Probes**: New quality metrics beyond simple solve relative error.

# Mathematical Methods & Architecture

This section documents the core mathematical foundations and architectural choices of the `fast-matrix-inverse-roots` project.

## Core Concepts

- **[Shared Implementation Notes](shared_tricks.md)**: Details on the preconditioning pipeline, symmetry controls, and low-level performance optimizations used across all methods.
- **[PE-Quad (Quadratic PE)](pe2.md)**: The primary method for SPD matrices, using quadratic polynomial expansions for fast convergence.
- **[Chebyshev Clenshaw Evaluation](chebyshev.md)**: Direct evaluation of $A^{-1/p} B$ using minimax polynomials, ideal for wide Gram matrices or large-scale applies.

## Architecture Overview

The library is designed for high-performance ML workloads, focusing on:
1. **GEMM Efficiency**: Minimizing kernel launches and maximizing hardware utilization.
2. **Numerical Stability**: Using preconditioning and symmetry guards to ensure robust results in finite precision.
3. **API Ergonomics**: Providing high-level solvers that encapsulate complex scheduling and preconditioning decisions.

## Archived Methods

Historical or underperforming methods are maintained in the `archive/` directory for reference:
- **[Uncoupled p-Root](../../archive/uncoupled_p_root.md)**: Legacy iterative kernels that track only $X$.
- **NSRC**: Neumann-Series Residual Correction (additive refinement).
- `archive/ns.md`: Newton-Schulz iterations.
- `archive/pe_ns3.md`: Affine PE scheduling.
- `archive/auto.md`: Early automated selection policies.

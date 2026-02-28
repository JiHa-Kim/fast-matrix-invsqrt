# Glossary of Terms & Abbreviations

This page defines common terms, abbreviations, and mathematical concepts used throughout the `fast-matrix-inverse-roots` project.

## Core Project Terms

- **`fast_iroot`**: The name of the Python package. "iroot" is short for **Inverse Root** (e.g., $A^{-1/p}$).
- **PE**: **Polynomial-Express**. The core iterative family used to compute inverse roots.
- **PE-Quad**: **Quadratic Polynomial-Express**. A specific PE variant that uses quadratic polynomial updates ($B = aI + bY + cY^2$) for rapid convergence.
- **Clenshaw**: Refers to the **Clenshaw Recurrence**, a numerically stable algorithm for evaluating a linear combination of Chebyshev polynomials.

## Archived Terms

- **NSRC**: **Neumann-Series Residual Correction**. An additive refinement method used primarily for linear solves ($p=1$). It refines a solution by iteratively adding terms of the Neumann series. Underperformed compared to PE-Quad and has been moved to the archive.

## Mathematical Abbreviations

- **SPD**: **Symmetric Positive Definite**. A property of a matrix $A$ where $A = A^T$ and $x^T A x > 0$ for all non-zero $x$. Most fast paths in this library are optimized for SPD matrices.
- **RHS**: **Right-Hand Side**. In the context of the problem $Z \approx A^{-1/p} B$, $B$ is the "Right-Hand Side" matrix.
- **EVD**: **Eigenvalue Decomposition**. Factoring a matrix into its eigenvalues and eigenvectors ($A = V \Lambda V^T$).
- **GEMM**: **General Matrix Multiply**. The fundamental operation for most kernels in this project.
- **Gram Matrix**: A matrix formed by the product of a matrix and its transpose (e.g., $A = G^T G$). These arise frequently in ML (e.g., covariance matrices).
- **Condition Number ($\kappa$)**: A measure of how sensitive a function is to changes or errors in the input. High condition numbers ("ill-conditioned") make inverse roots harder to compute accurately.
- **$\rho(I - Y)$ (Spectral Residual)**: The maximum absolute deviation of eigenvalues of the iteration matrix $Y$ from 1.0. This measures how close the iterative state is to the identity.

## Implementation Concepts

- **Coupled Iteration**: An iterative method that tracks both the solution $X$ and the current residual state $Y \approx X^p A$. This is generally more stable and faster for production.
- **Uncoupled Iteration**: An iterative method that tracks only the solution $X$, recalculating the state as needed.
- **Workspace (ws)**: A pre-allocated buffer of memory used to avoid expensive repeated allocations in high-frequency ML loops.
- **K < N**: A regime where the number of columns in the RHS ($k$) is significantly smaller than the dimension of the matrix ($n$). This often enables "matrix-free" or "polynomial-apply" optimizations.
- **BF16 / FP16 / FP32**: Floating-point precisions supported by the library (**Bfloat16**, **Half Precision**, and **Single Precision**).
- **CUDA Graph**: A PyTorch feature that allows "recording" a sequence of GPU operations to reduce CPU overhead during replay.

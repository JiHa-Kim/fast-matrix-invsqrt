# Fast Matrix Inverse Roots: Progress & Strategy Report

## 1. Core Objectives and Evaluation Methodology

Our primary goal is the efficient whitening of a symmetric positive definite (SPD) matrix $B$. We focus on the applied effect rather than the pure approximation of the inverse square root.

*   **Objective Function**: We maximize the whitening certificate:
    $$S = Z^T B Z \approx I$$
    where $Z$ is the preconditioning matrix.
*   **Primary Metric**: The Relative Frobenius Error:
    $$\delta_F = \|S - I\|_F$$
    measured in `fp32` to provide a high-fidelity assessment of stability and convergence.
*   **Performance Target**: We optimize for **Progress-per-Second** (wall-clock time) rather than raw operation counts, ensuring real-world performance on modern GPU hardware.

## 2. Implementation & Recent Achievements

### 2.1. Stable Chebyshev Evaluation (Clenshaw Recurrence)
We have successfully transitioned from unstable forward recurrences to the **Backward Clenshaw Algorithm**. This implementation provides significantly superior numerical stability for high-degree polynomials.

*   **Mechanism**: $B_k = c_k Z + 2 B_{k+1} t - B_{k+2}$
*   **Stability**: Prevents the exponential blow-up of recurrence matrices when eigenvalues stray outside the theoretical $[ \ell, 1 ]$ domain.
*   **Optimization**: Eliminated redundant $O(n^3)$ operations by refining the zero-matrix initialization start-point.

### 2.2. Optimized Monomial Design (Vandermonde Bypass)
To ensure fair comparisons, we fixed the ill-conditioning issues in Monomial polynomial design.
*   **Problem**: Standard Vandermonde bases for polynomials of degree $d \ge 4$ caused LP solver collapse due to poor conditioning.
*   **Solution**: We now solve the Phase-1 optimization in the **Chebyshev Basis** and implicitly project back to the Monomial basis. This enables finding tighter theoretical bounds (smaller $\mu^*$) for use with Horner evaluation.

### 2.3. Benchmarking Framework
*   **Hardware Acceleration**: Integrated `torch.compile(mode="max-autotune")` for all core evaluation kernels.
*   **Rigor**: Implemented exhaustive warmups and `inference_mode` wrappers to isolate performance metrics from JIT overhead and autograd tracking.

## 3. Stability & Precision Findings ($bf16$)

Our research has clarified several critical behaviors in $bf16$ arithmetic:
*   **Horner Collapse**: High-degree Monomials ($d \ge 5$) exhibit "rounding-out-of-bounds" where sequential additions cause the result to overshoot $1.0$ even when the theoretical polynomial is safe.
*   **Clenshaw Robustness**: The Clenshaw algorithm's backward structure acts as a natural stabilizer for the iterative accumulation of precision.

---

## 4. Roadmap & Strategic TODOs

### Phase 1: Robustness and Refinement
*   [ ] **Minimal GEMM-Robustness**:
    *   Implement $\epsilon_\beta$ padding: $\beta \leftarrow (1 + \epsilon_\beta)\beta$
    *   Fold $\alpha$ shrink factor into scale: $Z \leftarrow \alpha Z$
    *   Mandatory `fp32` symmetrization of $\widehat{S}$ before polynomial application.
*   [ ] **Unified Scaling Policy**: Enforce parity across `fro`, `trace`, and `maxdiag` estimators for $\beta$.

### Phase 2: Strategic Integration
*   [ ] **Ridge & Jacobi Analysis**: Conduct a localized sweep over $\delta$ (ridge) and $\epsilon$ (Jacobi) to optimize residual floors.
*   [ ] **Phase-2 Local Step**: Integrate the local minimax optimization step and policy switching logic.
*   [ ] **Full Policy Benchmarking**: Compare end-to-end policies ($P_1$-$P_4$) instead of isolated loops.

---
> *Status: Active research and development. All core evaluation kernels are modularized in `fast_iroot/eval.py`.*

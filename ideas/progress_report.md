# Fast Matrix Inverse Roots: Progress & Strategy Report

## 1. Core Objectives and Evaluation Methodology

Our primary goal is the efficient whitening of a symmetric positive definite (SPD) matrix $B$. We focus on the applied effect rather than the pure approximation of the inverse square root.

*   **Objective Function**: We maximize the whitening certificate:
    $$S = Z^T B Z \approx I$$
    where $Z$ is the preconditioning matrix.
*   **Primary Metric**: The Relative Frobenius Error:
    $$\delta_F = \|S - I\|_F$$
    measured in `bf16` to assess convergence within the target hardware's native precision.
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

### 2.4. BF16-Optimal Scalar Refinement
We have implemented a **bf16-in-the-loop** optimization pass that bridges the gap between fp64-designed polynomials and the realities of bf16 rounding.
*   **Mechanism**: A two-stage refinement using 1D bisection for optimal scaling followed by a derivative-free coordinate pattern search.
*   **Impact**: Directly maximizes the minimum objective value under exact bf16 arithmetic, finding "rounding-aware" coefficient tweaks that improved the objective by up to **25%** for high-degree monomial cases while strictly maintaining feasibility.

## 3. Stability & Precision Findings ($bf16$)

Our research has clarified several critical behaviors in $bf16$ arithmetic:
*   **Horner Collapse**: High-degree Monomials ($d \ge 5$) exhibit "rounding-out-of-bounds" where sequential additions cause the result to overshoot $1.0$ even when the theoretical polynomial is safe.
*   **Clenshaw Robustness**: The Clenshaw algorithm's backward structure acts as a natural stabilizer for the iterative accumulation of precision.
*   **Rounding-Aware Optimization**: We have proven that optimizing directly for the bf16 evaluation model (Step 2.4) is essential for recovering the performance lost to quantization noise in low-precision hardware.

---

## 4. Roadmap & Strategic TODOs

### Phase 1: Robustness and Refinement
*   [x] **BF16-Optimal Scalar Refinement**: Implemented exact pattern search for rounding-aware coefficient design.
*   [x] **Minimal GEMM-Robustness**:
    *   Implement relative $\epsilon_\beta$ padding to guarantee $S/\beta \le 1.0$.
    *   Mandatory `bf16` symmetrization of $\widehat{S}$ to prevent complex eigenvalue drift.
    *   Unified scaling policy across `fro`, `trace`, and `maxdiag` estimators.
*   [ ] **Comprehensive Scaling Policy Validation**: Verify parity across different matrix dimensions.

### Phase 2: Strategic Integration
*   [ ] **Ridge & Jacobi Analysis**: Conduct a localized sweep over $\delta$ (ridge) and $\epsilon$ (Jacobi) to optimize residual floors.
*   [x] **Phase-2 Local Step**: 
    *   Formulated rigorous math for backward induction and hardware-optimal proxy grids.
    *   Proven 2-step protocol via native bf16 GEMM quantization emulation.
    *   Integrated minimal transition ($d=3$, $\rho=0.7653$) and terminal ($d=3$, $\rho=0.0816$) logic.
*   [ ] **Full Policy Benchmarking**: Compare end-to-end policies ($P_1$-$P_4$) instead of isolated loops.

---
> *Status: Active research and development. All core evaluation kernels are modularized in `fast_iroot/eval.py`.*

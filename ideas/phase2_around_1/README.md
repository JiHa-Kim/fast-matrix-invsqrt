# Phase 2: Local Minimax Refinement (Work-in-Progress)

This directory contains the mathematical foundations, strategy, and verification scripts for the Phase 2 Local Refinement stage of the matrix whitening algorithm.

## Status: Active Research & Strategy Formulation
We have established a rigorous 2-step preconditioning protocol to bridge the gap between Phase 1 (Global Spectrum Compression) and the absolute hardware noise floor of `bf16` arithmetic.

### Key Mathematical Findings
1.  **Terminal Noise Floor ($\epsilon_{mach}$)**: Pure `bf16` precision near $1.0$ is limited to $2^{-7} \approx 0.0078$. This is our unbreachable target.
2.  **Hardware GEMM Drift ($\epsilon_{gemm}$)**: Native Tensor Core matrix multiplications introduce an eigenvalue drift of approximately $\pm 0.008$.
3.  **Optimal Terminal Boundary ($\rho_{terminal}$)**: The largest spectral radius that can be squeezed into the noise floor in a single step is **0.0816**.
4.  **Optimal Transition Boundary ($\rho_{in}$)**: A $d=3$ cubic polynomial can safely compress a spectrum from **0.7653** down to the terminal boundary, even accounting for hardware noise.

### Directory Structure
- `ideas/phase2_around_1/phase2_rigorous_math.md`: The definitive mathematical justification for the 2-step protocol and noise margins.
- `scripts/verify_phase2_policy.py`: A lean, high-signal script to reproduce the convergence from Phase 1 output to the hardware floor using native CUDA `bf16` GEMMs.


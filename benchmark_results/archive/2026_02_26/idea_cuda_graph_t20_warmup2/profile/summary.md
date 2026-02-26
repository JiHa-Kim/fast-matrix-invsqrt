# CUDA Graph Profile Summary

- Setup: fixed `n=1024`, `k=16`, `p=2`, `bf16`, `precond=jacobi`, `online-coeff-mode=greedy-affine-opt`.
- Timed kernel path: repeated `inverse_solve_pe_quadratic_coupled` calls (`online_stop_tol=None`).
- Raw profile: `off_vs_on_profile.txt`.

## Key Deltas (`on` vs `off`)

- Median per-call time (CUDA events): `1.973760 ms -> 1.900589 ms` (`-3.71%`).
- Profiler self CUDA total (20 profiled calls): `44.718 ms -> 37.281 ms` (`-16.63%`).
- Launch API shape:
  - Off: `cudaLaunchKernel` `500` calls, `9.070 ms` self CPU.
  - On: `cudaGraphLaunch` `20` calls, `4.061 ms` self CPU.

Interpretation: graph replay collapses host launch overhead and reduces idle gaps between kernels in this fixed-shape steady-state path.

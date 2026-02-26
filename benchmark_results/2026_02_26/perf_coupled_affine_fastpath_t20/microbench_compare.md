# Coupled Fast-Path Microbenchmark (CUDA Event Timing)

- Setup: fixed `A_norm` (`n=1024`), `bf16`, `precond=jacobi` applied once outside timed region.
- Mode: affine-opt planned schedule; timed function is `inverse_solve_pe_quadratic_coupled` only.
- Timing: per cell warmup `20`, then `20` outer x `10` inner calls; median and mean per-call ms from CUDA events.

| p | k | baseline_median_ms | optimized_median_ms | d_median | baseline_mean_ms | optimized_mean_ms | d_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.560 | 1.660 | +6.39% | 1.620 | 1.626 | +0.36% |
| 1 | 16 | 1.540 | 1.497 | -2.81% | 1.550 | 1.501 | -3.19% |
| 1 | 64 | 1.539 | 1.493 | -2.97% | 1.550 | 1.499 | -3.28% |
| 2 | 1 | 2.011 | 1.967 | -2.18% | 2.016 | 2.030 | +0.68% |
| 2 | 16 | 1.993 | 1.955 | -1.91% | 2.020 | 1.956 | -3.14% |
| 2 | 64 | 2.045 | 1.980 | -3.15% | 2.079 | 1.981 | -4.75% |
| 4 | 1 | 2.534 | 2.528 | -0.21% | 2.533 | 2.591 | +2.29% |
| 4 | 16 | 2.525 | 2.522 | -0.10% | 2.544 | 2.547 | +0.09% |
| 4 | 64 | 2.509 | 2.503 | -0.24% | 2.510 | 2.506 | -0.17% |

- Overall avg delta (median): **-0.82%**
- Overall avg delta (mean): **-1.01%**

# Solver Benchmark Report

Generated: 2026-02-28T01:04:13

Assessment metrics:
- `relerr`: median relative error across trials.
- `relerr_p90`: 90th percentile relative error (tail quality).
- `fail_rate`: fraction of failed/non-finite trials.
- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.
- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.

Run config:
- `dtype`: `bf16`
- `extra_args`: ``
- `only`: ``
- `timing_reps`: `10`
- `timing_warmup_reps`: `2`
- `trials`: `10`

## Non-Normal (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.54ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.70ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec` | T-Solve<br>(1.76ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | T-Solve<br>(1.31ms) | PE-Quad-Coupled<br>(1.7e-03) | **T-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.69ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(0.96ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.19ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.25ms) | PE-Quad-Coupled<br>(4.7e-03) | **T-Solve** |
| **1024** / **1**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.19ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(2.25ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(2.52ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **1024** / **1**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.23ms) | PE-Quad-Coupled<br>(1.9e-03) | **T-Solve** |
| **1024** / **16**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.20ms) | T-Solve<br>(1.6e-03) | **PE-Quad-Coupled** |
| **1024** / **16**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.17ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **16**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.18ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **16**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.35ms) | PE-Quad-Coupled<br>(7.7e-03) | **T-Solve** |
| **1024** / **64**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.21ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **64**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.24ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **64**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.27ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **64**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.27ms) | PE-Quad-Coupled<br>(1.2e-02) | **T-Solve** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.90ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(2.12ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.97ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.98ms) | PE-Quad-Coupled<br>(1.2e-02) | **T-Solve** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.56ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.87ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.83ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.82ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.35ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.36ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **16**<br>`gaussian_spd` | PE-Quad-Coupled<br>(1.87ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.99ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(3.44ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.86ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(2.97ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1024**<br>`illcond_1e6` | Inverse-Newton-Coupled<br>(3.01ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(1.75ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(1.87ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(1.99ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.24ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(1.88ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.03ms) | PE-Quad-Coupled-Primal-Gram<br>(2.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(1.56ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.62ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.07ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.17ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.75ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.01ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.36ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.47ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(9.51ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.39ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(3.67ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.51ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.76ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.52ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.16ms) | PE-Quad-Coupled-Primal-Gram<br>(1.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(4.68ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.31ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.18ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.75ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.69ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.31ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(1.68ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(5.39ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(5.02ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | relerr_p90 | fail_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Solve | 1.882e+00 | 1.629 | 1.653e-03 | 1.661e-03 | 0.0% | 1.891e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Solve | 1.890e+00 | 1.699 | 1.658e-03 | 1.663e-03 | 0.0% | 1.896e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Solve | 1.737e+00 | 1.757 | 1.661e-03 | 1.668e-03 | 0.0% | 1.744e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Solve | 2.554e+00 | 1.314 | 1.664e-03 | 1.668e-03 | 0.0% | 2.560e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.181e+00 | 2.228 | 4.385e-03 | 4.520e-03 | 0.0% | 1.217e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.415e+00 | 1.479 | 4.788e-03 | 6.180e-03 | 0.0% | 1.827e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 1.701e+00 | 1.722 | 4.899e-03 | 5.057e-03 | 0.0% | 1.756e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | Torch-Solve | 6.643e-01 | 3.460 | 4.714e-03 | 5.204e-03 | 0.0% | 7.334e-01 |
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 7.985e-01 | 2.993 | 5.730e-03 | 5.981e-03 | 0.0% | 8.335e-01 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.004e+00 | 2.288 | 5.012e-03 | 5.491e-03 | 0.0% | 1.100e+00 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | Torch-Solve | 3.191e-01 | 8.726 | 1.673e-03 | 1.704e-03 | 0.0% | 3.250e-01 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | Torch-Solve | 3.035e-01 | 4.433 | 1.857e-03 | 3.921e-03 | 0.0% | 6.409e-01 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.522e+00 | 1.619 | 5.627e-03 | 5.745e-03 | 0.0% | 1.554e+00 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.135e+00 | 1.588 | 4.138e-03 | 6.058e-03 | 0.0% | 1.662e+00 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.492e+00 | 1.541 | 7.364e-03 | 7.575e-03 | 0.0% | 1.535e+00 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | Torch-Solve | 1.084e-01 | 4.868 | 7.698e-03 | 3.300e-02 | 0.0% | 4.649e-01 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.546e+00 | 1.587 | 5.667e-03 | 5.720e-03 | 0.0% | 1.560e+00 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.209e+00 | 1.625 | 4.538e-03 | 6.129e-03 | 0.0% | 1.633e+00 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.401e+00 | 1.638 | 7.504e-03 | 7.661e-03 | 0.0% | 1.430e+00 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | Torch-Solve | 1.309e-01 | 4.800 | 1.157e-02 | 3.692e-02 | 0.0% | 4.178e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 9.033e-01 | 2.651 | 5.497e-03 | 5.540e-03 | 0.0% | 9.104e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 8.013e-01 | 2.784 | 4.938e-03 | 5.958e-03 | 0.0% | 9.668e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Solve | 4.598e-01 | 6.220 | 1.665e-03 | 1.667e-03 | 0.0% | 4.604e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | Torch-Solve | 2.955e-01 | 6.258 | 1.186e-02 | 1.276e-02 | 0.0% | 3.179e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.490e+01 | 1.559 | 1.662e-03 | 1.666e-03 | 0.0% | 1.494e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.758e+01 | 1.870 | 1.660e-03 | 1.664e-03 | 0.0% | 1.762e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.765e+00 | 1.829 | 1.662e-03 | 1.663e-03 | 0.0% | 4.768e+00 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.770e+00 | 1.819 | 1.662e-03 | 1.664e-03 | 0.0% | 4.776e+00 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.426e+01 | 1.347 | 1.654e-03 | 1.701e-03 | 0.0% | 1.467e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.242e+01 | 1.362 | 1.672e-03 | 1.713e-03 | 0.0% | 1.272e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 2.778e+00 | 2.189 | 1.666e-03 | 1.673e-03 | 0.0% | 2.790e+00 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.654e+00 | 1.989 | 1.667e-03 | 1.679e-03 | 0.0% | 3.680e+00 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.835e+00 | 3.440 | 1.658e-03 | 1.664e-03 | 0.0% | 4.852e+00 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.712e+00 | 1.861 | 1.662e-03 | 1.666e-03 | 0.0% | 4.723e+00 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.310e+00 | 3.353 | 1.660e-03 | 1.662e-03 | 0.0% | 1.312e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.309e+00 | 3.309 | 1.660e-03 | 1.661e-03 | 0.0% | 1.310e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.471e+00 | 1.755 | 2.828e-03 | 2.934e-03 | 0.0% | 4.639e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.332e+00 | 1.874 | 2.682e-03 | 2.717e-03 | 0.0% | 6.415e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 3.134e+00 | 1.994 | 2.936e-03 | 2.948e-03 | 0.0% | 3.147e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.657e+00 | 2.244 | 2.878e-03 | 3.158e-03 | 0.0% | 2.915e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.486e+00 | 1.875 | 2.657e-03 | 2.822e-03 | 0.0% | 5.827e+00 |
| spd | 2 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.960 | 2.136e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.525e+00 | 1.561 | 2.712e-03 | 2.858e-03 | 0.0% | 5.822e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.634e+00 | 1.621 | 2.660e-03 | 2.996e-03 | 0.0% | 5.219e+00 |
| spd | 2 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.925 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.148e+00 | 2.174 | 2.674e-03 | 2.705e-03 | 0.0% | 5.208e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.515e+00 | 2.748 | 2.714e-03 | 2.973e-03 | 0.0% | 2.755e+00 |
| spd | 2 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.898 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.551e+00 | 2.357 | 2.654e-03 | 2.949e-03 | 0.0% | 2.835e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.311e-01 | 3.472 | 4.551e-03 | 6.256e-03 | 0.0% | 1.005e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 5.326e-01 | 9.508 | 7.075e-03 | 7.079e-03 | 0.0% | 5.329e-01 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.793e+00 | 2.392 | 1.901e-03 | 1.907e-03 | 0.0% | 4.808e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 3.543e+00 | 3.671 | 1.901e-03 | 1.917e-03 | 0.0% | 3.573e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 1.912e+00 | 2.514 | 2.929e-03 | 3.630e-03 | 0.0% | 2.370e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.378e+00 | 2.764 | 3.610e-03 | 3.657e-03 | 0.0% | 2.409e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 3.224e+00 | 2.524 | 1.856e-03 | 1.949e-03 | 0.0% | 3.386e+00 |
| spd | 4 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.505 | 1.123e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 3.498e+00 | 4.677 | 1.907e-03 | 1.963e-03 | 0.0% | 3.601e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 3.798e+00 | 2.309 | 1.900e-03 | 1.905e-03 | 0.0% | 3.808e+00 |
| spd | 4 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.384 | 1.172e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.441e+00 | 1.753 | 1.908e-03 | 1.916e-03 | 0.0% | 5.464e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.975e+00 | 1.687 | 1.904e-03 | 1.918e-03 | 0.0% | 5.012e+00 |
| spd | 4 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.499 | 1.154e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 5.002e+00 | 1.683 | 1.904e-03 | 1.918e-03 | 0.0% | 5.039e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 5.993e-01 | 5.394 | 3.649e-03 | 3.736e-03 | 0.0% | 6.136e-01 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 7.145e-01 | 5.024 | 3.796e-03 | 3.803e-03 | 0.0% | 7.158e-01 |

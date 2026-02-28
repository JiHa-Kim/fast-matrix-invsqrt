# Solver Benchmark Report

Generated: 2026-02-28T00:47:06

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
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(0.84ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(0.88ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(0.84ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(0.79ms) | PE-Quad-Coupled<br>(1.7e-03) | **T-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(0.93ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.09ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(0.92ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(0.92ms) | PE-Quad-Coupled<br>(4.7e-03) | **T-Solve** |
| **1024** / **1**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.28ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.16ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.19ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **1024** / **1**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.19ms) | PE-Quad-Coupled<br>(1.9e-03) | **T-Solve** |
| **1024** / **16**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.17ms) | T-Solve<br>(1.6e-03) | **PE-Quad-Coupled** |
| **1024** / **16**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.15ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **16**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.17ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **16**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.53ms) | PE-Quad-Coupled<br>(7.7e-03) | **T-Solve** |
| **1024** / **64**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.57ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **64**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.40ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **64**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.43ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **64**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.42ms) | PE-Quad-Coupled<br>(1.2e-02) | **T-Solve** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.00ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.92ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.94ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.93ms) | PE-Quad-Coupled<br>(1.2e-02) | **T-Solve** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.72ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.66ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.89ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.96ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.48ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(1.35ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **16**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.68ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(2.14ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-Reuse<br>(1.81ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-Reuse<br>(2.07ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(2.95ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |
| **1024** / **1024**<br>`illcond_1e6` | Inverse-Newton-Coupled<br>(3.20ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-Reuse** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(1.78ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(1.72ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.09ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.33ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(1.61ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.02ms) | PE-Quad-Coupled-Primal-Gram<br>(2.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(1.67ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.75ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.11ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.84ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.71ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.04ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.33ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.30ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.98ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(1.69ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(1.60ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.20ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | Chebyshev<br>(2.98ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.08ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.10ms) | PE-Quad-Coupled-Primal-Gram<br>(1.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(2.04ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.63ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.23ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.61ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.98ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram-RHS<br>(1.13ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.33ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(4.23ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.68ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | relerr_p90 | fail_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Solve | 2.369e+00 | 1.337 | 1.653e-03 | 1.661e-03 | 0.0% | 2.380e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Solve | 3.348e+00 | 0.993 | 1.658e-03 | 1.663e-03 | 0.0% | 3.358e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Solve | 3.666e+00 | 0.968 | 1.661e-03 | 1.668e-03 | 0.0% | 3.681e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Solve | 3.690e+00 | 0.896 | 1.664e-03 | 1.668e-03 | 0.0% | 3.699e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.466e+00 | 1.752 | 4.385e-03 | 4.520e-03 | 0.0% | 1.511e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.240e+00 | 1.602 | 4.788e-03 | 6.180e-03 | 0.0% | 1.601e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 1.700e+00 | 1.473 | 4.899e-03 | 5.057e-03 | 0.0% | 1.755e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | Torch-Solve | 7.507e-01 | 2.952 | 4.714e-03 | 5.204e-03 | 0.0% | 8.287e-01 |
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.288e+00 | 1.891 | 5.730e-03 | 5.981e-03 | 0.0% | 1.344e+00 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.524e+00 | 1.536 | 5.012e-03 | 5.491e-03 | 0.0% | 1.670e+00 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | Torch-Solve | 6.465e-01 | 4.367 | 1.673e-03 | 1.704e-03 | 0.0% | 6.585e-01 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | Torch-Solve | 3.028e-01 | 4.420 | 1.857e-03 | 3.921e-03 | 0.0% | 6.394e-01 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.504e+00 | 1.614 | 5.627e-03 | 5.745e-03 | 0.0% | 1.536e+00 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.156e+00 | 1.562 | 4.138e-03 | 6.058e-03 | 0.0% | 1.693e+00 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.473e+00 | 1.558 | 7.364e-03 | 7.575e-03 | 0.0% | 1.515e+00 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | Torch-Solve | 1.070e-01 | 4.760 | 7.698e-03 | 3.300e-02 | 0.0% | 4.585e-01 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.381e+00 | 2.076 | 5.667e-03 | 5.720e-03 | 0.0% | 1.394e+00 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.214e+00 | 1.746 | 4.538e-03 | 6.129e-03 | 0.0% | 1.640e+00 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.018e+00 | 2.255 | 7.504e-03 | 7.661e-03 | 0.0% | 1.039e+00 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | Torch-Solve | 1.305e-01 | 4.852 | 1.157e-02 | 3.692e-02 | 0.0% | 4.163e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 8.801e-01 | 2.782 | 5.497e-03 | 5.540e-03 | 0.0% | 8.870e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 8.111e-01 | 2.514 | 4.938e-03 | 5.958e-03 | 0.0% | 9.787e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Solve | 4.567e-01 | 6.258 | 1.665e-03 | 1.667e-03 | 0.0% | 4.572e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | Torch-Solve | 2.955e-01 | 6.225 | 1.186e-02 | 1.276e-02 | 0.0% | 3.179e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.491e+01 | 1.722 | 1.662e-03 | 1.666e-03 | 0.0% | 1.495e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.538e+01 | 1.657 | 1.660e-03 | 1.664e-03 | 0.0% | 1.542e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.700e+00 | 1.894 | 1.662e-03 | 1.663e-03 | 0.0% | 4.703e+00 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.775e+00 | 1.962 | 1.662e-03 | 1.664e-03 | 0.0% | 4.781e+00 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.354e+01 | 1.476 | 1.654e-03 | 1.701e-03 | 0.0% | 1.392e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.424e+01 | 1.347 | 1.672e-03 | 1.713e-03 | 0.0% | 1.459e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 5.108e+00 | 1.678 | 1.666e-03 | 1.673e-03 | 0.0% | 5.129e+00 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 5.126e+00 | 2.137 | 1.667e-03 | 1.679e-03 | 0.0% | 5.163e+00 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.834e+00 | 1.809 | 1.658e-03 | 1.664e-03 | 0.0% | 4.851e+00 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.647e+00 | 2.067 | 1.662e-03 | 1.666e-03 | 0.0% | 4.658e+00 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.327e+00 | 3.348 | 1.660e-03 | 1.662e-03 | 0.0% | 1.329e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.326e+00 | 3.467 | 1.660e-03 | 1.661e-03 | 0.0% | 1.327e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.480e+00 | 1.779 | 2.828e-03 | 2.934e-03 | 0.0% | 4.648e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.337e+00 | 1.724 | 2.682e-03 | 2.717e-03 | 0.0% | 6.420e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 3.067e+00 | 2.086 | 2.936e-03 | 2.948e-03 | 0.0% | 3.080e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.413e+00 | 2.326 | 2.878e-03 | 3.158e-03 | 0.0% | 2.648e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.497e+00 | 1.607 | 2.657e-03 | 2.822e-03 | 0.0% | 5.838e+00 |
| spd | 2 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.032 | 2.136e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.509e+00 | 1.667 | 2.712e-03 | 2.858e-03 | 0.0% | 5.806e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.598e+00 | 1.752 | 2.660e-03 | 2.996e-03 | 0.0% | 5.179e+00 |
| spd | 2 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.918 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 4.876e+00 | 1.840 | 2.674e-03 | 2.705e-03 | 0.0% | 4.933e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.044e+00 | 1.706 | 2.714e-03 | 2.973e-03 | 0.0% | 4.430e+00 |
| spd | 2 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.916 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 3.969e+00 | 2.334 | 2.654e-03 | 2.949e-03 | 0.0% | 4.410e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.980e-01 | 3.296 | 4.551e-03 | 6.256e-03 | 0.0% | 1.097e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.006e+00 | 3.975 | 7.075e-03 | 7.079e-03 | 0.0% | 1.007e+00 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.944e+00 | 1.690 | 1.901e-03 | 1.907e-03 | 0.0% | 4.960e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.731e+00 | 1.598 | 1.901e-03 | 1.917e-03 | 0.0% | 6.788e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.350e+00 | 2.199 | 2.929e-03 | 3.630e-03 | 0.0% | 2.913e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.279e+00 | 2.982 | 1.896e-03 | 1.900e-03 | 0.0% | 2.284e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.880e+00 | 2.075 | 1.856e-03 | 1.949e-03 | 0.0% | 6.175e+00 |
| spd | 4 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.410 | 1.123e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.936e+00 | 2.044 | 1.907e-03 | 1.963e-03 | 0.0% | 6.110e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 5.481e+00 | 1.626 | 1.900e-03 | 1.905e-03 | 0.0% | 5.495e+00 |
| spd | 4 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.389 | 1.172e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.468e+00 | 1.610 | 1.908e-03 | 1.916e-03 | 0.0% | 5.491e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.959e+00 | 1.978 | 1.904e-03 | 1.918e-03 | 0.0% | 4.995e+00 |
| spd | 4 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.397 | 1.154e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.977e+00 | 2.327 | 1.904e-03 | 1.918e-03 | 0.0% | 5.014e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 8.760e-01 | 4.232 | 3.649e-03 | 3.736e-03 | 0.0% | 8.969e-01 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.019e+00 | 3.679 | 3.796e-03 | 3.803e-03 | 0.0% | 1.021e+00 |

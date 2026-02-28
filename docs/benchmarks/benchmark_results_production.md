# Solver Benchmark Report

Generated: 2026-02-28T04:34:51

## Run Configuration

- ab_baseline_rows_in: ``
- ab_extra_args_a: ``
- ab_extra_args_b: ``
- ab_interleave: `True`
- ab_label_a: `A`
- ab_label_b: `B`
- ab_match_on_method: `True`
- ab_out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\benchmark_results\runs\2026_02_28\042732_solver_benchmarks\solver_benchmarks_ab.md`
- baseline_rows_out: ``
- dtype: `bf16`
- extra_args: ``
- integrity_checksums: `True`
- manifest_out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\benchmark_results\runs\2026_02_28\042732_solver_benchmarks\run_manifest.json`
- markdown: `True`
- only: ``
- out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\docs\benchmarks\benchmark_results_production.md`
- prod: `True`
- run_name: `solver_benchmarks`
- timing_reps: `10`
- timing_warmup_reps: `2`
- trials: `10`

Assessment metrics:
- `relerr`: median relative error across trials.
- `relerr_p90`: 90th percentile relative error (tail quality).
- `fail_rate`: fraction of failed/non-finite trials.
- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.
- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.

## Non-Normal (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_shifted` | T-Linalg-Solve<br>(1.19ms) | T-Linalg-Solve<br>(3.4e-07) | **T-Linalg-Solve** |
| **256** / **256**<br>`nonnormal_upper` | T-Linalg-Solve<br>(1.05ms) | T-Linalg-Solve<br>(8.4e-08) | **T-Linalg-Solve** |
| **256** / **256**<br>`similarity_posspec` | T-Linalg-Solve<br>(1.15ms) | T-Linalg-Solve<br>(4.6e-07) | **T-Linalg-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | T-Linalg-Solve<br>(1.23ms) | T-Linalg-Solve<br>(4.4e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.18ms) | T-Linalg-Solve<br>(7.5e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.14ms) | T-Linalg-Solve<br>(6.6e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.12ms) | T-Linalg-Solve<br>(9.1e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.08ms) | T-Linalg-Solve<br>(4.4e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.03ms) | T-Linalg-Solve<br>(9.8e-05) | **T-Linalg-Solve** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(2.10ms) | T-Linalg-Solve<br>(9.1e-05) | **T-Linalg-Solve** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(2.02ms) | T-Linalg-Solve<br>(1.2e-04) | **T-Linalg-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(2.07ms) | T-Linalg-Solve<br>(1.2e-02) | **PE-Quad-Coupled** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.64ms) | T-Cholesky-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.20ms) | T-Cholesky-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.52ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.31ms) | T-Cholesky-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.84ms) | T-Cholesky-Solve<br>(3.5e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.64ms) | T-Cholesky-Solve<br>(3.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.07ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.16ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.67ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.69ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(3.78ms) | T-Cholesky-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(4.07ms) | T-Cholesky-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.39ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.49ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.81ms) | T-Linalg-Solve<br>(3.8e-04) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.07ms) | T-Linalg-Solve<br>(3.9e-04) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.19ms) | T-Linalg-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(2.27ms) | T-Linalg-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.27ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.31ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.38ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.42ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(4.37ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | Inverse-Newton-Coupled<br>(4.52ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(3.00ms) | T-Linalg-Solve<br>(3.7e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.46ms) | T-Linalg-Solve<br>(3.7e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.19ms) | T-Linalg-Solve<br>(4.0e-04) | **Chebyshev** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.82ms) | T-Linalg-Solve<br>(4.2e-04) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.33ms) | T-Linalg-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(2.16ms) | T-Linalg-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.45ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.45ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.67ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.60ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(5.35ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(5.17ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | resid | nf_rate | qf_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Linalg-Solve | 6.481e+00 | 1.188 | 3.407e-07 | 3.426e-07 | 0.0% | 0.0% | 6.822e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Linalg-Solve | 8.577e+00 | 1.049 | 8.427e-08 | 8.606e-08 | 0.0% | 0.0% | 8.622e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Linalg-Solve | 6.491e+00 | 1.149 | 4.613e-07 | 4.653e-07 | 0.0% | 0.0% | 6.715e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Linalg-Solve | 2.309e+00 | 1.232 | 4.431e-05 | 3.721e-04 | 0.0% | 10.0% | 4.249e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | Torch-Linalg-Solve | 1.440e+00 | 3.060 | 7.499e-05 | 7.819e-05 | 0.0% | 0.0% | 1.442e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | Torch-Linalg-Solve | 1.445e+00 | 3.066 | 6.601e-05 | 6.773e-05 | 0.0% | 0.0% | 1.456e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | Torch-Linalg-Solve | 1.427e+00 | 2.978 | 9.087e-05 | 1.037e-04 | 0.0% | 0.0% | 1.431e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 0.000e+00 | 3.921 | 4.714e-03 | 4.179e+00 | 0.0% | 100.0% | 6.243e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | Torch-Linalg-Solve | 6.516e-01 | 6.346 | 9.804e-05 | 1.018e-04 | 0.0% | 0.0% | 6.525e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | Torch-Linalg-Solve | 6.460e-01 | 6.358 | 9.095e-05 | 9.266e-05 | 0.0% | 0.0% | 6.576e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Linalg-Solve | 6.334e-01 | 6.353 | 1.212e-04 | 1.376e-04 | 0.0% | 0.0% | 6.339e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 0.000e+00 | 7.357 | 1.186e-02 | 5.537e+00 | 0.0% | 100.0% | 2.707e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.043e+01 | 2.640 | 1.642e-07 | 1.201e-04 | 0.0% | 0.0% | 3.151e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.367e+01 | 2.199 | 1.647e-07 | 1.559e-04 | 0.0% | 0.0% | 3.506e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.080e+01 | 2.523 | 1.663e-07 | 7.764e-05 | 0.0% | 0.0% | 1.124e+01 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.162e+01 | 2.315 | 1.635e-07 | 1.156e-04 | 0.0% | 0.0% | 1.183e+01 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.328e+01 | 1.837 | 3.513e-07 | 4.312e-05 | 0.0% | 0.0% | 3.445e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.357e+01 | 1.639 | 3.573e-07 | 8.167e-05 | 0.0% | 0.0% | 3.439e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.211e+01 | 2.066 | 1.676e-07 | 4.333e-05 | 0.0% | 0.0% | 1.245e+01 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.222e+01 | 2.162 | 1.679e-07 | 8.223e-05 | 0.0% | 0.0% | 1.250e+01 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.143e+01 | 2.667 | 1.677e-07 | 4.339e-05 | 0.0% | 0.0% | 1.166e+01 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.154e+01 | 2.686 | 1.681e-07 | 8.176e-05 | 0.0% | 0.0% | 1.176e+01 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.222e+00 | 3.872 | 1.891e-07 | 4.285e-05 | 0.0% | 0.0% | 3.261e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.200e+00 | 4.401 | 1.870e-07 | 8.165e-05 | 0.0% | 0.0% | 3.248e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.465e+00 | 2.389 | 2.828e-03 | 2.834e-03 | 0.0% | 0.0% | 4.632e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.297e+00 | 2.490 | 2.682e-03 | 2.686e-03 | 0.0% | 0.0% | 6.379e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.716e+00 | 2.810 | 2.936e-03 | 2.942e-03 | 0.0% | 0.0% | 2.727e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.096e+00 | 3.074 | 2.878e-03 | 2.881e-03 | 0.0% | 0.0% | 2.300e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.497e+00 | 2.185 | 2.657e-03 | 2.659e-03 | 0.0% | 0.0% | 5.838e+00 |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.504e+00 | 2.275 | 2.712e-03 | 2.714e-03 | 0.0% | 0.0% | 5.800e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.619e+00 | 2.269 | 2.660e-03 | 2.662e-03 | 0.0% | 0.0% | 5.202e+00 |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.134e+00 | 2.311 | 2.674e-03 | 2.675e-03 | 0.0% | 0.0% | 5.194e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.318e+00 | 2.383 | 2.714e-03 | 2.715e-03 | 0.0% | 0.0% | 4.730e+00 |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.290e+00 | 2.423 | 2.654e-03 | 2.655e-03 | 0.0% | 0.0% | 4.767e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.289e-01 | 4.369 | 4.551e-03 | 4.557e-03 | 0.0% | 0.0% | 1.002e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 6.368e-01 | 5.169 | 7.075e-03 | 7.079e-03 | 0.0% | 0.0% | 6.372e-01 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.928e+00 | 3.001 | 1.901e-03 | 1.902e-03 | 0.0% | 0.0% | 4.944e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.705e+00 | 2.455 | 1.901e-03 | 1.901e-03 | 0.0% | 0.0% | 6.761e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | Chebyshev-Apply | 2.347e+00 | 3.475 | 1.896e-03 | 1.897e-03 | 0.0% | 0.0% | 2.357e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.404e+00 | 2.859 | 1.896e-03 | 1.896e-03 | 0.0% | 0.0% | 2.409e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.889e+00 | 2.335 | 1.856e-03 | 1.856e-03 | 0.0% | 0.0% | 6.184e+00 |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.936e+00 | 2.163 | 1.907e-03 | 1.908e-03 | 0.0% | 0.0% | 6.110e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 5.496e+00 | 2.451 | 1.900e-03 | 1.900e-03 | 0.0% | 0.0% | 5.510e+00 |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.440e+00 | 2.449 | 1.908e-03 | 1.909e-03 | 0.0% | 0.0% | 5.463e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.938e+00 | 2.675 | 1.904e-03 | 1.904e-03 | 0.0% | 0.0% | 4.974e+00 |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.785e+00 | 2.597 | 1.904e-03 | 1.904e-03 | 0.0% | 0.0% | 4.820e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 1.002e+00 | 5.348 | 3.649e-03 | 3.651e-03 | 0.0% | 0.0% | 1.026e+00 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 7.062e-01 | 5.173 | 3.796e-03 | 3.801e-03 | 0.0% | 0.0% | 7.075e-01 |

# Solver Benchmark Report

Generated: 2026-02-28T02:32:27

## Run Configuration

- ab_baseline_rows_in: ``
- ab_extra_args_a: ``
- ab_extra_args_b: ``
- ab_interleave: `True`
- ab_label_a: `A`
- ab_label_b: `B`
- ab_match_on_method: `True`
- ab_out: `<REPO_ROOT>\benchmark_results\runs\2026_02_28\022703_solver_benchmarks\solver_benchmarks_ab.md`
- baseline_rows_out: ``
- dtype: `bf16`
- extra_args: ``
- integrity_checksums: `True`
- manifest_out: `<REPO_ROOT>\benchmark_results\runs\2026_02_28\022703_solver_benchmarks\run_manifest.json`
- markdown: `True`
- only: ``
- out: `docs/benchmarks/benchmark_results_production.md`
- prod: `False`
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
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(0.95ms) | T-Solve<br>(3.4e-07) | **T-Solve** |
| **256** / **256**<br>`nonnormal_upper` | T-Solve<br>(1.00ms) | T-Solve<br>(8.4e-08) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec` | T-Solve<br>(1.00ms) | T-Solve<br>(4.6e-07) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | T-Solve<br>(0.91ms) | T-Solve<br>(4.4e-05) | **T-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(0.94ms) | T-Solve<br>(7.5e-05) | **PE-Quad-Coupled** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.01ms) | T-Solve<br>(6.6e-05) | **T-Solve** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(0.91ms) | T-Solve<br>(9.1e-05) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(0.94ms) | T-Solve<br>(4.4e-03) | **T-Solve** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.02ms) | T-Solve<br>(9.8e-05) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.96ms) | T-Solve<br>(9.1e-05) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(2.06ms) | T-Solve<br>(1.2e-04) | **T-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(2.02ms) | T-Solve<br>(1.2e-02) | **T-Solve** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.00ms) | T-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.71ms) | T-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.38ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.84ms) | T-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.53ms) | T-Solve<br>(3.5e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(3.35ms) | T-Solve<br>(3.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.69ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.69ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.78ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.01ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.34ms) | T-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.27ms) | T-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(1.98ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.12ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.05ms) | T-EVD-Solve<br>(3.8e-04) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.65ms) | T-EVD-Solve<br>(3.9e-04) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(1.65ms) | T-EVD-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.10ms) | PE-Quad-Coupled-Primal-Gram<br>(2.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(1.65ms) | T-EVD-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.81ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.22ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.81ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.77ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.03ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.02ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.56ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.19ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.45ms) | T-EVD-Solve<br>(3.7e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.10ms) | T-EVD-Solve<br>(3.7e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.43ms) | T-EVD-Solve<br>(4.0e-04) | **Chebyshev** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.47ms) | T-EVD-Solve<br>(4.2e-04) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.13ms) | T-EVD-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.21ms) | PE-Quad-Coupled-Primal-Gram<br>(1.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(3.29ms) | T-EVD-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.35ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.27ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(3.41ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.84ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.15ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.16ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.96ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(4.05ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | relerr_p90 | fail_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Solve | 6.493e+00 | 1.105 | 3.407e-07 | 3.586e-07 | 0.0% | 6.834e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Solve | 8.551e+00 | 0.996 | 8.427e-08 | 8.471e-08 | 0.0% | 8.596e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Solve | 7.547e+00 | 1.003 | 4.613e-07 | 4.772e-07 | 0.0% | 7.807e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Solve | 3.647e+00 | 0.910 | 4.431e-05 | 7.338e-05 | 0.0% | 6.039e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.562e+00 | 1.626 | 4.385e-03 | 4.520e-03 | 0.0% | 1.610e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | Torch-Solve | 1.481e+00 | 2.995 | 6.601e-05 | 6.650e-05 | 0.0% | 1.492e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 1.574e+00 | 1.572 | 4.899e-03 | 5.057e-03 | 0.0% | 1.625e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | Torch-Solve | 7.439e-01 | 3.008 | 4.415e-03 | 4.937e-03 | 0.0% | 8.319e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 8.752e-01 | 2.735 | 5.497e-03 | 5.540e-03 | 0.0% | 8.820e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 7.656e-01 | 2.673 | 4.938e-03 | 5.958e-03 | 0.0% | 9.238e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Solve | 6.377e-01 | 6.318 | 1.212e-04 | 1.213e-04 | 0.0% | 6.382e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | Torch-Solve | 2.903e-01 | 6.361 | 1.175e-02 | 1.265e-02 | 0.0% | 3.125e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.625e+01 | 1.997 | 1.642e-07 | 1.700e-07 | 0.0% | 3.753e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.090e+01 | 1.707 | 1.647e-07 | 1.715e-07 | 0.0% | 4.259e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.110e+01 | 2.385 | 1.663e-07 | 1.730e-07 | 0.0% | 1.155e+01 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.149e+01 | 1.840 | 1.635e-07 | 1.665e-07 | 0.0% | 1.170e+01 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.300e+01 | 1.526 | 3.513e-07 | 3.636e-07 | 0.0% | 3.416e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.356e+01 | 3.353 | 3.573e-07 | 3.660e-07 | 0.0% | 3.438e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.232e+01 | 1.691 | 1.676e-07 | 1.723e-07 | 0.0% | 1.267e+01 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.228e+01 | 1.686 | 1.679e-07 | 1.718e-07 | 0.0% | 1.257e+01 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.157e+01 | 1.780 | 1.677e-07 | 1.710e-07 | 0.0% | 1.180e+01 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.164e+01 | 2.014 | 1.681e-07 | 1.713e-07 | 0.0% | 1.186e+01 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.207e+00 | 3.606 | 1.891e-07 | 1.914e-07 | 0.0% | 3.246e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.160e+00 | 3.545 | 1.870e-07 | 1.898e-07 | 0.0% | 3.207e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.479e+00 | 1.977 | 2.828e-03 | 2.934e-03 | 0.0% | 4.647e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.324e+00 | 2.115 | 2.682e-03 | 2.717e-03 | 0.0% | 6.407e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.894e+00 | 2.049 | 2.936e-03 | 2.948e-03 | 0.0% | 2.906e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.350e+00 | 2.645 | 2.878e-03 | 3.158e-03 | 0.0% | 2.579e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.493e+00 | 1.654 | 2.657e-03 | 2.822e-03 | 0.0% | 5.834e+00 |
| spd | 2 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.113 | 2.136e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.503e+00 | 1.650 | 2.712e-03 | 2.858e-03 | 0.0% | 5.799e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.616e+00 | 1.807 | 2.660e-03 | 2.996e-03 | 0.0% | 5.199e+00 |
| spd | 2 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.916 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.114e+00 | 1.812 | 2.674e-03 | 2.705e-03 | 0.0% | 5.173e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.332e+00 | 1.771 | 2.714e-03 | 2.973e-03 | 0.0% | 4.745e+00 |
| spd | 2 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.940 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.290e+00 | 2.024 | 2.654e-03 | 2.949e-03 | 0.0% | 4.767e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.289e-01 | 3.562 | 4.551e-03 | 6.256e-03 | 0.0% | 1.002e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.049e+00 | 3.187 | 7.075e-03 | 7.079e-03 | 0.0% | 1.050e+00 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.943e+00 | 2.446 | 1.901e-03 | 1.907e-03 | 0.0% | 4.959e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.729e+00 | 2.096 | 1.901e-03 | 1.917e-03 | 0.0% | 6.786e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | Chebyshev-Apply | 2.246e+00 | 2.695 | 1.896e-03 | 1.904e-03 | 0.0% | 2.255e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.428e+00 | 2.594 | 1.896e-03 | 1.900e-03 | 0.0% | 2.433e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.865e+00 | 2.131 | 1.856e-03 | 1.949e-03 | 0.0% | 6.159e+00 |
| spd | 4 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.882 | 1.123e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.964e+00 | 3.285 | 1.907e-03 | 1.963e-03 | 0.0% | 6.139e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 5.451e+00 | 2.349 | 1.900e-03 | 1.905e-03 | 0.0% | 5.465e+00 |
| spd | 4 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.403 | 1.172e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.423e+00 | 3.406 | 1.908e-03 | 1.916e-03 | 0.0% | 5.446e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.971e+00 | 1.839 | 1.904e-03 | 1.918e-03 | 0.0% | 5.008e+00 |
| spd | 4 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.394 | 1.154e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.964e+00 | 2.163 | 1.904e-03 | 1.918e-03 | 0.0% | 5.001e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 1.026e+00 | 3.959 | 3.649e-03 | 3.736e-03 | 0.0% | 1.050e+00 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.016e+00 | 4.050 | 3.796e-03 | 3.803e-03 | 0.0% | 1.018e+00 |

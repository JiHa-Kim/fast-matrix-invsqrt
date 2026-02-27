# Solver Benchmark A/B Report

Generated: 2026-02-26T21:24:45

A: gemm
B: shape_aware

| kind | p | n | k | case | method | gemm_total_ms | shape_aware_total_ms | delta_ms(B-A) | delta_pct | gemm_iter_ms | shape_aware_iter_ms | gemm_relerr | shape_aware_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 4.507 | 2.088 | -2.419 | -53.67% | 0.693 | 0.656 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 2 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.270 | 2.080 | -0.190 | -8.37% | 0.847 | 0.745 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.561 | 2.637 | 0.076 | 2.97% | 0.683 | 0.733 | 3.204e-03 | 3.204e-03 | 1.000 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.230 | 2.306 | 0.076 | 3.41% | 0.822 | 0.895 | 3.555e-03 | 3.555e-03 | 1.000 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.107 | 3.922 | -0.185 | -4.50% | 2.359 | 2.340 | 4.578e-03 | 4.578e-03 | 1.000 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.862 | 3.374 | -0.488 | -12.64% | 2.350 | 2.001 | 7.294e-03 | 7.294e-03 | 1.000 |
| spd | 4 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 3.034 | 2.808 | -0.226 | -7.45% | 0.875 | 0.701 | 3.693e-03 | 3.693e-03 | 1.000 |
| spd | 4 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.429 | 2.559 | 0.130 | 5.35% | 0.855 | 0.998 | 4.120e-03 | 4.120e-03 | 1.000 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 3.272 | 3.033 | -0.239 | -7.30% | 0.820 | 0.870 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 4 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.364 | 2.469 | 0.105 | 4.44% | 0.875 | 1.101 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.723 | 5.003 | 0.280 | 5.93% | 2.746 | 2.737 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.271 | 3.772 | -0.499 | -11.68% | 2.749 | 2.291 | 4.181e-03 | 4.181e-03 | 1.000 |

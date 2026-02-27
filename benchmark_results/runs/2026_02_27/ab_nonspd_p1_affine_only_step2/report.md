# Solver Benchmark A/B Report

Generated: 2026-02-27T18:07:35

A: baseline
B: affine_only

| kind | p | n | k | case | method | baseline_total_ms | affine_only_total_ms | delta_ms(B-A) | delta_pct | baseline_iter_ms | affine_only_iter_ms | baseline_relerr | affine_only_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.117 | 7.461 | 5.344 | 252.43% | 1.852 | 6.969 | 5.646e-03 | 0.000e+00 | 0.000 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.599 | 1.280 | -0.319 | -19.95% | 1.404 | 1.061 | 4.456e-03 | 7.660e-03 | 1.719 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.505 | 5.414 | 3.909 | 259.73% | 1.354 | 5.264 | 7.538e-03 | 0.000e+00 | 0.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 4.960 | 4.875 | -0.085 | -1.71% | 4.811 | 4.712 | 1.968e-03 | 1.968e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.567 | 5.742 | 4.175 | 266.43% | 1.418 | 5.595 | 5.829e-03 | 4.997e-04 | 0.086 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.592 | 1.381 | -0.211 | -13.25% | 1.436 | 1.204 | 5.676e-03 | 7.935e-03 | 1.398 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.595 | 5.773 | 4.178 | 261.94% | 1.443 | 5.590 | 7.507e-03 | 5.913e-04 | 0.079 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.333 | 5.101 | -0.232 | -4.35% | 5.179 | 4.948 | 9.033e-03 | 9.033e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.628 | 5.769 | 4.141 | 254.36% | 1.431 | 5.625 | 5.829e-03 | 5.684e-04 | 0.098 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.592 | 1.231 | -0.361 | -22.68% | 1.406 | 1.077 | 6.317e-03 | 8.484e-03 | 1.343 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.671 | 5.913 | 4.242 | 253.86% | 1.495 | 5.732 | 7.568e-03 | 6.409e-04 | 0.085 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.414 | 5.210 | -0.204 | -3.77% | 5.208 | 4.997 | 1.031e-02 | 1.031e-02 | 1.000 |

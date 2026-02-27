# Solver Benchmark A/B Report

Generated: 2026-02-27T18:24:47

A: safe_baseline
B: safe_diag_gated

| kind | p | n | k | case | method | safe_baseline_total_ms | safe_diag_gated_total_ms | delta_ms(B-A) | delta_pct | safe_baseline_iter_ms | safe_diag_gated_iter_ms | safe_baseline_relerr | safe_diag_gated_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.194 | 2.517 | 0.323 | 14.72% | 1.846 | 2.178 | 5.646e-03 | 5.646e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.688 | 1.564 | -0.124 | -7.35% | 1.434 | 1.398 | 4.456e-03 | 4.456e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.634 | 1.499 | -0.135 | -8.26% | 1.465 | 1.344 | 7.538e-03 | 7.599e-03 | 1.008 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.057 | 4.935 | -0.122 | -2.41% | 4.894 | 4.783 | 1.968e-03 | 1.968e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.579 | 1.816 | 0.237 | 15.01% | 1.419 | 1.670 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.547 | 2.048 | 0.501 | 32.39% | 1.348 | 1.788 | 5.676e-03 | 5.676e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.750 | 2.121 | 0.371 | 21.20% | 1.460 | 1.760 | 7.507e-03 | 7.538e-03 | 1.004 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.344 | 5.470 | 0.126 | 2.36% | 5.175 | 5.275 | 9.033e-03 | 9.033e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.560 | 1.522 | -0.038 | -2.44% | 1.411 | 1.373 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.715 | 1.528 | -0.187 | -10.90% | 1.484 | 1.354 | 6.317e-03 | 6.317e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.562 | 1.537 | -0.025 | -1.60% | 1.403 | 1.381 | 7.568e-03 | 7.629e-03 | 1.008 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.308 | 5.253 | -0.055 | -1.04% | 5.161 | 5.103 | 1.031e-02 | 1.031e-02 | 1.000 |

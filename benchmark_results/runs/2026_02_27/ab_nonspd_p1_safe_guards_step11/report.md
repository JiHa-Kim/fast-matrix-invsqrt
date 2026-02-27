# Solver Benchmark A/B Report

Generated: 2026-02-27T18:23:18

A: safe_on
B: safe_off

| kind | p | n | k | case | method | safe_on_total_ms | safe_off_total_ms | delta_ms(B-A) | delta_pct | safe_on_iter_ms | safe_off_iter_ms | safe_on_relerr | safe_off_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.011 | 1.591 | -0.420 | -20.89% | 1.797 | 1.414 | 5.646e-03 | 5.646e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.796 | 1.567 | -0.229 | -12.75% | 1.636 | 1.399 | 4.456e-03 | 4.456e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.626 | 1.315 | -0.311 | -19.13% | 1.467 | 1.133 | 7.538e-03 | 7.599e-03 | 1.008 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.000 | 1.265 | -3.735 | -74.70% | 4.848 | 1.113 | 1.968e-03 | 1.000e+00 | 508.130 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.629 | 1.297 | -0.332 | -20.38% | 1.474 | 1.111 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.528 | 1.251 | -0.277 | -18.13% | 1.373 | 1.096 | 5.676e-03 | 5.676e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.590 | 1.273 | -0.317 | -19.94% | 1.421 | 1.124 | 7.507e-03 | 7.538e-03 | 1.004 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.338 | 1.288 | -4.050 | -75.87% | 5.144 | 1.116 | 9.033e-03 | 1.000e+00 | 110.705 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.918 | 1.256 | -0.662 | -34.52% | 1.726 | 1.106 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.634 | 1.266 | -0.368 | -22.52% | 1.415 | 1.108 | 6.317e-03 | 6.317e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.720 | 1.263 | -0.457 | -26.57% | 1.564 | 1.115 | 7.568e-03 | 7.629e-03 | 1.008 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.273 | 1.275 | -3.998 | -75.82% | 5.122 | 1.118 | 1.031e-02 | 1.000e+00 | 96.993 |

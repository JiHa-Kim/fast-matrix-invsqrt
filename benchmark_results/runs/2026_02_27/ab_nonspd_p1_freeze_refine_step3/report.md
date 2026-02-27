# Solver Benchmark A/B Report

Generated: 2026-02-27T18:11:50

A: baseline
B: freeze_refine

| kind | p | n | k | case | method | baseline_total_ms | freeze_refine_total_ms | delta_ms(B-A) | delta_pct | baseline_iter_ms | freeze_refine_iter_ms | baseline_relerr | freeze_refine_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.959 | 1.961 | 0.002 | 0.10% | 1.806 | 1.804 | 5.646e-03 | 2.609e-03 | 0.462 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.555 | 1.921 | 0.366 | 23.54% | 1.386 | 1.738 | 4.456e-03 | 2.472e-03 | 0.555 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.531 | 1.639 | 0.108 | 7.05% | 1.382 | 1.484 | 7.538e-03 | 1.471e-02 | 1.951 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 4.998 | 5.921 | 0.923 | 18.47% | 4.848 | 5.769 | 1.968e-03 | 1.968e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.571 | 1.565 | -0.006 | -0.38% | 1.404 | 1.419 | 5.829e-03 | 2.518e-03 | 0.432 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.607 | 1.693 | 0.086 | 5.35% | 1.396 | 1.494 | 5.676e-03 | 2.304e-03 | 0.406 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.684 | 1.598 | -0.086 | -5.11% | 1.421 | 1.436 | 7.507e-03 | 1.556e-02 | 2.073 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.351 | 6.191 | 0.840 | 15.70% | 5.166 | 6.040 | 9.033e-03 | 9.033e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.581 | 1.578 | -0.003 | -0.19% | 1.405 | 1.421 | 5.829e-03 | 2.502e-03 | 0.429 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.550 | 1.649 | 0.099 | 6.39% | 1.397 | 1.483 | 6.317e-03 | 2.274e-03 | 0.360 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.677 | 1.612 | -0.065 | -3.88% | 1.465 | 1.457 | 7.568e-03 | 1.538e-02 | 2.032 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.306 | 6.176 | 0.870 | 16.40% | 5.150 | 6.026 | 1.031e-02 | 1.031e-02 | 1.000 |

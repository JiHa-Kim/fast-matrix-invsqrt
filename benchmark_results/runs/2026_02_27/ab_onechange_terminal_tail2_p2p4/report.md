# Solver Benchmark A/B Report

Generated: 2026-02-26T21:08:23

A: tail1
B: tail2

| kind | p | n | k | case | method | tail1_total_ms | tail2_total_ms | delta_ms(B-A) | delta_pct | tail1_iter_ms | tail2_iter_ms | tail1_relerr | tail2_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.316 | 2.561 | 0.245 | 10.58% | 0.708 | 0.687 | 3.799e-03 | 1.045e-01 | 27.507 |
| spd | 2 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.159 | 2.389 | 0.230 | 10.65% | 0.790 | 0.641 | 3.311e-03 | 1.108e-01 | 33.464 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.913 | 2.989 | 0.076 | 2.61% | 0.723 | 0.788 | 3.204e-03 | 1.118e-01 | 34.894 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.233 | 2.469 | 0.236 | 10.57% | 0.805 | 0.810 | 3.555e-03 | 1.118e-01 | 31.449 |
| spd | 2 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.274 | 4.170 | 0.896 | 27.37% | 1.585 | 0.907 | 3.769e-03 | 1.050e-01 | 27.859 |
| spd | 2 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.280 | 2.256 | -1.024 | -31.22% | 1.596 | 0.894 | 5.005e-03 | 1.006e-01 | 20.100 |
| spd | 2 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.297 | 5.431 | 2.134 | 64.73% | 1.601 | 3.803 | 3.937e-03 | 1.055e-01 | 26.797 |
| spd | 2 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.885 | 4.270 | 1.385 | 48.01% | 1.332 | 1.040 | 4.944e-03 | 1.011e-01 | 20.449 |
| spd | 2 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.121 | 5.527 | 2.406 | 77.09% | 1.332 | 2.418 | 3.860e-03 | 1.055e-01 | 27.332 |
| spd | 2 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.991 | 2.943 | -1.048 | -26.26% | 1.321 | 0.960 | 4.974e-03 | 1.011e-01 | 20.326 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.832 | 3.488 | -0.344 | -8.98% | 2.350 | 1.848 | 4.578e-03 | 1.118e-01 | 24.421 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.654 | 3.323 | -0.331 | -9.06% | 2.091 | 1.849 | 7.294e-03 | 1.050e-01 | 14.395 |
| spd | 4 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.959 | 3.683 | 0.724 | 24.47% | 0.941 | 0.645 | 3.693e-03 | 6.177e-02 | 16.726 |
| spd | 4 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.308 | 2.165 | -0.143 | -6.20% | 0.903 | 0.673 | 4.120e-03 | 6.079e-02 | 14.755 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 3.306 | 2.395 | -0.911 | -27.56% | 0.896 | 0.577 | 3.998e-03 | 6.030e-02 | 15.083 |
| spd | 4 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.679 | 2.148 | -0.531 | -19.82% | 1.030 | 0.657 | 3.418e-03 | 6.201e-02 | 18.142 |
| spd | 4 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.717 | 4.509 | 0.792 | 21.31% | 1.987 | 1.096 | 3.723e-03 | 5.884e-02 | 15.804 |
| spd | 4 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.192 | 3.178 | -0.014 | -0.44% | 1.575 | 1.503 | 3.479e-03 | 6.128e-02 | 17.614 |
| spd | 4 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 2.994 | 5.992 | 2.998 | 100.13% | 1.627 | 4.358 | 3.601e-03 | 5.884e-02 | 16.340 |
| spd | 4 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.082 | 2.796 | -0.286 | -9.28% | 1.544 | 1.389 | 3.464e-03 | 6.128e-02 | 17.691 |
| spd | 4 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.235 | 4.243 | 1.008 | 31.16% | 1.724 | 1.355 | 3.647e-03 | 5.884e-02 | 16.134 |
| spd | 4 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.344 | 7.563 | 4.219 | 126.17% | 1.654 | 4.429 | 3.464e-03 | 6.128e-02 | 17.691 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.662 | 4.839 | 0.177 | 3.80% | 2.747 | 2.023 | 3.998e-03 | 5.835e-02 | 14.595 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.608 | 3.238 | -1.370 | -29.73% | 2.362 | 1.901 | 4.181e-03 | 6.201e-02 | 14.831 |

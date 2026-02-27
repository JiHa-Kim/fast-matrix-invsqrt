# Solver Benchmark A/B Report

Generated: 2026-02-26T19:55:09

A: greedy_affine_opt
B: kappa_table_klt_only

| kind | n | k | case | method | greedy_affine_opt_total_ms | kappa_table_klt_only_total_ms | delta_ms(B-A) | delta_pct | greedy_affine_opt_iter_ms | kappa_table_klt_only_iter_ms | greedy_affine_opt_relerr | kappa_table_klt_only_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.480 | 3.212 | 0.732 | 29.52% | 0.550 | 0.553 | 2.029e-03 | 2.029e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.002 | 3.667 | 0.665 | 22.15% | 1.072 | 1.008 | 4.517e-03 | 4.517e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 3.451 | 3.989 | 0.538 | 15.59% | 1.521 | 1.329 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.996 | 3.829 | 0.833 | 27.80% | 1.066 | 1.170 | 3.693e-03 | 3.693e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 3.423 | 4.002 | 0.579 | 16.91% | 1.493 | 1.342 | 3.571e-03 | 3.571e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 2.420 | 2.749 | 0.329 | 13.60% | 0.631 | 0.556 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.786 | 3.180 | 0.394 | 14.14% | 0.998 | 0.987 | 3.784e-03 | 3.784e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 3.456 | 4.094 | 0.638 | 18.46% | 1.668 | 1.902 | 4.547e-03 | 4.547e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.825 | 3.306 | 0.481 | 17.03% | 1.036 | 1.113 | 4.120e-03 | 4.120e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 3.396 | 3.787 | 0.391 | 11.51% | 1.608 | 1.594 | 4.089e-03 | 4.089e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 4.727 | 3.974 | -0.753 | -15.93% | 1.249 | 1.204 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 6.098 | 4.137 | -1.961 | -32.16% | 2.620 | 1.367 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.115 | 4.523 | -0.592 | -11.57% | 1.637 | 1.753 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 4.492 | 3.908 | -0.584 | -13.00% | 1.014 | 1.138 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.107 | 4.194 | -0.913 | -17.88% | 1.629 | 1.424 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 3.304 | 3.619 | 0.315 | 9.53% | 1.109 | 1.110 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.255 | 3.544 | 0.289 | 8.88% | 1.059 | 1.035 | 4.700e-03 | 4.700e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.345 | 4.759 | 0.414 | 9.53% | 2.150 | 2.250 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.341 | 3.433 | 0.092 | 2.75% | 1.146 | 0.924 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.422 | 5.217 | 0.795 | 17.98% | 2.227 | 2.708 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.136 | 3.979 | 1.843 | 86.28% | 0.585 | 0.585 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.609 | 5.489 | 1.880 | 52.09% | 2.057 | 2.095 | 4.028e-03 | 4.028e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.337 | 7.426 | 2.089 | 39.14% | 3.786 | 4.033 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.181 | 5.156 | 1.975 | 62.09% | 1.630 | 1.762 | 3.723e-03 | 3.723e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.096 | 7.943 | 1.847 | 30.30% | 4.545 | 4.549 | 4.333e-03 | 4.333e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.124 | 2.389 | 0.265 | 12.48% | 0.583 | 0.596 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.644 | 4.044 | 0.400 | 10.98% | 2.104 | 2.251 | 4.639e-03 | 4.639e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.415 | 5.975 | 0.560 | 10.34% | 3.874 | 4.182 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.219 | 3.599 | 0.380 | 11.80% | 1.679 | 1.806 | 3.479e-03 | 3.357e-03 | 0.965 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.401 | 6.000 | 0.599 | 11.09% | 3.861 | 4.207 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.127 | 3.316 | 1.189 | 55.90% | 0.653 | 0.654 | 2.075e-03 | 2.075e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.606 | 5.294 | 1.688 | 46.81% | 2.132 | 2.632 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.422 | 6.718 | 1.296 | 23.90% | 3.947 | 4.056 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.181 | 4.419 | 1.238 | 38.92% | 1.706 | 1.758 | 3.601e-03 | 3.677e-03 | 1.021 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.439 | 6.721 | 1.282 | 23.57% | 3.965 | 4.060 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.024 | 2.385 | 0.361 | 17.84% | 0.659 | 0.720 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.514 | 3.907 | 0.393 | 11.18% | 2.148 | 2.243 | 4.639e-03 | 4.639e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.354 | 5.771 | 0.417 | 7.79% | 3.988 | 4.107 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.076 | 3.444 | 0.368 | 11.96% | 1.711 | 1.780 | 3.464e-03 | 3.357e-03 | 0.969 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.374 | 5.792 | 0.418 | 7.78% | 4.009 | 4.128 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.054 | 2.922 | 0.868 | 42.26% | 0.722 | 0.762 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.498 | 4.472 | 0.974 | 27.84% | 2.167 | 2.312 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.357 | 6.301 | 0.944 | 17.62% | 4.026 | 4.141 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.053 | 3.990 | 0.937 | 30.69% | 1.721 | 1.830 | 3.647e-03 | 3.693e-03 | 1.013 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.375 | 6.360 | 0.985 | 18.33% | 4.044 | 4.201 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.073 | 2.533 | 0.460 | 22.19% | 0.717 | 0.742 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.531 | 4.147 | 0.616 | 17.45% | 2.175 | 2.356 | 4.608e-03 | 4.608e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.352 | 6.063 | 0.711 | 13.28% | 3.996 | 4.272 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.105 | 3.615 | 0.510 | 16.43% | 1.749 | 1.825 | 3.464e-03 | 3.387e-03 | 0.978 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.377 | 6.151 | 0.774 | 14.39% | 4.021 | 4.360 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 12.344 | 12.692 | 0.348 | 2.82% | 9.269 | 10.140 | 2.151e-03 | 2.151e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 5.770 | 5.501 | -0.269 | -4.66% | 2.696 | 2.950 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 7.240 | 6.732 | -0.508 | -7.02% | 4.165 | 4.180 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 5.333 | 5.101 | -0.232 | -4.35% | 2.258 | 2.549 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 7.504 | 7.341 | -0.163 | -2.17% | 4.429 | 4.790 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 10.766 | 12.166 | 1.400 | 13.00% | 9.354 | 9.880 | 2.167e-03 | 2.167e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.150 | 5.326 | 1.176 | 28.34% | 2.739 | 3.040 | 4.944e-03 | 4.944e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.443 | 6.864 | 1.421 | 26.11% | 4.031 | 4.577 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.732 | 4.914 | 1.182 | 31.67% | 2.320 | 2.628 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.435 | 6.622 | 1.187 | 21.84% | 4.024 | 4.335 | 3.082e-03 | 3.082e-03 | 1.000 |

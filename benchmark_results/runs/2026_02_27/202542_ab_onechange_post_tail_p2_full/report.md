# Solver Benchmark A/B Report

Generated: 2026-02-26T20:25:42

A: post0
B: post1

| kind | n | k | case | method | post0_total_ms | post1_total_ms | delta_ms(B-A) | delta_pct | post0_iter_ms | post1_iter_ms | post0_relerr | post1_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.196 | 2.083 | -0.113 | -5.15% | 0.549 | 0.555 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.497 | 3.170 | 0.673 | 26.95% | 0.849 | 1.641 | 3.357e-03 | 3.571e-03 | 1.064 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.709 | 2.566 | -0.143 | -5.28% | 1.062 | 1.037 | 3.448e-03 | 3.448e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.320 | 2.481 | 0.161 | 6.94% | 0.673 | 0.952 | 3.799e-03 | 4.120e-03 | 1.084 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.656 | 2.580 | -0.076 | -2.86% | 1.009 | 1.051 | 3.448e-03 | 3.448e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 2.062 | 4.325 | 2.263 | 109.75% | 0.551 | 0.548 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.505 | 5.199 | 2.694 | 107.54% | 0.994 | 1.422 | 3.357e-03 | 3.601e-03 | 1.073 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.776 | 5.119 | 2.343 | 84.40% | 1.265 | 1.341 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.402 | 5.194 | 2.792 | 116.24% | 0.891 | 1.417 | 3.311e-03 | 3.525e-03 | 1.065 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.697 | 5.003 | 2.306 | 85.50% | 1.186 | 1.226 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 3.147 | 3.378 | 0.231 | 7.34% | 1.435 | 1.355 | 3.052e-03 | 3.052e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.566 | 3.154 | 0.588 | 22.92% | 0.853 | 1.131 | 3.311e-03 | 3.525e-03 | 1.065 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.808 | 3.104 | 0.296 | 10.54% | 1.096 | 1.081 | 1.328e-03 | 1.328e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.397 | 2.937 | 0.540 | 22.53% | 0.684 | 0.914 | 3.204e-03 | 3.448e-03 | 1.076 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.828 | 3.071 | 0.243 | 8.59% | 1.115 | 1.048 | 1.328e-03 | 1.328e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.655 | 2.595 | -0.060 | -2.26% | 1.107 | 1.109 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.391 | 2.647 | 0.256 | 10.71% | 0.843 | 1.161 | 3.311e-03 | 3.525e-03 | 1.065 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.920 | 2.843 | -0.077 | -2.64% | 1.373 | 1.356 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.691 | 2.781 | 0.090 | 3.34% | 1.143 | 1.295 | 3.555e-03 | 5.341e-03 | 1.502 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.993 | 2.870 | -0.123 | -4.11% | 1.445 | 1.383 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.317 | 2.147 | -0.170 | -7.34% | 0.584 | 0.584 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.397 | 3.865 | 0.468 | 13.78% | 1.664 | 2.302 | 3.860e-03 | 4.089e-03 | 1.059 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.501 | 4.687 | -0.814 | -14.80% | 3.768 | 3.124 | 6.409e-04 | 6.409e-04 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.170 | 3.499 | 0.329 | 10.38% | 1.436 | 1.936 | 3.769e-03 | 6.256e-03 | 1.660 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.512 | 5.342 | -0.170 | -3.08% | 3.779 | 3.779 | 6.180e-04 | 6.180e-04 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 4.346 | 2.296 | -2.050 | -47.17% | 0.586 | 0.631 | 3.021e-03 | 3.021e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 5.480 | 3.987 | -1.493 | -27.24% | 1.720 | 2.322 | 3.967e-03 | 4.181e-03 | 1.054 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 6.986 | 4.964 | -2.022 | -28.94% | 3.225 | 3.299 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 5.085 | 3.721 | -1.364 | -26.82% | 1.324 | 2.056 | 5.005e-03 | 5.859e-03 | 1.171 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 6.941 | 4.900 | -2.041 | -29.40% | 3.180 | 3.235 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.040 | 3.024 | 0.984 | 48.24% | 0.664 | 0.652 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.122 | 4.729 | 1.607 | 51.47% | 1.746 | 2.356 | 3.754e-03 | 3.937e-03 | 1.049 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.715 | 5.876 | 1.161 | 24.62% | 3.339 | 3.504 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 2.747 | 4.521 | 1.774 | 64.58% | 1.371 | 2.149 | 3.937e-03 | 6.500e-03 | 1.651 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 4.787 | 6.069 | 1.282 | 26.78% | 3.411 | 3.697 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.207 | 2.182 | -0.025 | -1.13% | 0.653 | 0.654 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.300 | 4.095 | 0.795 | 24.09% | 1.746 | 2.567 | 3.937e-03 | 4.150e-03 | 1.054 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.908 | 5.103 | 0.195 | 3.97% | 3.354 | 3.575 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.989 | 3.834 | 0.845 | 28.27% | 1.434 | 2.306 | 4.944e-03 | 5.768e-03 | 1.167 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.985 | 4.957 | -0.028 | -0.56% | 3.431 | 3.428 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.173 | 2.399 | 0.226 | 10.40% | 0.708 | 0.743 | 3.143e-03 | 3.143e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.253 | 4.017 | 0.764 | 23.49% | 1.788 | 2.361 | 3.754e-03 | 3.937e-03 | 1.049 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.844 | 5.161 | 0.317 | 6.54% | 3.379 | 3.505 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 2.894 | 3.875 | 0.981 | 33.90% | 1.429 | 2.219 | 3.860e-03 | 6.348e-03 | 1.645 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 4.787 | 5.021 | 0.234 | 4.89% | 3.322 | 3.365 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.233 | 2.581 | 0.348 | 15.58% | 0.744 | 0.721 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.258 | 4.238 | 0.980 | 30.08% | 1.769 | 2.378 | 3.937e-03 | 4.150e-03 | 1.054 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.894 | 5.215 | 0.321 | 6.56% | 3.405 | 3.355 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.973 | 3.977 | 1.004 | 33.77% | 1.484 | 2.117 | 4.974e-03 | 5.737e-03 | 1.153 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.834 | 5.421 | 0.587 | 12.14% | 3.346 | 3.561 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 10.820 | 11.202 | 0.382 | 3.53% | 9.266 | 9.286 | 3.113e-03 | 3.113e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.916 | 5.206 | 1.290 | 32.94% | 2.363 | 3.289 | 5.737e-03 | 5.859e-03 | 1.021 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.826 | 5.188 | 0.362 | 7.50% | 3.272 | 3.272 | 2.090e-03 | 2.090e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.482 | 4.665 | 1.183 | 33.97% | 1.929 | 2.749 | 4.578e-03 | 4.333e-03 | 0.946 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.502 | 5.861 | 0.359 | 6.52% | 3.949 | 3.944 | 2.090e-03 | 2.090e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 11.182 | 10.661 | -0.521 | -4.66% | 9.611 | 9.300 | 3.159e-03 | 3.159e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.084 | 4.669 | 0.585 | 14.32% | 2.513 | 3.309 | 5.951e-03 | 6.073e-03 | 1.021 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.060 | 4.670 | -0.390 | -7.71% | 3.490 | 3.309 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.650 | 4.170 | 0.520 | 14.25% | 2.079 | 2.809 | 7.294e-03 | 4.852e-03 | 0.665 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.018 | 4.666 | -0.352 | -7.01% | 3.447 | 3.306 | 3.326e-03 | 3.326e-03 | 1.000 |

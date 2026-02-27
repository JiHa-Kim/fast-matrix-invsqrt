# Solver Benchmark A/B Report

Generated: 2026-02-26T19:53:20

A: greedy_affine_opt
B: kappa_table_klt_only

| kind | n | k | case | method | greedy_affine_opt_total_ms | kappa_table_klt_only_total_ms | delta_ms(B-A) | delta_pct | greedy_affine_opt_iter_ms | kappa_table_klt_only_iter_ms | greedy_affine_opt_relerr | kappa_table_klt_only_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.035 | 2.348 | 0.313 | 15.38% | 0.548 | 0.549 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.326 | 2.708 | 0.382 | 16.42% | 0.839 | 0.909 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.560 | 2.889 | 0.329 | 12.85% | 1.073 | 1.090 | 3.448e-03 | 3.448e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.142 | 2.550 | 0.408 | 19.05% | 0.655 | 0.751 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.599 | 2.889 | 0.290 | 11.16% | 1.112 | 1.090 | 3.448e-03 | 3.448e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 1.953 | 2.094 | 0.141 | 7.22% | 0.549 | 0.548 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.341 | 2.524 | 0.183 | 7.82% | 0.937 | 0.978 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.801 | 2.767 | -0.034 | -1.21% | 1.397 | 1.221 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.138 | 2.368 | 0.230 | 10.76% | 0.734 | 0.823 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.726 | 2.759 | 0.033 | 1.21% | 1.321 | 1.214 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 3.067 | 3.285 | 0.218 | 7.11% | 1.271 | 1.436 | 3.052e-03 | 3.052e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.635 | 2.667 | 0.032 | 1.21% | 0.839 | 0.818 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.818 | 2.913 | 0.095 | 3.37% | 1.022 | 1.064 | 1.328e-03 | 1.328e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.446 | 2.507 | 0.061 | 2.49% | 0.650 | 0.658 | 3.204e-03 | 3.204e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.847 | 2.885 | 0.038 | 1.33% | 1.051 | 1.035 | 1.328e-03 | 1.328e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.616 | 2.550 | -0.066 | -2.52% | 1.188 | 1.107 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.464 | 2.474 | 0.010 | 0.41% | 1.037 | 1.031 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.703 | 2.710 | 0.007 | 0.26% | 1.276 | 1.268 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.354 | 2.270 | -0.084 | -3.57% | 0.927 | 0.827 | 3.555e-03 | 3.555e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.557 | 2.748 | 0.191 | 7.47% | 1.129 | 1.305 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.087 | 2.061 | -0.026 | -1.25% | 0.588 | 0.589 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.148 | 3.127 | -0.021 | -0.67% | 1.649 | 1.655 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.079 | 5.237 | 0.158 | 3.11% | 3.580 | 3.764 | 6.409e-04 | 6.409e-04 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 2.790 | 2.730 | -0.060 | -2.15% | 1.291 | 1.257 | 3.769e-03 | 3.769e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.279 | 5.238 | -0.041 | -0.78% | 3.780 | 3.765 | 6.180e-04 | 6.180e-04 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 1.953 | 1.983 | 0.030 | 1.54% | 0.591 | 0.582 | 3.021e-03 | 3.021e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.084 | 3.098 | 0.014 | 0.45% | 1.722 | 1.697 | 3.967e-03 | 3.967e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.603 | 4.621 | 0.018 | 0.39% | 3.241 | 3.220 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.737 | 2.756 | 0.019 | 0.69% | 1.375 | 1.355 | 5.005e-03 | 5.005e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.555 | 4.625 | 0.070 | 1.54% | 3.193 | 3.225 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.237 | 2.141 | -0.096 | -4.29% | 0.664 | 0.653 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.341 | 3.218 | -0.123 | -3.68% | 1.769 | 1.730 | 3.754e-03 | 3.754e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.951 | 4.768 | -0.183 | -3.70% | 3.378 | 3.280 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 2.983 | 2.848 | -0.135 | -4.53% | 1.410 | 1.360 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 4.948 | 4.738 | -0.210 | -4.24% | 3.376 | 3.250 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.293 | 2.054 | -0.239 | -10.42% | 0.658 | 0.653 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.396 | 3.119 | -0.277 | -8.16% | 1.761 | 1.719 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.022 | 4.680 | -0.342 | -6.81% | 3.388 | 3.280 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.052 | 2.787 | -0.265 | -8.68% | 1.417 | 1.387 | 4.944e-03 | 4.944e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.993 | 4.675 | -0.318 | -6.37% | 3.358 | 3.275 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.235 | 1.983 | -0.252 | -11.28% | 0.711 | 0.712 | 3.143e-03 | 3.143e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.306 | 3.038 | -0.268 | -8.11% | 1.782 | 1.767 | 3.754e-03 | 3.754e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.902 | 4.554 | -0.348 | -7.10% | 3.378 | 3.283 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 2.979 | 2.669 | -0.310 | -10.41% | 1.456 | 1.398 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 4.879 | 4.554 | -0.325 | -6.66% | 3.356 | 3.283 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.115 | 2.058 | -0.057 | -2.70% | 0.711 | 0.710 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.185 | 3.712 | 0.527 | 16.55% | 1.781 | 2.364 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.778 | 4.637 | -0.141 | -2.95% | 3.374 | 3.289 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.803 | 2.731 | -0.072 | -2.57% | 1.399 | 1.383 | 4.974e-03 | 4.974e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.784 | 4.607 | -0.177 | -3.70% | 3.380 | 3.259 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 11.424 | 11.124 | -0.300 | -2.63% | 9.248 | 9.273 | 3.113e-03 | 3.113e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.521 | 4.208 | -0.313 | -6.92% | 2.346 | 2.358 | 5.737e-03 | 5.737e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.991 | 5.121 | -0.870 | -14.52% | 3.815 | 3.271 | 2.090e-03 | 2.090e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.088 | 3.761 | -0.327 | -8.00% | 1.913 | 1.911 | 4.578e-03 | 4.578e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.071 | 5.791 | -0.280 | -4.61% | 3.896 | 3.941 | 2.090e-03 | 2.090e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 10.694 | 10.837 | 0.143 | 1.34% | 9.393 | 9.431 | 3.159e-03 | 3.159e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.728 | 3.841 | 0.113 | 3.03% | 2.427 | 2.435 | 5.951e-03 | 5.951e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.661 | 4.793 | 0.132 | 2.83% | 3.360 | 3.388 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.290 | 3.421 | 0.131 | 3.98% | 1.988 | 2.015 | 7.294e-03 | 7.294e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.628 | 4.807 | 0.179 | 3.87% | 3.327 | 3.402 | 3.326e-03 | 3.326e-03 | 1.000 |

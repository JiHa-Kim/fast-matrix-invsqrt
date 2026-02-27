# Solver Benchmark A/B Report

Generated: 2026-02-26T19:39:04

A: greedy_affine_opt
B: staged_minimax_newton

| kind | n | k | case | method | greedy_affine_opt_total_ms | staged_minimax_newton_total_ms | delta_ms(B-A) | delta_pct | greedy_affine_opt_iter_ms | staged_minimax_newton_iter_ms | greedy_affine_opt_relerr | staged_minimax_newton_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.655 | 2.246 | -0.409 | -15.40% | 0.547 | 0.548 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.915 | 2.496 | -0.419 | -14.37% | 0.808 | 0.797 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 3.194 | 2.707 | -0.487 | -15.25% | 1.087 | 1.009 | 3.448e-03 | 3.448e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.765 | 2.322 | -0.443 | -16.02% | 0.657 | 0.623 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 3.187 | 2.704 | -0.483 | -15.16% | 1.080 | 1.005 | 3.448e-03 | 3.448e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 1.973 | 2.045 | 0.072 | 3.65% | 0.549 | 0.549 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.509 | 2.315 | -0.194 | -7.73% | 1.084 | 0.819 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.706 | 2.765 | 0.059 | 2.18% | 1.282 | 1.269 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.238 | 2.381 | 0.143 | 6.39% | 0.814 | 0.884 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.660 | 2.813 | 0.153 | 5.75% | 1.235 | 1.316 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 3.380 | 3.179 | -0.201 | -5.95% | 1.292 | 1.437 | 3.052e-03 | 3.052e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.923 | 2.584 | -0.339 | -11.60% | 0.835 | 0.843 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 3.275 | 2.996 | -0.279 | -8.52% | 1.187 | 1.254 | 1.328e-03 | 1.328e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.821 | 2.536 | -0.285 | -10.10% | 0.733 | 0.795 | 3.204e-03 | 3.204e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 3.330 | 2.944 | -0.386 | -11.59% | 1.243 | 1.202 | 1.328e-03 | 1.328e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 3.082 | 2.908 | -0.174 | -5.65% | 1.289 | 1.184 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.768 | 2.563 | -0.205 | -7.41% | 0.975 | 0.839 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 3.045 | 3.058 | 0.013 | 0.43% | 1.251 | 1.334 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.656 | 2.526 | -0.130 | -4.89% | 0.863 | 0.802 | 3.555e-03 | 3.555e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 3.053 | 3.001 | -0.052 | -1.70% | 1.260 | 1.277 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.439 | 2.455 | 0.016 | 0.66% | 0.583 | 0.581 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.523 | 3.544 | 0.021 | 0.60% | 1.667 | 1.670 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.622 | 5.272 | -0.350 | -6.23% | 3.766 | 3.398 | 6.409e-04 | 6.409e-04 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.158 | 3.127 | -0.031 | -0.98% | 1.302 | 1.253 | 3.769e-03 | 3.525e-03 | 0.935 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.631 | 5.645 | 0.014 | 0.25% | 3.774 | 3.771 | 6.180e-04 | 6.180e-04 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 4.377 | 2.172 | -2.205 | -50.38% | 0.584 | 0.581 | 3.021e-03 | 3.021e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 5.497 | 3.291 | -2.206 | -40.13% | 1.704 | 1.700 | 3.967e-03 | 3.967e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 7.047 | 4.915 | -2.132 | -30.25% | 3.254 | 3.324 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 5.171 | 2.912 | -2.259 | -43.69% | 1.379 | 1.322 | 5.005e-03 | 5.219e-03 | 1.043 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 7.041 | 4.784 | -2.257 | -32.06% | 3.249 | 3.193 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.274 | 2.086 | -0.188 | -8.27% | 0.650 | 0.655 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.367 | 3.188 | -0.179 | -5.32% | 1.743 | 1.757 | 3.754e-03 | 3.754e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.952 | 4.996 | 0.044 | 0.89% | 3.328 | 3.565 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.032 | 2.803 | -0.229 | -7.55% | 1.408 | 1.372 | 3.937e-03 | 3.693e-03 | 0.938 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 4.923 | 4.698 | -0.225 | -4.57% | 3.299 | 3.268 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.173 | 2.355 | 0.182 | 8.38% | 0.658 | 0.669 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.271 | 3.441 | 0.170 | 5.20% | 1.757 | 1.756 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.857 | 5.064 | 0.207 | 4.26% | 3.342 | 3.379 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.923 | 3.079 | 0.156 | 5.34% | 1.409 | 1.393 | 4.944e-03 | 5.249e-03 | 1.062 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.866 | 5.020 | 0.154 | 3.16% | 3.351 | 3.334 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.092 | 2.866 | -1.226 | -29.96% | 0.714 | 0.714 | 3.143e-03 | 3.143e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 5.140 | 4.063 | -1.077 | -20.95% | 1.762 | 1.911 | 3.754e-03 | 3.754e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 6.738 | 5.708 | -1.030 | -15.29% | 3.360 | 3.556 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 4.799 | 3.667 | -1.132 | -23.59% | 1.420 | 1.516 | 3.860e-03 | 3.693e-03 | 0.957 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.728 | 5.751 | -0.977 | -14.52% | 3.350 | 3.600 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.132 | 2.393 | 0.261 | 12.24% | 0.715 | 0.712 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.191 | 3.465 | 0.274 | 8.59% | 1.773 | 1.784 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.781 | 5.045 | 0.264 | 5.52% | 3.363 | 3.365 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.824 | 3.057 | 0.233 | 8.25% | 1.406 | 1.376 | 4.974e-03 | 5.219e-03 | 1.049 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 4.799 | 5.045 | 0.246 | 5.13% | 3.381 | 3.365 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 10.928 | 10.779 | -0.149 | -1.36% | 9.271 | 9.265 | 3.113e-03 | 3.113e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.022 | 3.886 | -0.136 | -3.38% | 2.365 | 2.372 | 5.737e-03 | 5.737e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.959 | 4.817 | -0.142 | -2.86% | 3.302 | 3.303 | 2.090e-03 | 2.090e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.571 | 3.220 | -0.351 | -9.83% | 1.914 | 1.706 | 4.578e-03 | 4.211e-03 | 0.920 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.602 | 5.463 | -0.139 | -2.48% | 3.945 | 3.950 | 2.090e-03 | 2.090e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 11.444 | 11.032 | -0.412 | -3.60% | 9.614 | 9.498 | 3.159e-03 | 3.159e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.327 | 4.008 | -0.319 | -7.37% | 2.497 | 2.475 | 5.951e-03 | 5.951e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.327 | 5.011 | -0.316 | -5.93% | 3.497 | 3.478 | 3.326e-03 | 3.326e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.887 | 3.418 | -0.469 | -12.07% | 2.057 | 1.884 | 7.294e-03 | 5.981e-03 | 0.820 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.302 | 4.965 | -0.337 | -6.36% | 3.471 | 3.432 | 3.326e-03 | 3.326e-03 | 1.000 |

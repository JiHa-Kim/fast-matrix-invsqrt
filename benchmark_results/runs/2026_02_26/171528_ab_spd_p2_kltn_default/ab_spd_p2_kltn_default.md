# Solver Benchmark A/B Report

Generated: 2026-02-26T17:15:28

A: target0
B: target001

| kind | n | k | case | method | target0_total_ms | target001_total_ms | delta_ms(B-A) | delta_pct | target0_iter_ms | target001_iter_ms | target0_relerr | target001_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 7.394 | 3.345 | -4.049 | -54.76% | 1.693 | 1.662 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 8.843 | 4.816 | -4.027 | -45.54% | 3.142 | 3.134 | 6.409e-04 | 6.409e-04 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 7.591 | 2.987 | -4.604 | -60.65% | 1.890 | 1.305 | 3.616e-03 | 3.769e-03 | 1.042 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 8.904 | 5.448 | -3.456 | -38.81% | 3.203 | 3.765 | 6.180e-04 | 6.180e-04 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.967 | 3.367 | -0.600 | -15.12% | 1.998 | 1.701 | 3.967e-03 | 3.967e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.465 | 4.880 | -0.585 | -10.70% | 3.496 | 3.215 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.351 | 3.017 | -1.334 | -30.66% | 2.382 | 1.351 | 5.188e-03 | 5.005e-03 | 0.965 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.273 | 4.870 | -0.403 | -7.64% | 3.304 | 3.204 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.327 | 3.186 | -1.141 | -26.37% | 1.874 | 1.748 | 3.754e-03 | 3.754e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.809 | 4.809 | -1.000 | -17.21% | 3.355 | 3.371 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 4.502 | 2.859 | -1.643 | -36.49% | 2.049 | 1.421 | 3.769e-03 | 3.937e-03 | 1.045 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.800 | 4.790 | -1.010 | -17.41% | 3.346 | 3.352 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.261 | 3.316 | 0.055 | 1.69% | 1.774 | 1.772 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 4.995 | 4.909 | -0.086 | -1.72% | 3.508 | 3.365 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.538 | 3.004 | -0.534 | -15.09% | 2.051 | 1.460 | 5.096e-03 | 4.944e-03 | 0.970 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.044 | 4.938 | -0.106 | -2.10% | 3.557 | 3.394 | 3.235e-03 | 3.235e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.438 | 3.197 | -0.241 | -7.01% | 1.806 | 1.800 | 3.754e-03 | 3.754e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 4.977 | 4.783 | -0.194 | -3.90% | 3.346 | 3.386 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.687 | 2.814 | -0.873 | -23.68% | 2.056 | 1.417 | 3.769e-03 | 3.860e-03 | 1.024 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 4.979 | 4.795 | -0.184 | -3.70% | 3.347 | 3.398 | 1.343e-03 | 1.343e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.760 | 3.375 | -0.385 | -10.24% | 1.776 | 1.794 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.331 | 4.957 | -0.374 | -7.02% | 3.348 | 3.375 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.060 | 2.996 | -1.064 | -26.21% | 2.076 | 1.415 | 5.096e-03 | 4.974e-03 | 0.976 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.318 | 4.961 | -0.357 | -6.71% | 3.334 | 3.380 | 3.220e-03 | 3.220e-03 | 1.000 |
| spd | 2048 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 12.383 | 13.649 | 1.266 | 10.22% | 9.848 | 9.750 | 3.677e-03 | 3.677e-03 | 1.000 |
| spd | 2048 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 24.017 | 24.531 | 0.514 | 2.14% | 21.482 | 20.632 | 1.099e-03 | 1.099e-03 | 1.000 |
| spd | 2048 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 15.350 | 12.177 | -3.173 | -20.67% | 12.815 | 8.278 | 3.510e-03 | 3.815e-03 | 1.087 |
| spd | 2048 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 24.386 | 25.145 | 0.759 | 3.11% | 21.851 | 21.246 | 1.114e-03 | 1.114e-03 | 1.000 |
| spd | 2048 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 12.696 | 11.958 | -0.738 | -5.81% | 9.716 | 9.719 | 3.677e-03 | 3.677e-03 | 1.000 |
| spd | 2048 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 24.004 | 22.901 | -1.103 | -4.60% | 21.025 | 20.662 | 1.274e-03 | 1.274e-03 | 1.000 |
| spd | 2048 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 15.311 | 10.545 | -4.766 | -31.13% | 12.331 | 8.306 | 3.571e-03 | 3.830e-03 | 1.073 |
| spd | 2048 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 24.548 | 23.695 | -0.853 | -3.47% | 21.568 | 21.456 | 1.251e-03 | 1.251e-03 | 1.000 |
| spd | 2048 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 11.996 | 12.057 | 0.061 | 0.51% | 9.717 | 9.775 | 3.815e-03 | 3.815e-03 | 1.000 |
| spd | 2048 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 23.144 | 23.247 | 0.103 | 0.45% | 20.865 | 20.965 | 2.853e-03 | 2.853e-03 | 1.000 |
| spd | 2048 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 14.700 | 10.624 | -4.076 | -27.73% | 12.421 | 8.342 | 4.272e-03 | 4.456e-03 | 1.043 |
| spd | 2048 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 24.322 | 24.044 | -0.278 | -1.14% | 22.043 | 21.762 | 2.853e-03 | 2.853e-03 | 1.000 |
| spd | 2048 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 11.891 | 12.113 | 0.222 | 1.87% | 9.683 | 9.857 | 3.845e-03 | 3.845e-03 | 1.000 |
| spd | 2048 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 22.948 | 23.064 | 0.116 | 0.51% | 20.740 | 20.808 | 2.869e-03 | 2.869e-03 | 1.000 |
| spd | 2048 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 14.606 | 10.598 | -4.008 | -27.44% | 12.398 | 8.341 | 4.333e-03 | 4.517e-03 | 1.042 |
| spd | 2048 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 23.910 | 23.727 | -0.183 | -0.77% | 21.702 | 21.470 | 2.869e-03 | 2.869e-03 | 1.000 |
| spd | 2048 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 12.710 | 12.076 | -0.634 | -4.99% | 9.729 | 9.759 | 3.662e-03 | 3.662e-03 | 1.000 |
| spd | 2048 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 23.844 | 22.953 | -0.891 | -3.74% | 20.863 | 20.635 | 1.167e-03 | 1.167e-03 | 1.000 |
| spd | 2048 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 15.324 | 10.647 | -4.677 | -30.52% | 12.342 | 8.329 | 3.525e-03 | 3.784e-03 | 1.073 |
| spd | 2048 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 24.709 | 23.724 | -0.985 | -3.99% | 21.727 | 21.406 | 1.160e-03 | 1.160e-03 | 1.000 |
| spd | 2048 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 11.974 | 12.149 | 0.175 | 1.46% | 9.722 | 9.840 | 3.647e-03 | 3.647e-03 | 1.000 |
| spd | 2048 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 23.288 | 22.999 | -0.289 | -1.24% | 21.036 | 20.690 | 1.205e-03 | 1.205e-03 | 1.000 |
| spd | 2048 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 14.597 | 10.628 | -3.969 | -27.19% | 12.345 | 8.319 | 3.540e-03 | 3.830e-03 | 1.082 |
| spd | 2048 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 24.158 | 23.527 | -0.631 | -2.61% | 21.906 | 21.218 | 1.205e-03 | 1.205e-03 | 1.000 |

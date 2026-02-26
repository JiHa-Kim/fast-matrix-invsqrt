# Solver Benchmark A/B Report

Generated: 2026-02-26T17:17:56

A: target0
B: target001

| kind | n | k | case | method | target0_total_ms | target001_total_ms | delta_ms(B-A) | delta_pct | target0_iter_ms | target001_iter_ms | target0_relerr | target001_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.048 | 5.731 | 1.683 | 41.58% | 2.085 | 2.168 | 4.028e-03 | 4.028e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.717 | 7.629 | 1.912 | 33.44% | 3.755 | 4.066 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 4.418 | 5.332 | 0.914 | 20.69% | 2.456 | 1.769 | 3.662e-03 | 3.723e-03 | 1.017 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.505 | 8.370 | 1.865 | 28.67% | 4.543 | 4.807 | 4.333e-03 | 4.333e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.589 | 4.329 | 0.740 | 20.62% | 2.141 | 2.164 | 4.639e-03 | 4.639e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.346 | 6.202 | 0.856 | 16.01% | 3.899 | 4.037 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.990 | 3.891 | -0.099 | -2.48% | 2.542 | 1.725 | 3.937e-03 | 3.479e-03 | 0.884 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.300 | 6.210 | 0.910 | 17.17% | 3.852 | 4.044 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.683 | 3.550 | -0.133 | -3.61% | 2.193 | 2.197 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.525 | 5.411 | -0.114 | -2.06% | 4.034 | 4.058 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 4.132 | 3.119 | -1.013 | -24.52% | 2.642 | 1.766 | 3.677e-03 | 3.601e-03 | 0.979 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.506 | 5.439 | -0.067 | -1.22% | 4.016 | 4.086 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.881 | 3.708 | -0.173 | -4.46% | 2.183 | 2.216 | 4.639e-03 | 4.639e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.753 | 5.569 | -0.184 | -3.20% | 4.055 | 4.076 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.345 | 3.268 | -1.077 | -24.79% | 2.647 | 1.776 | 3.876e-03 | 3.464e-03 | 0.894 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.761 | 5.594 | -0.167 | -2.90% | 4.063 | 4.101 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.558 | 3.634 | 0.076 | 2.14% | 2.209 | 2.238 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.407 | 5.473 | 0.066 | 1.22% | 4.058 | 4.077 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 4.004 | 3.166 | -0.838 | -20.93% | 2.655 | 1.769 | 3.693e-03 | 3.647e-03 | 0.988 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.392 | 5.469 | 0.077 | 1.43% | 4.044 | 4.073 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.782 | 3.796 | 0.014 | 0.37% | 2.205 | 2.241 | 4.608e-03 | 4.608e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.633 | 5.638 | 0.005 | 0.09% | 4.056 | 4.083 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.261 | 3.317 | -0.944 | -22.15% | 2.685 | 1.762 | 3.876e-03 | 3.464e-03 | 0.894 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.664 | 5.642 | -0.022 | -0.39% | 4.087 | 4.087 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 2048 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 15.069 | 15.525 | 0.456 | 3.03% | 12.823 | 13.234 | 3.967e-03 | 3.967e-03 | 1.000 |
| spd | 2048 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 27.546 | 27.575 | 0.029 | 0.11% | 25.300 | 25.285 | 4.272e-03 | 4.272e-03 | 1.000 |
| spd | 2048 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 18.381 | 13.756 | -4.625 | -25.16% | 16.135 | 11.465 | 4.028e-03 | 3.937e-03 | 0.977 |
| spd | 2048 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 28.325 | 28.609 | 0.284 | 1.00% | 26.079 | 26.318 | 4.272e-03 | 4.272e-03 | 1.000 |
| spd | 2048 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 15.086 | 16.098 | 1.012 | 6.71% | 12.831 | 13.297 | 3.967e-03 | 3.967e-03 | 1.000 |
| spd | 2048 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 27.323 | 29.397 | 2.074 | 7.59% | 25.068 | 26.596 | 4.303e-03 | 4.303e-03 | 1.000 |
| spd | 2048 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 18.310 | 14.025 | -4.285 | -23.40% | 16.055 | 11.225 | 4.028e-03 | 3.876e-03 | 0.962 |
| spd | 2048 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 28.003 | 30.309 | 2.306 | 8.23% | 25.749 | 27.509 | 4.303e-03 | 4.303e-03 | 1.000 |
| spd | 2048 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 15.008 | 16.149 | 1.141 | 7.60% | 12.811 | 13.870 | 4.761e-03 | 4.761e-03 | 1.000 |
| spd | 2048 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 27.482 | 29.253 | 1.771 | 6.44% | 25.285 | 26.974 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 2048 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 18.318 | 13.531 | -4.787 | -26.13% | 16.121 | 11.252 | 4.578e-03 | 4.456e-03 | 0.973 |
| spd | 2048 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 28.844 | 30.004 | 1.160 | 4.02% | 26.647 | 27.726 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 2048 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 15.075 | 16.625 | 1.550 | 10.28% | 12.928 | 12.812 | 4.669e-03 | 4.669e-03 | 1.000 |
| spd | 2048 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 27.812 | 29.296 | 1.484 | 5.34% | 25.666 | 25.484 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 2048 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 18.251 | 14.575 | -3.676 | -20.14% | 16.104 | 10.763 | 4.547e-03 | 4.486e-03 | 0.987 |
| spd | 2048 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 30.027 | 30.113 | 0.086 | 0.29% | 27.880 | 26.301 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 2048 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 15.174 | 15.628 | 0.454 | 2.99% | 12.833 | 12.854 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 2048 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 27.439 | 28.289 | 0.850 | 3.10% | 25.099 | 25.516 | 4.303e-03 | 4.303e-03 | 1.000 |
| spd | 2048 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 18.465 | 13.505 | -4.960 | -26.86% | 16.124 | 10.731 | 4.028e-03 | 3.937e-03 | 0.977 |
| spd | 2048 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 28.992 | 29.307 | 0.315 | 1.09% | 26.651 | 26.534 | 4.303e-03 | 4.303e-03 | 1.000 |
| spd | 2048 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 16.109 | 16.584 | 0.475 | 2.95% | 13.818 | 12.817 | 3.967e-03 | 3.967e-03 | 1.000 |
| spd | 2048 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 27.816 | 29.161 | 1.345 | 4.84% | 25.524 | 25.394 | 4.333e-03 | 4.333e-03 | 1.000 |
| spd | 2048 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 18.446 | 14.605 | -3.841 | -20.82% | 16.155 | 10.838 | 4.028e-03 | 3.937e-03 | 0.977 |
| spd | 2048 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 29.449 | 30.039 | 0.590 | 2.00% | 27.158 | 26.272 | 4.333e-03 | 4.333e-03 | 1.000 |

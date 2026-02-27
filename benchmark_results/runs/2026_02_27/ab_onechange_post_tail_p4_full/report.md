# Solver Benchmark A/B Report

Generated: 2026-02-26T20:27:12

A: post0
B: post1

| kind | n | k | case | method | post0_total_ms | post1_total_ms | delta_ms(B-A) | delta_pct | post0_iter_ms | post1_iter_ms | post0_relerr | post1_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.185 | 2.591 | 0.406 | 18.58% | 0.550 | 0.548 | 2.029e-03 | 2.029e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.716 | 3.433 | 0.717 | 26.40% | 1.081 | 1.391 | 4.517e-03 | 3.998e-03 | 0.885 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.774 | 3.289 | 0.515 | 18.57% | 1.138 | 1.247 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.818 | 3.102 | 0.284 | 10.08% | 1.182 | 1.059 | 3.693e-03 | 3.998e-03 | 1.083 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.761 | 3.862 | 1.101 | 39.88% | 1.126 | 1.819 | 3.571e-03 | 3.571e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 2.121 | 1.963 | -0.158 | -7.45% | 0.552 | 0.549 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.521 | 2.673 | 0.152 | 6.03% | 0.952 | 1.259 | 3.784e-03 | 3.723e-03 | 0.984 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.947 | 2.807 | -0.140 | -4.75% | 1.378 | 1.393 | 4.547e-03 | 4.547e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.443 | 2.653 | 0.210 | 8.60% | 0.874 | 1.238 | 4.120e-03 | 4.150e-03 | 1.007 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.896 | 2.552 | -0.344 | -11.88% | 1.327 | 1.138 | 4.089e-03 | 4.089e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 3.038 | 3.341 | 0.303 | 9.97% | 1.181 | 1.330 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.767 | 3.201 | 0.434 | 15.68% | 0.910 | 1.190 | 3.860e-03 | 3.769e-03 | 0.976 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 3.048 | 3.238 | 0.190 | 6.23% | 1.191 | 1.227 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.596 | 3.984 | 1.388 | 53.47% | 0.740 | 1.972 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 3.059 | 3.185 | 0.126 | 4.12% | 1.202 | 1.174 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.593 | 2.660 | 0.067 | 2.58% | 1.167 | 1.200 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.379 | 2.668 | 0.289 | 12.15% | 0.953 | 1.209 | 4.700e-03 | 4.120e-03 | 0.877 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.815 | 2.897 | 0.082 | 2.91% | 1.389 | 1.437 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.339 | 2.554 | 0.215 | 9.19% | 0.913 | 1.095 | 3.418e-03 | 3.937e-03 | 1.152 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.847 | 2.864 | 0.017 | 0.60% | 1.421 | 1.405 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.126 | 2.331 | 0.205 | 9.64% | 0.584 | 0.590 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.586 | 4.433 | 0.847 | 23.62% | 2.045 | 2.692 | 4.028e-03 | 3.769e-03 | 0.936 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.285 | 5.475 | 0.190 | 3.60% | 3.744 | 3.734 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.145 | 4.149 | 1.004 | 31.92% | 1.604 | 2.408 | 3.723e-03 | 3.830e-03 | 1.029 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.063 | 6.281 | 0.218 | 3.60% | 4.521 | 4.540 | 4.333e-03 | 4.333e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 1.883 | 2.015 | 0.132 | 7.01% | 0.583 | 0.589 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.378 | 4.154 | 0.776 | 22.97% | 2.078 | 2.728 | 4.639e-03 | 4.059e-03 | 0.875 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.097 | 5.260 | 0.163 | 3.20% | 3.797 | 3.834 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.945 | 3.873 | 0.928 | 31.51% | 1.645 | 2.447 | 3.479e-03 | 4.120e-03 | 1.184 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.101 | 5.227 | 0.126 | 2.47% | 3.801 | 3.800 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.160 | 2.087 | -0.073 | -3.38% | 0.651 | 0.661 | 2.075e-03 | 2.075e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.646 | 4.186 | 0.540 | 14.81% | 2.136 | 2.761 | 3.998e-03 | 3.784e-03 | 0.946 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.428 | 5.343 | -0.085 | -1.57% | 3.919 | 3.917 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.214 | 3.945 | 0.731 | 22.74% | 1.704 | 2.519 | 3.601e-03 | 3.677e-03 | 1.021 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.432 | 5.327 | -0.105 | -1.93% | 3.923 | 3.901 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 1.959 | 2.041 | 0.082 | 4.19% | 0.655 | 0.656 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.448 | 4.160 | 0.712 | 20.65% | 2.144 | 2.775 | 4.639e-03 | 4.089e-03 | 0.881 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.253 | 5.308 | 0.055 | 1.05% | 3.948 | 3.923 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.997 | 3.924 | 0.927 | 30.93% | 1.693 | 2.539 | 3.464e-03 | 4.059e-03 | 1.172 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.240 | 5.287 | 0.047 | 0.90% | 3.936 | 3.902 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.118 | 2.046 | -0.072 | -3.40% | 0.715 | 0.713 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.553 | 4.130 | 0.577 | 16.24% | 2.150 | 2.798 | 3.998e-03 | 3.799e-03 | 0.950 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.343 | 5.279 | -0.064 | -1.20% | 3.940 | 3.947 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.114 | 3.884 | 0.770 | 24.73% | 1.712 | 2.552 | 3.647e-03 | 3.754e-03 | 1.029 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.347 | 5.260 | -0.087 | -1.63% | 3.944 | 3.927 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.033 | 2.165 | 0.132 | 6.49% | 0.712 | 0.716 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.453 | 4.230 | 0.777 | 22.50% | 2.132 | 2.781 | 4.608e-03 | 4.120e-03 | 0.894 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.296 | 5.398 | 0.102 | 1.93% | 3.974 | 3.949 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.022 | 3.999 | 0.977 | 32.33% | 1.700 | 2.550 | 3.464e-03 | 4.059e-03 | 1.172 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.300 | 5.387 | 0.087 | 1.64% | 3.979 | 3.938 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 11.254 | 10.965 | -0.289 | -2.57% | 9.292 | 9.304 | 2.151e-03 | 2.151e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.628 | 5.340 | 0.712 | 15.38% | 2.666 | 3.680 | 4.364e-03 | 4.181e-03 | 0.958 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.900 | 5.547 | -0.353 | -5.98% | 3.938 | 3.887 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.204 | 4.908 | 0.704 | 16.75% | 2.242 | 3.247 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.673 | 6.376 | -0.297 | -4.45% | 4.710 | 4.716 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 10.762 | 10.607 | -0.155 | -1.44% | 9.352 | 9.330 | 2.167e-03 | 2.167e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.121 | 5.002 | 0.881 | 21.38% | 2.710 | 3.725 | 4.944e-03 | 4.486e-03 | 0.907 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.422 | 5.268 | -0.154 | -2.84% | 4.011 | 3.992 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.715 | 4.596 | 0.881 | 23.71% | 2.304 | 3.319 | 4.181e-03 | 4.211e-03 | 1.007 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.391 | 5.253 | -0.138 | -2.56% | 3.981 | 3.976 | 3.082e-03 | 3.082e-03 | 1.000 |

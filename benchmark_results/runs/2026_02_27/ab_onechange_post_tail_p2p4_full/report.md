# Solver Benchmark A/B Report

Generated: 2026-02-26T20:23:28

A: post0
B: post1

| kind | n | k | case | method | post0_total_ms | post1_total_ms | delta_ms(B-A) | delta_pct | post0_iter_ms | post1_iter_ms | post0_relerr | post1_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.186 | 3.148 | 0.962 | 44.01% | 0.548 | 0.547 | 2.029e-03 | 2.029e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.621 | 3.814 | 1.193 | 45.52% | 0.983 | 1.213 | 4.517e-03 | 3.998e-03 | 0.885 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.959 | 3.814 | 0.855 | 28.89% | 1.321 | 1.213 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.419 | 3.649 | 1.230 | 50.85% | 0.781 | 1.048 | 3.693e-03 | 3.998e-03 | 1.083 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.938 | 3.796 | 0.858 | 29.20% | 1.300 | 1.194 | 3.571e-03 | 3.571e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 1.916 | 2.536 | 0.620 | 32.36% | 0.552 | 0.551 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.366 | 3.304 | 0.938 | 39.64% | 1.002 | 1.318 | 3.784e-03 | 3.723e-03 | 0.984 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.999 | 3.740 | 0.741 | 24.71% | 1.635 | 1.754 | 4.547e-03 | 4.547e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.226 | 3.401 | 1.175 | 52.79% | 0.862 | 1.416 | 4.120e-03 | 4.150e-03 | 1.007 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 3.761 | 3.677 | -0.084 | -2.23% | 2.397 | 1.692 | 4.089e-03 | 4.089e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 3.648 | 3.539 | -0.109 | -2.99% | 1.439 | 1.352 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.169 | 3.537 | 0.368 | 11.61% | 0.959 | 1.350 | 3.860e-03 | 3.769e-03 | 0.976 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 3.398 | 3.477 | 0.079 | 2.32% | 1.188 | 1.290 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.951 | 3.389 | 0.438 | 14.84% | 0.742 | 1.202 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 3.465 | 3.372 | -0.093 | -2.68% | 1.255 | 1.185 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.494 | 2.639 | 0.145 | 5.81% | 1.132 | 1.107 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.341 | 2.833 | 0.492 | 21.02% | 0.978 | 1.301 | 4.700e-03 | 4.120e-03 | 0.877 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 3.464 | 3.126 | -0.338 | -9.76% | 2.101 | 1.594 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.346 | 4.173 | 1.827 | 77.88% | 0.983 | 2.641 | 3.418e-03 | 3.937e-03 | 1.152 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.913 | 3.056 | 0.143 | 4.91% | 1.551 | 1.524 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.453 | 3.821 | 1.368 | 55.77% | 0.588 | 0.595 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.035 | 6.012 | 1.977 | 49.00% | 2.170 | 2.786 | 4.028e-03 | 3.769e-03 | 0.936 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.695 | 7.098 | 1.403 | 24.64% | 3.830 | 3.872 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.519 | 5.657 | 2.138 | 60.76% | 1.654 | 2.431 | 3.723e-03 | 3.830e-03 | 1.029 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.704 | 7.778 | 1.074 | 16.02% | 4.839 | 4.552 | 4.333e-03 | 4.333e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.744 | 2.413 | -0.331 | -12.06% | 0.586 | 0.584 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.416 | 4.662 | 0.246 | 5.57% | 2.258 | 2.833 | 4.639e-03 | 4.059e-03 | 0.875 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 6.441 | 6.075 | -0.366 | -5.68% | 4.282 | 4.246 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.855 | 4.524 | 0.669 | 17.35% | 1.696 | 2.695 | 3.479e-03 | 4.120e-03 | 1.184 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 6.191 | 6.024 | -0.167 | -2.70% | 4.033 | 4.196 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.154 | 2.196 | 0.042 | 1.95% | 0.659 | 0.668 | 2.075e-03 | 2.075e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.756 | 4.639 | 0.883 | 23.51% | 2.260 | 3.111 | 3.998e-03 | 3.784e-03 | 0.946 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.601 | 5.795 | 0.194 | 3.46% | 4.106 | 4.267 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.272 | 4.355 | 1.083 | 33.10% | 1.777 | 2.827 | 3.601e-03 | 3.677e-03 | 1.021 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.626 | 5.571 | -0.055 | -0.98% | 4.131 | 4.042 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.576 | 2.099 | -0.477 | -18.52% | 0.656 | 0.659 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.190 | 4.519 | 0.329 | 7.85% | 2.270 | 3.079 | 4.639e-03 | 4.089e-03 | 0.881 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 6.265 | 5.832 | -0.433 | -6.91% | 4.344 | 4.392 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.839 | 4.250 | 0.411 | 10.71% | 1.918 | 2.810 | 3.464e-03 | 4.059e-03 | 1.172 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 6.069 | 5.528 | -0.541 | -8.91% | 4.149 | 4.088 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.352 | 2.122 | -0.230 | -9.78% | 0.717 | 0.718 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.058 | 4.566 | 0.508 | 12.52% | 2.423 | 3.163 | 3.998e-03 | 3.799e-03 | 0.950 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.799 | 5.753 | -0.046 | -0.79% | 4.163 | 4.350 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.540 | 4.056 | 0.516 | 14.58% | 1.905 | 2.652 | 3.647e-03 | 3.754e-03 | 1.029 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.778 | 5.786 | 0.008 | 0.14% | 4.142 | 4.383 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.197 | 2.227 | 0.030 | 1.37% | 0.717 | 0.715 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.693 | 4.441 | 0.748 | 20.25% | 2.213 | 2.928 | 4.608e-03 | 4.120e-03 | 0.894 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.738 | 5.638 | -0.100 | -1.74% | 4.257 | 4.125 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.280 | 4.159 | 0.879 | 26.80% | 1.800 | 2.647 | 3.464e-03 | 4.059e-03 | 1.172 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.529 | 5.576 | 0.047 | 0.85% | 4.049 | 4.063 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 11.617 | 12.904 | 1.287 | 11.08% | 9.806 | 9.405 | 2.151e-03 | 2.151e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.708 | 7.260 | 2.552 | 54.21% | 2.897 | 3.760 | 4.364e-03 | 4.181e-03 | 0.958 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.979 | 7.448 | 1.469 | 24.57% | 4.167 | 3.948 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.164 | 6.794 | 2.630 | 63.16% | 2.353 | 3.294 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.531 | 8.226 | 1.695 | 25.95% | 4.719 | 4.727 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 11.358 | 11.064 | -0.294 | -2.59% | 9.463 | 9.555 | 2.167e-03 | 2.167e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.680 | 5.382 | 0.702 | 15.00% | 2.785 | 3.872 | 4.944e-03 | 4.486e-03 | 0.907 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 6.091 | 5.666 | -0.425 | -6.98% | 4.196 | 4.156 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 4.452 | 4.955 | 0.503 | 11.30% | 2.557 | 3.445 | 4.181e-03 | 4.211e-03 | 1.007 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.980 | 5.620 | -0.360 | -6.02% | 4.085 | 4.110 | 3.082e-03 | 3.082e-03 | 1.000 |

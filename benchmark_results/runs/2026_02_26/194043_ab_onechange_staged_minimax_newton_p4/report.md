# Solver Benchmark A/B Report

Generated: 2026-02-26T19:40:43

A: greedy_affine_opt
B: staged_minimax_newton

| kind | n | k | case | method | greedy_affine_opt_total_ms | staged_minimax_newton_total_ms | delta_ms(B-A) | delta_pct | greedy_affine_opt_iter_ms | staged_minimax_newton_iter_ms | greedy_affine_opt_relerr | staged_minimax_newton_relerr | relerr_ratio(B/A) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 256 | 256 | gaussian_spd | Chebyshev-Apply | 2.248 | 2.454 | 0.206 | 9.16% | 0.548 | 0.548 | 2.029e-03 | 2.029e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.640 | 2.923 | 0.283 | 10.72% | 0.940 | 1.018 | 4.517e-03 | 4.517e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 2.938 | 3.145 | 0.207 | 7.05% | 1.238 | 1.239 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.471 | 2.661 | 0.190 | 7.69% | 0.771 | 0.755 | 3.693e-03 | 3.418e-03 | 0.926 |
| spd | 256 | 256 | gaussian_spd | PE-Quad-Inverse-Multiply | 2.897 | 3.145 | 0.248 | 8.56% | 1.197 | 1.240 | 3.571e-03 | 3.571e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 2.060 | 1.915 | -0.145 | -7.04% | 0.548 | 0.553 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.465 | 2.308 | -0.157 | -6.37% | 0.953 | 0.945 | 3.784e-03 | 3.784e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.964 | 2.893 | -0.071 | -2.40% | 1.452 | 1.530 | 4.547e-03 | 4.547e-03 | 1.000 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.401 | 2.092 | -0.309 | -12.87% | 0.889 | 0.729 | 4.120e-03 | 4.150e-03 | 1.007 |
| spd | 256 | 256 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.924 | 2.677 | -0.247 | -8.45% | 1.412 | 1.315 | 4.089e-03 | 4.089e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Chebyshev-Apply | 3.243 | 3.726 | 0.483 | 14.89% | 1.340 | 1.435 | 2.045e-03 | 2.045e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | 2.814 | 3.241 | 0.427 | 15.17% | 0.912 | 0.950 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 3.318 | 3.583 | 0.265 | 7.99% | 1.416 | 1.292 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.655 | 3.026 | 0.371 | 13.97% | 0.753 | 0.735 | 3.998e-03 | 4.089e-03 | 1.023 |
| spd | 512 | 512 | gaussian_spd | PE-Quad-Inverse-Multiply | 3.592 | 3.444 | -0.148 | -4.12% | 1.690 | 1.153 | 4.456e-03 | 4.456e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.702 | 2.671 | -0.031 | -1.15% | 1.241 | 1.115 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 2.534 | 2.499 | -0.035 | -1.38% | 1.073 | 0.943 | 4.700e-03 | 4.700e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 2.889 | 3.189 | 0.300 | 10.38% | 1.428 | 1.633 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.383 | 2.536 | 0.153 | 6.42% | 0.922 | 0.980 | 3.418e-03 | 2.914e-03 | 0.853 |
| spd | 512 | 512 | illcond_1e6 | PE-Quad-Inverse-Multiply | 2.942 | 3.102 | 0.160 | 5.44% | 1.481 | 1.546 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.594 | 2.333 | -0.261 | -10.06% | 0.584 | 0.585 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.088 | 3.836 | -0.252 | -6.16% | 2.078 | 2.089 | 4.028e-03 | 4.028e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.816 | 5.572 | -0.244 | -4.20% | 3.806 | 3.824 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.658 | 3.362 | -0.296 | -8.09% | 1.648 | 1.614 | 3.723e-03 | 3.906e-03 | 1.049 |
| spd | 1024 | 1 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.551 | 6.287 | -0.264 | -4.03% | 4.541 | 4.539 | 4.333e-03 | 4.333e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.056 | 2.149 | 0.093 | 4.52% | 0.590 | 0.584 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.631 | 3.726 | 0.095 | 2.62% | 2.165 | 2.161 | 4.639e-03 | 4.639e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.408 | 5.573 | 0.165 | 3.05% | 3.942 | 4.007 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.167 | 3.289 | 0.122 | 3.85% | 1.701 | 1.723 | 3.479e-03 | 3.677e-03 | 1.057 |
| spd | 1024 | 1 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.395 | 5.515 | 0.120 | 2.22% | 3.929 | 3.950 | 3.006e-03 | 3.006e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.205 | 2.167 | -0.038 | -1.72% | 0.654 | 0.656 | 2.075e-03 | 2.075e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | 3.861 | 3.687 | -0.174 | -4.51% | 2.310 | 2.177 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 5.614 | 5.542 | -0.072 | -1.28% | 4.063 | 4.031 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.288 | 3.186 | -0.102 | -3.10% | 1.737 | 1.676 | 3.601e-03 | 3.967e-03 | 1.102 |
| spd | 1024 | 16 | gaussian_spd | PE-Quad-Inverse-Multiply | 5.634 | 5.564 | -0.070 | -1.24% | 4.083 | 4.053 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.143 | 2.322 | 0.179 | 8.35% | 0.659 | 0.662 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.722 | 3.862 | 0.140 | 3.76% | 2.238 | 2.202 | 4.639e-03 | 4.639e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.542 | 5.720 | 0.178 | 3.21% | 4.058 | 4.060 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.264 | 3.438 | 0.174 | 5.33% | 1.781 | 1.778 | 3.464e-03 | 3.677e-03 | 1.061 |
| spd | 1024 | 16 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.547 | 5.699 | 0.152 | 2.74% | 4.063 | 4.039 | 2.914e-03 | 2.914e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.826 | 2.450 | -0.376 | -13.31% | 0.717 | 0.720 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.354 | 3.943 | -0.411 | -9.44% | 2.245 | 2.212 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 6.176 | 5.825 | -0.351 | -5.68% | 4.067 | 4.094 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.853 | 3.523 | -0.330 | -8.56% | 1.745 | 1.792 | 3.647e-03 | 3.860e-03 | 1.058 |
| spd | 1024 | 64 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.173 | 5.824 | -0.349 | -5.65% | 4.064 | 4.093 | 4.181e-03 | 4.181e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.381 | 4.664 | 2.283 | 95.88% | 0.720 | 0.720 | 2.014e-03 | 2.014e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 3.898 | 6.119 | 2.221 | 56.98% | 2.237 | 2.175 | 4.608e-03 | 4.608e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.724 | 8.030 | 2.306 | 40.29% | 4.063 | 4.086 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.427 | 5.737 | 2.310 | 67.41% | 1.766 | 1.793 | 3.464e-03 | 3.677e-03 | 1.061 |
| spd | 1024 | 64 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.732 | 8.044 | 2.312 | 40.33% | 4.071 | 4.101 | 2.930e-03 | 2.930e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Chebyshev-Apply | 11.303 | 11.549 | 0.246 | 2.18% | 9.296 | 9.327 | 2.151e-03 | 2.151e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | 4.698 | 4.913 | 0.215 | 4.58% | 2.691 | 2.691 | 4.364e-03 | 4.364e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | Inverse-Newton-Inverse-Multiply | 6.144 | 6.125 | -0.019 | -0.31% | 4.137 | 3.903 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.260 | 4.284 | 0.024 | 0.56% | 2.253 | 2.062 | 3.998e-03 | 4.364e-03 | 1.092 |
| spd | 1024 | 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 6.430 | 6.941 | 0.511 | 7.95% | 4.423 | 4.719 | 4.395e-03 | 4.395e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Chebyshev-Apply | 10.796 | 11.655 | 0.859 | 7.96% | 9.477 | 9.596 | 2.167e-03 | 2.167e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | 4.114 | 4.939 | 0.825 | 20.05% | 2.795 | 2.880 | 4.944e-03 | 4.944e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Inverse-Multiply | 5.442 | 6.274 | 0.832 | 15.29% | 4.123 | 4.215 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.682 | 4.323 | 0.641 | 17.41% | 2.363 | 2.264 | 4.181e-03 | 3.998e-03 | 0.956 |
| spd | 1024 | 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 5.416 | 6.295 | 0.879 | 16.23% | 4.097 | 4.236 | 3.082e-03 | 3.082e-03 | 1.000 |

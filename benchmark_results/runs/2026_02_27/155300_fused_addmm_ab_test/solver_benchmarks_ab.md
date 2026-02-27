# Solver Benchmark A/B Report

Generated: 2026-02-27T15:54:09

A: A
B: B

| kind | p | n | k | case | method | A_total_ms | B_total_ms | delta_ms(B-A) | delta_pct | A_iter_ms | B_iter_ms | A_relerr | B_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.462 | 1.902 | -0.560 | -22.75% | 0.660 | 0.659 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 2 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.942 | 2.122 | 0.180 | 9.27% | 0.661 | 0.838 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.100 | 2.316 | 0.216 | 10.29% | 0.769 | 0.679 | 3.204e-03 | 3.204e-03 | 1.000 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.155 | 1.805 | -0.350 | -16.24% | 0.789 | 0.665 | 3.555e-03 | 3.555e-03 | 1.000 |
| spd | 2 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 2.799 | 2.689 | -0.110 | -3.93% | 1.605 | 1.590 | 3.769e-03 | 4.791e-03 | 1.271 |
| spd | 2 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.948 | 2.702 | -0.246 | -8.34% | 1.606 | 1.592 | 5.005e-03 | 5.890e-03 | 1.177 |
| spd | 2 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.249 | 2.876 | -0.373 | -11.48% | 1.610 | 1.598 | 3.937e-03 | 4.913e-03 | 1.248 |
| spd | 2 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.136 | 3.292 | 0.156 | 4.97% | 1.326 | 1.619 | 4.944e-03 | 5.859e-03 | 1.185 |
| spd | 2 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.210 | 2.759 | -0.451 | -14.05% | 1.366 | 1.594 | 3.860e-03 | 4.913e-03 | 1.273 |
| spd | 2 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.359 | 2.616 | -0.743 | -22.12% | 1.622 | 1.443 | 4.974e-03 | 5.890e-03 | 1.184 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.574 | 3.618 | 0.044 | 1.23% | 2.363 | 2.338 | 4.578e-03 | 4.578e-03 | 1.000 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.247 | 3.061 | -0.186 | -5.73% | 1.908 | 1.897 | 7.294e-03 | 7.294e-03 | 1.000 |
| spd | 4 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.701 | 1.954 | -0.747 | -27.66% | 0.719 | 0.716 | 3.693e-03 | 3.693e-03 | 1.000 |
| spd | 4 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.953 | 1.912 | -0.041 | -2.10% | 0.846 | 0.715 | 4.120e-03 | 4.120e-03 | 1.000 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.196 | 2.216 | 0.020 | 0.91% | 0.846 | 0.718 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 4 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.046 | 2.197 | 0.151 | 7.38% | 0.897 | 0.865 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 4 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.533 | 3.060 | -0.473 | -13.39% | 1.970 | 1.974 | 3.723e-03 | 4.578e-03 | 1.230 |
| spd | 4 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.890 | 2.896 | 0.006 | 0.21% | 1.642 | 1.794 | 3.479e-03 | 4.395e-03 | 1.263 |
| spd | 4 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 2.908 | 3.018 | 0.110 | 3.78% | 1.655 | 1.816 | 3.601e-03 | 4.517e-03 | 1.254 |
| spd | 4 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.219 | 3.242 | 0.023 | 0.71% | 2.021 | 1.993 | 3.464e-03 | 4.517e-03 | 1.304 |
| spd | 4 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 4.608 | 5.611 | 1.003 | 21.77% | 2.551 | 1.928 | 3.647e-03 | 4.517e-03 | 1.239 |
| spd | 4 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.223 | 3.085 | -0.138 | -4.28% | 1.693 | 1.902 | 3.464e-03 | 4.517e-03 | 1.304 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.602 | 3.926 | -0.676 | -14.69% | 2.736 | 2.720 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.582 | 3.624 | 0.042 | 1.17% | 2.289 | 2.264 | 4.181e-03 | 4.181e-03 | 1.000 |

# Solver Benchmark A/B Report

Generated: 2026-02-26T20:54:42

A: newton_ref
B: pe_best

| kind | p | n | k | case | newton_ref_method | pe_best_method | newton_ref_total_ms | pe_best_total_ms | delta_ms(B-A) | delta_pct | newton_ref_iter_ms | pe_best_iter_ms | newton_ref_relerr | pe_best_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 2.444 | 2.225 | -0.219 | -8.96% | 0.908 | 0.737 | 1.123e-01 | 3.799e-03 | 0.034 |
| spd | 2 | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 2.392 | 2.558 | 0.166 | 6.94% | 0.984 | 0.847 | 1.118e-01 | 3.311e-03 | 0.030 |
| spd | 2 | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 2.744 | 3.974 | 1.230 | 44.83% | 0.923 | 0.713 | 1.826e-01 | 3.204e-03 | 0.018 |
| spd | 2 | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 2.655 | 2.438 | -0.217 | -8.17% | 1.269 | 0.882 | 1.807e-01 | 3.555e-03 | 0.020 |
| spd | 2 | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 5.050 | 3.043 | -2.007 | -39.74% | 2.366 | 1.599 | 2.598e-01 | 3.769e-03 | 0.015 |
| spd | 2 | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3.799 | 2.967 | -0.832 | -21.90% | 2.354 | 1.606 | 2.617e-01 | 5.005e-03 | 0.019 |
| spd | 2 | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.007 | 3.263 | -0.744 | -18.57% | 2.423 | 1.669 | 2.598e-01 | 3.937e-03 | 0.015 |
| spd | 2 | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3.972 | 3.173 | -0.799 | -20.12% | 2.333 | 1.378 | 2.617e-01 | 4.944e-03 | 0.019 |
| spd | 2 | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.036 | 3.137 | -0.899 | -22.27% | 2.323 | 1.375 | 2.598e-01 | 3.860e-03 | 0.015 |
| spd | 2 | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3.763 | 3.186 | -0.577 | -15.33% | 2.121 | 1.342 | 2.598e-01 | 4.974e-03 | 0.019 |
| spd | 2 | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 5.031 | 4.427 | -0.604 | -12.01% | 3.165 | 2.357 | 2.598e-01 | 4.578e-03 | 0.018 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.192 | 3.659 | -0.533 | -12.71% | 2.801 | 2.188 | 2.617e-01 | 7.294e-03 | 0.028 |
| spd | 4 | 256 | 256 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3.649 | 3.135 | -0.514 | -14.09% | 1.324 | 0.829 | 3.613e-02 | 3.693e-03 | 0.102 |
| spd | 4 | 256 | 256 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3.176 | 2.546 | -0.630 | -19.84% | 1.454 | 0.924 | 3.589e-02 | 4.120e-03 | 0.115 |
| spd | 4 | 512 | 512 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3.223 | 3.308 | 0.085 | 2.64% | 1.335 | 0.835 | 6.177e-02 | 3.998e-03 | 0.065 |
| spd | 4 | 512 | 512 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.327 | 2.247 | -2.080 | -48.07% | 1.417 | 0.854 | 6.128e-02 | 3.418e-03 | 0.056 |
| spd | 4 | 1024 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.832 | 3.767 | -1.065 | -22.04% | 3.039 | 1.991 | 1.045e-01 | 3.723e-03 | 0.036 |
| spd | 4 | 1024 | 1 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.021 | 3.679 | -0.342 | -8.51% | 2.659 | 1.885 | 1.035e-01 | 3.479e-03 | 0.034 |
| spd | 4 | 1024 | 16 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 5.633 | 3.326 | -2.307 | -40.96% | 2.783 | 1.847 | 1.040e-01 | 3.601e-03 | 0.035 |
| spd | 4 | 1024 | 16 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.448 | 3.451 | -0.997 | -22.41% | 2.579 | 1.678 | 1.035e-01 | 3.464e-03 | 0.033 |
| spd | 4 | 1024 | 64 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 4.045 | 3.806 | -0.239 | -5.91% | 2.713 | 1.826 | 1.040e-01 | 3.647e-03 | 0.035 |
| spd | 4 | 1024 | 64 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 5.717 | 5.490 | -0.227 | -3.97% | 2.634 | 1.709 | 1.035e-01 | 3.464e-03 | 0.033 |
| spd | 4 | 1024 | 1024 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 6.791 | 5.669 | -1.122 | -16.52% | 3.760 | 2.738 | 1.045e-01 | 3.998e-03 | 0.038 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 6.331 | 3.865 | -2.466 | -38.95% | 3.333 | 2.313 | 1.035e-01 | 4.181e-03 | 0.040 |

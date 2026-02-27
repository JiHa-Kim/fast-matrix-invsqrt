# Solver Benchmark A/B Report

Generated: 2026-02-26T21:19:45

A: cheb32
B: cheb24

| kind | p | n | k | case | method | cheb32_total_ms | cheb24_total_ms | delta_ms(B-A) | delta_pct | cheb32_iter_ms | cheb24_iter_ms | cheb32_relerr | cheb24_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.483 | 1.932 | -0.551 | -22.19% | 0.658 | 0.502 | 3.067e-03 | 3.113e-03 | 1.015 |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.061 | 2.891 | 0.830 | 40.27% | 0.662 | 0.501 | 3.021e-03 | 3.052e-03 | 1.010 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.288 | 2.084 | -0.204 | -8.92% | 0.750 | 0.568 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.513 | 1.894 | -0.619 | -24.63% | 0.692 | 0.565 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 3.233 | 2.157 | -1.076 | -33.28% | 0.750 | 0.609 | 3.143e-03 | 3.143e-03 | 1.000 |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.402 | 2.211 | -0.191 | -7.95% | 0.731 | 0.591 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.389 | 2.669 | 0.280 | 11.72% | 0.662 | 0.501 | 1.968e-03 | 1.984e-03 | 1.008 |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.566 | 2.163 | -0.403 | -15.71% | 0.660 | 0.505 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.947 | 2.156 | -0.791 | -26.84% | 0.750 | 0.573 | 2.075e-03 | 2.075e-03 | 1.000 |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.211 | 3.116 | 0.905 | 40.93% | 0.746 | 0.565 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 7.054 | 5.121 | -1.933 | -27.40% | 4.845 | 3.632 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.306 | 2.241 | -0.065 | -2.82% | 0.761 | 0.541 | 2.014e-03 | 2.014e-03 | 1.000 |

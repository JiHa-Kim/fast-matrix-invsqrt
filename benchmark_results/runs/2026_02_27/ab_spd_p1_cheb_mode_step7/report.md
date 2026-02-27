# Solver Benchmark A/B Report

Generated: 2026-02-27T18:18:05

A: cheb_fixed
B: cheb_minimax_auto

| kind | p | n | k | case | method | cheb_fixed_total_ms | cheb_minimax_auto_total_ms | delta_ms(B-A) | delta_pct | cheb_fixed_iter_ms | cheb_minimax_auto_iter_ms | cheb_fixed_relerr | cheb_minimax_auto_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 1.877 | 1.805 | -0.072 | -3.84% | 0.502 | 0.661 | 8.179e-03 | 8.118e-03 | 0.993 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 1.583 | 1.819 | 0.236 | 14.91% | 0.502 | 0.655 | 8.057e-03 | 7.996e-03 | 0.992 |
| spd | 1 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 1.649 | 2.196 | 0.547 | 33.17% | 0.569 | 0.736 | 8.301e-03 | 8.362e-03 | 1.007 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 1.667 | 2.131 | 0.464 | 27.83% | 0.567 | 0.749 | 8.606e-03 | 8.606e-03 | 1.000 |
| spd | 1 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.215 | 6.129 | 3.914 | 176.70% | 0.611 | 4.764 | 8.118e-03 | 8.118e-03 | 1.000 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.054 | 1.804 | -0.250 | -12.17% | 0.830 | 0.712 | 7.751e-03 | 7.782e-03 | 1.004 |

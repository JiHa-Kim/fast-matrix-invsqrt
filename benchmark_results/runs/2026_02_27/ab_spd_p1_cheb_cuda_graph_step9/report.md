# Solver Benchmark A/B Report

Generated: 2026-02-27T18:20:21

A: cheb_graph_off
B: cheb_graph_on

| kind | p | n | k | case | method | cheb_graph_off_total_ms | cheb_graph_on_total_ms | delta_ms(B-A) | delta_pct | cheb_graph_off_iter_ms | cheb_graph_on_iter_ms | cheb_graph_off_relerr | cheb_graph_on_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 3.985 | 1.716 | -2.269 | -56.94% | 2.623 | 0.501 | 8.179e-03 | 8.179e-03 | 1.000 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 4.802 | 1.632 | -3.170 | -66.01% | 3.060 | 0.506 | 8.057e-03 | 8.057e-03 | 1.000 |
| spd | 1 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.327 | 2.561 | -1.766 | -40.81% | 2.986 | 0.568 | 8.301e-03 | 8.301e-03 | 1.000 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 4.767 | 1.675 | -3.092 | -64.86% | 3.528 | 0.573 | 8.606e-03 | 8.606e-03 | 1.000 |
| spd | 1 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 5.746 | 4.969 | -0.777 | -13.52% | 2.882 | 3.663 | 8.118e-03 | 8.118e-03 | 1.000 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.636 | 2.606 | -2.030 | -43.79% | 3.199 | 0.573 | 7.751e-03 | 7.751e-03 | 1.000 |

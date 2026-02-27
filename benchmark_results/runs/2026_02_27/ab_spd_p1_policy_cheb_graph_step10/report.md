# Solver Benchmark A/B Report

Generated: 2026-02-27T18:22:14

A: pe_quad
B: cheb_graph_on

| kind | p | n | k | case | pe_quad_method | cheb_graph_on_method | pe_quad_total_ms | cheb_graph_on_total_ms | delta_ms(B-A) | delta_pct | pe_quad_iter_ms | cheb_graph_on_iter_ms | pe_quad_relerr | cheb_graph_on_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.080 | 2.715 | 0.635 | 30.53% | 0.807 | 0.507 | 6.714e-03 | 8.179e-03 | 1.218 |
| spd | 1 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.252 | 1.576 | -0.676 | -30.02% | 0.816 | 0.502 | 7.080e-03 | 8.057e-03 | 1.138 |
| spd | 1 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.253 | 2.057 | -0.196 | -8.70% | 0.816 | 0.568 | 6.531e-03 | 8.301e-03 | 1.271 |
| spd | 1 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.070 | 1.887 | -0.183 | -8.84% | 0.809 | 0.553 | 7.080e-03 | 8.606e-03 | 1.216 |
| spd | 1 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | Chebyshev-Apply | 4.670 | 1.874 | -2.796 | -59.87% | 3.406 | 0.601 | 6.592e-03 | 8.118e-03 | 1.231 |
| spd | 1 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.515 | 2.122 | -0.393 | -15.63% | 1.066 | 0.793 | 1.044e-02 | 7.751e-03 | 0.742 |

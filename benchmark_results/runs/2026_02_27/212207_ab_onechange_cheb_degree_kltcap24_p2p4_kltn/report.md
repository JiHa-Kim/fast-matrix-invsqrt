# Solver Benchmark A/B Report

Generated: 2026-02-26T21:22:07

A: no_klt_cap
B: klt24_cap

| kind | p | n | k | case | method | no_klt_cap_total_ms | klt24_cap_total_ms | delta_ms(B-A) | delta_pct | no_klt_cap_iter_ms | klt24_cap_iter_ms | no_klt_cap_relerr | klt24_cap_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.170 | 2.008 | -0.162 | -7.47% | 0.665 | 0.503 | 3.067e-03 | 3.113e-03 | 1.015 |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.038 | 2.095 | 0.057 | 2.80% | 0.663 | 0.504 | 3.021e-03 | 3.052e-03 | 1.010 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.299 | 2.049 | -0.250 | -10.87% | 0.755 | 0.567 | 3.082e-03 | 3.082e-03 | 1.000 |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.621 | 2.295 | -0.326 | -12.44% | 0.750 | 0.568 | 3.067e-03 | 3.067e-03 | 1.000 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 3.407 | 1.965 | -1.442 | -42.32% | 0.802 | 0.609 | 3.143e-03 | 3.143e-03 | 1.000 |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.434 | 2.350 | -0.084 | -3.45% | 0.742 | 0.608 | 3.357e-03 | 3.357e-03 | 1.000 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.691 | 2.093 | -0.598 | -22.22% | 0.664 | 0.501 | 1.968e-03 | 1.984e-03 | 1.008 |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 2.514 | 2.112 | -0.402 | -15.99% | 0.650 | 0.506 | 1.968e-03 | 1.968e-03 | 1.000 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.299 | 2.537 | 0.238 | 10.35% | 0.729 | 0.570 | 2.075e-03 | 2.075e-03 | 1.000 |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.259 | 2.534 | 0.275 | 12.17% | 0.747 | 0.570 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.491 | 2.412 | -0.079 | -3.17% | 0.835 | 0.609 | 2.060e-03 | 2.060e-03 | 1.000 |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.361 | 2.547 | 0.186 | 7.88% | 0.806 | 0.759 | 2.014e-03 | 2.014e-03 | 1.000 |

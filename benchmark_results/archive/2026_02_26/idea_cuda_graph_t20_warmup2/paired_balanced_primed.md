# CUDA Graph Paired A/B (Balanced + Primed)

- Same prepared inputs and preconditioning reused per cell.
- Off/on order alternated by cell.
- Each variant is primed once, second measurement used for comparison.

- Mean delta total: **-2.87%**
- Mean delta iter: **-5.30%**
- Mean delta relerr: **+0.00%**
- Cell wins by total: on=18, off=0, ties=0

| p | k | case | order | off_total_ms | on_total_ms | d_total | off_iter_ms | on_iter_ms | d_iter |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | gaussian_spd | off->on | 3.270 | 3.204 | -2.02% | 1.440 | 1.374 | -4.58% |
| 1 | 1 | illcond_1e6 | on->off | 3.008 | 2.932 | -2.51% | 1.518 | 1.443 | -4.97% |
| 1 | 16 | gaussian_spd | off->on | 3.137 | 2.943 | -6.16% | 1.607 | 1.413 | -12.03% |
| 1 | 16 | illcond_1e6 | on->off | 3.395 | 3.301 | -2.76% | 1.520 | 1.426 | -6.17% |
| 1 | 64 | gaussian_spd | off->on | 3.233 | 3.123 | -3.38% | 1.542 | 1.433 | -7.09% |
| 1 | 64 | illcond_1e6 | on->off | 3.005 | 2.955 | -1.68% | 1.503 | 1.453 | -3.37% |
| 2 | 1 | gaussian_spd | off->on | 3.647 | 3.508 | -3.81% | 2.043 | 1.904 | -6.80% |
| 2 | 1 | illcond_1e6 | on->off | 3.477 | 3.421 | -1.63% | 1.979 | 1.922 | -2.87% |
| 2 | 16 | gaussian_spd | off->on | 3.730 | 3.563 | -4.46% | 2.055 | 1.889 | -8.10% |
| 2 | 16 | illcond_1e6 | on->off | 3.592 | 3.532 | -1.68% | 1.994 | 1.934 | -3.03% |
| 2 | 64 | gaussian_spd | off->on | 3.498 | 3.340 | -4.51% | 2.064 | 1.907 | -7.64% |
| 2 | 64 | illcond_1e6 | on->off | 3.964 | 3.881 | -2.09% | 2.011 | 1.929 | -4.12% |
| 4 | 1 | gaussian_spd | off->on | 4.486 | 4.283 | -4.53% | 2.600 | 2.396 | -7.82% |
| 4 | 1 | illcond_1e6 | on->off | 3.938 | 3.923 | -0.38% | 2.503 | 2.488 | -0.60% |
| 4 | 16 | gaussian_spd | off->on | 4.097 | 3.909 | -4.61% | 2.629 | 2.440 | -7.18% |
| 4 | 16 | illcond_1e6 | on->off | 4.032 | 4.015 | -0.43% | 2.513 | 2.496 | -0.69% |
| 4 | 64 | gaussian_spd | off->on | 4.497 | 4.305 | -4.26% | 2.608 | 2.416 | -7.35% |
| 4 | 64 | illcond_1e6 | on->off | 4.019 | 3.992 | -0.66% | 2.554 | 2.527 | -1.03% |

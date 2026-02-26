# Coupled Apply Fast-Path A/B (20 Trials, Same Env)

- Setup: `matrix_solve.py`, `n=1024`, `p in {1,2,4}`, `k in {1,16,64}`, cases `{gaussian_spd, illcond_1e6}`, `precond=jacobi`, `bf16`, `trials=20`, `timing_reps=5`.
- Mode: `--online-coeff-mode greedy-affine-opt` (default).
- Compared variants: baseline `HEAD` vs optimized `fast_iroot/coupled.py` affine fast paths.
- Execution control: both variants run with the same Python interpreter (`.venv/Scripts/python.exe`).

## Overall Means (PE-Quad-Coupled-Apply)
| variant | count | total_ms | iter_ms | relerr | newton_steps | minimax_steps | affineopt_steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 18 | 3.689 | 2.036 | 4.334e-03 | 0.33 | 0.00 | 0.67 |
| optimized | 18 | 3.851 | 2.015 | 4.198e-03 | 0.33 | 0.00 | 0.67 |

- Delta (optimized vs baseline): total **+4.38%**, iter **-1.04%**, relerr **-3.15%**.
- Cell wins (lower total_ms): optimized **9**, baseline **9**, ties **0** (out of 18).
- Note: `total_ms` includes preconditioning time; the optimized code only targets coupled iteration (`iter_ms`).

## Per-p Deltas (means over k x case)
| p | baseline_total_ms | optimized_total_ms | d_total | d_iter | d_relerr |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.184 | 3.127 | -1.81% | -0.62% | +0.00% |
| 2 | 3.612 | 3.906 | +8.13% | -2.43% | -8.56% |
| 4 | 4.271 | 4.521 | +5.83% | -0.18% | +0.00% |

## Per-k Deltas (means over p x case)
| k | baseline_total_ms | optimized_total_ms | d_total | d_iter | d_relerr |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.613 | 3.626 | +0.37% | -1.86% | -3.10% |
| 16 | 3.616 | 4.351 | +20.32% | -1.23% | -3.23% |
| 64 | 3.839 | 3.576 | -6.85% | -0.03% | -3.12% |

## Extremes (total_ms delta)
- Best cell: p=4, k=64, case=illcond_1e6, delta=-18.12% (4.939 -> 4.044 ms)
- Worst cell: p=4, k=16, case=gaussian_spd, delta=+54.59% (4.248 -> 6.567 ms)

- Raw parsed rows: `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/comparison_coupled_apply_sameenv.csv`

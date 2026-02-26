# Affine Online-Coefficient Ablation (20 Trials)

- Setup: `matrix_solve.py`, `n=1024`, `p in {1,2,4}`, `k in {1,16,64}`, 2 cases, `precond=jacobi`, `bf16`.
- Parsed rows: 72 (expected 72)
- Baseline for deltas: `greedy-newton`

## Overall Means (Coupled Apply)
| mode | count | total_ms | iter_ms | relerr | d_total_vs_newton | d_iter_vs_newton | d_relerr_vs_newton | newton_steps | minimax_steps | affineopt_steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy-affine-opt | 18 | 3.680 | 2.041 | 4.334e-03 | -3.69% | -1.28% | -0.24% | 0.33 | 0.00 | 0.67 |
| greedy-minimax | 18 | 3.758 | 2.083 | 4.331e-03 | -1.67% | +0.74% | -0.32% | 0.67 | 2.00 | 0.00 |
| greedy-newton | 18 | 3.821 | 2.068 | 4.345e-03 | +0.00% | +0.00% | +0.00% | 1.00 | 0.00 | 0.00 |
| off | 18 | 3.808 | 2.211 | 5.088e-03 | -0.36% | +6.91% | +17.11% | 0.00 | 0.00 | 0.00 |

## Per-p Total Means
| mode | p=1 | p=2 | p=4 |
|---|---:|---:|---:|
| greedy-affine-opt | 3.159 | 3.629 | 4.252 |
| greedy-minimax | 3.285 | 3.895 | 4.092 |
| greedy-newton | 3.296 | 3.691 | 4.477 |
| off | 3.308 | 3.773 | 4.343 |

## Cell Wins (best total_ms per cell)
| mode | wins |
|---|---:|
| greedy-affine-opt | 7 |
| greedy-minimax | 4 |
| greedy-newton | 5 |
| off | 2 |


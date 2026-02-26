# Solve Preconditioner Ablation (Coupled Apply, 20 Trials)

- Parsed rows: 72 (expected 72)
- Cells: 18 (p in {1,2,4}, k in {1,16,64}, cases=2)

## Overall Means (lower is better)
| mode | count | total_ms | iter_ms | pre_ms | relerr | d_total_vs_frob | d_iter_vs_frob | d_pre_vs_frob | d_relerr_vs_frob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aol | 18 | 3.802 | 2.068 | 1.734 | 4.820e-03 | +1.52% | +0.12% | +3.24% | -3.09% |
| frob | 18 | 3.745 | 2.065 | 1.680 | 4.974e-03 | +0.00% | +0.00% | +0.00% | +0.00% |
| jacobi | 18 | 3.687 | 2.053 | 1.633 | 4.345e-03 | -1.55% | -0.57% | -2.75% | -12.64% |
| ruiz | 18 | 3.819 | 2.048 | 1.771 | 4.648e-03 | +1.98% | -0.84% | +5.44% | -6.55% |

## Per-p Means (total_ms)
| mode | p=1 | p=2 | p=4 |
|---|---:|---:|---:|
| aol | 3.381 | 3.650 | 4.375 |
| frob | 3.423 | 3.661 | 4.151 |
| jacobi | 3.323 | 3.595 | 4.142 |
| ruiz | 3.336 | 3.824 | 4.297 |

## Cell Wins (best total_ms per cell)
| mode | wins |
|---|---:|
| aol | 5 |
| frob | 4 |
| jacobi | 9 |
| ruiz | 0 |

